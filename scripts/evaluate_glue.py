#!/usr/bin/env python
"""Fine-tune & evaluate a pretrained NexteraBERT backbone on GLUE.

The fine-tuning recipe follows ModernBERT-base (Warner et al., 2024, Appendix E.1 /
Table 6 plus the reference implementation's src/evals/glue_jobs.py): the per-task
lr/batch-size/weight-decay/epochs live in TASK_DEFAULTS_MODERNBERT and the globals
that table does not vary in OPTIM_DEFAULTS — AdamW with LINEAR decay to zero after
6% warmup. ModernBERT-base is NexteraBERT's own ~150M scale and its LRs sit at
NexteraBERT's pretraining end-LR, so fine-tuning resumes about where pretraining
left off. --beta1/--beta2/--eps/--warmup_pct/--alpha_f/--schedule override the
globals; --lr/--batch_size/--epochs/--weight_decay/--llrd override the table.

Weight decay reaches torch.optim.AdamW unchanged, so its per-step decay is the
ordinary lr_t * wd * p (see build_optimizer): one wd for every trained matrix, with
no LR-dependent rescaling.

Two more things aid comparability (both on by default):

  * MNLI intermediate transfer — RTE, MRPC and STS-B are fine-tuned from an
    MNLI-tuned encoder rather than the cold pretrained backbone (ModernBERT's
    round-1/round-2 split, plus QNLI as in NeoBERT Appendix C). This is the
    single biggest lift for RTE, which otherwise sits near its majority baseline
    (~0.55) regardless of pretraining quality. Disable with --no_mnli_transfer.
  * Multi-seed averaging — pass several --seeds and the per-task scores are
    reported as mean (± std) across them. Each task caps how many seeds it
    actually uses (small high-variance tasks average over several; large stable
    tasks — MNLI, QQP, QNLI — use one), so the big tasks are not needlessly
    retrained per seed. Override with --uniform_seeds. WNLI is excluded from the
    average by convention.

Two time-savers are on by default (roughly halving the fine-tuning compute);
both are approximations of the paper protocol, and both can be switched off:

  * Early stopping — a run ends once the val metric has not improved for
    --patience (default 2) consecutive epochs. ModernBERT does the same ("early
    stopping is used for all the fine-tuning runs"), so this is part of the
    recipe rather than a shortcut. Heuristic, not exact: a plateaued run almost
    never recovers (the reported score is the best epoch so far), but ties on
    the small quantized dev sets count as no improvement and the low-LR tail is
    never reached, so a late best epoch can occasionally be missed. Use
    --patience 0 to always train the full per-task budgets.
  * Per-task seed caps — TASK_MAX_SEEDS follows ModernBERT/MosaicBERT
    (rte/mrpc/stsb 5, cola 4, sst2 3, the large stable tasks 1). Caps only bite
    when several --seeds are passed; --uniform_seeds runs every task on all of
    them.

Two optimiser-side regularisers sit on top of the recipe, both ON BY DEFAULT. The
second is NOT part of the ModernBERT protocol, so a score produced with it
is this repo's recipe rather than a reproduction of those papers — --no_llrd
restores the published protocol exactly:

  * No weight decay on norms and biases — decay applies to the weight MATRICES
    only, the standard BERT fine-tuning convention. Norm parameters are found by
    module type (LayerNorm / RMSNorm / SeparableDyT), not by
    name, so nothing slips through; see build_optimizer.
  * Layerwise LR decay (LLRD, --llrd F, default 0.9) — a MULTIPLIER on whatever
    LR the task already has (the built-in table's, an --hparams entry, or --lr):
    the head keeps it, each block below trains at F x the block above, the
    embeddings at F**(n_layer+1) x it. Keeps a small task (RTE, MRPC, CoLA) from
    overwriting the pretrained lower layers. Per-task values can come from
    --hparams; --llrd 1.0 / --no_llrd is the flat published protocol.

Multi-GPU: prefer --task_parallel (one task per single GPU, tasks run concurrently),
which reproduces the per-task batch sizes exactly — they are TOTALS, so DDP-sharding
one task across GPUs would inflate the effective batch and hurt the small tasks. Launching via torchrun still works and falls back to the DDP-per-task path.

    # single GPU — runs all 8 standard GLUE tasks by default
    python scripts/evaluate_glue.py --model checkpoints/discriminator \
        --output glue_results.json

    # full benchmark, task-parallel over 4 GPUs (one task per GPU). These are
    # ModernBERT's own seeds; per-task caps apply (rte/mrpc/stsb use all 5,
    # cola 4, sst2 3, the large tasks 1 — add --uniform_seeds to run all 5 on
    # every task)
    python scripts/evaluate_glue.py --model checkpoints/discriminator \
        --seeds 19 8364 717 10536 90166 --task_parallel --num_gpus 4 \
        --output glue_results.json

The default (--tasks all) runs the 8 standard GLUE tasks (WNLI excluded); pass e.g.
--tasks sst2 rte for a quick subset.

Four flags exist for scripts/search_glue.py, which drives this script to tune the
per-task hyperparameters. They are search machinery — the first two make a run
CHEAPER THAN THE PROTOCOL and must never produce a reported score:

  * --train_subsample N   proxy training set (seed-fixed subset, dev set intact)
  * --stop_after_epochs N screening rung: train N epochs, keep the LR schedule
                          stretched over the full per-task budget
  * --hparams FILE        per-task lr/wd/epochs/batch_size overriding the recipe
                          table — the JSON the search writes
  * --mnli_source DIR     reuse an already fine-tuned MNLI encoder for transfer
                          instead of training one
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
# Imported at module level (not lazily inside compute_metric) so a missing eval
# dependency fails on EVERY rank at startup. compute_metric runs on rank 0 only;
# a rank-0-only ImportError mid-epoch would leave the other DDP ranks hanging in
# the early-stop broadcast.
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Windows consoles default to a legacy code page (e.g. cp932) that cannot encode
# the em-dashes / '±' / '·' in the progress lines — without this, a finished run
# dies in its own print. Workers inherit the parent's pipe, so they need it too.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert.data import build_tokenizer  # noqa: E402
from nexterabert.export import save_pretrained  # noqa: E402
from nexterabert.hf_baselines import (  # noqa: E402
    hf_llrd,
    is_hf_model_dir,
    load_hf_for_sequence_classification,
    load_hf_tokenizer,
    resolve_hub_model,
    save_hf_glue_backbone,
)
from nexterabert.loading import load_for_task  # noqa: E402

# task -> (sentence1, sentence2 or None, num_labels, metric)
GLUE_TASKS = {
    "cola": ("sentence", None, 2, "mcc"),
    "sst2": ("sentence", None, 2, "acc"),
    "mrpc": ("sentence1", "sentence2", 2, "acc_f1"),
    "stsb": ("sentence1", "sentence2", 1, "pearson_spearman"),
    "qqp": ("question1", "question2", 2, "acc_f1"),
    "mnli": ("premise", "hypothesis", 3, "acc"),
    "qnli": ("question", "sentence", 2, "acc"),
    "rte": ("sentence1", "sentence2", 2, "acc"),
    "wnli": ("sentence1", "sentence2", 2, "acc"),
}

# ModernBERT-base's per-task GLUE recipe — the only table this script has. A good
# match for NexteraBERT because (1) ModernBERT-base is ~149M params, i.e.
# NexteraBERT's own scale, and (2) these LRs sit right at NexteraBERT's pretraining
# end-LR (peak 8e-4/4e-4 decayed to 10% = 8e-5/4e-5), so fine-tuning resumes about
# where pretraining left off instead of barely moving the weights.
#
# lr / epochs are the ModernBERT-BASE column of Warner et al., 2024 ("Smarter,
# Better, Faster, Longer"), Appendix E.1 Table 6; the paper's search grid was lr in
# {1e-5,3e-5,5e-5,8e-5}, wd in {1e-6,5e-6,8e-6,1e-5}, epochs in {1,2,3}
# (sst2/mnli/rte) or {2,5,10} (qnli/qqp/cola/mrpc/stsb).
#
# batch_size is NOT in Table 6 — it comes from the reference implementation
# (AnswerDotAI/ModernBERT, src/evals/glue_jobs.py, the per-*Job* defaults that the
# paper's sweep never overrides): mnli 64, cola/mrpc/stsb 32, rte/qqp/qnli/sst2 16.
# The same file fixes max_sequence_length=256 for every task, which is this
# script's --max_len default. Optimiser/schedule globals live in OPTIM_DEFAULTS.
#
# weight_decay is a TORCH weight decay: it reaches torch.optim.AdamW unchanged, so
# the per-step decay is lr_t * wd * p. ModernBERT's published wds are composer
# DecoupledAdamW values, whose decay is LR-independent, so they are stored here
# already converted — wd_torch = wd_published / lr:
#
#   5e-6 at lr 5e-5   -> 0.1     (cola/sst2/mrpc/stsb/qqp/qnli/rte)
#   5e-6 at lr 1.5e-4 -> 0.0333  (mnli)
#   1e-5 at lr 5e-5   -> 0.2     (wnli)
#
# Change an entry's lr and the equivalent wd moves with it; keep the values on this
# torch scale (a composer-scale 5e-6 here would decay by ~2.5e-10 a step, i.e. not
# at all).
#
# NOTE qqp/mrpc/stsb carry 10-epoch budgets; on the large qqp that is real compute
# even with early stopping (min 2 epochs over ~364k examples).
TASK_DEFAULTS_MODERNBERT = {
    "cola": {"lr": 4e-5, "epochs": 5,  "batch_size": 4, "weight_decay": 5e-3},
    "sst2": {"lr": 4e-5, "epochs": 2,  "batch_size": 16, "weight_decay": 5e-3},
    "mrpc": {"lr": 4e-5, "epochs": 10, "batch_size": 16, "weight_decay": 5e-3},
    "stsb": {"lr": 4e-5, "epochs": 10, "batch_size": 4, "weight_decay": 5e-3},
    "qqp":  {"lr": 4e-5, "epochs": 10, "batch_size": 8, "weight_decay": 5e-3},
    # ModernBERT trains the MNLI transfer source for a single epoch. That is kept
    # faithful here; --mnli_epochs N raises only this task's budget (early stopping
    # then caps it at the best epoch) if a better-converged transfer source for
    # rte/mrpc/stsb is wanted.
    "mnli": {"lr": 1e-4, "epochs": 2,  "batch_size": 128, "weight_decay": 1e-2},
    "qnli": {"lr": 4e-5, "epochs": 2,  "batch_size": 16, "weight_decay": 5e-3},
    "rte":  {"lr": 4e-5, "epochs": 3,  "batch_size": 8, "weight_decay": 5e-3},
    # WNLI is not in ModernBERT's table (excluded from the GLUE average); sane default.
    "wnli": {"lr": 5e-5, "epochs": 5,  "batch_size": 16, "weight_decay": 0.2},
}

# Optimiser / schedule globals that the per-task table above does not vary. Every
# entry is overridable from the CLI (--beta1/--beta2/--eps/--schedule/--warmup_pct/
# --alpha_f), whose defaults are None = "take the value here".
#
# From AnswerDotAI/ModernBERT src/evals/glue_jobs.py (betas, eps) and
# yamls/finetuning/glue (linear_decay_with_warmup, t_warmup 0.06dur, alpha_f 0.0,
# i.e. decay to zero, not to a floor).
OPTIM_DEFAULTS = {
    "beta1": 0.9, "beta2": 0.95, "eps": 1e-6,
    "schedule": "linear", "warmup_pct": 0.06, "alpha_f": 0.0,
}

# Tasks whose fine-tuning is initialised from the MNLI-tuned encoder instead of the
# cold pretrained backbone. ModernBERT's round-1/round-2 split promotes exactly RTE,
# MRPC and STS-B (glue.py: round_2_task_names = {"mnli": {"rte", "mrpc", "stsb"}});
# QNLI is added following NeoBERT Appendix C.
MNLI_INIT_TASKS = ("rte", "mrpc", "stsb", "qnli")


# Per-task fine-tuning seed budget, from ModernBERT's GLUE config
# (yamls/finetuning/glue: rte/stsb/mrpc [19, 8364, 717, 10536, 90166], cola the
# first 4, sst2 the first 3, every other task the single default_seed 19). The
# small, high-variance tasks are averaged over several seeds; the large, stable
# ones use one. This caps how many of --seeds each task actually consumes, so
# passing 5 seeds does NOT retrain QQP/MNLI/QNLI five times. Disable the caps with
# --uniform_seeds to run every task on all --seeds.
TASK_MAX_SEEDS = {
    "rte": 5, "mrpc": 5, "stsb": 5, "cola": 4, "sst2": 3,
    "mnli": 1, "qqp": 1, "qnli": 1, "wnli": 1,
}

# The 8 tasks that make up the standard GLUE average. WNLI is excluded by
# convention (everyone reports its majority baseline), so MosaicBERT's headline
# average is over exactly these — matching it here keeps the numbers comparable.
GLUE_AVG_TASKS = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")

# Fixed shuffle seed for --train_subsample. Deliberately NOT the run's --seeds
# value: a hyperparameter search must rank its candidates on identical data, so
# the proxy subset has to be the same for every run of a task.
SUBSAMPLE_SEED = 20242024

# Per-task keys a --hparams file may override (anything else is ignored, so the
# "_search" history block search_glue.py writes alongside them is harmless).
HPARAM_KEYS = ("lr", "weight_decay", "epochs", "batch_size", "llrd")

# Default layerwise-LR-decay factor: each depth trains at this multiple of the
# depth above it, on top of the task's own LR. 0.9 keeps the embeddings around 9%
# of the head's LR at this backbone's depth (0.9**23 = 0.089 at
# 22 blocks) — enough to protect the pretrained lower layers on the small tasks
# without freezing them. --no_llrd (or --llrd 1.0) restores a flat LR.
DEFAULT_LLRD = 0.9

# Torch weight decay for a task with no row in the table above (every GLUE task has
# one, so this is only a floor for hand-added tasks). Matches the table's own scale.
DEFAULT_WEIGHT_DECAY = 0.1

_HPARAMS_CACHE = {}


def load_hparams(path):
    """Per-task hyperparameter overrides from a --hparams JSON file.

    Returns {task: {lr/weight_decay/epochs/batch_size}}; {} when no file was
    given. Parsed once per process (workers re-read it, which is why the path —
    not the parsed values — is what gets forwarded to them)."""
    if not path:
        return {}
    if path not in _HPARAMS_CACHE:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        table = {}
        for task, cfg in raw.items():
            if task.startswith("_") or not isinstance(cfg, dict):
                continue
            if task not in GLUE_TASKS:
                print(f"[glue][warn] --hparams: ignoring unknown task '{task}'")
                continue
            table[task] = {k: cfg[k] for k in HPARAM_KEYS if k in cfg}
        _HPARAMS_CACHE[path] = table
    return _HPARAMS_CACHE[path]


def is_ddp():
    return dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_ddp() else 0


def get_world_size():
    return dist.get_world_size() if is_ddp() else 1


def set_seed(seed: int):
    """Seed python / numpy / torch so each fine-tuning run is reproducible.

    The small GLUE tasks are famously seed-sensitive, so this (plus averaging
    over several ``--seeds``) is what makes the reported scores stable enough to
    compare against MosaicBERT.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_dataset(ds, tokenizer, c1, c2, max_len):
    def _map(batch):
        if c2 is None:
            enc = tokenizer(batch[c1], truncation=True, max_length=max_len)
        else:
            enc = tokenizer(batch[c1], batch[c2], truncation=True, max_length=max_len)
        return enc

    cols = [c for c in (c1, c2) if c is not None]
    ds = ds.map(_map, batched=True, remove_columns=cols)
    return ds


class GlueCollator:
    def __init__(self, pad_id: int, regression: bool):
        self.pad_id = pad_id
        self.regression = regression

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, token_type_ids, labels = [], [], [], []
        for b in batch:
            n = len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * (maxlen - n))
            attn.append([1] * n + [0] * (maxlen - n))
            # token_type_ids from the tokenizer (0 for sentence A, 1 for B); pad
            # with segment 0. Falls back to all-zeros if the tokenizer omits them.
            tt = b.get("token_type_ids", [0] * n)
            token_type_ids.append(tt + [0] * (maxlen - n))
            labels.append(b["label"])
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        if self.regression:
            out["labels"] = torch.tensor(labels, dtype=torch.float).unsqueeze(-1)
        else:
            out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out


def compute_metric(metric, preds, labels):
    if metric == "mcc":
        return {"mcc": float(matthews_corrcoef(labels, preds))}
    if metric == "acc":
        return {"acc": float((preds == labels).mean())}
    if metric == "acc_f1":
        return {"acc": float((preds == labels).mean()),
                "f1": float(f1_score(labels, preds))}
    if metric == "pearson_spearman":
        return {"pearson": float(pearsonr(preds, labels)[0]),
                "spearman": float(spearmanr(preds, labels)[0])}
    raise ValueError(metric)


# ---------------------------------------------------------------------------
# Optimiser construction: no-decay groups + layerwise LR decay (LLRD)
# ---------------------------------------------------------------------------

def norm_param_names(module):
    """Fully-qualified names of every parameter owned by a normalisation layer.

    Weight decay is kept off norms by MODULE TYPE rather than by a name pattern,
    because only one of this encoder's three norm flavours is a torch builtin:
    LayerNorm (out_norm / pooler.norm), RMSNorm and SeparableDyT. Their gains are called 'weight'/'alpha'/'beta' with
    nothing in the parameter's own name saying "norm", so a name-only rule would
    silently decay them (and decayed norm gains shrink the residual stream — the
    classic cause of a fine-tuning run that quietly under-performs).
    """
    names = set()
    for mod_name, mod in module.named_modules():
        cls = type(mod).__name__.lower()
        is_norm = (isinstance(mod, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                                    torch.nn.modules.batchnorm._BatchNorm))
                   or "norm" in cls or "dyt" in cls)
        if not is_norm:
            continue
        for p_name, _ in mod.named_parameters(recurse=False):
            names.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


def is_no_decay(name, param, norm_names):
    """True for parameters excluded from weight decay: biases and norms.

    The standard BERT fine-tuning convention (Devlin et al.'s run_classifier,
    every HuggingFace example since): decay the weight MATRICES only. Three
    rules, in order of reliability — the parameter belongs to a norm module,
    it is 0/1-dimensional (biases, gates, per-channel scales, DyT alphas), or
    its name says bias/alpha/beta/dyt.
    """
    if name in norm_names or param.ndim < 2:
        return True
    lowered = name.lower()
    return any(tag in lowered for tag in ("bias", "alpha", "beta", "dyt"))


def llrd_depth(name, n_layer):
    """Depth index used by layerwise LR decay.

    0 = the embedding layer, 1..n_layer = encoder blocks bottom-to-top,
    n_layer+1 = everything above the blocks (final norm, pooler, classifier),
    which keeps the freshly initialised head at the full peak LR.
    """
    if not name.startswith("encoder."):
        return n_layer + 1                      # classifier / any head parameter
    body = name[len("encoder."):]
    if body.startswith(("embedding.", "token_type_embeddings.", "_emb_proj.")):
        return 0
    if body.startswith("blocks."):
        return int(body.split(".")[1]) + 1
    return n_layer + 1                          # out_norm, pooler


def build_optimizer(model, lr, weight_decay, betas, eps,
                    llrd=1.0, n_layer=None, depth_fn=None):
    """AdamW over depth x decay parameter groups. Returns (optimizer, summary).

    ``n_layer`` / ``depth_fn`` default to the NexteraBERT layout (``llrd_depth``
    over ``model.encoder.blocks``); a foreign backbone passes its own -- see
    ``nexterabert.hf_baselines.hf_llrd`` for the ModernBERT / NeoBERT / BERT map.

    * Weight decay is applied to matrices only (see is_no_decay).
    * Layerwise LR decay (LLRD, Howard & Ruder 2018 discriminative fine-tuning;
      the standard BERT/ELECTRA/DeBERTa fine-tuning trick) scales each depth's
      LR by ``llrd ** (depth_from_top)`` — a MULTIPLIER on the task's own LR,
      never a replacement for it: the head trains at the full per-task ``lr``,
      block n_layer at lr*llrd, ..., the embeddings at lr*llrd**(n_layer+1).
      Lower layers hold the general features learned in pretraining and are the
      ones a small GLUE task destroys first, so they move least. ``llrd=1.0`` is
      a no-op and reproduces the plain two-group optimiser.

    torch.optim.AdamW, which ALWAYS applies Adam's bias-correction terms (it has
    no switch to disable them). That matters most here: Zhang et al. (2021,
    "Revisiting Few-sample BERT Fine-tuning", Sec. 4) trace degenerate few-sample
    fine-tuning runs to the debiasing omission in the legacy BERTAdam optimizer —
    without correction, 48% of 50 RTE runs scored under 55% (near random). RTE,
    MRPC, STS-B and CoLA — exactly the tasks they study — are in this benchmark,
    so do NOT substitute a BERTAdam-style optimizer (old pytorch_pretrained_bert,
    transformers' AdamW with correct_bias=False, or a hand-rolled Adam that skips
    the m_hat/v_hat division) for the call below.

    ``weight_decay`` reaches AdamW unchanged: every decayed group carries the
    same value and the per-step decay is torch's ordinary lr_t * wd * p, so a
    layer that LLRD slowed down also decays proportionally less. Nothing here
    rescales wd by the LR (composer's DecoupledAdamW would — see
    TASK_DEFAULTS_MODERNBERT for the conversion if a published decoupled value is
    being copied in).
    """
    if n_layer is None:
        n_layer = len(model.encoder.blocks)
    if depth_fn is None:
        def depth_fn(name):
            return llrd_depth(name, n_layer)
    norm_names = norm_param_names(model)
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = depth_fn(name) if llrd != 1.0 else n_layer + 1
        buckets.setdefault((depth, is_no_decay(name, p, norm_names)), []).append(p)

    param_groups, summary = [], []
    for (depth, no_decay), params in sorted(buckets.items()):
        group_lr = lr * (llrd ** (n_layer + 1 - depth))
        group_wd = 0.0 if no_decay else weight_decay
        param_groups.append({"params": params, "lr": group_lr,
                             "weight_decay": group_wd})
        summary.append({"depth": depth, "no_decay": no_decay, "lr": group_lr,
                        "weight_decay": group_wd,
                        "n_params": sum(p.numel() for p in params)})
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    # LambdaLR reads each group's own lr as its base, so the schedule multiplies
    # the LLRD ladder rather than flattening it.
    return optimizer, summary


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_task(task, args, seed, model_path=None, save_backbone_to=None):
    """Fine-tune one GLUE task for one seed and return the best-epoch scores.

    ``model_path`` overrides where the encoder is initialised from (used to load
    the MNLI-tuned encoder for RTE/MRPC/STS-B); ``save_backbone_to`` exports the
    fine-tuned encoder afterwards (used to produce that MNLI checkpoint).
    """
    from datasets import load_dataset

    set_seed(seed)
    rank = get_rank()
    world_size = get_world_size()
    ddp = is_ddp()
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    c1, c2, num_labels, metric = GLUE_TASKS[task]
    regression = num_labels == 1
    # A Hugging Face baseline (ModernBERT, NeoBERT, LFM2.5-Encoder, ...) goes
    # through the same recipe with NexteraBERT's head over its AutoModel
    # backbone and its own tokenizer -- see nexterabert.hf_baselines.
    is_hf = is_hf_model_dir(model_path or args.model)
    tokenizer = (load_hf_tokenizer(args.tokenizer) if is_hf
                 else build_tokenizer(args.tokenizer))

    # recipe table < --hparams file < explicit CLI override
    file_defaults = load_hparams(args.hparams).get(task, {})
    defaults = dict(TASK_DEFAULTS_MODERNBERT.get(task, {}))
    defaults.update(file_defaults)
    g = OPTIM_DEFAULTS
    lr = args.lr if args.lr is not None else defaults.get("lr", 2e-5)
    epochs = args.epochs if args.epochs is not None else defaults.get("epochs", 3)
    if task == "mnli" and args.mnli_epochs is not None:
        epochs = args.mnli_epochs   # --mnli_epochs beats --epochs for the transfer source
    batch_size = args.batch_size if args.batch_size is not None else defaults.get("batch_size", 32)
    # NOTE --weight_decay must be checked BEFORE the table, like every other
    # override: a hyperparameter search sweeping wd is otherwise silently pinned
    # to the recipe's value.
    weight_decay = args.weight_decay if args.weight_decay is not None \
        else defaults.get("weight_decay", DEFAULT_WEIGHT_DECAY)
    # LLRD's per-layer factor. It MULTIPLIES the task LR resolved above (the
    # recipe table, an --hparams entry or --lr): the head keeps that LR and each
    # depth below it is scaled by llrd**(depth from the top). --no_llrd (llrd 1.0)
    # gives every depth the task LR, i.e. the published flat protocol.
    llrd = args.llrd if args.llrd is not None         else defaults.get("llrd", DEFAULT_LLRD)
    if not args.llrd_enabled:
        llrd = 1.0
    # CLI overrides are None unless explicitly passed; otherwise take the recipe's.
    beta1 = args.beta1 if args.beta1 is not None else g["beta1"]
    beta2 = args.beta2 if args.beta2 is not None else g["beta2"]
    eps = args.eps if args.eps is not None else g["eps"]
    schedule = args.schedule if args.schedule is not None else g["schedule"]
    warmup_pct = args.warmup_pct if args.warmup_pct is not None else g["warmup_pct"]
    alpha_f = args.alpha_f if args.alpha_f is not None else g["alpha_f"]

    raw = load_dataset("nyu-mll/glue", task)
    train_split = raw["train"]
    val_key = "validation_matched" if task == "mnli" else "validation"
    val_split = raw[val_key]

    # Proxy training set for hyperparameter search. Shuffled with a FIXED seed
    # (not `seed`), so every candidate ranks on exactly the same examples and the
    # comparison isn't confounded by which subset each one happened to draw. The
    # dev set is never subsampled — the score has to stay comparable.
    if args.train_subsample and args.train_subsample < len(train_split):
        train_split = (train_split.shuffle(seed=SUBSAMPLE_SEED)
                       .select(range(args.train_subsample)))
        if rank == 0:
            print(f"  [{task}] train subsampled to {args.train_subsample} examples "
                  f"(proxy; search only)")

    train_split = encode_dataset(train_split, tokenizer, c1, c2, args.max_len)
    val_split = encode_dataset(val_split, tokenizer, c1, c2, args.max_len)
    train_split.set_format("python")
    val_split.set_format("python")

    collate = GlueCollator(tokenizer.pad_token_id, regression)

    if ddp:
        train_sampler = DistributedSampler(train_split, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        train_loader = DataLoader(train_split, batch_size=batch_size, sampler=train_sampler,
                                  collate_fn=collate, num_workers=2, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=2, pin_memory=True)
    # Validation is NOT sharded, even under DDP: every rank evaluates the full
    # dev set. GLUE dev sets are tiny, and a sharded DistributedSampler pads
    # ranks by repeating examples — that duplicate-biased metric would steer
    # best-epoch selection and early stopping. Full-set eval keeps DDP and
    # single-GPU runs of the same seed identical.
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)

    src_path = model_path or args.model
    depth_kwargs = {}
    if is_hf:
        model, config, info = load_hf_for_sequence_classification(
            src_path, num_labels, max_len=args.max_len)
        n_layer, depth_fn = hf_llrd(model)
        depth_kwargs = {"n_layer": n_layer, "depth_fn": depth_fn}
    else:
        model, config, info = load_for_task(src_path, "sequence-classification",
                                            num_labels=num_labels)
    if info["missing"] and rank == 0:
        print(f"  [warn] missing encoder keys: {info['missing'][:4]} ...")
    model.to(dev)
    raw_model = model   # uncompiled, unwrapped handle for params / checkpoint export

    if ddp:
        model = DDP(model, device_ids=[rank])
    # torch.compile after the DDP wrap. CUDA only; dynamic=True so the variable
    # GLUE sequence lengths don't trigger a recompile per batch. Dynamo's
    # DDPOptimizer (bucket-aligned graph splitting) is incompatible with
    # dynamic shapes here — a SymInt crossing a submodule boundary crashes AOT
    # compile with "'int' object has no attribute 'meta'" — so disable it and
    # compile the DDP wrapper as one graph; DDP's backward-hook allreduce still
    # runs, we only lose comm/compute overlap, which is negligible at GLUE
    # fine-tuning sizes. compile / DDP share raw_model's parameter tensors, so
    # the optimizer and export still use raw_model.
    if args.compile and dev.type == "cuda":
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)

    # Parameter groups: weight decay on matrices only (norms and biases excluded,
    # unlike pretraining), and an optional per-depth LR ladder. See build_optimizer
    # for the optimiser choice.
    optimizer, groups = build_optimizer(
        raw_model, lr, weight_decay, (beta1, beta2), eps, llrd=llrd, **depth_kwargs)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # Decay after linear warmup, from the peak LR down to a floor of
        # alpha_f * peak. 'linear' is ModernBERT's linear_decay_with_warmup
        # (alpha_f 0.0 -> straight to zero); 'cosine' is the --schedule alternative.
        t = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        decay = (1.0 - t) if schedule == "linear" else 0.5 * (1.0 + math.cos(math.pi * t))
        return alpha_f + (1.0 - alpha_f) * decay

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Screening rung: stop early but leave total_steps (and therefore the whole LR
    # schedule) on the FULL budget, so candidates with different epoch counts are
    # all compared at the same point of their own schedule.
    run_epochs = min(epochs, args.stop_after_epochs) if args.stop_after_epochs else epochs

    use_amp = args.dtype != "fp32" and dev.type == "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    autocast = (
        torch.autocast(device_type=dev.type, dtype=dtype)
        if use_amp else contextlib.nullcontext()
    )

    if rank == 0:
        gpu_label = f" x{world_size} GPUs" if ddp else ""
        init_label = "  init=MNLI" if model_path else ""
        if run_epochs < epochs:
            init_label += f"  [screening: {run_epochs}/{epochs} epochs, full schedule]"
        print(f"  seed={seed}  lr={lr:.1e}  epochs={epochs}  bs={batch_size}{gpu_label}"
              f"  wd={weight_decay:.1e}  betas=({beta1},{beta2})  eps={eps:.0e}"
              f"  sched={schedule}  warmup={warmup_pct}  alpha_f={alpha_f}  dtype={args.dtype}{init_label}")
        n_decayed = sum(gr["n_params"] for gr in groups if not gr["no_decay"])
        n_free = sum(gr["n_params"] for gr in groups if gr["no_decay"])
        fmt = lambda n: f"{n/1e6:.2f}M" if n >= 1e6 else f"{n:,}"   # noqa: E731
        print(f"  no-decay (norms+biases): {fmt(n_free)} params  |  decayed: "
              f"{fmt(n_decayed)} params")
        lrs = [gr["lr"] for gr in groups]
        if llrd != 1.0:
            print(f"  llrd={llrd:g}/layer on the task lr: {len(groups)} param "
                  f"groups, lr {min(lrs):.2e} (embeddings) .. {max(lrs):.2e} (head)")

    best = None
    best_state = None      # best-epoch weights (kept only for the transfer source)
    epochs_since_best = 0
    for epoch in range(run_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with autocast:
                loss, _ = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"],
                                labels=batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()

        preds, labels = evaluate(model, val_loader, dev, regression, autocast)
        # Early stopping: the reported score is the BEST epoch, so once the val
        # metric has not improved for --patience epochs the remaining ones can
        # only burn compute. rank 0 decides; the decision is broadcast so every
        # DDP rank leaves the loop together (the broadcast also doubles as the
        # end-of-epoch barrier).
        stop = False
        improved = False
        if rank == 0:
            scores = compute_metric(metric, preds, labels)
            primary = float(np.mean(list(scores.values())))
            print(f"  [{task}] epoch {epoch+1}/{run_epochs}: {scores}")
            if best is None or primary > best["_primary"]:
                best = {**scores, "_primary": primary}
                epochs_since_best = 0
                improved = True
            else:
                epochs_since_best += 1
            stop = args.patience > 0 and epochs_since_best >= args.patience
        if ddp:
            flags = torch.tensor([int(stop), int(improved)], device=dev)
            dist.broadcast(flags, src=0)
            stop, improved = bool(flags[0].item()), bool(flags[1].item())
        # NeoBERT-style transfer uses the BEST MNLI checkpoint, and early
        # stopping must not change what gets exported — snapshot the weights on
        # every improvement (DDP replicas are identical, so each rank snapshots
        # locally; no communication needed).
        if improved and save_backbone_to is not None:
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in raw_model.state_dict().items()}
        if stop:
            if rank == 0:
                print(f"  [{task}] early stop after epoch {epoch+1}/{run_epochs} "
                      f"(no improvement for {args.patience} epochs; "
                      f"--patience 0 disables)")
            break

    # Export the fine-tuned encoder so a later task can initialise from it
    # (MNLI -> RTE/MRPC/STS-B transfer). DDP keeps every replica identical, so
    # each rank writes its own copy to a rank-local path — no shared filesystem
    # or cross-rank barrier required, which also keeps multi-node runs correct.
    if save_backbone_to is not None:
        if best_state is not None:
            # export the best-epoch weights, not wherever the loop happened to
            # stop (patience-independent, and the actual NeoBERT recipe)
            raw_model.load_state_dict(best_state)
        if is_hf:
            save_hf_glue_backbone(raw_model, save_backbone_to, tokenizer=tokenizer)
        else:
            save_pretrained(raw_model, save_backbone_to, config=config,
                            tokenizer=tokenizer)

    return best


@torch.no_grad()
def evaluate(model, loader, dev, regression, autocast):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(dev)
        attn = batch["attention_mask"].to(dev)
        token_type_ids = batch["token_type_ids"].to(dev)
        with autocast:
            _, logits = model(input_ids=input_ids, attention_mask=attn,
                              token_type_ids=token_type_ids)
        if regression:
            all_preds.append(logits.squeeze(-1).float().cpu())
        else:
            all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(batch["labels"].squeeze(-1).cpu())

    # val_loader is never sharded (each rank sees the full dev set), so there is
    # nothing to all_gather — every rank already holds the complete predictions.
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="pretrained backbone dir or checkpoint; a Hugging Face "
                        "baseline (Hub id or directory, e.g. "
                        "LiquidAI/LFM2.5-Encoder-230M, answerdotai/ModernBERT-base) "
                        "is fine-tuned through the same recipe and head")
    p.add_argument("--tokenizer", default=None,
                   help="default: ModernBERT's for a NexteraBERT backbone, the "
                        "baseline's own for a Hugging Face model")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="GLUE tasks to run; 'all' (default) = the 8 standard tasks "
                        "(cola sst2 mrpc stsb qqp mnli qnli rte; WNLI excluded)")
    p.add_argument("--seeds", nargs="+", type=int, default=[19],
                   help="fine-tuning seeds (default 19, ModernBERT's default_seed); "
                        "per-task scores are the mean (± std) over them. Each task "
                        "caps how many it consumes (rte/mrpc/stsb 5, cola 4, sst2 3, "
                        "large stable tasks 1 — see TASK_MAX_SEEDS), so ModernBERT's "
                        "full set is --seeds 19 8364 717 10536 90166")
    p.add_argument("--uniform_seeds", action="store_true",
                   help="run every task on all --seeds, disabling the per-task caps "
                        "(large stable tasks would otherwise use a single seed)")
    p.add_argument("--no_mnli_transfer", dest="mnli_transfer", action="store_false",
                   help="disable MNLI -> {rte,mrpc,stsb,qnli} intermediate "
                        "fine-tuning; those tasks then start from the cold "
                        "pretrained backbone")
    p.set_defaults(mnli_transfer=True)
    p.add_argument("--mnli_cache_dir", default=None,
                   help="directory for the transient MNLI-tuned encoder used for "
                        "transfer (default: system temp)")
    p.add_argument("--mnli_source", default=None,
                   help="reuse an ALREADY fine-tuned MNLI encoder (a directory "
                        "written by a previous run's transfer export, e.g. "
                        "search_glue.py's work_dir/mnli_best) instead of training "
                        "one. rte/mrpc/stsb/qnli start from it "
                        "immediately. The directory is never deleted. If it also "
                        "holds a glue_score.json for the first --seeds entry, that "
                        "score is reused for the mnli task itself; otherwise mnli "
                        "(when requested) is trained as an ordinary task.")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs for all tasks (default: per-task)")
    p.add_argument("--mnli_epochs", type=int, default=None,
                   help="override epochs for MNLI only (beats --epochs there). "
                        "ModernBERT trains the transfer source for 1 epoch; raise "
                        "this for a better-converged encoder behind rte/mrpc/stsb "
                        "(early stopping still reports/exports the best epoch).")
    p.add_argument("--patience", type=int, default=2,
                   help="early-stop a fine-tuning run after this many epochs without "
                        "val improvement. Heuristic: plateaued runs rarely improve "
                        "again, but ties count as no improvement, so a late best "
                        "epoch can occasionally be missed. 0 disables and always "
                        "trains the full per-task epoch budget.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override batch size for all tasks (default: per-task)")
    p.add_argument("--lr", type=float, default=None,
                   help="override learning rate for all tasks (default: per-task)")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="override weight decay for all tasks. A TORCH AdamW weight "
                        "decay (per-step decay lr_t*wd*p), used verbatim — a composer "
                        "DecoupledAdamW value converts as wd/lr. Default: per-task "
                        f"(TASK_DEFAULTS_MODERNBERT), {DEFAULT_WEIGHT_DECAY} for a "
                        "task absent from the table")
    p.add_argument("--llrd", type=float, default=None,
                   help="layerwise LR decay (LLRD) factor in (0, 1], applied as a "
                        "MULTIPLIER on the task's own LR (the table's, an "
                        "--hparams entry, or --lr): the head keeps that LR, block "
                        "n_layer trains at llrd x it, and so on down to the "
                        "embeddings at llrd**(n_layer+1) x it. The standard "
                        "BERT/DeBERTa discriminative fine-tuning trick — it keeps a "
                        "small task (RTE/MRPC/CoLA) from overwriting the pretrained "
                        f"lower layers. Default {DEFAULT_LLRD}; typical 0.8-0.95; 1.0 "
                        "(or --no_llrd) is a flat LR, the protocol as published. "
                        "Per-task values can also come from --hparams.")
    p.add_argument("--no_llrd", dest="llrd_enabled", action="store_false",
                   help="disable layerwise LR decay: every depth trains at the task's "
                        "LR (the ModernBERT protocol as published)")
    p.set_defaults(llrd_enabled=True)
    p.add_argument("--hparams", default=None,
                   help="JSON file of per-task hyperparameters that overrides the "
                        "built-in table: {\"rte\": {\"lr\": 3e-5, \"weight_decay\": "
                        "0.1, \"epochs\": 3, \"batch_size\": 8, \"llrd\": 0.9}, ...}. "
                        "Exactly the "
                        "format search_glue.py writes (top-level keys starting with "
                        "'_' are ignored). Explicit --lr/--weight_decay/--epochs/"
                        "--batch_size/--llrd still win over the file.")
    p.add_argument("--train_subsample", type=int, default=0,
                   help="cap each task's TRAINING set at this many examples — a "
                        "seed-fixed random subset (independent of --seeds, so every "
                        "candidate of a hyperparameter search sees identical data). "
                        "0 (default) = full data. Proxy for search runs only; never "
                        "use it for reported scores.")
    p.add_argument("--stop_after_epochs", type=int, default=0,
                   help="train at most this many epochs while keeping the LR "
                        "schedule stretched over the FULL per-task epoch budget — "
                        "the screening rung of a successive-halving search, where "
                        "candidates are compared mid-schedule on equal footing. "
                        "0 (default) = run the full budget.")
    # AdamW / schedule globals. Default None = take the value from OPTIM_DEFAULTS;
    # per-task lr/bs/wd/epochs come from TASK_DEFAULTS_MODERNBERT.
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--schedule", choices=["linear", "cosine"], default=None,
                   help="post-warmup LR decay shape (default: OPTIM_DEFAULTS)")
    p.add_argument("--warmup_pct", type=float, default=None)
    p.add_argument("--alpha_f", type=float, default=None)
    # 256 is ModernBERT's GLUE max_sequence_length (glue_jobs.py, every task).
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no_compile", dest="compile", action="store_false",
                   help="disable torch.compile (on by default on CUDA, matching the "
                        "pretraining speed setup)")
    p.set_defaults(compile=True)
    p.add_argument("--task_parallel", action="store_true",
                   help="run each GLUE task on its OWN single GPU, tasks in parallel "
                        "across GPUs (instead of DDP-sharding one task over all GPUs). "
                        "This is what reproduces the recipe exactly: the per-task "
                        "batch sizes are totals, and single-GPU-per-task keeps them "
                        "intact (DDP inflates the effective batch by num_gpus). The "
                        "MNLI transfer source trains first; its dependents "
                        "(rte/mrpc/stsb/qnli) wait for it while every other task "
                        "starts immediately.")
    p.add_argument("--num_gpus", type=int, default=None,
                   help="number of GPUs to schedule across for --task_parallel "
                        "(default: all visible CUDA devices)")
    # Internal flags: --task_parallel re-invokes this script per (task, seed) with a
    # single GPU pinned via CUDA_VISIBLE_DEVICES. Not intended for direct use.
    p.add_argument("--worker_task", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker_seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker_init", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker_save_backbone", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker_out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--output", default="glue_results.json")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="log GLUE results to this W&B project")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    args = p.parse_args()
    # a Hub id becomes a local snapshot once, so every worker / seed / task reads
    # the same directory and the loaders can dispatch on its model_type
    args.model = resolve_hub_model(args.model)
    if args.tokenizer is None:
        args.tokenizer = (args.model if is_hf_model_dir(args.model)
                          else "answerdotai/ModernBERT-base")
    return args


def aggregate_runs(runs):
    """Average a task's per-seed score dicts into mean (+ std when >1 seed)."""
    metric_keys = [k for k in runs[0] if not k.startswith("_")]
    agg = {}
    for k in metric_keys:
        vals = [r[k] for r in runs]
        agg[k] = float(np.mean(vals))
        if len(vals) > 1:
            agg[f"{k}_std"] = float(np.std(vals))
    primaries = [r["_primary"] for r in runs]
    agg["_primary"] = float(np.mean(primaries))
    if len(primaries) > 1:
        agg["_primary_std"] = float(np.std(primaries))
    agg["_n_seeds"] = len(runs)
    return agg


def external_mnli_source(args):
    """Resolve --mnli_source into (encoder_dir, cached_mnli_score).

    The directory is an MNLI-tuned encoder exported by an earlier run (typically
    search_glue.py's work_dir/mnli_best, trained on FULL data with the winning
    recipe), so the dependents can start from it without retraining. If it also
    carries a glue_score.json for this run's first seed — i.e. the very run the
    final evaluation would otherwise repeat — that score is reused for the mnli
    task itself. Returns (None, None) when no source was given."""
    if not args.mnli_source:
        return None, None
    src = Path(args.mnli_source)
    if not src.is_dir():
        sys.exit(f"--mnli_source: not a directory: {src}")
    if not args.mnli_transfer:
        print("[glue][warn] --mnli_source ignored (--no_mnli_transfer was passed)")
        return None, None
    score = None
    score_file = src / "glue_score.json"
    if score_file.exists():
        try:
            rec = json.loads(score_file.read_text(encoding="utf-8"))
            # Seed must match, or the reused number would not be the one this
            # run's protocol produces.
            if rec.get("task") == "mnli" and rec.get("seed") == args.seeds[0]:
                score = rec.get("best")
        except (OSError, json.JSONDecodeError, TypeError) as e:
            print(f"[glue][warn] unreadable {score_file}: {e}")
    return str(src), score


def run_serial(args):
    """Sequential execution: tasks run one after another, each optionally sharded
    across GPUs via DDP (under torchrun). This is the fallback path; --task_parallel
    (one task per GPU, tasks concurrent) is preferred for reproducing the recipe."""
    # torchrun sets LOCAL_RANK — init DDP if present
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        # set_device first, and pass device_id, so NCCL binds the group to this
        # rank's GPU instead of inferring it from the current context (which warns).
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    rank = get_rank()
    world_size = get_world_size()

    if rank == 0 and world_size > 1:
        print(f"DDP active: {world_size} GPUs")

    # RTE/MRPC/STS-B/QNLI initialise from an MNLI-tuned encoder.
    init_tasks = MNLI_INIT_TASKS
    dependents = [t for t in args.tasks if t in init_tasks]
    need_mnli = args.mnli_transfer and len(dependents) > 0
    if rank == 0 and need_mnli:
        print(f"[glue] MNLI-transfer ON: {dependents} initialise from an MNLI-tuned "
              f"encoder (disable with --no_mnli_transfer).")

    cache_root = args.mnli_cache_dir or tempfile.gettempdir()
    model_tag = hashlib.md5(str(Path(args.model).resolve()).encode()).hexdigest()[:8]

    def seeds_for(task):
        """Seeds this task consumes — the per-task cap unless --uniform_seeds."""
        if args.uniform_seeds:
            return args.seeds
        cap = TASK_MAX_SEEDS.get(task, len(args.seeds))
        return args.seeds[:max(1, min(cap, len(args.seeds)))]

    per_task_runs = {t: [] for t in args.tasks}
    mnli_dirs = []

    # MNLI transfer source: trained ONCE (first seed) and shared by every seed of
    # RTE/MRPC/STS-B. MNLI is stable, so a single seed is the reference recipe and
    # avoids retraining the most expensive task once per downstream seed.
    # --mnli_source supplies that encoder from an earlier run instead.
    ext_mnli, ext_mnli_score = external_mnli_source(args)
    mnli_backbone = None
    mnli_scored = False        # is the mnli task's own score already accounted for?
    if need_mnli and ext_mnli:
        mnli_backbone = ext_mnli
        if rank == 0:
            print(f"[glue] reusing the MNLI transfer source at {ext_mnli} "
                  f"(not retrained, not deleted)")
        if "mnli" in args.tasks and ext_mnli_score is not None:
            per_task_runs["mnli"].append(ext_mnli_score)
            mnli_scored = True
            if rank == 0:
                print(f"[glue] reusing its recorded mnli score "
                      f"(seed {args.seeds[0]}, primary={ext_mnli_score['_primary']:.4f})")
        elif "mnli" in args.tasks and rank == 0:
            print("[glue] no matching glue_score.json in the source — mnli will be "
                  "trained as an ordinary task for its own score")
    elif need_mnli:
        src_seed = args.seeds[0]
        # rank-local path — every replica writes its own identical copy, so no
        # shared filesystem or extra barrier is needed (multi-node safe).
        mnli_backbone = os.path.join(cache_root, f"nb_glue_mnli_{model_tag}_r{rank}")
        mnli_dirs.append(mnli_backbone)
        if rank == 0:
            print(f"=== GLUE: mnli (transfer source) · seed {src_seed} ===")
        mnli_best = run_task("mnli", args, src_seed, save_backbone_to=mnli_backbone)
        mnli_scored = True
        if "mnli" in args.tasks and mnli_best is not None:
            per_task_runs["mnli"].append(mnli_best)

    for task in args.tasks:
        if task == "mnli" and mnli_scored:
            continue   # already trained above as the transfer source (or reused)
        init_path = mnli_backbone if task in init_tasks else None
        task_seeds = seeds_for(task)
        for seed in task_seeds:
            if rank == 0:
                tag = "  (init: MNLI)" if init_path else ""
                nseed = f" · seed {seed}" if len(task_seeds) > 1 else ""
                print(f"=== GLUE: {task}{nseed}{tag} ===")
            best = run_task(task, args, seed, model_path=init_path)
            if best is not None:
                per_task_runs[task].append(best)

    # each rank removes the MNLI checkpoints it wrote
    for d in mnli_dirs:
        shutil.rmtree(d, ignore_errors=True)

    if rank == 0:
        finalize_results(args, per_task_runs, need_mnli)

    if is_ddp():
        dist.destroy_process_group()


def finalize_results(args, per_task_runs, need_mnli):
    """Aggregate per-task/per-seed runs, print the GLUE table, optionally log to
    W&B, and write the results JSON. Shared by run_serial and run_task_parallel."""
    results = {}
    for task in args.tasks:
        if per_task_runs.get(task):
            results[task] = aggregate_runs(per_task_runs[task])

    avg_tasks = [t for t in results if t != "wnli"]
    avg = (float(np.mean([results[t]["_primary"] for t in avg_tasks]))
           if avg_tasks else 0.0)
    results["glue_avg"] = avg
    results["_meta"] = {
        "seeds": list(args.seeds),
        "mnli_transfer": need_mnli,
        "avg_tasks": avg_tasks,
    }

    print(f"\nGLUE average ({len(avg_tasks)} tasks, WNLI excluded): {avg:.4f}")
    for t in avg_tasks:
        std = results[t].get("_primary_std")
        extra = f" ± {std:.4f}" if std is not None else ""
        n = results[t].get("_n_seeds", 1)   # per-task: TASK_MAX_SEEDS caps --seeds
        print(f"  {t:5s} {results[t]['_primary']:.4f}{extra}  "
              f"({n} seed{'s' if n != 1 else ''})")
    if "wnli" in results:
        print(f"  (wnli {results['wnli']['_primary']:.4f} — reported, not in average)")

    # Write the results JSON BEFORE touching W&B: by this point the evaluation
    # is fully done (and task-parallel has already deleted the per-worker temp
    # results), so a wandb.init failure — invalid/rotated API key, no network,
    # not logged in — must not lose the aggregated scores.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"results -> {args.output}")

    if args.wandb_project is not None:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"glue-{Path(args.model).name}",
            config={k: v for k, v in vars(args).items()
                    if k not in ("wandb_project", "wandb_entity", "wandb_run_name")},
        )
        for task, scores in results.items():
            if isinstance(scores, dict) and "_primary" in scores:
                for k, v in scores.items():
                    if not k.startswith("_"):
                        wandb.summary[f"glue_{task}_{k}"] = v
                wandb.log({f"glue/{task}/primary": scores["_primary"]})
        wandb.log({"glue/average": avg})
        wandb.summary["glue_avg"] = avg
        wandb.finish()

    return results


def run_worker(args):
    """Internal: fine-tune exactly one (task, seed) on the single visible GPU and
    write its best-epoch scores to args.worker_out. Spawned by run_task_parallel
    with CUDA_VISIBLE_DEVICES pinned to one device, so run_task sees only cuda:0."""
    best = run_task(
        args.worker_task, args, args.worker_seed,
        model_path=args.worker_init or None,
        save_backbone_to=args.worker_save_backbone or None,
    )
    with open(args.worker_out, "w", encoding="utf-8") as f:
        json.dump({"task": args.worker_task, "seed": args.worker_seed, "best": best}, f)


def _worker_cmd(args, job):
    """Build the subprocess argv that runs `job` as a single-GPU worker."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--max_len", str(args.max_len),
        "--dtype", args.dtype,
        "--patience", str(args.patience),
        "--worker_task", job["task"],
        "--worker_seed", str(job["seed"]),
        "--worker_out", job["out"],
    ]
    # The optimiser/schedule globals default to None ("use OPTIM_DEFAULTS"), so only
    # forward the ones actually overridden — passing "None" through argparse would
    # fail its float/choices conversion.
    for flag in ("beta1", "beta2", "eps", "schedule", "warmup_pct", "alpha_f"):
        val = getattr(args, flag)
        if val is not None:
            cmd += [f"--{flag}", str(val)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.mnli_epochs is not None:
        cmd += ["--mnli_epochs", str(args.mnli_epochs)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.weight_decay is not None:
        cmd += ["--weight_decay", str(args.weight_decay)]
    if args.llrd is not None:
        cmd += ["--llrd", str(args.llrd)]
    # LLRD is ON by default, so the OFF direction has to be forwarded too: a
    # worker that inherited the default instead of the parent's flags would
    # silently fine-tune under a different recipe than the one asked for.
    if not args.llrd_enabled:
        cmd += ["--no_llrd"]
    if args.hparams:
        cmd += ["--hparams", os.path.abspath(args.hparams)]
    if args.train_subsample:
        cmd += ["--train_subsample", str(args.train_subsample)]
    if args.stop_after_epochs:
        cmd += ["--stop_after_epochs", str(args.stop_after_epochs)]
    if not args.compile:
        cmd += ["--no_compile"]
    if job.get("init"):
        cmd += ["--worker_init", job["init"]]
    if job.get("save_bb"):
        cmd += ["--worker_save_backbone", job["save_bb"]]
    return cmd


def run_task_parallel(args):
    """Run each GLUE task on its own single GPU, tasks in parallel across GPUs.

    Each (task, seed) is a separate single-GPU subprocess (CUDA_VISIBLE_DEVICES
    pinned), so the table's batch sizes — which are totals — are reproduced
    exactly instead of being inflated by DDP. A simple GPU-pool scheduler respects
    the one dependency in the recipe: the MNLI transfer source trains first, its
    dependents (rte/mrpc/stsb/qnli) wait for it, and every task starts as
    soon as a GPU is free.
    """
    num_gpus = args.num_gpus if (args.num_gpus and args.num_gpus > 0) else torch.cuda.device_count()
    num_gpus = max(1, num_gpus)
    print(f"[glue] task-parallel across {num_gpus} GPU(s): one task per GPU (single-GPU each)")

    init_tasks = MNLI_INIT_TASKS
    dependents = [t for t in args.tasks if t in init_tasks]
    need_mnli = args.mnli_transfer and len(dependents) > 0
    if need_mnli:
        print(f"[glue] MNLI-transfer ON: {dependents} initialise from an MNLI-tuned "
              f"encoder (disable with --no_mnli_transfer).")

    cache_root = args.mnli_cache_dir or tempfile.gettempdir()
    model_tag = hashlib.md5(str(Path(args.model).resolve()).encode()).hexdigest()[:8]
    work_dir = tempfile.mkdtemp(prefix="nb_glue_par_")
    ext_mnli, ext_mnli_score = external_mnli_source(args)
    mnli_dir = None
    if need_mnli:
        mnli_dir = ext_mnli or os.path.join(cache_root, f"nb_glue_mnli_{model_tag}")

    def seeds_for(task):
        if args.uniform_seeds:
            return args.seeds
        cap = TASK_MAX_SEEDS.get(task, len(args.seeds))
        return args.seeds[:max(1, min(cap, len(args.seeds)))]

    # Build the job list. The MNLI source (first seed) is trained once and shared by
    # every dependent seed, exactly as in the serial path — unless --mnli_source
    # already supplies that encoder, in which case nothing blocks the dependents.
    per_task_runs = {t: [] for t in args.tasks}
    jobs = []
    mnli_scored = False        # is the mnli task's own score already accounted for?
    if need_mnli and ext_mnli:
        print(f"[glue] reusing the MNLI transfer source at {ext_mnli} "
              f"(not retrained, not deleted) — dependents start immediately")
        if "mnli" in args.tasks and ext_mnli_score is not None:
            per_task_runs["mnli"].append(ext_mnli_score)
            mnli_scored = True
            print(f"[glue] reusing its recorded mnli score "
                  f"(seed {args.seeds[0]}, primary={ext_mnli_score['_primary']:.4f})")
        elif "mnli" in args.tasks:
            print("[glue] no matching glue_score.json in the source — mnli will be "
                  "trained as an ordinary task for its own score")
    elif need_mnli:
        jobs.append({"task": "mnli", "seed": args.seeds[0], "init": None,
                     "save_bb": mnli_dir, "needs_mnli": False, "is_mnli_src": True})
        mnli_scored = True
    for task in args.tasks:
        if task == "mnli" and mnli_scored:
            continue   # already scheduled as the transfer source (or reused)
        is_dep = need_mnli and task in init_tasks
        for seed in seeds_for(task):
            jobs.append({"task": task, "seed": seed,
                         "init": mnli_dir if is_dep else None,
                         "save_bb": None, "needs_mnli": is_dep, "is_mnli_src": False})
    for i, j in enumerate(jobs):
        j["out"] = os.path.join(work_dir, f"res_{i:03d}_{j['task']}_{j['seed']}.json")

    free = list(range(num_gpus))
    pending = list(jobs)
    running = {}                       # proc -> (job, gpu)
    mnli_done = (not need_mnli) or bool(ext_mnli)   # a reused source is ready now

    def collect(job, ret):
        best = None
        if ret == 0 and os.path.exists(job["out"]):
            try:
                with open(job["out"], encoding="utf-8") as f:
                    best = json.load(f).get("best")
            except (OSError, json.JSONDecodeError) as e:
                print(f"[glue][error] unreadable result for {job['task']} "
                      f"seed={job['seed']}: {e}")
        if ret != 0:
            print(f"[glue][error] {job['task']} seed={job['seed']} exited with code {ret}")
        if job["is_mnli_src"]:
            if "mnli" in args.tasks and best is not None:
                per_task_runs["mnli"].append(best)
        elif best is not None:
            per_task_runs[job["task"]].append(best)

    while pending or running:
        progressed = False
        for job in list(pending):
            if not free:
                break
            if job["needs_mnli"] and not mnli_done:
                continue
            gpu = free.pop()
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(_worker_cmd(args, job), env=env)
            running[proc] = (job, gpu)
            pending.remove(job)
            progressed = True
            tag = "  (init: MNLI)" if job["init"] else ""
            src = "  [transfer source]" if job["is_mnli_src"] else ""
            print(f"[glue] launch {job['task']} seed={job['seed']} -> GPU{gpu}{tag}{src}")
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            job, gpu = running.pop(proc)
            free.append(gpu)
            progressed = True
            collect(job, ret)
            if job["is_mnli_src"]:
                # Release dependents even on failure (they will then error out
                # loudly on the missing backbone) so the scheduler never deadlocks.
                mnli_done = True
                print("[glue] MNLI transfer source finished — dependents released")
        if not progressed:
            time.sleep(2)

    shutil.rmtree(work_dir, ignore_errors=True)
    if mnli_dir and not ext_mnli:
        # only the transient encoder this run trained — never a --mnli_source
        shutil.rmtree(mnli_dir, ignore_errors=True)
    finalize_results(args, per_task_runs, need_mnli)


def main():
    args = parse_args()
    if args.llrd is not None and not 0.0 < args.llrd <= 1.0:
        raise SystemExit(f"--llrd must be in (0, 1] (got {args.llrd}); "
                         "1.0 disables layerwise decay")
    if args.tasks == ["all"]:
        args.tasks = list(GLUE_AVG_TASKS)   # 8 standard tasks (WNLI excluded)

    # Match the pretraining throughput setup: TF32 matmuls (large Ampere+ speed-up at
    # negligible accuracy cost).
    torch.set_float32_matmul_precision("high")

    # Internal single-(task, seed) worker spawned by --task_parallel.
    if args.worker_task is not None:
        run_worker(args)
        return

    # Task-parallel (one GPU per task) unless launched under torchrun (LOCAL_RANK set).
    if int(os.environ.get("LOCAL_RANK", -1)) < 0 and args.task_parallel:
        run_task_parallel(args)
        return

    run_serial(args)


if __name__ == "__main__":
    main()
