#!/usr/bin/env python
"""ELECTRA-style or BERT-style (MLM) pretraining for NexteraBERT with optional DDP.

Single GPU / CPU:
    python scripts/pretrain.py --config configs/pretrain_bert_phase1.yaml

Multi-GPU (DDP, single node, 8 GPUs):
    torchrun --standalone --nproc_per_node=8 scripts/pretrain.py \
        --config configs/pretrain_bert_phase1.yaml

Multi-node:
    torchrun --nnodes=2 --node_rank=$RANK --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        scripts/pretrain.py --config configs/pretrain_bert_phase1.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

# A fast (Rust) tokenizer used inside forked DataLoader workers (num_workers > 0)
# deadlocks unless tokenizer parallelism is disabled before the fork. Without this
# training hangs on the first batch. Set before transformers/tokenizers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Growable (VM-backed) allocator segments. A long-context phase mixes fixed 8192
# packed blocks with varlen short-replay batches; the many distinct allocation
# sizes fragment the fixed-size-segment caching allocator badly (observed on a
# phase-2 run: 64 GiB "reserved but unallocated" at the moment of an OOM).
# Expandable segments grow/shrink in place instead of pinning whole segments to
# one size class. setdefault: an explicit user setting always wins. Must be set
# before the first CUDA allocation; harmless on CPU-only runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert import (  # noqa: E402
    NexteraBERT,
    NexteraBERTConfig,
    NexteraBERTForElectraTrainer,
    NexteraBERTForBertTrainer,
    NexteraBERTForCocoLMTrainer,
)
from nexterabert.data import (  # noqa: E402
    CudaPrefetchLoader,
    PackedStreamDataset,
    PretokenizedDataset,
    VarLengthDataset,
    build_tokenizer,
    collate,
    is_varlen_shard,
    make_dynamic_pad_collate,
    padded_vocab_size,
)
from nexterabert.optim import build_optimizer  # noqa: E402
from nexterabert.training_utils import (  # noqa: E402
    WarmupCosineDecay,
    WarmupStableDecay,
    cleanup_distributed,
    count_parameters,
    freeze_all_but,
    save_checkpoint,
    setup_distributed,
)


def _arch_overrides(value):
    """``--arch_overrides`` accepts a JSON string (CLI) or an already-parsed mapping
    (YAML config), and always yields a dict.

    argparse's ``type`` is also what parse_args() uses to cast a value that arrived
    from the YAML file, and PyYAML hands that one over as a real dict — so this has
    to be idempotent on dicts rather than only parsing strings.
    """
    if value is None or isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--arch_overrides must be a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    # data
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_mix", type=str, default=None,
                   help="JSON list or YAML list of dataset sources with weights; "
                        "overrides --dataset_name for the streaming fallback")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--pretokenized", type=str, default=None,
                   help="path to a prepare_data.py shard; overrides dataset_name "
                        "and gives an instant, deterministic start")
    p.add_argument("--tokenizer", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--pad_multiple_of", type=int, default=8,
                   help="for a variable-length (--pack_mode docs) shard, the dynamic-pad "
                        "collate rounds each batch's length up to a multiple of this "
                        "(>= NexteraHRA block_size). Larger = fewer distinct sequence "
                        "lengths for torch.compile, at the cost of a little more padding.")
    p.add_argument("--shuffle_buffer", type=int, default=1000,
                   help="streaming shuffle-buffer size (docs filled before the first "
                        "batch); lower it for a faster start, 0 disables shuffling")
    # held-out evaluation data (MosaicBERT-style: eval on a separate split, not the
    # training stream). Defaults to streaming the dataset's `validation` split so
    # eval never overlaps the training data. Falls back to the training source if
    # unset / the split is unavailable.
    p.add_argument("--eval_dataset_name", type=str, default=None,
                   help="dataset for held-out eval; defaults to --dataset_name")
    p.add_argument("--eval_dataset_config", type=str, default=None,
                   help="config for the eval dataset; defaults to --dataset_config")
    p.add_argument("--eval_split", type=str, default="validation",
                   help="split used for held-out eval (streamed, no shuffle)")
    p.add_argument("--eval_pretokenized", type=str, default=None,
                   help="optional pre-tokenised held-out shard; overrides the "
                        "streaming eval split for an instant, deterministic eval set")
    # Short-context replay during a long-context extension phase (ProLong-style).
    # Training a phase-2 extension on ONLY long packed blocks reliably degrades
    # short-sequence quality (and re-fits the SSMax length scale to the long
    # regime); mixing ~40% short batches back in is the established fix.
    p.add_argument("--short_pretokenized", type=str, default=None,
                   help="optional short-context shard replayed during long-context "
                        "training; --short_ratio of optimizer steps draw all their "
                        "micro-batches from it instead of --pretokenized")
    p.add_argument("--short_ratio", type=float, default=0.4,
                   help="fraction of optimizer steps drawn from --short_pretokenized "
                        "(ProLong's short-data share); only used when that shard is set")
    p.add_argument("--short_batch_size", type=int, default=None,
                   help="per-device micro-batch for short replay steps; defaults to "
                        "batch_size * (max_seq_len // short-shard max_seq_len) so a "
                        "short micro-batch carries the same token count as a long one")
    p.add_argument("--short_eval_pretokenized", type=str, default=None,
                   help="optional held-out SHORT shard (e.g. the phase-1 eval slice) "
                        "scored at every --eval_every next to --eval_pretokenized, as "
                        "'eval_short/*'. Tracks whether a long-context phase keeps its "
                        "short-sequence quality (the SSSMax scale is length-dependent). "
                        "Batch = --short_batch_size, else token-parity with batch_size.")
    # Partial fine-tuning: train only the parameters whose name contains one of
    # these substrings and freeze everything else (requires_grad=False, so they
    # are outside the optimizer and DDP entirely). The SSSMax re-fit uses
    # "scalable_factor" to move only the per-layer (s, b) scale scalars.
    p.add_argument("--train_only", type=str, nargs="*", default=None,
                   metavar="SUBSTR",
                   help="freeze every parameter whose name contains none of these "
                        "substrings (e.g. scalable_factor). Empty/unset = train all.")
    # model
    p.add_argument("--training_mode", type=str, default="electra",
                   choices=["electra", "bert", "cocolm"],
                   help="'electra' = generator+discriminator ELECTRA training; "
                        "'bert' = single-model MLM with 80/10/10 masking; "
                        "'cocolm' = COCO-LM (corrective LM + Matryoshka sequence "
                        "contrastive learning) on top of an ELECTRA-style generator")
    p.add_argument("--disc_size", type=str, default="mezzoforte")
    p.add_argument("--gen_size", type=str, default="piano")
    # Architecture switch. Defaults to None meaning "leave it to
    # NexteraBERTConfig's own default", so the library stays the single source of
    # truth for the default architecture and this only ever overrides it.
    p.add_argument("--mlp_class", type=str, default=None, choices=["mlp", "ugm"],
                   help="feed-forward inside every block: 'mlp' = dense squared-ReLU; "
                        "'ugm' = routed RippleBloomUGM (activates the router z-loss "
                        "and load-balance auxiliary losses)")
    p.add_argument("--arch_overrides", type=_arch_overrides, default=None,
                   metavar="JSON",
                   help="JSON object of NexteraBERTConfig fields that override "
                        "the --disc_size preset for the MAIN model, e.g. "
                        "{\"n_embd\": 768, \"n_layer\": 12, \"block_types\": [...]}. "
                        "In a YAML config the same key may be given as a mapping. "
                        "Whatever is set here is baked into the exported config.json, "
                        "so the checkpoint rebuilds the architecture it was trained "
                        "with. Used by the paper's ablation runs.")
    p.add_argument("--rope_ntk_train_len", type=int, default=0,
                   help="baked into the model config: enable dynamic NTK-aware RoPE "
                        "scaling for sequences LONGER than this trained length "
                        "(train-free context extension; 0 disables). Sequences at or "
                        "below it are unchanged, so short-context behaviour is "
                        "untouched. Set to 1024 to serve/extend a phase-1-only "
                        "checkpoint beyond its trained context without a phase 2.")
    p.add_argument("--mask_ratio", type=float, default=0.15,
                   help="ELECTRA mask ratio (phase 2); keep ~0.15 so the generator "
                        "has enough context to produce non-trivial replacements")
    # BERT-mode: ramp the MLM mask ratio over the run instead of a fixed value.
    p.add_argument("--mask_ratio_start", type=float, default=None,
                   help="BERT mode only: if set (with --mask_ratio_end), the MLM mask "
                        "ratio is scheduled from this value at step 0 ...")
    p.add_argument("--mask_ratio_end", type=float, default=None,
                   help="... up to this value at max_steps. Enables the mask-ratio schedule.")
    p.add_argument("--mask_ratio_power", type=float, default=2.0,
                   help="convexity of the mask-ratio ramp: ratio = start + (end-start) * "
                        "(step/max_steps)**power. power>1 rises gently early and steeper "
                        "late (2.0 = quadratic); 1.0 = linear.")
    p.add_argument("--gen_warmup_steps", type=int, default=0,
                   help="train ONLY the generator for this many steps (discriminator "
                        "is not run/updated) before starting joint ELECTRA training")
    p.add_argument("--gen_warmup_frac", type=float, default=None,
                   help="if set, gen_warmup_steps = gen_warmup_frac * max_steps, resolved "
                        "after --train_tokens/--epochs fix max_steps. Keeps the "
                        "generator-only phase a fixed SHARE of the run when the token "
                        "budget changes (an absolute step count would silently become "
                        "most of a small budget). Overrides --gen_warmup_steps.")
    p.add_argument("--gen_warmup_mask_ratio", type=float, default=0.3,
                   help="mask ratio used during the generator-only warmup phase")
    p.add_argument("--head_warmup_steps", type=int, default=0,
                   help="for the first N optimizer steps, update ONLY the non-encoder "
                        "head parameters (prediction-head transform, decoder bias, "
                        "RTD/copy heads) by dropping encoder gradients before the "
                        "step. Use together with --resume_weights_only when resuming "
                        "a checkpoint that predates the ModernBERT-style heads, so "
                        "the fresh-init heads settle against the trained encoder "
                        "before their gradients can distort it.")
    p.add_argument("--disc_lambda", type=float, default=50.0)
    p.add_argument("--router_z_loss_coef", type=float, default=1e-3,
                   help="ST-MoE router z-loss weight on the RippleBloomUGM routers; "
                        "0 disables it")
    p.add_argument("--gen_temperature", type=float, default=1.0,
                   help="generator sampling temperature; >1 = harder fakes")
    p.add_argument("--load_balance_loss_coef", type=float, default=1e-2,
                   help="MoE load-balance auxiliary loss weight; 0 disables it")
    p.add_argument("--tie_gen_disc_embeddings",
                   action=argparse.BooleanOptionalAction, default=True)
    # COCO-LM (training_mode=cocolm) — the main model = disc_size, generator = gen_size
    p.add_argument("--copy_lambda", type=float, default=50.0,
                   help="COCO-LM: weight of the binary copy/RTD detection loss "
                        "(paper default 50)")
    p.add_argument("--scl_lambda", type=float, default=1.0,
                   help="COCO-LM: weight of the sequence-contrastive (SCL) loss")
    p.add_argument("--scl_temperature", type=float, default=1.0,
                   help="COCO-LM SCL InfoNCE temperature (paper default 1.0; the "
                        "reference MNRL uses ~0.05)")
    p.add_argument("--scl_proj_dim", type=int, default=512,
                   help="COCO-LM SCL projection width (largest Matryoshka dim)")
    p.add_argument("--scl_crop_ratio", type=float, default=0.9,
                   help="COCO-LM SCL: fraction of the original sequence kept for "
                        "the cropped positive view")
    p.add_argument("--matryoshka_dims", type=str, default="512,256,128,64,32",
                   help="COCO-LM SCL: comma-separated nested embedding dims (only "
                        "those <= scl_proj_dim are used)")
    # optimisation
    p.add_argument("--batch_size", type=int, default=32, help="per-device micro batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    # AdamW / schedule defaults follow OptiBERT (Dervishi et al., EMNLP 2025,
    # Appendix A): standard AdamW, fixed wd 0.1, betas (0.9, 0.95), eps 1e-8,
    # linear warmup then cosine decay to 10% of the peak LR. --wd_mode decoupled
    # + --lr_schedule wsd switches to the ModernBERT recipe instead (wd 1e-5
    # fully decoupled and exempting biases/norms, beta2 0.98, eps 1e-6,
    # trapezoidal LR with a 1-sqrt decay tail); see nexterabert.optim. Both
    # recipes run torch.optim.AdamW, which always applies Adam's bias
    # correction — never substitute a legacy BERTAdam-style optimizer (Zhang
    # et al., 2021, "Revisiting Few-sample BERT Fine-tuning").
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--wd_mode", choices=["standard", "decoupled"], default="standard",
                   help="'standard' (OptiBERT, default): plain torch AdamW over every "
                        "parameter, per-step shrink lr_t*wd*p. 'decoupled' "
                        "(ModernBERT): shrink is schedule_factor*wd*p, independent of "
                        "the LR magnitude, and biases/norms are exempt. The two "
                        "conventions' wd values are NOT interchangeable — 0.1 read as "
                        "decoupled would shrink weights 10%%/step.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--lr_schedule", choices=["cosine", "wsd"], default="cosine",
                   help="'cosine' (OptiBERT/NeoBERT, default): warmup then cosine "
                        "decay to 10%% of peak. 'wsd' (ModernBERT): linear warmup, "
                        "constant peak LR, then a 1-sqrt decay over the last "
                        "--decay_frac of the run.")
    p.add_argument("--decay_frac", type=float, default=0.0,
                   help="--lr_schedule wsd only (ignored by cosine): fraction of the "
                        "run spent in the 1-sqrt decay tail. 0 = warmup + constant "
                        "with no anneal, which is ModernBERT-base's 1024-token phase; "
                        "it spends the decay at the end of the long-context phase "
                        "instead (50 B of its 300 B tokens = 0.167).")
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--epochs", type=float, default=None,
                   help="if set (and the shard is map-style, e.g. --pack_mode docs), "
                        "derive max_steps = epochs * n_docs / (batch * grad_accum * gpus) "
                        "from the actual document count, and drive the LR schedule from it. "
                        "Makes the token budget exact for variable-length shards without "
                        "guessing the average document length. Overrides --max_steps.")
    p.add_argument("--train_tokens", type=float, default=None,
                   help="target TRAINING TOKEN budget (e.g. 13e9). max_steps is derived "
                        "from the shard's real token statistics at startup: a "
                        "variable-length (--pack_mode docs) shard is sized by its own "
                        "mean segment length, a fixed-block shard by max_seq_len. Counts "
                        "CONTENT tokens — collate padding is masked out and carries no "
                        "data, so it is excluded. Overrides --max_steps and --epochs.")
    p.add_argument("--warmup_frac", type=float, default=None,
                   help="if set, warmup_steps = warmup_frac * max_steps (computed after "
                        "--epochs resolves max_steps), so the LR reaches its peak at exactly "
                        "this fraction of the run. Overrides --warmup_steps.")
    p.add_argument("--warmup_steps", type=int, default=2_00)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dataloader_mp_context", type=str, default="spawn",
                   choices=["spawn", "forkserver", "fork"],
                   help="DataLoader worker start method. 'spawn' avoids the "
                        "fork-after-CUDA hang; use 'fork' only if spawn misbehaves")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="recompute each encoder block in backward instead of storing "
                        "its activations: ~1/3 more compute for a large cut in "
                        "activation memory. Intended for the long-context phase "
                        "(--max_seq_len 8192), where memory — not arithmetic — caps "
                        "the batch size; leave it off at 1024 where a full batch "
                        "already fits.")
    # logging / io
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=2_000)
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_log_model", action="store_true",
                   help="upload the exported discriminator backbone to W&B as an artifact")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume_weights_only", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="load only model weights from --resume (skip optimizer/"
                        "scheduler/step); use to warm-start a fresh run from a checkpoint")
    args = p.parse_args(argv)

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # CLI flags explicitly set on the command line win over the YAML file
        cli_argv = sys.argv[1:] if argv is None else argv
        provided_options = {
            token.split("=", 1)[0] for token in cli_argv if token.startswith("--")
        }
        # Compare YAML keys with argparse destinations, rather than spelling CLI
        # options by hand.  This is important for BooleanOptionalAction: both
        # --resume_weights_only and --no-resume_weights_only target the same
        # ``resume_weights_only`` setting and must override the YAML value.
        provided = {
            action.dest
            for action in p._actions
            if provided_options.intersection(action.option_strings)
        }
        # An explicit --warmup_steps must not be clobbered by a YAML warmup_frac
        # (main() lets warmup_frac override warmup_steps) — drop the YAML value
        # unless --warmup_frac was itself given on the command line.
        if "warmup_steps" in provided and "warmup_frac" not in provided:
            cfg.pop("warmup_frac", None)
        # PyYAML's float resolver rejects an unsigned exponent, so "2e-3" / "13e9"
        # arrive as STRINGS. Cast them to the type argparse declared for the flag.
        arg_types = {a.dest: a.type for a in p._actions}
        for k, v in cfg.items():
            if k not in provided and hasattr(args, k):
                # Cast the YAML value to the same type as the argparse default so
                # that e.g. "2e-3" (YAML string) becomes 0.002 (float) rather than
                # staying as a str and breaking f"{v:g}" format specs later.
                default_val = getattr(args, k)
                if isinstance(default_val, bool):
                    caster = bool
                elif isinstance(default_val, int):
                    caster = int
                elif isinstance(default_val, float):
                    caster = float
                elif default_val is None:
                    # optional flag (--train_tokens, --epochs, ...): there is no
                    # default to copy a type from, so use the declared one.
                    caster = arg_types.get(k)
                else:
                    caster = None
                # A YAML list / mapping (nargs flags such as --train_only, the
                # --arch_overrides mapping) is already structured; casting it
                # through the scalar type would stringify it.
                if caster is not None and v is not None and not isinstance(v, (list, dict)):
                    v = caster(v)
                setattr(args, k, v)
    return args


def _worker_kwargs(args):
    """DataLoader keyword args for multi-process loading.

    Forking workers *after* CUDA is initialised in the parent hangs on many
    setups (Colab / cloud GPUs). Starting them with 'spawn' (a clean process that
    does not inherit the CUDA context / tokenizer threadpool) avoids it. Only
    valid when ``num_workers > 0``.
    """
    if args.num_workers <= 0:
        return {}
    import multiprocessing as mp

    return {
        "multiprocessing_context": mp.get_context(args.dataloader_mp_context),
        "persistent_workers": True,
    }


def build_dataloader(args, tokenizer, env):
    if args.pretokenized:
        if not os.path.exists(args.pretokenized):
            # Fail fast: the config asks to train from a pre-tokenised shard but it
            # isn't there. Don't silently fall back to mid-run streaming (the thing
            # we pre-tokenise to avoid) — tell the user to build it first.
            cmd = (
                "python scripts/prepare_data.py "
                f"--dataset_name {args.dataset_name} "
                f"--dataset_config {args.dataset_config} "
                f"--text_column {args.text_column} "
                f"--tokenizer {args.tokenizer} "
                f"--max_seq_len {args.max_seq_len} "
                f"--output {args.pretokenized}"
            )
            raise FileNotFoundError(
                f"pre-tokenised shard not found: {args.pretokenized}\n"
                f"Build it BEFORE training (this downloads + packs the corpus up "
                f"front so the run never streams):\n    {cmd}\n"
                f"Or set pretokenized: null in the config to stream during training."
            )
        # Variable-length (ragged, one-doc-per-example) shard -> pad dynamically per
        # batch; fixed-block shard -> stack as-is.
        if is_varlen_shard(args.pretokenized):
            ds = VarLengthDataset(args.pretokenized)
            collate_fn = make_dynamic_pad_collate(ds.pad_token_id, args.pad_multiple_of)
        else:
            ds = PretokenizedDataset(args.pretokenized)
            collate_fn = collate
        sampler = DistributedSampler(
            ds, num_replicas=env.world_size, rank=env.rank, shuffle=True, seed=args.seed,
        ) if env.is_distributed else None
        loader = DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            shuffle=sampler is None, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True, drop_last=True,
            **_worker_kwargs(args),
        )
        return loader, sampler

    mix = getattr(args, "dataset_mix", None)
    if mix is not None and isinstance(mix, str):
        mix = json.loads(mix)

    ds = PackedStreamDataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_mix=mix if isinstance(mix, list) else None,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_column=args.text_column,
        rank=env.rank,
        world_size=env.world_size,
        seed=args.seed,
        buffer_docs=args.shuffle_buffer,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        **_worker_kwargs(args),
    )
    return loader, None


def build_short_dataloader(args, env):
    """Loader over the short-context replay shard (``--short_pretokenized``).

    Used during a long-context extension phase: ``--short_ratio`` of optimizer
    steps draw their micro-batches from this shard instead of the long one, so
    the model keeps seeing the short-sequence regime (sequence lengths, SSMax
    key counts, single-document attention) it learned in phase 1. Mirrors the
    pretokenized branch of :func:`build_dataloader`; the micro-batch defaults to
    ``batch_size * (max_seq_len // short_max_seq_len)`` so a short step carries
    roughly the same token count as a long one.
    """
    path = args.short_pretokenized
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"short replay shard not found: {path}\n"
            f"Build it with scripts/prepare_data.py (any pack mode), or set "
            f"short_pretokenized: null to train on the long shard only.")
    if is_varlen_shard(path):
        ds = VarLengthDataset(path)
        collate_fn = make_dynamic_pad_collate(ds.pad_token_id, args.pad_multiple_of)
    else:
        ds = PretokenizedDataset(path)
        collate_fn = collate
    if args.short_batch_size is None:
        args.short_batch_size = args.batch_size * max(1, args.max_seq_len // ds.max_seq_len)
    # seed+1: decorrelated from the long sampler's permutation stream
    sampler = DistributedSampler(
        ds, num_replicas=env.world_size, rank=env.rank, shuffle=True, seed=args.seed + 1,
    ) if env.is_distributed else None
    loader = DataLoader(
        ds, batch_size=args.short_batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
        **_worker_kwargs(args),
    )
    return loader, sampler


def build_eval_dataloader(args, tokenizer, env, *, pretokenized=None,
                          batch_size=None, label="eval"):
    """Build a held-out evaluation loader that never overlaps the training data.

    Preference order:
      1. ``--eval_pretokenized`` shard (instant, deterministic), if it exists.
      2. Streaming the ``--eval_split`` (default ``validation``) of the eval
         dataset, with shuffling disabled so every eval sees the same subset.

    ``pretokenized`` names a different shard (the ``--short_eval_pretokenized``
    slice); such a loader is shard-only, never the streaming fallback, and sizes
    its batch for token parity with the long batch unless ``batch_size`` is given.

    Returns ``None`` when no held-out source can be built (e.g. a streaming-only
    dataset without the requested split); the caller then skips eval rather than
    leaking training data into the metric.
    """
    shard = pretokenized or args.eval_pretokenized
    if shard:
        if not os.path.exists(shard):
            if env.is_main:
                print(f"[NexteraBERT] {label} shard not found ({shard}); "
                      f"skipping held-out {label}", flush=True)
            return None
        if is_varlen_shard(shard):
            ds = VarLengthDataset(shard)
            collate_fn = make_dynamic_pad_collate(ds.pad_token_id, args.pad_multiple_of)
        else:
            ds = PretokenizedDataset(shard)
            collate_fn = collate
        if batch_size is None:
            batch_size = args.batch_size
            if pretokenized:
                batch_size *= max(1, args.max_seq_len // ds.max_seq_len)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            collate_fn=collate_fn, pin_memory=True, drop_last=True,
            **_worker_kwargs(args),
        )
        return loader

    eval_name = args.eval_dataset_name or args.dataset_name
    if args.eval_dataset_name:
        # custom eval dataset: take its config as given (may be None)
        eval_config = args.eval_dataset_config
    else:
        # eval on the training dataset's held-out split: inherit its config
        eval_config = args.eval_dataset_config or args.dataset_config
    ds = PackedStreamDataset(
        dataset_name=eval_name,
        dataset_config=eval_config,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_column=args.text_column,
        split=args.eval_split,
        rank=env.rank,
        world_size=env.world_size,
        seed=args.seed,
        buffer_docs=0,  # deterministic: same held-out subset every eval
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=max(1, args.num_workers // 2),
        collate_fn=collate, pin_memory=True, drop_last=True,
        **_worker_kwargs(args),
    )
    return loader


def main():
    args = parse_args()
    env = setup_distributed()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed + env.rank)

    use_wandb = args.wandb_project is not None and env.is_main
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))

    tokenizer = build_tokenizer(args.tokenizer)
    # Model vocab is padded to a multiple of 64 for GPU tile alignment
    # (ModernBERT-base's 50,368 is already 64-aligned, so this is a no-op there);
    # any padded ids are never produced by the tokenizer, so the trainers cap all
    # token sampling at true_vocab_size.
    vocab_size = padded_vocab_size(tokenizer)
    specials = dict(
        mask_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        true_vocab_size=len(tokenizer),
    )
    # Bake the tokenizer's special-token ids into the model config too, so an exported
    # config.json matches the actual tokenizer (they differ per tokenizer, e.g.
    # ModernBERT's CLS/SEP/MASK/PAD are 50281/50282/50284/50283, not BERT's
    # 101/102/103/0).
    config_specials = dict(
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        # Baked into the exported config.json so a served checkpoint keeps the
        # dynamic-NTK extension setting (0 = disabled, identity for all lengths).
        rope_ntk_train_len=args.rope_ntk_train_len,
    )
    # Architecture switch, also baked into config.json so the checkpoint rebuilds
    # the same FFN it was trained with. The presets only fix the *sizes*, so this
    # rides on top of whatever --disc_size/--gen_size selects and applies to the
    # generator and discriminator alike. It is passed through only when actually
    # set, so an unset flag inherits NexteraBERTConfig's own default rather than
    # silently pinning it to a second copy of the default kept here.
    if args.mlp_class is not None:
        config_specials["mlp_class"] = args.mlp_class

    # Arbitrary per-run architecture overrides on top of the preset (ablations).
    # They apply to the MAIN model only — the ELECTRA/COCO-LM generator keeps its
    # own --gen_size preset, since the ablations vary the model being evaluated.
    arch_overrides = dict(args.arch_overrides or {})
    unknown = [k for k in arch_overrides
               if k not in NexteraBERTConfig().to_dict() and k != "block_types"]
    if unknown and env.is_main:
        print(f"[NexteraBERT] warning: --arch_overrides has key(s) the config does "
              f"not define: {unknown} (they will be stored verbatim)")

    is_bert_mode = args.training_mode == "bert"
    is_cocolm = args.training_mode == "cocolm"
    disc_config = NexteraBERTConfig.from_preset(
        args.disc_size, vocab_size=vocab_size, **config_specials, **arch_overrides)

    if is_bert_mode:
        model = NexteraBERTForBertTrainer(
            config=disc_config,
            mask_ratio=args.mask_ratio,
            router_z_loss_coef=args.router_z_loss_coef,
            load_balance_loss_coef=args.load_balance_loss_coef,
            **specials,
        )
        gen_config = None
    elif is_cocolm:
        gen_config = NexteraBERTConfig.from_preset(
            args.gen_size, vocab_size=vocab_size, **config_specials)
        # matryoshka_dims may arrive as a "512,256,..." string (CLI/YAML) or a
        # YAML list; accept both.
        _md = args.matryoshka_dims
        if isinstance(_md, str):
            matryoshka_dims = [int(x) for x in _md.replace(" ", "").split(",") if x]
        else:
            matryoshka_dims = [int(x) for x in _md]
        model = NexteraBERTForCocoLMTrainer(
            generator_config=gen_config,
            main_config=disc_config,
            mask_ratio=args.mask_ratio,
            copy_lambda=args.copy_lambda,
            scl_lambda=args.scl_lambda,
            scl_temperature=args.scl_temperature,
            scl_proj_dim=args.scl_proj_dim,
            scl_crop_ratio=args.scl_crop_ratio,
            matryoshka_dims=matryoshka_dims,
            gen_temperature=args.gen_temperature,
            router_z_loss_coef=args.router_z_loss_coef,
            load_balance_loss_coef=args.load_balance_loss_coef,
            tie_embeddings=args.tie_gen_disc_embeddings,
            **specials,
        )
    else:
        gen_config = NexteraBERTConfig.from_preset(
            args.gen_size, vocab_size=vocab_size, **config_specials)
        model = NexteraBERTForElectraTrainer(
            generator_config=gen_config,
            discriminator_config=disc_config,
            mask_ratio=args.mask_ratio,
            disc_lambda=args.disc_lambda,
            gen_temperature=args.gen_temperature,
            router_z_loss_coef=args.router_z_loss_coef,
            load_balance_loss_coef=args.load_balance_loss_coef,
            tie_embeddings=args.tie_gen_disc_embeddings,
            **specials,
        )

    model.to(env.device)
    for m in model.eval_metrics.values():
        m.to(env.device)

    if env.is_main:
        if is_bert_mode:
            print(f"[NexteraBERT] BERT mode: model={args.disc_size} "
                  f"({count_parameters(model.model):,} params)")
        elif is_cocolm:
            print(f"[NexteraBERT] COCO-LM mode: main={args.disc_size} "
                  f"({count_parameters(model.model):,} params), "
                  f"generator={args.gen_size} ({count_parameters(model.generator):,} params); "
                  f"SCL matryoshka dims={model.matryoshka_dims} "
                  f"(proj={args.scl_proj_dim}, τ={args.scl_temperature:g})")
        else:
            print(f"[NexteraBERT] ELECTRA mode: discriminator={args.disc_size} "
                  f"({count_parameters(model.discriminator):,} params), "
                  f"generator={args.gen_size} ({count_parameters(model.generator):,} params)")
        # Read back from the config actually built, not from args — an unset flag
        # inherits the config-class default and must still be reported correctly.
        print(f"[NexteraBERT] mlp={disc_config.mlp_class}")
        if arch_overrides:
            print(f"[NexteraBERT] arch overrides: "
                  f"{json.dumps(arch_overrides, ensure_ascii=False)}")
        print(f"[NexteraBERT] arch: n_layer={disc_config.n_layer} "
              f"n_embd={disc_config.n_embd} n_head={disc_config.n_head} "
              f"n_inter={disc_config.n_inter} ssmax={disc_config.ssmax_mode}")
        print(f"[NexteraBERT] blocks: {' '.join(disc_config.block_types)}")

    # Partial fine-tuning (--train_only): freeze everything else now, before the
    # model is compiled / wrapped in DDP and before the optimizer collects its
    # parameter groups. Frozen weights still get loaded by --resume and saved by
    # save_checkpoint; they just never receive an update.
    if isinstance(args.train_only, str):      # YAML `train_only: scalable_factor`
        args.train_only = [args.train_only]
    if args.train_only:
        frozen = freeze_all_but(model, args.train_only)
        if frozen["trainable_tensors"] == 0:
            raise ValueError(
                f"--train_only {args.train_only} matches no trainable parameter; "
                f"parameter names look like "
                f"{next(n for n, _ in model.named_parameters())!r}")
        if env.is_main:
            print(f"[NexteraBERT] train_only={args.train_only}: "
                  f"{frozen['trainable_tensors']} tensor(s) / "
                  f"{frozen['trainable_params']:,} params trainable, "
                  f"{frozen['frozen_tensors']} tensor(s) frozen")
            shown = frozen["names"][:6]
            more = len(frozen["names"]) - len(shown)
            print(f"[NexteraBERT]   trainable: {', '.join(shown)}"
                  f"{f', ... (+{more})' if more > 0 else ''}")

    if args.grad_checkpointing:
        # Every NexteraBERT encoder in the trainer (generator + discriminator /
        # main model), whichever objective is running.
        encoders = [m for m in model.modules() if isinstance(m, NexteraBERT)]
        for enc in encoders:
            enc.set_gradient_checkpointing(True)
        if env.is_main:
            print(f"[NexteraBERT] gradient checkpointing ON "
                  f"({len(encoders)} encoder(s))")

    if args.compile:
        # DynamicPadCollate pads each batch to its own max rounded to
        # --pad_multiple_of (8 by default), so a 1024-token shard presents up to 128
        # distinct sequence lengths. Dynamo specialises on the first shape, marks the
        # dim dynamic on the second, and re-specialises whenever a guard fails; past
        # the default limit of 8 it STOPS COMPILING and silently runs the rest of the
        # run in eager — throwing away the whole ~1.5x that --compile is here for, with
        # nothing in the log to say so. Raise the ceiling so the handful of genuinely
        # distinct graphs (dynamic vs static shape, warmup vs discriminator) all fit.
        # `recompile_limit` is the current name; `cache_size_limit` is its older alias.
        for _name in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, _name):
                setattr(torch._dynamo.config, _name, max(
                    64, getattr(torch._dynamo.config, _name)))
        model = torch.compile(model)

    ddp_model = model
    if env.is_distributed:
        ddp_model = DDP(
            model,
            device_ids=[env.local_rank] if env.device.type == "cuda" else None,
            # gen_warmup_frac resolves to a step count only after the loader sizes
            # max_steps (below), so consider it here too — during the warmup the
            # discriminator is not run and DDP must tolerate its unused params.
            find_unused_parameters=(not is_bert_mode
                                    and (args.gen_warmup_steps > 0
                                         or (args.gen_warmup_frac or 0) > 0)),
            # The gradients ARE the reducer's bucket storage rather than a copy into
            # it: one fewer full-size copy of every gradient per step, and the peak
            # drops by roughly one gradient replica.
            gradient_as_bucket_view=True,
            # NexteraBERT registers no buffers at all (checked: named_buffers() is
            # empty — the rotary table is a plain attribute), so the per-forward
            # buffer broadcast has nothing to send and is pure latency.
            broadcast_buffers=False,
        )

    _base = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
    core_model = getattr(_base, "_orig_mod", _base)

    # Build the loader first so --epochs / --train_tokens can derive max_steps from the
    # real shard contents (variable-length shards can't be sized by tokens up front).
    loader, sampler = build_dataloader(args, tokenizer, env)
    blocks_per_step = args.batch_size * args.grad_accum * env.world_size

    # Short-context replay loader (ProLong-style long/short mixing, phase 2 only).
    short_loader, short_sampler = None, None
    use_short_replay = bool(args.short_pretokenized) and args.short_ratio > 0
    if use_short_replay:
        short_loader, short_sampler = build_short_dataloader(args, env)
        if env.is_main:
            print(f"[NexteraBERT] short replay: {args.short_pretokenized} — "
                  f"{args.short_ratio:.0%} of steps use micro-batch "
                  f"{args.short_batch_size} x {short_loader.dataset.max_seq_len} "
                  f"(long: {args.batch_size} x {args.max_seq_len})")
    if args.epochs is not None:
        try:
            steps_per_epoch = max(1, len(loader) // args.grad_accum)
        except TypeError:
            steps_per_epoch = None                      # streaming loader has no length
        if steps_per_epoch is not None:
            args.max_steps = max(1, int(args.epochs * steps_per_epoch))
            if env.is_main:
                print(f"[NexteraBERT] --epochs {args.epochs:g}: {steps_per_epoch:,} "
                      f"opt-steps/epoch -> max_steps={args.max_steps:,}")
        elif env.is_main:
            print("[NexteraBERT] --epochs ignored: streaming loader has no length; "
                  "using --max_steps")

    if args.train_tokens is not None:
        # CONTENT tokens per example. A variable-length (one-doc-per-example) shard
        # stores documents shorter than max_seq_len — the collate pads each batch to
        # its longest member, but that padding is masked out and carries no data, so
        # the budget is sized by the shard's OWN mean segment length rather than by
        # max_seq_len (which would overcount by the padding ratio). Fixed-block
        # shards (--pack_mode unpad/concat) and the streaming packer emit full
        # max_seq_len blocks of densely-packed content, so there the two coincide.
        shard_tokens = None
        if args.pretokenized and is_varlen_shard(args.pretokenized):
            varlen_ds = loader.dataset
            shard_tokens = int(varlen_ds.offsets[-1])
            tok_per_block = shard_tokens / max(1, len(varlen_ds))
        elif args.pretokenized:
            # Fixed-block shard: blocks are max_seq_len wide but their CONTENT can be
            # narrower — --pack_mode unpad aligns each document segment to
            # --unpad_align tokens, and a --pack_mode pad shard pads every document on
            # its own. Measure it from a strided sample of blocks (attention_mask is 0
            # exactly on padding) instead of assuming every block is full.
            block_ds = loader.dataset
            n_blocks = len(block_ds)
            sampled = [int(block_ds[i]["attention_mask"].sum())
                       for i in range(0, n_blocks, max(1, n_blocks // 512))]
            tok_per_block = sum(sampled) / len(sampled)
            shard_tokens = int(tok_per_block * n_blocks)     # sample-based estimate
        else:
            tok_per_block = float(args.max_seq_len)     # streaming packer emits full blocks
        tokens_per_step = blocks_per_step * tok_per_block
        if use_short_replay:
            # Blend in the replay steps' (smaller) token count so the budget still
            # counts every content token seen: short_ratio of steps draw
            # short_batch_size examples of the short shard's own mean length.
            sds = short_loader.dataset
            if is_varlen_shard(args.short_pretokenized):
                short_tok_per_ex = int(sds.offsets[-1]) / max(1, len(sds))
            else:
                n = len(sds)
                sampled = [int(sds[i]["attention_mask"].sum())
                           for i in range(0, n, max(1, n // 512))]
                short_tok_per_ex = sum(sampled) / len(sampled)
            short_tok_per_step = (args.short_batch_size * args.grad_accum
                                  * env.world_size * short_tok_per_ex)
            tokens_per_step = ((1.0 - args.short_ratio) * tokens_per_step
                               + args.short_ratio * short_tok_per_step)
            if env.is_main:
                print(f"[NexteraBERT] short replay budget: {short_tok_per_ex:.0f} "
                      f"tok/example -> {short_tok_per_step / 1e6:.2f} M tok/short-step "
                      f"({args.short_ratio:.0%} of steps)")
        args.max_steps = max(1, round(args.train_tokens / tokens_per_step))
        if env.is_main:
            print(f"[NexteraBERT] --train_tokens {args.train_tokens / 1e9:g}B: "
                  f"{tok_per_block:.0f} tok/example x {blocks_per_step} examples/step "
                  f"= {tokens_per_step / 1e6:.2f} M tok/step (blended) -> "
                  f"max_steps={args.max_steps:,}"
                  if use_short_replay else
                  f"[NexteraBERT] --train_tokens {args.train_tokens / 1e9:g}B: "
                  f"{tok_per_block:.0f} tok/example x {blocks_per_step} examples/step "
                  f"= {tokens_per_step / 1e6:.2f} M tok/step -> "
                  f"max_steps={args.max_steps:,}")
            if shard_tokens is not None:
                print(f"[NexteraBERT] shard {args.pretokenized}: {len(loader.dataset):,} "
                      f"examples / {shard_tokens / 1e9:.2f} B tokens -> this budget is "
                      f"{args.train_tokens / shard_tokens:.2f} epochs over it")

    if args.gen_warmup_frac is not None:
        args.gen_warmup_steps = max(0, round(args.gen_warmup_frac * args.max_steps))
        if env.is_main:
            print(f"[NexteraBERT] gen_warmup_frac {args.gen_warmup_frac:g}: "
                  f"gen_warmup_steps={args.gen_warmup_steps:,} "
                  f"({args.gen_warmup_frac:.0%} of {args.max_steps:,} steps)")

    if args.warmup_frac is not None:
        # warmup reaches peak LR at exactly warmup_frac of the (resolved) total steps —
        # the linear ramp finishes at args.warmup_steps, so this is that fraction exactly.
        # OptiBERT (Appendix A) constrains its 10%-of-steps warmup to [500, 50,000]
        # steps to keep it consistent across training regimes; apply the same clamp
        # (capped at max_steps so tiny debug runs still finish their warmup).
        args.warmup_steps = max(1, round(args.warmup_frac * args.max_steps))
        args.warmup_steps = min(max(args.warmup_steps, 500), 50_000, args.max_steps)
        if env.is_main:
            print(f"[NexteraBERT] warmup_frac {args.warmup_frac:g}: warmup_steps="
                  f"{args.warmup_steps:,} (peak LR at "
                  f"{args.warmup_steps / args.max_steps:.0%} of "
                  f"{args.max_steps:,} steps; clamped to [500, 50k])")

    decoupled_wd = args.wd_mode == "decoupled"
    optimizer = build_optimizer(
        ddp_model, lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay, eps=args.eps, decoupled=decoupled_wd,
    )
    # Remember the intended per-group decay: --resume re-asserts the configured
    # recipe over the checkpoint's, and optimizer.defaults holds only the decayed
    # group's value — replaying it into every group would start decaying the
    # biases/norms that build_optimizer deliberately exempts.
    wd_by_group = [g["weight_decay"] for g in optimizer.param_groups]
    if args.lr_schedule == "wsd":
        decay_steps = max(0, round(args.decay_frac * args.max_steps))
        scheduler = WarmupStableDecay(optimizer, args.warmup_steps, args.max_steps,
                                      decay_steps=decay_steps)
        sched_label = (f"wsd (warmup {args.warmup_steps:,} -> constant -> 1-sqrt decay "
                       f"over the last {scheduler.decay_steps:,} steps)"
                       if scheduler.decay_steps else
                       f"wsd (warmup {args.warmup_steps:,} -> constant, no decay tail)")
    else:
        scheduler = WarmupCosineDecay(optimizer, args.warmup_steps, args.max_steps)
        sched_label = f"cosine (warmup {args.warmup_steps:,} -> 10% of peak)"
    if env.is_main:
        # 'standard' builds ONE group over every parameter, 'decoupled' adds a
        # second, decay-free group for biases/norms — so only describe the
        # exemption when that group actually exists.
        if decoupled_wd:
            wd_note = (f" [decoupled -> torch wd {wd_by_group[0]:g}], "
                       f"{len(optimizer.param_groups[1]['params'])} bias/norm "
                       f"params exempt")
        else:
            wd_note = " [standard], every parameter decayed"
        print(f"[NexteraBERT] optimizer=adamw (lr={args.lr:g}, "
              f"betas=({args.beta1:g}, {args.beta2:g}), eps={args.eps:g}, "
              f"wd={args.weight_decay:g}{wd_note})")
        print(f"[NexteraBERT] lr_schedule={sched_label}")
    if env.is_main and args.warmup_steps < 100:
        print(f"[NexteraBERT] WARNING: warmup_steps={args.warmup_steps} is very low — "
              f"the optimizer hits the peak LR ({args.lr:g}) almost immediately, which "
              f"tends to diverge the generator. Use >= a few hundred warmup steps.")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_amp = dtype != torch.float32 and env.device.type == "cuda"
    autocast = (
        torch.autocast(device_type=env.device.type, dtype=dtype)
        if use_amp else contextlib.nullcontext()
    )
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    if env.device.type == "cuda":
        loader = CudaPrefetchLoader(loader, env.device)
        if short_loader is not None:
            short_loader = CudaPrefetchLoader(short_loader, env.device)

    # Held-out eval loader (separate split — never overlaps the training data).
    eval_loader = build_eval_dataloader(args, tokenizer, env)
    if eval_loader is not None and env.is_main:
        src = (args.eval_pretokenized
               or f"{args.eval_dataset_name or args.dataset_name}[{args.eval_split}]")
        print(f"[NexteraBERT] held-out eval source: {src}")
    # Optional second held-out loader over a SHORT shard, scored alongside the
    # long one so a long-context phase reports both regimes (the SSSMax scale is
    # a function of length, so the two can move in opposite directions).
    short_eval_loader = None
    if args.short_eval_pretokenized:
        short_eval_loader = build_eval_dataloader(
            args, tokenizer, env, pretokenized=args.short_eval_pretokenized,
            batch_size=args.short_batch_size, label="short eval")
        if short_eval_loader is not None and env.is_main:
            print(f"[NexteraBERT] held-out short eval source: "
                  f"{args.short_eval_pretokenized} (micro-batch "
                  f"{short_eval_loader.batch_size} x "
                  f"{short_eval_loader.dataset.max_seq_len})")

    start_step = 0
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"resume checkpoint not found: {args.resume}\n"
            "Refusing to start from random weights. Check --resume, or omit it "
            "explicitly for a fresh run."
        )
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["model"]
        # Strip _orig_mod. prefix saved by older checkpoints (torch.compile artifact)
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        try:
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # strict=False tolerates missing/unexpected keys but not a shape
            # mismatch. The one shape that has changed across revisions is the
            # SSSMax offset (a (1,) tensor in older formulations, a scalar now), so
            # point at the converter instead of leaving a bare size-mismatch trace.
            if "scalable_factor" in str(e):
                raise RuntimeError(
                    f"{args.resume} was written under an older SSSMax formulation. "
                    f"Convert it first:\n"
                    f"  python scripts/migrate_ssmax_checkpoint.py --ckpt {args.resume} "
                    f"--output <new.pt>\nthen resume the converted file with "
                    f"--resume_weights_only.") from e
            raise
        if env.is_main and (missing or unexpected):
            if missing:
                print(f"[resume] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"[resume] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not getattr(args, "resume_weights_only", False):
            opt_state = ckpt["optimizer"]
            if "param_groups" not in opt_state:
                raise ValueError(
                    f"{args.resume} was saved by a pre-AdamW-migration optimizer "
                    "(incompatible state format). Pass --resume_weights_only to "
                    "load the model weights with a fresh optimizer/LR schedule.")
            if len(opt_state["param_groups"]) != len(optimizer.param_groups):
                raise ValueError(
                    f"{args.resume} has {len(opt_state['param_groups'])} optimizer "
                    f"param group(s), this run builds {len(optimizer.param_groups)} "
                    "(decayed + bias/norm-exempt, the ModernBERT recipe). Pass "
                    "--resume_weights_only to load the model weights with a fresh "
                    "optimizer/LR schedule.")
            ckpt_sizes = [len(g["params"]) for g in opt_state["param_groups"]]
            live_sizes = [len(g["params"]) for g in optimizer.param_groups]
            if ckpt_sizes != live_sizes:
                raise ValueError(
                    f"{args.resume} optimizer state covers {ckpt_sizes} params per "
                    f"group, this run builds {live_sizes} -- the set of trainable "
                    "parameters changed since the checkpoint was written (e.g. the "
                    "2026-08 ModernBERT-style prediction/RTD head transforms). Pass "
                    "--resume_weights_only to load the matching model weights with a "
                    "fresh optimizer/LR schedule, or restart pretraining from scratch "
                    "for a clean run.")
            optimizer.load_state_dict(opt_state)
            # load_state_dict adopts the checkpoint's group hyperparameters;
            # re-assert the configured recipe so a checkpoint written under a
            # different one (other betas/eps, the standard-AdamW wd convention,
            # no fused flag) cannot silently override it. lr is left alone — the
            # scheduler rewrites it every step. ``defaults`` holds
            # build_optimizer's construction values, but weight_decay is
            # per-group (0.0 for biases/norms), so replay the captured list.
            for group, wd in zip(optimizer.param_groups, wd_by_group):
                for k in ("betas", "eps", "fused"):
                    group[k] = optimizer.defaults[k]
                group["weight_decay"] = wd
            scheduler._step = ckpt.get("scheduler_step", 0)
            # scheduler_step counts optimizer steps COMPLETED (incremented after
            # each step); "step" is the last executed loop index. Resume from the
            # former — starting at "step" would re-run the checkpointed iteration.
            start_step = ckpt.get("scheduler_step", ckpt.get("step", 0))
        if env.is_main:
            if getattr(args, "resume_weights_only", False):
                print(f"loaded model weights from {args.resume} (fresh optimizer/schedule)")
            else:
                print(f"resumed from {args.resume} at step {start_step}")

    os.makedirs(args.output_dir, exist_ok=True)
    ddp_model.train()
    # One optimiser step consumes grad_accum micro-batches, so a full pass over the
    # shard is len(loader) // grad_accum STEPS -- len(loader) is counted in
    # micro-batches. Sizing the epoch counter by len(loader) fires set_epoch only once
    # every grad_accum passes, and DistributedSampler then hands out the SAME
    # permutation for every pass in between, so a >1-epoch token budget replays the
    # shard in identical order. Epoch advance is therefore driven by the iterator
    # actually running dry (below), which cannot desync from grad_accum; this step
    # count is only used to place a resumed run on the right permutation.
    # (A streaming loader has no sampler and no length; it reshuffles itself.)
    steps_per_epoch = (max(1, len(loader) // max(1, args.grad_accum))
                       if sampler is not None else None)
    epoch = start_step // steps_per_epoch if steps_per_epoch else 0

    def new_data_iter():
        """Begin the next pass over the data.

        set_epoch must precede iter(): DistributedSampler bakes its permutation in
        at __iter__ time, so setting the epoch afterwards has no effect on the
        iterator just created.
        """
        nonlocal epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
            if env.is_main and epoch > 0:
                print(f"[NexteraBERT] starting pass {epoch + 1} over "
                      f"{args.pretokenized} (reshuffled)", flush=True)
        epoch += 1
        return iter(loader)

    data_iter = new_data_iter()

    # Short-replay iterator: its own epoch counter and reshuffle cycle, advanced
    # only on short steps. The shard is small next to the long one, so it simply
    # reshuffles and goes around again whenever it runs dry.
    short_iter = None
    short_epoch = 0

    def next_short_batch():
        nonlocal short_iter, short_epoch
        for _ in range(2):
            if short_iter is None:
                if short_sampler is not None:
                    short_sampler.set_epoch(short_epoch)
                short_epoch += 1
                short_iter = iter(short_loader)
            try:
                return next(short_iter)
            except StopIteration:
                short_iter = None
        raise RuntimeError(f"short replay shard {args.short_pretokenized} yields no "
                           f"batches (fewer than short_batch_size examples?)")

    t0 = time.time()
    # Logging accumulators live ON THE DEVICE and are read back once per
    # ``log_every`` steps, not once per micro-batch. Calling ``.item()`` on each
    # component right after ``backward()`` (what this used to do) blocks the host
    # until every backward kernel retires — 5-7 full device syncs per micro-batch,
    # which serialises CPU and GPU and defeats the CudaPrefetchLoader entirely,
    # since the host can never run ahead to enqueue the next batch's H2D copy.
    # Order matches RUNNING_KEYS.
    RUNNING_KEYS = ("loss", "gen", "disc", "corr", "scl", "zloss", "bal")
    running = torch.zeros(len(RUNNING_KEYS), device=env.device, dtype=torch.float32)
    running_tokens = torch.zeros((), device=env.device, dtype=torch.float32)
    _zero = torch.zeros((), device=env.device, dtype=torch.float32)
    # `loss` is already divided by grad_accum before it is stacked; the raw
    # components are not, so they carry the 1/grad_accum here.
    _running_scale = torch.tensor(
        [1.0] + [1.0 / args.grad_accum] * (len(RUNNING_KEYS) - 1),
        device=env.device, dtype=torch.float32)

    if env.is_main and not is_bert_mode and args.gen_warmup_steps > 0:
        print(f"[NexteraBERT] generator-only warmup for {args.gen_warmup_steps} steps "
              f"(mask_ratio={args.gen_warmup_mask_ratio}); discriminator starts after that "
              f"(mask_ratio={args.mask_ratio})")
    # Head-only warmup: every trainable parameter OUTSIDE the encoders (the
    # prediction-head transform, decoder bias, RTD/copy heads) keeps its gradient;
    # encoder gradients are dropped before the optimizer step, so AdamW skips
    # those params entirely (no moment update, no weight decay). Done via
    # grad-dropping rather than requires_grad toggling because DDP forbids
    # changing requires_grad after wrapping, and this way the DDP bucketing /
    # compiled graphs are identical in and out of the warmup.
    head_warmup_encoder_params = []
    if args.head_warmup_steps > 0:
        head_warmup_encoder_params = [
            p for name, p in core_model.named_parameters()
            if p.requires_grad and "encoder." in name]
        if env.is_main and start_step < args.head_warmup_steps:
            n_head = sum(p.requires_grad and "encoder." not in name
                         for name, p in core_model.named_parameters())
            print(f"[NexteraBERT] head-only warmup until step {args.head_warmup_steps}: "
                  f"encoder gradients dropped ({len(head_warmup_encoder_params)} params "
                  f"held fixed, {n_head} head params training)")
    if env.is_main and is_bert_mode and args.mask_ratio_end is not None:
        _dir = "decreasing" if args.mask_ratio_end < args.mask_ratio_start else "increasing"
        print(f"[NexteraBERT] MLM mask-ratio schedule: {args.mask_ratio_start:g} -> "
              f"{args.mask_ratio_end:g} over {args.max_steps:,} steps "
              f"({_dir}, power={args.mask_ratio_power:g})")

    if env.is_main:
        if args.pretokenized:
            print(f"[NexteraBERT] fetching the first batch (pretokenized "
                  f"{args.pretokenized})...", flush=True)
        else:
            print(f"[NexteraBERT] fetching the first batch (streaming "
                  f"{args.dataset_name}); filling a {args.shuffle_buffer}-doc shuffle "
                  f"buffer first, which can take a while. For a faster start use "
                  f"--shuffle_buffer 0 or pre-tokenise with scripts/prepare_data.py.",
                  flush=True)

    for step in range(start_step, args.max_steps):
        if is_bert_mode:
            in_gen_warmup = False
            run_disc = True
            # Optional MLM mask-ratio schedule: ramp start -> end over the run with a
            # convex (power>1) curve — gentle early, steeper late. Read live by the
            # BERT trainer's forward (self.mask_ratio).
            if args.mask_ratio_end is not None:
                progress = min(1.0, step / max(1, args.max_steps))
                core_model.mask_ratio = (
                    args.mask_ratio_start
                    + (args.mask_ratio_end - args.mask_ratio_start)
                    * progress ** args.mask_ratio_power)
        else:
            in_gen_warmup = step < args.gen_warmup_steps
            core_model.mask_ratio = (args.gen_warmup_mask_ratio if in_gen_warmup
                                     else args.mask_ratio)
            run_disc = not in_gen_warmup
            if env.is_main and args.gen_warmup_steps > 0 and step == args.gen_warmup_steps:
                print(f"[NexteraBERT] step {step}: generator warmup done -> enabling "
                      f"discriminator (mask_ratio {args.gen_warmup_mask_ratio} -> {args.mask_ratio})")

        in_head_warmup = args.head_warmup_steps > 0 and step < args.head_warmup_steps
        if env.is_main and args.head_warmup_steps > 0 and step == args.head_warmup_steps:
            print(f"[NexteraBERT] step {step}: head-only warmup done -> encoder "
                  "updates enabled")

        optimizer.zero_grad(set_to_none=True)
        # Short-replay source choice for THIS optimizer step (all its micro-batches
        # come from the same loader so the gradient mixes at the step level, like
        # ProLong). Keyed off the step index alone — deterministic, identical on
        # every DDP rank, and stable across resumes without a shared RNG stream.
        use_short_step = (short_loader is not None
                          and random.Random((args.seed << 20) ^ step).random()
                          < args.short_ratio)

        for micro in range(args.grad_accum):
            if use_short_step:
                batch = next_short_batch()
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # The shard ran dry: this is the real epoch boundary, whatever
                    # grad_accum is. Advance the sampler epoch so the next pass gets a
                    # fresh permutation instead of replaying this one.
                    data_iter = new_data_iter()
                    batch = next(data_iter)
                except Exception as e:  # noqa: BLE001
                    # The streaming dataset already retries transient Hub/network
                    # errors internally; if one still escapes (e.g. a worker died),
                    # rebuild the iterator and retry a few times before giving up so
                    # a blip doesn't kill the run.
                    rebuilt = False
                    for attempt in range(1, 6):
                        if env.is_main:
                            print(f"[NexteraBERT] step {step}: dataloader error "
                                  f"({type(e).__name__}: {e}); rebuilding iterator "
                                  f"({attempt}/5)", flush=True)
                        try:
                            # new_data_iter (not a bare iter) so the retry draws a fresh
                            # permutation: restarting the current epoch's would re-feed
                            # the samples already seen before the error.
                            data_iter = new_data_iter()
                            batch = next(data_iter)
                            rebuilt = True
                            break
                        except Exception as e2:  # noqa: BLE001
                            e = e2
                            time.sleep(min(2.0 * attempt, 10.0))
                    if not rebuilt:
                        raise
            input_ids = batch["input_ids"].to(env.device, non_blocking=True)
            attn = batch["attention_mask"].to(env.device, non_blocking=True)
            if use_short_replay:
                # Real content tokens this rank consumed (padding excluded) — used
                # for the tok/s log, where the fixed max_seq_len formula would
                # overcount the short steps ~8x. Kept on-device; read at log time.
                running_tokens += attn.sum()

            sync_ctx = contextlib.nullcontext()
            if env.is_distributed and micro < args.grad_accum - 1:
                sync_ctx = ddp_model.no_sync()
            with sync_ctx, autocast:
                # Call the DDP wrapper, not the bare module: DDP installs the
                # reducer's expectation of the backward hooks in ITS forward
                # (prepare_for_backward). Going straight to `model` leaves the
                # grad-accumulator hooks inert, so NO all-reduce happens and each
                # rank silently trains its own divergent copy — and it also makes
                # the no_sync() above a no-op. Outside DDP, ddp_model IS model.
                outputs = ddp_model(input_ids, attention_mask=attn,
                                    run_discriminator=run_disc)
                loss, comps = model.loss(outputs, return_components=True)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            # COCO-LM-only components (absent in electra/bert -> treated as 0)
            corr_c = comps.get("correction_loss")
            scl_c = comps.get("scl_loss")
            # One stack + one add on the device instead of 5-7 blocking .item()
            # calls. Raw component losses (before weighting), averaged over
            # micro-steps; order must match RUNNING_KEYS. Everything is cast to
            # fp32 first: during the generator warmup ``discriminator_loss`` is a
            # ``new_zeros`` off the bf16 generator logits, and torch.stack rejects
            # a mixed-dtype tuple.
            running += _running_scale * torch.stack([
                p.float() for p in (
                    loss.detach(),
                    comps["generator_loss"],
                    comps["discriminator_loss"],
                    corr_c if corr_c is not None else _zero,
                    scl_c if scl_c is not None else _zero,
                    comps["router_z_loss"],
                    comps["load_balance_loss"],
                )
            ])

        if in_head_warmup:
            # Encoder grads were computed (and DDP-reduced) normally; dropping
            # them here means clip_grad_norm_ measures only the head grads and
            # AdamW skips the encoder entirely for this step.
            for p in head_warmup_encoder_params:
                p.grad = None

        # the hybrid optimizer wraps several real optimizers; the grad-scaler must
        # see each of them (a plain AdamW is its own single-element list)
        real_optims = getattr(optimizer, "optimizers", [optimizer])
        if args.grad_clip > 0:
            for opt in real_optims:
                scaler.unscale_(opt)
            if is_bert_mode:
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), args.grad_clip)
            elif is_cocolm:
                torch.nn.utils.clip_grad_norm_(core_model.generator.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(core_model.model.parameters(), args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(core_model.generator.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(core_model.discriminator.parameters(), args.grad_clip)
        for opt in real_optims:
            scaler.step(opt)
        scaler.update()
        scheduler.step()
        if step % args.log_every == 0 and env.is_main:
            dt = time.time() - t0
            # The ONE device->host read per logging interval.
            stats = dict(zip(RUNNING_KEYS, running.tolist()))
            tokens_seen = running_tokens.item()
            if use_short_replay:
                # Mixed long/short steps: report the REAL content tokens/s counted
                # per micro-batch (this rank x world_size; ranks draw same-source
                # batches, so the extrapolation is close).
                tok_per_s = tokens_seen * env.world_size / max(dt, 1e-6)
            else:
                tok_per_s = args.log_every * args.batch_size * args.grad_accum * \
                    args.max_seq_len * env.world_size / max(dt, 1e-6)
            denom = args.log_every if step > start_step else 1
            avg = stats["loss"] / denom
            avg_gen = stats["gen"] / denom
            avg_disc = stats["disc"] / denom
            avg_corr = stats["corr"] / denom
            avg_scl = stats["scl"] / denom
            avg_zloss = stats["zloss"] / denom
            avg_bal = stats["bal"] / denom
            lr = scheduler.get_last_lr()[0]
            # COCO-LM logs 'disc' as the copy/RTD loss and adds corr/scl columns
            extra = f"corr {avg_corr:.4f} | scl {avg_scl:.4f} | " if is_cocolm else ""
            # During the generator-only warmup the main model is idle, so
            # disc/corr/scl are 0 by design — flag it so that isn't mistaken for a bug.
            warm = (f" | GEN-WARMUP {step}/{args.gen_warmup_steps} "
                    f"(main model{'/SCL' if is_cocolm else ''} idle)") if in_gen_warmup else ""
            if in_head_warmup:
                warm += (f" | HEAD-WARMUP {step}/{args.head_warmup_steps} "
                         "(encoder frozen)")
            sched_mask = is_bert_mode and args.mask_ratio_end is not None
            mask_str = f" | mask {core_model.mask_ratio:.3f}" if sched_mask else ""
            print(f"step {step:>7} | loss {avg:.4f} | gen {avg_gen:.4f} | "
                  f"disc {avg_disc:.4f} | {extra}zloss {avg_zloss:.4f} | bal {avg_bal:.4f} | "
                  f"lr {lr:.2e}{mask_str} | {tok_per_s/1e3:.1f}k tok/s{warm}")
            if use_wandb:
                import wandb
                log_dict = {
                    "train/loss": avg,
                    "train/generator_loss": avg_gen,
                    "train/discriminator_loss": avg_disc,
                    "train/router_z_loss": avg_zloss,
                    "train/load_balance_loss": avg_bal,
                    "lr": lr,
                    "tok_per_s": tok_per_s,
                }
                if is_cocolm:
                    log_dict["train/correction_loss"] = avg_corr
                    log_dict["train/scl_loss"] = avg_scl
                if sched_mask:
                    log_dict["train/mask_ratio"] = core_model.mask_ratio
                wandb.log(log_dict, step=step)
            running.zero_()
            running_tokens.zero_()
            t0 = time.time()

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            if eval_loader is not None:
                run_eval(model, eval_loader, env, args, step, use_wandb, autocast)
            if short_eval_loader is not None:
                run_eval(model, short_eval_loader, env, args, step, use_wandb,
                         autocast, prefix="eval_short")
            ddp_model.train()

        if args.save_every and step > 0 and step % args.save_every == 0 and env.is_main:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step,
                            disc_config.to_dict())
            print(f"saved checkpoint -> {ckpt_path}")

    if env.is_main:
        final = os.path.join(args.output_dir, "final.pt")
        save_checkpoint(final, model, optimizer, scheduler, args.max_steps,
                        disc_config.to_dict())
        if is_bert_mode:
            backbone_dir = export_bert_backbone(model, disc_config, tokenizer,
                                                args.output_dir)
            print(f"training complete. final checkpoint -> {final}")
            if use_wandb and args.wandb_log_model:
                upload_backbone_to_wandb(backbone_dir, args, role="bert", size=args.disc_size)
        elif is_cocolm:
            backbone_dir = export_cocolm_backbone(model, disc_config, tokenizer,
                                                  args.output_dir)
            print(f"training complete. final checkpoint -> {final}")
            if use_wandb and args.wandb_log_model:
                upload_backbone_to_wandb(backbone_dir, args, role="cocolm", size=args.disc_size)
        else:
            gen_dir, disc_dir = export_backbones(model, gen_config, disc_config,
                                                 tokenizer, args.output_dir)
            print(f"training complete. final checkpoint -> {final}")
            if use_wandb and args.wandb_log_model:
                upload_backbone_to_wandb(disc_dir, args, role="discriminator", size=args.disc_size)
                upload_backbone_to_wandb(gen_dir, args, role="generator", size=args.gen_size)

    cleanup_distributed(env)


def upload_backbone_to_wandb(backbone_dir, args, role, size):
    """Log an exported backbone to W&B as a model artifact."""
    import wandb

    name = f"{args.run_name or 'nexterabert'}-{role}"
    artifact = wandb.Artifact(name, type="model",
                              metadata={"role": role, "size": size,
                                        "max_steps": args.max_steps})
    artifact.add_dir(str(backbone_dir))
    wandb.log_artifact(artifact)
    print(f"uploaded {role} backbone to W&B as artifact '{name}'")


@torch.no_grad()
def run_eval(model, loader, env, args, step, use_wandb, autocast_ctx=None,
             prefix="eval"):
    """Held-out eval on ``model`` — which must be the (possibly torch.compile'd)
    trainer, called through ``__call__``, under the same autocast as training.
    ``prefix`` names the metric group (``eval`` for the main shard, ``eval_short``
    for ``--short_eval_pretokenized``).

    Both details are load-bearing at 8192. ``eval_forward`` is a plain method, so
    on an ``OptimizedModule`` it resolves through ``__getattr__`` to the ORIGINAL
    module and bypasses the compiled graph entirely; eager FlexAttention then
    falls back to the dense math kernel that materialises the full (B, H, T, T)
    score matrix — 32 GiB in fp32 at T=8192 — and OOMs the eval. Calling the
    wrapper's ``__call__`` keeps eval on the fused kernels (one extra compiled
    graph for eval/no-grad mode, covered by the raised recompile limit), and
    running it under ``autocast_ctx`` keeps activations bf16 as in training
    instead of silently doubling them to fp32.
    """
    if autocast_ctx is None:
        autocast_ctx = contextlib.nullcontext()
    model.eval()
    metrics = model.get_metrics(is_train=False)
    for m in metrics.values():
        m.reset()
    it = iter(loader)
    seen = 0
    for _ in range(args.eval_steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        except Exception as e:  # noqa: BLE001
            # eval is best-effort: a transient streaming/Hub error here must not
            # take down the whole training run. Skip the rest of this eval.
            if env.is_main:
                print(f"[eval] step {step}: skipping eval after a data error "
                      f"({type(e).__name__}: {e})", flush=True)
            return
        input_ids = batch["input_ids"].to(env.device)
        attn = batch["attention_mask"].to(env.device)
        with autocast_ctx:
            outputs = model(input_ids, attention_mask=attn)
        for m in metrics.values():
            model.update_metric({"input_ids": input_ids}, outputs, m)
        seen += 1
    if seen == 0:
        return
    results = {name: m.compute().item() for name, m in metrics.items()}
    if env.is_main:
        msg = " | ".join(f"{k} {v:.4f}" for k, v in results.items())
        print(f"[{prefix}] step {step} | {msg}")
        if use_wandb:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in results.items()}, step=step)


def export_bert_backbone(model, config, tokenizer, output_dir):
    """Save the BERT encoder as a standalone binary backbone."""
    from nexterabert.export import save_mlm_head, save_pretrained

    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)

    backbone_dir = save_pretrained(raw.model, os.path.join(output_dir, "backbone"),
                                   config=config, tokenizer=tokenizer)
    # The MLM output head lives outside the encoder, so save_pretrained does not
    # write it; without it the backbone cannot be scored as a masked LM.
    head_path = save_mlm_head(raw.model, backbone_dir)
    print(f"exported backbone -> {backbone_dir}"
          + (f" (+ MLM head -> {head_path})" if head_path else ""))
    return backbone_dir


def export_cocolm_backbone(model, config, tokenizer, output_dir):
    """Save the COCO-LM main-model encoder as a standalone binary backbone.

    The main model (``trainer.model``, a ``NexteraBERTForCocoLM``) is the reusable
    part; ``save_pretrained`` resolves its ``.encoder`` and writes the same backbone
    directory layout as the ELECTRA discriminator / BERT model.
    """
    from nexterabert.export import save_mlm_head, save_pretrained

    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)

    backbone_dir = save_pretrained(raw.model, os.path.join(output_dir, "backbone"),
                                   config=config, tokenizer=tokenizer)
    # The MLM output head lives outside the encoder, so save_pretrained does not
    # write it; without it the backbone cannot be scored as a masked LM.
    head_path = save_mlm_head(raw.model, backbone_dir)
    print(f"exported backbone -> {backbone_dir}"
          + (f" (+ MLM head -> {head_path})" if head_path else ""))
    return backbone_dir


def export_backbones(model, gen_config, disc_config, tokenizer, output_dir):
    """Save the generator and discriminator encoders as standalone binary backbones.

    Each is written to its own directory (config.json + pytorch_model.bin +
    model.safetensors + tokenizer) under ``output_dir``.
    """
    from nexterabert.export import save_mlm_head, save_pretrained

    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)

    disc_dir = save_pretrained(raw.discriminator, os.path.join(output_dir, "discriminator"),
                               config=disc_config, tokenizer=tokenizer)
    gen_dir = save_pretrained(raw.generator, os.path.join(output_dir, "generator"),
                              config=gen_config, tokenizer=tokenizer)
    # Only the generator has an MLM head (the discriminator's objective is RTD).
    save_mlm_head(raw.generator, gen_dir)
    print(f"exported backbones -> {disc_dir} (discriminator), {gen_dir} (generator)")
    return gen_dir, disc_dir


if __name__ == "__main__":
    main()
