#!/usr/bin/env python
"""Contrastive fine-tuning: the stage MTEB numbers are measured after.

MTEB scores one vector per text, and an MLM backbone was never trained to place
related texts near each other -- so published encoder numbers on it come from a
model that first learned to embed. This script runs that stage under the protocol
of the MTEB comparison:

  --protocol mteb-nli  (default)  OptiBERT, Dervishi et al. EMNLP 2025, App. D.2
      Attentive pooling head + supervised-SimCSE InfoNCE on MNLI+SNLI triplets,
      3 epochs (Gao et al., 2021). Their Table 5 MTEB(eng, v2) scores follow this.
          -> feed the result to scripts/evaluate_mteb.py

    python scripts/finetune_contrastive.py --model checkpoints/phase2/backbone \
        --output_dir checkpoints/phase2/simcse
    python scripts/evaluate_mteb.py --model checkpoints/phase2/simcse

    # A Hugging Face baseline through the SAME stage, for a like-for-like MTEB row
    python scripts/finetune_contrastive.py --hf_model answerdotai/ModernBERT-base         --output_dir checkpoints/baselines/ModernBERT-base/simcse
    python scripts/finetune_contrastive.py --hf_model chandar-lab/NeoBERT         --output_dir checkpoints/baselines/NeoBERT/simcse
    python scripts/evaluate_mteb.py --model checkpoints/baselines/NeoBERT/simcse

--hf_model opens any AutoModel (ModernBERT, NeoBERT, BERT, ...) with the same
attentive head / mean pooling, optimiser and data (nexterabert.hf_baselines), and
uses the baseline's own tokenizer unless --tokenizer says otherwise. The result
directory is an ordinary save_pretrained directory plus the repo's markers, and
both eval scripts open it automatically through --model.

Everything the protocol pins is in PROTOCOLS below and overridable from the CLI.
The optimiser side is `evaluate_glue.py`'s recipe (AdamW betas (0.9, 0.95), eps
1e-6, linear decay to zero after warmup, grad clip 1.0, weight decay on matrices
only, torch-scale), reusing its `build_optimizer` so the no-decay grouping and LR
ladder stay identical to GLUE fine-tuning. As in this repo's GLUE recipe, layerwise
LR decay is on by default (`--llrd 0.9`): the upper layers retain the protocol LR
while lower pretrained layers move progressively less. This is an extra
regulariser rather than part of the published protocol; `--no_llrd` restores its
flat LR exactly (the other side of the LLRD comparison). A Hugging Face baseline
gets the same default, with its layer names mapped onto the same ladder.

Batch size is part of the objective, not just the memory knob: InfoNCE negatives
only come from tensors sharing a forward pass, so gradient accumulation cannot
substitute for it (see info_nce_loss). For the same reason DDP is deliberately not
supported -- per-rank batches would quietly weaken the loss instead of scaling it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate_glue import (  # noqa: E402
    DEFAULT_LLRD,
    OPTIM_DEFAULTS,
    build_optimizer,
    set_seed,
)
from nexterabert.data import build_tokenizer  # noqa: E402
from nexterabert.hf_baselines import (  # noqa: E402
    hf_llrd,
    is_hf_model_dir,
    load_hf_contrastive_model,
    load_hf_tokenizer,
    save_hf_contrastive_model,
)
from nexterabert.simcse import (  # noqa: E402
    CONTRASTIVE_RUN_FILE,
    DEFAULT_TEMPERATURE,
    freeze_unused,
    info_nce_loss,
    load_contrastive_model,
    save_contrastive_model,
)

DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"

# Protocol defaults. `None` for a CLI flag means "take the value here"; every
# one of them is overridable.
PROTOCOLS = {
    # OptiBERT App. D.2 -> supervised SimCSE (Gao et al., 2021, Sec. 6.1) for the
    # constants the appendix delegates rather than states.
    "mteb-nli": {
        "pooling": "attentive", "epochs": 3, "batch_size": 512,
        "max_len": 64, "lr": 5e-5, "warmup_pct": 0.06, "max_samples": 0,
        "eval_every": 250,
    },
}

# NLI label ids shared by MNLI and SNLI: 0 entailment, 1 neutral, 2 contradiction
# (-1 marks SNLI's examples with no gold label, which are dropped).
ENTAILMENT, CONTRADICTION = 0, 2
NLI_SOURCES = ("nyu-mll/multi_nli", "stanfordnlp/snli")


# ---------------------------------------------------------------------------
# Data: (anchor, positive, hard negative) triplets
# ---------------------------------------------------------------------------

def build_nli_triplets() -> list[tuple[str, str, str]]:
    """Group NLI rows into SimCSE's supervised triplets.

    A premise that carries both an entailment and a contradiction hypothesis
    yields one triplet per pair it can form -- the construction behind SimCSE's
    released `nli_for_simcse.csv` (~275k triplets from SNLI+MNLI). The premise is
    the anchor, the entailed hypothesis the positive, the contradicting one the
    hard negative.
    """
    from datasets import load_dataset

    grouped: dict[str, tuple[list, list]] = {}
    for name in NLI_SOURCES:
        ds = load_dataset(name)["train"]
        premises, hypotheses, labels = ds["premise"], ds["hypothesis"], ds["label"]
        kept = 0
        for premise, hypothesis, label in zip(premises, hypotheses, labels):
            if label not in (ENTAILMENT, CONTRADICTION) or not premise or not hypothesis:
                continue
            entailed, contradicting = grouped.setdefault(premise, ([], []))
            (entailed if label == ENTAILMENT else contradicting).append(hypothesis)
            kept += 1
        print(f"[data] {name}: {kept} usable rows ({len(ds)} total)")

    triplets = []
    for premise, (entailed, contradicting) in grouped.items():
        for positive, negative in zip(entailed, contradicting):
            triplets.append((premise, positive, negative))
    print(f"[data] {len(triplets)} triplets from {len(grouped)} premises")
    return triplets


def build_triplets(cache: str | None = None, max_samples: int = 0):
    """The NLI triplets, read from / written to ``cache`` when one is given."""
    if cache and Path(cache).exists():
        with open(cache, encoding="utf-8") as f:
            triplets = [tuple(t) for t in json.load(f)]
        print(f"[data] {len(triplets)} triplets from cache {cache}")
        return triplets[:max_samples] if max_samples else triplets

    triplets = build_nli_triplets()

    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(triplets, f)
        print(f"[data] cached -> {cache}")
    return triplets


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        return self.triplets[i]


class TripletCollator:
    """Tokenise a batch of triplets into padded id/mask tensors: one flat (3B, T)
    batch with the views interleaved (a0, p0, n0, a1, ...), which the embeddings
    split back out of with a stride-3 slice."""

    def __init__(self, tokenizer, max_len):
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        enc = self.tokenizer([text for triplet in batch for text in triplet],
                             padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# STS-B dev: SimCSE's checkpoint selection metric
# ---------------------------------------------------------------------------

def load_stsb_dev():
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "stsb")["validation"]
    return ds["sentence1"], ds["sentence2"], np.asarray(ds["label"], dtype=np.float64)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, dev, max_len, batch_size, autocast):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        with autocast:
            emb = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev))
        out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(out)


def stsb_spearman(model, tokenizer, stsb, dev, max_len, batch_size, autocast) -> float:
    """Spearman correlation between cosine similarity and the human score."""
    was_training = model.training
    model.eval()
    s1, s2, gold = stsb
    e1 = embed_texts(model, tokenizer, s1, dev, max_len, batch_size, autocast)
    e2 = embed_texts(model, tokenizer, s2, dev, max_len, batch_size, autocast)
    sims = (e1 * e2).sum(dim=-1).numpy()
    model.train(was_training)
    return float(spearmanr(sims, gold)[0])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    # Match GLUE/pretraining throughput on Ampere+ without changing the requested
    # autocast dtype. This is a no-op for CPU and unsupported CUDA hardware.
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit(
            "finetune_contrastive.py is single-process by design: InfoNCE negatives "
            "come from one forward pass, so DDP would shrink the objective's "
            "negatives per rank. Run it without torchrun and raise --batch_size.")

    hf_source = args.hf_model or (args.model if is_hf_model_dir(args.model) else None)
    n_layer, depth_fn = None, None
    if hf_source:
        # a Hugging Face baseline: its own tokenizer, the same pooling head /
        # InfoNCE / optimiser recipe (nexterabert.hf_baselines)
        tokenizer = load_hf_tokenizer(args.tokenizer)
        model, config, info = load_hf_contrastive_model(
            hf_source, pooling=args.pooling, max_len=args.max_len)
        frozen = model.freeze_unused()
        n_layer, depth_fn = hf_llrd(model)
        print(f"[hf] {hf_source}: {type(model.encoder).__name__}, "
              f"{n_layer} layers, hidden {model.hidden_size}, "
              f"{sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M params")
    else:
        tokenizer = build_tokenizer(args.tokenizer)
        model, config, info = load_contrastive_model(args.model, pooling=args.pooling)
        frozen = freeze_unused(model)
    if info["missing"]:
        print(f"[warn] missing encoder keys: {info['missing'][:4]} ...")
    if info.get("pooling_head"):
        print("[info] resuming from an already-trained pooling head")
    model.to(dev)

    triplets = build_triplets(cache=args.triplets_cache, max_samples=args.max_samples)
    loader = DataLoader(TripletDataset(triplets), batch_size=args.batch_size,
                        shuffle=True, drop_last=True,
                        collate_fn=TripletCollator(tokenizer, args.max_len),
                        num_workers=args.num_workers, pin_memory=True)

    optimizer, groups = build_optimizer(
        model, args.lr, args.weight_decay, (args.beta1, args.beta2), args.eps,
        llrd=args.llrd, n_layer=n_layer, depth_fn=depth_fn)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        decay = (1.0 - t) if args.schedule == "linear" else 0.5 * (1.0 + math.cos(math.pi * t))
        return args.alpha_f + (1.0 - args.alpha_f) * decay

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = args.dtype != "fp32" and dev.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    autocast = (torch.autocast(device_type=dev.type, dtype=amp_dtype)
                if use_amp else contextlib.nullcontext())

    train_model = model
    if args.compile and dev.type == "cuda":
        # dynamic=True: the batch's padded length changes every step
        train_model = torch.compile(model, dynamic=True)

    stsb = load_stsb_dev() if args.eval_every > 0 else None

    print(f"[{args.protocol}] {len(triplets)} triplets  pooling={args.pooling}  "
          f"bs={args.batch_size}  epochs={args.epochs}  steps={total_steps}  "
          f"lr={args.lr:.1e}  wd={args.weight_decay}  tau={args.temperature}  "
          f"max_len={args.max_len}  "
          f"llrd={args.llrd:g}  warmup={warmup_steps}  dtype={args.dtype}  "
          f"dev={dev.type}")
    n_decayed = sum(g["n_params"] for g in groups if not g["no_decay"])
    n_free = sum(g["n_params"] for g in groups if g["no_decay"])
    print(f"[{args.protocol}] decayed {n_decayed/1e6:.1f}M params | no-decay "
          f"{n_free/1e6:.1f}M | frozen {len(frozen)} unused tensors")
    if args.llrd != 1.0:
        group_lrs = [group["lr"] for group in groups]
        print(f"[{args.protocol}] llrd={args.llrd:g}/layer on the protocol lr: "
              f"{len(groups)} param groups, lr {min(group_lrs):.2e} "
              f"(embeddings) .. {max(group_lrs):.2e} (top)")

    best_score, best_state, step = None, None, 0
    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        for ids, mask in loader:
            with autocast:
                embedded = train_model(input_ids=ids.to(dev, non_blocking=True),
                                       attention_mask=mask.to(dev, non_blocking=True))
            anchor, positive, negative = (embedded[i::3] for i in range(3))  # interleaved a,p,n
            loss = info_nce_loss(anchor, positive, negative,
                                 temperature=args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                print(f"  step {step}/{total_steps}  epoch {epoch+1}/{args.epochs}  "
                      f"loss {loss.item():.4f}  "
                      f"top_lr {max(sched.get_last_lr()):.2e}  "
                      f"{time.time()-start:.0f}s")
            if args.eval_every > 0 and step % args.eval_every == 0:
                score = stsb_spearman(model, tokenizer, stsb, dev, args.max_len,
                                      args.batch_size, autocast)
                flag = ""
                if best_score is None or score > best_score:
                    best_score = score
                    # CPU snapshot so the best step survives the rest of the run
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()}
                    flag = "  *best*"
                print(f"  step {step}: STS-B dev spearman {score:.4f}{flag}")

    if best_state is not None:
        print(f"[{args.protocol}] restoring best checkpoint (STS-B dev {best_score:.4f})")
        model.load_state_dict(best_state)
    elif args.eval_every > 0:
        print("[warn] no evaluation ran (fewer steps than --eval_every); keeping last")

    summary = {
        "protocol": args.protocol,
        "pooling": args.pooling,
        "dataset": "nli",
        "model": hf_source or args.model,
        "backbone": "hf" if hf_source else "nexterabert",
        "hf_model": hf_source,
        "tokenizer": args.tokenizer,
        "triplets": len(triplets),
        "steps": total_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "max_len": args.max_len,
        "llrd": args.llrd,
        "betas": [args.beta1, args.beta2],
        "eps": args.eps,
        "schedule": args.schedule,
        "warmup_pct": args.warmup_pct,
        "warmup_steps": warmup_steps,
        "alpha_f": args.alpha_f,
        "grad_clip": args.grad_clip,
        "dtype": args.dtype,
        "compile": args.compile,
        "seed": args.seed,
        "stsb_dev_spearman": best_score,
        "wall_clock_seconds": time.time() - start,
    }
    if hf_source:
        out = save_hf_contrastive_model(model, args.output_dir, tokenizer=tokenizer,
                                        run_summary=summary)
    else:
        out = save_contrastive_model(model, args.output_dir, config=config,
                                     tokenizer=tokenizer, run_summary=summary)
    print(json.dumps(summary, indent=2))
    print(f"[{args.protocol}] saved -> {out} (protocol recorded in {CONTRASTIVE_RUN_FILE})")
    print(f"[{args.protocol}] next: python scripts/evaluate_mteb.py --model {out}")


def parse_args(argv=None):
    g = OPTIM_DEFAULTS
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="mteb-nli", choices=sorted(PROTOCOLS),
                   help="the recipe to follow (OptiBERT App. D.2); sets the defaults below")
    p.add_argument("--model",
                   help="pretrained NexteraBERT backbone directory (or a directory "
                        "written by an earlier --hf_model run, to continue it)")
    p.add_argument("--hf_model",
                   help="a Hugging Face baseline instead, e.g. answerdotai/ModernBERT-base "
                        "or chandar-lab/NeoBERT: same head, loss, data and optimiser, "
                        "so its MTEB row is comparable to NexteraBERT's")
    p.add_argument("--output_dir", required=True, help="where the tuned model goes")
    p.add_argument("--tokenizer", default=None,
                   help="default: ModernBERT's for a NexteraBERT backbone, the "
                        "baseline's own for --hf_model / a baseline directory")
    # --- protocol-pinned; None means "take the protocol's value" ---
    p.add_argument("--pooling", default=None, choices=["attentive", "mean"],
                   help="attentive adds a trained pooling head; mean tunes the "
                        "encoder under the backbone's own masked mean pooling")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="triplets per step; also the number of in-batch negatives, "
                        "so halving it on OOM weakens the objective (gradient "
                        "accumulation is not an equivalent substitute)")
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap the triplet count (0 = all)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--warmup_pct", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    # --- optimiser (evaluate_glue.py's recipe) ---
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="torch-scale (per-step decay lr_t*wd*p), matrices only")
    p.add_argument("--llrd", type=float, default=DEFAULT_LLRD,
                   help="layerwise LR decay (LLRD) factor in (0, 1], applied as a "
                        "multiplier on the protocol LR: the top keeps that LR and "
                        "the embeddings use llrd**(n_layer+1) times it. Matches "
                        f"the GLUE default ({DEFAULT_LLRD}); typical 0.8-0.95. "
                        "1.0 (or --no_llrd) restores the flat published protocol")
    p.add_argument("--no_llrd", dest="llrd_enabled", action="store_false",
                   help="disable layerwise LR decay and use the protocol LR at "
                        "every depth (the published protocol)")
    p.set_defaults(llrd_enabled=True)
    p.add_argument("--beta1", type=float, default=g["beta1"])
    p.add_argument("--beta2", type=float, default=g["beta2"])
    p.add_argument("--eps", type=float, default=g["eps"])
    p.add_argument("--schedule", default=g["schedule"], choices=["linear", "cosine"])
    p.add_argument("--alpha_f", type=float, default=g["alpha_f"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    # --- run control ---
    p.add_argument("--eval_every", type=int, default=None,
                   help="STS-B dev checkpoint selection, as in SimCSE; 0 disables")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true",
                               help="enable torch.compile on CUDA (default)")
    compile_group.add_argument("--no_compile", dest="compile", action="store_false",
                               help="disable torch.compile")
    p.set_defaults(compile=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--triplets_cache", default=None,
                   help="JSON file to cache the built triplets in (skips the rebuild)")

    args = p.parse_args(argv)
    if bool(args.model) == bool(args.hf_model):
        p.error("pass exactly one of --model (NexteraBERT backbone) or --hf_model "
                "(Hugging Face baseline)")
    if args.tokenizer is None:
        if args.hf_model or is_hf_model_dir(args.model):
            args.tokenizer = args.hf_model or args.model
        else:
            args.tokenizer = DEFAULT_TOKENIZER
    for key, value in PROTOCOLS[args.protocol].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not args.llrd_enabled:
        args.llrd = 1.0
    if not 0.0 < args.llrd <= 1.0:
        p.error(f"--llrd must be in (0, 1] (got {args.llrd}); "
                "1.0 disables layerwise decay")
    return args


if __name__ == "__main__":
    train(parse_args())
