#!/usr/bin/env python
"""ColBERT (multi-vector) training for the ModernBERT retrieval protocol.

Reproduces the "Multi vector retrieval" stage of Warner et al. (2024), 3.1.2 +
App. E.2, on a NexteraBERT backbone:

  * the JaColBERTv2.5 (Clavie, 2024) recipe: knowledge distillation with the
    KL-divergence between the normalised teacher and student MaxSim scores
    (``pylate.losses.Distillation``),
  * 810k MS MARCO queries, each with 32 candidate passages scored by the BGE-M3
    teacher (``lightonai/ms-marco-en-bge``: ``train`` / ``queries`` / ``documents``),
  * PyLate (Chaffin and Sourty, 2024), batch 16 (their public examples/train_pylate.py;
    App. E.2 writes it as 8 x 2 for base), 5% linear warmup, one epoch,
  * a learning rate chosen by sweeping [1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4] on
    NFCorpus/SciFact/TREC-COVID/FiQA (Table 9: 1e-4 for ModernBERT-base). The
    sweep loop lives in scripts/eval_colbert.sh; this script trains one point.

The backbone is loaded through ``transformers`` remote code (``AutoModel`` +
``trust_remote_code``), exactly as a Hub user would, so ``--model`` can be a Hub
repo id (``RikkaBotan/NexteraBERT-Mezzoforte-220M-en``) or a directory that ``upload_to_hub.py``
/ ``export.save_pretrained(bundle_source=True)`` produced. PyLate appends a
128-d projection and the ``[Q] `` / ``[D] `` marker tokens (the input embedding is
resized), then ``model.save`` writes an ordinary sentence-transformers directory
that ``scripts/evaluate_colbert.py`` reads back.

    python scripts/train_colbert.py --model RikkaBotan/NexteraBERT-Mezzoforte-220M-en \\
        --output_dir checkpoints/colbert/lr1e-4 --lr 1e-4

PyLate pins sentence-transformers (5.3.x at the time of writing) below what the
rest of this repo locks, so eval_colbert.sh runs this in its own virtualenv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

KD_DATASET = "lightonai/ms-marco-en-bge"
RUN_FILE = "colbert_run.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Hub repo id or local directory of the NexteraBERT backbone "
                        "(loaded with trust_remote_code)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="peak learning rate (the Table 9 value for ModernBERT-base)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16,
                   help="per-device batch of queries (each with --n_ways documents); "
                        "16 as in ModernBERT's examples/train_pylate.py (App. E.2 "
                        "writes it as 8 x 2 for base)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps (reference: 1)")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--n_ways", type=int, default=32,
                   help="teacher-scored documents kept per query")
    p.add_argument("--max_samples", type=int, default=0,
                   help="cap the 810k training queries (0 = all); a fixed-seed shuffle "
                        "makes a capped run a random sample")
    p.add_argument("--query_length", type=int, default=32)
    p.add_argument("--document_length", type=int, default=180,
                   help="document truncation during training (the PyLate default; the "
                        "paper does not state one)")
    p.add_argument("--embedding_size", type=int, default=128)
    p.add_argument("--dataset", default=KD_DATASET)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_trust_remote_code", dest="trust_remote_code",
                   action="store_false", help="for plain HF architectures (baselines)")
    p.add_argument("--revision", default=None, help="Hub revision of --model")
    p.add_argument("--work_dir", default=None,
                   help="trainer scratch directory (default: <output_dir>/_trainer)")
    return p.parse_args(argv)


def load_kd_dataset(name: str, max_samples: int, n_ways: int, seed: int):
    from datasets import load_dataset
    from pylate import utils

    train = load_dataset(name, "train", split="train")
    queries = load_dataset(name, "queries", split="train")
    documents = load_dataset(name, "documents", split="train")
    if max_samples and max_samples < len(train):
        train = train.shuffle(seed=seed).select(range(max_samples))
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents, n_ways=n_ways).transform)
    print(f"[data] {name}: {len(train)} queries x {n_ways} teacher-scored documents")
    return train


def main(argv=None):
    args = parse_args(argv)
    import torch
    from pylate import losses, models, utils
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    torch.manual_seed(args.seed)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from nexterabert.modeling_nexterabert import ensure_dynamo_recompile_budget

        ensure_dynamo_recompile_budget()   # see train_st_dpr.py: keeps FlexAttention fused
    except Exception as err:  # noqa: BLE001 - a plain HF baseline has no such module
        print(f"[colbert] dynamo recompile budget left at default ({err})")
    model = models.ColBERT(
        model_name_or_path=args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        embedding_size=args.embedding_size,
        query_length=args.query_length,
        document_length=args.document_length,
    )
    try:
        from nexterabert.hf_baselines import repair_wrapped_neobert

        # PyLate opens the encoder with a plain AutoModel call: a NeoBERT inside has a
        # zero-filled RoPE table (uniform attention, no error) until it is rebuilt.
        repair_wrapped_neobert(model, max(args.document_length, args.query_length))
    except ImportError as err:
        print(f"[colbert] NeoBERT RoPE repair unavailable ({err})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[colbert] {args.model}: {n_params / 1e6:.1f}M params, "
          f"projection -> {args.embedding_size}d, q_len {args.query_length}, "
          f"d_len {args.document_length}")

    train = load_kd_dataset(args.dataset, args.max_samples, args.n_ways, args.seed)

    work_dir = args.work_dir or os.path.join(args.output_dir, "_trainer")
    train_args = SentenceTransformerTrainingArguments(
        output_dir=work_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        logging_steps=args.log_every,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        run_name=Path(args.output_dir).name,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        loss=losses.Distillation(model=model),
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    effective = args.batch_size * args.grad_accum
    print(f"[colbert] KD training: lr={args.lr:.1e} bs={args.batch_size}x{args.grad_accum}"
          f" (={effective}) epochs={args.epochs} warmup={args.warmup_ratio:.0%} "
          f"dtype={args.dtype}")
    start = time.time()
    trainer.train()
    wall = time.time() - start

    # under torchrun every rank runs this script; only rank 0 writes the checkpoint
    if int(os.environ.get("RANK", "0")) != 0:
        return
    model.save(args.output_dir, create_model_card=False)
    summary = {
        "protocol": "colbert-msmarco-kd",
        "source": "Warner et al. 2024, 3.1.2 + App. E.2 (JaColBERTv2.5 recipe)",
        "model": args.model,
        "dataset": args.dataset,
        "queries": len(train),
        "n_ways": args.n_ways,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "query_length": args.query_length,
        "document_length": args.document_length,
        "embedding_size": args.embedding_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "wall_clock_seconds": wall,
    }
    with open(os.path.join(args.output_dir, RUN_FILE), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[colbert] saved -> {args.output_dir}")
    print(f"[colbert] next: python scripts/evaluate_colbert.py --model {args.output_dir}")


if __name__ == "__main__":
    main()
