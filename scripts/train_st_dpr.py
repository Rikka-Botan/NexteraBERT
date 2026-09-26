#!/usr/bin/env python
"""ModernBERT's single-vector (DPR) retrieval stage, reproduced with sentence-transformers.

A line-for-line port of the public reference script
(AnswerDotAI/ModernBERT, examples/train_st.py) behind Warner et al. (2024) Table 1
"IR (DPR)", so the resulting BEIR / MLDR / CoIR numbers are comparable to the
paper's -- nothing of this repo's own training loop (InfoNCE implementation,
layerwise LR decay, weight decay, collator) is in the path:

  * plain ``SentenceTransformer(model)``: the encoder's last hidden state under
    **mean pooling**, no extra head (the Hub model loads through trust_remote_code);
  * data ``sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1``
    config ``triplet-hard`` (query / positive / mined hard negative),
    ``train_test_split(test_size=1000, seed=12)`` then the first **1.25M** rows;
  * ``CachedMultipleNegativesRankingLoss(mini_batch_size=16)`` = InfoNCE at scale
    20 over all positives + hard negatives of the batch, with
    ``per_device_train_batch_size=512`` (1023 negatives per query) and
    ``BatchSamplers.NO_DUPLICATES``;
  * one epoch, 5% linear warmup, bf16, AdamW defaults (lr 2e-5 ... 1e-4 swept in
    the paper; **8e-5** selected for ModernBERT-base, Table 9);
  * a ``TripletEvaluator`` on the 1000 held-out triplets before and after.

    python scripts/train_st_dpr.py --model RikkaBotan/NexteraBERT-Mezzoforte-220M-en \
        --output_dir checkpoints/st_dpr/lr8e-5 --lr 8e-5

The output is an ordinary sentence-transformers directory plus this repo's
``contrastive_run.json`` marker (protocol ``st-msmarco``), which
``scripts/evaluate_retrieval.py`` recognises and scores through mteb exactly as
the reference ``evaluate_st.py`` does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MSMARCO_DATASET = ("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
                   "triplet-hard")
RUN_FILE = "contrastive_run.json"
PROTOCOL = "st-msmarco"
# ModernBERT examples/train_st.py, verbatim values
DEFAULTS = {"batch_size": 512, "max_samples": 1_250_000, "eval_size": 1000, "lr": 8e-5,
            "eval_batch_size": 16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Hub repo id or directory: a NexteraBERT export (trust_remote_code), "
                        "any HF encoder (bert-base-uncased, answerdotai/ModernBERT-base), or "
                        "a sentence-transformers directory to continue from")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=None,
                   help="peak learning rate (default 8e-5, ModernBERT-base's Table 9 value)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="per-device batch = in-batch negatives + 1 (default 512)")
    p.add_argument("--mini_batch_size", type=int, default=16,
                   help="CachedMNRL forward chunk; memory only, does not change the loss")
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="TripletEvaluator encode batch (default 16)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="training rows (default 1,250,000; 0 = all)")
    p.add_argument("--eval_size", type=int, default=None,
                   help="held-out triplets for the TripletEvaluator (reference 1000); "
                        "0 = none")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=None,
                   help="override the model's max_seq_length (the reference leaves it "
                        "at the model's own)")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"],
                   help="pooling of a freshly wrapped encoder (the reference: mean)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=12, help="train_test_split seed (reference: 12)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=0,
                   help="checkpoint every N steps into <output_dir>/_trainer (0 = off)")
    p.add_argument("--no_eval", action="store_true", help="skip the TripletEvaluator")
    p.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--attn_implementation", default=None,
                   help="passed to from_pretrained, e.g. flash_attention_2 (a baseline "
                        "whose default path builds a dense (B,1,T,T) mask -- "
                        "LFM2.5-Encoder -- needs it, or a small batch, at 8192 tokens)")
    p.add_argument("--revision", default=None)
    p.add_argument("--work_dir", default=None, help="trainer scratch dir (default <output_dir>/_trainer)")
    args = p.parse_args(argv)
    for key, value in DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def load_model(args):
    from sentence_transformers import SentenceTransformer, models

    path = args.model
    is_st_dir = os.path.isdir(path) and os.path.exists(os.path.join(path, "modules.json"))
    model_kwargs = ({"attn_implementation": args.attn_implementation}
                    if args.attn_implementation else {})
    if is_st_dir:
        model = SentenceTransformer(path, trust_remote_code=args.trust_remote_code,
                                    model_kwargs=model_kwargs or None)
        source = "sentence-transformers directory"
    else:
        # sentence-transformers opens the encoder with a plain AutoModel call. For
        # LFM2.5-Encoder that silently yields RANDOM weights (checkpoint saved under
        # the MaskedLM prefix); swap in a repaired copy of the backbone when needed.
        from nexterabert.hf_baselines import ensure_automodel_loadable

        repaired = ensure_automodel_loadable(path)
        if repaired != path:
            print(f"[st] {path}: not usable by a plain AutoModel + tokenizer as-is "
                  f"(see {repaired}/backbone_repair.json); training from {repaired}")
            path = repaired
        args.loaded_from = path
        # exactly what SentenceTransformer(name) builds for a bare encoder: a
        # Transformer module + mean Pooling. Built explicitly so the pooling is
        # stated (and cls is available) whatever the checkpoint's metadata says.
        hub_args = {"trust_remote_code": args.trust_remote_code}
        if args.revision:
            hub_args["revision"] = args.revision
        transformer = models.Transformer(path, model_args={**hub_args, **model_kwargs},
                                         tokenizer_args=dict(hub_args),
                                         config_args=dict(hub_args))
        pooling = models.Pooling(transformer.get_word_embedding_dimension(),
                                 pooling_mode=args.pooling)
        model = SentenceTransformer(modules=[transformer, pooling])
        source = f"bare encoder + {args.pooling} pooling"
        if "message" in getattr(transformer, "modality_config", {}):
            print("[st] [warn] this tokenizer carries a chat template, so "
                  "sentence-transformers wraps every text in it -- not the framing an "
                  "MLM encoder was pretrained on.")
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    # A NeoBERT opened by sentence-transformers has a zero-filled RoPE table (uniform
    # attention, silently) sized to 4096; rebuild it before a single step is taken.
    from nexterabert.hf_baselines import repair_wrapped_neobert

    repair_wrapped_neobert(model, model.max_seq_length)
    print(f"[st] {path}: {source}, max_seq_length={model.max_seq_length}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def complete_remote_code(model, output_dir) -> list:
    """Make a saved trust_remote_code checkpoint reloadable.

    ``save_pretrained`` keeps ``auto_map`` in config.json but copies the module
    file only for a class transformers registered for its auto class -- which it
    does not for an encoder opened from a local directory (the repaired
    LFM2.5-Encoder backbone). The directory would then name a
    ``modeling_*.py`` it does not contain and ``SentenceTransformer(dir)`` fails at
    evaluation time, after the whole training run.
    """
    from nexterabert.hf_baselines import _copy_remote_code

    try:
        auto_model = model[0].auto_model
    except (AttributeError, IndexError, TypeError):
        return []
    return _copy_remote_code(auto_model, Path(output_dir))


def load_triplets(args):
    """(train Dataset, eval Dataset | None) with columns query / positive / negative."""
    from datasets import load_dataset

    name, config = MSMARCO_DATASET
    dataset = load_dataset(name, config, split="train")
    eval_dataset = None
    if args.eval_size and args.eval_size > 0:
        split = dataset.train_test_split(test_size=args.eval_size, seed=args.split_seed)
        dataset, eval_dataset = split["train"], split["test"]
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))   # the reference takes the head
    print(f"[data] {name} [{config}]: {len(dataset)} train triplets"
          + (f", {len(eval_dataset)} held out" if eval_dataset is not None else ""))
    return dataset, eval_dataset


def protocol_name(n_triplets: int) -> str:
    """``st-msmarco`` only for the published budget. A run on fewer triplets
    (DPR_BUDGET=lite, a smoke run) records e.g. ``st-msmarco-250k``, so no summary
    can file it as the full stage."""
    if n_triplets < DEFAULTS["max_samples"]:
        return f"{PROTOCOL}-{max(n_triplets // 1000, 1)}k"
    return PROTOCOL


def check_dependencies():
    """Fail early with a plain message: the HF Trainer behind SentenceTransformerTrainer
    needs accelerate, which is not a dependency of sentence-transformers itself."""
    try:
        import accelerate  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            "train_st_dpr.py needs `accelerate>=1.1.0` (the Hugging Face Trainer behind "
            "SentenceTransformerTrainer). Install the repo's eval extras, which include it:\n"
            "    pip install -e '.[eval]'      # or: pip install 'accelerate>=1.1.0'"
        ) from err


def main(argv=None):
    args = parse_args(argv)
    check_dependencies()
    import torch
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    # NexteraBERT's FlexAttention is torch.compiled; dynamo's default recompile
    # budget (8) is exhausted by train -> no_grad eval -> ragged batches, after
    # which it silently falls back to the dense O(T^2) path (64 GiB at 8192 tokens).
    # The model may load through remote code (a bundled copy of the modeling file),
    # so raise the process-wide budget from here rather than trusting that copy.
    from nexterabert.modeling_nexterabert import ensure_dynamo_recompile_budget

    ensure_dynamo_recompile_budget()
    from sentence_transformers.evaluation import TripletEvaluator
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    torch.manual_seed(args.seed)
    model = load_model(args)
    train_dataset, eval_dataset = load_triplets(args)

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    work_dir = args.work_dir or os.path.join(args.output_dir, "_trainer")
    train_args = SentenceTransformerTrainingArguments(
        output_dir=work_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=2,
        logging_steps=args.log_every,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        run_name=Path(args.output_dir).name,
    )

    evaluator = None
    before = None
    if eval_dataset is not None and not args.no_eval:
        evaluator = TripletEvaluator(anchors=eval_dataset["query"],
                                     positives=eval_dataset["positive"],
                                     negatives=eval_dataset["negative"],
                                     name="msmarco-co-condenser-dev",
                                     batch_size=args.eval_batch_size)
        before = evaluator(model)
        print(f"[st] triplet accuracy before training: {before}")

    steps = int(len(train_dataset) // args.batch_size * args.epochs)
    print(f"[st] {protocol_name(len(train_dataset))}: lr={args.lr:.1e} bs={args.batch_size} "
          f"(mini {args.mini_batch_size}) epochs={args.epochs} steps~{steps} "
          f"warmup={args.warmup_ratio:.0%} max_seq_length={model.max_seq_length} dtype={args.dtype}")
    from transformers import TrainerCallback

    diverged = {}

    class StopOnNonFinite(TrainerCallback):
        """The HF trainer does not stop on a NaN loss: it keeps stepping, clipping a
        NaN gradient norm turns every weight NaN, and the run "finishes" normally.
        Stop at the first non-finite loss or gradient norm that gets logged."""

        def on_log(self, _args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                value = (logs or {}).get(key)
                if value is not None and not math.isfinite(float(value)):
                    diverged.update(step=state.global_step, what=key, value=value)
                    control.should_training_stop = True

    trainer = SentenceTransformerTrainer(
        model=model, args=train_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset if evaluator is not None else None,
        loss=loss, evaluator=evaluator, callbacks=[StopOnNonFinite()],
    )
    start = time.time()
    trainer.train()
    wall = time.time() - start

    from nexterabert.hf_baselines import non_finite_parameters

    bad = non_finite_parameters(model)
    if diverged or bad:
        # No checkpoint and, above all, no contrastive_run.json: eval_common.sh takes
        # that file to mean "this stage is done" and would score the dead model.
        raise SystemExit(
            f"[st] [ERROR] training diverged at lr={args.lr:.1e}: "
            + (f"{diverged['what']}={diverged['value']} at step {diverged['step']}"
               if diverged else "non-finite weights after training")
            + (f"; non-finite tensors e.g. {bad[:3]}" if bad else "")
            + ". Nothing was saved. Use a lower --lr (DPR_LR=5e-5 bash "
              "scripts/eval_dpr.sh) or --dtype fp32.")

    # Save BEFORE the post-training evaluation: the evaluation is optional
    # bookkeeping, the checkpoint is the product, and an OOM during it must not
    # throw the training away.
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(args.output_dir, create_model_card=False)
    copied = complete_remote_code(model, args.output_dir)
    if copied:
        print(f"[st] copied remote code into {args.output_dir}: {', '.join(copied)}")
    print(f"[st] saved -> {args.output_dir}")

    after = None
    if evaluator is not None:
        # release the trainer's optimizer state / cached activations first
        del trainer, loss
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.eval()
        try:
            after = evaluator(model)
            print(f"[st] triplet accuracy after training: {after}")
        except torch.OutOfMemoryError as err:
            print(f"[warn] post-training TripletEvaluator ran out of GPU memory "
                  f"({str(err).splitlines()[0][:120]}); the checkpoint is saved, only "
                  f"triplet_eval_after is missing. Re-run with a smaller "
                  f"--eval_batch_size to record it.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {
        "protocol": protocol_name(len(train_dataset)),
        "source": "Warner et al. 2024, 3.1.2 + App. E.2; AnswerDotAI/ModernBERT examples/train_st.py",
        "framework": "sentence-transformers",
        "pooling": args.pooling if not (os.path.isdir(args.model) and
                                        os.path.exists(os.path.join(args.model, "modules.json")))
        else "inherited",
        "model": args.model,
        "loaded_from": getattr(args, "loaded_from", args.model),
        "dataset": MSMARCO_DATASET[0] + "/" + MSMARCO_DATASET[1],
        "triplets": len(train_dataset),
        "steps": steps,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": model.max_seq_length,
        "loss": "CachedMultipleNegativesRankingLoss (scale 20)",
        "dtype": args.dtype,
        "seed": args.seed,
        "eval_size": args.eval_size,
        "triplet_eval_before": before,
        "triplet_eval_after": after,
        "wall_clock_seconds": wall,
    }
    with open(os.path.join(args.output_dir, RUN_FILE), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"[st] next: python scripts/evaluate_retrieval.py --model {args.output_dir} --benchmark BEIR")


if __name__ == "__main__":
    main()
