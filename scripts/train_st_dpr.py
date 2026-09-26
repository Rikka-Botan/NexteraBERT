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
    python scripts/train_st_dpr.py --model bert-base-uncased ...        # the paper's BERT-base row

``--dataset mldr`` runs the paper's "Single Vector - In Domain" pass (3.1.3):
continue from a trained DPR directory on the English MLDR training split
(``--max_seq_length 8192``). The paper states no hyperparameters for it. A first
guess (the DPR lr 8e-5) pushed the in-domain score below the out-of-domain one,
so the defaults are conservative -- lr 2e-5, one sampled negative per query (~10k
triplets; the best measured setting, 41.6, while all 20 mined negatives over-trained
to 37.8; ``--negatives k`` samples k), batch 32, one epoch -- and 500 training
queries are held out
for a TripletEvaluator whose accuracy (``heldout_triplet_accuracy`` in
``contrastive_run.json``) lets scripts/eval_dpr.sh pick the lr of a sweep without
touching the test split.

The output is an ordinary sentence-transformers directory plus this repo's
``contrastive_run.json`` marker (protocol ``st-msmarco`` / ``st-mldr``), which
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MSMARCO_DATASET = ("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
                   "triplet-hard")
RUN_FILE = "contrastive_run.json"
PROTOCOLS = {
    # ModernBERT examples/train_st.py, verbatim values
    "msmarco": {"protocol": "st-msmarco", "batch_size": 512, "max_samples": 1_250_000,
                "eval_size": 1000, "lr": 8e-5, "eval_batch_size": 16},
    # 3.1.3 in-domain pass; the paper states no hyperparameters and publishes no
    # script for it (only the MS MARCO stage). A first guess at the DPR lr (8e-5)
    # LOWERED the in-domain score below the out-of-domain one (36.3 -> 29.5 on
    # nextera-130B), so the lr is a conservative 2e-5. One sampled negative per
    # query (~10k triplets of 8192-token documents, batch 32, one epoch) is the
    # best measured setting: 36.3 OOD -> 41.6 ID on nextera-130B, whereas all 20
    # mined negatives (~190k triplets, 20x the steps over the same 10k queries)
    # over-trained and fell to 37.8. --negatives <k> samples k per query; select
    # such variants on the MLDR dev split (diagnose_retrieval.py --split dev),
    # never on test. 500 whole queries are held out for the TripletEvaluator.
    "mldr": {"protocol": "st-mldr", "batch_size": 32, "max_samples": 0,
             "eval_size": 500, "lr": 2e-5, "negatives": "one", "eval_batch_size": 4},
}
PROTOCOLS["msmarco"]["negatives"] = "one"   # the dataset already has one per row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Hub repo id or directory: a NexteraBERT export (trust_remote_code), "
                        "any HF encoder (bert-base-uncased, answerdotai/ModernBERT-base), or "
                        "a sentence-transformers directory to continue from")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", default="msmarco", choices=sorted(PROTOCOLS))
    p.add_argument("--lr", type=float, default=None,
                   help="peak learning rate (default 8e-5, ModernBERT-base's Table 9 value)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="per-device batch = in-batch negatives + 1 (default 512 / mldr 32)")
    p.add_argument("--mini_batch_size", type=int, default=16,
                   help="CachedMNRL forward chunk; memory only, does not change the loss")
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="TripletEvaluator encode batch (default 16; mldr 4 -- its "
                        "held-out documents are 8192 tokens long)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="training rows (default 1,250,000; 0 = all)")
    p.add_argument("--eval_size", type=int, default=None,
                   help="held-out triplets (msmarco: rows, reference 1000) or queries "
                        "(mldr: default 500) for the TripletEvaluator; 0 = none")
    p.add_argument("--negatives", default=None,
                   help="mldr only: 'one' (default, ~10k triplets; best measured), an "
                        "integer k to sample k of the 20 mined negatives per query, or "
                        "'all' (~190k triplets, 20x the steps; over-trained: 37.8 vs 41.6)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=None,
                   help="override the model's max_seq_length (the reference leaves it "
                        "at the model's own; use 8192 for --dataset mldr)")
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
    for key, value in PROTOCOLS[args.dataset].items():
        if key != "protocol" and getattr(args, key, None) is None:
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


def triplet_dataset(rows):
    """A ``datasets.Dataset`` of (query, positive, negative) rows.

    The text columns are typed ``large_string`` (64-bit offsets): MLDR documents run
    to tens of thousands of characters and ~70k triplets put a column past the 2 GB
    that pyarrow's default ``string`` type can address, which surfaces as
    "offset overflow while concatenating arrays" inside datasets' fingerprinting.
    """
    from datasets import Dataset, Features, Value

    features = Features({"query": Value("large_string"), "positive": Value("large_string"),
                         "negative": Value("large_string")})
    return Dataset.from_dict({"query": [t[0] for t in rows],
                              "positive": [t[1] for t in rows],
                              "negative": [t[2] for t in rows]}, features=features)


def load_triplets(args):
    """(train Dataset, eval Dataset | None) with columns query / positive / negative."""
    from datasets import load_dataset

    if args.dataset == "msmarco":
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

    import random

    from finetune_contrastive import build_mldr_triplets

    triplets = build_mldr_triplets(max_samples=args.max_samples or 0,
                                   negatives=args.negatives or "one")
    # hold out whole queries (a query's triplets are adjacent in the list), so the
    # evaluator never sees a query that was trained on
    eval_triplets = []
    if args.eval_size and args.eval_size > 0:
        queries = list(dict.fromkeys(t[0] for t in triplets))
        held = set(random.Random(args.split_seed).sample(
            queries, min(args.eval_size, max(len(queries) // 10, 1))))
        eval_triplets = [t for t in triplets if t[0] in held]
        triplets = [t for t in triplets if t[0] not in held]

    def as_dataset(rows):
        return triplet_dataset(rows)

    print(f"[data] MLDR-en train: {len(triplets)} train triplets"
          + (f", {len(eval_triplets)} held out ({len(held)} queries)" if eval_triplets else ""))
    return as_dataset(triplets), (as_dataset(eval_triplets) if eval_triplets else None)


def protocol_name(args, n_triplets: int) -> str:
    """``st-msmarco`` only for the published budget. A run on fewer triplets
    (DPR_BUDGET=lite, a smoke run) records e.g. ``st-msmarco-250k``, so no summary
    can file it as the paper-comparable stage."""
    name = PROTOCOLS[args.dataset]["protocol"]
    reference = PROTOCOLS[args.dataset]["max_samples"]
    if args.dataset == "msmarco" and reference and n_triplets < reference:
        return f"{name}-{max(n_triplets // 1000, 1)}k"
    return name


def triplet_accuracy(metrics) -> float | None:
    """The accuracy out of a TripletEvaluator result (a {name_fn_accuracy: value} dict)."""
    if isinstance(metrics, (int, float)):
        return float(metrics)
    if isinstance(metrics, dict):
        values = [float(v) for k, v in metrics.items() if "accuracy" in k]
        return max(values) if values else None
    return None


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
                                     name="msmarco-co-condenser-dev" if args.dataset == "msmarco"
                                     else "mldr-en-heldout",
                                     batch_size=args.eval_batch_size)
        before = evaluator(model)
        print(f"[st] triplet accuracy before training: {before}")

    steps = int(len(train_dataset) // args.batch_size * args.epochs)
    print(f"[st] {protocol_name(args, len(train_dataset))}: lr={args.lr:.1e} bs={args.batch_size} "
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
            + ". Nothing was saved. Use a lower --lr (the protocol sweeps 1e-5 ... "
              "1e-4 per model for exactly this reason: DPR_LRS=\"1e-5 2e-5 3e-5 5e-5\" "
              "bash scripts/eval_dpr.sh) or --dtype fp32.")

    # Save BEFORE the post-training evaluation: the evaluation is optional
    # bookkeeping, the checkpoint is the product, and an OOM while encoding
    # 8192-token held-out documents must not throw the training away.
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
                  f"heldout_triplet_accuracy is missing. Re-run with a smaller "
                  f"--eval_batch_size to record it.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {
        "protocol": protocol_name(args, len(train_dataset)),
        "source": "Warner et al. 2024, 3.1.2 + App. E.2; AnswerDotAI/ModernBERT examples/train_st.py",
        "framework": "sentence-transformers",
        "pooling": args.pooling if not (os.path.isdir(args.model) and
                                        os.path.exists(os.path.join(args.model, "modules.json")))
        else "inherited",
        "model": args.model,
        "loaded_from": getattr(args, "loaded_from", args.model),
        "dataset": MSMARCO_DATASET[0] + "/" + MSMARCO_DATASET[1] if args.dataset == "msmarco"
        else "Shitao/MLDR en train",
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
        "negatives": args.negatives,
        "eval_size": args.eval_size,
        "triplet_eval_before": before,
        "triplet_eval_after": after,
        # the number eval_dpr.sh selects an MLDR lr sweep on
        "heldout_triplet_accuracy": triplet_accuracy(after),
        "wall_clock_seconds": wall,
    }
    with open(os.path.join(args.output_dir, RUN_FILE), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"[st] next: python scripts/evaluate_retrieval.py --model {args.output_dir} --benchmark BEIR")


if __name__ == "__main__":
    main()
