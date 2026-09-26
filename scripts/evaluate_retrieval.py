#!/usr/bin/env python
"""Retrieval evaluation (BEIR / NanoBEIR / MLDR / CoIR) of an MS MARCO checkpoint.

Mirrors the single-vector (DPR) half of the ModernBERT evaluation suite, which is
a TWO-STAGE protocol (Warner et al., 2024, 3.1.2): the backbone is first
fine-tuned contrastively on MS MARCO with mined hard negatives
(scripts/train_st_dpr.py), and only then scored from mean-pooled, L2-normalised
embeddings with nDCG@10. This script is the second stage; mteb scores the
sentence-transformers checkpoint directly, as ModernBERT's evaluate_st.py does:

    python scripts/train_st_dpr.py --model RikkaBotan/NexteraBERT-Mezzoforte-220M-en \
        --output_dir checkpoints/st_dpr/lr8e-5
    python scripts/evaluate_retrieval.py --model checkpoints/st_dpr/lr8e-5 \
        --benchmark BEIR --output beir_results.json

    # MLDR: the English test split at the full 8192 context
    python scripts/evaluate_retrieval.py --model checkpoints/st_dpr/lr8e-5 \
        --tasks MultiLongDocRetrieval --languages eng --eval_splits test --max_len 8192
    # Code (CoIR): CodeSearchNet code->text and StackOverflow-QA
    python scripts/evaluate_retrieval.py --model checkpoints/st_dpr/lr8e-5 \
        --tasks COIRCodeSearchNetRetrieval,StackOverflowQA --max_len 8192

The model directory records which stage produced it (contrastive_run.json); this
script reports that alongside the scores.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert.hf_baselines import (  # noqa: E402
    non_finite_parameters,
    repair_wrapped_neobert,
)


def is_st_model_dir(path) -> bool:
    """True for a sentence-transformers directory (what scripts/train_st_dpr.py writes)."""
    return bool(path) and Path(path).is_dir() and (Path(path) / "modules.json").exists()


def load_st_model(path, max_len, trust_remote_code=True, attn_implementation=None):
    """A SentenceTransformer scored the way ModernBERT's evaluate_st.py scores it:
    mteb wraps the model itself (its own pooling and tokenizer), only the sequence
    length is pinned to --max_len."""
    from sentence_transformers import SentenceTransformer

    from nexterabert.modeling_nexterabert import ensure_dynamo_recompile_budget

    # the remote-code copy of the model compiles FlexAttention; give dynamo enough
    # recompile budget that long-document batches never fall back to the dense path
    ensure_dynamo_recompile_budget()
    model_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else None
    model = SentenceTransformer(path, trust_remote_code=trust_remote_code,
                                model_kwargs=model_kwargs)
    model.max_seq_length = max_len
    # sentence-transformers loads the encoder with a plain AutoModel call: a NeoBERT
    # inside comes out with a zero-filled RoPE table sized to 4096 -- uniform attention
    # at any length, and an AssertionError past 4096 (--max_len 8192 for MLDR / CoIR).
    repair_wrapped_neobert(model, max_len)
    # A diverged run saves NaN weights like any other, and NaN similarities score a
    # quiet nDCG@10 = 0.0. That is not a result; refuse to produce it.
    bad = non_finite_parameters(model)
    if bad:
        raise SystemExit(f"[encoder] [ERROR] {path} holds non-finite weights (e.g. "
                         f"{bad[:3]}): its training diverged. Retrain it (lower lr) "
                         f"instead of scoring it -- every metric would read 0.0.")
    # mteb keys its result cache by the model NAME it reads from the model card
    # (model_card_data.model_name, else base_model). Two checkpoints fine-tuned
    # from the same backbone -- the full and the 250k-triplet MS MARCO stage --
    # therefore share a key and the second one silently gets the first one's
    # cached scores. Name each directory distinctly.
    tag = st_model_tag(path)
    try:
        model.model_card_data.model_name = tag
    except Exception as err:  # noqa: BLE001 - metadata only
        print(f"[warn] could not set the model card name ({err}); mteb may reuse cached "
              f"results of another checkpoint from the same base model")
    print(f"[encoder] sentence-transformers {path} (mteb name {tag}): {model}  "
          f"max_seq_length={max_len}")
    return model


def st_model_tag(path) -> str:
    """A cache key unique to one *checkpoint*: ``<parent>/<name>@<fingerprint>``.

    The directory name alone is not enough -- retraining into the same directory
    must not be served the previous run's cached scores. The fingerprint is taken
    from ``contrastive_run.json`` (unique per training run: it records wall-clock
    time) and, failing that, from the weight file's size and mtime, so an untouched
    checkpoint keeps its key and re-runs still hit the cache.
    """
    import hashlib

    p = Path(path).resolve()
    base = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
    marker = p / "contrastive_run.json"
    if marker.exists():
        digest = hashlib.sha1(marker.read_bytes()).hexdigest()[:8]
    else:
        weights = [f for f in ("model.safetensors", "pytorch_model.bin") if (p / f).exists()]
        if not weights:
            return base
        st = (p / weights[0]).stat()
        digest = hashlib.sha1(f"{weights[0]}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]
    return f"{base}@{digest}"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="sentence-transformers directory written by scripts/train_st_dpr.py")
    p.add_argument("--benchmark", default="NanoBEIR",
                   help="an mteb benchmark name (NanoBEIR, BEIR, CoIR, ...), or "
                        "comma-separated task names")
    p.add_argument("--tasks", default="",
                   help="comma-separated mteb task names; overrides --benchmark")
    p.add_argument("--languages", default="",
                   help="comma-separated ISO-639-3 codes to keep for multilingual "
                        "tasks, e.g. 'eng' to score only the English MLDR subset")
    p.add_argument("--eval_splits", default="",
                   help="comma-separated splits to score (default: the task's own)")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--attn_implementation", default=None,
                   help="passed to from_pretrained (e.g. flash_attention_2)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--st_dtype", default="fp32", choices=["fp32", "tf32", "bf16"],
                   help="compute precision of the evaluation. sentence-transformers "
                        "encodes in plain fp32, which on an Ampere+/Hopper GPU is "
                        "several times slower than it needs to be: 'tf32' lets fp32 "
                        "matmuls use TensorFloat-32, 'bf16' wraps the evaluation in "
                        "bf16 autocast (how these checkpoints were trained). Scores "
                        "move in the 3rd-4th decimal, so use ONE setting for every "
                        "model of a comparison; the value is recorded in the results "
                        "file as _st_dtype.")
    p.add_argument("--output", default="retrieval_results.json")
    p.add_argument("--output_folder", default="mteb_results",
                   help="mteb per-task result cache (a finished task is not rerun)")
    p.add_argument("--cache_dir", default=None, help=argparse.SUPPRESS)  # alias
    p.add_argument("--overwrite", default="only-missing",
                   choices=["never", "only-missing", "only-cache", "always"],
                   help="mteb >= 2 cache policy")
    args = p.parse_args(argv)
    if args.cache_dir:
        args.output_folder = args.cache_dir
    if not is_st_model_dir(args.model):
        p.error(f"--model {args.model} is not a sentence-transformers directory "
                f"(scripts/train_st_dpr.py writes one)")
    return args


def select_tasks(mteb, args):
    """--tasks / --benchmark (+ --languages / --eval_splits) -> list of mteb tasks.

    A benchmark keeps its own split choices (BEIR's MSMARCO is scored on dev, as
    in the original suite); the filters only apply to explicitly named tasks,
    which is how the English MLDR test subset is selected:
    ``--tasks MultiLongDocRetrieval --languages eng --eval_splits test``.
    """
    kwargs = {}
    if args.languages:
        kwargs["languages"] = [x.strip() for x in args.languages.split(",") if x.strip()]
    if args.eval_splits:
        kwargs["eval_splits"] = [x.strip() for x in args.eval_splits.split(",") if x.strip()]
    if args.tasks:
        names = [t.strip() for t in args.tasks.split(",") if t.strip()]
        return list(mteb.get_tasks(tasks=names, **kwargs))
    try:
        return list(mteb.get_benchmark(args.benchmark))
    except Exception:
        names = [t.strip() for t in args.benchmark.split(",") if t.strip()]
        return list(mteb.get_tasks(tasks=names, **kwargs))


def run_tasks(mteb, encoder, tasks, args):
    """Evaluate ``tasks`` and return the list of task results (mteb 1.x and 2.x)."""
    encode_kwargs = {"batch_size": args.batch_size}
    if hasattr(mteb, "evaluate"):  # mteb >= 2.0
        cache = mteb.ResultCache(args.output_folder) if args.output_folder else None
        result = mteb.evaluate(
            encoder, tasks, encode_kwargs=encode_kwargs, cache=cache,
            overwrite_strategy=args.overwrite, co2_tracker=False, raise_error=True,
        )
        return list(result.task_results)
    return list(mteb.MTEB(tasks=tasks).run(
        encoder, output_folder=args.output_folder, encode_kwargs=encode_kwargs))


def task_ndcg10(res) -> float:
    """nDCG@10 of one task result, averaged over its scored splits and subsets."""
    try:
        return float(res.get_score(getter=lambda s: s.get("ndcg_at_10", s["main_score"])))
    except Exception:
        values = [s.get("ndcg_at_10", s.get("main_score"))
                  for split in res.scores.values() for s in split]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    import mteb

    from nexterabert.simcse import contrastive_run

    encoder = load_st_model(args.model, args.max_len,
                            attn_implementation=args.attn_implementation)
    run = contrastive_run(args.model)

    # --tasks / --benchmark (+ --languages / --eval_splits), then mteb.evaluate on
    # mteb >= 2 (the legacy MTEB.run wrapper there rejects output_folder) or
    # MTEB.run on mteb 1.x.
    tasks = select_tasks(mteb, args)
    precision = contextlib.nullcontext()
    if args.st_dtype != "fp32":
        import torch

        if args.st_dtype == "tf32":
            torch.set_float32_matmul_precision("high")
        elif torch.cuda.is_available():
            # mteb encodes in this thread, so the autocast region covers every forward
            precision = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print(f"[encoder] sentence-transformers path computing in {args.st_dtype}")
    with precision:
        results = run_tasks(mteb, encoder, tasks, args)

    summary = {res.task_name: task_ndcg10(res) for res in results}
    if summary:
        summary["average_ndcg@10"] = float(np.mean(list(summary.values())))
    # Record the protocol alongside the scores: nDCG@10 after the full MS MARCO stage
    # and after the 250k-triplet one are not the same measurement.
    summary["_protocol"] = run.get("protocol") if run else "none"
    summary["_pooling"] = "sentence-transformers"
    summary["_max_len"] = args.max_len
    summary["_st_dtype"] = args.st_dtype
    summary["_batch_size"] = args.batch_size
    summary["_tasks"] = [t.metadata.name for t in tasks]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"results -> {args.output}")
    if not run:
        print("[warn] no contrastive_run.json in this directory: the stage that "
              "produced it is not recorded (see scripts/train_st_dpr.py)")


if __name__ == "__main__":
    main()
