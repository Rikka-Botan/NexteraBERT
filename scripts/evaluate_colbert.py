#!/usr/bin/env python
"""BEIR / MLDR evaluation of a PyLate ColBERT model trained by train_colbert.py.

The multi-vector half of ModernBERT's retrieval evaluation (Warner et al., 2024,
3.1.2 / 3.1.3): nDCG@10 over BEIR, and "Multi-Vector - Out-Of-Domain" MLDR, where
the BEIR checkpoint is scored on long documents "without any further fine-tuning".

Scoring goes through mteb's own PyLate wrapper (``MultiVectorModel``: PLAID index
+ MaxSim retrieval), so the task definitions, splits and metric are the same
objects ``scripts/evaluate_retrieval.py`` uses for the single-vector numbers.

    # BEIR (15 datasets; MSMARCO on dev, as in the original suite)
    python scripts/evaluate_colbert.py --model checkpoints/colbert/best \
        --benchmark BEIR --output beir_colbert.json
    # MLDR out-of-domain, English test split, documents read at the full context
    python scripts/evaluate_colbert.py --model checkpoints/colbert/best \
        --tasks MultiLongDocRetrieval --languages eng --eval_splits test \
        --document_length 8192 --output mldr_ood_colbert.json

Runs in the PyLate virtualenv that scripts/eval_colbert.sh sets up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate_retrieval import run_tasks, select_tasks, task_ndcg10  # noqa: E402
from nexterabert.mteb_encoder import build_model_meta  # noqa: E402

RUN_FILE = "colbert_run.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="PyLate ColBERT directory (train_colbert.py output) or Hub id")
    p.add_argument("--benchmark", default="BEIR",
                   help="an mteb benchmark name, or comma-separated task names")
    p.add_argument("--tasks", default="", help="comma-separated task names; overrides --benchmark")
    p.add_argument("--languages", default="", help="e.g. 'eng' for the English MLDR subset")
    p.add_argument("--eval_splits", default="")
    p.add_argument("--query_length", type=int, default=None,
                   help="override the trained model's query length")
    p.add_argument("--document_length", type=int, default=None,
                   help="override the trained document length (8192 for MLDR)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--nbits", type=int, default=4, help="PLAID residual compression")
    p.add_argument("--index_dir", default=None,
                   help="where PLAID indexes are built (default: a temp dir per task)")
    p.add_argument("--keep_index", action="store_true")
    p.add_argument("--output", default="colbert_results.json")
    p.add_argument("--output_folder", default="mteb_results_colbert",
                   help="mteb per-task result cache")
    p.add_argument("--cache_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", default="only-missing",
                   choices=["never", "only-missing", "only-cache", "always"])
    p.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.cache_dir:
        args.output_folder = args.cache_dir
    return args


def colbert_run(path) -> dict | None:
    marker = Path(path) / RUN_FILE
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_wrapper(args):
    from mteb.models.model_implementations.pylate_models import MultiVectorModel

    from nexterabert.modeling_nexterabert import ensure_dynamo_recompile_budget

    # the remote-code copy of NexteraBERT compiles FlexAttention; without a larger
    # dynamo recompile budget long-document batches fall back to the dense path
    ensure_dynamo_recompile_budget()

    kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.query_length is not None:
        kwargs["query_length"] = args.query_length
    if args.document_length is not None:
        kwargs["document_length"] = args.document_length
    if args.device:
        kwargs["device"] = args.device
    wrapper = MultiVectorModel(
        model_name=args.model,
        index_dir=args.index_dir,
        index_autodelete=not args.keep_index,
        index_kwargs={"nbits": args.nbits},
        **kwargs,
    )
    # mteb only sets model_prompts when the checkpoint carries prompts
    if not hasattr(wrapper, "model_prompts"):
        wrapper.model_prompts = {}
    colbert = wrapper.model
    from nexterabert.hf_baselines import repair_wrapped_neobert

    # PyLate, like sentence-transformers, opens the encoder with a plain AutoModel call.
    repair_wrapped_neobert(colbert, max(int(colbert.document_length or 0),
                                        int(colbert.query_length or 0)))
    wrapper.mteb_model_meta = build_model_meta(
        name=f"NexteraBERT-ColBERT/{Path(args.model).name}",
        embed_dim=None,
        n_parameters=sum(p.numel() for p in colbert.parameters()),
        max_tokens=float(colbert.document_length),
        similarity="max_sim",
    )
    print(f"[colbert] {args.model}: q_len={colbert.query_length} "
          f"d_len={colbert.document_length} PLAID nbits={args.nbits}")
    return wrapper


def main(argv=None):
    args = parse_args(argv)
    import mteb

    run = colbert_run(args.model)
    if run:
        print(f"[colbert] protocol={run.get('protocol')} lr={run.get('lr')} "
              f"({run.get('queries')} queries x {run.get('n_ways')} docs)")
    else:
        print("[warn] no colbert_run.json in the model directory: scoring a ColBERT "
              "model this script did not train (or a raw backbone with a random "
              "projection, which is not comparable to any published number).")

    wrapper = build_wrapper(args)
    tasks = select_tasks(mteb, args)
    start = time.time()
    results = run_tasks(mteb, wrapper, tasks, args)

    summary = {res.task_name: task_ndcg10(res) for res in results}
    if summary:
        summary["average_ndcg@10"] = float(np.mean(list(summary.values())))
    summary["_protocol"] = run.get("protocol") if run else "none (untrained ColBERT)"
    summary["_lr"] = run.get("lr") if run else None
    summary["_query_length"] = wrapper.model.query_length
    summary["_document_length"] = wrapper.model.document_length
    summary["_tasks"] = [t.metadata.name for t in tasks]
    summary["_wall_clock_seconds"] = time.time() - start
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
