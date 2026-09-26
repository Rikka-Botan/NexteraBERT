#!/usr/bin/env python
"""Full MTEB evaluation for a NexteraBERT embedding model.

Runs any `mteb` benchmark (default: the English v2 suite) over the L2-normalised
encoder embedding and reports the two aggregates the MTEB leaderboard uses:
Mean(Task) -- the mean over every task -- and Mean(TaskType) -- the mean of the
per-task-type means, which stops the task-heavy types (Classification, Retrieval)
from dominating the headline number. OptiBERT's tables call the latter "Avg".

Point --model at a directory produced by scripts/finetune_contrastive.py and the
trained attentive pooling head is used automatically (--pooling auto); a raw
backbone falls back to masked mean pooling, which is NOT the protocol published
MTEB numbers are measured under -- see that script's docstring. The protocol that
produced the checkpoint is reported in the summary.

    # English MTEB v2 (41 tasks, hours on one GPU)
    python scripts/finetune_contrastive.py --model checkpoints/discriminator         --output_dir checkpoints/simcse
    python scripts/evaluate_mteb.py --model checkpoints/simcse

    # A quick subset: two task types, 4 tasks, small model context
    python scripts/evaluate_mteb.py --model checkpoints/discriminator \
        --task_types STS,PairClassification --max_tasks 4

    # Named tasks only
    python scripts/evaluate_mteb.py --model checkpoints/discriminator \
        --tasks STS12,STSBenchmark,Banking77Classification

    # Baselines under the SAME protocol (see nexterabert.hf_baselines):
    python scripts/finetune_contrastive.py --hf_model answerdotai/ModernBERT-base         --output_dir checkpoints/baselines/ModernBERT-base/simcse
    python scripts/evaluate_mteb.py --model checkpoints/baselines/ModernBERT-base/simcse
    #   zero-shot mean pooling of a raw Hub checkpoint, the old reference path:
    python scripts/evaluate_mteb.py --hf_model chandar-lab/NeoBERT

Results are cached per task under --cache_dir, so an interrupted run resumes
where it stopped (the default --overwrite only-missing reruns missing splits
only). `scripts/evaluate_retrieval.py` is the retrieval-only shortcut; both share
the encoder in `nexterabert.mteb_encoder`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert.hf_baselines import is_hf_model_dir  # noqa: E402
from nexterabert.mteb_encoder import HFEncoder, NexteraEncoder  # noqa: E402

DEFAULT_BENCHMARK = "MTEB(eng, v2)"
DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   help="NexteraBERT backbone directory, or a directory written by "
                        "finetune_contrastive.py --hf_model (opened as an AutoModel)")
    p.add_argument("--hf_model",
                   help="score a Hugging Face AutoModel (answerdotai/ModernBERT-base, "
                        "chandar-lab/NeoBERT, bert-base-uncased, ...) through the "
                        "identical tokenisation/pooling/scoring path instead; a raw "
                        "Hub id is mean-pooled zero-shot -- the reference point for "
                        "judging whether a score is low")
    p.add_argument("--tokenizer", default=None,
                   help="default: ModernBERT's for --model, the baseline's own for --hf_model")
    p.add_argument("--pooling", default="auto", choices=["auto", "mean", "attentive"],
                   help="auto uses the SimCSE attentive head when the model dir has "
                        "one (scripts/finetune_contrastive.py), else masked mean pooling")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                   help="an mteb benchmark name, e.g. 'MTEB(eng, v2)' or 'NanoBEIR'")
    p.add_argument("--tasks", default="",
                   help="comma-separated task names; overrides --benchmark")
    p.add_argument("--exclude_tasks", default="",
                   help="comma-separated task names to omit, e.g. MindSmallReranking")
    p.add_argument("--task_types", default="",
                   help="comma-separated task types to keep, e.g. STS,Retrieval")
    p.add_argument("--languages", default="eng",
                   help="comma-separated ISO 639-3 codes used when --benchmark is unknown")
    p.add_argument("--max_tasks", type=int, default=0,
                   help="evaluate at most N tasks (0 = all); handy for smoke runs")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", default="mteb_results.json")
    p.add_argument("--cache_dir", default="mteb_results",
                   help="per-task result cache; reused to resume an interrupted run")
    p.add_argument("--embedding_cache_dir", default=None,
                   help="directory for disk-backed large embedding matrices; "
                        "default: <cache_dir>/_embedding_spill")
    p.add_argument("--embedding_memmap_threshold_mb", type=float, default=256,
                   help="spill an embedding matrix of at least this size to disk "
                        "instead of host RAM (0 disables spilling)")
    p.add_argument("--overwrite", default="only-missing",
                   choices=["only-missing", "always", "never", "only-cache"])
    p.add_argument("--num_proc", type=int, default=None,
                   help="processes used for dataset loading/transformation")
    p.add_argument("--skip_errors", action="store_true",
                   help="keep going when a task fails instead of raising")
    args = p.parse_args()
    if not args.model and not args.hf_model:
        p.error("pass --model (a NexteraBERT backbone) or --hf_model (a baseline)")
    return args


def build_encoder(args):
    embedding_cache_dir = args.embedding_cache_dir
    if embedding_cache_dir is None and args.cache_dir:
        embedding_cache_dir = str(Path(args.cache_dir) / "_embedding_spill")
    if args.hf_model or is_hf_model_dir(args.model):
        # a baseline uses its own tokenizer unless one was named explicitly; a
        # SimCSE'd baseline directory picks up its pooling head like --model does
        return HFEncoder(args.hf_model or args.model, args.tokenizer,
                         args.max_len, args.batch_size, pooling=args.pooling,
                         embedding_cache_dir=embedding_cache_dir,
                         embedding_memmap_threshold_mb=
                         args.embedding_memmap_threshold_mb)
    tokenizer = args.tokenizer or DEFAULT_TOKENIZER
    return NexteraEncoder(args.model, tokenizer, args.max_len, args.batch_size,
                          pooling=args.pooling,
                          embedding_cache_dir=embedding_cache_dir,
                          embedding_memmap_threshold_mb=
                          args.embedding_memmap_threshold_mb)


def select_tasks(mteb, args):
    """Resolve --tasks / --benchmark / --task_types into a list of mteb tasks."""
    if args.tasks:
        tasks = list(mteb.get_tasks(tasks=[t.strip() for t in args.tasks.split(",") if t.strip()]))
    else:
        try:
            tasks = list(mteb.get_benchmark(args.benchmark))
        except Exception:
            # not a benchmark name -- fall back to filtering the task registry
            langs = [x.strip() for x in args.languages.split(",") if x.strip()] or None
            tasks = list(mteb.get_tasks(languages=langs))
    if args.task_types:
        keep = {t.strip() for t in args.task_types.split(",") if t.strip()}
        tasks = [t for t in tasks if t.metadata.type in keep]
    if args.exclude_tasks:
        exclude = {t.strip() for t in args.exclude_tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t.metadata.name not in exclude]
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise SystemExit("no tasks selected -- check --benchmark / --tasks / --task_types")
    return tasks


def run_tasks(mteb, encoder, tasks, args):
    """Evaluate `tasks`, returning (task_results, {task_name: error_message})."""
    encode_kwargs = {"batch_size": args.batch_size}
    if hasattr(mteb, "evaluate"):  # mteb >= 2.0
        cache = mteb.ResultCache(args.cache_dir) if args.cache_dir else None
        result = mteb.evaluate(
            encoder,
            tasks,
            encode_kwargs=encode_kwargs,
            cache=cache,
            overwrite_strategy=args.overwrite,
            co2_tracker=False,
            raise_error=not args.skip_errors,
            num_proc=args.num_proc,
        )
        errors = {e.task_name: str(e.exception) for e in (result.exceptions or [])}
        return list(result.task_results), errors
    # mteb 1.x
    results = mteb.MTEB(tasks=tasks).run(
        encoder, output_folder=args.cache_dir, encode_kwargs=encode_kwargs,
        raise_error=not args.skip_errors,
    )
    return list(results), {}


def task_score(res) -> float:
    """Main score of one task, averaged over its splits/subsets like mteb does."""
    try:
        return float(res.get_score())
    except Exception:
        values = [s["main_score"] for split in res.scores.values() for s in split
                  if s.get("main_score") is not None]
        return float(np.mean(values)) if values else float("nan")


def summarise(results, errors, type_by_name, args, wall_clock, pooling="mean",
              protocol="none (raw backbone)"):
    per_task, per_type = {}, defaultdict(list)
    for res in results:
        # the selected task carries the type; res.task_type re-resolves it from
        # the registry, which is only needed for cached results we didn't select
        ttype = type_by_name.get(res.task_name) or getattr(res, "task_type", "Unknown")
        score = task_score(res)
        per_task[res.task_name] = {
            "score": score,
            "task_type": ttype,
            "seconds": getattr(res, "evaluation_time", None),
        }
        if not np.isnan(score):
            per_type[ttype].append(score)

    type_means = {t: float(np.mean(v)) for t, v in sorted(per_type.items())}
    all_scores = [v["score"] for v in per_task.values() if not np.isnan(v["score"])]
    return {
        "model": args.hf_model or args.model,
        "benchmark": args.tasks or args.benchmark,
        "max_len": args.max_len,
        "pooling": pooling,
        "protocol": protocol,
        "n_tasks": len(per_task),
        # the two headline aggregates: Mean(Task) weights every task equally,
        # Mean(TaskType) weights every task type equally
        "mean_task": float(np.mean(all_scores)) if all_scores else float("nan"),
        "mean_task_type": float(np.mean(list(type_means.values()))) if type_means else float("nan"),
        "task_type_scores": type_means,
        "tasks": per_task,
        "wall_clock_seconds": wall_clock,
        "errors": errors,
    }


def print_summary(summary):
    print()
    print(f"MTEB: {summary['benchmark']}  ({summary['n_tasks']} tasks, "
          f"{summary['wall_clock_seconds'] / 60:.1f} min)   scores x100")
    print("-" * 62)
    for name, info in sorted(summary["tasks"].items(),
                             key=lambda kv: (kv[1]["task_type"], kv[0])):
        seconds = info["seconds"]
        took = f"{seconds:8.1f}s" if isinstance(seconds, (int, float)) else " " * 9
        print(f"  {info['task_type']:<20} {name:<28} {info['score'] * 100:6.2f} {took}")
    print("-" * 62)
    for ttype, score in summary["task_type_scores"].items():
        print(f"  {ttype:<49} {score * 100:6.2f}")
    print("-" * 62)
    print(f"  {'Mean(TaskType)':<49} {summary['mean_task_type'] * 100:6.2f}")
    print(f"  {'Mean(Task)':<49} {summary['mean_task'] * 100:6.2f}")
    for name, err in summary["errors"].items():
        print(f"  [failed] {name}: {err}")


def main():
    args = parse_args()
    import mteb

    tasks = select_tasks(mteb, args)
    type_by_name = {t.metadata.name: t.metadata.type for t in tasks}
    print(f"[mteb {getattr(mteb, '__version__', '?')}] {len(tasks)} tasks: "
          f"{', '.join(sorted(type_by_name))[:200]}")

    encoder = build_encoder(args)

    start = time.time()
    results, errors = run_tasks(mteb, encoder, tasks, args)
    wall_clock = time.time() - start

    run = getattr(encoder, "contrastive_run", None)
    summary = summarise(results, errors, type_by_name, args, wall_clock,
                        pooling=getattr(encoder, "pooling", args.pooling),
                        protocol=run.get("protocol") if run else "none (raw backbone)")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"results -> {args.output}  (per-task cache: {args.cache_dir})")


if __name__ == "__main__":
    main()
