#!/usr/bin/env python
"""Collect the ModernBERT-style evaluation suite into one Table-1 row.

Reads the JSON files the scripts/eval_*.sh stages write into one results
directory and prints (and saves) the eight columns of Warner et al. (2024)
Table 1 next to the paper's BERT-base / ModernBERT-base rows:

    IR (DPR): BEIR, MLDR_OOD, MLDR_ID   IR (ColBERT): BEIR, MLDR_OOD
    NLU: GLUE                            Code: CSN, SQA

    python scripts/summarize_eval_suite.py eval_results/NexteraBERT-Mezzoforte-220M-en

A second mode picks the learning rate of a sweep, the way the paper does
(best mean nDCG@10 over NFCorpus / SciFact / TREC-COVID / FiQA):

    python scripts/summarize_eval_suite.py pick-lr 1e-5=select_1e-5.json 3e-5=select_3e-5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# results file -> (column, how to read the score)
FILES = {
    "beir_dpr.json": ("dpr_beir", "average_ndcg@10"),
    "mldr_ood_dpr.json": ("dpr_mldr_ood", "MultiLongDocRetrieval"),
    "mldr_id_dpr.json": ("dpr_mldr_id", "MultiLongDocRetrieval"),
    "beir_colbert.json": ("colbert_beir", "average_ndcg@10"),
    "mldr_ood_colbert.json": ("colbert_mldr_ood", "MultiLongDocRetrieval"),
    "glue_results.json": ("glue", "glue_avg"),
    "code_results.json": ("csn", "COIRCodeSearchNetRetrieval"),
    "code_results.json#sqa": ("sqa", "StackOverflowQA"),
}
COLUMNS = ["dpr_beir", "dpr_mldr_ood", "dpr_mldr_id", "colbert_beir",
           "colbert_mldr_ood", "glue", "csn", "sqa"]
HEADERS = ["BEIR", "MLDR_OOD", "MLDR_ID", "BEIR", "MLDR_OOD", "GLUE", "CSN", "SQA"]
# Contrastive stages whose single-vector numbers are comparable to the paper's:
# the sentence-transformers port of ModernBERT's own train_st.py, and its MLDR
# in-domain continuation.
PAPER_PROTOCOLS = {"st-msmarco", "st-mldr"}
# Warner et al. (2024), Table 1, base rows.
REFERENCE = {
    "BERT-base (Table 1)": [38.9, 23.9, 32.2, 49.0, 28.1, 84.7, 41.2, 59.5],
    "ModernBERT-base (Table 1)": [41.6, 27.4, 44.0, 51.3, 80.2, 88.4, 56.4, 73.6],
}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(results_dir: str | Path) -> dict:
    """{column: score x100 or None} plus the provenance of every file found."""
    results_dir = Path(results_dir)
    row = {c: None for c in COLUMNS}
    provenance = {}
    for spec, (column, key) in FILES.items():
        fname = spec.split("#")[0]
        data = _load(results_dir / fname)
        if not data or key not in data or data[key] is None:
            continue
        row[column] = round(float(data[key]) * 100.0, 2)
        provenance[column] = {
            "file": fname,
            "protocol": data.get("_protocol"),
            "max_len": data.get("_max_len", data.get("_document_length")),
            "n_tasks": len(data["_tasks"]) if isinstance(data.get("_tasks"), list) else None,
        }
    return {"scores": row, "provenance": provenance}


BEIR_N_TASKS = 15   # Table 7 / 8 average over the full suite (MSMARCO on dev)


def _fmt(v):
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  -  "


def render(summary: dict, model_name: str) -> str:
    lines = []
    lines.append("| Model | " + " | ".join(HEADERS) + " |")
    lines.append("|" + "---|" * (len(HEADERS) + 1))
    for name, vals in REFERENCE.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(v) for v in vals) + " |")
    row = summary["scores"]
    lines.append(f"| {model_name} | " + " | ".join(_fmt(row[c]) for c in COLUMNS) + " |")
    lines.append("")
    lines.append("columns: IR (DPR) = BEIR, MLDR_OOD, MLDR_ID; IR (ColBERT) = BEIR, "
                 "MLDR_OOD; NLU = GLUE; Code = CSN (CoIR CodeSearchNet), SQA "
                 "(StackOverflow-QA). nDCG@10 / GLUE dev average, x100.")
    missing = [c for c in COLUMNS if row[c] is None]
    if missing:
        lines.append("missing (stage not run or failed): " + ", ".join(missing))
    for col in ("dpr_beir", "colbert_beir"):
        n = summary["provenance"].get(col, {}).get("n_tasks")
        if n is not None and n < BEIR_N_TASKS:
            lines.append(f"NOTE: {col} averages a {n}/{BEIR_N_TASKS}-dataset BEIR subset, "
                         f"not the paper's 15-dataset average.")
    protocols = {c: p.get("protocol") for c, p in summary["provenance"].items()
                 if p.get("protocol")}
    if protocols:
        lines.append("contrastive stage per column: " +
                     ", ".join(f"{c}={p}" for c, p in protocols.items()))
        single_vector = [p for c, p in protocols.items()
                         if c.startswith("dpr") or c in ("csn", "sqa")]
        if any(p not in PAPER_PROTOCOLS for p in single_vector):
            lines.append("NOTE: the single-vector columns were NOT produced by ModernBERT's "
                         "sentence-transformers DPR stage (st-msmarco); they are comparable "
                         "to each other, not to the paper's IR (DPR) / Code numbers.")
    return "\n".join(lines)


def pick_lr(pairs: list[str]) -> tuple[str, dict]:
    """``lr=path`` pairs -> (best lr, {lr: mean nDCG@10 of the selection subset})."""
    scores = {}
    for pair in pairs:
        lr, _, path = pair.partition("=")
        data = _load(Path(path))
        if not data or "average_ndcg@10" not in data:
            continue
        scores[lr] = float(data["average_ndcg@10"])
    if not scores:
        raise SystemExit("pick-lr: no readable selection results among " + ", ".join(pairs))
    best = max(scores, key=scores.get)
    return best, scores


def pick_triplet(pairs: list[str]) -> tuple[str, dict]:
    """``lr=checkpoint dir`` pairs -> (best lr, {lr: held-out triplet accuracy}).

    Selection for the MLDR in-domain sweep, read from each checkpoint's
    contrastive_run.json (train_st_dpr.py records heldout_triplet_accuracy)."""
    scores = {}
    for pair in pairs:
        lr, _, path = pair.partition("=")
        data = _load(Path(path) / "contrastive_run.json")
        if not data or data.get("heldout_triplet_accuracy") is None:
            continue
        scores[lr] = float(data["heldout_triplet_accuracy"])
    if not scores:
        raise SystemExit("pick-triplet: no heldout_triplet_accuracy among " + ", ".join(pairs))
    best = max(scores, key=scores.get)
    return best, scores


def merge_results(parts: list[str | Path]) -> dict:
    """Union of per-task result files (evaluate_retrieval / evaluate_colbert output)
    into one file of the same shape; the average is recomputed over the union."""
    merged, tasks, meta = {}, [], {}
    for part in parts:
        data = _load(Path(part))
        if not data:
            raise SystemExit(f"merge: missing or unreadable part {part}")
        for key, value in data.items():
            if key == "_tasks":
                tasks.extend(t for t in value if t not in tasks)
            elif key.startswith("_"):
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    # per-task metadata (_metrics, _split_scores): union over the parts
                    meta[key] = {**meta[key], **value}
                else:
                    meta.setdefault(key, value)
            elif key != "average_ndcg@10":
                merged[key] = value
    scores = [v for v in merged.values() if isinstance(v, (int, float))]
    out = dict(merged)
    out["average_ndcg@10"] = float(sum(scores) / len(scores)) if scores else float("nan")
    out.update(meta)
    out["_tasks"] = tasks or list(merged)
    out["_merged_from"] = [str(p) for p in parts]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge":
        if len(argv) < 3:
            raise SystemExit("usage: summarize_eval_suite.py merge <out.json> <part.json>...")
        out = Path(argv[1])
        merged = merge_results(argv[2:])
        out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"merged {len(argv) - 2} parts -> {out}  "
              f"(avg nDCG@10 {merged['average_ndcg@10'] * 100:.2f} over {len(merged['_tasks'])} tasks)")
        return
    if argv and argv[0] == "pick-triplet":
        best, scores = pick_triplet(argv[1:])
        for lr, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  lr={lr:<8} held-out triplet accuracy {s * 100:6.2f}", file=sys.stderr)
        print(best)
        return
    if argv and argv[0] == "pick-lr":
        best, scores = pick_lr(argv[1:])
        for lr, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  lr={lr:<8} selection nDCG@10 {s * 100:6.2f}", file=sys.stderr)
        print(best)
        return
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--model_name", default=None)
    p.add_argument("--output", default=None,
                   help="summary JSON (default <results_dir>/summary.json)")
    args = p.parse_args(argv)
    results_dir = Path(args.results_dir)
    name = args.model_name or results_dir.name
    summary = collect(results_dir)
    summary["model"] = name
    text = render(summary, name)
    print(text)
    out = Path(args.output) if args.output else results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(f"summary -> {out} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
