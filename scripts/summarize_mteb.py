#!/usr/bin/env python
"""One MTEB comparison table from several ``evaluate_mteb.py`` result files.

    python scripts/summarize_mteb.py eval_results/*/mteb_simcse.json
    python scripts/summarize_mteb.py eval_results/NexteraBERT-Mezzoforte-220M-en/mteb_simcse.json \
        eval_results/ModernBERT-base/mteb_simcse.json eval_results/NeoBERT/mteb_simcse.json \
        --output eval_results/mteb_comparison.json

Prints (and saves as JSON + Markdown) one row per file: the protocol the model
went through, the seven task-type means, and the two leaderboard aggregates --
Mean(TaskType), which OptiBERT's tables call Avg, and Mean(Task). Reference rows
are OptiBERT's Table 5 (MTEB(eng, v2), the same protocol scripts/eval_mteb.sh
runs). ModernBERT's and NeoBERT's *paper* MTEB numbers are deliberately not
listed: they are MTEB(eng, v1) after each paper's own, heavier contrastive
pipeline, so the like-for-like comparison is the rows this script prints.

Rows measured under different protocols (a zero-shot mean-pool run next to a
SimCSE run) are printed with their protocol so the table never hides it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TASK_TYPES = ["Classification", "Clustering", "PairClassification", "Reranking",
              "Retrieval", "STS", "Summarization"]
SHORT = {"Classification": "Class", "Clustering": "Clust", "PairClassification": "PairCls",
         "Reranking": "Rerank", "Retrieval": "Retr", "STS": "STS", "Summarization": "Summ"}
# Dervishi et al. (EMNLP 2025), Table 5: MTEB(eng, v2) after attentive pooling +
# supervised SimCSE -- the protocol scripts/eval_mteb.sh reproduces.
REFERENCE = {
    "OptiBERTneo 198M / 13B tok (Table 5)": {
        "Classification": 70.1, "Clustering": 39.1, "PairClassification": 74.7,
        "Reranking": 41.0, "Retrieval": 24.4, "STS": 77.8, "Summarization": 26.2,
        "mean_task_type": 50.5,
    },
}


def load_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    types = data.get("task_type_scores", {})
    row = {
        "file": str(path),
        "model": data.get("model", path.parent.name),
        "name": path.parent.name if path.parent.name not in ("", ".") else path.stem,
        "protocol": data.get("protocol", "?"),
        "pooling": data.get("pooling", "?"),
        "n_tasks": data.get("n_tasks"),
        "max_len": data.get("max_len"),
        "mean_task_type": _x100(data.get("mean_task_type")),
        "mean_task": _x100(data.get("mean_task")),
        "errors": sorted((data.get("errors") or {}).keys()),
    }
    for t in TASK_TYPES:
        row[t] = _x100(types.get(t))
    return row


def _x100(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v:  # nan
        return None
    return v * 100.0


def _fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def render(rows: list[dict], reference: bool = True) -> str:
    header = ["Model", "Protocol", *[SHORT[t] for t in TASK_TYPES], "Avg=Mean(Type)", "Mean(Task)", "n"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        proto = r["protocol"] if r["protocol"] != "none (raw backbone)" else "zero-shot mean"
        cells = [r["name"], proto, *[_fmt(r[t]) for t in TASK_TYPES],
                 _fmt(r["mean_task_type"]), _fmt(r["mean_task"]), str(r["n_tasks"] or "-")]
        lines.append("| " + " | ".join(cells) + " |")
    if reference:
        for name, ref in REFERENCE.items():
            cells = [name, "mteb-nli (paper)", *[_fmt(ref.get(t)) for t in TASK_TYPES],
                     _fmt(ref.get("mean_task_type")), "  -  ", "41"]
            lines.append("| " + " | ".join(cells) + " |")
    notes = []
    for r in rows:
        if r["errors"]:
            notes.append(f"{r['name']}: {len(r['errors'])} task(s) failed and are missing "
                         f"from its means: {', '.join(r['errors'][:6])}"
                         + (" ..." if len(r["errors"]) > 6 else ""))
    protocols = {r["protocol"] for r in rows}
    if len(protocols) > 1:
        notes.append("rows were measured under DIFFERENT protocols -- compare within a "
                     "protocol only (scores x100).")
    text = "\n".join(lines)
    if notes:
        text += "\n\n" + "\n".join(f"- {n}" for n in notes)
    return text


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+",
                   help="evaluate_mteb.py JSON files, or directories containing mteb_simcse.json")
    p.add_argument("--output", default=None, help="comparison JSON (+ .md beside it)")
    p.add_argument("--no_reference", action="store_true", help="omit the OptiBERT Table 5 row")
    args = p.parse_args(argv)

    paths = []
    for item in args.results:
        path = Path(item)
        if path.is_dir():
            for name in ("mteb_simcse.json", "mteb_results.json", "mteb_zeroshot.json"):
                if (path / name).is_file():
                    paths.append(path / name)
                    break
            else:
                print(f"[warn] no MTEB results in {path}", file=sys.stderr)
        elif path.is_file():
            paths.append(path)
        else:
            print(f"[warn] not found: {path}", file=sys.stderr)
    if not paths:
        raise SystemExit("no results to summarise")

    rows = [load_row(path) for path in paths]
    rows.sort(key=lambda r: -(r["mean_task_type"] if r["mean_task_type"] is not None else -1))
    text = render(rows, reference=not args.no_reference)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "reference": REFERENCE}, indent=2),
                       encoding="utf-8")
        out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        print(f"comparison -> {out} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
