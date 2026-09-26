#!/usr/bin/env python
"""Merge per-task ``evaluate_retrieval.py`` result files into one.

``parallel_tasks`` in ``scripts/eval_common.sh`` scores one task per GPU and then
writes the combined file with this, the average recomputed over the union:

    python scripts/merge_retrieval_results.py beir_dpr.json beir_dpr.MSMARCO.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def merge_results(parts: list[str | Path]) -> dict:
    """Union of per-task result files into one file of the same shape; the average
    is recomputed over the union."""
    merged, tasks, meta = {}, [], {}
    for part in parts:
        data = _load(Path(part))
        if not data:
            raise SystemExit(f"merge: missing or unreadable part {part}")
        for key, value in data.items():
            if key == "_tasks":
                tasks.extend(t for t in value if t not in tasks)
            elif key.startswith("_"):
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
    if len(argv) < 2:
        raise SystemExit("usage: merge_retrieval_results.py <out.json> <part.json>...")
    out = Path(argv[0])
    merged = merge_results(argv[1:])
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"merged {len(argv) - 1} parts -> {out}  "
          f"(avg nDCG@10 {merged['average_ndcg@10'] * 100:.2f} over {len(merged['_tasks'])} tasks)")


if __name__ == "__main__":
    main()
