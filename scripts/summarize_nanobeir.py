#!/usr/bin/env python
"""One NanoBEIR comparison table from several ``evaluate_retrieval.py`` result files.

    python scripts/summarize_nanobeir.py eval_results/*/nanobeir_dpr_lite.json
    python scripts/summarize_nanobeir.py eval_results/NexteraBERT-Mezzoforte-220M-en eval_results/ModernBERT-base \
        --output eval_results/nanobeir_dpr_lite_comparison.json

Prints (and with --output saves as JSON + Markdown) one row per file: the
contrastive stage the scored checkpoint went through, nDCG@10 x100 on each of the
13 NanoBEIR subsets and their mean. Rows whose recorded settings (protocol,
pooling, max_len, precision, batch size) differ are called out so the table never
hides it. A directory argument resolves to its ``nanobeir_dpr_lite.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# mteb's NanoBEIR benchmark, in BEIR's usual column order
TASKS = ["NanoMSMARCORetrieval", "NanoNFCorpusRetrieval", "NanoNQRetrieval",
         "NanoHotpotQARetrieval", "NanoFiQA2018Retrieval", "NanoArguAnaRetrieval",
         "NanoTouche2020Retrieval", "NanoQuoraRetrieval", "NanoDBPediaRetrieval",
         "NanoSCIDOCSRetrieval", "NanoFEVERRetrieval", "NanoClimateFeverRetrieval",
         "NanoSciFactRetrieval"]
SETTINGS = ["_protocol", "_pooling", "_max_len", "_st_dtype", "_batch_size"]
RESULT_NAMES = ("nanobeir_dpr_lite.json",)


def short(task: str) -> str:
    return task.removeprefix("Nano").removesuffix("Retrieval")


def _x100(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v * 100.0


def _fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def load_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = {t: _x100(data.get(t)) for t in TASKS}
    done = [v for v in scores.values() if v is not None]
    return {
        "file": str(path),
        "name": path.parent.name if path.parent.name not in ("", ".") else path.stem,
        "scores": scores,
        # recomputed over the known subsets, so a partial file is never averaged
        # against a full one without the n column showing it
        "average": sum(done) / len(done) if done else None,
        "n_tasks": len(done),
        **{k: data.get(k) for k in SETTINGS},
    }


def render(rows: list[dict]) -> str:
    header = ["Model", "Stage", *[short(t) for t in TASKS], "NanoBEIR", "n"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        cells = [r["name"], str(r["_protocol"] or "?"),
                 *[_fmt(r["scores"][t]) for t in TASKS],
                 _fmt(r["average"]), str(r["n_tasks"])]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ["nDCG@10 x100; NanoBEIR = mean over the n subsets scored."]
    for key in SETTINGS:
        values = {str(r[key]) for r in rows}
        if len(values) > 1:
            notes.append(f"rows differ in {key.lstrip('_')}: "
                         + ", ".join(f"{r['name']}={r[key]}" for r in rows)
                         + " -- not a like-for-like comparison.")
    for r in rows:
        if r["n_tasks"] < len(TASKS):
            missing = [short(t) for t in TASKS if r["scores"][t] is None]
            notes.append(f"{r['name']}: {len(missing)} subset(s) missing from its mean: "
                         + ", ".join(missing))
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+",
                   help="evaluate_retrieval.py NanoBEIR JSON files, or directories "
                        "containing nanobeir_dpr_lite.json")
    p.add_argument("--output", default=None, help="comparison JSON (+ .md beside it)")
    args = p.parse_args(argv)

    paths = []
    for item in args.results:
        path = Path(item)
        if path.is_dir():
            for name in RESULT_NAMES:
                if (path / name).is_file():
                    paths.append(path / name)
                    break
            else:
                print(f"[warn] no NanoBEIR results in {path}", file=sys.stderr)
        elif path.is_file():
            paths.append(path)
        else:
            print(f"[warn] not found: {path}", file=sys.stderr)
    if not paths:
        raise SystemExit("no results to summarise")

    rows = [load_row(path) for path in paths]
    rows.sort(key=lambda r: -(r["average"] if r["average"] is not None else -1))
    text = render(rows)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        print(f"comparison -> {out} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
