#!/usr/bin/env python
"""One NanoBEIR comparison table from several ``evaluate_retrieval.py`` result files.

    python scripts/summarize_nanobeir.py eval_results/*/nanobeir_zeroshot.json
    python scripts/summarize_nanobeir.py eval_results/NexteraBERT-Mezzoforte-220M-en eval_results/ModernBERT-base \
        --output eval_results/nanobeir_zeroshot_comparison.json

Prints (and with --output saves as JSON + Markdown) one row per file: the stage
the scored model went through ("zero-shot" = the raw backbone, mean pooling -- the
default of scripts/eval_nanobeir.sh -- or the contrastive protocol of a DPR
checkpoint), nDCG@10 x100 on each of the 13 NanoBEIR subsets and their mean. A
DPR row also gets the full 15-dataset BEIR average its NanoBEIR mean is a proxy
of, when the same directory holds the ``beir_dpr.json`` of scripts/eval_dpr.sh.
Rows whose recorded settings (protocol, pooling, max_len, precision, batch size)
differ are called out so the table never hides it. A directory argument resolves
to its zero-shot file first, then the DPR one.
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
RESULT_NAMES = ("nanobeir_zeroshot.json", "nanobeir_dpr.json", "nanobeir_results.json")
RAW = "none (raw backbone)"   # evaluate_retrieval.py's _protocol without a contrastive stage


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
    row = {
        "file": str(path),
        "name": path.parent.name if path.parent.name not in ("", ".") else path.stem,
        "scores": scores,
        # recomputed over the known subsets, so a partial file is never averaged
        # against a full one without the n column showing it
        "average": sum(done) / len(done) if done else None,
        "n_tasks": len(done),
        "beir_average": None,
        "beir_n_tasks": None,
        **{k: data.get(k) for k in SETTINGS},
    }
    # beir_dpr.json scores the DPR checkpoint: it says nothing about a raw backbone
    beir = path.parent / "beir_dpr.json"
    if row["_protocol"] != RAW and beir.is_file():
        try:
            b = json.loads(beir.read_text(encoding="utf-8"))
            row["beir_average"] = _x100(b.get("average_ndcg@10"))
            row["beir_n_tasks"] = len(b["_tasks"]) if isinstance(b.get("_tasks"), list) else None
        except (OSError, ValueError):
            pass
    return row


def render(rows: list[dict]) -> str:
    # the BEIR column only exists for DPR rows; an all-zero-shot table drops it
    with_beir = any(r["_protocol"] != RAW for r in rows)
    header = ["Model", "Stage", *[short(t) for t in TASKS], "NanoBEIR", "n",
              *(["BEIR"] if with_beir else [])]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        beir = _fmt(r["beir_average"])
        if r["beir_average"] is not None and r["beir_n_tasks"] not in (None, 15):
            beir += f" ({r['beir_n_tasks']}/15)"
        stage = "zero-shot" if r["_protocol"] == RAW else str(r["_protocol"] or "?")
        cells = [r["name"], stage,
                 *[_fmt(r["scores"][t]) for t in TASKS],
                 _fmt(r["average"]), str(r["n_tasks"]), *([beir] if with_beir else [])]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ["nDCG@10 x100; NanoBEIR = mean over the n subsets scored."]
    if with_beir:
        notes.append("BEIR = that DPR checkpoint's beir_dpr.json average "
                     "(scripts/eval_dpr.sh), when present.")
    if any(r["_protocol"] == RAW for r in rows):
        notes.append("zero-shot = the raw pretrained backbone, masked mean pooling, no "
                     "contrastive stage: comparable across these rows, far below (and "
                     "not comparable to) BEIR numbers measured after MS MARCO training.")
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
                        "containing nanobeir_zeroshot.json / nanobeir_dpr.json")
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
