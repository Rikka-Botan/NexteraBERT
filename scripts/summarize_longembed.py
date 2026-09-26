#!/usr/bin/env python
"""One LongEmbed comparison table from several ``evaluate_retrieval.py`` result files.

    python scripts/summarize_longembed.py eval_results/*/longembed_dpr.json
    python scripts/summarize_longembed.py eval_results/NexteraBERT-Mezzoforte-220M-en eval_results/ModernBERT-base \
        --output eval_results/longembed_comparison.json

Prints (and with --output saves as JSON + Markdown) two tables:

1. one row per file -- the contrastive stage the checkpoint went through, the four
   real tasks (nDCG@10 x100), needle and passkey (nDCG@1 = accuracy x100, mean over
   the 8 context lengths 256..32768) and the LongEmbed average of the six, which is
   the number of the paper (Zhu et al., 2024) and of the mteb leaderboard. When the
   same directory holds the ``mldr_ood_dpr.json`` of scripts/eval_dpr.sh, the MLDR
   out-of-domain score of that checkpoint is shown beside it.
2. needle / passkey accuracy per context length. A model read at max_len tokens
   only sees the first max_len tokens of a longer document, so the splits past
   max_len sit near chance whatever the model; "<=len" is the mean over the lengths
   that fit and separates "cannot use its context" from "context too short".

Rows whose recorded settings (protocol, max_len, precision, batch size, metric)
differ are called out so the table never hides it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# mteb's LongEmbed benchmark, in the paper's column order
REAL_TASKS = ["LEMBNarrativeQARetrieval", "LEMBQMSumRetrieval",
              "LEMBSummScreenFDRetrieval", "LEMBWikimQARetrieval"]
SYNTHETIC_TASKS = ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]
TASKS = REAL_TASKS + SYNTHETIC_TASKS
LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
SETTINGS = ["_protocol", "_pooling", "_max_len", "_st_dtype", "_batch_size", "_metric"]
# scripts/diagnose_longembed.py --model static-mean at max_len 8192: the mean of RANDOM
# token vectors, no transformer. At one length the candidates share their filler, which
# cancels out of a mean-pooled ranking, so this context-free control already scores:
STATIC_MEAN_8192 = {
    "LEMBNeedleRetrieval": [60, 48, 54, 50, 46, 60, 40, 24],
    "LEMBPasskeyRetrieval": [100, 100, 100, 100, 100, 100, 54, 26],
}
SHORT_MAX = 1024     # "short" lengths of the trend columns: 256..1024
# a directory argument: the full run first, else a LONGEMBED_SUBSET run
RESULT_NAMES = ("longembed_dpr.json", "longembed_results.json",
                "longembed_synthetic_dpr.json", "longembed_real_dpr.json")


def short(task: str) -> str:
    return task.removeprefix("LEMB").removesuffix("Retrieval")


def _x100(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v * 100.0


def _fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = {t: _x100(data.get(t)) for t in TASKS}
    done = [v for v in scores.values() if v is not None]
    splits = data.get("_split_scores") or {}
    by_length = {t: {n: _x100((splits.get(t) or {}).get(f"test_{n}")) for n in LENGTHS}
                 for t in SYNTHETIC_TASKS}
    max_len = data.get("_max_len")
    fits = [n for n in LENGTHS if isinstance(max_len, int) and n <= max_len]
    # the model directory names the row; longembed_dpr_len<N>.json (a run at another
    # LONGEMBED_MAX_LEN) keeps its suffix so two lengths of one model stay apart
    name = path.parent.name if path.parent.name not in ("", ".") else path.stem
    tag = [t for t in path.stem.split("_")
           if t not in ("longembed", "dpr", "results", "synthetic", "real")]
    if tag:
        name += f" [{'_'.join(tag)}]"
    row = {
        "file": str(path),
        "name": name,
        "scores": scores,
        # recomputed over the known tasks, so a partial file is never averaged
        # against a full one without the n column showing it
        "average": sum(done) / len(done) if done else None,
        "real_average": _mean(scores[t] for t in REAL_TASKS),
        "synthetic_average": _mean(scores[t] for t in SYNTHETIC_TASKS),
        "n_tasks": len(done),
        "by_length": by_length,
        "within_max_len": {t: _mean(by_length[t][n] for n in fits) for t in SYNTHETIC_TASKS},
        "mldr_ood": None,
        **{k: data.get(k) for k in SETTINGS},
    }
    mldr = path.parent / "mldr_ood_dpr.json"
    if mldr.is_file():
        try:
            row["mldr_ood"] = _x100(json.loads(mldr.read_text(encoding="utf-8")).get("average_ndcg@10"))
        except (OSError, ValueError):
            pass
    return row


def _table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def render(rows: list[dict]) -> str:
    # a LONGEMBED_SUBSET run (needle + passkey only, ...) shows just its own columns
    shown = [t for t in TASKS if any(r["scores"][t] is not None for r in rows)] or TASKS
    halves = [(label, key) for label, key, group in (("Real", "real_average", REAL_TASKS),
                                                     ("Synth", "synthetic_average", SYNTHETIC_TASKS))
              if any(t in shown for t in group)]
    if len(halves) < 2:
        halves = []          # one half only: it equals the mean column
    mean_label = "LongEmbed" if len(shown) == len(TASKS) else "Mean"
    main = _table(
        ["Model", "Stage", "max_len", *[short(t) for t in shown], *[h[0] for h in halves],
         mean_label, "n", "MLDR_OOD"],
        [[r["name"], str(r["_protocol"] or "?"), str(r["_max_len"] or "?"),
          *[_fmt(r["scores"][t]) for t in shown], *[_fmt(r[h[1]]) for h in halves],
          _fmt(r["average"]), str(r["n_tasks"]), _fmt(r["mldr_ood"])] for r in rows])

    def trend(by_length, max_len):
        """(mean over 256..1024, mean over 2048..max_len, their difference)"""
        near = _mean(by_length[n] for n in LENGTHS if n <= SHORT_MAX)
        far = _mean(by_length[n] for n in LENGTHS
                    if n > SHORT_MAX and isinstance(max_len, int) and n <= max_len)
        return near, far, (far - near if near is not None and far is not None else None)

    body = []
    for t in SYNTHETIC_TASKS:
        scored = [r for r in rows if any(v is not None for v in r["by_length"][t].values())]
        for r in scored:
            near, far, delta = trend(r["by_length"][t], r["_max_len"])
            body.append([r["name"], short(t), *[_fmt(r["by_length"][t][n]) for n in LENGTHS],
                         _fmt(near), _fmt(far), "  -  " if delta is None else f"{delta:+5.1f}",
                         _fmt(r["within_max_len"][t]), _fmt(r["scores"][t])])
        if scored and all(r["_max_len"] == 8192 for r in scored):
            ref = dict(zip(LENGTHS, map(float, STATIC_MEAN_8192[t])))
            near, far, delta = trend(ref, 8192)
            body.append(["*static-mean control*", short(t), *[_fmt(ref[n]) for n in LENGTHS],
                         _fmt(near), _fmt(far), f"{delta:+5.1f}",
                         _fmt(_mean(ref[n] for n in LENGTHS if n <= 8192)), _fmt(_mean(ref.values()))])
    curve = _table(["Model", "Task", *[str(n) for n in LENGTHS], f"<={SHORT_MAX}", "long", "trend",
                    "<=len", "all 8"], body) if body else ""

    notes = ["NarrativeQA / QMSum / SummScreenFD / WikimQA: nDCG@10 x100. Needle / Passkey: "
             "nDCG@1 (accuracy) x100, mean over the 8 context lengths 256..32768. LongEmbed = "
             "mean over the n tasks scored (the paper's average when n = 6); Real / Synth = its "
             "two halves. MLDR_OOD = that checkpoint's mldr_ood_dpr.json (scripts/eval_dpr.sh), "
             "when present.",
             *([f"only {', '.join(short(t) for t in shown)} scored (LONGEMBED_SUBSET / "
                "LONGEMBED_TASKS): 'Mean' is NOT the 6-task LongEmbed average."]
               if len(shown) < len(TASKS) else []),
             *([f"per-length table: '<={SHORT_MAX}' / 'long' = mean over 256..{SHORT_MAX} / over "
                f"{SHORT_MAX * 2}..max_len, 'trend' = long - short (about 0: accuracy does not depend "
                "on length; strongly negative: the model loses what lies deep in a long document). "
                "50 queries per cell = +-6 points of sampling noise per cell, about +-4 on 'trend'. "
                "'static-mean control' = mean of RANDOM token vectors, no transformer "
                "(scripts/diagnose_longembed.py): the candidates of one length share their filler, "
                "so a context-free model already solves passkey -- a row at the control's level "
                "reads 8192 tokens without breaking, and only rows BELOW it say something."]
               if body else []),
             "documents longer than max_len are truncated: on the needle / passkey splits past "
             "max_len only the targets that fall inside the first max_len tokens can be found "
             "(chance is 1 of 100 candidates), whatever the model; '<=len' averages only the "
             "lengths that fit."]
    for key in SETTINGS:
        values = {str(r[key]) for r in rows}
        if len(values) > 1:
            notes.append(f"rows differ in {key.lstrip('_')}: "
                         + ", ".join(f"{r['name']}={r[key]}" for r in rows)
                         + " -- not a like-for-like comparison.")
    for r in rows:
        if r["n_tasks"] < len(shown):
            missing = [short(t) for t in shown if r["scores"][t] is None]
            notes.append(f"{r['name']}: {len(missing)} task(s) missing from its mean: "
                         + ", ".join(missing))
        if r["_metric"] != "main_score":
            notes.append(f"{r['name']}: scored with --metric {r['_metric'] or 'ndcg_at_10'}; "
                         "needle / passkey are then nDCG@10, not the paper's accuracy.")
        if r["_protocol"] == "none (raw backbone)":
            notes.append(f"{r['name']}: scored WITHOUT a contrastive stage (raw backbone).")
    parts = [main] + ([curve] if curve else []) + ["\n".join(f"- {n}" for n in notes)]
    return "\n\n".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+",
                   help="evaluate_retrieval.py LongEmbed JSON files, or directories "
                        "containing longembed_dpr.json")
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
                print(f"[warn] no LongEmbed results in {path}", file=sys.stderr)
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
