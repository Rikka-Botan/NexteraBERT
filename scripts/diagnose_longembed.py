#!/usr/bin/env python
"""Why do all models score alike on LongEmbed needle / passkey? Measure, don't guess.

The two synthetic LongEmbed tasks have a peculiar structure: at one context
length the 100 candidate documents share the SAME filler text and differ only in
one inserted sentence (the needle / the pass key). Two very different failure
modes -- and one non-failure -- then produce "every model gets the same number":

  saturated   every model finds every needle it can see: the task is solved and
              simply does not separate the models (under mean pooling the shared
              filler cancels out of the ranking, so this is likely);
  squashed    the contrastive stage (MS MARCO passages of ~80 tokens) left models
              that only use the head of a document: needles are found when they
              sit in the first few hundred tokens and missed beyond, although
              the tokens were read;
  truncated   the documents never reached the model at full length (a pipeline
              bug): same picture as "squashed", but also for the raw backbone.

They are told apart by WHERE the found needles are. For each query the needle is
located in its document (the sentence a sibling document does not have), mapped
to a token position, and accuracy@1 is reported per position bucket:

    [0,512)  [512,2048)  [2048,8192)   cut (needle beyond --max_len: unreachable)

Healthy long-context use = high accuracy in every bucket before "cut". Squashed /
truncated = high in [0,512), near chance (1%) after it. Run it on the DPR
checkpoint AND on the raw backbone of the same model: a drop that exists only
after fine-tuning is the stage's doing.

    # the fine-tuned checkpoint vs the raw backbone vs the no-transformer control
    python scripts/diagnose_longembed.py \
        --model static-mean eval_results/NexteraBERT-Mezzoforte-220M-en/dpr/lr8e-5 checkpoints/hub/RikkaBotan__NexteraBERT-Mezzoforte-220M-en
    # a baseline (its DPR checkpoint and the Hub backbone)
    python scripts/diagnose_longembed.py \
        --model eval_results/ModernBERT-base/dpr/lr8e-5 answerdotai/ModernBERT-base
    # does reading more tokens help at all? same checkpoint, 512 vs 8192
    python scripts/diagnose_longembed.py --model <dpr dir> --max_len 512 --output diag_512.json
    python scripts/diagnose_longembed.py --model <dpr dir> --max_len 8192 --output diag_8192.json

A raw backbone is scored with masked mean pooling (as NANOBEIR_MODE=zeroshot).
Cheap: 2 tasks x 8 lengths x (100 documents + 50 queries) per model.

``--model static-mean`` is the control that calibrates every other row: NO
transformer at all, just the mean of fixed random token vectors (ModernBERT
tokenizer), i.e. a model with zero contextualisation and zero long-context
ability. Measured 2026-09-21 at max_len 8192: passkey 100% at every length up to
8192, then 54% / 26% at 16384 / 32768 (exactly the share of pass keys that survive
truncation) = 85.0 on the 8-length mean; needle ~50% at EVERY length (47.8 mean).
So under mean pooling these two tasks are lexical matching -- the shared filler
cancels out of the ranking -- and a passkey score of ~85 says "reads 8192 tokens
without breaking", nothing more. Models can only differ here by breaking.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATASET = {"path": "dwzhu/LongEmbed", "revision": "6e346642246bfb4928c560ee08640dc84d074e8c"}
LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ssmax_drift", nargs=2, metavar=("RAW_DIR", "TUNED_DIR"),
                   help="no encoding: compare NexteraBERT's SSMax parameters (s, b of "
                        "scale = s*log(n) + base + relu(b)) before / after the contrastive "
                        "stage and print the attention temperature each implies at n = 80 "
                        "(an MS MARCO passage) and n = 8192")
    p.add_argument("--model", nargs="+", default=[],
                   help="one or more of: a sentence-transformers directory (train_st_dpr.py "
                        "output), a NexteraBERT backbone directory, a Hugging Face id / dir")
    p.add_argument("--tokenizer", default=None, help="NexteraBERT backbone directories only")
    p.add_argument("--tasks", default="needle,passkey")
    p.add_argument("--lengths", default=",".join(str(n) for n in LENGTHS),
                   help="context-length splits to score")
    p.add_argument("--max_len", type=int, default=8192, help="truncation, as LONGEMBED_MAX_LEN")
    p.add_argument("--position_edges", default="512,2048,8192",
                   help="token-position bucket edges; positions >= --max_len form the 'cut' bucket")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default="diag_longembed.json")
    return p.parse_args(argv)


def load_split(name: str, length: int):
    """(query texts, doc ids, doc texts, index of each query's relevant document)."""
    import datasets

    ds = datasets.load_dataset(name=name, **DATASET)
    queries = [r for r in ds["queries"] if r["context_length"] == length]
    corpus = [r for r in ds["corpus"] if r["context_length"] == length]
    rel = {r["qid"]: r["doc_id"] for r in ds["qrels"] if r["context_length"] == length}
    doc_ids = [r["doc_id"] for r in corpus]
    index = {d: i for i, d in enumerate(doc_ids)}
    return ([r["text"] for r in queries], doc_ids, [r["text"] for r in corpus],
            [index[rel[r["qid"]]] for r in queries])


def needle_char_start(doc: str, sibling: str) -> int:
    """Character offset of the first sentence ``doc`` has and ``sibling`` has not --
    the inserted needle, since the candidates of one length share their filler."""
    for sent in re.split(r"(?<=[.!?])\s+", doc):
        if sent and sent not in sibling:
            return doc.find(sent)
    return -1


def needle_token_positions(tokenizer, docs, targets):
    """Token position of the needle in each query's relevant document (-1 = not found)."""
    out = []
    for t in targets:
        start = needle_char_start(docs[t], docs[(t + 1) % len(docs)])
        if start < 0:
            out.append(-1)
            continue
        ids = tokenizer(docs[t][:start], add_special_tokens=False, truncation=False)["input_ids"]
        out.append(len(ids) + 1)   # + the leading special token
    return out


class StaticMeanEncoder:
    """The control: mean of fixed random token vectors. No transformer, no context."""

    def __init__(self, max_len: int, dim: int = 768):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        self.table = np.random.default_rng(0).standard_normal(
            (len(self.tokenizer), dim)).astype(np.float32)
        self.max_len = max_len

    def encode(self, texts, batch_size=None, **kwargs):
        out = []
        for text in texts:
            ids = self.tokenizer(text, truncation=True, max_length=self.max_len)["input_ids"]
            v = self.table[ids].mean(0)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def build(model: str, args):
    if model == "static-mean":
        return StaticMeanEncoder(args.max_len), "control: no transformer"
    from diagnose_retrieval import STAdapter

    from nexterabert.hf_baselines import is_hf_model_dir
    from nexterabert.mteb_encoder import HFEncoder, NexteraEncoder

    path = Path(model)
    if path.is_dir() and (path / "modules.json").exists():
        return STAdapter(model, args.max_len, args.batch_size), "sentence-transformers"
    if not path.is_dir() or is_hf_model_dir(model):
        return HFEncoder(model, None, args.max_len, args.batch_size, pooling="mean"), "raw, mean"
    return NexteraEncoder(model, args.tokenizer or DEFAULT_TOKENIZER, args.max_len,
                          args.batch_size, pooling="mean"), "raw, mean"


def bucket_label(pos: int, edges, max_len: int) -> str:
    if pos < 0:
        return "unlocated"
    if pos >= max_len:
        return "cut"
    lo = 0
    for hi in edges:
        if pos < hi:
            return f"[{lo},{hi})"
        lo = hi
    return f"[{lo},inf)"


def diagnose(encoder, args, edges):
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    report = {}
    for task in tasks:
        per_length, buckets = {}, {}
        for length in lengths:
            queries, _, docs, targets = load_split(task, length)
            positions = needle_token_positions(encoder.tokenizer, docs, targets)
            q = np.asarray(encoder.encode(queries, batch_size=args.batch_size), dtype=np.float32)
            d = np.asarray(encoder.encode(docs, batch_size=args.batch_size), dtype=np.float32)
            sims = q @ d.T
            hit = sims.argmax(axis=1) == np.asarray(targets)
            pos_sim = sims[np.arange(len(targets)), targets]
            masked = sims.copy()
            masked[np.arange(len(targets)), targets] = -np.inf
            margin = pos_sim - masked.max(axis=1)
            off = ~np.eye(len(d), dtype=bool)
            per_length[length] = {
                "accuracy": float(hit.mean()),
                "visible": float(np.mean([0 <= p < args.max_len for p in positions])),
                "margin": float(margin.mean()),
                # how much of a document embedding is the shared filler: -> 1.0 means the
                # needle moves the vector by almost nothing (mean-pool dilution)
                "doc_doc_cosine": float((d @ d.T)[off].mean()),
            }
            for p, h, m in zip(positions, hit, margin):
                b = buckets.setdefault(bucket_label(p, edges, args.max_len), {"hits": [], "margins": []})
                b["hits"].append(bool(h))
                b["margins"].append(float(m))
            print(f"  {task:8s} L={length:<6d} acc {hit.mean() * 100:5.1f}  visible "
                  f"{per_length[length]['visible'] * 100:5.1f}%  margin {margin.mean():+.4f}  "
                  f"doc-doc cos {per_length[length]['doc_doc_cosine']:.4f}", flush=True)
        report[task] = {
            "by_length": per_length,
            "by_position": {k: {"n": len(v["hits"]), "accuracy": float(np.mean(v["hits"])),
                                "margin": float(np.mean(v["margins"]))}
                            for k, v in buckets.items()},
        }
    return report


def verdict(by_position: dict, edges) -> str:
    head = by_position.get(f"[0,{edges[0]})")
    beyond = [v for k, v in by_position.items() if k.startswith("[") and not k.startswith("[0,")]
    n = sum(v["n"] for v in beyond)
    if not head or not n:
        return "not enough located needles to judge"
    far = sum(v["accuracy"] * v["n"] for v in beyond) / n
    if head["accuracy"] < 0.5:
        return f"fails even in the first {edges[0]} tokens ({head['accuracy'] * 100:.0f}%): not a length effect"
    if far >= 0.9 * head["accuracy"]:
        return (f"NO LENGTH EFFECT: needles past token {edges[0]} are found as often as early "
                f"ones ({far * 100:.0f}% vs {head['accuracy'] * 100:.0f}%) -- not squashed"
                + ("; the task is saturated and cannot separate models"
                   if head["accuracy"] >= 0.95 else
                   "; what it measures is query-needle matching, not length (compare static-mean)"))
    if far <= 0.25 * head["accuracy"]:
        return (f"HEAD-ONLY: {head['accuracy'] * 100:.0f}% in the first {edges[0]} tokens, "
                f"{far * 100:.0f}% beyond although the tokens were read -- squashed (or truncated: "
                f"check the raw backbone)")
    return (f"DECAYING: {head['accuracy'] * 100:.0f}% in the first {edges[0]} tokens, "
            f"{far * 100:.0f}% beyond")


def render(results: dict, edges, max_len: int) -> str:
    labels = [f"[{lo},{hi})" for lo, hi in zip([0, *edges], edges) if lo < max_len] + ["cut"]
    lines = []
    for task in next(iter(results.values()))["report"]:
        lines += [f"### {task}: accuracy@1 (x100) by needle token position, max_len {max_len}", "",
                  "| Model | " + " | ".join(labels) + " | verdict |",
                  "|---|" + "---|" * (len(labels) + 1)]
        for name, r in results.items():
            pos = r["report"][task]["by_position"]
            cells = [f"{pos[k]['accuracy'] * 100:5.1f} (n={pos[k]['n']})" if k in pos else "-"
                     for k in labels]
            lines.append(f"| {name} [{r['kind']}] | " + " | ".join(cells)
                         + f" | {verdict(pos, edges)} |")
        lines.append("")
    lines.append("- chance is 1.0 (1 of 100 candidates). 'cut' = the needle starts at or past "
                 "max_len, so no model can find it; it is what pulls the 16384 / 32768 splits down.")
    return "\n".join(lines)


def ssmax_drift(raw_dir: str, tuned_dir: str, base: float = 0.1) -> None:
    """The short-text stage only ever sees n ~ 10..300 keys, so s and b are fitted to
    that range alone -- but s is multiplied by log(n), which is ~2x larger at 8192
    than at 80. A drift that is harmless where it was trained can therefore move the
    long-context attention temperature a lot; this prints by how much, per layer."""
    import math

    from safetensors import safe_open

    def load(d):
        out = {}
        files = sorted(Path(d).glob("*.safetensors"))
        if not files:
            raise SystemExit(f"no .safetensors in {d}")
        for f in files:
            with safe_open(str(f), framework="np") as sf:
                for k in sf.keys():
                    if "scalable_factor" in k:
                        # key by layer path from the block index on: prefixes differ
                        # between a backbone export and a sentence-transformers save
                        m = re.search(r"(\d+(?:\.[A-Za-z_]+)*\.scalable_factor\.[sb])$", k)
                        out[m.group(1) if m else k] = float(sf.get_tensor(k))
        return out

    raw, tuned = load(raw_dir), load(tuned_dir)
    layers = sorted({k.rsplit(".", 1)[0] for k in raw if k in tuned},
                    key=lambda x: int(re.match(r"\d+", x).group()))
    if not layers:
        raise SystemExit("no shared scalable_factor parameters (ssmax_mode none, or not NexteraBERT)")

    def scale(w, layer, n):
        return w[layer + ".s"] * math.log(n) + base + max(w.get(layer + ".b", 0.0), 0.0)

    print("| layer | s raw -> tuned | b raw -> tuned | scale@80 raw -> tuned | scale@8192 raw -> tuned |")
    print("|---|---|---|---|---|")
    for layer in layers:
        r80, t80 = scale(raw, layer, 80), scale(tuned, layer, 80)
        r8k, t8k = scale(raw, layer, 8192), scale(tuned, layer, 8192)
        print(f"| {layer} | {raw[layer + '.s']:.4f} -> {tuned[layer + '.s']:.4f} | "
              f"{raw.get(layer + '.b', 0.0):+.4f} -> {tuned.get(layer + '.b', 0.0):+.4f} | "
              f"{r80:.3f} -> {t80:.3f} ({t80 - r80:+.3f}) | {r8k:.3f} -> {t8k:.3f} ({t8k - r8k:+.3f}) |")
    print()
    print("- scale multiplies the attention logits (assumes ssmax_base 0.1). The stage only "
          "sees n ~ 80, where it can trade s against b freely; at 8192 the same s shift moves the "
          "scale about twice as far (log 8192 / log 80 = 2.06) and a b shift does not compensate. "
          "Shifts of a few hundredths are noise; tenths mean the long-context attention now runs "
          "off its pretrained temperature.")


def main(argv=None):
    args = parse_args(argv)
    if args.ssmax_drift:
        ssmax_drift(*args.ssmax_drift)
        return
    if not args.model:
        raise SystemExit("pass --model (or --ssmax_drift RAW_DIR TUNED_DIR)")
    edges = sorted(int(x) for x in args.position_edges.split(",") if x.strip())
    results = {}
    for model in args.model:
        print(f"== {model} (max_len {args.max_len})", flush=True)
        encoder, kind = build(model, args)
        p = Path(model)
        name = f"{p.parent.parent.name}/{p.parent.name}/{p.name}" if p.is_dir() else model
        results[name] = {"kind": kind, "max_len": args.max_len,
                         "report": diagnose(encoder, args, edges)}
        del encoder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    text = render(results, edges, args.max_len)
    print("\n" + text)
    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(f"report -> {out} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
