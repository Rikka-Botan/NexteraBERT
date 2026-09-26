#!/usr/bin/env python
"""Verify unpadded inference on a pretrained NexteraBERT: it is exact, and it is faster.

    python scripts/verify_unpadding.py --model RikkaBotan/NexteraBERT-Mezzoforte-220M-en \
        --compile --baseline answerdotai/ModernBERT-base --output eval_results/unpadding.json

Unpadding (``NexteraBERT.set_unpadding``) packs the real tokens of a padded batch into
one stream, so no GEMM, conv or attention is spent on pad slots. Both halves of the
claim are checked on real weights and real text (NanoBEIR-en queries and passages,
fetched from the Hub):

Correctness -- fp32 with TF32 off, eager. The reference for every text is the same
text encoded ALONE (B=1, no padding). Padded and unpadded batches of mixed lengths are
compared against it:
  * last hidden states (max |diff| over real tokens) and mean-pooled embeddings
    (min cosine),
  * masked LM on the same texts with ``--mask_ratio`` of the tokens masked: accuracy,
    top-1 agreement with the reference and max |diff| of the masked logits.
Unpadded rows must agree with the reference to float rounding. Padded rows need not:
NexteraHRA's token-level convs read the pad slots next to a row's last tokens (the
"HRA padding leak"), which unpadding removes.

Speed -- bf16 autocast under inference_mode. Texts are batched as they come
(``--batch_size``, each batch padded to its longest text, tokenised up front) and one
pass over all batches is timed, padded vs unpadded interleaved, median of
``--repeats``. ``--compile`` repeats it under ``torch.compile`` (default mode; warm-up
passes compile every shape and are not timed). ``--baseline`` adds an HF encoder run
exactly as transformers loads it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
for _stream in (sys.stdout, sys.stderr):
    _enc = (getattr(_stream, "encoding", None) or "").replace("-", "").replace("_", "")
    if _enc.lower() not in ("utf8", "utf8sig") and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
del _stream, _enc

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NANOBEIR = "sentence-transformers/NanoBEIR-en"
PASSAGE_SETS = ("NanoMSMARCO", "NanoNQ", "NanoSciFact")


def load_texts(per_set: int):
    """NanoBEIR-en: every query of the 13 sets, and ``per_set`` passages of each of
    ``PASSAGE_SETS`` (short web passages, Wikipedia paragraphs, paper abstracts)."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    patterns = ["queries/*.parquet"] + [f"corpus/{s}-*.parquet" for s in PASSAGE_SETS]
    root = Path(snapshot_download(NANOBEIR, repo_type="dataset", allow_patterns=patterns))
    queries = []
    for f in sorted(root.glob("queries/*.parquet")):
        queries += pq.read_table(f).column("text").to_pylist()
    passages = []
    for name in PASSAGE_SETS:
        for f in sorted(root.glob(f"corpus/{name}-*.parquet")):
            passages += pq.read_table(f).column("text").to_pylist()[:per_set]
    return [q for q in queries if q.strip()], [p for p in passages if p.strip()]


# --------------------------------------------------------------------------- correctness


def check_correctness(model, tok, texts, batch_size, max_len, mask_ratio, seed):
    """Every text alone (B=1) vs the same texts in padded / unpadded batches."""
    device = next(model.parameters()).device
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    enc = model.encoder
    g = torch.Generator().manual_seed(seed)
    specials = torch.tensor(sorted({tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}))
    paths = ("padded", "unpadded")
    stats = {p: dict(max_abs_hidden=0.0, min_cos_embedding=1.0, correct=0, agree=0,
                     max_abs_masked_logit=0.0) for p in paths}
    alone_correct = n_masked = n_tokens = 0
    try:
        for start in range(0, len(texts), batch_size):
            b = tok(texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            ids, am = b["input_ids"], b["attention_mask"]
            sel = (am.bool() & ~torch.isin(ids, specials)
                   & (torch.rand(ids.shape, generator=g) < mask_ratio))
            masked = ids.masked_fill(sel, tok.mask_token_id)
            ids, am, masked, sel = (t.to(device) for t in (ids, am, masked, sel))
            # rows of the (n_selected, V) logits that belong to each text
            row0 = F.pad(sel.sum(1).cumsum(0), (1, 0)).tolist()
            with torch.no_grad():
                out = {}
                for p in paths:
                    enc.set_unpadding(p == "unpadded")
                    hidden, _ = enc(ids, None, am)
                    out[p] = (hidden, model(masked, None, am, select_mask=sel))
                enc.set_unpadding(False)
                for i in range(ids.size(0)):
                    L = int(am[i].sum())                  # the tokenizer pads right
                    one = torch.ones(1, L, dtype=am.dtype, device=device)
                    ref_h = enc(ids[i:i + 1, :L], None, one)[0][0]
                    ref_logits = model(masked[i:i + 1, :L], None, one,
                                       select_mask=sel[i:i + 1, :L])
                    target = ids[i, :L][sel[i, :L]]
                    ref_pred = ref_logits.argmax(-1)
                    alone_correct += int((ref_pred == target).sum())
                    n_masked += target.numel()
                    n_tokens += L
                    for p in paths:
                        s = stats[p]
                        hidden, logits = out[p]
                        h = hidden[i, :L]
                        s["max_abs_hidden"] = max(s["max_abs_hidden"],
                                                  (h - ref_h).abs().max().item())
                        s["min_cos_embedding"] = min(s["min_cos_embedding"], F.cosine_similarity(
                            h.mean(0), ref_h.mean(0), dim=0).item())
                        mine = logits[row0[i]:row0[i + 1]]
                        pred = mine.argmax(-1)
                        s["correct"] += int((pred == target).sum())
                        s["agree"] += int((pred == ref_pred).sum())
                        if target.numel():
                            s["max_abs_masked_logit"] = max(
                                s["max_abs_masked_logit"], (mine - ref_logits).abs().max().item())
    finally:
        enc.set_unpadding(False)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])
    result = dict(texts=len(texts), tokens=n_tokens, masked_tokens=n_masked,
                  alone_mlm_accuracy=alone_correct / max(n_masked, 1))
    for p in paths:
        s = stats[p]
        result[p] = dict(max_abs_hidden=s["max_abs_hidden"],
                         min_cos_embedding=s["min_cos_embedding"],
                         mlm_accuracy=s["correct"] / max(n_masked, 1),
                         top1_agreement_with_alone=s["agree"] / max(n_masked, 1),
                         max_abs_masked_logit=s["max_abs_masked_logit"])
    return result


# --------------------------------------------------------------------------- speed


def tokenize_batches(tok, texts, batch_size, max_len, device):
    batches = []
    for start in range(0, len(texts), batch_size):
        b = tok(texts[start:start + batch_size], padding=True, truncation=True,
                max_length=max_len, return_tensors="pt")
        batches.append((b["input_ids"].to(device), b["attention_mask"].to(device)))
    return batches


def time_pass(run, batches) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ids, am in batches:
        run(ids, am)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(runners: dict, batches, repeats: int, warmup: int) -> dict:
    """Median seconds per pass of each runner; runners interleaved within a repeat."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for run in runners.values():
            for _ in range(warmup):
                time_pass(run, batches)
        times = {name: [] for name in runners}
        for _ in range(repeats):
            for name, run in runners.items():
                times[name].append(time_pass(run, batches))
    return {name: statistics.median(t) for name, t in times.items()}


def nextera_runners(fn, enc):
    def make(unpad):
        def run(ids, am):
            enc.unpadding = unpad
            fn(ids, None, am)
        return run
    return {"nextera padded": make(False), "nextera unpadded": make(True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="RikkaBotan/NexteraBERT-Mezzoforte-220M-en",
                    help="Hub id or local directory with config.json + model.safetensors "
                         "+ mlm_head.safetensors + the tokenizer")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--passages_per_set", type=int, default=700)
    ap.add_argument("--correctness_texts", type=int, default=128,
                    help="texts (half queries, half passages) for the B=1 comparison")
    ap.add_argument("--correctness_batch_size", type=int, default=16)
    ap.add_argument("--mask_ratio", type=float, default=0.15)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compile", action="store_true",
                    help="also time both paths under torch.compile")
    ap.add_argument("--baseline", default=None,
                    help="HF encoder to time alongside, e.g. answerdotai/ModernBERT-base")
    ap.add_argument("--skip_correctness", action="store_true")
    ap.add_argument("--skip_speed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="write all numbers here (JSON)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU (FlexAttention, bf16 timing)")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from nexterabert.loading import load_masked_lm

    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model)
    model, config, info = load_masked_lm(path)
    if info["missing"] or info["unexpected"] or not info["has_mlm_head"]:
        sys.exit(f"incomplete checkpoint: {info}")
    model = model.cuda().eval()
    enc = model.encoder
    tok = AutoTokenizer.from_pretrained(path)
    queries, passages = load_texts(args.passages_per_set)
    rng = random.Random(args.seed)
    report = dict(model=args.model, gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, batch_size=args.batch_size)
    print(f"model {args.model} on {report['gpu']} (torch {torch.__version__})")
    print(f"texts: {len(queries)} queries, {len(passages)} passages (NanoBEIR-en)")

    if not args.skip_correctness:
        half = args.correctness_texts // 2
        texts = rng.sample(queries, half) + rng.sample(passages, args.correctness_texts - half)
        rng.shuffle(texts)
        res = check_correctness(model, tok, texts, args.correctness_batch_size,
                                args.max_length, args.mask_ratio, args.seed)
        report["correctness"] = res
        print(f"\n== correctness: {res['texts']} texts in batches of "
              f"{args.correctness_batch_size}, {res['tokens']} tokens, {res['masked_tokens']} "
              f"masked; reference = each text alone (fp32, TF32 off)")
        print(f"   MLM accuracy alone: {res['alone_mlm_accuracy']:.4f}")
        print(f"   {'path':10s} {'max|dh|':>10s} {'min cos(emb)':>13s} {'max|dlogit|':>12s}"
              f" {'top1 agree':>11s} {'MLM acc':>8s}")
        for p in ("padded", "unpadded"):
            r = res[p]
            print(f"   {p:10s} {r['max_abs_hidden']:10.2e} {r['min_cos_embedding']:13.8f}"
                  f" {r['max_abs_masked_logit']:12.2e} {r['top1_agreement_with_alone']:11.4f}"
                  f" {r['mlm_accuracy']:8.4f}")

    if not args.skip_speed:
        mixed = queries + passages
        rng.shuffle(mixed)
        workloads = {"queries": queries, "passages": passages, "mixed": mixed}
        baseline = None
        if args.baseline:
            from transformers import AutoModel
            baseline = AutoModel.from_pretrained(args.baseline).cuda().eval()
            print(f"baseline {args.baseline}: attn_implementation="
                  f"{baseline.config._attn_implementation}")
        report["speed"] = {}
        modes = ["eager"] + (["compile"] if args.compile else [])
        for mode in modes:
            fn = enc if mode == "eager" else torch.compile(enc)
            runners = nextera_runners(fn, enc)
            if baseline is not None:
                bfn = baseline if mode == "eager" else torch.compile(baseline)
                runners["baseline padded"] = lambda ids, am, bfn=bfn: bfn(
                    input_ids=ids, attention_mask=am)
            print(f"\n== speed, {mode}: bf16 autocast, batch {args.batch_size}, "
                  f"median of {args.repeats} passes")
            print(f"   {'workload':9s} {'texts':>6s} {'real/slots':>10s}  " + "  ".join(
                f"{n:>18s}" for n in runners) + "   unpad speedup")
            for wl, texts in workloads.items():
                batches = tokenize_batches(tok, texts, args.batch_size, args.max_length, "cuda")
                real = sum(int(am.sum()) for _, am in batches)
                slots = sum(am.numel() for _, am in batches)
                t = bench(runners, batches, args.repeats, warmup=1 if mode == "eager" else 2)
                speedup = t["nextera padded"] / t["nextera unpadded"]
                report["speed"].setdefault(mode, {})[wl] = dict(
                    texts=len(texts), real_tokens=real, padded_slots=slots,
                    seconds=t, real_tokens_per_s={k: real / v for k, v in t.items()},
                    unpadded_speedup=speedup)
                print(f"   {wl:9s} {len(texts):6d} {real / slots:10.2f}  " + "  ".join(
                    f"{v * 1e3:12.1f} ms   " for v in t.values()) + f"   x{speedup:.2f}")
        enc.set_unpadding(False)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
