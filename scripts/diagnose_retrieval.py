#!/usr/bin/env python
"""Why does a retrieval model score well on a small corpus and collapse on a big one?

Scores one BEIR-style task end to end (same encoder, pooling, truncation and
cosine scoring as scripts/evaluate_retrieval.py) and then breaks the result down
along the axes that separate "the embedding space is weak at scale" from "the
evaluation is doing something to this model":

  * ranking quality      nDCG@10 / MRR@10 / Recall@100, plus the score margin
                         between a query's positives and random documents, and how
                         often a random document outscores the best positive.
  * hubness              how many distinct documents fill all the top-10 lists, the
                         most frequent ones (with their token lengths). A handful
                         of documents that are close to every query is the classic
                         signature of an anisotropic space; it costs little on a
                         small corpus and everything on a large one.
  * document length      mean cosine to random queries and share of top-10 hits per
                         length bucket, against each bucket's share of the corpus.
                         Mean pooling over long / truncated documents drifting
                         towards a "mean direction" shows up here.
  * padding sensitivity  queries encoded one at a time vs. in length-sorted
                         batches vs. padded to a fixed length. NexteraHRA runs a
                         conv before the padding mask (see mteb_encoder.py), so
                         this isolates that leak: cosines near 1.0 and identical
                         nDCG mean it is not the problem.
  * truncation sweep     optional: re-encode the corpus at several max_len values.

    python scripts/diagnose_retrieval.py --model checkpoints/dpr --task SCIDOCS
    python scripts/diagnose_retrieval.py --model checkpoints/dpr --task FiQA2018 \
        --doc_len_sweep 128,256,512 --output diag_fiqa.json
    # the same for a reference model through the identical path
    python scripts/diagnose_retrieval.py --hf_model bert-base-uncased --task SCIDOCS

Pick a task that collapsed but is small enough to re-encode a few times
(SCIDOCS: 25k docs / 1k queries; FiQA2018: 57k / 648; NFCorpus as a control).

``--model`` also accepts a sentence-transformers directory (scripts/train_st_dpr.py
output). With ``--split dev --max_docs N`` this doubles as a cheap, test-free
selection signal for the MLDR in-domain stage: the 200 MLDR dev queries against
their positives plus N random documents, e.g.

    python scripts/diagnose_retrieval.py --model eval_results/<m>/dpr/mldr_id_lr2e-5 \
        --task MultiLongDocRetrieval --languages eng --split dev --max_len 8192 \
        --max_docs 20000 --padding_queries 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="NexteraBERT model directory (e.g. the DPR checkpoint)")
    p.add_argument("--hf_model", help="a Hugging Face AutoModel instead, through the same path")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--pooling", default="auto", choices=["auto", "mean", "attentive"])
    p.add_argument("--task", default="SCIDOCS", help="an mteb retrieval task name")
    p.add_argument("--split", default="test")
    p.add_argument("--languages", default="", help="e.g. eng for multilingual tasks")
    p.add_argument("--max_len", type=int, default=512, help="document truncation")
    p.add_argument("--query_max_len", type=int, default=None, help="default: --max_len")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_docs", type=int, default=0,
                   help="subsample the corpus to N documents (positives kept); 0 = all")
    p.add_argument("--max_queries", type=int, default=0, help="subsample queries; 0 = all")
    p.add_argument("--padding_queries", type=int, default=256,
                   help="queries used for the padding-sensitivity check (0 = skip)")
    p.add_argument("--padded_len", type=int, default=64,
                   help="fixed length the padding check pads queries to")
    p.add_argument("--doc_len_sweep", default="",
                   help="comma-separated max_len values to re-encode the corpus at")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="JSON report (default diag_<task>.json)")
    args = p.parse_args(argv)
    if not args.model and not args.hf_model:
        p.error("pass --model or --hf_model")
    return args


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_task_data(name: str, split: str, languages: str = ""):
    """(doc_ids, doc_texts, query_ids, query_texts, qrels) of one mteb retrieval task."""
    import mteb

    kwargs = {"eval_splits": [split]}
    if languages:
        kwargs["languages"] = [x.strip() for x in languages.split(",") if x.strip()]
    task = mteb.get_task(name, **kwargs)
    task.load_data()
    subset = next(iter(task.dataset))
    data = task.dataset[subset][split]
    corpus, queries, qrels = data["corpus"], data["queries"], data["relevant_docs"]

    def doc_text(row):
        title = row.get("title") or ""
        body = row.get("text") if row.get("text") is not None else row.get("body", "")
        return (title + " " + body).strip() if title else (body or "").strip()

    doc_ids = [str(i) for i in corpus["id"]]
    doc_texts = [doc_text(row) for row in corpus]
    query_ids = [str(i) for i in queries["id"]]
    query_texts = [row["text"] for row in queries]
    qrels = {str(q): {str(d): int(s) for d, s in rel.items() if int(s) > 0}
             for q, rel in qrels.items()}
    print(f"[data] {name}/{subset}/{split}: {len(doc_ids)} documents, "
          f"{len(query_ids)} queries, {sum(len(v) for v in qrels.values())} qrels")
    return doc_ids, doc_texts, query_ids, query_texts, qrels


def subsample(doc_ids, doc_texts, query_ids, query_texts, qrels, max_docs, max_queries, seed):
    rng = random.Random(seed)
    if max_queries and max_queries < len(query_ids):
        keep = sorted(rng.sample(range(len(query_ids)), max_queries))
        query_ids = [query_ids[i] for i in keep]
        query_texts = [query_texts[i] for i in keep]
    qrels = {q: qrels.get(q, {}) for q in query_ids}
    if max_docs and max_docs < len(doc_ids):
        positives = {d for rel in qrels.values() for d in rel}
        pos_idx = [i for i, d in enumerate(doc_ids) if d in positives]
        rest = [i for i, d in enumerate(doc_ids) if d not in positives]
        rng.shuffle(rest)
        keep = sorted(pos_idx + rest[:max(0, max_docs - len(pos_idx))])
        doc_ids = [doc_ids[i] for i in keep]
        doc_texts = [doc_texts[i] for i in keep]
    return doc_ids, doc_texts, query_ids, query_texts, qrels


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def topk_search(q: np.ndarray, d: np.ndarray, k: int, chunk: int = 50_000):
    """Cosine top-k of every query over the corpus, chunked like mteb's search."""
    import torch

    qt = torch.nn.functional.normalize(torch.as_tensor(q).float(), dim=-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qt = qt.to(dev)
    best_scores = torch.full((qt.shape[0], 0), -2.0, device=dev)
    best_idx = torch.zeros((qt.shape[0], 0), dtype=torch.long, device=dev)
    for start in range(0, d.shape[0], chunk):
        block = torch.nn.functional.normalize(
            torch.as_tensor(d[start:start + chunk]).float(), dim=-1).to(dev)
        scores = qt @ block.T
        kk = min(k, scores.shape[1])
        vals, idx = torch.topk(scores, kk, dim=1)
        best_scores = torch.cat([best_scores, vals], dim=1)
        best_idx = torch.cat([best_idx, idx + start], dim=1)
        if best_scores.shape[1] > k:
            vals, order = torch.topk(best_scores, k, dim=1)
            best_idx = torch.gather(best_idx, 1, order)
            best_scores = vals
    return best_scores.cpu().numpy(), best_idx.cpu().numpy()


def ranking_metrics(query_ids, doc_ids, top_scores, top_idx, qrels):
    """nDCG@10, MRR@10, Recall@100 (graded qrels, BEIR style) over queries with qrels."""
    ndcg, mrr, recall = [], [], []
    for qi, qid in enumerate(query_ids):
        rel = qrels.get(qid) or {}
        if not rel:
            continue
        ranked = [doc_ids[j] for j in top_idx[qi]]
        gains = [rel.get(d, 0) for d in ranked[:10]]
        dcg = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for r, d in enumerate(ranked[:10]):
            if d in rel:
                rr = 1.0 / (r + 1)
                break
        mrr.append(rr)
        hits = sum(1 for d in ranked[:100] if d in rel)
        recall.append(hits / len(rel))
    return {"ndcg@10": float(np.mean(ndcg)), "mrr@10": float(np.mean(mrr)),
            "recall@100": float(np.mean(recall)), "n_queries_scored": len(ndcg)}


def margin_analysis(q_emb, d_emb, query_ids, doc_ids, qrels, seed=0, n_random=256):
    """Positive-vs-random cosine margins, and how often a random doc beats the best positive."""
    import torch

    rng = np.random.default_rng(seed)
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    qn = torch.nn.functional.normalize(torch.as_tensor(q_emb).float(), dim=-1)
    dn = torch.nn.functional.normalize(torch.as_tensor(d_emb).float(), dim=-1)
    pos_scores, rand_means, rand_max, beaten = [], [], [], 0
    for qi, qid in enumerate(query_ids):
        pos = [doc_index[d] for d in (qrels.get(qid) or {}) if d in doc_index]
        if not pos:
            continue
        sample = rng.choice(dn.shape[0], size=min(n_random, dn.shape[0]), replace=False)
        p = (qn[qi] @ dn[pos].T)
        r = (qn[qi] @ dn[sample].T)
        pos_scores.append(p.max().item())
        rand_means.append(r.mean().item())
        rand_max.append(r.max().item())
        beaten += int(r.max().item() > p.max().item())
    n = len(pos_scores)
    return {
        "mean_best_positive_cos": float(np.mean(pos_scores)),
        "mean_random_cos": float(np.mean(rand_means)),
        "mean_max_of_random_cos": float(np.mean(rand_max)),
        "margin_best_positive_minus_random_mean": float(np.mean(pos_scores) - np.mean(rand_means)),
        f"frac_queries_where_max_of_{n_random}_random_beats_best_positive": beaten / max(n, 1),
        "corpus_cos_std_over_random": float(np.std(rand_means)),
    }


def hubness(top_idx, doc_ids, doc_lengths, k=10, n_hubs=10):
    """Which documents fill the top-k lists, and how concentrated that is."""
    counts = Counter(int(j) for row in top_idx[:, :k] for j in row)
    total = int(top_idx.shape[0] * k)
    freq = np.array(sorted(counts.values(), reverse=True), dtype=float)
    # Gini over the documents that appear at all (1 = one doc fills everything)
    if len(freq) > 1:
        cum = np.cumsum(np.sort(freq))
        gini = 1.0 - 2.0 * float(np.sum(cum) / (cum[-1] * len(freq))) + 1.0 / len(freq)
    else:
        gini = 1.0
    hubs = [{"doc_id": doc_ids[j], "in_topk_lists": c,
             "share_of_all_slots": c / total, "tokens": int(doc_lengths[j])}
            for j, c in counts.most_common(n_hubs)]
    return {
        "distinct_docs_in_topk": len(counts),
        "topk_slots": total,
        "distinct_over_slots": len(counts) / total,
        "share_of_slots_taken_by_top10_docs": float(freq[:10].sum() / total),
        "share_of_slots_taken_by_top1pct_docs": float(
            freq[:max(1, int(math.ceil(len(freq) * 0.01)))].sum() / total),
        "gini_over_appearing_docs": gini,
        "hubs": hubs,
    }


def length_analysis(q_emb, d_emb, doc_lengths, top_idx, max_len, seed=0, n_queries=512,
                    edges=(0, 32, 64, 128, 256, 512, 1024, 10**9)):
    """Per length bucket: corpus share, share of top-10 hits, mean cosine to random queries."""
    import torch

    rng = np.random.default_rng(seed)
    qs = rng.choice(q_emb.shape[0], size=min(n_queries, q_emb.shape[0]), replace=False)
    qn = torch.nn.functional.normalize(torch.as_tensor(q_emb[qs]).float(), dim=-1)
    dn = torch.nn.functional.normalize(torch.as_tensor(d_emb).float(), dim=-1)
    mean_cos = torch.empty(dn.shape[0])
    for start in range(0, dn.shape[0], 50_000):
        mean_cos[start:start + 50_000] = (qn @ dn[start:start + 50_000].T).mean(dim=0)
    mean_cos = mean_cos.numpy()
    lengths = np.asarray(doc_lengths)
    hit_counts = Counter(int(j) for row in top_idx[:, :10] for j in row)
    hits = np.zeros(len(lengths))
    for j, c in hit_counts.items():
        hits[j] = c
    truncated = float(np.mean(lengths >= max_len))
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & (lengths < hi)
        if not m.any():
            continue
        buckets.append({
            "tokens": f"[{lo}, {hi if hi < 10**9 else 'inf'})",
            "corpus_share": float(m.mean()),
            "top10_hit_share": float(hits[m].sum() / max(hits.sum(), 1)),
            "mean_cos_to_random_queries": float(mean_cos[m].mean()),
            "n_docs": int(m.sum()),
        })
    corr = float(np.corrcoef(np.minimum(lengths, max_len), mean_cos)[0, 1]) \
        if len(lengths) > 2 else float("nan")
    return {"frac_docs_truncated_at_max_len": truncated,
            "corr_doc_tokens_vs_mean_cos_to_queries": corr,
            "buckets": buckets}


def token_lengths(encoder, texts, max_len=None):
    tok = encoder.tokenizer
    out = []
    for i in range(0, len(texts), 1024):
        enc = tok(texts[i:i + 1024], truncation=False, padding=False)["input_ids"]
        out.extend(len(x) for x in enc)
    return out


def padding_sensitivity(encoder, query_texts, query_ids, doc_ids, d_emb, qrels, n, padded_len,
                        max_len, seed=0):
    """Encode queries three ways and compare embeddings + nDCG@10 on the same corpus."""
    import contextlib

    import torch

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(query_texts)), min(n, len(query_texts))))
    texts = [query_texts[i] for i in idx]
    qids = [query_ids[i] for i in idx]
    old_max = encoder.max_len
    encoder.max_len = max_len
    batched = np.asarray(encoder.encode(texts, batch_size=encoder.batch_size))
    alone = np.asarray(encoder.encode(texts, batch_size=1))
    # padded to a fixed length: what a naive, unsorted, oversized batch would do
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if encoder.device.type == "cuda" else contextlib.nullcontext())
    padded = []
    with torch.inference_mode(), autocast:
        for i in range(0, len(texts), encoder.batch_size):
            enc = encoder.tokenizer(texts[i:i + encoder.batch_size], padding="max_length",
                                    truncation=True, max_length=padded_len, return_tensors="pt")
            emb = encoder._embed(enc["input_ids"].to(encoder.device),
                                 enc["attention_mask"].to(encoder.device))
            padded.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
    padded = np.concatenate(padded)
    encoder.max_len = old_max

    def cos(a, b):
        return float(np.mean(np.sum(a * b, axis=1) /
                             (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)))

    def ndcg(q):
        s, i = topk_search(q, d_emb, 100)
        return ranking_metrics(qids, doc_ids, s, i, qrels)["ndcg@10"]

    return {
        "n_queries": len(texts),
        "cos(batched_sorted, alone)": cos(batched, alone),
        f"cos(padded_to_{padded_len}, alone)": cos(padded, alone),
        "ndcg@10_batched_sorted": ndcg(batched),
        "ndcg@10_alone": ndcg(alone),
        f"ndcg@10_padded_to_{padded_len}": ndcg(padded),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class STAdapter:
    """A sentence-transformers checkpoint (scripts/train_st_dpr.py output) behind the
    small encoder surface this script uses: ``encode`` / ``tokenizer`` / ``max_len``
    / ``batch_size`` / ``device`` / ``_embed``. Embeddings are L2-normalised like the
    repo encoders', so cosine scoring below is identical to evaluate_retrieval.py's."""

    def __init__(self, path, max_len, batch_size):
        import torch
        from sentence_transformers import SentenceTransformer

        from nexterabert.modeling_nexterabert import ensure_dynamo_recompile_budget

        ensure_dynamo_recompile_budget()
        self.model = SentenceTransformer(path, trust_remote_code=True)
        self.model.max_seq_length = max_len
        from nexterabert.hf_baselines import repair_wrapped_neobert

        repair_wrapped_neobert(self.model, max_len)   # zero-filled / 4096-row RoPE table
        self.tokenizer = self.model.tokenizer
        self.batch_size = batch_size
        self.device = torch.device(str(self.model.device))
        self.contrastive_run = None
        marker = Path(path) / "contrastive_run.json"
        if marker.exists():
            self.contrastive_run = json.loads(marker.read_text(encoding="utf-8"))

    @property
    def max_len(self):
        return self.model.max_seq_length

    @max_len.setter
    def max_len(self, value):
        self.model.max_seq_length = value
        from nexterabert.hf_baselines import repair_wrapped_neobert

        repair_wrapped_neobert(self.model, value)

    def encode(self, texts, batch_size=None, **kwargs):
        return self.model.encode(list(texts), batch_size=batch_size or self.batch_size,
                                 convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)

    def _embed(self, input_ids, attention_mask):
        return self.model({"input_ids": input_ids, "attention_mask": attention_mask})[
            "sentence_embedding"]


def build_encoder(args):
    from nexterabert.mteb_encoder import HFEncoder, NexteraEncoder

    if args.hf_model:
        return HFEncoder(args.hf_model, args.tokenizer, args.max_len, args.batch_size,
                         pooling=args.pooling)
    if args.model and Path(args.model).is_dir() and (Path(args.model) / "modules.json").exists():
        return STAdapter(args.model, args.max_len, args.batch_size)
    return NexteraEncoder(args.model, args.tokenizer or DEFAULT_TOKENIZER,
                          args.max_len, args.batch_size, pooling=args.pooling)


def run(args, encoder, data):
    doc_ids, doc_texts, query_ids, query_texts, qrels = data
    report = {"model": args.hf_model or args.model, "task": args.task, "split": args.split,
              "max_len": args.max_len, "n_docs": len(doc_ids), "n_queries": len(query_ids)}
    t0 = time.time()
    doc_lengths = token_lengths(encoder, doc_texts)
    report["doc_tokens"] = {
        "mean": float(np.mean(doc_lengths)), "median": float(np.median(doc_lengths)),
        "p90": float(np.percentile(doc_lengths, 90)), "max": int(np.max(doc_lengths)),
    }
    q_lengths = token_lengths(encoder, query_texts)
    report["query_tokens"] = {"mean": float(np.mean(q_lengths)), "max": int(np.max(q_lengths))}

    encoder.max_len = args.query_max_len or args.max_len
    q_emb = np.asarray(encoder.encode(query_texts, batch_size=args.batch_size))
    encoder.max_len = args.max_len
    d_emb = np.asarray(encoder.encode(doc_texts, batch_size=args.batch_size))
    report["encode_seconds"] = time.time() - t0

    scores, idx = topk_search(q_emb, d_emb, 100)
    report["ranking"] = ranking_metrics(query_ids, doc_ids, scores, idx, qrels)
    report["margins"] = margin_analysis(q_emb, d_emb, query_ids, doc_ids, qrels, args.seed)
    report["hubness"] = hubness(idx, doc_ids, doc_lengths)
    report["length"] = length_analysis(q_emb, d_emb, doc_lengths, idx, args.max_len, args.seed)
    if args.padding_queries > 0:
        report["padding"] = padding_sensitivity(
            encoder, query_texts, query_ids, doc_ids, d_emb, qrels,
            args.padding_queries, args.padded_len, args.query_max_len or args.max_len, args.seed)
    if args.doc_len_sweep:
        sweep = {}
        for L in [int(x) for x in args.doc_len_sweep.split(",") if x.strip()]:
            encoder.max_len = L
            d_L = np.asarray(encoder.encode(doc_texts, batch_size=args.batch_size))
            s_L, i_L = topk_search(q_emb, d_L, 100)
            sweep[str(L)] = ranking_metrics(query_ids, doc_ids, s_L, i_L, qrels)["ndcg@10"]
        encoder.max_len = args.max_len
        report["doc_len_sweep_ndcg@10"] = sweep
    return report


def print_report(r):
    print(f"\n== {r['task']} ({r['n_docs']} docs, {r['n_queries']} queries, max_len {r['max_len']})")
    rk = r["ranking"]
    print(f"  nDCG@10 {rk['ndcg@10'] * 100:6.2f}   MRR@10 {rk['mrr@10'] * 100:6.2f}   "
          f"Recall@100 {rk['recall@100'] * 100:6.2f}")
    m = r["margins"]
    print(f"  cos: best positive {m['mean_best_positive_cos']:.3f} | random mean "
          f"{m['mean_random_cos']:.3f} | max of 256 random {m['mean_max_of_random_cos']:.3f}")
    beat = [v for k, v in m.items() if k.startswith("frac_queries_where")][0]
    print(f"  a random document beats the best positive for {beat * 100:.1f}% of queries")
    h = r["hubness"]
    print(f"  hubness: {h['distinct_docs_in_topk']} distinct docs fill {h['topk_slots']} top-10 slots "
          f"(ratio {h['distinct_over_slots']:.2f}); top-10 docs take "
          f"{h['share_of_slots_taken_by_top10_docs'] * 100:.1f}% of slots, gini {h['gini_over_appearing_docs']:.2f}")
    for hub in h["hubs"][:5]:
        print(f"     doc {hub['doc_id']}: in {hub['in_topk_lists']} lists, {hub['tokens']} tokens")
    ln = r["length"]
    print(f"  length: {ln['frac_docs_truncated_at_max_len'] * 100:.1f}% of docs truncated; "
          f"corr(tokens, mean cos to queries) = {ln['corr_doc_tokens_vs_mean_cos_to_queries']:+.2f}")
    print("     bucket        corpus%  top10-hit%  mean cos")
    for b in ln["buckets"]:
        print(f"     {b['tokens']:<13} {b['corpus_share'] * 100:6.1f}  {b['top10_hit_share'] * 100:9.1f}  "
              f"{b['mean_cos_to_random_queries']:8.3f}")
    if "padding" in r:
        p = r["padding"]
        keys = list(p)
        print(f"  padding ({p['n_queries']} queries): {keys[1]} {p[keys[1]]:.4f} | {keys[2]} {p[keys[2]]:.4f}")
        print(f"     nDCG@10 batched {p['ndcg@10_batched_sorted'] * 100:.2f} | alone "
              f"{p['ndcg@10_alone'] * 100:.2f} | {keys[5]} {p[keys[5]] * 100:.2f}")
    if "doc_len_sweep_ndcg@10" in r:
        print("  doc max_len sweep nDCG@10: " +
              ", ".join(f"{k}: {v * 100:.2f}" for k, v in r["doc_len_sweep_ndcg@10"].items()))


def main(argv=None):
    args = parse_args(argv)
    data = load_task_data(args.task, args.split, args.languages)
    data = subsample(*data, args.max_docs, args.max_queries, args.seed)
    encoder = build_encoder(args)
    report = run(args, encoder, data)
    print_report(report)
    out = args.output or f"diag_{args.task}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
