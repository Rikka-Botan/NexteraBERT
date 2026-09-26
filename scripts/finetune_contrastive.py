#!/usr/bin/env python
"""Contrastive fine-tuning: the stage MTEB and BEIR numbers are measured after.

Both benchmarks score one vector per text, and an MLM/RTD backbone was never
trained to place related texts near each other -- so published encoder numbers on
either come from a model that first learned to embed. This script runs that stage,
under whichever prior-work protocol matches the benchmark being reported:

  --protocol mteb-nli  (default)  OptiBERT, Dervishi et al. EMNLP 2025, App. D.2
      Attentive pooling head + supervised-SimCSE InfoNCE on MNLI+SNLI triplets,
      3 epochs (Gao et al., 2021). Their Table 5 MTEB(eng, v2) scores follow this.
          -> feed the result to scripts/evaluate_mteb.py

  --protocol retrieval-msmarco    ModernBERT, Warner et al. 2024, 3.1.2 + App. E.2
      Mean pooling (no extra head) + InfoNCE over MS MARCO query/passage pairs
      with mined hard negatives, 1.25M samples, batch 16, 5% warmup, lr swept over
      [1e-5 .. 1e-4] (8e-5 chosen for ModernBERT-base, 5e-5 for BERT-base). Their
      Table 7 BEIR nDCG@10 -- BERT-base 38.9, ModernBERT-base 41.6 -- follows this.
          -> feed the result to scripts/evaluate_retrieval.py

  --protocol retrieval-mldr       ModernBERT, Warner et al. 2024, 3.1.3 "Single Vector - In Domain"
      The MS MARCO-tuned model (above) is further fine-tuned on the English MLDR
      training split (Shitao/MLDR, long documents: max_len 8192), then scored on
      the MLDR test split. Their Table 1 MLDR_ID column follows this. The paper
      gives no hyperparameters for this stage: one epoch at the DPR learning
      rate, batch 8, is the assumption made here (all overridable). Pooling
      follows the source checkpoint (attentive head if it has one, else mean).
          -> scripts/evaluate_retrieval.py --tasks MultiLongDocRetrieval --languages eng

    # MTEB
    python scripts/finetune_contrastive.py --model checkpoints/phase2/backbone \
        --output_dir checkpoints/phase2/simcse
    python scripts/evaluate_mteb.py --model checkpoints/phase2/simcse

    # Retrieval (BEIR / NanoBEIR)
    python scripts/finetune_contrastive.py --protocol retrieval-msmarco \
        --model checkpoints/phase2/backbone --output_dir checkpoints/phase2/dpr
    python scripts/evaluate_retrieval.py --model checkpoints/phase2/dpr

    # A Hugging Face baseline through the SAME stage, for a like-for-like MTEB row
    python scripts/finetune_contrastive.py --hf_model answerdotai/ModernBERT-base         --output_dir checkpoints/baselines/ModernBERT-base/simcse
    python scripts/finetune_contrastive.py --hf_model chandar-lab/NeoBERT         --output_dir checkpoints/baselines/NeoBERT/simcse
    python scripts/evaluate_mteb.py --model checkpoints/baselines/NeoBERT/simcse

--hf_model opens any AutoModel (ModernBERT, NeoBERT, BERT, ...) with the same
attentive head / mean pooling, optimiser and data (nexterabert.hf_baselines), and
uses the baseline's own tokenizer unless --tokenizer says otherwise. The result
directory is an ordinary save_pretrained directory plus the repo's markers, and
both eval scripts open it automatically through --model.

Everything a protocol pins is in PROTOCOLS below and overridable from the CLI. The
optimiser side both share is `evaluate_glue.py`'s recipe (AdamW betas (0.9, 0.95),
eps 1e-6, linear decay to zero after warmup, grad clip 1.0, weight decay on
matrices only, torch-scale), reusing its `build_optimizer` so the no-decay grouping
and LR ladder stay identical to GLUE fine-tuning. As in this repo's GLUE recipe,
layerwise LR decay is on by default (`--llrd 0.9`): the upper layers retain the
protocol LR while lower pretrained layers move progressively less. This is an
extra regulariser rather than part of either published protocol; `--no_llrd`
restores their flat LR exactly. A Hugging Face baseline gets the same default,
with its layer names mapped onto the same ladder.

Batch size is part of the objective, not just the memory knob: InfoNCE negatives
only come from tensors sharing a forward pass, so gradient accumulation cannot
substitute for it (see info_nce_loss). For the same reason DDP is deliberately not
supported -- per-rank batches would quietly weaken the loss instead of scaling it.
ModernBERT's batch 16 is small for a contrastive objective (31 negatives per
anchor, against 1023 for the NLI protocol's 512); it is reproduced here because it
is what the paper states, but raising it is usually the first thing to try if the
retrieval numbers disappoint. Its lr was swept AT batch 16, so re-sweep --lr
alongside, and record that the run deviates from the published protocol.

Asymmetric protocols pad the anchors apart from the documents (--split_views, on
for retrieval-msmarco): sentence-transformers tokenises each dataset column in its
own forward pass, so this matches the reference implementation as well as avoiding
the padding leak described in TripletCollator.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate_glue import (  # noqa: E402
    DEFAULT_LLRD,
    OPTIM_DEFAULTS,
    build_optimizer,
    set_seed,
)
from nexterabert.data import build_tokenizer  # noqa: E402
from nexterabert.hf_baselines import (  # noqa: E402
    hf_llrd,
    is_hf_model_dir,
    load_hf_contrastive_model,
    load_hf_tokenizer,
    save_hf_contrastive_model,
)
from nexterabert.simcse import (  # noqa: E402
    CONTRASTIVE_RUN_FILE,
    DEFAULT_TEMPERATURE,
    freeze_unused,
    info_nce_loss,
    load_contrastive_model,
    save_contrastive_model,
)

DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"

# Per-protocol defaults. `None` for a CLI flag means "take the value here"; every
# one of them is overridable.
PROTOCOLS = {
    # OptiBERT App. D.2 -> supervised SimCSE (Gao et al., 2021, Sec. 6.1) for the
    # constants the appendix delegates rather than states.
    "mteb-nli": {
        "dataset": "nli", "pooling": "attentive", "epochs": 3, "batch_size": 512,
        "max_len": 64, "lr": 5e-5, "warmup_pct": 0.06, "max_samples": 0,
        "eval_every": 250, "split_views": False,
    },
    # ModernBERT 3.1.2 ("1.25M samples ... batch size of 16", 5% warmup, via
    # sentence-transformers, whose MultipleNegativesRankingLoss is InfoNCE at
    # scale 20 = temperature 0.05 with mean pooling) + App. E.2 (lr sweep).
    "retrieval-msmarco": {
        "dataset": "msmarco", "pooling": "mean", "epochs": 1, "batch_size": 16,
        "max_len": 256, "lr": 8e-5, "warmup_pct": 0.05, "max_samples": 1_250_000,
        # queries and passages are padded apart: see TripletCollator
        "split_views": True,
        # ModernBERT selects by an lr sweep scored on NFCorpus/SciFact/TREC-COVID/
        # FiQA, not by a mid-training metric, so nothing is evaluated inline.
        "eval_every": 0,
    },
    # ModernBERT 3.1.3, single-vector in-domain: "Models trained on MS-MARCO are
    # further fine-tuned on long-context MLDR training set before being
    # evaluated." Start it from the retrieval-msmarco output, not the raw
    # backbone. Batch/epochs/lr are this repo's assumptions (see module docstring).
    # pooling "auto" = keep whatever the source checkpoint trained: the attentive
    # head when --model carries pooling_head.pt (an mteb-nli run), else mean.
    "retrieval-mldr": {
        "dataset": "mldr", "pooling": "auto", "epochs": 1, "batch_size": 8,
        "max_len": 8192, "lr": 8e-5, "warmup_pct": 0.05, "max_samples": 0,
        "split_views": True, "eval_every": 0,
    },
}

# NLI label ids shared by MNLI and SNLI: 0 entailment, 1 neutral, 2 contradiction
# (-1 marks SNLI's examples with no gold label, which are dropped).
ENTAILMENT, CONTRADICTION = 0, 2
NLI_SOURCES = ("nyu-mll/multi_nli", "stanfordnlp/snli")
# MS MARCO with hard negatives mined by msmarco-distilbert-base-tas-b, in the
# sentence-transformers "Embedding Model Datasets" collection: (query, positive,
# negative) columns, which is the shape ModernBERT's ST training consumes.
MSMARCO_DATASET = ("sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
                   "triplet")
# MLDR (Chen et al., 2024): the English train split carries, per query, a list of
# positive and a list of (BM25-mined) negative long documents. The Hub repo is a
# loading-script dataset (MLDR.py), which datasets >= 4 refuses to run, so the
# raw gzipped JSONL is fetched and streamed directly (1.25 GB; long documents).
MLDR_DATASET = ("Shitao/MLDR", "mldr-v1.0-en/train.jsonl.gz")


# ---------------------------------------------------------------------------
# Data: (anchor, positive, hard negative) triplets
# ---------------------------------------------------------------------------

def build_nli_triplets(max_samples: int = 0) -> list[tuple[str, str, str]]:
    """Group NLI rows into SimCSE's supervised triplets.

    A premise that carries both an entailment and a contradiction hypothesis
    yields one triplet per pair it can form -- the construction behind SimCSE's
    released `nli_for_simcse.csv` (~275k triplets from SNLI+MNLI). The premise is
    the anchor, the entailed hypothesis the positive, the contradicting one the
    hard negative.
    """
    from datasets import load_dataset

    grouped: dict[str, tuple[list, list]] = {}
    for name in NLI_SOURCES:
        ds = load_dataset(name)["train"]
        premises, hypotheses, labels = ds["premise"], ds["hypothesis"], ds["label"]
        kept = 0
        for premise, hypothesis, label in zip(premises, hypotheses, labels):
            if label not in (ENTAILMENT, CONTRADICTION) or not premise or not hypothesis:
                continue
            entailed, contradicting = grouped.setdefault(premise, ([], []))
            (entailed if label == ENTAILMENT else contradicting).append(hypothesis)
            kept += 1
        print(f"[data] {name}: {kept} usable rows ({len(ds)} total)")

    triplets = []
    for premise, (entailed, contradicting) in grouped.items():
        for positive, negative in zip(entailed, contradicting):
            triplets.append((premise, positive, negative))
    print(f"[data] {len(triplets)} triplets from {len(grouped)} premises")
    return triplets


def build_msmarco_triplets(max_samples: int = 0) -> list[tuple[str, str, str]]:
    """(query, positive passage, mined hard negative) rows from MS MARCO."""
    from datasets import load_dataset

    name, config = MSMARCO_DATASET
    ds = load_dataset(name, config)["train"]
    if max_samples and max_samples < len(ds):
        # a fixed-seed shuffle, so a capped run is a random sample of the corpus
        # rather than its first N rows (which are ordered by query id)
        ds = ds.shuffle(seed=42).select(range(max_samples))
    print(f"[data] {name} [{config}]: {len(ds)} triplets")
    return list(zip(ds["query"], ds["positive"], ds["negative"]))


def build_mldr_triplets(max_samples: int = 0, path: str | None = None,
                        negatives: str = "one") -> list[tuple[str, str, str]]:
    """(query, positive long document, mined negative long document) from MLDR-en.

    ``negatives="one"`` samples one (positive, negative) pair per query; ``"all"``
    pairs every mined negative of a query with a sampled positive -- MLDR-en TRAIN
    carries 20 BM25 negatives per query (dev: 7), so this is ~20x the triplets;
    an integer ``k`` samples ``k`` of them per query. Triplets of one query stay
    adjacent, so a caller can hold out whole queries.

    ``path`` overrides the Hub download with a local ``.jsonl[.gz]`` in the same
    record format (``query_id`` / ``query`` / ``positive_passages`` /
    ``negative_passages``, each passage ``{"docid", "text"}``).
    """
    per_query = None                       # None = all, else how many to sample
    if negatives == "one":
        per_query = 1
    elif negatives != "all":
        try:
            per_query = int(negatives)
        except (TypeError, ValueError):
            raise ValueError(
                f"negatives must be 'one', 'all' or an integer, got {negatives!r}") from None
        if per_query < 1:
            raise ValueError(f"negatives must be >= 1, got {per_query}")
    import gzip
    import random

    if path is None:
        from huggingface_hub import hf_hub_download

        repo, filename = MLDR_DATASET
        path = hf_hub_download(repo, filename, repo_type="dataset")
        label = f"{repo} [{filename}]"
    else:
        label = path

    rng = random.Random(42)
    opener = gzip.open if str(path).endswith(".gz") else open
    triplets, n_rows = [], 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            positives = [p["text"] for p in (row.get("positive_passages") or []) if p.get("text")]
            negatives = [n["text"] for n in (row.get("negative_passages") or []) if n.get("text")]
            if not positives or not negatives:
                continue
            # sampled (not the first k) so a rerun with a different cap still sees a
            # random mix; every pair reuses one sampled positive for the query
            chosen = negatives if per_query is None else rng.sample(
                negatives, min(per_query, len(negatives)))
            positive = rng.choice(positives)
            for neg in chosen:
                triplets.append((row["query"], positive, neg))
    if max_samples and max_samples < len(triplets):
        rng.shuffle(triplets)
        triplets = triplets[:max_samples]
    print(f"[data] {label}: {len(triplets)} triplets from {n_rows} queries")
    return triplets


def build_triplets(dataset: str, cache: str | None = None, max_samples: int = 0):
    if cache and Path(cache).exists():
        with open(cache, encoding="utf-8") as f:
            triplets = [tuple(t) for t in json.load(f)]
        print(f"[data] {len(triplets)} triplets from cache {cache}")
        return triplets[:max_samples] if max_samples else triplets

    builder = {"nli": build_nli_triplets, "msmarco": build_msmarco_triplets,
               "mldr": build_mldr_triplets}[dataset]
    triplets = builder(max_samples)

    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(triplets, f)
        print(f"[data] cached -> {cache}")
    return triplets


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        return self.triplets[i]


class TripletCollator:
    """Tokenise a batch of triplets into padded id/mask tensors.

    ``split_views=False`` (symmetric data, e.g. NLI): one flat (3B, T) batch with
    the views interleaved (a0, p0, n0, a1, ...), which the embeddings split back
    out of with a stride-3 slice.

    ``split_views=True`` (asymmetric data, e.g. MS MARCO): the anchors are
    tokenised separately from the passages, so a ~10-token query is padded among
    other queries instead of out to a 256-token passage. That matters here beyond
    wasted compute -- NexteraHRA's conv runs before the mask (see
    ``nexterabert.mteb_encoder``), so padding leaks into the last real tokens of
    the shortest text in the batch. Mixing queries and passages in one padded
    batch therefore trains the query side on contaminated states, while retrieval
    evaluation (which batches by length) encodes them clean: a train/eval
    mismatch on exactly the side retrieval is most sensitive to.
    """

    def __init__(self, tokenizer, max_len, split_views=False):
        self.tokenizer, self.max_len = tokenizer, max_len
        self.split_views = split_views

    def _encode(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, batch):
        if not self.split_views:
            return (self._encode([text for triplet in batch for text in triplet]),)
        anchors = [triplet[0] for triplet in batch]
        # positives then negatives: one pass over everything that is a document,
        # so both sides of the loss see the same padding regime
        documents = [t[1] for t in batch] + [t[2] for t in batch]
        return self._encode(anchors), self._encode(documents)


# ---------------------------------------------------------------------------
# STS-B dev: SimCSE's checkpoint selection metric
# ---------------------------------------------------------------------------

def load_stsb_dev():
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "stsb")["validation"]
    return ds["sentence1"], ds["sentence2"], np.asarray(ds["label"], dtype=np.float64)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, dev, max_len, batch_size, autocast):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        with autocast:
            emb = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev))
        out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(out)


def stsb_spearman(model, tokenizer, stsb, dev, max_len, batch_size, autocast) -> float:
    """Spearman correlation between cosine similarity and the human score."""
    was_training = model.training
    model.eval()
    s1, s2, gold = stsb
    e1 = embed_texts(model, tokenizer, s1, dev, max_len, batch_size, autocast)
    e2 = embed_texts(model, tokenizer, s2, dev, max_len, batch_size, autocast)
    sims = (e1 * e2).sum(dim=-1).numpy()
    model.train(was_training)
    return float(spearmanr(sims, gold)[0])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    # Match GLUE/pretraining throughput on Ampere+ without changing the requested
    # autocast dtype. This is a no-op for CPU and unsupported CUDA hardware.
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit(
            "finetune_contrastive.py is single-process by design: InfoNCE negatives "
            "come from one forward pass, so DDP would shrink the objective's "
            "negatives per rank. Run it without torchrun and raise --batch_size.")

    hf_source = args.hf_model or (args.model if is_hf_model_dir(args.model) else None)
    n_layer, depth_fn = None, None
    if hf_source:
        # a Hugging Face baseline: its own tokenizer, the same pooling head /
        # InfoNCE / optimiser recipe (nexterabert.hf_baselines)
        tokenizer = load_hf_tokenizer(args.tokenizer)
        model, config, info = load_hf_contrastive_model(
            hf_source, pooling=args.pooling, max_len=args.max_len)
        frozen = model.freeze_unused()
        n_layer, depth_fn = hf_llrd(model)
        print(f"[hf] {hf_source}: {type(model.encoder).__name__}, "
              f"{n_layer} layers, hidden {model.hidden_size}, "
              f"{sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M params")
    else:
        tokenizer = build_tokenizer(args.tokenizer)
        model, config, info = load_contrastive_model(args.model, pooling=args.pooling)
        frozen = freeze_unused(model)
    if info["missing"]:
        print(f"[warn] missing encoder keys: {info['missing'][:4]} ...")
    if info.get("pooling_head"):
        print("[info] resuming from an already-trained pooling head")
    model.to(dev)

    triplets = build_triplets(args.dataset, cache=args.triplets_cache,
                              max_samples=args.max_samples)
    loader = DataLoader(TripletDataset(triplets), batch_size=args.batch_size,
                        shuffle=True, drop_last=True,
                        collate_fn=TripletCollator(tokenizer, args.max_len,
                                                   split_views=args.split_views),
                        num_workers=args.num_workers, pin_memory=True)

    optimizer, groups = build_optimizer(
        model, args.lr, args.weight_decay, (args.beta1, args.beta2), args.eps,
        llrd=args.llrd, n_layer=n_layer, depth_fn=depth_fn)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        decay = (1.0 - t) if args.schedule == "linear" else 0.5 * (1.0 + math.cos(math.pi * t))
        return args.alpha_f + (1.0 - args.alpha_f) * decay

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = args.dtype != "fp32" and dev.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    autocast = (torch.autocast(device_type=dev.type, dtype=amp_dtype)
                if use_amp else contextlib.nullcontext())

    train_model = model
    if args.compile and dev.type == "cuda":
        # dynamic=True: the batch's padded length changes every step
        train_model = torch.compile(model, dynamic=True)

    stsb = load_stsb_dev() if args.eval_every > 0 else None

    print(f"[{args.protocol}] {len(triplets)} triplets  pooling={args.pooling}  "
          f"bs={args.batch_size}  epochs={args.epochs}  steps={total_steps}  "
          f"lr={args.lr:.1e}  wd={args.weight_decay}  tau={args.temperature}  "
          f"max_len={args.max_len}  split_views={args.split_views}  "
          f"llrd={args.llrd:g}  warmup={warmup_steps}  dtype={args.dtype}  "
          f"dev={dev.type}")
    n_decayed = sum(g["n_params"] for g in groups if not g["no_decay"])
    n_free = sum(g["n_params"] for g in groups if g["no_decay"])
    print(f"[{args.protocol}] decayed {n_decayed/1e6:.1f}M params | no-decay "
          f"{n_free/1e6:.1f}M | frozen {len(frozen)} unused tensors")
    if args.llrd != 1.0:
        group_lrs = [group["lr"] for group in groups]
        print(f"[{args.protocol}] llrd={args.llrd:g}/layer on the protocol lr: "
              f"{len(groups)} param groups, lr {min(group_lrs):.2e} "
              f"(embeddings) .. {max(group_lrs):.2e} (top)")

    best_score, best_state, step = None, None, 0
    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        for views in loader:
            with autocast:
                embedded = [train_model(input_ids=ids.to(dev, non_blocking=True),
                                        attention_mask=mask.to(dev, non_blocking=True))
                            for ids, mask in views]
            if len(embedded) == 1:                      # interleaved a,p,n
                anchor, positive, negative = (embedded[0][i::3] for i in range(3))
            else:                                       # anchors | positives+negatives
                anchor, documents = embedded
                positive, negative = documents.chunk(2, dim=0)
            loss = info_nce_loss(anchor, positive, negative,
                                 temperature=args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                print(f"  step {step}/{total_steps}  epoch {epoch+1}/{args.epochs}  "
                      f"loss {loss.item():.4f}  "
                      f"top_lr {max(sched.get_last_lr()):.2e}  "
                      f"{time.time()-start:.0f}s")
            if args.eval_every > 0 and step % args.eval_every == 0:
                score = stsb_spearman(model, tokenizer, stsb, dev, args.max_len,
                                      args.batch_size, autocast)
                flag = ""
                if best_score is None or score > best_score:
                    best_score = score
                    # CPU snapshot so the best step survives the rest of the run
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()}
                    flag = "  *best*"
                print(f"  step {step}: STS-B dev spearman {score:.4f}{flag}")

    if best_state is not None:
        print(f"[{args.protocol}] restoring best checkpoint (STS-B dev {best_score:.4f})")
        model.load_state_dict(best_state)
    elif args.eval_every > 0:
        print("[warn] no evaluation ran (fewer steps than --eval_every); keeping last")

    summary = {
        "protocol": args.protocol,
        "pooling": args.pooling,
        "dataset": args.dataset,
        "model": hf_source or args.model,
        "backbone": "hf" if hf_source else "nexterabert",
        "hf_model": hf_source,
        "tokenizer": args.tokenizer,
        "triplets": len(triplets),
        "steps": total_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "max_len": args.max_len,
        "split_views": args.split_views,
        "llrd": args.llrd,
        "betas": [args.beta1, args.beta2],
        "eps": args.eps,
        "schedule": args.schedule,
        "warmup_pct": args.warmup_pct,
        "warmup_steps": warmup_steps,
        "alpha_f": args.alpha_f,
        "grad_clip": args.grad_clip,
        "dtype": args.dtype,
        "compile": args.compile,
        "seed": args.seed,
        "stsb_dev_spearman": best_score,
        "wall_clock_seconds": time.time() - start,
    }
    if hf_source:
        out = save_hf_contrastive_model(model, args.output_dir, tokenizer=tokenizer,
                                        run_summary=summary)
    else:
        out = save_contrastive_model(model, args.output_dir, config=config,
                                     tokenizer=tokenizer, run_summary=summary)
    print(json.dumps(summary, indent=2))
    print(f"[{args.protocol}] saved -> {out} (protocol recorded in {CONTRASTIVE_RUN_FILE})")
    nxt = ("scripts/evaluate_retrieval.py" if args.dataset in ("msmarco", "mldr")
           else "scripts/evaluate_mteb.py")
    print(f"[{args.protocol}] next: python {nxt} --model {out}")


def parse_args(argv=None):
    g = OPTIM_DEFAULTS
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="mteb-nli", choices=sorted(PROTOCOLS),
                   help="which prior-work recipe to follow; sets the defaults below")
    p.add_argument("--model",
                   help="pretrained NexteraBERT backbone directory (or a directory "
                        "written by an earlier --hf_model run, to continue it)")
    p.add_argument("--hf_model",
                   help="a Hugging Face baseline instead, e.g. answerdotai/ModernBERT-base "
                        "or chandar-lab/NeoBERT: same head, loss, data and optimiser, "
                        "so its MTEB row is comparable to NexteraBERT's")
    p.add_argument("--output_dir", required=True, help="where the tuned model goes")
    p.add_argument("--tokenizer", default=None,
                   help="default: ModernBERT's for a NexteraBERT backbone, the "
                        "baseline's own for --hf_model / a baseline directory")
    # --- protocol-pinned; None means "take the protocol's value" ---
    p.add_argument("--dataset", default=None, choices=["nli", "msmarco", "mldr"])
    p.add_argument("--pooling", default=None, choices=["attentive", "mean"],
                   help="attentive adds a trained pooling head; mean tunes the "
                        "encoder under the backbone's own masked mean pooling")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="triplets per step; also the number of in-batch negatives, "
                        "so halving it on OOM weakens the objective (gradient "
                        "accumulation is not an equivalent substitute)")
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--split_views", dest="split_views", default=None,
                   action="store_true",
                   help="pad anchors apart from documents (asymmetric data)")
    p.add_argument("--no_split_views", dest="split_views", action="store_false")
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap the triplet count (0 = all)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--warmup_pct", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    # --- optimiser (evaluate_glue.py's recipe) ---
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="torch-scale (per-step decay lr_t*wd*p), matrices only")
    p.add_argument("--llrd", type=float, default=DEFAULT_LLRD,
                   help="layerwise LR decay (LLRD) factor in (0, 1], applied as a "
                        "multiplier on the protocol LR: the top keeps that LR and "
                        "the embeddings use llrd**(n_layer+1) times it. Matches "
                        f"the GLUE default ({DEFAULT_LLRD}); typical 0.8-0.95. "
                        "1.0 (or --no_llrd) restores the flat published protocol")
    p.add_argument("--no_llrd", dest="llrd_enabled", action="store_false",
                   help="disable layerwise LR decay and use the protocol LR at "
                        "every depth (the published contrastive protocols)")
    p.set_defaults(llrd_enabled=True)
    p.add_argument("--beta1", type=float, default=g["beta1"])
    p.add_argument("--beta2", type=float, default=g["beta2"])
    p.add_argument("--eps", type=float, default=g["eps"])
    p.add_argument("--schedule", default=g["schedule"], choices=["linear", "cosine"])
    p.add_argument("--alpha_f", type=float, default=g["alpha_f"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    # --- run control ---
    p.add_argument("--eval_every", type=int, default=None,
                   help="STS-B dev checkpoint selection, as in SimCSE; 0 disables")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true",
                               help="enable torch.compile on CUDA (default)")
    compile_group.add_argument("--no_compile", dest="compile", action="store_false",
                               help="disable torch.compile")
    p.set_defaults(compile=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--triplets_cache", default=None,
                   help="JSON file to cache the built triplets in (skips the rebuild)")

    args = p.parse_args(argv)
    if bool(args.model) == bool(args.hf_model):
        p.error("pass exactly one of --model (NexteraBERT backbone) or --hf_model "
                "(Hugging Face baseline)")
    if args.tokenizer is None:
        if args.hf_model or is_hf_model_dir(args.model):
            args.tokenizer = args.hf_model or args.model
        else:
            args.tokenizer = DEFAULT_TOKENIZER
    for key, value in PROTOCOLS[args.protocol].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.pooling == "auto":
        from nexterabert.simcse import has_pooling_head

        args.pooling = ("attentive" if args.model and has_pooling_head(args.model)
                        else "mean")
    if not args.llrd_enabled:
        args.llrd = 1.0
    if not 0.0 < args.llrd <= 1.0:
        p.error(f"--llrd must be in (0, 1] (got {args.llrd}); "
                "1.0 disables layerwise decay")
    return args


if __name__ == "__main__":
    train(parse_args())
