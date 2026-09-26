"""MTEB-compatible encoder wrapper around ``NexteraBERTForEmbedding``.

Shared by `scripts/evaluate_retrieval.py` (NanoBEIR/BEIR) and
`scripts/evaluate_mteb.py` (the full MTEB suite). Kept out of `__init__`'s lazy
export table on purpose: it is only useful when the optional `mteb` extra is
installed.

mteb >= 2.0 dispatches on `runtime_checkable` protocols instead of duck typing,
so an encoder must expose ``encode`` / ``similarity`` / ``similarity_pairwise``
/ ``mteb_model_meta`` or `AbsRetrieval._evaluate_subset` rejects it with
"expects a SearchInterface, Encoder, or CrossEncoder". `encode` is also handed a
``DataLoader`` of collated batches there, where mteb 1.x passed a list of
strings; both call styles are accepted below.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import weakref
from pathlib import Path

import numpy as np
import torch

from .data import build_tokenizer
from .loading import load_for_task


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_texts(batch) -> list[str]:
    """Pull the text out of one mteb batch (v2) or one corpus item (v1)."""
    text = batch.get("text")
    if text is None:
        text = batch.get("query", batch.get("body"))
    if text is None:
        return []
    if isinstance(text, str):
        # mteb 1.x passes corpus documents one dict at a time, with the title in
        # a separate field that the caller is expected to prepend.
        title = batch.get("title")
        if isinstance(title, str) and title:
            return [(title + " " + text).strip()]
        return [text]
    # mteb 2.x collates a batch into {"id": [...], "text": [...], ...} and has
    # already folded the title into "text", so it must not be prepended twice.
    return [t if isinstance(t, str) else str(t) for t in text]


def extract_texts(inputs) -> list[str]:
    """Flatten whatever mteb hands to ``encode`` into a list of strings.

    mteb >= 2.0 passes a ``DataLoader`` of collated dicts; mteb 1.x passed a
    plain list of strings (queries) or of ``{"title", "text"}`` dicts (corpus).
    """
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, dict):
        return batch_texts(inputs)
    texts: list[str] = []
    for item in inputs:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.extend(batch_texts(item))
        else:
            texts.append(str(item))
    return texts


def _unlink_memmap(path: str) -> None:
    """Remove an embedding spill after its last NumPy reference is released."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        # On Windows a derived view can briefly keep the mapping open. The file
        # is harmless and can be removed with the other cache contents later.
        pass


def build_model_meta(name, embed_dim, n_parameters, max_tokens, similarity="cosine"):
    """Best-effort ``ModelMeta`` so results land in a named output folder.

    mteb builds an anonymous meta when this is None, so schema drift between
    mteb versions degrades to that instead of breaking the run. ``similarity`` is
    ``"cosine"`` for the single-vector encoders here and ``"max_sim"`` for the
    ColBERT wrapper in ``scripts/evaluate_colbert.py``.
    """
    try:
        from mteb.models.model_meta import ModelMeta, ScoringFunction

        fn = {"cosine": ScoringFunction.COSINE, "dot": ScoringFunction.DOT_PRODUCT,
              "max_sim": ScoringFunction.MAX_SIM}[similarity]
        return ModelMeta(
            loader=None,
            name=name,
            revision="no_revision_available",
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=n_parameters,
            memory_usage_mb=None,
            max_tokens=max_tokens,
            embed_dim=embed_dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name=fn,
            use_instructions=False,
            training_datasets=None,
        )
    except Exception:  # noqa: BLE001 - optional metadata, never fatal
        return None


class _MeanPoolEncoder:
    """The MTEB protocol surface shared by every encoder here.

    Subclasses set ``tokenizer`` / ``model`` / ``device`` / ``max_len`` /
    ``batch_size`` / ``embed_dim`` / ``mteb_model_meta`` and implement
    ``_embed``; everything MTEB actually calls lives on this class.
    """

    #: Group texts of similar length into the same batch before encoding.
    #: NexteraHRA's block mean runs a depthwise conv (kernel ``block_size//2+1``)
    #: over the TOKEN sequence before the mask is applied, so the last couple of
    #: real tokens mix in whatever the padding holds -- measured at cos 0.98
    #: between an 8-token text encoded alone and the same text padded to 32
    #: (attn / lily / swa layers are exactly pad-invariant; only hra is not).
    #: The contamination is a fixed number of positions, so it hurts short texts
    #: far more than long ones -- i.e. queries more than documents, exactly the
    #: asymmetry retrieval scores on. Length-bucketed batching drives the padding
    #: inside a batch to near zero, which also makes encoding faster.
    sort_by_length = True
    #: A DataLoader is consumed in bounded groups, rather than flattened into
    #: one Python list. Sorting each group retains most length-bucketing benefit
    #: without retaining millions of strings (MindSmall has 2.36 M queries).
    length_sort_buffer_size = 4096
    #: Spill large output matrices to disk. MTEB requires ``encode`` to return a
    #: full matrix, but a memmap keeps that matrix out of resident host memory.
    embedding_memmap_threshold_bytes = 256 * 1024**2
    embedding_cache_dir: str | None = None

    def _allocate_embeddings(self, rows: int) -> np.ndarray:
        if self.embed_dim is None:
            raise ValueError("embed_dim must be known before MTEB encoding")
        nbytes = rows * self.embed_dim * np.dtype(np.float32).itemsize
        threshold = self.embedding_memmap_threshold_bytes
        if threshold <= 0 or nbytes < threshold:
            return np.empty((rows, self.embed_dim), dtype=np.float32)

        cache_dir = Path(self.embedding_cache_dir) if self.embedding_cache_dir else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(
            prefix="nexterabert-mteb-", suffix=".f32.mmap",
            dir=str(cache_dir) if cache_dir is not None else None,
        )
        os.close(fd)
        output = np.memmap(path, mode="w+", dtype=np.float32,
                           shape=(rows, self.embed_dim))
        weakref.finalize(output, _unlink_memmap, path)
        print(f"[encoder] spilling {nbytes / 1024**3:.1f} GiB embedding matrix "
              f"to {path}")
        return output

    def _encode_buffer(self, indexed_texts, output, batch_size, autocast_ctx):
        """Encode ``(original index, text)`` pairs into preallocated output."""
        if self.sort_by_length:
            indexed_texts.sort(key=lambda item: len(item[1]), reverse=True)
        for i in range(0, len(indexed_texts), batch_size):
            batch = indexed_texts[i:i + batch_size]
            indices = [item[0] for item in batch]
            chunk = [item[1] for item in batch]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            with autocast_ctx:
                emb = self._embed(input_ids, attn)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            output[indices] = emb.cpu().numpy()

    @staticmethod
    def _dataloader_size(inputs) -> int | None:
        """Return row count for an MTEB DataLoader without consuming it."""
        dataset = getattr(inputs, "dataset", None)
        if dataset is None:
            return None
        try:
            return len(dataset)
        except TypeError:
            return None

    @torch.inference_mode()
    def encode(self, inputs, batch_size: int | None = None, **kwargs):
        batch_size = batch_size or self.batch_size
        # bf16 autocast on CUDA: the encoder's matmuls/convs run in bfloat16 for a
        # sizeable inference speedup. The pooled embedding is cast back to fp32
        # before L2-normalising so retrieval scoring keeps full precision.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" else contextlib.nullcontext()
        )

        # MTEB 2.x supplies a DataLoader with a known-size dataset. Consume it in
        # bounded buffers and write directly into caller order. The old code first
        # retained every string, then held the per-batch arrays, their concatenated
        # copy, and a reordered copy simultaneously.
        total = self._dataloader_size(inputs)
        if total is not None:
            if total == 0:
                return np.zeros((0, self.embed_dim or 0), dtype=np.float32)
            output = self._allocate_embeddings(total)
            buffer = []
            seen = 0
            buffer_size = max(batch_size, self.length_sort_buffer_size)
            for item in inputs:
                texts = extract_texts(item)
                buffer.extend((seen + i, text) for i, text in enumerate(texts))
                seen += len(texts)
                if len(buffer) >= buffer_size:
                    self._encode_buffer(buffer, output, batch_size, autocast_ctx)
                    buffer.clear()
            if buffer:
                self._encode_buffer(buffer, output, batch_size, autocast_ctx)
            if seen != total:
                raise ValueError(f"MTEB DataLoader yielded {seen} rows, expected {total}")
            if isinstance(output, np.memmap):
                output.flush()
            return output

        # MTEB 1.x and direct callers pass an in-memory collection. Preallocate
        # one output matrix and scatter batches into it, avoiding concatenate and
        # the final full-size reorder copy.
        sentences = extract_texts(inputs)
        if not sentences:
            return np.zeros((0, self.embed_dim or 0), dtype=np.float32)
        output = self._allocate_embeddings(len(sentences))
        indexed_texts = list(enumerate(sentences))
        self._encode_buffer(indexed_texts, output, batch_size, autocast_ctx)
        if isinstance(output, np.memmap):
            output.flush()
        return output

    @staticmethod
    def _as_matrix(embeddings) -> torch.Tensor:
        t = torch.as_tensor(embeddings).float()
        return t if t.ndim > 1 else t.unsqueeze(0)

    def similarity(self, embeddings1, embeddings2):
        a = torch.nn.functional.normalize(self._as_matrix(embeddings1), dim=-1)
        b = torch.nn.functional.normalize(self._as_matrix(embeddings2), dim=-1)
        return a @ b.transpose(-2, -1)

    def similarity_pairwise(self, embeddings1, embeddings2):
        a = torch.nn.functional.normalize(self._as_matrix(embeddings1), dim=-1)
        b = torch.nn.functional.normalize(self._as_matrix(embeddings2), dim=-1)
        return (a * b).sum(dim=-1)

    # mteb 1.x calls encode_queries / encode_corpus; both defer to encode
    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        return self.encode(corpus, **kwargs)

    def _embed(self, input_ids, attention_mask):
        """(B, T) ids -> (B, C) pooled sentence embedding, pre-normalisation."""
        raise NotImplementedError


class NexteraEncoder(_MeanPoolEncoder):
    """Minimal MTEB-compatible encoder around ``NexteraBERTForEmbedding``.

    Every task type in MTEB (classification, clustering, pair classification,
    reranking, retrieval, STS, summarisation) is scored from the embeddings
    returned by ``encode``, so this one wrapper covers the whole suite.
    """

    def __init__(self, model_path, tokenizer_name, max_len=512, batch_size=64,
                 pooling="auto", embedding_cache_dir=None,
                 embedding_memmap_threshold_mb=256):
        from .simcse import contrastive_run, has_pooling_head, load_for_sentence_embedding

        self.tokenizer = build_tokenizer(tokenizer_name)
        # A directory written by scripts/finetune_contrastive.py under the MTEB
        # protocol carries a trained attentive pooling head next to the backbone.
        # Without one, fall back to the backbone's own masked mean pooling --
        # which is what the retrieval protocol trains (see nexterabert.simcse).
        self.pooling = pooling
        if pooling == "attentive" or (pooling == "auto" and has_pooling_head(model_path)):
            self.model, config, info = load_for_sentence_embedding(model_path)
            self.pooling = "attentive"
            if not info["pooling_head"]:
                print("[warn] --pooling attentive but no pooling_head.pt: the head is "
                      "randomly initialised. Run scripts/finetune_contrastive.py first.")
        else:
            self.model, config, info = load_for_task(model_path, "embedding")
            self.pooling = "mean"
        # Which contrastive stage (if any) produced this directory. Published MTEB
        # and BEIR numbers are all measured after one, so scoring a raw backbone
        # against them compares protocols rather than models -- say so loudly.
        self.contrastive_run = contrastive_run(model_path)
        if self.contrastive_run:
            print(f"[encoder] {model_path}  pooling={self.pooling}  "
                  f"protocol={self.contrastive_run.get('protocol')} "
                  f"({self.contrastive_run.get('dataset')}, "
                  f"{self.contrastive_run.get('triplets')} pairs)")
        else:
            print(f"[encoder] {model_path}  pooling={self.pooling}  "
                  f"protocol=NONE (raw backbone)")
            print("[warn] no contrastive stage recorded for this model. Published "
                  "MTEB/BEIR numbers are measured after one -- run "
                  "scripts/finetune_contrastive.py first for a comparable score.")
        if info["missing"]:
            print(f"[warn] missing encoder keys: {info['missing'][:4]} ...")
        self.device = resolve_device()
        self.model.to(self.device).eval()
        self.max_len = max_len
        self.batch_size = batch_size
        self.embedding_cache_dir = embedding_cache_dir
        self.embedding_memmap_threshold_bytes = int(
            embedding_memmap_threshold_mb * 1024**2
        )
        self.embed_dim = int(getattr(config, "n_embd", 0)) or None
        self.mteb_model_meta = build_model_meta(
            name=f"NexteraBERT/{Path(model_path).name}",
            embed_dim=self.embed_dim,
            n_parameters=sum(p.numel() for p in self.model.parameters()),
            max_tokens=float(max_len),
        )

    def _embed(self, input_ids, attention_mask):
        # Both heads take the same call and return (B, C): NexteraBERTForEmbedding
        # is encoder -> out_norm -> masked mean pool (the pretraining-untrained
        # pooler is deliberately not in that path), and
        # NexteraBERTForSentenceEmbedding replaces the mean with the trained
        # attentive pooling head.
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class HFEncoder(_MeanPoolEncoder):
    """The same recipe over any Hugging Face ``AutoModel`` (ModernBERT, NeoBERT, BERT).

    A raw MLM/RTD encoder has never been trained to put semantically similar
    sentences near each other, so its zero-shot MTEB scores are low by
    construction -- the question is only whether NexteraBERT is low *for its
    class*. This runs a reference encoder through the identical tokenisation,
    pooling, normalisation and scoring path, which turns "is the score too low?"
    into a comparison instead of a guess.

    ``model_name`` is a Hub id (scored zero-shot with masked mean pooling) or a
    directory written by ``scripts/finetune_contrastive.py --hf_model``, whose
    trained attentive head is picked up with ``pooling="auto"`` exactly as
    :class:`NexteraEncoder` does -- so a baseline and NexteraBERT can be scored
    after the *same* contrastive stage. NeoBERT's remote-code quirks (xformers
    import, zero-filled RoPE table under transformers 5, bool mask) are handled
    in ``nexterabert.hf_baselines``.
    """

    def __init__(self, model_name, tokenizer_name=None, max_len=512, batch_size=64,
                 pooling="auto", embedding_cache_dir=None,
                 embedding_memmap_threshold_mb=256):
        from .hf_baselines import (
            hf_source_name,
            load_hf_contrastive_model,
            load_hf_tokenizer,
        )

        model_name = str(model_name)
        # a baseline uses its own tokenizer unless one was named explicitly; a
        # saved fine-tuning directory carries it
        self.tokenizer = load_hf_tokenizer(tokenizer_name or model_name)
        self.model, config, info = load_hf_contrastive_model(
            model_name, pooling=pooling, max_len=max_len)
        self.pooling = self.model.pooling_kind
        if pooling == "attentive" and not info["pooling_head"]:
            print("[warn] --pooling attentive but no pooling_head.pt: the head is "
                  "randomly initialised. Run scripts/finetune_contrastive.py "
                  "--hf_model first.")
        self.contrastive_run = info.get("contrastive_run")
        source = hf_source_name(model_name)
        if self.contrastive_run:
            print(f"[encoder] {model_name} ({source})  pooling={self.pooling}  "
                  f"protocol={self.contrastive_run.get('protocol')} "
                  f"({self.contrastive_run.get('dataset')}, "
                  f"{self.contrastive_run.get('triplets')} pairs)")
        else:
            print(f"[encoder] {model_name}  pooling={self.pooling}  "
                  f"protocol=NONE (raw backbone)")
        self.device = resolve_device()
        self.model.to(self.device).eval()
        self.max_len = max_len
        self.batch_size = batch_size
        self.embedding_cache_dir = embedding_cache_dir
        self.embedding_memmap_threshold_bytes = int(
            embedding_memmap_threshold_mb * 1024**2
        )
        self.embed_dim = int(self.model.hidden_size) or None
        if Path(model_name).is_dir():
            meta_name = f"{Path(source).name}/{Path(model_name).name}"
        else:
            meta_name = model_name if "/" in model_name else f"baseline/{model_name}"
        self.mteb_model_meta = build_model_meta(
            name=meta_name,
            embed_dim=self.embed_dim,
            n_parameters=sum(p.numel() for p in self.model.encoder.parameters()),
            max_tokens=float(max_len),
        )

    def _embed(self, input_ids, attention_mask):
        # HFForSentenceEmbedding: AutoModel last_hidden_state -> attentive head
        # or masked mean, the same two poolings NexteraEncoder scores with.
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


#: Old name: the zero-shot mean-pool baseline path. ``HFEncoder`` with a Hub id
#: and ``pooling="auto"`` is exactly that.
HFMeanPoolEncoder = HFEncoder
