"""Data loading for ELECTRA / BERT pretraining.

The trainer performs masking / replacement *inside* its ``forward``, so the data
layer only has to deliver clean, packed token blocks.  Documents are packed into
fixed-length blocks **without crossing document boundaries**: each document (or
chunk of a long document) is bookended by ``[CLS]`` / ``[SEP]``, and leftover
positions are filled with ``[PAD]`` (attention_mask = 0).
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

# Streaming + tokenizer helpers live in the torch-free `streaming` module so the
# pre-tokenisation script can use them without importing torch; re-export here so
# existing `from nexterabert.data import build_tokenizer, ...` call sites keep working.
from .streaming import (  # noqa: F401
    DEFAULT_TOKENIZER,
    build_tokenizer,
    iter_rows_resilient,
    padded_vocab_size,
    _is_transient_network_error,
    _resilient_rows,
)


def _split_for_worker_and_rank(iterable, rank: int, world_size: int):
    """Shard a stream across DDP ranks *and* DataLoader workers (round-robin)."""
    info = get_worker_info()
    num_workers = info.num_workers if info is not None else 1
    worker_id = info.id if info is not None else 0
    total = world_size * num_workers
    offset = rank * num_workers + worker_id
    for i, item in enumerate(iterable):
        if i % total == offset:
            yield item


class PackedStreamDataset(IterableDataset):
    """Stream a 🤗 dataset, tokenise on the fly and emit packed token blocks.

    Each block is ``[CLS] doc1... [SEP] doc2... [SEP] ... [PAD]...`` - documents
    never cross block boundaries.  Long documents are split into chunks of at most
    ``max_seq_len - 2`` tokens, each forming its own ``[CLS]...[SEP]`` block.
    Short documents are packed together until the block is full.  Remaining
    positions are padded and ``attention_mask`` is set to 0 there.

    The stream is round-robin sharded across (DDP rank x DataLoader worker).
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_len: int = 128,
        text_column: str = "text",
        split: str = "train",
        dataset_config: str | None = None,
        dataset_mix: list | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1234,
        buffer_docs: int = 1000,
        max_stream_retries: int = 8,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_mix = dataset_mix
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_column = text_column
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.buffer_docs = buffer_docs
        self.max_stream_retries = max_stream_retries
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _raw_stream(self):
        from datasets import load_dataset

        if self.dataset_mix:
            from datasets import interleave_datasets

            datasets = []
            for src in self.dataset_mix:
                ds = load_dataset(src["name"], src.get("config"),
                                  split=self.split, streaming=True)
                min_score = src.get("min_score")
                if min_score is not None:
                    # Keep only high-quality rows (e.g. FineWeb-Edu int_score >= 4);
                    # filter BEFORE the rename so the score column is still present.
                    score_col = src.get("score_column", "score")
                    ds = ds.filter(
                        lambda r, _c=score_col, _m=min_score:
                        r.get(_c) is not None and r[_c] >= _m)
                tc = src.get("text_column", self.text_column)
                if tc != self.text_column:
                    ds = ds.rename_column(tc, self.text_column)
                datasets.append(ds)
            probs = [src.get("weight", 1.0) for src in self.dataset_mix]
            total = sum(probs)
            probs = [p / total for p in probs]
            ds = interleave_datasets(datasets, probabilities=probs,
                                     seed=self.seed,
                                     stopping_strategy="all_exhausted")
        else:
            ds = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=True,
            )

        if self.buffer_docs and self.buffer_docs > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_docs)
        return ds

    def _pack(self, stream) -> Iterator[dict]:
        max_chunk = self.max_seq_len - 2          # max tokens per doc chunk ([CLS] + chunk + [SEP])
        body_cap = self.max_seq_len - 1           # body capacity after [CLS]
        block_buf: list[int] = []

        for row in stream:
            text = row.get(self.text_column)
            if not text:
                continue
            doc_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not doc_ids:
                continue

            for i in range(0, len(doc_ids), max_chunk):
                chunk = doc_ids[i:i + max_chunk]
                needed = len(chunk) + 1           # chunk + [SEP]
                remaining = body_cap - len(block_buf)

                if needed <= remaining:
                    block_buf.extend(chunk)
                    block_buf.append(self.sep_id)
                else:
                    if block_buf:
                        yield self._emit_block(block_buf)
                    block_buf = list(chunk)
                    block_buf.append(self.sep_id)

        if block_buf:
            yield self._emit_block(block_buf)

    def _emit_block(self, body: list[int]) -> dict:
        input_ids = [self.cls_id] + body
        real_len = len(input_ids)
        pad_len = self.max_seq_len - real_len
        if pad_len > 0:
            input_ids += [self.pad_id] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * real_len + [0] * pad_len, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[dict]:
        # rebuild-and-continue on transient Hub/network errors so a single 408 /
        # SSL timeout doesn't tear down the worker (and with it the whole run)
        rows = _resilient_rows(self._raw_stream, max_retries=self.max_stream_retries)
        stream = _split_for_worker_and_rank(rows, self.rank, self.world_size)
        yield from self._pack(stream)


class PretokenizedDataset(torch.utils.data.Dataset):
    """Map-style dataset over a pre-tokenised, pre-packed int32 memmap shard.

    Produced by ``scripts/prepare_data.py``.  The ``.meta`` file is JSON with
    ``max_seq_len`` and ``pad_token_id``.  ``attention_mask`` is derived at read
    time: 1 for real tokens, 0 for padding.
    """

    def __init__(self, path: str, pad_token_id: int = 0):
        import numpy as np

        meta = path + ".meta"
        if not os.path.exists(meta):
            raise FileNotFoundError(f"missing meta file: {meta}")
        with open(meta, encoding="utf-8") as f:
            content = f.read().strip()
        try:
            meta_dict = json.loads(content)
            self.max_seq_len = meta_dict["max_seq_len"]
            self.pad_token_id = meta_dict.get("pad_token_id", pad_token_id)
            dtype = meta_dict.get("dtype", "int32")   # older shards were int32
        except (json.JSONDecodeError, KeyError):
            self.max_seq_len = int(content)
            self.pad_token_id = pad_token_id
            dtype = "int32"
        self.path = path
        self.dtype = dtype
        # Open the memmap in THIS process for the sanity checks. It is NOT pickled (see
        # __getstate__): np.memmap pickles by COPYING its whole buffer, so shipping the
        # dataset to spawn DataLoader workers would duplicate the (many-GB) shard per
        # worker x rank and OOM the host. Each worker re-opens it lazily instead.
        self._open()
        if self.data.size % self.max_seq_len != 0:
            raise ValueError(
                f"{path}: token count {self.data.size} is not a multiple of "
                f"max_seq_len {self.max_seq_len} for dtype={dtype!r} - the .meta "
                f"'dtype' disagrees with the file. Fix {meta} or rebuild the shard.")
        blocks = self.data.reshape(-1, self.max_seq_len)
        # A shard written as uint16 but read as int32 (or vice versa) decodes token
        # ids as huge/garbage values and later dies deep in the embedding as an
        # out-of-bounds CUDA assert. Catch that here with a clear, actionable error
        # by sanity-checking a spread of blocks (a correct shard never exceeds its
        # vocab, so any id in the millions means the dtype is wrong).
        self._len = int(blocks.shape[0])
        sample = blocks[:: max(1, self._len // 64)][:64]
        if sample.size:
            hi, lo = int(sample.max()), int(sample.min())
            if hi >= (1 << 20) or lo < 0:
                raise ValueError(
                    f"{path}: token id out of range (min={lo}, max={hi}) for "
                    f"dtype={dtype!r}. The shard was written with a different dtype - "
                    f"set the correct 'dtype' in {meta} (e.g. \"uint16\") or rebuild "
                    f"it with scripts/prepare_data.py.")

    def _open(self):
        import numpy as np
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r").reshape(
            -1, self.max_seq_len)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if getattr(self, "data", None) is None:     # reopened lazily in each worker
            self._open()
        ids = torch.from_numpy(self.data[idx].astype("int64"))
        attention_mask = (ids != self.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attention_mask}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = None         # drop the memmap so pickling stays tiny
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)  # memmap reopened lazily on first __getitem__


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


class VarLengthDataset(torch.utils.data.Dataset):
    """Map-style dataset over a **variable-length** (ragged) shard: one document per
    example, with NO padding stored.

    Produced by ``scripts/prepare_data.py --pack_mode docs``: a flat token stream
    (``path``) plus a ``path.lens`` int64 sidecar of per-document token counts. The
    ``.meta`` JSON carries ``{"format": "varlen", "pad_token_id", "dtype", "n_docs"}``.
    ``__getitem__`` returns a single document's ids (variable length) — padding into a
    rectangular batch is the collate's job (see :func:`make_dynamic_pad_collate`), so
    NexteraHRA sees exactly one document per sequence.
    """

    def __init__(self, path: str, pad_token_id: int = 0):
        import numpy as np

        meta_path = path + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"missing meta file: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.loads(f.read())
        if meta.get("format") != "varlen":
            raise ValueError(f"{path}: not a varlen shard (meta format={meta.get('format')!r}); "
                             f"use PretokenizedDataset for fixed-block shards.")
        self.path = path
        self.dtype = meta.get("dtype", "uint16")
        self.off_path = path + ".offsets"
        self.pad_token_id = meta.get("pad_token_id", pad_token_id)
        self.max_seq_len = meta["max_seq_len"]

        # offsets[i]:offsets[i+1] slices document i out of the flat token stream.
        # Precompute cumulative offsets ONCE and cache them to <path>.offsets (derived
        # from the .lens sidecar), so every process just memmaps the file.
        lens_path = path + ".lens"
        if (not os.path.exists(self.off_path)
                or os.path.getmtime(self.off_path) < os.path.getmtime(lens_path)):
            lens = np.fromfile(lens_path, dtype=np.int64)
            offs = np.empty(len(lens) + 1, dtype=np.int64)
            offs[0] = 0
            np.cumsum(lens, out=offs[1:])
            tmp = f"{self.off_path}.tmp.{os.getpid()}"   # atomic write (ranks idempotent)
            offs.tofile(tmp)
            os.replace(tmp, self.off_path)
            del lens, offs

        # Open the memmaps in THIS process for the sanity check + length. They are NOT
        # pickled (see __getstate__): np.memmap pickles by COPYING its whole buffer, so
        # sending the dataset to spawn DataLoader workers would duplicate the (many-GB)
        # token map per worker x rank and OOM the host (SIGKILL). Instead each worker
        # re-opens the memmaps lazily on first access, sharing the files via the OS.
        self._open()
        if int(self.offsets[-1]) != self.tokens.size:
            raise ValueError(
                f"{path}: .lens sum ({int(self.offsets[-1])}) != token count "
                f"({self.tokens.size}); .bin/.lens out of sync (delete {self.off_path}).")
        self._len = int(self.offsets.shape[0] - 1)

    def _open(self):
        import numpy as np
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.offsets = np.memmap(self.off_path, dtype=np.int64, mode="r")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if getattr(self, "tokens", None) is None:   # reopened lazily in each worker
            self._open()
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return {"input_ids": torch.from_numpy(self.tokens[s:e].astype("int64"))}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokens"] = None       # drop the memmaps so pickling stays tiny
        state["offsets"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)  # memmaps reopened lazily on first __getitem__


class DynamicPadCollate:
    """Collate variable-length documents into a rectangular batch by padding each
    batch to its longest example, rounded up to a multiple of ``multiple_of``.

    Padding is added here (at model-input time), not baked into the shard. Rounding to
    ``multiple_of`` (>= NexteraHRA's block_size, ideally a multiple of it) keeps the
    block-pooling clean and limits the number of distinct sequence lengths torch.compile
    sees. Emits ``attention_mask`` (1 for real tokens, 0 for padding).

    Implemented as a module-level **class** (not a closure) so it is picklable, which
    DataLoader workers require under the ``spawn`` / ``forkserver`` start methods.
    """

    def __init__(self, pad_token_id: int = 0, multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.multiple_of = multiple_of

    def __call__(self, batch):
        lengths = [b["input_ids"].numel() for b in batch]
        max_len = max(lengths)
        if self.multiple_of > 1:
            max_len = ((max_len + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = lengths[i]
            input_ids[i, :n] = b["input_ids"]
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_dynamic_pad_collate(pad_token_id: int = 0, multiple_of: int = 8):
    """Return a picklable dynamic-pad collate (a :class:`DynamicPadCollate` instance)."""
    return DynamicPadCollate(pad_token_id, multiple_of)


def is_varlen_shard(path: str) -> bool:
    """True if ``path`` is a variable-length (ragged) shard, by its .meta format."""
    meta_path = path + ".meta"
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.loads(f.read()).get("format") == "varlen"
    except (OSError, ValueError):
        return False


class CudaPrefetchLoader:
    """Overlap CPU-to-GPU transfer with compute by prefetching on a side stream.

    The side stream is created once per instance, not once per ``__iter__``: the
    training loop rebuilds the iterator at every epoch boundary and on every short-
    shard wrap-around, and a fresh ``torch.cuda.Stream`` each time leaks stream
    handles into the pool for no benefit.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device)

    def _to_dev(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _release(self, batch):
        """Hand the prefetched tensors over to the consuming (default) stream.

        The blocks were allocated on ``self.stream`` but are read by kernels on the
        default stream. ``wait_stream`` orders those reads after the copy, but the
        caching allocator tracks liveness per stream: once the last Python reference
        drops it is free to hand the same block to the NEXT ``_to_dev`` on the side
        stream, which can then overwrite data the default stream has not finished
        reading. ``record_stream`` tells the allocator the consuming stream also uses
        the block, which is what makes the reuse safe.
        """
        for v in batch.values():
            v.record_stream(torch.cuda.current_stream(self.device))
        return batch

    def __iter__(self):
        stream = self.stream
        it = iter(self.loader)
        batch = next(it)
        with torch.cuda.stream(stream):
            batch = self._to_dev(batch)
        for cpu_batch in it:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            ready = self._release(batch)
            with torch.cuda.stream(stream):
                batch = self._to_dev(cpu_batch)
            yield ready
        torch.cuda.current_stream(self.device).wait_stream(stream)
        yield self._release(batch)

    def __len__(self):
        return len(self.loader)
