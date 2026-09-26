#!/usr/bin/env python
"""Pre-tokenise and pack a corpus into a flat int32 memmap for fast pretraining.

Documents are packed into fixed-length blocks **without crossing document
boundaries**.  Each document (or chunk of a long document) is bookended by
``[CLS]`` / ``[SEP]``, and remaining positions are filled with ``[PAD]``.
The ``.meta`` file is JSON: ``{"max_seq_len": ..., "pad_token_id": ...}``.

Supports mixing multiple datasets with configurable weights via ``--dataset_mix``.
Each mix entry may also carry a ``min_score`` / ``score_column`` pair to keep only
high-quality rows (e.g. FineWeb-Edu's integer classifier score):

    python scripts/prepare_data.py \\
        --dataset_mix '[{"name":"HuggingFaceFW/fineweb-edu","config":"sample-100BT","weight":10,
                          "min_score":4,"score_column":"int_score"},
                        {"name":"wikimedia/wikipedia","config":"20231101.en","weight":3.5},
                        {"name":"bookcorpus/bookcorpus","weight":1.5}]' \\
        --max_seq_len 1024 --max_blocks 2000000 --output data/mix_1024.bin --pack_mode unpad

Or a single dataset (backward-compatible):

    python scripts/prepare_data.py \\
        --dataset_name HuggingFaceFW/fineweb-edu \\
        --max_seq_len 1024 --max_blocks 2000000 --output data/mix_1024.bin

To carve one mix into **disjoint** phase-1 / phase-2 shards without sharing a single
token, partition the (deterministically interleaved) document stream into
``--holdout_mod`` round-robin buckets by global document index and build each phase
from a different slice: ``--holdout_role train`` keeps every bucket except the
held-out one (the large phase-1 split), ``--holdout_role holdout`` keeps only the
held-out bucket (the small phase-2 split). Because both builds share the same mix,
weights and ``--seed``, the interleaved order is identical, so the split is exact.

The build is **resilient** (transient Hub/network errors pause-and-retry instead of
crashing) and **resumable**: progress is checkpointed to ``<output>.progress`` and
re-running the same command continues from where it stopped. The checkpoint stores
the streaming dataset's own ``state_dict`` (shard index + in-shard offset), so a
resume re-reads at most the current shard; checkpoints written by older versions
lack it and fall back to one last slow ``.skip()`` fast-forward.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Use the torch-free `streaming` module (NOT `nexterabert.data`, which pulls in
# torch). build_encoder wraps the raw `tokenizers` Rust engine WITHOUT importing
# transformers/torch: a pure-CPU tokenisation process that loads torch can crash the
# interpreter at shutdown on some CUDA boxes (torch worker threads +
# "PyGILState_Release ... during finalization").
from nexterabert.streaming import ResilientRows, build_encoder  # noqa: E402

# uint16 halves the on-disk shard vs int32; the tokenizer vocab (ModernBERT 50,368,
# ids <= 50367) fits under uint16's 65535 ceiling. The dtype is recorded in the .meta so the reader
# (nexterabert.data.PretokenizedDataset) picks it up; older int32 shards without a
# "dtype" key still load as int32.
TOKEN_DTYPE = "uint16"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="allenai/c4")
    p.add_argument("--dataset_config", type=str, default="en")
    p.add_argument("--dataset_mix", type=str, default=None,
                   help='JSON list of {"name","config","data_dir","text_column",'
                        '"weight"} dicts (all but "name" optional); '
                        "overrides --dataset_name when set")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_splits", type=int, default=1,
                   help="partition the source into N disjoint file-level shards "
                        "(datasets.split_dataset_by_node) and build only one — lets "
                        "phase 1 and phase 2 use non-overlapping C4 data cheaply")
    p.add_argument("--split_index", type=int, default=0,
                   help="which split in [0, num_splits) to build")
    p.add_argument("--holdout_mod", type=int, default=1,
                   help="partition the (interleaved) document stream into N deterministic "
                        "round-robin buckets by global doc index; 1 disables. Combine with "
                        "--holdout_role to build disjoint phase-1 / phase-2 shards from a "
                        "single mix without a slow .skip(). "
                        "In --pack_mode docs (variable length) it selects docs, not blocks.")
    p.add_argument("--holdout_index", type=int, default=0,
                   help="which bucket in [0, holdout_mod) is the held-out (phase-2) slice")
    p.add_argument("--holdout_role", type=str, default="train",
                   choices=["train", "holdout"],
                   help="'train' keeps every bucket EXCEPT holdout_index (the large phase-1 "
                        "split); 'holdout' keeps ONLY holdout_index (the small phase-2 split)")
    p.add_argument("--holdout_keep", type=str, default=None,
                   help="keep only buckets in the half-open range 'lo:hi' of "
                        "[0, holdout_mod). Overrides --holdout_role/--holdout_index and lets "
                        "one stream be carved into 3+ disjoint slices (train / phase-2 / eval).")
    p.add_argument("--pack_mode", type=str, default="concat",
                   choices=["concat", "pad", "unpad", "docs"],
                   help="'concat' packs multiple documents per block separated by [SEP] "
                        "(dense, but tokens attend across document boundaries). 'pad' puts "
                        "exactly ONE document (or chunk) per block and pads the rest (no "
                        "cross-document mixing, but wastes compute on padding). 'unpad' packs "
                        "documents densely BUT aligns each [CLS] doc [SEP] segment to a "
                        "--unpad_align boundary with a few pad tokens, so the block-pooling "
                        "NexteraHRA layers (and local mixers) never straddle two documents "
                        "while wasting almost no padding. 'docs' stores ONE document per "
                        "example as a variable-length ragged shard (flat tokens + a .lens "
                        "sidecar, NO padding baked in); padding is added at model-input time "
                        "by the dynamic-padding collate, so NexteraHRA sees a single document "
                        "per sequence.")
    p.add_argument("--unpad_align", type=int, default=8,
                   help="in --pack_mode unpad, pad each document segment up to a multiple of "
                        "this many tokens (>= NexteraHRA's block_size, and a multiple of it, "
                        "so a pooled block never mixes documents). Adds 1..unpad_align pad "
                        "tokens per document as a boundary gap. Default 8 (= 2x block_size 4).")
    p.add_argument("--stream_prefetch", type=int, default=1,
                   help="parquet streaming readahead: number of coalesced ranges "
                        "pyarrow prefetches in the background (C++ threads), "
                        "overlapping download with decode and hiding request "
                        "latency. 0 restores the default synchronous reads "
                        "(32MiB requests, no prefetch). Pure I/O tuning: row "
                        "order, outputs and checkpoints are unaffected.")
    p.add_argument("--stream_range_mb", type=int, default=128,
                   help="parquet streaming request size in MiB (datasets default "
                        "32). Bigger ranges = fewer round-trips on high-latency "
                        "links; RAM cost ~ stream_prefetch x this.")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--tokenizer", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--tokenizer_backend", type=str, default="auto",
                   choices=["auto", "gigatoken", "tokenizers"],
                   help="batch-encode engine: 'gigatoken' (Rust SIMD BPE, ~50x) "
                        "requires the package and its output must pass the exact-"
                        "parity self-test against 'tokenizers'; 'auto' tries "
                        "gigatoken and falls back. Both produce identical ids, "
                        "so shards and checkpoints are interchangeable.")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_blocks", type=int, default=2_000_000)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for dataset interleaving (deterministic resume)")
    p.add_argument("--max_retries", type=int, default=50,
                   help="consecutive transient-error retries before giving up")
    p.add_argument("--flush_every", type=int, default=10_000,
                   help="flush the memmap + progress checkpoint every N blocks")
    p.add_argument("--tokenize_batch", type=int, default=1000,
                   help="documents tokenised per fast-tokenizer batch call; higher "
                        "= more CPU (Rust-thread) parallelism at some memory cost")
    p.add_argument("--tokenize_char_budget", type=int, default=4_000_000,
                   help="also flush the pending batch once its combined text reaches "
                        "this many characters. Bounds peak tokenizer memory on "
                        "long-document corpora (e.g. FineWiki) so the Rust tokenizer "
                        "can't hit an allocation failure and abort() the process. "
                        "0 disables the char cap (batch purely by --tokenize_batch).")
    p.add_argument("--max_doc_chars", type=int, default=5_000_000,
                   help="hard-truncate any single document to this many characters "
                        "before tokenising, so one pathological row (a multi-MB dump "
                        "article) can't blow up memory. 0 disables the per-doc cap.")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing progress checkpoint and start over")
    return p.parse_args()


# --- dataset mix helpers ------------------------------------------------------

def _parse_dataset_mix(raw):
    """Parse a dataset-mix specification from a JSON string, list, or None."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return None


# --- identity / progress checkpoint -------------------------------------------

def _identity(args) -> dict:
    """The build parameters that must match for a resume to be valid."""
    mix = _parse_dataset_mix(args.dataset_mix)
    if mix is not None:
        dataset_info = {"dataset_mix": mix}
    else:
        dataset_info = {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
        }
    return {
        **dataset_info,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "max_blocks": args.max_blocks,
        "dtype": TOKEN_DTYPE,
        "num_splits": args.num_splits,
        "split_index": args.split_index,
        "holdout_mod": args.holdout_mod,
        "holdout_index": args.holdout_index,
        "holdout_role": args.holdout_role,
        "holdout_keep": args.holdout_keep,
        "pack_mode": args.pack_mode,
        "unpad_align": args.unpad_align,
    }


def _in_split(idx: int, args) -> bool:
    """True if document ``idx`` belongs to this build's holdout slice.

    Buckets are assigned round-robin by global document index, so builds that share
    the same mix/weights/seed (hence the same interleaved order) partition the
    identical stream. With ``--holdout_keep lo:hi`` a build keeps only buckets in
    ``[lo, hi)``; disjoint ranges (e.g. 0:89 / 89:99 / 99:100) give non-overlapping
    train / phase-2 / eval slices. Otherwise the legacy ``train`` / ``holdout`` role
    keeps all-but-one / only-one bucket.
    """
    if args.holdout_mod <= 1:
        return True
    b = idx % args.holdout_mod
    if args.holdout_keep:
        lo, hi = args.holdout_keep.split(":")
        return int(lo) <= b < int(hi)
    is_hold = (b == args.holdout_index)
    return is_hold if args.holdout_role == "holdout" else not is_hold


def _progress_path(output: str) -> str:
    return output + ".progress"


def _save_progress(output: str, identity: dict, written: int, docs_seen: int,
                   stream=None, buf=None):
    """Atomically checkpoint progress so a kill mid-flush can't corrupt it.

    ``stream`` is :meth:`ResilientRows.state` (or None): the dataset's own
    ``state_dict`` so a resume jumps straight to the right shard offset instead
    of re-streaming ``docs_seen`` rows through ``.skip()``. ``buf`` is the
    partially packed block (concat/unpad), persisted so a resumed build produces
    byte-identical output to an uninterrupted one.
    """
    path = _progress_path(output)
    tmp = path + ".tmp"
    payload = {"identity": identity, "written": written, "docs_seen": docs_seen}
    if stream is not None:
        payload["stream"] = stream
    if buf:
        # int() each element: the gigatoken backend fills buf with numpy
        # uint32 scalars, which json.dump rejects.
        payload["buf"] = [int(x) for x in buf]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _load_progress(output: str, identity: dict):
    """Return ``(written, docs_seen, stream, buf)`` to resume from, or ``None``."""
    path = _progress_path(output)
    if not (os.path.exists(path) and os.path.exists(output)):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    if state.get("identity") != identity:
        print("[prepare_data] existing progress checkpoint was built with "
              "different parameters; ignoring it and starting fresh "
              "(pass --fresh to silence this / force a rebuild).")
        return None
    return (int(state["written"]), int(state["docs_seen"]),
            state.get("stream"), state.get("buf") or [])


# --- streaming ---------------------------------------------------------------

def _scan_options(args):
    """pyarrow prefetch/range options for parquet streaming, or ``None``.

    ``datasets`` reads streamed parquet in synchronous 32MiB ranges by default;
    ``CacheOptions(prefetch_limit=N, range_size_limit=M)`` makes pyarrow fetch
    up to N coalesced ranges of up to M bytes in background C++ threads, so the
    download overlaps decode and request latency stops serialising the stream.
    Pure I/O tuning - row order/content and checkpoint state are unaffected.
    """
    if not getattr(args, "stream_prefetch", 0):
        return None
    try:
        import pyarrow
        import pyarrow.dataset as pads
        return pads.ParquetFragmentScanOptions(
            cache_options=pyarrow.CacheOptions(
                prefetch_limit=args.stream_prefetch,
                range_size_limit=args.stream_range_mb << 20))
    except Exception as err:  # noqa: BLE001 - old pyarrow etc.: just go without
        print(f"[prepare_data] parquet prefetch unavailable "
              f"({type(err).__name__}: {err}); using default reads", flush=True)
        return None


def _load_streaming(name, config=None, data_dir=None, split="train", columns=None,
                    scan_options=None):
    """``load_dataset(streaming=True)`` with column pruning + read prefetch.

    Packing only reads the text (and, when filtering, score) column; parquet is
    columnar, so skipping id/url/metadata columns cuts both download bytes and
    decode CPU. ``scan_options`` (see :func:`_scan_options`) adds background
    readahead. Builders that reject either kwarg fall back stepwise - row order
    and count are identical in every case, so checkpoints stay valid.
    """
    from datasets import load_dataset

    kwargs = {}
    if columns:
        kwargs["columns"] = sorted(set(columns))
    if scan_options is not None:
        kwargs["fragment_scan_options"] = scan_options
    while True:
        try:
            return load_dataset(name, config, data_dir=data_dir, split=split,
                                streaming=True, **kwargs)
        except (TypeError, ValueError) as err:
            if "fragment_scan_options" in kwargs:
                print(f"[prepare_data] read prefetch unsupported for {name} "
                      f"({type(err).__name__}); retrying without it", flush=True)
                del kwargs["fragment_scan_options"]
            elif "columns" in kwargs:
                print(f"[prepare_data] column pruning unavailable for {name} "
                      f"({type(err).__name__}); streaming all columns", flush=True)
                del kwargs["columns"]
            else:
                raise


def _make_stream(args):
    mix = _parse_dataset_mix(args.dataset_mix)
    text_col = args.text_column

    def _apply_split(ds):
        # Partition the stream into num_splits disjoint pieces (file-level when the
        # source has enough shards, e.g. C4's 1024 files) and keep split_index. This
        # lets phase 1 and phase 2 draw from non-overlapping C4 data without a slow
        # .skip() that would re-read every already-used document.
        if args.num_splits > 1:
            from datasets.distributed import split_dataset_by_node
            ds = split_dataset_by_node(ds, args.split_index, args.num_splits)
        return ds

    if mix is not None:
        from datasets import interleave_datasets

        def make(skip):
            datasets = []
            for src in mix:
                tc = src.get("text_column", text_col)
                min_score = src.get("min_score")
                cols = [tc]
                if min_score is not None:
                    cols.append(src.get("score_column", "score"))
                # data_dir: for repos organised by subdirectory instead of named
                # configs (e.g. bigcode/starcoderdata's per-language dirs).
                ds = _load_streaming(src["name"], src.get("config"),
                                     data_dir=src.get("data_dir"),
                                     split=args.split, columns=cols,
                                     scan_options=_scan_options(args))
                if min_score is not None:
                    # Keep only high-quality rows (e.g. FineWeb-Edu int_score >= 4).
                    # Filter BEFORE the rename so the score column is still present.
                    score_col = src.get("score_column", "score")
                    ds = ds.filter(
                        lambda r, _c=score_col, _m=min_score:
                        r.get(_c) is not None and r[_c] >= _m)
                if tc != text_col:
                    ds = ds.rename_column(tc, text_col)
                datasets.append(ds)
            if len(datasets) == 1:
                # A one-entry "mix" is just that dataset (build_data.sh packs each
                # source to its own token budget this way, so per-source token ratios
                # are exact); skip interleave to avoid pointless oversampling.
                ds = datasets[0]
            else:
                probs = [src.get("weight", 1.0) for src in mix]
                total = sum(probs)
                probs = [p / total for p in probs]
                ds = interleave_datasets(datasets, probabilities=probs,
                                         seed=args.seed,
                                         stopping_strategy="all_exhausted")
            ds = _apply_split(ds)
            if skip:
                ds = ds.skip(skip)
            return ds

        return make

    def make(skip):
        ds = _load_streaming(args.dataset_name, args.dataset_config,
                             split=args.split, columns=[text_col],
                             scan_options=_scan_options(args))
        ds = _apply_split(ds)
        if skip:
            ds = ds.skip(skip)
        return ds

    return make


def _make_rows(args, docs_seen: int, stream_resume):
    """Build the resilient ``(index, row)`` iterator for a (possibly resumed) run.

    With a ``stream`` section in the checkpoint the dataset state is restored
    natively (shard + offset: near-instant). Legacy checkpoints without one fall
    back to a slow-but-exact ``.skip(docs_seen)`` fast-forward one last time; the
    next checkpoint written then carries a stream state.
    """
    if stream_resume is not None:
        import datasets
        saved_ver = stream_resume.get("datasets_version")
        if saved_ver != datasets.__version__:
            # A library upgrade can reshape the state dict; loading it into a
            # differently-shaped pipeline could silently mis-position the stream.
            print(f"[prepare_data] WARNING: checkpoint stream state was written "
                  f"by datasets=={saved_ver}, installed is {datasets.__version__}; "
                  f"falling back to slow .skip() fast-forward to stay exact")
        else:
            try:
                return ResilientRows(
                    _make_stream(args),
                    start_index=int(stream_resume["pos"]),
                    stream_state=stream_resume["state"],
                    base_skip=int(stream_resume.get("skip", 0)),
                    max_retries=args.max_retries)
            except (KeyError, TypeError, ValueError) as err:
                print(f"[prepare_data] WARNING: malformed stream state in "
                      f"checkpoint ({type(err).__name__}: {err}); falling back "
                      f"to slow .skip() fast-forward")
    if docs_seen:
        print(f"[prepare_data] fast-forwarding {docs_seen:,} rows via .skip() - "
              f"this checkpoint has no usable stream state, so the skipped rows "
              f"are re-streamed once (slow, no progress output); the next "
              f"checkpoint stores a stream state that resumes instantly",
              flush=True)
    return ResilientRows(_make_stream(args), start_index=docs_seen,
                         max_retries=args.max_retries)


class _PrefetchBatches:
    """Read, filter and batch documents on a background thread.

    The packing loop used to alternate serially: pull rows (network +
    decompress + arrow->python), THEN tokenise, THEN write - each phase idled
    while the others ran. The reader thread overlaps streaming with the Rust
    tokenizer (``encode_batch`` releases the GIL), hiding the smaller of the
    two costs entirely.

    Yields ``(batch, snapshot)``: ``batch`` is the same ``[(global_idx, text),
    ...]`` the old inline loop built (identical boundaries: ``tokenize_batch``
    docs or ``tokenize_char_budget`` chars, whichever first, so outputs are
    byte-identical) and ``snapshot`` is ``rows.state()`` captured by the reader
    - the only thread allowed to touch the stream - at the batch boundary, so
    a checkpoint written after packing the batch is exact.
    """

    _DONE = object()

    def __init__(self, args, rows, depth: int = 8):
        self._args = args
        self._rows = rows
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._err = None
        self.wait_s = 0.0            # packer time spent starved of batches
        self._thread = threading.Thread(target=self._read, daemon=True,
                                        name="prepare-data-reader")
        self._thread.start()

    def _read(self):
        args = self._args
        pending, chars = [], 0
        try:
            for idx, row in self._rows:
                if self._stop.is_set():
                    return
                if not _in_split(idx, args):
                    continue
                text = row.get(args.text_column) or ""
                if args.max_doc_chars and len(text) > args.max_doc_chars:
                    text = text[:args.max_doc_chars]
                pending.append((idx, text))
                chars += len(text)
                if (len(pending) >= args.tokenize_batch
                        or (args.tokenize_char_budget
                            and chars >= args.tokenize_char_budget)):
                    self._put((pending, self._rows.state()))
                    pending, chars = [], 0
            if pending:
                self._put((pending, self._rows.state()))
            self._put(self._DONE)
        except BaseException as err:  # noqa: BLE001 - re-raised by the packer
            self._err = err
            self._put(self._DONE)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def __iter__(self):
        while True:
            t0 = time.perf_counter()
            item = self._q.get()
            self.wait_s += time.perf_counter() - t0
            if item is self._DONE:
                if self._err is not None:
                    raise self._err
                return
            yield item

    def stop(self):
        self._stop.set()


def _timing_start(docs: int, tokens: int) -> dict:
    return {"wall": time.perf_counter(), "tok": 0.0, "pack": 0.0,
            "wait": 0.0, "docs": docs, "tokens": tokens}


def _report_throughput(tim, prefetch, docs_seen: int, tokens: int):
    """Print throughput and where the wall-clock went since the last report.

    ``stream-wait`` is packer time starved waiting on the reader thread
    (network + decompress + arrow->python), ``tokenize`` the Rust
    ``encode_batch``, ``pack`` chunking + memmap writes. The biggest share
    names the bottleneck; the shares can sum below 100% (checkpoint writes,
    scheduling) - what matters is their ratio.
    """
    now = time.perf_counter()
    wall = max(now - tim["wall"], 1e-9)
    shares = [("stream-wait", prefetch.wait_s - tim["wait"]),
              ("tokenize", tim["tok"]), ("pack", tim["pack"])]
    label = max(shares, key=lambda kv: kv[1])[0]
    pct = " ".join(f"{name} {100 * s / wall:.0f}%" for name, s in shares)
    print(f"[timing] {(docs_seen - tim['docs']) / wall:,.0f} docs/s, "
          f"{(tokens - tim['tokens']) / wall / 1e6:.2f} M tokens/s | {pct} "
          f"-> bottleneck: {label}", flush=True)
    tim.update(wall=now, tok=0.0, pack=0.0, wait=prefetch.wait_s,
               docs=docs_seen, tokens=tokens)


def _build_varlen(args, tok, cls_id, sep_id, pad_id, identity):
    """One-document-per-example, variable-length (ragged) build.

    Writes a FLAT token stream (documents back-to-back, each ``[CLS] chunk [SEP]``,
    NO padding) plus a ``<output>.lens`` sidecar of per-document token counts. The
    training loader (``VarLengthDataset``) slices documents out and the dynamic-padding
    collate pads each *batch* to its longest example — so padding is added at
    model-input time (which NexteraHRA then block-pools cleanly, one doc per sequence),
    never baked into the shard.
    """
    max_chunk = args.max_seq_len - 2
    capacity = args.max_blocks * args.max_seq_len   # flat token budget (= blocks x seq)
    prog = _progress_path(args.output)
    lens_path = args.output + ".lens"

    resume = None
    stream_resume = None
    if (not args.fresh and os.path.exists(prog)
            and os.path.exists(args.output) and os.path.exists(lens_path)):
        try:
            with open(prog, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = None
        if st and st.get("identity") == identity and st.get("format") == "varlen":
            resume = (int(st["cur"]), int(st["n_docs"]), int(st["docs_seen"]))
            stream_resume = st.get("stream")
        elif st:
            # Mirror the fixed-block path: never restart a multi-day build
            # silently - say WHY the checkpoint was rejected.
            print("[prepare_data] existing progress checkpoint was built with "
                  "different parameters; ignoring it and starting fresh "
                  "(pass --fresh to silence this / force a rebuild).")
    if resume is not None:
        cur, n_docs, docs_seen = resume
        arr = np.memmap(args.output, dtype=TOKEN_DTYPE, mode="r+", shape=(capacity,))
        lens = list(np.fromfile(lens_path, dtype=np.int64)[:n_docs])
        print(f"[prepare_data] resuming varlen build: {n_docs:,} docs / {cur:,} tokens")
    else:
        cur, n_docs, docs_seen = 0, 0, 0
        arr = np.memmap(args.output, dtype=TOKEN_DTYPE, mode="w+", shape=(capacity,))
        lens = []

    rows = _make_rows(args, docs_seen, stream_resume)

    def _checkpoint(snapshot):
        # Only called between fully packed batches: every row the reader handed
        # over is on disk, and ``snapshot`` (captured by the reader thread at
        # this batch's boundary) matches ``docs_seen`` exactly.
        arr.flush()
        np.asarray(lens, dtype=np.int64).tofile(lens_path)
        payload = {"identity": identity, "format": "varlen",
                   "cur": cur, "n_docs": n_docs, "docs_seen": docs_seen}
        if snapshot is not None:
            payload["stream"] = snapshot
        tmp = prog + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, prog)
        print(f"packed {n_docs:,} docs / {cur:,}/{capacity:,} tokens "
              f"({docs_seen:,} docs read)")

    def _emit(doc_ids) -> bool:
        """Append one document's chunks (each an example); True once capacity hit."""
        nonlocal cur, n_docs
        for cstart in range(0, len(doc_ids), max_chunk):
            chunk = doc_ids[cstart:cstart + max_chunk]
            seg_len = len(chunk) + 2               # [CLS] + chunk + [SEP]
            if cur + seg_len > capacity:
                return True
            arr[cur] = cls_id
            arr[cur + 1: cur + 1 + len(chunk)] = np.asarray(chunk, dtype=TOKEN_DTYPE)
            arr[cur + 1 + len(chunk)] = sep_id
            cur += seg_len
            lens.append(seg_len)
            n_docs += 1
        return False

    # The reader thread streams + batches while the Rust tokenizer (GIL-free)
    # and the packer run here; batch boundaries match the old inline loop, so
    # the output is byte-identical to a single-threaded build.
    prefetch = _PrefetchBatches(args, rows)
    stop = False
    last_ckpt = n_docs
    tim = _timing_start(docs_seen, cur)
    for batch, snapshot in prefetch:
        t0 = time.perf_counter()
        encoded = tok([t for _, t in batch],
                      add_special_tokens=False, truncation=False)["input_ids"]
        t1 = time.perf_counter()
        tim["tok"] += t1 - t0
        for (idx, _), doc_ids in zip(batch, encoded):
            docs_seen = idx + 1
            # len() not truthiness: gigatoken returns numpy views, where bool()
            # is ambiguous for >1 element and element-wise for exactly 1.
            if len(doc_ids) and _emit(doc_ids):
                stop = True
                break
        tim["pack"] += time.perf_counter() - t1
        if stop:
            break
        if snapshot is not None:
            # The snapshot position also covers trailing holdout-filtered rows
            # the reader consumed after the batch's last kept doc.
            docs_seen = snapshot["pos"]
        # Checkpoint only at batch boundaries: the stream snapshot was captured
        # between cleanly consumed rows and matches docs_seen exactly.
        if n_docs - last_ckpt >= args.flush_every:
            _checkpoint(snapshot)
            _report_throughput(tim, prefetch, docs_seen, cur)
            last_ckpt = n_docs
    prefetch.stop()

    arr.flush()
    trimmed = np.array(arr[:cur])                  # trim the flat stream to real tokens
    del arr
    trimmed.tofile(args.output)
    np.asarray(lens, dtype=np.int64).tofile(lens_path)
    with open(args.output + ".meta", "w", encoding="utf-8") as f:
        json.dump({"format": "varlen", "max_seq_len": args.max_seq_len,
                   "pad_token_id": pad_id, "dtype": TOKEN_DTYPE,
                   "n_docs": n_docs, "total_tokens": int(cur), "docs_seen": docs_seen}, f)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"done: {n_docs:,} docs / {cur:,} tokens (varlen) -> {args.output}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    args = parse_args()

    tok = build_encoder(args.tokenizer,   # torch-free batch encoder (no transformers)
                        backend=args.tokenizer_backend)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    align = args.unpad_align
    # In 'unpad' mode a chunk must leave room for [CLS], [SEP] and up to `align` pad
    # tokens so the aligned segment still fits one block; the other modes fill to seq-2.
    if args.pack_mode == "unpad":
        max_chunk = args.max_seq_len - 2 - align
    else:
        max_chunk = args.max_seq_len - 2      # max doc chunk fitting [CLS] + chunk + [SEP]
    body_cap = args.max_seq_len - 1           # body capacity after [CLS]
    identity = _identity(args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mix = _parse_dataset_mix(args.dataset_mix)
    if mix is not None:
        total_w = sum(s.get("weight", 1.0) for s in mix) or 1.0
        names = ", ".join(
            f'{s["name"]}({s.get("weight", 1.0) / total_w:.0%})' for s in mix)
        print(f"[prepare_data] dataset mix: {names}")

    if args.holdout_mod > 1:
        if args.holdout_keep:
            lo, hi = (int(x) for x in args.holdout_keep.split(":"))
            print(f"[prepare_data] holdout: keep buckets [{lo},{hi}) of "
                  f"{args.holdout_mod} ({hi - lo}/{args.holdout_mod} of the stream)")
        else:
            kept = 1 if args.holdout_role == "holdout" else args.holdout_mod - 1
            print(f"[prepare_data] holdout: role={args.holdout_role} keeps "
                  f"{kept}/{args.holdout_mod} of the stream "
                  f"(bucket {args.holdout_index} is the held-out slice)")
    _pm = {
        "pad": " (one document per block, padded - no cross-document mixing)",
        "unpad": f" (dense packing, each doc segment {args.unpad_align}-aligned so "
                 f"NexteraHRA blocks never straddle documents)",
        "concat": " (documents concatenated per block)",
        "docs": " (one document per example, variable length; padding added dynamically "
                "at collate time)",
    }
    print(f"[prepare_data] pack_mode: {args.pack_mode}{_pm.get(args.pack_mode, '')}")

    # Variable-length build has its own writer (flat tokens + .lens); dispatch BEFORE
    # opening the fixed-block memmap so the two never touch the same file at once.
    if args.pack_mode == "docs":
        _build_varlen(args, tok, cls_id, sep_id, pad_id, identity)
        return                                     # unreachable: _build_varlen os._exit(0)s

    resume = None if args.fresh else _load_progress(args.output, identity)
    if resume is not None:
        written, docs_seen, stream_resume, buf = resume
        arr = np.memmap(args.output, dtype=TOKEN_DTYPE, mode="r+",
                        shape=(args.max_blocks, args.max_seq_len))
        print(f"[prepare_data] resuming from checkpoint: {written:,} blocks "
              f"already packed ({docs_seen:,} docs read).")
    else:
        written, docs_seen = 0, 0
        stream_resume, buf = None, []
        arr = np.memmap(args.output, dtype=TOKEN_DTYPE, mode="w+",
                        shape=(args.max_blocks, args.max_seq_len))

    rows = _make_rows(args, docs_seen, stream_resume)

    def _write_block(ids):
        """Pad ``ids`` to seq_len and write it as the next block."""
        nonlocal written
        pad_len = args.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
        arr[written] = np.asarray(ids, dtype=TOKEN_DTYPE)
        written += 1

    def _checkpoint(snapshot):
        # Only called between fully packed batches: every row the reader handed
        # over is in blocks (or sits in ``buf``, persisted alongside), so
        # ``docs_seen`` and the reader-captured stream snapshot agree exactly.
        arr.flush()
        _save_progress(args.output, identity, written, docs_seen,
                       stream=snapshot, buf=buf)
        print(f"packed {written:,}/{args.max_blocks:,} blocks "
              f"({docs_seen:,} docs read)")

    def _flush_block():
        # concat / pad: a single [CLS] leads the block, docs joined by [SEP].
        nonlocal buf
        _write_block([cls_id] + buf)
        buf = []

    def _flush_unpad():
        # unpad: buf already holds whole [CLS] doc [SEP] <pad> segments (each already
        # a multiple of `align`); write it as-is, tail-padded to seq_len.
        nonlocal buf
        _write_block(buf)
        buf = []

    def _pack_doc(doc_ids) -> bool:
        """Pack one document's ids into blocks; return True once max_blocks hit.

        ``pad``   : each chunk is its own block (``[CLS] chunk [SEP] [PAD]...``) — a
                    sequence never mixes documents, but short docs waste padding.
        ``unpad`` : documents are packed densely, but each ``[CLS] chunk [SEP]``
                    segment is padded up to a multiple of ``align`` so a NexteraHRA
                    pooling block (block_size divides align) never straddles two
                    documents; near-zero wasted padding.
        ``concat``: short documents share a block joined by [SEP] (dense, but they
                    attend across document boundaries).
        """
        nonlocal buf
        for chunk_start in range(0, len(doc_ids), max_chunk):
            if written >= args.max_blocks:
                return True
            chunk = doc_ids[chunk_start:chunk_start + max_chunk]
            if args.pack_mode == "pad":
                buf = list(chunk)                 # one document per block
                buf.append(sep_id)
                _flush_block()                    # emits + pads + resets buf
            elif args.pack_mode == "unpad":
                seg = [cls_id, *chunk, sep_id]
                seg += [pad_id] * (align - (len(seg) % align))   # 1..align pad: 8-align gap
                if len(buf) + len(seg) > args.max_seq_len:
                    if buf:                       # segment can't fit -> flush current block
                        _flush_unpad()
                        if written >= args.max_blocks:
                            return True
                buf.extend(seg)
                if len(buf) == args.max_seq_len:
                    _flush_unpad()
            else:
                needed = len(chunk) + 1           # chunk + [SEP]
                remaining = body_cap - len(buf)
                if needed <= remaining:
                    buf.extend(chunk)
                    buf.append(sep_id)
                else:
                    if buf and written < args.max_blocks:
                        _flush_block()
                    buf = list(chunk)
                    buf.append(sep_id)
            if written >= args.max_blocks:
                return True
        return False

    # The reader thread streams + batches (tokenize_batch docs / char budget,
    # same boundaries as the old inline loop) while the Rust tokenizer
    # (GIL-free) and the packer run here.
    prefetch = _PrefetchBatches(args, rows)
    stop = False
    last_saved = written
    tim = _timing_start(docs_seen, written * args.max_seq_len)
    for batch, snapshot in prefetch:
        t0 = time.perf_counter()
        encoded = tok([t for _, t in batch],
                      add_special_tokens=False, truncation=False)["input_ids"]
        t1 = time.perf_counter()
        tim["tok"] += t1 - t0
        for (idx, _), doc_ids in zip(batch, encoded):
            docs_seen = idx + 1
            # len() not truthiness: gigatoken returns numpy views (see varlen).
            if len(doc_ids) and _pack_doc(doc_ids):
                stop = True
                break
        tim["pack"] += time.perf_counter() - t1
        if stop:
            break
        if snapshot is not None:
            # The snapshot position also covers trailing holdout-filtered rows
            # the reader consumed after the batch's last kept doc.
            docs_seen = snapshot["pos"]
        # Checkpoint only at batch boundaries: the stream snapshot was captured
        # between cleanly consumed rows and matches docs_seen exactly.
        if written - last_saved >= args.flush_every:
            _checkpoint(snapshot)
            _report_throughput(tim, prefetch, docs_seen, written * args.max_seq_len)
            last_saved = written
    prefetch.stop()

    if not stop:
        if buf and written < args.max_blocks:     # final partial block
            _flush_unpad() if args.pack_mode == "unpad" else _flush_block()

    arr.flush()
    if written < args.max_blocks:
        trimmed = np.array(arr[:written])
        del arr
        trimmed.tofile(args.output)

    with open(args.output + ".meta", "w", encoding="utf-8") as f:
        json.dump({"max_seq_len": args.max_seq_len, "pad_token_id": pad_id,
                   "dtype": TOKEN_DTYPE, "docs_seen": docs_seen}, f)

    # the build finished cleanly — drop the resume checkpoint
    prog = _progress_path(args.output)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"done: {written:,} blocks of {args.max_seq_len} tokens -> {args.output}")

    # Everything (memmap, .meta) is flushed to disk above. Some native extensions —
    # HF datasets' streaming prefetch threads, the Rust tokenizer's Rayon pool,
    # pyarrow — leave daemon threads running that touch the GIL during interpreter
    # shutdown and crash with "PyGILState_Release: thread state must be current",
    # turning a fully successful build into a non-zero exit (which build_data.sh
    # would treat as a failure and delete the finished shard). Skip the fragile
    # finalization entirely with a hard exit now that the output is safely written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
