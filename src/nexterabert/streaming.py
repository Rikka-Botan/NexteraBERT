"""Torch-free streaming + tokenizer helpers shared by training and data prep.

These utilities deliberately import **no torch** (and nothing that pulls it in), so
``scripts/prepare_data.py`` can pre-tokenise a corpus without loading torch/CUDA into
the process. That matters: torch spawns background worker threads that crash the
interpreter at shutdown ("PyGILState_Release: thread state must be current when
releasing" during finalization) on some CUDA boxes, turning a finished build into a
non-zero exit. A pure-CPU tokenisation job has no reason to touch torch, so it does
not import it. ``nexterabert.data`` re-exports these for the training-time loaders.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from itertools import islice

DEFAULT_TOKENIZER = "answerdotai/ModernBERT-base"

# Class names / top-level modules that signal a *transient* network hiccup while
# streaming shards from the Hub (HTTP 408/5xx, SSL handshake timeout, dropped
# connection, ...). These should pause-and-retry, not crash a multi-day run.
_TRANSIENT_ERR_NAMES = frozenset({
    "HfHubHTTPError", "HTTPStatusError", "HTTPError", "ConnectionError",
    "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError", "PoolTimeout",
    "RemoteProtocolError", "RemoteDisconnected", "ProtocolError", "SSLError",
    "IncompleteRead", "ChunkedEncodingError", "Timeout", "TimeoutError",
})
_TRANSIENT_ERR_MODULES = frozenset({
    "httpx", "httpcore", "requests", "urllib3", "huggingface_hub", "fsspec",
    "aiohttp", "ssl", "socket",
})

_TRANSIENT_MSG_FRAGMENTS = (
    "client has been closed",
    "connection reset",
    "broken pipe",
)


def build_tokenizer(name: str = DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def padded_vocab_size(tokenizer, multiple: int = 64) -> int:
    """``len(tokenizer)`` rounded up to a multiple of ``multiple``.

    Embedding / lm_head GEMMs hit aligned tile sizes when the vocab dimension is
    a multiple of 64 (ModernBERT does the same: 50,368 = 64*787 with 83 unused
    ids). The padded rows correspond to no real token, so every path that
    *samples* token ids must stay capped to ``len(tokenizer)`` — the trainers
    take that bound separately as ``true_vocab_size``.
    """
    return ((len(tokenizer) + multiple - 1) // multiple) * multiple


# Candidate special-token surface forms, most specific first, covering WordPiece
# ([CLS]/[SEP]/[PAD]) and BPE (<s>/</s>/<pad>) tokenizers.
_CLS_CANDIDATES = ("[CLS]", "<s>", "<cls>", "<CLS>")
_SEP_CANDIDATES = ("[SEP]", "</s>", "<sep>", "<SEP>")
_PAD_CANDIDATES = ("[PAD]", "<pad>", "<PAD>")


def _first_token_id(tk, candidates):
    for tokn in candidates:
        tid = tk.token_to_id(tokn)
        if tid is not None:
            return tid
    return None


class _PackEncoder:
    """Minimal, **torch-free** batch encoder for the pre-tokenisation script.

    Wraps a raw ``tokenizers.Tokenizer`` (the same Rust engine the fast HF tokenizer
    uses) and exposes only what ``scripts/prepare_data.py`` needs. Critically it does
    NOT import ``transformers`` — in transformers 5.x, importing ``AutoTokenizer``
    eagerly imports torch, whose worker threads can crash the interpreter at shutdown
    ("PyGILState_Release ... during finalization") on some CUDA boxes. Packing is
    pure CPU, so it has no business loading torch.
    """

    def __init__(self, tk):
        self._tk = tk
        tk.no_truncation()   # we chunk long docs ourselves; never truncate here
        tk.no_padding()      # we pad packed blocks ourselves
        self.cls_token_id = _first_token_id(tk, _CLS_CANDIDATES)
        self.sep_token_id = _first_token_id(tk, _SEP_CANDIDATES)
        self.pad_token_id = _first_token_id(tk, _PAD_CANDIDATES)
        self.model_max_length = 10 ** 12   # accepted + ignored (API compatibility)

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        encs = self._tk.encode_batch(texts, add_special_tokens=add_special_tokens)
        return {"input_ids": [e.ids for e in encs]}


class _GigatokenEncoder:
    """Batch encoder backed by gigatoken (Rust SIMD BPE), ~50x ``tokenizers``.

    Exposes the same interface as :class:`_PackEncoder`. Ids are returned as
    zero-copy numpy views into the ragged batch (materialising Python lists
    would eat most of the speedup). Output parity with the ``tokenizers``
    backend is verified by :func:`build_encoder` before this is ever used, so
    shards are byte-identical no matter which backend encoded them.
    """

    def __init__(self, g, base: _PackEncoder):
        self._g = g
        self.cls_token_id = base.cls_token_id
        self.sep_token_id = base.sep_token_id
        self.pad_token_id = base.pad_token_id
        self.model_max_length = base.model_max_length

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        import awkward as ak
        import numpy as np

        enc = self._g.encode_batch(texts)
        counts = ak.to_numpy(ak.num(enc))
        flat = ak.to_numpy(ak.flatten(enc))
        offs = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offs[1:])
        return {"input_ids": [flat[offs[i]:offs[i + 1]]
                              for i in range(len(counts))]}


# Startup parity suite for the fast backend: pretokenizer-regex and added-token
# edge cases (specials appearing literally in text, CJK, emoji, whitespace
# runs, empty docs). Divergence on any sample rejects the backend.
_PARITY_SAMPLES = (
    "",
    " ",
    "   \t\n\n  ",
    "Hello world",
    "def f(x):\n    return x + 1  # comment\n",
    "こんにちは世界。BPE境界の確認。",
    "\U0001f600\U0001f680 emoji mix \U0001f1ef\U0001f1f5",
    "[CLS] literal special tokens in text [SEP] [PAD] [MASK]",
    "<|endoftext|> and <s> and </s>",
    "éé combining vs precomposed",
    "1234567890 3.14159 1e-9 0xDEADBEEF",
    "a" * 10000,
    ("word " * 2000).strip(),
    "中文测试" * 200,
    "\r\n\r\nwindows line endings\r\n",
    "tab\tsep\tvalues\tand   runs    of spaces",
)


def _verify_encoder_parity(ref, fast):
    a = ref(list(_PARITY_SAMPLES))["input_ids"]
    b = fast(list(_PARITY_SAMPLES))["input_ids"]
    for i, (x, y) in enumerate(zip(a, b)):
        if list(map(int, x)) != list(map(int, y)):
            raise ValueError(f"ids diverge from the tokenizers backend on "
                             f"parity sample {i}")


def build_encoder(name: str = DEFAULT_TOKENIZER, backend: str = "auto"):
    """Return a torch-free batch encoder for data packing.

    Use this instead of :func:`build_tokenizer` wherever only batch encoding + the
    CLS/SEP/PAD ids are needed and torch must stay out of the process.

    ``backend``: ``"tokenizers"`` forces the HF Rust engine; ``"gigatoken"``
    requires the ~50x gigatoken engine (raises if missing or failing the parity
    suite); ``"auto"`` (default) tries gigatoken and silently falls back.
    """
    from tokenizers import Tokenizer

    tk = Tokenizer.from_pretrained(name)
    enc = _PackEncoder(tk)
    if enc.cls_token_id is None or enc.sep_token_id is None:
        raise ValueError(
            f"tokenizer {name!r} has no recognisable CLS/SEP token "
            f"(looked for {_CLS_CANDIDATES} / {_SEP_CANDIDATES}); packing needs both.")
    if backend == "tokenizers":
        return enc
    try:
        import gigatoken

        fast = _GigatokenEncoder(gigatoken.Tokenizer(name), enc)
        _verify_encoder_parity(enc, fast)
    except Exception as err:  # noqa: BLE001 - any failure falls back cleanly
        if backend == "gigatoken":
            raise
        print(f"[NexteraBERT] gigatoken backend unavailable "
              f"({type(err).__name__}: {err}); using tokenizers", flush=True)
        return enc
    print("[NexteraBERT] tokenizer backend: gigatoken "
          "(parity with tokenizers verified)", flush=True)
    return fast


def _is_transient_network_error(err: BaseException) -> bool:
    """True if ``err`` (or anything in its cause/context chain) looks like a
    transient network/Hub error worth retrying rather than crashing on."""
    seen: set[int] = set()
    e: BaseException | None = err
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, (OSError, TimeoutError)):
            return True
        if type(e).__name__ in _TRANSIENT_ERR_NAMES:
            return True
        if type(e).__module__.split(".")[0] in _TRANSIENT_ERR_MODULES:
            return True
        msg = str(e).lower()
        if any(frag in msg for frag in _TRANSIENT_MSG_FRAGMENTS):
            return True
        e = e.__cause__ or e.__context__
    return False


def _resilient_rows(make_stream, max_retries: int = 50, base_wait: float = 2.0):
    """Yield rows from a (re-creatable) 🤗 streaming dataset, rebuilding the
    stream and continuing on transient network errors instead of propagating
    them (which would kill the DataLoader worker and the whole run).

    ``make_stream`` is a zero-arg callable returning a fresh iterable. On a
    transient failure the stream is rebuilt (restarting from the top - fine for a
    huge, shuffled pretraining corpus) after an exponential back-off. A
    non-transient error, or more than ``max_retries`` consecutive failures, is
    re-raised.
    """
    consecutive = 0
    while True:
        stream = make_stream()
        try:
            for row in stream:
                consecutive = 0
                yield row
            return
        except Exception as err:  # noqa: BLE001 - classified below
            if not _is_transient_network_error(err) or consecutive >= max_retries:
                raise
            consecutive += 1
            wait = min(base_wait * (2 ** (consecutive - 1)), 120.0)
            print(f"[NexteraBERT] transient streaming error "
                  f"({type(err).__name__}); retrying in {wait:.0f}s "
                  f"({consecutive}/{max_retries})", flush=True)
            time.sleep(wait)


class ResilientRows:
    """Yield ``(index, row)`` from a re-creatable 🤗 streaming dataset, resuming.

    Unlike :func:`_resilient_rows` (which restarts from the top - fine for a huge,
    shuffled *training* stream), this preserves position across both transient
    network errors and process restarts, so an offline pre-tokenisation pass
    (``scripts/prepare_data.py``) can run to completion despite Hub/SSL hiccups
    and resume from a checkpoint without re-streaming the whole corpus.

    Position is restored with the dataset's native ``state_dict`` /
    ``load_state_dict``: the state records shard index + in-shard offset, so a
    resume re-reads at most the current shard instead of re-downloading every
    earlier row the way ``.skip(n)`` does (hours of silent fast-forward on a
    multi-billion-token build). :meth:`state` returns a JSON-safe snapshot taken
    *between cleanly yielded rows* - the only moment the state is guaranteed
    exact - for the caller's progress checkpoint; pass its fields back as
    ``start_index`` / ``stream_state`` / ``base_skip`` to resume.

    ``make_stream`` is a one-arg callable ``make_stream(skip) -> IterableDataset``
    returning a fresh stream with the first ``skip`` rows dropped via the
    dataset's native ``.skip``. The skip path is used for legacy checkpoints
    (no stream state recorded) and as a fallback when state save/load is
    unavailable. State-based rebuilds always call ``make_stream(base_skip)``
    with the SAME skip the state was captured under: ``.skip`` inserts a
    ``SkipExamplesIterable`` node whose entry appears in the state dict, so the
    pipeline must be built identically for the structures to line up.
    """

    def __init__(self, make_stream, start_index: int = 0, stream_state=None,
                 base_skip: int | None = None, max_retries: int = 50,
                 base_wait: float = 2.0, state_every: int = 1000):
        self._make_stream = make_stream
        self._pos = start_index          # global index of the next row to yield
        self._max_retries = max_retries
        self._base_wait = base_wait
        self._state_every = state_every
        # skip the CURRENT pipeline was built with (fixes the state-dict shape)
        self._skip = start_index if base_skip is None else base_skip
        self._state = stream_state       # ds.state_dict() as of row ``_state_pos``
        self._state_pos = start_index
        self._ds = None
        self._can_state = True           # flips off if the stream can't state_dict

    # -- state capture / export ------------------------------------------------

    def _capture(self):
        """Snapshot the dataset state; only called between cleanly yielded rows
        (mid-error state may sit at a partially consumed batch and is unsafe)."""
        if not (self._can_state and self._ds is not None):
            return
        try:
            state = self._ds.state_dict()
            json.dumps(state)            # progress checkpoints are JSON: verify now
        except Exception as err:  # noqa: BLE001 - degrade to skip-based resume
            print(f"[NexteraBERT] stream state_dict unavailable "
                  f"({type(err).__name__}: {err}); falling back to slow "
                  f".skip() resume", flush=True)
            self._can_state = False
            return
        self._state = state
        self._state_pos = self._pos

    def state(self):
        """JSON-safe resume snapshot as of the last yielded row, or ``None``.

        ``pos`` is the global index of the next row, ``skip`` the ``make_stream``
        argument the state's pipeline was built with, ``state`` the dataset's own
        state dict. ``datasets_version`` guards against loading a state into a
        differently-shaped pipeline after a library upgrade.
        """
        self._capture()
        if not self._can_state or self._state is None:
            return None
        import datasets
        return {"pos": self._state_pos, "skip": self._skip,
                "datasets_version": datasets.__version__,
                "state": deepcopy(self._state)}

    # -- iteration -------------------------------------------------------------

    def _build(self):
        """(Re)build the stream and return ``(iterator, rows_to_discard)``."""
        if self._state is not None and self._can_state:
            ds = self._make_stream(self._skip)
            try:
                ds.load_state_dict(deepcopy(self._state))
            except Exception as err:  # noqa: BLE001 - shape mismatch and kin
                print(f"[NexteraBERT] stream load_state_dict failed "
                      f"({type(err).__name__}: {err}); falling back to slow "
                      f".skip({self._pos}) fast-forward", flush=True)
                self._state = None
                self._can_state = False
            else:
                self._ds = ds
                return iter(ds), self._pos - self._state_pos
        # Legacy / fallback: native .skip() fast-forward - slow (re-streams every
        # earlier row) but exact. Recording the skip keeps later state captures
        # loadable: they must be restored into a pipeline built with the same skip.
        self._skip = self._pos
        ds = self._make_stream(self._pos)
        has_state = hasattr(ds, "state_dict") and hasattr(ds, "load_state_dict")
        self._ds = ds if has_state else None
        if not has_state:
            self._can_state = False
        return iter(ds), 0

    def __iter__(self):
        consecutive = 0
        while True:
            try:
                it, discard = self._build()
                if discard:
                    # Fast-forward from the state's position (<= state_every rows
                    # behind) back to the current one after an error rebuild.
                    if sum(1 for _ in islice(it, discard)) < discard:
                        return   # stream ended inside the fast-forward window
                for row in it:
                    consecutive = 0
                    idx = self._pos
                    self._pos += 1
                    if (self._can_state
                            and self._pos - self._state_pos >= self._state_every):
                        self._capture()
                    yield idx, row
                return
            except Exception as err:  # noqa: BLE001 - classified below
                if not _is_transient_network_error(err) or consecutive >= self._max_retries:
                    raise
                consecutive += 1
                wait = min(self._base_wait * (2 ** (consecutive - 1)), 120.0)
                print(f"[NexteraBERT] transient streaming error "
                      f"({type(err).__name__}) at row {self._pos}; resuming in "
                      f"{wait:.0f}s ({consecutive}/{self._max_retries})", flush=True)
                time.sleep(wait)


# Backward-compatible alias: existing call sites iterate the object directly, and
# the old generator signature is a strict subset of the constructor's.
iter_rows_resilient = ResilientRows
