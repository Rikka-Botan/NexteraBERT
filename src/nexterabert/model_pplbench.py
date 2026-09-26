"""Masked-LM (pseudo-)perplexity vs sequence length, NexteraBERT vs the field.

The sibling ``model_speedbench.py`` plots *inference time* against sequence length
for a fresh (untrained) model; this plots **pseudo-perplexity** against sequence
length for the *trained* NexteraBERT next to ModernBERT / NeoBERT / LFM2.5-Encoder,
so the long-context quality picture lines up with the speed picture. The plot
styling (pastel palette, figure size, markers, grid) matches ``model_speedbench``.

Metric — pseudo-perplexity (PPPL). A bidirectional encoder has no autoregressive
factorisation, so ordinary perplexity is undefined. Following Salazar et al. (2020,
"Masked Language Model Scoring") we score each model by its masked-LM loss: a fixed
fraction (``--mask_ratio``, 15% by default) of a contiguous window is replaced with
``[MASK]``, the model predicts the originals, and ``exp(mean cross-entropy over the
masked positions)`` is the pseudo-perplexity. Exact PPPL masks one token at a time
(``L`` forwards per window) — far too costly at ``L=16384`` — so this is the standard
one-pass subsampled estimator (identical in form to the MLM training loss), summed
over many windows for a stable mean. Each window is framed as ``[CLS] body [SEP]``
(how all of these encoders were pretrained; ``--no_special_tokens`` opts out) and the
framing tokens are excluded from the target set, matching the trainers.

Masked-LM accuracy rides along for free: the same forward that yields the cross-entropy
also yields the top-1 prediction at each masked position, so every measurement records
``acc`` (fraction of masked tokens recovered exactly) next to its PPPL. Being bounded,
it stays readable where PPPL does not -- a model past its context wall reads ~0% rather
than 1e5 -- but it is per-token like PPPL, so the cross-tokenizer caveat below applies
to it equally. Written to ``<output>_acc.png`` and to ``acc_mean`` / ``acc_std`` /
``acc_runs`` in the JSON.

Every length is measured with ``--seeds`` independent draws (default 25) over
**disjoint** corpus slices, and reported as ``mean +/- sample std``; the plot draws the
std as error bars and ``--json`` records every individual value alongside the summary,
so a published number can be traced back to the run that produced it.

The JSON is rewritten (atomically) the moment each model finishes, not once at the end,
so a crash or a Ctrl-C costs only the model in flight; models the run has not reached
yet keep whatever points they already had.

The JSON doubles as the resume log: on a rerun, any ``(model, length)`` already
recorded there is reused instead of re-measured, a model whose lengths are *all*
recorded is never even loaded, and the corpus is only streamed if something actually
needs computing. So adding one model or one length to a finished sweep costs just that
model or that length. The recorded settings must match the current ones -- change
``--mask_ratio``, ``--seeds`` or the corpus and the whole file is ignored with a
printed reason, because all of those move the numbers. ``--refresh``
re-measures regardless. Only a *finite* recorded value counts: a null is either a
length the model cannot run (deciding that again is free) or a failure worth retrying.

Baselines. ``--models`` takes short aliases as well as raw Hub ids::

    modernbert  answerdotai/ModernBERT-base            RoPE, 8192 context
    neobert     chandar-lab/NeoBERT                    RoPE, 4096 context
    lfm         LiquidAI/LFM2.5-Encoder-230M           RoPE, 128k context
    nextera     RikkaBotan/NexteraBERT-Mezzoforte-220M-en   (or --nextera_path)

Windows are framed the way each tokenizer itself frames text, discovered by diffing an
encoding with and without special tokens rather than assumed from ``cls_token_id``.
That is what gets LFM2.5-Encoder right: it prepends ``<|startoftext|>`` and appends
*nothing*, so pairing its ``bos`` with its ``eos`` would have appended ``<|im_end|>`` —
a chat-template marker the encoder never sees in that position.

What the sweep shows. Each contiguous window of length ``L`` is drawn from one shared
corpus (FineWeb-Edu by default — the training distribution) tokenised with *each
model's own* tokenizer, and every model reads the complete window: no chunking and no
extended position tables. Longer windows give the encoder more bidirectional context,
so PPPL falls with ``L`` until a model hits its context wall.

**Every model is attempted at every requested length.** Nothing is skipped up front on
the strength of a config number: RoPE models (ModernBERT to 8192, NeoBERT to 4096,
LFM2.5-Encoder, NexteraBERT) have no position table to run out of, and
``max_position_embeddings`` is their *trained* context, not a wall -- beyond it they
extrapolate, and how gracefully is the informative part of the picture. A length is
only absent when the model actually raised, and the error is printed when it does.

The one exception is a model that looks positions up in a learned absolute table:
an out-of-range lookup raises a catchable ``RuntimeError`` on CPU but fires a
*device-side assert* on CUDA, which poisons the context and takes the rest of the
sweep down with it. Lengths past the table's real reach are therefore skipped.

Caveats (printed at runtime too):
  * Cross-tokenizer PPPL is only *roughly* comparable — a 30k WordPiece vocab and a
    128k SentencePiece vocab put different amounts of information in a "token", so the
    per-token numbers are not on an identical scale. Read the *trends and context
    walls*, not hairline gaps.
  * **Whether a checkpoint really has a trained MLM head is checked, not assumed.**
    ``AutoModelForMaskedLM.from_pretrained`` never fails over a missing output head —
    it quietly builds a random one — so the loader inspects the newly-initialised keys
    it reports; such a line is marked ``(!)`` in the table and
    ``"mlm_head_loaded": false`` in the JSON.
  * NeoBERT's remote code imports xformers for its SwiGLU feed-forward; where
    xformers has no wheel (e.g. Windows / CPU-only) an equivalent eager stand-in is
    registered automatically so the checkpoint still loads. Its RoPE table is also
    pre-sized at construction to cover the longest requested length (the released
    config bakes it to 4096).
  * PPL needs *trained* weights. By default the NexteraBERT line is the published
    ``RikkaBotan/NexteraBERT-Mezzoforte-220M-en`` (encoder + ``mlm_head.safetensors``
    + tokenizer, snapshotted into the HF cache -- a failed download leaves the line
    out and says why, it never substitutes another model). ``--nextera_path`` also
    takes a local backbone directory. Only a local path that does not exist falls back
    to an untrained preset, whose line is meaningless (loud warning). The JSON records
    which weights were scored, and points recorded for different ones are re-measured.
  * **The exported backbone is the encoder only.** ``save_pretrained`` writes
    ``model.safetensors`` from ``_encoder_state_dict``, so the MLM output head — the
    ModernBERT-style prediction transform (dense + LayerNorm) and the untied decoder
    bias — is *not* in it. Reconstructing ``NexteraBERTForMaskedLM`` from the backbone
    alone therefore leaves that transform at its random init, which scrambles the vocab
    projection and reports a pseudo-perplexity in the thousands for a perfectly healthy
    model. The head is recovered from the full training checkpoint
    (``<run_dir>/final.pt`` or the newest ``step_*.pt``, found automatically next to the
    backbone, or passed with ``--nextera_ckpt``); newer exports also drop a
    ``mlm_head.safetensors`` inside the backbone directory, which is preferred when
    present. A loud warning fires when neither is found.

Examples
--------
    # the full comparison (ModernBERT / NeoBERT / LFM2.5-Encoder + ours, ours = the
    # published Hub weights; 1,024 to 65,536 tokens)
    python src/nexterabert/model_pplbench.py \
        --output ppl_benchmark.png            # writes ppl_benchmark.json too

    # score a local backbone instead of the Hub weights
    python src/nexterabert/model_pplbench.py \
        --nextera_path checkpoints/mezzoforte_bert/phase2/backbone

    # quick smoke test: one baseline, short windows, fewer seeds
    python src/nexterabert/model_pplbench.py --max_seqs 8 --seeds 2 \
        --models modernbert nextera --seq_lengths 128 256 512 1024
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Force UTF-8 stdout/stderr so progress logs never crash on a legacy Windows console
# (the default cp932 code page can't encode the arrows / em-dashes this script prints).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # non-reconfigurable stream (e.g. redirected)
        pass

# This file lives INSIDE the package (src/nexterabert/), so put the `src` root on
# the path and import through the `nexterabert` package — that keeps the relative
# imports inside loading.py / streaming.py working (unlike model_speedbench's bare
# `from modeling_nexterabert import ...`, which only resolves from this directory).
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from nexterabert.configuration_nexterabert import NexteraBERTConfig  # noqa: E402
from nexterabert.loading import load_backbone_state  # noqa: E402
from nexterabert.modeling_nexterabert import NexteraBERTForMaskedLM  # noqa: E402
from nexterabert.streaming import (  # noqa: E402
    DEFAULT_TOKENIZER,
    build_tokenizer,
    padded_vocab_size,
)

import torch._dynamo  # noqa: E402

torch._dynamo.config.suppress_errors = True

# ==============================
# 設定 (mirrors model_speedbench.py)
# ==============================
NEXTERA_NAME = "NexteraBERT (Ours)"

# The masked-LM encoders of the comparison, by short alias. ``--models modernbert
# neobert ...`` resolves through here; any raw Hub id still works unchanged.
MODEL_ALIASES = {
    "nextera": NEXTERA_NAME,
    "modernbert": "answerdotai/ModernBERT-base",
    "neobert": "chandar-lab/NeoBERT",
    "lfm": "LiquidAI/LFM2.5-Encoder-230M",
}

# Legend / table labels.
DISPLAY_NAMES = {
    "answerdotai/ModernBERT-base": "ModernBERT-base",
    "chandar-lab/NeoBERT": "NeoBERT",
    "LiquidAI/LFM2.5-Encoder-230M": "LFM2.5-Encoder-230M",
}

DEFAULT_MODELS = ["modernbert", "neobert", "lfm", "nextera"]


def resolve_model(name):
    """Map a short alias to a Hub id; pass anything else through untouched."""
    return MODEL_ALIASES.get(name.lower(), name)


def display_name(name):
    return DISPLAY_NAMES.get(name, name.split("/")[-1] if "/" in name else name)

DEFAULT_SEQUENCE_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

# The published NexteraBERT weights scored when --nextera_path is not given
# (config.json + model.safetensors + mlm_head.safetensors + tokenizer). A local
# backbone directory still works.
DEFAULT_NEXTERA_REPO = "RikkaBotan/NexteraBERT-Mezzoforte-220M-en"

# Only what the benchmark reads -- a repo also carries source modules, a README and
# possibly sentence-transformers folders that are of no use here.
_NEXTERA_HUB_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def looks_like_repo_id(path):
    """``org/name`` that is not a directory on disk. A relative checkpoint path such as
    ``checkpoints/run/phase2/backbone`` has more than one separator and never matches,
    so a mistyped local path still gets the 'no checkpoint found' treatment instead of
    a confusing Hub 404."""
    import re

    return bool(path) and not os.path.exists(path) and \
        re.fullmatch(r"[\w.\-]+/[\w.\-]+", path) is not None


def resolve_nextera_path(path):
    """A local directory for ``path``: itself, or a snapshot of the Hub repo it names.

    The repo is private by default (``HF_PRIVATE=true`` in the pipeline), so the usual
    failure is authentication, and the Hub reports a private repo the token cannot
    read as *404 Not Found* rather than 403 -- say so, or it reads as a typo."""
    if not looks_like_repo_id(path):
        return path
    from huggingface_hub import snapshot_download

    print(f"[ppl] NexteraBERT: fetching {path} from the Hub ...")
    try:
        return snapshot_download(path, allow_patterns=_NEXTERA_HUB_PATTERNS)
    except Exception as e:  # noqa: BLE001 - re-raised with the likely cause attached
        raise RuntimeError(
            f"could not download {path!r} from the Hub ({type(e).__name__}). A private "
            f"repo answers 404 to a token without read access to it, so check "
            f"`huggingface-cli whoami` / HF_TOKEN (a fine-grained token needs read "
            f"access to this repo) and that the upload has run. Or pass "
            f"--nextera_path <local backbone dir>.") from e

# Ten distinct hues, because the sweep now compares seven-plus models and a
# five-colour palette silently gave BERT and LFM the same blue and ELECTRA and
# NexteraBERT the same pink -- two pairs of lines that could not be told apart.
# Markers and line styles cycle on a different period as a second cue, so the
# figure still reads in greyscale or to a colour-blind reader.
pastel_colors = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#B279A2",  # purple
    "#72B7B2",  # teal
    "#EECA3B",  # yellow
    "#FF9DA6",  # pink
    "#9D755D",  # brown
    "#8C8C8C",  # grey
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def series_style(index, name):
    """Colour / marker / width for one model. NexteraBERT is always the thick black
    line so the eye finds it first, whatever else is in the comparison."""
    if name == NEXTERA_NAME:
        return {"color": "#111111", "marker": "o", "linewidth": 2.6,
                "markersize": 7, "zorder": 5}
    return {"color": pastel_colors[index % len(pastel_colors)],
            "marker": MARKERS[index % len(MARKERS)], "linewidth": 1.6,
            "markersize": 5, "zorder": 2}

torch.backends.cudnn.benchmark = True


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ==============================
# コーパス
# ==============================
def collect_corpus_text(dataset, config, split, text_column, char_budget):
    """Stream ``dataset`` ONCE and return a list of text strings covering
    ``char_budget`` characters. Streaming keeps the download to a few hundred KB
    even for a huge corpus like FineWeb-Edu. The raw text is cached and re-tokenised
    per model (each family uses its own tokenizer), so the Hub is hit only once."""
    from datasets import load_dataset

    print(f"[ppl] streaming corpus {dataset}"
          f"{'/' + config if config else ''}[{split}] "
          f"(~{char_budget/1e6:.1f}M chars)...")
    ds = load_dataset(dataset, config, split=split, streaming=True)
    texts, total = [], 0
    for row in ds:
        text = row.get(text_column)
        if not text or not text.strip():
            continue
        texts.append(text)
        total += len(text)
        if total >= char_budget:
            break
    if not texts:
        raise RuntimeError(
            f"no text collected from {dataset}[{split}] (column {text_column!r}); "
            "check --dataset / --text_column.")
    print(f"[ppl] collected {len(texts)} documents (~{total/1e6:.1f}M chars)")
    return texts


def tokenize_concat(tokenizer, texts, n_tokens_needed):
    """Concatenate ``texts`` into one flat id stream (each doc bookended by the
    tokenizer's separator so window boundaries stay clean), stopping once
    ``n_tokens_needed`` ids are available. One stream per tokenizer — vocabularies
    differ, so BERT WordPiece and RoBERTa BPE produce different-length streams over
    the same text."""
    sep = tokenizer.sep_token_id
    if sep is None:
        sep = tokenizer.eos_token_id
    # Per-document tensors, not one flat Python list: at the default 25 seeds the
    # stream reaches ~50M ids, which is ~2 GB as list-of-int and 0.4 GB as int64.
    chunks: list[torch.Tensor] = []
    total = 0
    for text in texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if not toks:
            continue
        if sep is not None:
            toks = list(toks) + [sep]
        chunks.append(torch.tensor(toks, dtype=torch.long))
        total += len(toks)
        if total >= n_tokens_needed:
            break
    if not chunks:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(chunks)[:n_tokens_needed]


# ==============================
# モデルのロード
# ==============================
def hf_max_len(config):
    """Best-effort maximum sequence length a HF model supports, or None (uncapped).
    Values under 128 are ignored — those are generation defaults (e.g. the legacy
    ``max_length=20`` some configs still carry), not positional caps."""
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "max_length"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and 128 <= v <= 1_000_000:
            return v
    return None


def uses_rope(config, model=None):
    """True when the model's positions are computed, not looked up in a fixed table.

    For such models ``max_position_embeddings`` is the *trained* context, not a hard
    limit — longer inputs run fine, they merely extrapolate. Two families qualify:

      * **RoPE.** Custom (trust_remote_code) configs name their RoPE fields freely, so
        beyond the well-known attribute names any config key containing
        "rope"/"rotary" counts; some (NeoBERT) advertise nothing at all in their
        config, so as a last resort look for a rotary buffer (``freqs_cis`` /
        ``inv_freq``) on the instantiated model.
      * **Disentangled relative attention (DeBERTa).** ``relative_attention=True``
        with ``position_biased_input=False`` means the encoder never adds an absolute
        position embedding at all — position enters only as relative buckets. DeBERTa
        v1 and v3 both run at 1024+ despite advertising
        ``max_position_embeddings=512``, so treating that number as a wall silently
        truncated the DeBERTa line at 512 for no reason."""
    if getattr(config, "relative_attention", False) and \
            getattr(config, "position_biased_input", True) is False:
        return True
    for attr in ("rope_parameters", "rope_scaling", "rope_theta",
                 "global_rope_theta", "rotary_emb_base"):
        if getattr(config, attr, None):
            return True
    try:
        cfg = config.to_dict()
    except Exception:  # noqa: BLE001 - exotic custom config; the attrs above had to do
        cfg = {}
    if any(("rope" in k.lower() or "rotary" in k.lower()) and bool(v)
           for k, v in cfg.items()):
        return True
    if model is not None:
        return any("freqs_cis" in n or "rotary" in n or "inv_freq" in n
                   for n, _ in model.named_buffers())
    return False


def _position_embedding_module(model):
    """The submodule holding the learned absolute position table, or ``(None, None)``.

    ``bert.embeddings`` / ``electra.embeddings`` / ``roberta.embeddings`` all expose it
    as ``.position_embeddings``; looking for the attribute rather than hard-coding the
    path covers every ``*Embeddings`` class shaped like BERT's.
    """
    for name, mod in model.named_modules():
        emb = getattr(mod, "position_embeddings", None)
        if isinstance(emb, torch.nn.Embedding):
            return name, mod
    return None, None


def usable_position_len(model):
    """The longest sequence this model can actually index, or ``None``.

    Not the same as ``config.max_position_embeddings``: RoBERTa declares 514 but
    numbers real positions from ``padding_idx + 1 == 2``, so only 512 are reachable.
    Feeding it 514 tokens indexes row 515 and, on CUDA, fires a device-side assert
    that kills the process rather than raising something catchable -- so the real
    figure is what every cap decision has to use.
    """
    _, mod = _position_embedding_module(model)
    if mod is None:
        return None
    emb = mod.position_embeddings
    pad = emb.padding_idx
    return emb.num_embeddings - ((pad + 1) if pad is not None else 0)


def _ensure_xformers_swiglu():
    """NeoBERT's remote code needs ``xformers.ops.SwiGLU``; register an eager
    stand-in when xformers is missing. Shared with the MTEB baseline path --
    see ``nexterabert.hf_baselines.ensure_xformers_swiglu``."""
    from nexterabert.hf_baselines import ensure_xformers_swiglu

    ensure_xformers_swiglu()


# Parameter-name fragments that belong to an MLM *output head* rather than the
# encoder. If transformers reports one of these as newly initialised, the checkpoint
# did not supply a trained head and whatever perplexity comes out is noise.
_HEAD_KEY_HINTS = ("predictions", "lm_head", "mlm", "cls.", "decoder.bias")


def _head_keys(keys):
    return [k for k in keys if any(h in k.lower() for h in _HEAD_KEY_HINTS)]


def _check_head(model_name, loading_info):
    """Did this checkpoint actually supply a trained MLM head?

    ``from_pretrained`` never fails over a missing output head -- it silently builds a
    random one -- so the only reliable signal is the list of newly-initialised keys.
    Returns True when the head is real.
    """
    missing = _head_keys(loading_info.get("missing_keys", []) or [])
    if not missing:
        return True
    print(f"[ppl] [WARNING] {model_name}: the checkpoint supplies NO trained MLM head "
          f"({len(missing)} head params newly initialised, e.g. {missing[:3]}). Its "
          f"perplexity is NOT meaningful -- this is a feature-extraction checkpoint, "
          f"not a masked LM.")
    return False


def load_hf_masked_lm(model_name, device, target_len):
    """Load a 🤗 ``AutoModelForMaskedLM`` + tokenizer.

    Returns ``(tokenizer, model, forward, max_len, hard_cap, head_ok)``: ``max_len``
    is the config's positional cap (None if uncapped) and ``hard_cap`` says whether
    exceeding it is impossible (learned absolute positions) or merely extrapolation
    (RoPE). ``target_len`` is the longest length the sweep will request — needed by
    models that pre-size a RoPE cache at construction time (NeoBERT)."""
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    is_neobert = "neobert" in model_name.lower()
    if is_neobert:
        _ensure_xformers_swiglu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    max_len = hf_max_len(config)   # the *trained* cap, before any override below
    if is_neobert and max_len is not None and target_len > max_len:
        # NeoBERT bakes its RoPE table to config.max_length in __init__ (a
        # non-persistent buffer, absent from the checkpoint), so request a table
        # covering the whole sweep up front; the trained weights are unaffected.
        config.max_length = int(target_len)
    model, loading_info = AutoModelForMaskedLM.from_pretrained(
        model_name, config=config, trust_remote_code=True,
        output_loading_info=True)
    head_ok = _check_head(model_name, loading_info)
    model = model.to(device).eval()
    if is_neobert:
        # transformers>=5 materialises models via the meta device, which leaves
        # computed non-persistent buffers (NeoBERT's RoPE table ``freqs_cis``)
        # ZERO-FILLED instead of running the __init__ computation -- position info
        # is silently destroyed and PPL lands in the tens of thousands. Recompute
        # the table post-load with the remote module's own function, sized to
        # config.max_length (already bumped above to cover the whole sweep).
        neo = model.model if hasattr(model, "model") else model
        remote_mod = sys.modules[type(neo).__module__]
        neo.freqs_cis = remote_mod.precompute_freqs_cis(
            config.dim_head, int(config.max_length)).to(device)
        # ... and keep its padding mask broadcastable rather than (B, H, L, L).
        from nexterabert.hf_baselines import broadcast_neobert_mask

        broadcast_neobert_mask(neo)
    hard_cap = not uses_rope(config, model)
    # Prefer the table's real reach over the config number; see usable_position_len.
    real_cap = usable_position_len(model)
    if hard_cap and real_cap and (max_len is None or real_cap < max_len):
        max_len = real_cap
    # Always say what cap was detected and why -- when a model unexpectedly skips a
    # length, this line is the first thing to check.
    if max_len is None:
        print(f"[ppl] {model_name}: no positional cap found in config; "
              f"evaluating every requested length.")
    elif hard_cap:
        print(f"[ppl] {model_name}: fixed position table of {max_len} (no RoPE and no "
              f"disentangled relative attention in the config); longer lengths are "
              f"skipped.")
    else:
        why = ("disentangled relative attention, no absolute position embedding"
               if (getattr(config, "relative_attention", False) and
                   getattr(config, "position_biased_input", True) is False)
               else "computed (RoPE) positions")
        print(f"[ppl] {model_name}: {why} -- the config cap ({max_len}) is the "
              f"TRAINED context, not a limit. Longer lengths run and are reported as "
              f"extrapolation.")

    def forward(input_ids, attention_mask):
        if is_neobert:
            # NeoBERT expands the (B, L) mask to (B, H, L, L) with `repeat`; bool
            # keeps that expansion 8x smaller than the int64 mask the bench passes.
            attention_mask = attention_mask.bool()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return _logits_of(out)

    return tokenizer, model, forward, max_len, hard_cap, head_ok


def _find_mlm_head_state(backbone_path, explicit_ckpt=None, match_encoder=None):
    """Locate the trained MLM head for a backbone directory.

    ``save_pretrained`` writes only the **encoder** -- the prediction-head transform
    (dense + LayerNorm) and the decoder bias are NOT in ``model.safetensors``. Left
    at their random init they destroy the vocab projection, which is why an
    otherwise-healthy backbone reports a pseudo-perplexity in the thousands.

    Delegates to :func:`nexterabert.loading.find_mlm_head` so this benchmark, the
    exporter and the Hub uploader all search identically -- when they disagreed, this
    script found a head via the sibling ``final.pt`` while an upload of the same
    directory shipped none.

    Returns ``(state_dict, source_str)`` or ``(None, None)``.
    """
    from nexterabert.loading import find_mlm_head

    return find_mlm_head(backbone_path, explicit_ckpt,
                         match_encoder=match_encoder)


def load_nextera_masked_lm(path, tokenizer_name, preset, device, ckpt=None):
    """Reconstruct ``NexteraBERTForMaskedLM`` from a backbone directory and load its
    encoder weights **and its trained MLM head**. Falls back to an untrained preset
    (loud warning) when no checkpoint is found.

    The LM head *weight* is tied to the input embedding, which the backbone stores,
    but the prediction-head transform and the decoder bias are separate trained
    parameters that ``save_pretrained`` does not write — see
    :func:`_find_mlm_head_state`.

    ``path`` may be a Hub repo id (the default); it is snapshotted to the local cache
    and then treated exactly like a backbone directory. A repo that cannot be fetched
    raises -- silently plotting an untrained preset under the published model's name
    would be worse than no line.
    """
    path = resolve_nextera_path(path)
    trained = bool(path) and os.path.isdir(path) and \
        os.path.exists(os.path.join(path, "config.json"))
    head_state = None      # only assigned on the trained path; see the return below

    if trained:
        # Prefer a tokenizer bundled alongside the checkpoint; else the training default.
        tok_src = path if os.path.exists(os.path.join(path, "tokenizer.json")) \
            or os.path.exists(os.path.join(path, "vocab.json")) else tokenizer_name
        tokenizer = build_tokenizer(tok_src)
        config = NexteraBERTConfig.from_pretrained(path)
        model = NexteraBERTForMaskedLM(config)
        enc_state = load_backbone_state(path)
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
        # pooler / token_type_embeddings are never trained in pretraining and can be
        # absent from a backbone; don't flag them as a real gap.
        missing = [m for m in missing
                   if not m.startswith(("pooler", "token_type_embeddings"))]
        print(f"[ppl] NexteraBERT loaded from {path}")
        if missing:
            print(f"[ppl]   [warn] missing encoder keys: {missing[:4]}"
                  f"{' ...' if len(missing) > 4 else ''}")
        if unexpected:
            print(f"[ppl]   [warn] unexpected keys: {unexpected[:4]}"
                  f"{' ...' if len(unexpected) > 4 else ''}")

        # --- the trained MLM head (NOT part of the exported backbone) ---
        head_state, head_src = _find_mlm_head_state(path, ckpt,
                                                   match_encoder=enc_state)
        if head_state:
            # lm_head.weight is tied to the embedding the backbone already restored;
            # dropping it here keeps the tie intact and avoids a shape clash if the
            # checkpoint was written with a different vocab padding.
            head_state = {k: v for k, v in head_state.items() if k != "lm_head.weight"}
            h_missing, h_unexpected = model.load_state_dict(head_state, strict=False)
            # lm_head.weight is *expected* to be reported missing -- we dropped it
            # above on purpose to keep the embedding tie.
            h_missing = [m for m in h_missing
                         if m.startswith(("head.", "lm_head."))
                         and m != "lm_head.weight"]
            print(f"[ppl]   MLM head restored from {head_src}")
            if h_missing or h_unexpected:
                print(f"[ppl]   [warn] head missing={h_missing} unexpected={h_unexpected}")
        else:
            print("[ppl]   [WARNING] no trained MLM head found for this backbone. "
                  "save_pretrained() exports the ENCODER ONLY -- the prediction-head "
                  "transform (dense + LayerNorm) and the decoder bias stay at their "
                  "RANDOM init, which inflates pseudo-perplexity by orders of "
                  "magnitude. Pass --nextera_ckpt <run_dir>/final.pt (the full "
                  "training checkpoint), or keep the backbone next to it so it is "
                  "found automatically.")
    else:
        tokenizer = build_tokenizer(tokenizer_name)
        config = NexteraBERTConfig.from_preset(
            preset, vocab_size=padded_vocab_size(tokenizer))
        model = NexteraBERTForMaskedLM(config)
        print(f"[ppl] [WARNING] no trained NexteraBERT checkpoint found "
              f"(looked for {path!r}); using an UNTRAINED '{preset}' preset. The "
              f"NexteraBERT perplexity line will be meaningless -- pass --nextera_path "
              f"<backbone dir> to plot the trained model.")

    model = model.to(device).eval()

    def forward(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask)  # (B,L,V)

    # RoPE + local SSM + block-pooled HRA: no positional cap, evaluate every length.
    # head_ok is None on the untrained-preset fallback: there is no trained head to
    # look for, and that whole line is already flagged as meaningless above.
    return (tokenizer, model, forward, None, False,
            (bool(head_state) if trained else None))


def _logits_of(out):
    """Pull the (B, L, V) logits out of whatever a HF MaskedLM forward returned."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out.get("logits", out.get("prediction_logits"))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ==============================
# 擬似パープレキシティ
# ==============================
@torch.inference_mode()
def pseudo_ppl_at_length(forward, token_ids, seq_len, mask_token_id, device,
                         batch_tokens, max_batch_size, max_seqs, mask_ratio,
                         autocast_ctx, seed, window_offset=0, bookend=None,
                         protected_ids=()):
    """Pseudo-perplexity and masked-LM accuracy of one model at one sequence length.

    Cuts ``token_ids`` into contiguous windows of ``seq_len``, masks ``mask_ratio`` of
    each window (deterministically, seeded per window so runs reproduce), predicts the
    masked originals, and returns ``(ppl, acc)`` over up to ``max_seqs`` windows:
    ``exp(total_ce / total_masked)`` and the top-1 accuracy ``total_correct /
    total_masked`` on exactly the same masked positions, from the same forward.
    Batch size shrinks with length (``batch_tokens // seq_len``) so long sequences
    don't OOM; a CUDA OOM retries at batch size 1, then gives up (NaN for both) for
    that length only.

    ``window_offset`` starts the window cut that many tokens into the stream, so
    repeated runs score *disjoint* corpus slices — that, not the mask pattern alone,
    is what the reported standard deviation is measuring.

    ``bookend`` is an optional ``(cls_id, sep_id)`` pair; when given every sequence
    handed to the model is ``[CLS] body [SEP]``, matching how these encoders were
    pretrained (bare token streams are out of distribution and read as an inflated
    perplexity for all of them). ``protected_ids`` are never masked — the trainers
    exclude CLS/SEP/PAD from the MLM target set.
    """
    nothing = (float("nan"), float("nan"))      # (ppl, acc) when nothing was scored
    pre, post = bookend if bookend else ([], [])
    # Accept the older ``(cls_id, sep_id)`` scalar form as well as id lists.
    pre = [pre] if isinstance(pre, int) else list(pre or [])
    post = [post] if isinstance(post, int) else list(post or [])
    n_pre, n_post = len(pre), len(post)
    body_len = seq_len - n_pre - n_post
    if body_len <= 0:
        return nothing
    avail = max(0, (token_ids.numel() - window_offset) // body_len)
    n_windows = min(max_seqs, avail)
    if n_windows == 0:
        return nothing
    windows = token_ids[window_offset: window_offset + n_windows * body_len] \
        .view(n_windows, body_len)

    protected = torch.tensor(sorted({int(i) for i in protected_ids if i is not None}),
                             dtype=windows.dtype)

    def run(batch_size):
        total_ce, total_correct, total_masked = 0.0, 0, 0
        for b0 in range(0, n_windows, batch_size):
            body = windows[b0: b0 + batch_size].to(device)      # (B, body_len)
            # Deterministic mask over the window's content.
            gen = torch.Generator().manual_seed(seed + b0)
            sel = (torch.rand(body.shape, generator=gen) < mask_ratio).to(device)
            scoreable = (~torch.isin(body, protected.to(device))
                         if protected.numel() else torch.ones_like(sel))
            sel &= scoreable
            if not bool(sel.any()):
                # Guarantee >=1 target, but never a protected one. A batch made up
                # entirely of special tokens has nothing to score -- skip it.
                idx = scoreable.reshape(-1).nonzero()
                if idx.numel() == 0:
                    continue
                sel.view(-1)[idx[0, 0]] = True

            masked = body.clone()
            masked[sel] = mask_token_id
            if n_pre or n_post:
                def _cols(ids_):
                    return torch.tensor(ids_, dtype=masked.dtype,
                                        device=device).expand(masked.size(0), -1)
                parts = ([_cols(pre)] if n_pre else []) + [masked] + \
                        ([_cols(post)] if n_post else [])
                masked = torch.cat(parts, 1)
            attn = torch.ones_like(masked)
            with autocast_ctx:
                logits = forward(masked, attn)              # (B, L, V)
            if n_pre or n_post:
                # Drop the framing columns so the logits line up with `body`.
                logits = logits[:, n_pre: logits.size(1) - n_post] \
                    if n_post else logits[:, n_pre:]
            vocab = logits.size(-1)
            flat = sel.reshape(-1)
            rows = logits.reshape(-1, vocab)[flat].float()
            targets = body.reshape(-1)[flat]
            total_ce += F.cross_entropy(rows, targets, reduction="sum").item()
            total_correct += int((rows.argmax(-1) == targets).sum().item())
            total_masked += int(targets.numel())
        if total_masked == 0:      # nothing scoreable (all-special windows)
            return nothing
        acc = total_correct / total_masked
        try:
            return math.exp(total_ce / total_masked), acc
        except OverflowError:   # mean CE past exp()'s range -- degenerate, not a crash
            return float("inf"), acc

    def is_oom(e):
        # CUDA says "out of memory"; the CPU allocator says "not enough memory" /
        # "can't allocate" / (Windows) "bad allocation".
        msg = str(e).lower()
        return any(h in msg for h in ("out of memory", "not enough memory",
                                      "can't allocate", "bad allocation"))

    # IndexError = a learned position table indexed past its cap; AssertionError =
    # a custom implementation's own length check (e.g. NeoBERT's rotary reshape).
    # Degrade to nan for that length instead of crashing the sweep.
    bs = max(1, min(max_batch_size, batch_tokens // seq_len))
    try:
        return run(bs)
    except (RuntimeError, IndexError, AssertionError) as e:
        if not is_oom(e):
            print(f"[ppl]   [warn] seq_len={seq_len}: {type(e).__name__}: {e}")
            return nothing
        if device == "cuda":
            torch.cuda.empty_cache()
        if bs == 1:
            print(f"[ppl]   [warn] seq_len={seq_len}: out of memory at bs=1 -- this "
                  f"length does not fit on this machine ({type(e).__name__}); skipping.")
            return nothing
        print(f"[ppl]   seq_len={seq_len}: OOM at bs={bs}, retrying at bs=1")
        try:
            return run(1)
        except (RuntimeError, IndexError, AssertionError) as e2:
            print(f"[ppl]   [warn] seq_len={seq_len}: OOM even at bs=1, skipping ({e2})")
            if device == "cuda":
                torch.cuda.empty_cache()
            return nothing


# Missing optional dependencies are the usual reason a baseline refuses to load,
# and the raw exception rarely names the package to install.
_LOAD_HINTS = (
    ("sentencepiece", "pip install sentencepiece  (DeBERTa-v2/v3 and many "
                      "SentencePiece tokenizers need it)"),
    ("tiktoken", "pip install tiktoken"),
    ("protobuf", "pip install protobuf"),
    ("xformers", "NeoBERT's remote code wants xformers; the script registers an "
                 "eager stand-in, so this usually means the stand-in was bypassed"),
    ("trust_remote_code", "the checkpoint needs trust_remote_code=True; the script "
                          "already passes it, so this is likely a network/cache issue"),
    ("connection", "network error reaching the Hub -- retry, or pre-download the "
                   "model with huggingface-cli"),
)


def _load_failure_hint(name, err):
    blob = f"{type(err).__name__} {err}".lower()
    for needle, hint in _LOAD_HINTS:
        if needle in blob:
            return hint
    return None


def framing_tokens(tokenizer):
    """What this tokenizer wraps a sequence in, as ``(prefix_ids, suffix_ids)``.

    Asked of the tokenizer rather than assumed, by diffing an encoding with and
    without special tokens. Guessing from ``cls_token_id``/``sep_token_id`` gets the
    BERT family right and everything else wrong: RoBERTa uses ``<s>``/``</s>``, and
    LFM2.5-Encoder prepends ``<|startoftext|>`` and appends **nothing** — pairing its
    ``bos`` with its ``eos`` would append ``<|im_end|>``, a chat-template marker the
    encoder never sees in this position.

    Returns ``([], [])`` when the tokenizer adds nothing.
    """
    try:
        core = tokenizer("hello world", add_special_tokens=False)["input_ids"]
        full = tokenizer("hello world", add_special_tokens=True)["input_ids"]
    except Exception:  # noqa: BLE001 - exotic tokenizer; fall back to no framing
        return [], []
    if not core:
        return [], []
    for i in range(len(full) - len(core) + 1):
        if list(full[i:i + len(core)]) == list(core):
            return list(full[:i]), list(full[i + len(core):])
    return [], []


def _nan_result(n_lengths, repeats):
    """The result shape ``evaluate_model`` returns, all-NaN (model skipped)."""
    return {
        "mean": [float("nan")] * n_lengths,
        "std": [float("nan")] * n_lengths,
        "runs": [[float("nan")] * repeats for _ in range(n_lengths)],
        "acc_mean": [float("nan")] * n_lengths,
        "acc_std": [float("nan")] * n_lengths,
        "acc_runs": [[float("nan")] * repeats for _ in range(n_lengths)],
        "seeds": [], "max_len": None, "hard_cap": None, "head_ok": None,
        "error": None,
    }


def _mean_std(values):
    """Mean and *sample* std (ddof=1) over the finite entries of ``values``.
    A single finite value has no dispersion estimate, so its std is 0.0."""
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan")
    mean = sum(good) / len(good)
    if len(good) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in good) / (len(good) - 1)
    return mean, math.sqrt(var)


def _point_fields(pt):
    """One recorded JSON point as ``(mean, std, runs, acc_mean, acc_std, acc_runs)``,
    with the nulls ``_json_safe`` wrote turned back into NaN."""
    def num(key):
        return pt[key] if pt.get(key) is not None else float("nan")

    def seq(key):
        return [v if v is not None else float("nan") for v in pt.get(key, [])]

    return (num("mean"), num("std"), seq("runs"),
            num("acc_mean"), num("acc_std"), seq("acc_runs"))


def _result_from_cache(args, points, meta):
    """Assemble a result from recorded points alone, without loading the model.

    Normally every requested length is recorded. The per-model JSON checkpoint also
    calls this for a model still waiting its turn, where some lengths may be missing;
    those come back NaN, are written as null, and so still count as unmeasured."""
    points = {L: points.get(L, {}) for L in args.seq_lengths}
    fields = [_point_fields(points[L]) for L in args.seq_lengths]
    return {
        "mean": [f[0] for f in fields],
        "std": [f[1] for f in fields],
        "runs": [f[2] for f in fields],
        "acc_mean": [f[3] for f in fields],
        "acc_std": [f[4] for f in fields],
        "acc_runs": [f[5] for f in fields],
        "seeds": meta.get("seeds", []),
        "max_len": meta.get("max_position"),
        "hard_cap": meta.get("hard_positional_cap"),
        "head_ok": meta.get("mlm_head_loaded"),
        "error": meta.get("load_error"),
    }


def evaluate_model(name, args, corpus_texts, device, autocast_ctx,
                   cached=None, cached_meta=None):
    """Load one model, compute its pseudo-PPL at every sequence length, free it.

    Each length is measured ``args.repeats`` times over *disjoint* corpus slices
    (and independent mask draws); the returned ``mean``/``std`` summarise those
    repeats, and ``runs`` keeps the individual values.

    ``cached`` holds points already recorded for this model in a previous run. Any
    length found there is reused verbatim, and when *every* requested length is
    present the model is never loaded at all -- which is the whole point, since
    downloading and instantiating it is most of the cost of a short sweep."""
    is_nextera = name == NEXTERA_NAME
    n_lengths = len(args.seq_lengths)
    points = dict(cached or {})
    todo = [L for L in args.seq_lengths if L not in points]
    if not todo:
        print(f"[ppl] {display_name(name)}: all {n_lengths} length(s) already "
              f"recorded -- skipping (model not loaded).")
        return _result_from_cache(args, points, cached_meta or {})
    if points:
        print(f"[ppl] {display_name(name)}: reusing {sorted(points)}, "
              f"measuring {todo}.")

    if is_nextera:
        try:
            (tokenizer, model, forward, max_len, hard_cap,
             head_ok) = load_nextera_masked_lm(
                args.nextera_path, args.tokenizer, args.preset, device,
                ckpt=args.nextera_ckpt)
        except Exception as e:  # noqa: BLE001 - keep the baselines' results
            print(f"[ppl] [ERROR] could not load NexteraBERT from "
                  f"{args.nextera_path!r}: {e}")
            res = _nan_result(n_lengths, args.repeats)
            res["error"] = f"{type(e).__name__}: {e}"
            return res
    else:
        try:
            (tokenizer, model, forward, max_len, hard_cap,
             head_ok) = load_hf_masked_lm(name, device, max(args.seq_lengths))
        except Exception as e:  # noqa: BLE001 - report and skip a model that won't load
            import traceback

            print(f"[ppl] [ERROR] could not load {name} as a masked LM -- this model "
                  f"will be absent from every length, not just the long ones:")
            traceback.print_exc()
            hint = _load_failure_hint(name, e)
            if hint:
                print(f"[ppl]   hint: {hint}")
            res = _nan_result(n_lengths, args.repeats)
            res["error"] = f"{type(e).__name__}: {e}"
            return res


    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        print(f"[ppl] [warn] {name} tokenizer has no [MASK] token; skipping.")
        del model
        _free(device)
        return _nan_result(n_lengths, args.repeats)

    # Frame each window the way this tokenizer itself frames text, and never score
    # the framing tokens -- the trainers exclude them from the MLM target set.
    pre, post = framing_tokens(tokenizer) if args.special_tokens else ([], [])
    bookend = (pre, post) if (pre or post) else None
    if args.special_tokens:
        if bookend:
            shown = "".join(tokenizer.decode([i]) for i in pre) + " ... " + \
                    ("".join(tokenizer.decode([i]) for i in post) or "(nothing)")
            print(f"[ppl] {name}: framing windows as {shown!r}")
        else:
            print(f"[ppl] [warn] {name}: its tokenizer adds no special tokens; "
                  f"windows are scored unframed.")
    protected = set(pre) | set(post) | {tokenizer.pad_token_id}

    # Enough tokens for `repeats` disjoint sweeps at the longest length, plus slack.
    longest = max(args.seq_lengths)
    need = longest * args.max_seqs * args.repeats + longest
    texts = corpus_texts() if callable(corpus_texts) else corpus_texts
    token_ids = tokenize_concat(tokenizer, texts, need)
    if token_ids.numel() < min(args.seq_lengths):
        print(f"[ppl] [warn] {name}: corpus tokenised to only {token_ids.numel()} tokens.")

    means, stds, runs = [], [], []
    acc_means, acc_stds, acc_runs = [], [], []
    for seq_len in args.seq_lengths:
        # Only a learned absolute-position table is a *hard* wall (indexing past it is
        # impossible). RoPE models run at any length -- past max_len they extrapolate
        # beyond their trained context, which is exactly what the sweep should show.
        if seq_len in points:
            pt = points[seq_len]
            m, s, rs, am, as_, ars = _point_fields(pt)
            means.append(m)
            stds.append(s)
            runs.append(rs)
            acc_means.append(am)
            acc_stds.append(as_)
            acc_runs.append(ars)
            print(f"{name} | seq_len={seq_len} | ppl={m:.3f} | "
                  f"mlm_acc={am * 100:.2f}% (recorded)")
            continue

        # Indexing a position table out of range raises a catchable RuntimeError on
        # CPU but a device-side assert on CUDA -- which poisons the context and kills
        # every model after this one, not just this length. So skip rather than
        # attempt: there is nothing to learn from the attempt, and the downside is
        # losing the whole sweep.
        if max_len is not None and seq_len > max_len and hard_cap:
            print(f"{name} | seq_len={seq_len} | skip (its position table stops at "
                  f"{max_len})")
            means.append(float("nan"))
            stds.append(float("nan"))
            runs.append([float("nan")] * args.repeats)
            acc_means.append(float("nan"))
            acc_stds.append(float("nan"))
            acc_runs.append([float("nan")] * args.repeats)
            continue

        body = seq_len - (len(pre) + len(post) if bookend else 0)
        avail = token_ids.numel() // max(body, 1)
        per_run = args.max_seqs
        trial, acc_trial = [], []
        for r in range(args.repeats):
            # Disjoint slice per repeat; wrap around (with a shifted seed, so the
            # mask pattern still differs) if the corpus cannot cover them all.
            offset = r * per_run * body
            if offset + body > token_ids.numel():
                offset = (offset % max(avail, 1)) * body
            ppl, acc = pseudo_ppl_at_length(
                forward, token_ids, seq_len, mask_id, device,
                args.batch_tokens, args.max_batch_size, per_run,
                args.mask_ratio, autocast_ctx, args.seed + r * 100_003,
                window_offset=offset, bookend=bookend, protected_ids=protected)
            trial.append(ppl)
            acc_trial.append(acc)
        mean, std = _mean_std(trial)
        acc_mean, acc_std = _mean_std(acc_trial)
        means.append(mean)
        stds.append(std)
        runs.append(trial)
        acc_means.append(acc_mean)
        acc_stds.append(acc_std)
        acc_runs.append(acc_trial)

        shown = f"{mean:.3f} +/- {std:.3f}" if math.isfinite(mean) else "nan"
        acc_shown = (f"{acc_mean * 100:.2f} +/- {acc_std * 100:.2f}%"
                     if math.isfinite(acc_mean) else "nan")
        each = " ".join(f"{v:.3f}" if math.isfinite(v) else "nan" for v in trial)
        if not math.isfinite(mean):
            note = " -- no finite value from any seed"
        else:
            note = (f" (extrapolating beyond trained context {max_len})"
                    if max_len is not None and seq_len > max_len else "")
        print(f"{name} | seq_len={seq_len} | ppl={shown} | mlm_acc={acc_shown} "
              f"(n={args.repeats}: {each}){note}")

    del model
    _free(device)
    return {"mean": means, "std": stds, "runs": runs,
            "acc_mean": acc_means, "acc_std": acc_stds, "acc_runs": acc_runs,
            "seeds": [args.seed + r * 100_003 for r in range(args.repeats)],
            "max_len": max_len, "hard_cap": hard_cap, "head_ok": head_ok}


def _free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ==============================
# 結果の記録
# ==============================
def _fmt_cell(mean, std):
    """``mean +/- std`` in a fixed width. A meaningless model can score in the
    millions, and %.3f on that shoves every later column out of alignment."""
    if not math.isfinite(mean):
        return "-"
    fmt = "{:.3e}" if mean >= 1e5 else "{:.3f}"
    return f"{fmt.format(mean)} +/- {fmt.format(std) if math.isfinite(std) else '?'}"


# Everything here changes the numbers, so a cached point is only reusable when all
# of it matches. batch_tokens / max_batch_size are deliberately absent: they change
# how the work is split across forwards, not the result.
def estimator_settings(args):
    return {
        "mask_ratio": args.mask_ratio,
        "windows_per_seed": args.max_seqs,
        "seeds": args.repeats,
        "base_seed": args.seed,
        "special_token_framing": args.special_tokens,
    }


def corpus_settings(args):
    return {
        "dataset": args.dataset,
        "config": args.dataset_config,
        "split": args.split,
        "text_column": args.text_column,
    }


def load_cached_results(path, args):
    """Points already recorded in ``path``, keyed ``{model_id: {seq_len: point}}``.

    Only a *finite* mean counts as recorded. A null is either a length the model
    cannot run (re-deciding that is free) or a failure worth retrying, and silently
    inheriting it would make a transient OOM permanent.

    The whole cache is discarded unless the recorded settings match this run's, since
    every one of them moves the numbers. Returns ``(points, meta, prior_doc_count)``;
    the last is carried into the rewritten file so a fully-resumed run -- which never
    touches the corpus -- does not overwrite it with zero.
    """
    import json

    if not path or not os.path.exists(path):
        return {}, {}, None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # noqa: BLE001 - a truncated file is not a reason to stop
        print(f"[ppl] [warn] could not read {path} ({type(e).__name__}); "
              f"starting fresh.")
        return {}, {}, None

    for label, want, got in (("estimator", estimator_settings(args),
                              payload.get("estimator", {})),
                             ("corpus", corpus_settings(args),
                              {k: payload.get("corpus", {}).get(k)
                               for k in corpus_settings(args)})):
        differing = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if differing:
            shown = ", ".join(f"{k}: recorded {old!r} vs now {new!r}"
                              for k, (old, new) in list(differing.items())[:3])
            print(f"[ppl] {path} was written with different {label} settings "
                  f"({shown}) -- ignoring it and re-measuring everything.")
            return {}, {}, None

    cached, meta = {}, {}
    for entry in payload.get("models", []):
        # "NexteraBERT (Ours)" is one id for whatever weights --nextera_path names, so
        # points recorded for other weights are a different model's numbers.
        if entry.get("id") == NEXTERA_NAME and (
                entry.get("nextera_path") != args.nextera_path
                or entry.get("nextera_ckpt") != args.nextera_ckpt):
            print(f"[ppl] {path}: NexteraBERT points were recorded for "
                  f"{entry.get('nextera_path')!r}, not {args.nextera_path!r} -- "
                  f"re-measuring them.")
            continue
        # A point written before accuracy was recorded (schema /1) has no acc_mean;
        # treat it as unmeasured so it is redone rather than plotted with a hole.
        pts = {p["seq_len"]: p for p in entry.get("by_seq_len", [])
               if p.get("mean") is not None and p.get("acc_mean") is not None}
        if pts:
            cached[entry["id"]] = pts
            meta[entry["id"]] = entry
    if cached:
        total = sum(len(v) for v in cached.values())
        print(f"[ppl] reusing {total} recorded point(s) from {path} "
              f"({len(cached)} model(s)); pass --refresh to re-measure them.")
    return cached, meta, payload.get("corpus", {}).get("documents")


def _json_safe(x):
    """JSON has no NaN/Infinity; emit null so the file is valid, parseable JSON."""
    return x if isinstance(x, (int, str, bool, type(None))) or (
        isinstance(x, float) and math.isfinite(x)) else None


def write_json(path, args, results, device, corpus_docs):
    """Record every measurement, not just the summary.

    The per-repeat values and their seeds go in alongside the mean and std, so a
    number in the paper can be traced back to the run that produced it and re-derived
    without re-running the sweep.
    """
    import json
    import platform
    from datetime import datetime, timezone

    payload = {
        "schema": "nexterabert.model_pplbench/2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": "pseudo-perplexity (Salazar et al. 2020, one-pass subsampled)",
        "accuracy_metric": "masked-LM top-1 accuracy over the same masked positions "
                           "(fraction in [0, 1])",
        "environment": {
            "device": device,
            "dtype": args.dtype,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "corpus": {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "text_column": args.text_column,
            "documents": corpus_docs,
        },
        "estimator": estimator_settings(args),
        "seq_lengths": list(args.seq_lengths),
        "models": [],
    }

    for name, res in results.items():
        entry = {
            "id": name,
            "display": display_name(name),
            "is_nextera": name == NEXTERA_NAME,
            "max_position": res.get("max_len"),
            "hard_positional_cap": res.get("hard_cap"),
            "seeds": res.get("seeds", []),
            # False => the checkpoint had no trained MLM head and the numbers below
            # are noise. Recorded so a downstream table can drop or flag the row.
            "mlm_head_loaded": res.get("head_ok"),
            # Set when the model could not be loaded at all -- distinguishes "absent
            # because it failed" from "absent because the length was out of range".
            "load_error": res.get("error"),
            "by_seq_len": [],
        }
        if name == NEXTERA_NAME:
            entry["nextera_path"] = args.nextera_path
            entry["nextera_ckpt"] = args.nextera_ckpt
        for j, seq_len in enumerate(args.seq_lengths):
            entry["by_seq_len"].append({
                "seq_len": seq_len,
                "mean": _json_safe(res["mean"][j]),
                "std": _json_safe(res["std"][j]),
                "runs": [_json_safe(v) for v in res["runs"][j]],
                # Top-1 accuracy on the masked positions, one value per seed, from
                # the same forwards as `runs` (so acc_runs[i] pairs with runs[i]).
                "acc_mean": _json_safe(res["acc_mean"][j]),
                "acc_std": _json_safe(res["acc_std"][j]),
                "acc_runs": [_json_safe(v) for v in res["acc_runs"][j]],
            })
        payload["models"].append(entry)

    # Written after every model, and it is the resume log: replace it atomically so a
    # kill mid-write leaves the previous complete file, never a truncated one.
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    print(f"[ppl] results -> {path}")


# ==============================
# プロット
# ==============================
def _zoom_limits(results, seq_lengths, factor):
    """The y-range worth a second panel, or ``None``.

    A sweep like this spans five orders of magnitude — a model that has fallen apart
    reads 1e5 while the ones still working sit near 3 — and on a shared log axis every
    difference and error bar among the working models collapses into one flat line.
    So find the *low cluster*: everything within ``factor`` of the best value anywhere
    in the sweep. If that cluster covers only part of the data, it earns its own panel.
    """
    vals = [m for res in results.values() for m in res["mean"] if math.isfinite(m)]
    if not vals:
        return None
    lo = min(vals)
    inside = [v for v in vals if v <= lo * factor]
    if len(inside) == len(vals):
        return None                      # nothing extreme; one panel is enough
    hi = max(inside)
    # Tight padding: the point of the panel is resolution, and a generous margin
    # spends it on empty space below the best curve.
    return lo * 0.97, hi * 1.06


# What a figure plots: result keys for the centre and the spread, a multiplier into
# display units, and the axis label. Accuracy is stored as a fraction, shown in %.
METRICS = {
    "ppl": ("mean", "std", 1.0, "Pseudo-Perplexity"),
    "acc": ("acc_mean", "acc_std", 100.0, "Masked-LM Accuracy (top-1, %)"),
}


def _draw_series(ax, results, seq_lengths, log_y, clip_above=None, styles=None,
                 metric="ppl"):
    """Draw every model onto one axes as mean +/- 1 std error bars.

    ``clip_above`` drops points beyond the panel's range instead of letting matplotlib
    draw the near-vertical connector to an off-screen value: in a zoomed figure those
    spikes are pure noise, and the overview already shows where each model goes.
    """
    mean_key, std_key, scale, _ = METRICS[metric]
    for i, (name, res) in enumerate(results.items()):
        # Styles are resolved against the FULL model list, not this axes' subset, so a
        # model keeps the same colour and marker in every figure of the run.
        style = (styles or {}).get(name) or series_style(i, name)
        # Plot only the valid points so each line cleanly ends at the model's context
        # wall (NaN past max length / OOM, inf on overflow) instead of drawing a gap.
        pts = [(s, m * scale, e * scale) for s, m, e
               in zip(seq_lengths, res[mean_key], res[std_key])
               if math.isfinite(m) and (clip_above is None or m <= clip_above)]
        if not pts:
            continue
        label = display_name(name)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] if math.isfinite(p[2]) else 0.0 for p in pts]
        # Error bars = std over the seeds. On a log y-axis a symmetric bar can dip
        # to <= 0 and matplotlib drops the point, so clamp the lower arm.
        lower = [min(e, y * 0.99) for y, e in zip(ys, es)] if log_y else es
        upper = es
        if metric == "acc":
            # A percentage cannot leave [0, 100]; a symmetric std bar near either end
            # would draw an accuracy that does not exist.
            lower = [min(e, y) for y, e in zip(ys, es)]
            upper = [min(e, 100.0 - y) for y, e in zip(ys, es)]
        ax.errorbar(xs, ys, yerr=[lower, upper], label=label, capsize=3,
                    elinewidth=1.1, **style)


def _write_figure(path, results, seq_lengths, styles, *, log_y, title, subtitle=None,
                  ylim=None, clip_above=None, legend_cols=1, metric="ppl"):
    """Render one self-contained figure and save it. Returns the path."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_series(ax, results, seq_lengths, log_y, clip_above=clip_above,
                 styles=styles, metric=metric)
    if log_y:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(s) for s in seq_lengths], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(METRICS[metric][3] + (" (log)" if log_y else ""))
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=12 if not subtitle else 11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=legend_cols, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _suffixed(output, suffix):
    base, ext = os.path.splitext(output)
    return f"{base}_{suffix}{ext or '.png'}"


def models_under(results, seq_lengths, threshold):
    """The models that stay at or below ``threshold`` for the whole sweep.

    "For the whole sweep" is meant literally: every recorded value must be within the
    threshold *and* the longest requested length must have produced one. A model that
    is excellent up to 4096 and then explodes, or that stops reporting partway, is not
    what this figure is about -- it exists so the models that hold up end to end can be
    compared against each other, at a scale where a +/-0.05 standard deviation is
    actually visible rather than a rounding error three decades down a log axis.
    """
    # Index of the longest length, not the last entry: --seq_lengths is whatever the
    # caller passed and need not be in ascending order.
    longest = max(range(len(seq_lengths)), key=lambda j: seq_lengths[j])
    keep = {}
    for name, res in results.items():
        finite = [m for m in res["mean"] if math.isfinite(m)]
        if not finite or max(finite) > threshold:
            continue
        if not math.isfinite(res["mean"][longest]):
            continue
        keep[name] = res
    return keep


def plot(results, seq_lengths, output, log_y, ylim=None, zoom_factor=2.5,
         focus_max=10.0):
    """Write the figures, one per file.

    Always ``output`` itself: every model, log y, the whole picture. Then, when they
    apply, two more beside it -- each a standalone image rather than a panel, so any
    one of them can go into a paper or a slide on its own:

      ``<output>_zoom.png``   the same models, y-range cropped to the low cluster.
      ``<output>_focus.png``  only the models that stay under ``focus_max`` throughout.

    The two differ in what they hide: the zoom keeps every model and crops the axis
    (a line simply leaves the frame), while the focus drops the models that blow up
    and keeps the axis honest for the ones that remain.
    """
    styles = {name: series_style(i, name) for i, name in enumerate(results)}
    written = [_write_figure(
        output, results, seq_lengths, styles, log_y=log_y, ylim=ylim,
        title="Masked-LM Perplexity vs Sequence Length",
        legend_cols=2 if len(results) > 4 else 1)]
    print(f"[ppl] saved -> {output}")

    # Accuracy is bounded in [0, 100], so it needs neither the log axis nor the
    # zoom/focus treatment: a collapsed model reads ~0% instead of 1e5 and flattens
    # nobody. Skipped only when there is nothing to draw (e.g. every model failed).
    if any(math.isfinite(m) for res in results.values()
           for m in res.get("acc_mean", [])):
        path = _write_figure(
            _suffixed(output, "acc"), results, seq_lengths, styles, log_y=False,
            metric="acc", title="Masked-LM Accuracy vs Sequence Length",
            subtitle="top-1 on the masked positions (bars = +/-1 std over seeds)",
            legend_cols=2 if len(results) > 4 else 1)
        written.append(path)
        print(f"[ppl] saved -> {path}")

    zoom = None if (ylim or zoom_factor <= 0) else \
        _zoom_limits(results, seq_lengths, zoom_factor)
    if zoom:
        # Linear here on purpose: inside the zoom the values differ by a factor of a
        # few, and a linear axis is where a 2.97 +/- 0.05 error bar is actually visible.
        path = _write_figure(
            _suffixed(output, "zoom"), results, seq_lengths, styles, log_y=False,
            ylim=zoom, clip_above=zoom[1],
            title="Masked-LM Perplexity vs Sequence Length",
            subtitle=(f"zoom: within {zoom_factor:g}x of the best result "
                      f"(a line stops where it leaves the frame)"))
        written.append(path)
        print(f"[ppl] saved -> {path}  (y {zoom[0]:.2f}-{zoom[1]:.2f})")

    if focus_max and focus_max > 0:
        subset = models_under(results, seq_lengths, focus_max)
        if len(subset) < 2:
            print(f"[ppl] focus figure skipped: {len(subset)} model(s) stay at or "
                  f"below {focus_max:g} across every length (need at least 2).")
        else:
            path = _write_figure(
                _suffixed(output, "focus"), subset, seq_lengths, styles, log_y=False,
                title="Masked-LM Perplexity vs Sequence Length",
                subtitle=(f"models staying at or below {focus_max:g} across the whole "
                          f"sweep (bars = +/-1 std over seeds)"))
            written.append(path)
            print(f"[ppl] saved -> {path}  ({len(subset)} model(s): "
                  f"{', '.join(display_name(n) for n in subset)})")
    return written


# ==============================
# CLI / 実行
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nextera_path", default=DEFAULT_NEXTERA_REPO,
                   help="trained NexteraBERT weights: a Hub repo id (downloaded to the "
                        "HF cache; needs a token with read access if the repo is "
                        "private) or a local backbone directory (config.json + "
                        f"weights). Default: {DEFAULT_NEXTERA_REPO}")
    p.add_argument("--nextera_ckpt", default=None,
                   help="full training checkpoint (.pt from scripts/pretrain.py) holding "
                        "the trained MLM head. save_pretrained() exports the ENCODER "
                        "ONLY, so without the head the prediction transform + decoder "
                        "bias stay randomly initialised and the perplexity is garbage. "
                        "Not needed for a Hub repo or a current export, which carry "
                        "mlm_head.safetensors. Default: 'final.pt' / the newest "
                        "'step_*.pt' in the backbone's parent run directory.")
    p.add_argument("--preset", default="mezzoforte",
                   help="preset used ONLY for the untrained fallback when no checkpoint exists")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="NexteraBERT tokenizer (used when the backbone bundles none)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"models to compare, as short aliases or raw Hub ids. "
                        f"Aliases: {', '.join(sorted(MODEL_ALIASES))}. "
                        f"Default: {' '.join(DEFAULT_MODELS)}")
    p.add_argument("--seq_lengths", nargs="+", type=int, default=DEFAULT_SEQUENCE_LENGTHS,
                   help=f"sequence lengths to sweep (default: {DEFAULT_SEQUENCE_LENGTHS})")
    # --- corpus (FineWeb-Edu, streamed) ---
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset streamed for evaluation text (the training distribution)")
    p.add_argument("--dataset_config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    # --- estimator knobs ---
    p.add_argument("--mask_ratio", type=float, default=0.15,
                   help="fraction of each window masked for the MLM loss (Salazar et al.; 0.15)")
    p.add_argument("--max_seqs", type=int, default=32,
                   help="windows averaged per sequence length (more = smoother, slower)")
    p.add_argument("--repeats", "--seeds", dest="repeats", type=int, default=25,
                   help="seeds (independent measurements) per sequence length, each "
                        "over a DISJOINT corpus slice with its own mask draw. Both the "
                        "pseudo-perplexity and the masked-LM accuracy are reported as "
                        "mean +/- sample std, with error bars on the plots and every "
                        "individual value kept in the JSON. Costs seeds x the corpus "
                        "and the compute. Default 25.")
    p.add_argument("--no_special_tokens", dest="special_tokens", action="store_false",
                   help="score bare token windows instead of '[CLS] body [SEP]'. Every "
                        "model here was pretrained on framed sequences, so unframed "
                        "windows are out of distribution and read high for all of them.")
    p.set_defaults(special_tokens=True)
    p.add_argument("--batch_tokens", type=int, default=8192,
                   help="target tokens per batch; batch size = batch_tokens // seq_len")
    p.add_argument("--max_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--linear_y", dest="log_y", action="store_false",
                   help="use a linear perplexity axis (default: log, since PPL spans "
                        "orders of magnitude across models)")
    p.set_defaults(log_y=True)
    p.add_argument("--ylim", nargs=2, type=float, default=None,
                   metavar=("LOW", "HIGH"),
                   help="clamp the perplexity axis. Implies a single panel -- use it "
                        "when you want one figure showing exactly this range.")
    p.add_argument("--zoom_factor", type=float, default=2.5,
                   help="the second panel covers every point within this factor of the "
                        "best result anywhere in the sweep (default 2.5). A collapsed "
                        "model reading 1e5 otherwise flattens everyone else into a "
                        "single line with invisible error bars. Raise it to bring the "
                        "weaker models into the zoom too.")
    p.add_argument("--no_zoom", dest="zoom_factor", action="store_const", const=0.0,
                   help="one panel only, whatever the spread.")
    p.add_argument("--output", default="ppl_benchmark.png")
    p.add_argument("--focus_max", type=float, default=10.0,
                   help="also write <output>_focus.png containing ONLY the models "
                        "whose perplexity stays at or below this value across every "
                        "length (and that still report one at the longest length). "
                        "The overview's log axis compresses those models into a single "
                        "band; this figure gives them a linear axis where the seed "
                        "spread is readable. 0 disables it.")
    p.add_argument("--refresh", action="store_true",
                   help="re-measure everything, ignoring points already recorded in "
                        "the --json file. Without this, a model whose lengths are all "
                        "recorded is not even loaded, and a partly-recorded one only "
                        "measures what is missing -- so an interrupted sweep resumes "
                        "where it stopped. Recorded settings must match the current "
                        "ones or the file is ignored anyway.")
    p.add_argument("--json", dest="json_output", default=None,
                   help="where to write the full results (means, stds, every "
                        "per-seed value, and the run's settings). Default: --output "
                        "with a .json suffix. Pass 'none' to skip.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    # Aliases -> Hub ids, keeping the caller's ordering (and therefore plot colours).
    args.models = [resolve_model(m) for m in args.models]
    # De-duplicate: two aliases can resolve to the same id, and evaluating a model
    # twice would just draw the same line over itself at twice the cost.
    seen = set()
    args.models = [m for m in args.models if not (m in seen or seen.add(m))]

    if args.json_output is None:
        base = os.path.splitext(args.output)[0]
        args.json_output = base + ".json"

    args.repeats = max(1, args.repeats)

    cached, cached_meta, prior_docs = ({}, {}, None) if args.refresh else \
        load_cached_results(args.json_output, args)

    device = resolve_device()
    print(f"[ppl] device={device}  models={args.models}  repeats={args.repeats}  "
          f"special_tokens={args.special_tokens}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype)
        if device in ("cuda", "cpu") and dtype != torch.float32 else contextlib.nullcontext()
    )

    # One shared corpus for every model (re-tokenised per family). ~6 chars/token gives
    # comfortable headroom over the largest token budget any single model consumes.
    # Fetched on first use: a fully-resumed run needs no corpus at all, and streaming
    # tens of MB from the Hub just to reprint recorded numbers would be the slowest
    # part of it.
    char_budget = (max(args.seq_lengths) * args.max_seqs * max(args.repeats, 1)
                   + max(args.seq_lengths)) * 6
    _corpus = []

    def corpus_texts():
        if not _corpus:
            _corpus.extend(collect_corpus_text(
                args.dataset, args.dataset_config, args.split, args.text_column,
                char_budget))
        return _corpus

    write = bool(args.json_output) and args.json_output.lower() != "none"
    results = {}
    for name in args.models:
        print(f"\n=== {display_name(name)}  [{name}] ===")
        results[name] = evaluate_model(name, args, corpus_texts, device, autocast_ctx,
                                       cached.get(name), cached_meta.get(name))
        if write:
            # Checkpoint as soon as a model finishes, so a crash or a Ctrl-C three
            # models in costs the model in flight, not the sweep. Models still waiting
            # keep the points they already had: dropping them here would turn an
            # interrupted *resume* into a loss of everything it had not reached yet.
            snapshot = {
                n: results[n] if n in results
                else _result_from_cache(args, cached[n], cached_meta.get(n, {}))
                for n in args.models if n in results or n in cached}
            write_json(args.json_output, args, snapshot, device,
                       len(_corpus) if _corpus else prior_docs)

    print(f"\n[ppl] pseudo-perplexity summary (mean +/- std over "
          f"{args.repeats} seeds)")
    print("[ppl]   (!) = no trained MLM head, number is meaningless")
    header = "seq_len".ljust(10) + "".join(display_name(n)[:26].ljust(30)
                                            for n in results)
    print(header)
    for j, seq_len in enumerate(args.seq_lengths):
        row = str(seq_len).ljust(10)
        for res in results.values():
            m, s = res["mean"][j], res["std"][j]
            cell = _fmt_cell(m, s)
            if res.get("head_ok") is False and math.isfinite(m):
                cell += " (!)"        # no trained MLM head: not a real perplexity
            row += cell.ljust(30)
        print(row)

    print(f"\n[ppl] masked-LM accuracy summary (top-1 %, mean +/- std over "
          f"{args.repeats} seeds; same flags as above)")
    print(header)
    for j, seq_len in enumerate(args.seq_lengths):
        row = str(seq_len).ljust(10)
        for res in results.values():
            m, s = res["acc_mean"][j], res["acc_std"][j]
            cell = "-"
            if math.isfinite(m):
                cell = f"{m * 100:.2f} +/- " + \
                    (f"{s * 100:.2f}" if math.isfinite(s) else "?")
                if res.get("head_ok") is False:
                    cell += " (!)"
            row += cell.ljust(30)
        print(row)

    # The JSON is already complete: the checkpoint after the last model holds them all.
    plot(results, args.seq_lengths, args.output, args.log_y,
         ylim=tuple(args.ylim) if args.ylim else None,
         zoom_factor=args.zoom_factor, focus_max=args.focus_max)


if __name__ == "__main__":
    main()
