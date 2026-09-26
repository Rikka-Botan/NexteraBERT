"""Inference latency vs sequence length, NexteraBERT vs the field.

The sibling ``model_pplbench.py`` plots *pseudo-perplexity* against sequence length;
this plots **forward latency** for the same set of encoders, so the speed picture lines
up with the quality picture. The model registry (``--models`` aliases, display names,
plot styling) and the baseline repairs are shared with it rather than copied.

Measurement. The optimisation settings are the original script's, untouched:
``torch.compile(model, mode="reduce-overhead")`` applied **once per model**, ``eval()``
+ ``inference_mode`` + bf16 autocast, ``cudnn.benchmark``, and ``--warmup`` untimed
forwards under that same context so tracing and the CUDA-graph capture land outside
the measured region. One **set** is the original measurement itself: ``--runs``
(default **5**) forwards queued back to back with a single synchronise at the end. A
pair of CUDA events brackets each forward, and since every start event sits on the GPU
timeline right behind the previous end event, the samples tile the block -- a set's
mean *is* the old ``elapsed / num_runs``.

The sweep covers 1024 to 65536 tokens by default. Batch size follows the length:
``batch = --batch_tokens // seq_len`` (default 65536 tokens, clamped to
``1..--max_batch_size``), so every point pushes the same number of tokens through the
model -- 64 x 1024, 32 x 2048, ... 1 x 65536. A fixed batch of 2 (the old behaviour,
still available as ``--batch_size 2``) hands a GPU 2048 tokens at length 1024: the
device idles between kernel launches and the measured tokens/s *falls* towards short
lengths, which says nothing about the model. The
latency figure therefore shows seconds **per sequence** (batch time / batch size),
which stays comparable across lengths; the JSON keeps the raw per-batch runs and the
batch size of every point.

Rounds. The sweep is organised as ``--rounds`` (default **5**) passes over the *whole*
model list: round 1 loads, compiles and times every model at every length (one set
each), then round 2 does all of it again, and so on -- 5 sets x 5 runs = 25 forwards
per point. Interleaving is the point: five sets taken back to back inside one process
state share their compile, their autotuned kernels, their allocator layout and the
GPU's clock/thermal state, so their spread says nothing about whether the number would
come out the same next time. A round repeats all of that from scratch, and spreading a
model's sets across the session keeps slow drift from landing on one model only.
Reported per length:

  * ``mean`` = mean of the set means, with the two-sided Student-t **confidence
    interval** over the **set means** (``--confidence``, default 0.95; at 5 rounds the
    half-width is 2.776 x std / sqrt(5)). The sets are the independent replicates; the
    runs inside a set are not, so they are not pooled into n=25. With a single round
    the interval falls back to the runs of that one set (``ci_basis`` says which),
  * ``std`` of the set means, ``pooled_std`` over all runs, ``median``, ``min``, and
    the throughput in tokens/s,
  * every individual run of every set in the JSON, so a number can be re-derived.

The plot draws the interval as error bars plus a shaded band. Successive forwards on one
machine are not perfectly independent (clock/thermal drift), so read the interval as the
run-to-run noise of *this* session, not as machine-to-machine variation.

One opt-in variant measures something different and says so in the figure title:

  * ``--sync_each_run`` synchronises after every forward. Each sample is then the
    latency of one *isolated* request from an idle GPU, which adds the CPU-side launch
    cost (dynamo guards, CUDA-graph input copies, any Python between graph segments)
    that back-to-back queueing hides behind the previous forward's GPU work. It weighs
    most on a launch-bound model at short lengths -- NexteraBERT more than the plain
    transformers -- so never compare it against numbers taken without it.

Models. ``--models`` takes the same aliases as ``model_pplbench`` (bert, roberta,
electra, deberta, deberta-v3, modernbert, neobert, lfm, lfm-350m, nextera) or any raw
Hub id / local directory; the default set and its order are the PPL bench's (BERT,
ELECTRA, DeBERTa, ModernBERT, NeoBERT, LFM2.5, NexteraBERT), so a model keeps its colour
and marker across the two figures. One alias differs on purpose: ``electra`` here is
``electra-base-discriminator``, the 110M encoder people actually deploy. The PPL bench
has to use the *generator* because only that half has a trained MLM head, but it is a
34M model (hidden 256) and timing it under the name "ELECTRA-base" would flatter it by
3x; ``electra-generator`` is still available. Baselines are loaded as bare encoders through
``hf_baselines.load_hf_encoder`` (NeoBERT's xformers stand-in and RoPE-table repair,
LFM's prefix repair). Speed does not depend on the weights, so NexteraBERT defaults to
an untrained ``--preset``; ``--nextera_path`` times a trained backbone's config instead,
and ``nextera:<preset>`` adds further presets as separate lines::

    --models modernbert nextera:piano nextera:mezzoforte

No length caps, as in the PPL bench: **every model is timed at every requested
length**, and ``max_position_embeddings`` is never used to skip one. RoPE models
(ModernBERT 8192, NeoBERT 4096 -- its baked RoPE table is re-sized to the sweep) and
DeBERTa's relative attention simply run longer. BERT / ELECTRA / RoBERTa look positions
up in a learned table of 512, and past its end a CUDA lookup is a device-side assert
that kills the whole sweep, so ``--extend_positions copy`` (default) tiles the table as
Longformer does -- the compute at a given length is identical to a natively longer
table, so the timing is fair -- and those points are marked ``[e]``. Only an explicit
``--extend_positions none`` brings the cap back (those lengths are then skipped). What
does end a line early is memory, below.

Attention backend. transformers >= 5 loads every baseline with ``sdpa`` unless told
otherwise -- ModernBERT no longer switches itself to FlashAttention. SDPA never
materialises the score matrix, but it is handed a materialised ``(B, 1, L, L)`` mask:
the padding mask cannot be dropped while tracing (checking "is it all ones?" is data
dependent, so under ``torch.compile`` it is always built), and ModernBERT's local
layers need a band mask on top. A custom mask also rules out the flash kernel, so the
memory-efficient one runs and a sliding-window layer pays the full L^2. That, not
attention scores, is why a baseline's memory grows with the length under sdpa.

So ``--attn_implementation auto`` (the default) puts **every baseline that can on
FlashAttention-2**: it is requested per model when the ``flash-attn`` package (or
``kernels``, from which transformers fetches ``kernels-community/flash-attn2``) is
installed -- with neither, every baseline silently stays on sdpa, which is warned
about once -- and a model that does not support it (DeBERTa's disentangled attention has
no FlashAttention form; remote-code models decide for themselves) falls back to its
default backend. A model that ends up on FlashAttention is not handed the all-ones
mask -- FlashAttention is unpadded by definition, the batch has no padding, and
transformers would otherwise test ``mask.all()`` in every forward, a data-dependent
graph break. The backend each model really ran on, and whether it got the mask, are
printed and recorded per model (``attention_backend``, ``attention_mask_passed``), so
a figure can say which lines are FlashAttention and which are not. ``default`` /
``sdpa`` reproduce the all-sdpa condition. What ``auto`` resolved to is recorded too
(``attention_requested``): a baseline's cached sets are dropped when it differs, so
installing flash-attn after an sdpa run re-measures instead of reusing that run.

Out of memory. On CUDA the peak allocation of every measured length is recorded, and
before each new length the activation memory *per sequence* is extrapolated from the
two longest measured ones as ``a*L + b*L^2`` (linear for fused attention, quadratic
once a model materialises the L x L scores -- the fit finds the mix) and multiplied by
the batch size that length will use. If the predicted peak exceeds
``--oom_margin`` (default 0.9) of what the device can still give this process, that
length and every longer one are skipped for that model *without being attempted*:
a real OOM costs a full compile first, can leave the allocator fragmented for the
models after it, and on Windows may not raise at all -- the driver spills into shared
system memory and the "measurement" crawls for minutes. A length that still runs out
of memory despite the prediction is caught and skips the longer ones the same way.
Skipped points carry ``skip_reason`` in the JSON and are re-decided on every rerun; a
*predicted* skip is also re-decided in every round, since round 1 runs cold and records
the highest peaks. Memory is logged as two numbers per set: the **peak**, taken over
warm-up and timed forwards together, and the peak of the **timed** forwards alone.
Read the first one. Under ``reduce-overhead`` a CUDA-graph replay does not go through
the caching allocator at all -- its activations live in the graph's private pool,
which the allocator counts as reserved, not allocated -- so the timed-only number
collapses to the weights for *every* model and says nothing; the capture during
warm-up is where the activation memory is actually seen. The timed-only number is
meaningful with ``--compile none`` / ``default``, where it separates the model's
steady need from one-off costs (tracing, autotuning, cuDNN algorithm search).

Because every length carries the same token budget, the peak is flat across lengths
for a model whose memory is linear in tokens, and grows with the length only through
an L x L term: ``peak - weights = tokens * (a + b*L)``. A flat line is therefore the
signature of "nothing quadratic is materialised", not of a broken measurement.

The JSON doubles as the resume log and is rewritten after every model of every round:
a ``(model, length)`` that already holds a round's set is not re-measured for that
round, a model with nothing left to do in a round is not loaded, and raising
``--rounds`` later only measures the additional rounds. The settings and the device
must match or the file is ignored; ``--refresh`` re-measures everything and
``--refresh deberta`` only the named models.

Examples
--------
    # the default comparison: 5 rounds x 5 runs, 95 percent CI over the rounds
    python src/nexterabert/model_speedbench.py

    # quick CPU smoke test
    python src/nexterabert/model_speedbench.py --models bert nextera:pianissimo \
        --seq_lengths 128 256 --runs 5 --warmup 1 --compile none
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import torch

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from nexterabert.configuration_nexterabert import (  # noqa: E402
    PRESET_NAMES,
    NexteraBERTConfig,
)
from nexterabert.hf_baselines import is_neobert, load_hf_encoder  # noqa: E402
from nexterabert.loading import load_backbone_state  # noqa: E402
# Importing the PPL bench also forces UTF-8 stdout, enables cudnn.benchmark and sets
# dynamo's suppress_errors, exactly as this script used to do for itself.
from nexterabert.model_pplbench import (  # noqa: E402
    MODEL_ALIASES,
    NEXTERA_NAME,
    display_name as _ppl_display_name,
    extend_position_embeddings,
    hf_max_len,
    make_deberta_autocast_safe,
    resolve_device,
    resolve_model,
    series_style,
    usable_position_len,
    uses_rope,
)
from nexterabert.modeling_nexterabert import NexteraBERT  # noqa: E402
from nexterabert.streaming import (  # noqa: E402
    DEFAULT_TOKENIZER,
    build_tokenizer,
    padded_vocab_size,
)

import torch._dynamo  # noqa: E402

# ==============================
# 設定
# ==============================
# Same set and order as model_pplbench.DEFAULT_MODELS: series_style() is indexed by
# position, so this is what gives a model the same colour in both figures.
DEFAULT_MODELS = ["bert", "electra", "deberta", "modernbert", "neobert", "lfm",
                  "nextera"]

# Where the speed bench deliberately departs from the PPL bench's registry. The PPL
# bench needs ELECTRA's generator (the only half with an MLM head); for latency the
# subject is the discriminator -- the BERT-base-sized encoder that gets fine-tuned.
SPEED_ALIASES = {
    "electra": "google/electra-base-discriminator",
    "electra-generator": "google/electra-base-generator",
}
# The PPL bench labels the RTD checkpoints "(untrained MLM head)", which is a warning
# about perplexity and means nothing on a latency plot.
SPEED_DISPLAY_NAMES = {
    "google/electra-base-discriminator": "ELECTRA-base",
    "microsoft/deberta-v3-base": "DeBERTa-v3-base",
}


def display_name(name):
    return SPEED_DISPLAY_NAMES.get(name) or _ppl_display_name(name)

DEFAULT_SEQUENCE_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
# One full-length sequence at the top of the sweep; shorter lengths fill the same budget.
DEFAULT_BATCH_TOKENS = max(DEFAULT_SEQUENCE_LENGTHS)
DEFAULT_RUNS = 5        # forwards per set
DEFAULT_ROUNDS = 5      # sets per (model, length), one per pass over the models
DEFAULT_WARMUP = 5

NEXTERA_PRESET_PREFIX = "nextera:"


def is_nextera(name):
    return name == NEXTERA_NAME or name.startswith("NexteraBERT-")


def resolve_speed_model(name, optibert=None):
    """``model_pplbench.resolve_model`` plus ``nextera:<preset>``.

    Returns ``(model_id, preset)``; ``preset`` is only set for the explicit-preset form,
    which always builds an untrained model of that size."""
    if name.lower().startswith(NEXTERA_PRESET_PREFIX):
        preset = name[len(NEXTERA_PRESET_PREFIX):].lower()
        if preset not in PRESET_NAMES:
            raise SystemExit(f"[speed] unknown preset {preset!r}; choose one of "
                             f"{', '.join(PRESET_NAMES)}")
        return f"NexteraBERT-{preset} (Ours)", preset
    if name.lower() in SPEED_ALIASES:
        return SPEED_ALIASES[name.lower()], None
    return resolve_model(name, optibert), None


# ==============================
# 統計
# ==============================
def t_critical(df, confidence):
    """Two-sided Student-t critical value, e.g. 2.093 for df=19 at 95 percent."""
    p = 0.5 + confidence / 2.0
    try:
        from scipy import stats

        return float(stats.t.ppf(p, df))
    except ImportError:
        pass
    # Cornish-Fisher expansion around the normal quantile; within 1e-3 for df >= 5.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def summarize(times, confidence):
    """Mean, sample std and the t confidence interval of the mean over ``times``."""
    nan = float("nan")
    good = sorted(t for t in times if math.isfinite(t))
    n = len(good)
    if n == 0:
        return {"mean": nan, "std": nan, "ci_low": nan, "ci_high": nan,
                "median": nan, "min": nan, "n": 0}
    mean = sum(good) / n
    median = good[n // 2] if n % 2 else 0.5 * (good[n // 2 - 1] + good[n // 2])
    if n == 1:      # no dispersion estimate from a single run
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean,
                "median": median, "min": good[0], "n": 1}
    std = math.sqrt(sum((t - mean) ** 2 for t in good) / (n - 1))
    half = t_critical(n - 1, confidence) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half,
            "median": median, "min": good[0], "n": n}


def summarize_point(pt, confidence):
    """Summary of one ``(model, length)`` from its sets (one per round).

    The interval is over the *set means*: each set comes from its own load + compile,
    so those are the independent replicates, while the runs inside a set share
    everything. Only when a single set exists is the interval taken over its runs."""
    sets = [[t for t in st if math.isfinite(t)] for st in pt.get("sets", [])]
    sets = [st for st in sets if st]
    runs = [t for st in sets for t in st]
    set_means = [sum(st) / len(st) for st in sets]
    basis = "sets" if len(sets) >= 2 else "runs"
    out = summarize(set_means if basis == "sets" else runs, confidence)
    pooled = summarize(runs, confidence)
    out.update(median=pooled["median"], min=pooled["min"], pooled_std=pooled["std"],
               n_sets=len(sets), n_runs=len(runs), set_means=set_means,
               ci_basis=basis if runs else None)
    return out


# ==============================
# モデルのロード
# ==============================
def load_nextera_encoder(path, preset, tokenizer_name, device):
    """A bare ``NexteraBERT`` encoder. Latency does not depend on the weights, so an
    untrained preset is a valid subject; a backbone directory fixes the *config*
    (and loads its weights, which is cheap and keeps the run honest about what it is)."""
    trained = bool(path) and os.path.isdir(path) and \
        os.path.exists(os.path.join(path, "config.json"))
    if trained:
        config = NexteraBERTConfig.from_pretrained(path)
        model = NexteraBERT(config)
        model.load_state_dict(load_backbone_state(path), strict=False)
        print(f"[speed] NexteraBERT config + weights from {path}")
    else:
        tokenizer = build_tokenizer(tokenizer_name)
        config = NexteraBERTConfig.from_preset(
            preset, vocab_size=padded_vocab_size(tokenizer))
        model = NexteraBERT(config)
        print(f"[speed] NexteraBERT untrained '{preset}' preset "
              f"(weights do not affect latency)")
    return model.to(device).eval(), config.vocab_size, None, False, None


def make_deberta_compile_friendly(config):
    """Let DeBERTa (v1) compile into ONE graph, like every other model here.

    Its ``compute_attention_span`` wraps a plain Python int in ``torch.tensor(...)``
    (a leftover of jit.trace support) and the attention then uses that tensor as a
    slice bound, which is a ``Tensor.item()`` call: dynamo breaks the graph there in
    every layer -- 9 graphs and 8 breaks for one forward. Under ``reduce-overhead``
    that means a chain of small CUDA graphs with Python in between instead of one
    replay, a handicap that comes from the wrapper, not from the architecture being
    timed. Returning the int itself gives 1 graph / 0 breaks and bit-identical output.
    The module-level function is replaced, which is what the attention looks up."""
    if getattr(config, "model_type", None) != "deberta":
        return False
    import transformers.models.deberta.modeling_deberta as deberta

    def compute_attention_span(query_layer, key_layer, max_relative_positions):
        return min(max(query_layer.size(-2), key_layer.size(-2)),
                   max_relative_positions)

    deberta.compute_attention_span = compute_attention_span
    print("[speed] DeBERTa: attention span kept as a Python int so torch.compile "
          "captures one graph (the released code graph-breaks on Tensor.item() in "
          "every layer).")
    return True


_FLASH_ATTN_NOTE = []


def flash_attn_installed():
    """Whether FlashAttention-2 can be requested here: the flash-attn package, or the
    ``kernels`` package (transformers then fetches ``kernels-community/flash-attn2``
    in its place). Says so once when neither is there."""
    import importlib.util

    found = any(importlib.util.find_spec(pkg) is not None
                for pkg in ("flash_attn", "kernels"))
    if not found and not _FLASH_ATTN_NOTE:
        _FLASH_ATTN_NOTE.append(True)
        print("[speed] [warn] neither flash-attn nor kernels is installed in this "
              "environment -- EVERY baseline (ModernBERT, LFM, ...) runs on "
              "transformers' default backend (sdpa). `pip install flash-attn` (or "
              "`pip install kernels`) lets --attn_implementation auto put them on "
              "FlashAttention; note that `uv sync` removes a flash-attn that was "
              "pip-installed by hand.")
    return found


def requested_attention(choice):
    """What ``--attn_implementation`` hands to ``from_pretrained`` on this machine
    (``None`` = transformers' own choice)."""
    if choice == "default":
        return None
    if choice == "auto":
        return "flash_attention_2" if flash_attn_installed() else None
    return choice


def uses_flash_attention(model):
    backend = getattr(getattr(model, "config", None), "_attn_implementation", None)
    return isinstance(backend, str) and "flash" in backend


def load_hf_speed_encoder(name, device, target_len, extend_positions,
                          autocast_dtype=None, attn_implementation=None):
    """Bare 🤗 encoder + the positional-cap facts the sweep needs.

    Returns ``(model, vocab_size, max_len, hard_cap, extended_from)`` with the same
    meaning as in ``model_pplbench.load_hf_masked_lm``."""
    auto = attn_implementation == "auto"
    attn_implementation = requested_attention(attn_implementation)
    try:
        model, config = load_hf_encoder(name, max_len=target_len,
                                        attn_implementation=attn_implementation)
    except Exception as e:  # noqa: BLE001 - the kernels fallback raises anything
        if attn_implementation is None:
            raise
        # A model (or this machine) that cannot do the requested backend is timed on
        # its default one rather than dropped; the JSON records which it really was.
        # Under 'auto' that is the expected outcome for some models (DeBERTa's
        # disentangled attention has no FlashAttention form), not a problem.
        print(f"[speed] {'' if auto else '[warn] '}{name}: "
              f"attn_implementation={attn_implementation!r} is not available "
              f"({type(e).__name__}: {str(e)[:400]}); using the default backend.")
        model, config = load_hf_encoder(name, max_len=target_len)
    print(f"[speed] {name}: attention backend = "
          f"{getattr(model.config, '_attn_implementation', None)!r}")
    model = model.to(device).eval()
    make_deberta_autocast_safe(model, config, autocast_dtype, tag="speed")
    make_deberta_compile_friendly(config)
    max_len = hf_max_len(config)
    # Only a learned table that is actually there can be indexed out of range.
    real_cap = usable_position_len(model)
    hard_cap = (not uses_rope(config, model)) and real_cap is not None
    if hard_cap and (max_len is None or real_cap < max_len):
        max_len = real_cap
    extended_from = None
    if hard_cap and extend_positions != "none" and max_len and target_len > max_len:
        if extend_position_embeddings(model, int(target_len), extend_positions):
            extended_from, hard_cap = max_len, False
    if hard_cap:
        print(f"[speed] {name}: fixed position table of {max_len}; longer lengths are "
              f"skipped (--extend_positions none).")
    return model, int(config.vocab_size), max_len, hard_cap, extended_from


# ==============================
# 計測
# ==============================
def _is_oom(err):
    msg = str(err).lower()
    return any(h in msg for h in ("out of memory", "not enough memory",
                                  "can't allocate", "bad allocation"))


def _free(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def batch_size_for(args, seq_len):
    """Sequences per batch at ``seq_len``: a fixed ``--batch_size`` when given, else
    whatever keeps the batch at about ``--batch_tokens`` tokens."""
    if args.batch_size:
        return args.batch_size
    return max(1, min(args.max_batch_size, args.batch_tokens // seq_len))


def predict_peak_memory(history, seq_len, base_now=None, batch_size=1):
    """Peak bytes expected for ``batch_size`` sequences of ``seq_len``, from
    ``history = [(L, base, peak, batch), ...]`` (``batch`` optional, default 1).
    ``base_now`` replaces the last measured base with what is allocated right now
    (when compiling once, earlier lengths' CUDA-graph pools stay resident and count).

    ``base`` is what was allocated before the forward (weights), ``peak - base`` the
    activation memory, taken per sequence since the batch size varies with the
    length. That part is fitted as ``a*L + b*L^2`` through the two longest
    measured lengths below ``seq_len``, both coefficients kept non-negative: ``b == 0``
    is a fused-attention model, ``a == 0`` one dominated by a materialised L x L score
    matrix. From a single point only the linear term can be had, which under-predicts
    -- on purpose, so a model is never skipped on the strength of one measurement
    unless even linear growth would not fit. Returns ``None`` with no usable history."""
    below = sorted((h for h in history
                    if h[0] < seq_len and h[1] is not None and h[2] is not None),
                   key=lambda h: h[0])
    if not below:
        return None
    def per_sequence(h):
        return max(h[2] - h[1], 0) / (h[3] if len(h) > 3 and h[3] else 1)

    L2, base = below[-1][0], below[-1][1]
    m2 = per_sequence(below[-1])
    a, b = m2 / L2, 0.0
    if len(below) >= 2:
        L1, m1 = below[-2][0], per_sequence(below[-2])
        b = (m2 / L2 - m1 / L1) / (L2 - L1)
        if b <= 0:
            a, b = m2 / L2, 0.0
        else:
            a = m1 / L1 - b * L1
            if a < 0:
                a, b = 0.0, m2 / L2 ** 2
    return (base if base_now is None else base_now) \
        + batch_size * (a * seq_len + b * seq_len ** 2)


def cuda_capacity():
    """Bytes this process could hold at once: what the device reports free (other
    processes' share is already excluded) plus what our allocator has reserved."""
    free, _total = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved()


def time_forward(step, runs, warmup, device, sync_each_run=False, after_warmup=None):
    """``runs`` individually-timed forwards, in seconds, after ``warmup`` untimed ones.

    On CUDA the forwards are queued back to back and synchronised once at the end, as
    the original single-block timing did; each one is merely bracketed by its own pair
    of events. Start event i lands on the stream directly behind end event i-1, so the
    samples tile the block (any GPU idle gap waiting on the CPU falls into the next
    sample) and their mean equals the old ``elapsed / runs``. ``sync_each_run`` instead
    drains the GPU after every forward: isolated-request latency, CPU launch included."""
    for _ in range(warmup):
        step()
    if after_warmup is not None:
        after_warmup()
    if device == "cuda":
        torch.cuda.synchronize()
        events = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            if sync_each_run:
                torch.cuda.synchronize()
            events.append((start, end))
        torch.cuda.synchronize()
        return [start.elapsed_time(end) / 1000.0 for start, end in events]
    if device == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        step()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return times


def _new_point():
    return {"sets": [], "steady_memory": [], "warmup_memory": [],
            "extended": False, "base_memory": None, "peak_memory": None,
            "skip_reason": None, "batch_size": None}


def _point_from_cache(pt):
    out = _new_point()
    out["sets"] = [[v if v is not None else float("nan") for v in st]
                   for st in pt.get("sets", [])]
    out["extended"] = bool(pt.get("position_table_extended"))
    out["base_memory"] = pt.get("base_memory_bytes")
    out["peak_memory"] = pt.get("peak_memory_bytes")
    out["steady_memory"] = list(pt.get("steady_memory_bytes_per_set") or [])
    out["warmup_memory"] = list(pt.get("warmup_memory_bytes_per_set") or [])
    out["batch_size"] = pt.get("batch_size")
    return out


def new_result(args, cached=None, cached_meta=None):
    """The mutable per-model record the rounds append to."""
    meta = cached_meta or {}
    points = {L: _point_from_cache(p) for L, p in (cached or {}).items()}
    for L in args.seq_lengths:
        points.setdefault(L, _new_point())
    return {"points": points, "params": meta.get("params"),
            "attn_requested": meta.get("attention_requested"),
            "attn_backend": meta.get("attention_backend"),
            "mask_passed": meta.get("attention_mask_passed"),
            "max_len": meta.get("max_position"),
            "extended_from": meta.get("position_table_extended_from"),
            "error": meta.get("load_error")}


def benchmark_model(name, preset, args, device, autocast_ctx, res, round_idx):
    """One round for one model: load it, time ONE set at every length that does not
    have this round's set yet, free it. Appends to ``res`` in place."""
    points = res["points"]
    # A length skipped earlier in this process for a hard reason (position cap, a real
    # OOM, an error) would be skipped again; do not reload a model just to re-decide
    # that. A *predicted* OOM is different: the prediction rests on the peaks recorded
    # so far, and a cold first round records higher ones (compile, autotuning, one-off
    # table builds) than a warm later round -- so it is re-decided every round.
    todo = [L for L in args.seq_lengths
            if len(points[L]["sets"]) <= round_idx
            and points[L]["skip_reason"] in (None, "predicted_oom")]
    for L in todo:
        points[L]["skip_reason"] = None
    if not todo:
        print(f"[speed] {display_name(name)}: nothing to measure in round "
              f"{round_idx + 1} -- model not loaded.")
        return
    recorded = [L for L in args.seq_lengths if len(points[L]["sets"]) > round_idx]
    if recorded:
        print(f"[speed] {display_name(name)}: round {round_idx + 1} already recorded "
              f"for {recorded}, measuring {todo}.")

    drop_mask = False
    try:
        if is_nextera(name):
            model, vocab, max_len, hard_cap, extended_from = load_nextera_encoder(
                None if preset else args.nextera_path, preset or args.preset,
                args.tokenizer, device)
        else:
            model, vocab, max_len, hard_cap, extended_from = load_hf_speed_encoder(
                name, device, max(args.seq_lengths), args.extend_positions,
                getattr(autocast_ctx, "fast_dtype", None),
                None if args.attn_implementation == "default"
                else args.attn_implementation)
            if uses_flash_attention(model) and args.attention_mask:
                # FlashAttention is unpadded by definition: handed a mask, transformers
                # first asks `mask.all()` (data dependent -> a graph break in every
                # forward) and, were there padding, would unpad into variable-length
                # kernels that CUDA graphs cannot hold. This batch has no padding, so
                # the all-ones mask carries no information -- leave it out.
                drop_mask = True
                print(f"[speed] {display_name(name)}: on FlashAttention -- the "
                      f"all-ones attention mask is not passed (no padding in the "
                      f"batch; it would only add a data-dependent graph break).")
    except Exception as e:  # noqa: BLE001 - report and skip a model that won't load
        import traceback

        print(f"[speed] [ERROR] could not load {name}:")
        traceback.print_exc()
        res["error"] = f"{type(e).__name__}: {e}"
        for L in todo:
            points[L]["skip_reason"] = "load_error"
        return

    params = sum(p.numel() for p in model.parameters())
    res.update(params=params, max_len=max_len, extended_from=extended_from, error=None,
               attn_requested=(None if is_nextera(name)
                               else requested_attention(args.attn_implementation)),
               attn_backend=getattr(getattr(model, "config", None),
                                    "_attn_implementation", None),
               mask_passed=bool(args.attention_mask and not drop_mask))
    print(f"[speed] {display_name(name)}: {params / 1e6:.1f}M parameters")
    neobert = is_neobert(model)
    gen = torch.Generator().manual_seed(args.seed + round_idx)
    oom_at, oom_reason = None, None
    if args.compile != "none":
        # Exactly the original script: one compile per model, reused for every length.
        model = torch.compile(model, mode=args.compile)

    for seq_len in args.seq_lengths:
        pt = points[seq_len]
        if seq_len not in todo:
            continue
        if hard_cap and max_len is not None and seq_len > max_len:
            # Out-of-range position lookup = device-side assert on CUDA, which takes
            # the rest of the sweep down with it. Never attempt it.
            print(f"{name} | seq_len={seq_len} | skip (position table stops at "
                  f"{max_len}; use --extend_positions copy)")
            pt["skip_reason"] = "position_cap"
            continue
        if oom_at is not None and seq_len >= oom_at:
            print(f"{name} | seq_len={seq_len} | skip (out of memory "
                  f"{'predicted ' if oom_reason == 'predicted_oom' else ''}at {oom_at})")
            pt["skip_reason"] = oom_reason
            continue
        batch = batch_size_for(args, seq_len)
        if device == "cuda" and args.oom_margin > 0:
            _free(device)
            predicted = predict_peak_memory(
                [(L, q["base_memory"], q["peak_memory"], q["batch_size"])
                 for L, q in points.items()],
                seq_len, base_now=torch.cuda.memory_allocated(), batch_size=batch)
            budget = cuda_capacity() * args.oom_margin
            if predicted is not None and predicted > budget:
                oom_at, oom_reason = seq_len, "predicted_oom"
                print(f"{name} | seq_len={seq_len} | skip -- predicted peak "
                      f"{predicted / 2**30:.1f} GiB exceeds the "
                      f"{budget / 2**30:.1f} GiB budget (--oom_margin "
                      f"{args.oom_margin:g}); skipping this and longer lengths")
                pt["skip_reason"] = oom_reason
                continue

        input_ids = torch.randint(0, vocab, (batch, seq_len),
                                  generator=gen).to(device)
        attention_mask = None
        if args.attention_mask and not drop_mask:
            attention_mask = torch.ones_like(input_ids)
            if neobert:
                # NeoBERT expands the (B, L) mask to (B, H, L, L) with `repeat`; bool
                # keeps that 8x smaller than int64.
                attention_mask = attention_mask.bool()

        def step(input_ids=input_ids, attention_mask=attention_mask):
            model(input_ids=input_ids, attention_mask=attention_mask)

        base_mem = peak_mem = steady_mem = None
        warm = {}

        def after_warmup():
            # Split the peak in two. The warm-up holds everything that happens once:
            # tracing, autotuning, cuDNN algorithm search, CUDA-graph capture, shape-
            # keyed tables being built. What the timed forwards need is the number that
            # describes the model.
            if device == "cuda":
                torch.cuda.synchronize()
                warm["peak"] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

        if device == "cuda":
            torch.cuda.synchronize()
            base_mem = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode(), autocast_ctx:
                times = time_forward(step, args.runs, args.warmup, device,
                                     args.sync_each_run, after_warmup)
            if device == "cuda":
                steady_mem = torch.cuda.max_memory_allocated()
                # The OOM prediction keeps using the overall peak: a length has to get
                # through its warm-up before it can be timed at all.
                peak_mem = max(steady_mem, warm.get("peak", 0))
        except (RuntimeError, IndexError, AssertionError) as e:
            oom = _is_oom(e)
            if oom:
                oom_at, oom_reason = seq_len, "oom"
                print(f"{name} | seq_len={seq_len} | out of memory -- skipping this "
                      f"and longer lengths")
            else:
                print(f"{name} | seq_len={seq_len} | [warn] {type(e).__name__}: {e}")
            pt["skip_reason"] = "oom" if oom else "error"
            del step
            _free(device)
            continue

        pt["sets"].append(times)
        pt["batch_size"] = batch
        pt["extended"] = extended_from is not None and seq_len > extended_from
        if peak_mem is not None:
            pt["base_memory"], pt["peak_memory"] = base_mem, peak_mem
            pt["steady_memory"].append(steady_mem)
            pt["warmup_memory"].append(warm.get("peak"))
        this = summarize(times, args.confidence)
        tok_s = batch * seq_len / this["mean"]
        print(f"{name} | round {round_idx + 1} | seq_len={seq_len} x batch {batch} | "
              f"{this['mean'] * 1e3:.3f} ms/batch (set of {this['n']}, "
              f"std={this['std'] * 1e3:.3f} ms, {tok_s:,.0f} tok/s"
              + (f", peak {peak_mem / 2**30:.2f} GiB "
                 f"(timed-only {steady_mem / 2**30:.2f})" if peak_mem else "")
              + ")"
              f"{' [e]' if pt['extended'] else ''}")
        del step, input_ids, attention_mask

    del model
    if args.compile != "none":
        # Between models only: releases this model's CUDA-graph pools so seven models
        # do not pile up on the device. The next model compiles from scratch either way.
        torch._dynamo.reset()
    _free(device)


# ==============================
# 結果の記録
# ==============================
def host_cpu():
    """CPU model and logical core count of the host, for the record.

    The GPU does the timed work, but the host still shows through wherever a model is
    launch-bound (eager graph segments, --sync_each_run, --compile none): that part
    scales with single-thread speed. Results taken on two hosts should be told apart,
    so the JSON says which one it was. Not part of the resume key -- swapping the CPU
    does not invalidate GPU-bound points."""
    import platform

    name = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"model": name or platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count()}


def device_name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    import platform

    return platform.processor() or platform.machine()


# Everything here moves the numbers, so a recorded point is only reusable when all of
# it matches -- including the device it was measured on.
def measurement_settings(args, device):
    return {
        "batch_size": args.batch_size,          # null = follow --batch_tokens
        "batch_tokens": args.batch_tokens,
        "max_batch_size": args.max_batch_size,
        # --rounds is deliberately absent: more rounds only ADD sets to a record.
        "runs_per_set": args.runs,
        "warmup": args.warmup,
        "confidence": args.confidence,
        "dtype": args.dtype,
        "compile": args.compile,
        "sync_each_run": args.sync_each_run,
        "attn_implementation": args.attn_implementation,
        "attention_mask": args.attention_mask,
        "extend_positions": args.extend_positions,
        "device": device,
        "device_name": device_name(device),
        "torch": torch.__version__,
    }


def load_cached_results(path, args, device):
    """Finite points already recorded in ``path``: ``({model_id: {seq_len: point}},
    {model_id: entry})``. Discarded wholesale when the settings differ."""
    import json

    if not path or not os.path.exists(path):
        return {}, {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # noqa: BLE001 - a truncated file is not a reason to stop
        print(f"[speed] [warn] could not read {path} ({type(e).__name__}); "
              f"starting fresh.")
        return {}, {}
    want, got = measurement_settings(args, device), payload.get("measurement", {})
    differing = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if differing:
        shown = ", ".join(f"{k}: recorded {old!r} vs now {new!r}"
                          for k, (old, new) in list(differing.items())[:3])
        print(f"[speed] {path} was written with different settings ({shown}) -- "
              f"ignoring it and re-measuring everything.")
        return {}, {}
    cached, meta = {}, {}
    for entry in payload.get("models", []):
        # The NexteraBERT line is only the same subject if it was built the same way.
        if entry.get("is_nextera") and (
                entry.get("nextera_path") != nextera_path_of(entry["id"], args)
                or entry.get("preset") != entry_preset(entry["id"], args)):
            continue
        # 'auto' is a per-machine decision: a baseline recorded while flash-attn was
        # missing (-> sdpa) must not be reused once FlashAttention can be requested,
        # or the model is never reloaded and stays on sdpa for good.
        if not entry.get("is_nextera"):
            backend = entry.get("attention_backend") or ""
            now = requested_attention(args.attn_implementation)
            # A record from before this field existed: only 'auto' is ambiguous.
            legacy = (("flash_attention_2" if "flash" in backend else None)
                      if args.attn_implementation == "auto" else now)
            was = entry.get("attention_requested", legacy)
            if was != now:
                if any(p.get("sets") for p in entry.get("by_seq_len", [])):
                    print(f"[speed] {entry.get('display', entry['id'])}: recorded "
                          f"with attn_implementation={was!r} (ran on {backend!r}), "
                          f"now {now!r} -- discarding its sets and re-measuring.")
                continue
        pts = {p["seq_len"]: p for p in entry.get("by_seq_len", []) if p.get("sets")}
        if pts:
            cached[entry["id"]] = pts
            meta[entry["id"]] = entry
    if cached:
        total = sum(len(p["sets"]) for v in cached.values() for p in v.values())
        print(f"[speed] reusing {total} recorded set(s) from {path} "
              f"({len(cached)} model(s)); pass --refresh to re-measure them.")
    return cached, meta


def entry_preset(name, args):
    if name.startswith("NexteraBERT-"):
        return name[len("NexteraBERT-"):].split(" ")[0]
    return args.preset


def nextera_path_of(name, args):
    """Only the plain 'nextera' entry reads --nextera_path; 'nextera:<preset>' never."""
    return args.nextera_path if name == NEXTERA_NAME else None


def _json_safe(x):
    return x if isinstance(x, (int, str, bool, type(None))) or (
        isinstance(x, float) and math.isfinite(x)) else None


def write_json(path, args, results, device, carry=None):
    """Record every run. ``carry`` holds previously recorded entries for models this
    call has no result for (not reached yet, or not requested this time); they are
    written back untouched so an interrupted or narrower sweep never erases them."""
    import json
    import platform
    from datetime import datetime, timezone

    payload = {
        "schema": "nexterabert.model_speedbench/2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": "forward latency in seconds per BATCH (batch_size is per point; "
                  "mean_per_sequence = mean / batch_size). sets = one list of "
                  "individually-timed runs per round; mean = mean of the set means; "
                  "ci = two-sided Student-t interval over the set means (over the runs "
                  "when only one set exists -- see ci_basis)",
        "rounds": args.rounds,
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "cpu": host_cpu()},
        "measurement": measurement_settings(args, device),
        "seq_lengths": list(args.seq_lengths),
        "models": [],
    }
    for name, res in results.items():
        entry = {
            "id": name,
            "display": display_name(name),
            "is_nextera": is_nextera(name),
            "params": res.get("params"),
            # What the model really ran on (sdpa / flash_attention_2 / ...). With sdpa
            # a padding or sliding-window mask is an L x L tensor and rules out the
            # flash kernel, so this decides both the memory curve and the speed.
            "attention_backend": res.get("attn_backend"),
            # What from_pretrained was asked for (null = transformers' choice). 'auto'
            # resolves per machine, so a record is only reused while this still matches.
            "attention_requested": res.get("attn_requested"),
            # False for a model on FlashAttention: the all-ones mask is left out.
            "attention_mask_passed": res.get("mask_passed"),
            "max_position": res.get("max_len"),
            "position_table_extended_from": res.get("extended_from"),
            "load_error": res.get("error"),
            "by_seq_len": [],
        }
        if is_nextera(name):
            entry["nextera_path"] = nextera_path_of(name, args)
            entry["preset"] = entry_preset(name, args)
        for seq_len in args.seq_lengths:
            pt = res["points"][seq_len]
            sm = summarize_point(pt, args.confidence)
            mean = sm["mean"]
            batch = pt.get("batch_size") or batch_size_for(args, seq_len)
            entry["by_seq_len"].append({
                "seq_len": seq_len,
                "batch_size": batch,
                "mean_per_sequence": _json_safe(mean / batch),
                "n_sets": sm["n_sets"],
                "n_runs": sm["n_runs"],
                "mean": _json_safe(mean),
                "std": _json_safe(sm["std"]),
                "ci_low": _json_safe(sm["ci_low"]),
                "ci_high": _json_safe(sm["ci_high"]),
                "ci_basis": sm["ci_basis"],
                "pooled_std": _json_safe(sm["pooled_std"]),
                "median": _json_safe(sm["median"]),
                "min": _json_safe(sm["min"]),
                "tokens_per_sec": _json_safe(
                    batch * seq_len / mean if math.isfinite(mean) else mean),
                "set_means": [_json_safe(v) for v in sm["set_means"]],
                "sets": [[_json_safe(v) for v in st] for st in pt["sets"]],
                "position_table_extended": bool(pt["extended"]),
                "base_memory_bytes": pt.get("base_memory"),
                "peak_memory_bytes": pt.get("peak_memory"),
                # One entry per set. warmup = peak during the untimed forwards
                # (compile, autotune, CUDA-graph capture): THE memory figure under
                # reduce-overhead. steady = peak during the timed forwards only, which
                # under CUDA graphs is just the weights (replays bypass the allocator).
                "steady_memory_bytes_per_set": pt.get("steady_memory"),
                "warmup_memory_bytes_per_set": pt.get("warmup_memory"),
                # position_cap / predicted_oom / oom / error; null when measured
                "skip_reason": pt.get("skip_reason"),
            })
        payload["models"].append(entry)
    payload["models"].extend(e for k, e in (carry or {}).items() if k not in results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[speed] results -> {path}")


# ==============================
# プロット
# ==============================
def _style(index, name, n_nextera_seen):
    if is_nextera(name):
        # Every NexteraBERT line is the thick black one; presets differ by dash.
        style = dict(series_style(index, NEXTERA_NAME))
        style["linestyle"] = ["-", "--", ":", "-."][n_nextera_seen % 4]
        return style
    return series_style(index, name)


def _label(name, res):
    params = res.get("params")
    tags = [f"{params / 1e6:.0f}M"] if params else []
    backend = res.get("attn_backend") or ""
    if "flash" in backend:      # say which baselines are on FlashAttention
        tags.append("FA3" if backend.endswith("3") else "FA2")
    return display_name(name) + (f" [{', '.join(tags)}]" if tags else "")


def batch_note(args):
    if args.batch_size:
        return f"batch={args.batch_size}"
    return f"batch={args.batch_tokens} tokens / length (max {args.max_batch_size})"


def _write_figure(path, results, args, *, throughput):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_nextera = 0
    for i, (name, res) in enumerate(results.items()):
        style = _style(i, name, n_nextera)
        n_nextera += is_nextera(name)
        pts = [(L, summarize_point(res["points"][L], args.confidence))
               for L in args.seq_lengths]
        pts = [(L, p) for L, p in pts if math.isfinite(p["mean"])]
        if not pts:
            continue
        xs = [L for L, _ in pts]
        batches = [res["points"][L]["batch_size"] or batch_size_for(args, L)
                   for L in xs]
        if throughput:
            # tokens/s is monotone decreasing in latency, so the interval flips.
            tokens = [B * L for B, L in zip(batches, xs)]
            ys = [t / p["mean"] for t, (_, p) in zip(tokens, pts)]
            lo = [t / p["ci_high"] for t, (_, p) in zip(tokens, pts)]
            hi = [t / max(p["ci_low"], 1e-12) for t, (_, p) in zip(tokens, pts)]
        else:
            # Per sequence: the batch size differs from length to length, so the raw
            # per-batch time would not be comparable along the x axis.
            ys = [p["mean"] / B for B, (_, p) in zip(batches, pts)]
            # Clamp for the log axis: a lower bound <= 0 drops the point.
            lo = [max(p["ci_low"], p["mean"] * 0.01) / B
                  for B, (_, p) in zip(batches, pts)]
            hi = [p["ci_high"] / B for B, (_, p) in zip(batches, pts)]
        yerr = [[y - a for y, a in zip(ys, lo)], [b - y for y, b in zip(ys, hi)]]
        ax.errorbar(xs, ys, yerr=yerr, label=_label(name, res), capsize=3,
                    elinewidth=1.1, **style)
        ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, linewidth=0)

    ax.set_xscale("log", base=2)
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xticks(args.seq_lengths)
    ax.set_xticklabels([str(s) for s in args.seq_lengths], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(("Throughput (tokens/sec)" if throughput
                   else "Inference Time (sec / sequence)")
                  + (" (log)" if args.log_y else ""))
    ax.set_title(
        ("Model Throughput vs Sequence Length" if throughput
         else "Model Speed vs Sequence Length")
        + f"\n{batch_note(args)}, {args.rounds} rounds x {args.runs} runs, "
          f"bars/band = {args.confidence * 100:g}% CI over rounds"
          f"\ncompile={args.compile}, {args.dtype}"
          f"{', synchronised per run' if args.sync_each_run else ''}",
        fontsize=10)
    ax.grid(alpha=0.3, which="both" if args.log_y else "major")
    ax.legend(fontsize=9, ncol=2 if len(results) > 4 else 1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[speed] saved -> {path}")


def plot(results, args):
    _write_figure(args.output, results, args, throughput=False)
    base, ext = os.path.splitext(args.output)
    _write_figure(f"{base}_throughput{ext or '.png'}", results, args, throughput=True)


# ==============================
# CLI / 実行
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"models to time, as short aliases, 'nextera:<preset>', or raw "
                        f"Hub ids / local dirs. Aliases: "
                        f"{', '.join(sorted(set(MODEL_ALIASES) | set(SPEED_ALIASES)))} "
                        f"('electra' is the 110M discriminator here, not the PPL "
                        f"bench's 34M generator). "
                        f"Default: {' '.join(DEFAULT_MODELS)}")
    p.add_argument("--optibert", default=None,
                   help="Hub id or local directory to time as the 'optibert' entry "
                        "(the paper released a recipe, not weights).")
    p.add_argument("--nextera_path", default=None,
                   help="NexteraBERT backbone directory whose config (and weights) the "
                        "'nextera' entry uses. Default: an untrained --preset, since "
                        "latency does not depend on the weights.")
    p.add_argument("--preset", default="mezzoforte", choices=PRESET_NAMES,
                   help="preset for the 'nextera' entry when --nextera_path is not given")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="tokenizer that sizes an untrained NexteraBERT's vocabulary")
    p.add_argument("--seq_lengths", nargs="+", type=int,
                   default=DEFAULT_SEQUENCE_LENGTHS,
                   help=f"sequence lengths to sweep (default: {DEFAULT_SEQUENCE_LENGTHS})")
    p.add_argument("--batch_tokens", type=int, default=DEFAULT_BATCH_TOKENS,
                   help="token budget per batch: batch size = batch_tokens // seq_len, "
                        "so short lengths get a large batch and keep the device busy "
                        f"(default {DEFAULT_BATCH_TOKENS} = 64 x 1024 ... 1 x 65536)")
    p.add_argument("--max_batch_size", type=int, default=128,
                   help="upper clamp on the derived batch size (default 128)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="fixed batch size at every length instead of the token budget "
                        "(the old behaviour was 2). Under-fills the device at short "
                        "lengths, so tokens/s drops there.")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help=f"individually-timed forwards in one set, queued back to back "
                        f"(default {DEFAULT_RUNS})")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help=f"passes over the whole model list; every round reloads and "
                        f"recompiles each model and adds one set per length. The "
                        f"confidence interval is over the set means "
                        f"(default {DEFAULT_ROUNDS})")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help="untimed forwards before measuring; absorbs torch.compile "
                        f"tracing and CUDA-graph capture (default {DEFAULT_WARMUP})")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="level of the two-sided Student-t interval of the mean "
                        "(default 0.95)")
    p.add_argument("--compile", default="reduce-overhead",
                   choices=["none", "default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. reduce-overhead = CUDA graphs, which is "
                        "what helps a launch-bound model at short lengths; "
                        "max-autotune favours long, compute-bound sequences; none = "
                        "eager.")
    p.add_argument("--sync_each_run", action="store_true",
                   help="synchronise after every forward: isolated-request latency, "
                        "including the CPU launch cost that back-to-back queueing "
                        "hides. Off by default -- the forwards are queued back to back "
                        "and synchronised once, as the original script did. Penalises "
                        "launch-bound models at short lengths; not comparable with "
                        "numbers taken without it.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="autocast dtype on CUDA (CPU / MPS run without autocast)")
    p.add_argument("--no_attention_mask", dest="attention_mask", action="store_false",
                   help="pass attention_mask=None instead of an all-ones mask. The "
                        "batch has no padding either way, but some implementations "
                        "only take their fused-attention path without a mask.")
    p.set_defaults(attention_mask=True)
    p.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "default", "sdpa", "flash_attention_2",
                            "flash_attention_3", "flex_attention", "eager"],
                   help="attention backend requested from the Hugging Face baselines "
                        "(NexteraBERT picks its own kernels). 'auto' (the default) "
                        "gives every baseline FlashAttention-2 when the flash-attn "
                        "package is installed and the model supports it, and its "
                        "default backend otherwise -- per model, so BERT / ELECTRA / "
                        "ModernBERT / LFM can be on FlashAttention while DeBERTa "
                        "stays on its own attention. 'default' is whatever "
                        "transformers chooses, which since v5 is sdpa for every model "
                        "-- ModernBERT included; it no longer auto-selects "
                        "FlashAttention. Under sdpa a padding / sliding-window mask "
                        "is a materialised L x L tensor, which forces the "
                        "memory-efficient kernel, makes memory grow with the length "
                        "and makes local-attention layers pay the full L^2. "
                        "A model on FlashAttention is not handed the all-ones mask "
                        "(the batch has no padding, and the mask would only add a "
                        "data-dependent graph break). A model that cannot do the "
                        "requested backend falls back to its default, and the JSON "
                        "records what each model actually ran on.")
    p.add_argument("--extend_positions", default="copy",
                   choices=["copy", "interpolate", "none"],
                   help="how a model with a LEARNED position table (BERT, ELECTRA, "
                        "RoBERTa) reaches lengths past it: tile the table (copy, the "
                        "default), stretch it, or skip those lengths (none). The "
                        "compute is that of a natively longer table; points are "
                        "marked [e].")
    p.add_argument("--oom_margin", type=float, default=0.9,
                   help="CUDA only. Skip a length (and every longer one for that "
                        "model) without attempting it when the peak memory "
                        "extrapolated from the measured lengths exceeds this fraction "
                        "of what the device can give this process. 0 disables the "
                        "prediction; a real out-of-memory error is still caught.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--linear_y", dest="log_y", action="store_false",
                   help="linear time axis (default: log, since latency spans orders of "
                        "magnitude across the sweep)")
    p.set_defaults(log_y=True)
    p.add_argument("--output", default="benchmark.png",
                   help="latency figure; <output>_throughput.png is written beside it")
    p.add_argument("--json", dest="json_output", default=None,
                   help="where to write every run, the summaries and the settings. "
                        "Default: --output with a .json suffix. Pass 'none' to skip.")
    p.add_argument("--refresh", nargs="*", default=None, metavar="MODEL",
                   help="re-measure instead of reusing what --json recorded: bare "
                        "--refresh for everything, or '--refresh deberta lfm' for "
                        "just those models (same aliases / ids as --models)")
    args = p.parse_args()
    if args.batch_tokens < 1 or args.max_batch_size < 1 or \
            (args.batch_size is not None and args.batch_size < 1):
        p.error("--batch_tokens, --max_batch_size and --batch_size must be positive")
    if args.runs < 1 or args.rounds < 1:
        p.error("--runs and --rounds must be at least 1")
    if args.rounds == 1 and args.runs < 2:
        p.error("a single round needs --runs >= 2 to estimate a confidence interval")
    if not 0.0 < args.confidence < 1.0:
        p.error("--confidence must be strictly between 0 and 1")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    presets, models = {}, []
    for raw in args.models:
        name, preset = resolve_speed_model(raw, args.optibert)
        if name not in models:      # two aliases can resolve to the same id
            models.append(name)
            presets[name] = preset

    if args.json_output is None:
        args.json_output = os.path.splitext(args.output)[0] + ".json"
    use_json = args.json_output.lower() != "none"

    device = resolve_device()
    if device != "cuda":
        args.dtype = "fp32"     # no autocast off CUDA; record what actually ran
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=dtype)
                    if device == "cuda" and dtype != torch.float32
                    else contextlib.nullcontext())
    print(f"[speed] device={device} ({device_name(device)})  models={models}  "
          f"{batch_note(args)}  rounds={args.rounds} x runs={args.runs}  "
          f"warmup={args.warmup}  "
          f"compile={args.compile}  "
          f"sync_each_run={args.sync_each_run}")

    refresh_all = args.refresh is not None and not args.refresh
    cached, cached_meta = ({}, {}) if (refresh_all or not use_json) else \
        load_cached_results(args.json_output, args, device)
    for raw in args.refresh or []:
        name = resolve_speed_model(raw, args.optibert)[0]
        if cached.pop(name, None) is not None:
            print(f"[speed] --refresh: discarding the recorded sets of "
                  f"{display_name(name)}")
        cached_meta.pop(name, None)

    results = {name: new_result(args, cached.get(name), cached_meta.get(name))
               for name in models}
    for name, res in results.items():
        for pt in res["points"].values():
            del pt["sets"][args.rounds:]      # a record from a longer run: keep ours
    for round_idx in range(args.rounds):
        print(f"\n########## round {round_idx + 1} / {args.rounds} ##########")
        for name in models:
            print(f"\n=== {display_name(name)}  [{name}]  "
                  f"(round {round_idx + 1}/{args.rounds}) ===")
            benchmark_model(name, presets[name], args, device, autocast_ctx,
                            results[name], round_idx)
            # Rewritten after every model so an interrupted sweep resumes from here.
            if use_json:
                write_json(args.json_output, args, results, device, cached_meta)

    print(f"\n[speed] latency summary, ms per SEQUENCE ({batch_note(args)}): mean +/- "
          f"{args.confidence * 100:g}% CI half-width over the set means of "
          f"{args.rounds} rounds x {args.runs} runs "
          f"((k) = fewer sets than rounds, [e] = extended position table)")
    width = 28
    print("seq_len".ljust(8) + "batch".ljust(7) + "".join(display_name(n)[:width - 2].ljust(width)
                                        for n in results))
    for seq_len in args.seq_lengths:
        B = batch_size_for(args, seq_len)
        row = str(seq_len).ljust(8) + str(B).ljust(7)
        for res in results.values():
            pt = res["points"][seq_len]
            sm = summarize_point(pt, args.confidence)
            if math.isfinite(sm["mean"]):
                cell = (f"{sm['mean'] / B * 1e3:.3f} +/- "
                        f"{(sm['ci_high'] - sm['mean']) / B * 1e3:.3f}"
                        f"{'' if sm['n_sets'] == args.rounds else ' (%d)' % sm['n_sets']}"
                        f"{' [e]' if pt['extended'] else ''}")
            else:
                cell = "-"
            row += cell.ljust(width)
        print(row)

    plot(results, args)


if __name__ == "__main__":
    main()
