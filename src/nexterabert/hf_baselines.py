"""Hugging Face baseline encoders (ModernBERT, NeoBERT, BERT, ...) on this repo's
embedding protocols.

Published MTEB numbers for every encoder in NexteraBERT's class are measured
after a contrastive stage (OptiBERT App. D.2: attentive pooling + supervised
SimCSE on NLI; see ``nexterabert.simcse``). Scoring NexteraBERT after that stage
next to ModernBERT's or NeoBERT's *paper* numbers still compares two pipelines,
because each paper ran its own -- ModernBERT's and NeoBERT's are heavier. The
only clean comparison is to push the baselines through the identical stage and
the identical scorer, which is what this module makes possible:

    python scripts/finetune_contrastive.py --hf_model answerdotai/ModernBERT-base \
        --output_dir checkpoints/baselines/modernbert-base/simcse
    python scripts/evaluate_mteb.py --model checkpoints/baselines/modernbert-base/simcse

Everything model-specific lives here:

* **Loading.** ``AutoModel`` with ``trust_remote_code`` (NeoBERT ships its own
  code). NeoBERT's remote module imports ``xformers`` unconditionally, which has
  no CPU/Windows wheels, so an eager ``SwiGLU`` stand-in with the same parameter
  packing is registered when it is missing. transformers >= 5 builds models on
  the meta device, which leaves NeoBERT's computed, non-persistent RoPE table
  (``freqs_cis``) zero-filled -- positions silently vanish -- so it is recomputed
  after loading, sized to the longest sequence the caller will feed.
  LFM2.5-Encoder's checkpoint is saved under the MaskedLM class's ``lfm2.``
  prefix, which its bare ``AutoModel`` class does not share: ``AutoModel``
  returns a RANDOM encoder and only logs a table. The loader detects an
  uninitialised backbone and takes it from ``AutoModelForMaskedLM`` instead.
* **GLUE.** :class:`HFForSequenceClassification` puts NexteraBERT's pooler +
  classifier (which is also ModernBERT's head) over any backbone, with the
  ``(loss, logits)`` contract ``scripts/evaluate_glue.py`` trains.
* **Forward.** NeoBERT expands the ``(B, T)`` key-padding mask to ``(B, H, T,
  T)`` with ``repeat`` -- 96 GiB of int64 for 16 x 8192 tokens. Every NeoBERT opened
  here (and, through :func:`repair_wrapped_neobert`, inside sentence-transformers /
  PyLate) is given a broadcastable ``(B, 1, 1, T)`` bool mask instead, with
  bit-identical outputs; see :func:`broadcast_neobert_mask`.
* **Pooling.** :class:`HFForSentenceEmbedding` puts the same
  :class:`~nexterabert.simcse.AttentivePooling` head (or the same masked mean)
  over ``last_hidden_state`` that the NexteraBERT variants use.
* **Optimiser.** :func:`hf_llrd` maps a baseline's parameter names onto the
  depth ladder ``evaluate_glue.build_optimizer`` expects, so layerwise LR decay
  behaves identically across architectures.
* **Saving.** A directory written by :func:`save_hf_contrastive_model` is an
  ordinary ``save_pretrained`` model directory plus the repo's ``pooling_head.pt``
  and ``contrastive_run.json`` markers; :func:`is_hf_model_dir` tells the eval
  scripts to open it through this module instead of the NexteraBERT loader.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simcse import (
    CONTRASTIVE_RUN_FILE,
    POOLING_HEAD_FILE,
    AttentivePooling,
    contrastive_run,
)

#: Value of ``contrastive_run.json["backbone"]`` for a directory saved here.
HF_BACKBONE_KIND = "hf"
NEXTERA_MODEL_TYPE = "nexterabert"

#: The baselines the MTEB / GLUE comparisons are written for.
DEFAULT_BASELINES = ("answerdotai/ModernBERT-base", "chandar-lab/NeoBERT",
                     "LiquidAI/LFM2.5-Encoder-230M")
#: Pooler-head weights of a GLUE-tuned baseline (the MNLI -> RTE/MRPC/STS-B
#: transfer carries them, as a NexteraBERT export carries its ``pooler``).
GLUE_POOLER_FILE = "glue_pooler.pt"


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def is_neobert(obj) -> bool:
    """True for a NeoBERT name, config or model."""
    if isinstance(obj, str):
        return "neobert" in obj.lower()
    config = getattr(obj, "config", obj)
    model_type = str(getattr(config, "model_type", "") or "")
    return "neobert" in model_type.lower() or "neobert" in type(obj).__name__.lower()


def hf_model_type(path) -> str | None:
    """``model_type`` recorded in ``<path>/config.json`` (None if not a model dir)."""
    cfg = Path(path) / "config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return None


def is_hf_model_dir(path) -> bool:
    """True when ``path`` is a local model directory that is NOT a NexteraBERT
    export -- i.e. one this module (``AutoModel``) has to open.

    A Hub id is not a directory and returns False; the callers treat those
    through ``--hf_model`` explicitly.
    """
    if path is None:
        return False
    model_type = hf_model_type(path)
    return model_type is not None and model_type != NEXTERA_MODEL_TYPE


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def ensure_xformers_swiglu() -> bool:
    """Make ``from xformers.ops import SwiGLU`` succeed without xformers.

    NeoBERT's remote code does that import unconditionally, but xformers ships
    no CPU/Windows wheels. When it is missing, register a minimal eager
    equivalent with xformers' parameter names and packing (``w12`` fused gate +
    up projection, ``w3`` down projection, silu on the first chunk) so the
    checkpoint loads unchanged. Returns True when the stand-in was installed.
    """
    try:
        import xformers.ops  # noqa: F401
        return False
    except ImportError:
        pass
    import types

    class SwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features, out_features=None, bias=True,
                     **kwargs):
            super().__init__()
            out_features = out_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    ops = types.ModuleType("xformers.ops")
    ops.SwiGLU = SwiGLU
    pkg = types.ModuleType("xformers")
    pkg.ops = ops
    sys.modules["xformers"] = pkg
    sys.modules["xformers.ops"] = ops
    print("[hf] xformers is not installed -- registered an eager SwiGLU stand-in "
          "so NeoBERT's remote code can load.")
    return True


def _recompute_neobert_rope(model, config, device=None) -> None:
    """Rebuild NeoBERT's ``freqs_cis`` buffer with the remote module's own function.

    transformers >= 5 materialises models via the meta device, so a computed
    non-persistent buffer comes out ZERO-FILLED instead of running the __init__
    computation. Always recompute: it is cheap and the check for an all-zero
    table would be the only thing standing between a silent no-position model
    and a published number.
    """
    remote_mod = sys.modules[type(model).__module__]
    dim_head = getattr(config, "dim_head", None) or (
        config.hidden_size // config.num_attention_heads)
    table = remote_mod.precompute_freqs_cis(int(dim_head), int(config.max_length))
    model.freqs_cis = table.to(device or model.freqs_cis.device)


def non_finite_parameters(module, limit: int = 5) -> list:
    """Names of parameters / buffers holding a NaN or inf (at most ``limit``).

    A run that diverges does not stop: the trainer keeps stepping, every weight turns
    NaN, the checkpoint is written as usual, and the retrieval scorer turns NaN
    similarities into a quiet ``nDCG@10 = 0.0`` -- a number that then sits in a results
    table looking like a (terrible) measurement. Trainers call this before saving and
    scorers right after loading, so a dead checkpoint is an error, not a score.
    """
    bad = []
    tensors = list(module.named_parameters()) + list(module.named_buffers())
    for name, tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all()):
                bad.append(name)
                if len(bad) >= limit:
                    break
    return bad


def broadcast_neobert_mask(model) -> bool:
    """Stop NeoBERT from materialising its padding mask as ``(B, H, L, L)``.

    Its forward turns the ``(B, L)`` mask into ``(B, H, L, L)`` with ``repeat`` -- a
    real copy, in the mask's own dtype. sentence-transformers hands it int64, so a
    batch of 16 x 8192 tokens is 16 * 12 * 8192^2 * 8 B = 96 GiB of mask before a
    single layer has run, and SDPA then converts it to a float bias of the same shape
    on top (the "tried to allocate 48 GiB" that kills an MLDR / CoIR run on a 140 GiB
    card). Nothing in the maths needs it: a key-padding mask is the same for every
    head and every query, which is exactly what a broadcastable ``(B, 1, 1, L)`` mask
    says, and both SDPA and the eager path broadcast it.

    The remote forward is left alone. It is called with ``attention_mask=None`` (so it
    repeats nothing) and a pre-hook on every encoder block puts the broadcastable mask
    back into the block's ``attention_mask`` argument. Outputs are unchanged. With no
    mask at all the released code crashes on ``None.bool()``; a ``(1, 1, 1, 1)``
    all-True mask is supplied instead. Packed (``cu_seqlens``) calls pass through
    untouched. Returns True when the model was patched by this call.
    """
    if getattr(model, "_nextera_broadcast_mask", False):
        return False
    blocks = getattr(model, "transformer_encoder", None)
    if blocks is None:
        return False
    state = {"mask": None}
    original_forward = model.forward

    def forward(*args, **kwargs):
        state["mask"] = None
        mask = kwargs.get("attention_mask")
        if kwargs.get("cu_seqlens") is None and len(args) < 4 and \
                (mask is None or mask.dim() == 2):
            if mask is not None:
                state["mask"] = mask[:, None, None, :].bool()
            else:
                ref = kwargs.get("input_ids") if kwargs.get("input_ids") is not None \
                    else (args[0] if args else kwargs.get("inputs_embeds"))
                state["mask"] = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=ref.device)
            kwargs["attention_mask"] = None
        out = original_forward(*args, **kwargs)
        state["mask"] = None
        return out

    def put_mask_back(_block, block_args):
        # blocks are called as layer(x, attention_mask, freqs_cis, ...)
        if state["mask"] is None or len(block_args) < 2:
            return None
        return (block_args[0], state["mask"], *block_args[2:])

    for block in blocks:
        block.register_forward_pre_hook(put_mask_back)
    model.forward = forward
    model._nextera_broadcast_mask = True
    return True


def repair_wrapped_neobert(wrapper, max_len: int | None = None) -> int:
    """Repair every NeoBERT living inside ``wrapper`` (a SentenceTransformer, a PyLate
    ColBERT, any ``nn.Module``). Returns how many were repaired.

    sentence-transformers and PyLate open the encoder with their own plain
    ``AutoModel.from_pretrained``, so none of :func:`load_hf_encoder`'s repairs reach
    it, and NeoBERT needs both of them:

      * its RoPE table ``freqs_cis`` comes out **all zeros** (transformers >= 5, see
        :func:`_recompute_neobert_rope`). Rotating by a zero table zeroes every query
        and key, so all attention scores are equal: the model runs with *uniform
        attention* -- no error, just a baseline that trains and scores far below what
        the checkpoint can do;
      * the table is ``config.max_length`` (4096) rows long and the forward slices
        ``freqs_cis[:T]``, so a longer input (``--max_len 8192`` for MLDR / CoIR)
        dies in ``reshape_for_broadcast`` with a bare ``AssertionError``.

    ``max_len`` is the longest sequence the caller will feed; the table is rebuilt to
    cover it. Call this right after constructing the wrapper and again whenever its
    max sequence length is raised. Safe to call on a wrapper with no NeoBERT in it.
    """
    repaired = 0
    for module in wrapper.modules():
        if not (hasattr(module, "freqs_cis") and is_neobert(module)):
            continue
        config = module.config
        if max_len is not None and int(max_len) > int(getattr(config, "max_length", 0) or 0):
            config.max_length = int(max_len)
        _recompute_neobert_rope(module, config)
        broadcast_neobert_mask(module)
        repaired += 1
    if repaired:
        print(f"[hf] NeoBERT inside {type(wrapper).__name__}: RoPE table recomputed "
              f"for {int(config.max_length)} positions (a plain AutoModel load leaves "
              f"it zero-filled, i.e. uniform attention, and sized to 4096); padding "
              f"mask kept broadcastable instead of repeated to (B, H, L, L).",
              file=sys.stderr)
    return repaired


def load_hf_encoder(model_name, max_len: int | None = None, torch_dtype=None,
                    attn_implementation: str | None = None):
    """``AutoModel.from_pretrained`` with the baseline-specific repairs applied.

    ``max_len`` is the longest sequence the caller will feed; NeoBERT bakes its
    RoPE table to ``config.max_length`` at construction, so the table is sized
    to cover it (the trained weights are unaffected).

    ``attn_implementation`` is handed to ``from_pretrained`` when given (``sdpa`` /
    ``flash_attention_2`` / ``flex_attention`` / ``eager``); ``None`` keeps
    transformers' own choice, which since v5 is ``sdpa`` -- it no longer switches to
    FlashAttention on its own, not even for ModernBERT.

    Returns ``(model, config)``.
    """
    from transformers import AutoConfig, AutoModel

    neobert = is_neobert(str(model_name))
    if neobert:
        ensure_xformers_swiglu()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not neobert:
        neobert = is_neobert(config)
        if neobert:
            ensure_xformers_swiglu()
    if neobert and max_len is not None:
        current = int(getattr(config, "max_length", 0) or 0)
        if max_len > current:
            config.max_length = int(max_len)
    kwargs = {"trust_remote_code": True, "config": config}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model, loading = AutoModel.from_pretrained(model_name, output_loading_info=True,
                                               **kwargs)
    uninitialised = _uninitialised_fraction(model, loading)
    if uninitialised > 0.5:
        # The checkpoint was saved from a task class whose base-model prefix the
        # bare AutoModel class does not share (LFM2.5-Encoder: keys are
        # ``lfm2.*`` while Lfm2BidirectionalModel inherits prefix ``model``), so
        # AutoModel "loads" a RANDOM encoder and only logs a table about it.
        # Open the task class that owns the checkpoint and take its backbone.
        print(f"[hf] {model_name}: AutoModel left {uninitialised:.0%} of the encoder "
              f"uninitialised (base-model prefix mismatch) -- loading it through "
              f"AutoModelForMaskedLM and taking the backbone instead.")
        model = _backbone_via_masked_lm(model_name, kwargs)
        model._nextera_repaired_backbone = True
    if neobert:
        _recompute_neobert_rope(model, config)
        broadcast_neobert_mask(model)
    _lfm_mask_for_flash_kernels(model)
    return model, config


def _lfm_mask_for_flash_kernels(model):
    """LFM2.5-Encoder's remote code picks its FlashAttention mask path with
    ``_attn_implementation == "flash_attention_2"``. Without the flash-attn package
    transformers substitutes ``kernels-community/flash-attn2``, which fails that test
    and would hand the flash kernel a 4D additive mask. Route every flash backend to
    the 2D-padding-mask-or-None path."""
    import sys

    backend = getattr(model.config, "_attn_implementation", None)
    if not isinstance(backend, str) or "flash" not in backend \
            or backend == "flash_attention_2":
        return
    remote = sys.modules.get(type(model).__module__)
    lfm2_mod = getattr(remote, "_lfm2_mod", None)
    original = getattr(remote, "_bidirectional_mask", None)
    if lfm2_mod is None or original is None \
            or getattr(lfm2_mod.create_causal_mask, "_nextera_flash_kernels", False):
        return

    def bidirectional_mask(config, *args, attention_mask=None, **kwargs):
        if "flash" in str(config._attn_implementation):
            if attention_mask is not None and not attention_mask.all():
                return attention_mask
            return None
        return original(config, *args, attention_mask=attention_mask, **kwargs)

    bidirectional_mask._nextera_flash_kernels = True
    lfm2_mod.create_causal_mask = bidirectional_mask


def _uninitialised_fraction(model, loading_info) -> float:
    """Share of the model's tensors that ``from_pretrained`` did not find."""
    missing = set((loading_info or {}).get("missing_keys", []) or [])
    total = len(model.state_dict())
    return (len(missing) / total) if total else 0.0


def _backbone_via_masked_lm(model_name, kwargs):
    from transformers import AutoModelForMaskedLM

    mlm, loading = AutoModelForMaskedLM.from_pretrained(
        model_name, output_loading_info=True, **kwargs)
    prefix = getattr(mlm, "base_model_prefix", "") or ""
    backbone = getattr(mlm, prefix, None) if prefix else None
    if backbone is None or backbone is mlm:
        backbone = getattr(mlm, "base_model", None)
    if backbone is None or backbone is mlm:
        raise RuntimeError(f"{model_name}: cannot locate the encoder inside "
                           f"{type(mlm).__name__} (base_model_prefix={prefix!r})")
    backbone_keys = {f"{prefix}.{k}" for k in backbone.state_dict()}
    still_missing = [k for k in loading.get("missing_keys", []) if k in backbone_keys]
    if len(still_missing) > 0.5 * max(len(backbone_keys), 1):
        raise RuntimeError(
            f"{model_name}: the encoder is still uninitialised when loaded through "
            f"{type(mlm).__name__} ({len(still_missing)}/{len(backbone_keys)} tensors "
            f"missing, e.g. {still_missing[:3]}); refusing to score a random model.")
    return backbone


#: Marker written into a directory produced by :func:`ensure_automodel_loadable`.
BACKBONE_REPAIR_FILE = "backbone_repair.json"


def ensure_automodel_loadable(name_or_dir, cache_dir="checkpoints/hub") -> str:
    """A path that a PLAIN ``AutoModel.from_pretrained`` opens correctly.

    The loaders in this module repair a broken checkpoint layout in memory, but
    third-party trainers build their own model: sentence-transformers
    (``scripts/train_st_dpr.py``, the DPR stage behind BEIR / MLDR / CoIR) and
    PyLate both call ``AutoModel.from_pretrained`` themselves. For
    LFM2.5-Encoder that call returns a **randomly initialised** encoder (see
    ``load_hf_encoder``) -- the run would train and score noise without a single
    error. This materialises the repaired backbone once, as an ordinary model
    directory (weights under the keys the bare class expects, remote code and
    tokenizer beside them), and returns it; a model that loads correctly, and a
    NexteraBERT export, are returned untouched.

    The tokenizer is saved WITHOUT its chat template. LFM2.5-Encoder inherits one
    from the LFM2.5 chat models, and sentence-transformers >= 5 routes every text
    through ``apply_chat_template`` when it finds one: the encoder would be fed
    ``<|startoftext|><|im_start|>user
...<|im_end|>
`` instead of the
    ``<|startoftext|>`` + text it was pretrained on (and that the MTEB / GLUE
    paths here use). A chat template alone is therefore also a reason to
    materialise the directory.
    """
    import os
    import shutil
    import tempfile

    path = str(name_or_dir)
    if Path(path).is_dir() and not is_hf_model_dir(path):
        return path                      # a NexteraBERT export (or an ST directory)
    if Path(path).is_dir() and (Path(path) / BACKBONE_REPAIR_FILE).is_file():
        return path
    stem = Path(path).name if Path(path).is_dir() else path.replace("/", "__")
    dest = Path(cache_dir) / f"{stem}__backbone"
    if (dest / BACKBONE_REPAIR_FILE).is_file():
        return str(dest)

    tokenizer = load_hf_tokenizer(path)
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    model, _ = load_hf_encoder(path)
    repaired = getattr(model, "_nextera_repaired_backbone", False)
    if not repaired and not has_chat_template:
        return path
    reasons = []
    if repaired:
        reasons.append("AutoModel.from_pretrained left the encoder uninitialised "
                       "(base-model prefix mismatch); backbone re-saved from "
                       "AutoModelForMaskedLM")
    if has_chat_template:
        reasons.append("tokenizer chat template removed (sentence-transformers would "
                       "wrap every text in it)")
        tokenizer.chat_template = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{stem}__backbone.", dir=str(dest.parent)))
    try:
        model.save_pretrained(tmp, safe_serialization=True)
        _copy_remote_code(model, tmp)
        tokenizer.save_pretrained(tmp)
        for stale in ("chat_template.jinja", "chat_template.json"):
            if has_chat_template and (tmp / stale).exists():
                (tmp / stale).unlink()
        with open(tmp / BACKBONE_REPAIR_FILE, "w", encoding="utf-8") as f:
            json.dump({"source": path, "reasons": reasons}, f, indent=2)
        try:
            os.replace(tmp, dest)
        except OSError:                  # a parallel sweep job got there first
            if not (dest / BACKBONE_REPAIR_FILE).is_file():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[hf] {path}: plain-AutoModel backbone materialised -> {dest} "
          f"({'; '.join(reasons)})", file=sys.stderr)
    return str(dest)


def hf_last_hidden(model, input_ids, attention_mask=None) -> torch.Tensor:
    """``(B, T) -> (B, T, C)`` token states from any ``AutoModel``."""
    if attention_mask is not None and is_neobert(model):
        attention_mask = attention_mask.bool()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0]
    return hidden


def load_hf_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except (ValueError, OSError):
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HFForSentenceEmbedding(nn.Module):
    """``AutoModel`` encoder + the repo's pooling: text -> one vector.

    ``pooling="attentive"`` mirrors ``NexteraBERTForSentenceEmbedding`` (the
    OptiBERT MTEB protocol); ``"mean"`` mirrors ``NexteraBERTForEmbedding`` (the
    ModernBERT retrieval protocol, no new parameters). The encoder is exposed as
    ``.encoder`` so ``build_optimizer``'s decay rules and ``save`` see the same
    layout as the NexteraBERT variants.
    """

    def __init__(self, encoder: nn.Module, pooling: str = "attentive"):
        super().__init__()
        if pooling not in ("attentive", "mean"):
            raise ValueError(f"pooling must be 'attentive' or 'mean' (got {pooling!r})")
        self.encoder = encoder
        self.config = encoder.config
        self.pooling_kind = pooling
        hidden = int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))
        self.pooling = AttentivePooling(hidden) if pooling == "attentive" else None

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        from .modeling_nexterabert import masked_mean_pool

        hidden = hf_last_hidden(self.encoder, input_ids, attention_mask)
        if self.pooling is None:
            return masked_mean_pool(hidden, attention_mask)
        return self.pooling(hidden, attention_mask)

    def freeze_unused(self) -> list:
        """Freeze what this objective can never reach.

        BERT-style backbones carry a ``pooler`` (dense + tanh over [CLS]) that no
        sentence-embedding path calls; ModernBERT / NeoBERT have none.
        """
        frozen = []
        for name, p in self.encoder.named_parameters():
            if name.startswith("pooler.") or ".pooler." in name:
                p.requires_grad_(False)
                frozen.append(name)
        return frozen


class HFPooler(nn.Module):
    """``NexteraBERTPooler`` over a foreign backbone: masked mean -> dense (no
    bias) -> GELU -> LayerNorm (gain only). It is also ModernBERT's own
    classification head (``classifier_pooling="mean"``), so every GLUE row is
    read through one head regardless of what the baseline ships -- LFM2.5-Encoder
    ships no sequence-classification class at all."""

    def __init__(self, hidden: int):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden, bias=False)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states, attention_mask=None):
        from .modeling_nexterabert import masked_mean_pool

        pooled = masked_mean_pool(hidden_states, attention_mask)
        return self.norm(self.activation(self.dense(pooled)))


class HFForSequenceClassification(nn.Module):
    """``AutoModel`` encoder + the repo's GLUE head, with the call contract of
    ``NexteraBERTForSequenceClassification``: ``forward(...) -> (loss, logits)``.

    ``scripts/evaluate_glue.py`` drives it unchanged -- same per-task recipe,
    optimiser, early stopping and MNLI transfer -- so a baseline's GLUE row is
    measured the way NexteraBERT's is.
    """

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.0):
        super().__init__()
        import inspect

        self.encoder = encoder
        self.config = encoder.config
        hidden = int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))
        self.pooler = HFPooler(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        # segment ids only where the backbone has segment embeddings (BERT);
        # ModernBERT / NeoBERT / LFM2 have none and would reject the argument
        accepts = "token_type_ids" in inspect.signature(encoder.forward).parameters
        self._use_token_types = (
            accepts and int(getattr(self.config, "type_vocab_size", 0) or 0) > 1)

    def forward(self, input_ids, cross_ids=None, attention_mask=None,
                token_type_ids=None, labels=None):
        if self._use_token_types and token_type_ids is not None:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out[0]
        else:
            hidden = hf_last_hidden(self.encoder, input_ids, attention_mask)
        logits = self.classifier(self.dropout(self.pooler(hidden, attention_mask)))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.squeeze(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def freeze_unused(self) -> list:
        """BERT's own [CLS] pooler is never reached by this head."""
        frozen = []
        for name, p in self.encoder.named_parameters():
            if name.startswith("pooler.") or ".pooler." in name:
                p.requires_grad_(False)
                frozen.append(name)
        return frozen


def load_hf_for_sequence_classification(name_or_dir, num_labels: int,
                                        max_len: int | None = None):
    """``(model, config, info)`` like ``loading.load_for_task``.

    A directory written by :func:`save_hf_glue_backbone` also restores the tuned
    pooler head; the classifier is always fresh (its width is per task).
    """
    encoder, config = load_hf_encoder(name_or_dir, max_len=max_len)
    model = HFForSequenceClassification(encoder, num_labels)
    pooler_path = Path(str(name_or_dir)) / GLUE_POOLER_FILE
    restored = pooler_path.is_file()
    if restored:
        model.pooler.load_state_dict(torch.load(pooler_path, map_location="cpu"))
    model.freeze_unused()
    return model, config, {"missing": [], "unexpected": [], "glue_pooler": restored}


def save_hf_glue_backbone(model: HFForSequenceClassification, output_dir, *,
                          tokenizer=None) -> str:
    """Export a GLUE-tuned baseline encoder (+ pooler head) for task transfer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out, safe_serialization=True)
    _copy_remote_code(model.encoder, out)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    torch.save(model.pooler.state_dict(), out / GLUE_POOLER_FILE)
    return str(out)


def resolve_hub_model(name, cache_dir="checkpoints/hub") -> str:
    """A local directory for ``name``: itself when it exists, else a one-time
    ``snapshot_download`` (the same layout ``scripts/eval_common.sh`` uses), so
    the scripts can dispatch on ``config.json``'s ``model_type``."""
    text = str(name)
    if Path(text).exists():
        return text
    looks_like_hub_id = (text.count("/") == 1 and not text.startswith((".", "/", "~"))
                         and ":" not in text and chr(92) not in text)
    if not looks_like_hub_id:
        return text                                   # let the caller report it
    from huggingface_hub import snapshot_download

    dest = Path(cache_dir) / text.replace("/", "__")
    if not (dest / "config.json").is_file():
        print(f"[hf] downloading {text} -> {dest}", file=sys.stderr)
        snapshot_download(repo_id=text, local_dir=str(dest), repo_type="model")
    return str(dest)


# ---------------------------------------------------------------------------
# Layerwise LR decay over a foreign parameter namespace
# ---------------------------------------------------------------------------

_LAYER_INDEX = re.compile(r"\.(?:layers|layer|transformer_encoder|blocks|h)\.(\d+)\.")


def hf_llrd(model: nn.Module):
    """``(n_layer, depth_fn)`` for ``evaluate_glue.build_optimizer``.

    Depth 0 is every ``nn.Embedding`` (plus anything under an ``embedding(s)``
    module: ModernBERT's embedding norm rides with its table, as NexteraBERT's
    does), 1..n_layer the transformer blocks bottom-to-top (``layers.N`` for
    ModernBERT, ``encoder.layer.N`` for BERT, ``transformer_encoder.N`` for
    NeoBERT), and n_layer+1 everything above them -- final norm and the freshly
    initialised pooling head, which trains at the full LR.
    """
    config = model.config
    n_layer = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    names = [name for name, _ in model.named_parameters()]
    if not n_layer:
        indices = [int(m.group(1)) for m in map(_LAYER_INDEX.search, names) if m]
        n_layer = (max(indices) + 1) if indices else 0
    n_layer = int(n_layer)

    embedding_params = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            for p_name, _ in mod.named_parameters(recurse=False):
                embedding_params.add(f"{mod_name}.{p_name}" if mod_name else p_name)

    def depth_fn(name: str) -> int:
        m = _LAYER_INDEX.search(name)
        if m:
            return min(int(m.group(1)), n_layer - 1) + 1
        # "embed" also catches a free-standing embedding norm (LFM2's
        # ``embedding_norm``), which belongs with the table it normalises
        if name in embedding_params or "embed" in name.lower():
            return 0
        return n_layer + 1

    return n_layer, depth_fn


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _copy_remote_code(encoder: nn.Module, out: Path) -> list:
    """Make a saved remote-code model reloadable.

    ``save_pretrained`` copies a ``trust_remote_code`` model's source only for a
    class that was opened through its own auto class. A backbone lifted out of a
    task model (LFM2.5-Encoder, see ``_backbone_via_masked_lm``) is not, yet its
    config still carries ``auto_map`` -- the directory would then name a module
    file it does not contain. Copy whatever ``auto_map`` references and is missing.
    """
    import shutil

    auto_map = getattr(encoder.config, "auto_map", None) or {}
    module = sys.modules.get(type(encoder).__module__)
    src_dir = Path(module.__file__).parent if getattr(module, "__file__", None) else None
    if not auto_map or src_dir is None:
        return []
    copied = []
    for py in src_dir.glob("*.py"):
        if py.name != "__init__.py" and not (out / py.name).exists():
            shutil.copy(py, out / py.name)
            copied.append(py.name)
    return copied


def save_hf_contrastive_model(model: HFForSentenceEmbedding, output_dir, *,
                              tokenizer=None, run_summary: dict | None = None) -> str:
    """Write an ``AutoModel`` directory + the repo's contrastive markers.

    ``save_pretrained`` also copies a remote-code model's source files (NeoBERT)
    so the directory reloads with ``trust_remote_code``. The pooling head goes
    to ``pooling_head.pt`` exactly as for a NexteraBERT run, and
    ``contrastive_run.json`` gains ``"backbone": "hf"`` so the loaders know which
    module owns the directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out, safe_serialization=True)
    _copy_remote_code(model.encoder, out)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    if isinstance(model.pooling, nn.Module):
        torch.save(model.pooling.state_dict(), out / POOLING_HEAD_FILE)
    if run_summary is not None:
        summary = {**run_summary, "backbone": HF_BACKBONE_KIND}
        with open(out / CONTRASTIVE_RUN_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return str(out)


def load_hf_contrastive_model(name_or_dir, pooling: str = "auto",
                              max_len: int | None = None):
    """Rebuild :class:`HFForSentenceEmbedding` from a Hub id or a saved directory.

    ``pooling="auto"`` follows the directory: a trained ``pooling_head.pt`` makes
    it attentive, otherwise masked mean -- a raw Hub id therefore scores exactly
    as the old mean-pool baseline path did. Returns ``(model, config, info)``
    with ``info["pooling_head"]`` saying whether a trained head was found.
    """
    encoder, config = load_hf_encoder(name_or_dir, max_len=max_len)
    head_path = Path(str(name_or_dir)) / POOLING_HEAD_FILE
    trained_head = head_path.is_file()
    if pooling == "auto":
        pooling = "attentive" if trained_head else "mean"
    model = HFForSentenceEmbedding(encoder, pooling)
    loaded = False
    if pooling == "attentive" and trained_head:
        model.pooling.load_state_dict(torch.load(head_path, map_location="cpu"))
        loaded = True
    info = {"missing": [], "unexpected": [], "pooling_head": loaded,
            "contrastive_run": contrastive_run(name_or_dir) if Path(str(name_or_dir)).is_dir() else None}
    return model, config, info


def hf_source_name(name_or_dir) -> str:
    """The Hub id a saved directory was fine-tuned from (or the name itself)."""
    run = contrastive_run(name_or_dir) if Path(str(name_or_dir)).is_dir() else None
    if run and run.get("hf_model"):
        return str(run["hf_model"])
    return str(name_or_dir)
