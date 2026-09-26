"""
coding = utf-8
Licensed under "MIT License"
Commercial use is of course permitted
SEA Model Op.0+: NexteraBERT
Bidirectional Encoder Representations from SEA
PyTorch implementation

NexteraBERT = SnowLily + NexteraSelfAttention + NexteraSlidingWindowAttention + NexteraHRA
              + squared-ReLU MLP (or routed RippleBloomUGM with mlp_class="ugm")
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import List, NamedTuple
try:
    from torchmetrics import Metric
except ImportError:  # pragma: no cover - inference-only install
    # This file is bundled into uploaded Hub repos and imported by
    # `trust_remote_code=True`, where only torch + transformers are guaranteed.
    # The two Metric subclasses below are training-eval bookkeeping and are never
    # touched by inference, so degrade to a stub rather than making torchmetrics a
    # hard requirement for merely loading the model.
    class Metric(torch.nn.Module):  # type: ignore[no-redef]
        def add_state(self, name, default, dist_reduce_fx=None):
            setattr(self, name, default)

        def update(self, *args, **kwargs):
            raise RuntimeError(
                "torchmetrics is not installed; NexteraBERT's training metrics are "
                "unavailable (inference is unaffected). pip install torchmetrics")

        compute = update

from transformers.integrations.hub_kernels import use_kernel_func_from_hub, use_kernelized_func
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from collections.abc import Callable

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True

class DiscriminatorAccuracy(Metric):
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, outputs, batch):
        logits = outputs["disc_logits"]
        
        # A token counts as replaced only if it actually differs from the original.
        if 'disc_input_ids' in outputs and 'original_token_ids' in outputs:
            labels = (outputs['disc_input_ids'] != outputs['original_token_ids']).long()
        else:
            # Fallback to replace_mask for backward compatibility
            labels = outputs["replace_mask"]
        
        input_ids = batch['input_ids']

        preds = (logits > 0).long()
        
        valid = (input_ids != self.pad_token_id)

        correct = ((preds == labels) & valid).sum()
        total = valid.sum()

        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct / (self.total + 1e-8)



class TokenAccuracy(Metric):
    """Token-level accuracy on the masked positions of the MLM task."""
    def __init__(self, pad_token_id, logits_key: str = "gen_logits"):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.logits_key = logits_key   # which logits tensor in `outputs` to score
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, outputs, batch):
        logits = outputs[self.logits_key]
        labels = outputs['original_token_ids']
        replace_mask = outputs['replace_mask'] # Shape: [B, L], boolean

        # When the vocabulary projection only runs on masked positions, logits are
        # compacted to (N, V), so the labels are gathered at the same positions. The
        # COCO-LM corrective-LM path returns full (B, L, V) logits (handled below).
        if logits.dim() == 2:
            preds = torch.argmax(logits, dim=-1)          # (N,)
            correct = (preds == labels[replace_mask])
            self.correct += correct.sum()
            self.total += torch.as_tensor(
                preds.numel(), device=preds.device, dtype=self.total.dtype)
            return

        preds = torch.argmax(logits, dim=-1)

        correct = (preds == labels) & replace_mask

        self.correct += correct.sum()
        self.total += replace_mask.sum()

    def compute(self):
        return self.correct / (self.total + 1e-8)


def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def norm(x, eps=1e-8):
    """
    Parameter-free RMSNorm (no affine, no partial)

    Args:
        x: input tensor (..., d)
        eps: numerical stability

    Returns:
        normalized tensor
    """
    # F.rms_norm reduces in the accumulate dtype (fp32 for bf16/fp16 inputs) inside a
    # single fused kernel, without materialising an fp32 copy of x.
    return F.rms_norm(x, x.shape[-1:], None, eps)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable per-channel gain (no bias, no mean subtraction).

    Used for every pre-norm inside the encoder blocks in place of LayerNorm:
    one reduction instead of two, and no centring — the residual stream keeps
    its mean, which the gated mixers read. The final ``out_norm`` stays a
    LayerNorm so the exported hidden states are still centred.
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-8
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # ``F.rms_norm`` is the fused kernel for exactly this: it accumulates the
        # mean-square in fp32 for bf16/fp16 inputs (same precision as the explicit
        # ``x.float()`` this replaces) but never materialises the fp32 copy of x —
        # which, at 2 norms x n_layer sites, was the largest single source of
        # activation traffic in the encoder.
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SeparableDyT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        alpha_init: float = 0.5,
        bias: bool = False
    ):
        super().__init__()
        self.alpha = nn.Parameter(alpha_init*torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.ones(hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.bias = None

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        pre = self.alpha * x
        if self.bias is not None:
            pre = pre + self.bias
        x = self.beta * F.tanh(pre)
        return x


class ScalableFactor(nn.Module):
    """Length-dependent query scaling for the attention softmax.

    Which of the three formulations is used is set by ``config.ssmax_mode`` (see
    :class:`~nexterabert.configuration_nexterabert.NexteraBERTConfig`) and is an
    ablation axis, not a runtime switch: the modes have different parameter sets,
    so a checkpoint must be scored with the mode it was trained under.

    ``"sssmax"``  Stable Scalable-Softmax — what NexteraBERT ships (default):
                  ``s·log(T) + base + relu(b)`` with ``T`` the number of attended
                  keys, ``s`` and ``b`` learnable (init ``ssmax_s`` = 0.43,
                  ``ssmax_b`` = 0.1, so ``relu(b)`` starts inside its linear range
                  and ``b`` receives a gradient from the first step) and ``base`` a
                  fixed constant (``ssmax_base``, 0.1). The offset is what keeps it
                  stable: ``s·log T`` alone is 0 at a single key and the published SSMax
                  collapses there, whereas here the scale is at least ``base``
                  however small ``T`` is, and ``relu`` keeps the learnable part
                  non-negative so no training trajectory can push the scale below
                  ``base`` or flip its sign. (``relu`` has zero gradient at ``b <= 0``:
                  a layer whose offset is driven to the floor stays there.)
    ``"ssmax"``   Scalable-Softmax exactly as published (Nakanishi 2025,
                  *Scalable-Softmax Is Superior for Attention*): ``s·log(n)``. No
                  offset, so the scale is already ~3x at n=1024 and falls to 0 at
                  n=1 (a single attended key, where the softmax is degenerate
                  anyway).
    ``"none"``    No length scaling: plain softmax with the usual ``1/sqrt(d)``.
                  ``forward`` returns ``None`` and **no parameter is registered** —
                  an always-unused parameter would abort DDP training under
                  ``find_unused_parameters=False`` (see ``freeze_pretraining_unused``).
    """

    MODES = ("sssmax", "ssmax", "none")

    def __init__(
        self,
        s: float = 0.43,
        b: float = 0.1,
        base: float = 0.1,
        mode: str = "sssmax",
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(
                f"Unknown ssmax mode '{mode}'. Choose one of {self.MODES}.")
        self.mode = mode
        self.base = float(base)
        # Only the modes that use a parameter register one (see "none" above), and
        # only "sssmax" has the offset b -- so an "ssmax" state_dict has exactly the
        # key it always had.
        if mode == "none":
            self.s = None            # plain attribute, NOT a Parameter
            self.b = None
        else:
            if s <= 0:
                raise ValueError(f"Scaling parameter s must be positive, got {s}")
            self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
            self.b = (nn.Parameter(torch.tensor(b, dtype=torch.float32))
                      if mode == "sssmax" else None)

    @classmethod
    def from_config(cls, config) -> "ScalableFactor":
        """Build from a model config, defaulting to the shipped SSSMax.

        ``getattr`` with defaults so a ``config.json`` written before these keys
        existed still rebuilds the module it was trained with.
        """
        return cls(
            s=float(getattr(config, "ssmax_s", 0.43)),
            b=float(getattr(config, "ssmax_b", 0.1)),
            base=float(getattr(config, "ssmax_base", 0.1)),
            mode=str(getattr(config, "ssmax_mode", "sssmax")),
        )

    def forward(
        self,
        lengths: torch.Tensor,
    ) -> torch.Tensor | None:
        """``lengths``: ``(B,)`` count of attended keys per sequence — the number
        of *real* (non-pad) tokens for full attention, or valid blocks for HRA.

        Returns a ``(B, 1, 1, 1)`` fp32 scale that broadcasts over a
        ``(B, n_head, T, head_dim)`` query, or ``None`` under ``mode="none"``.
        Scaling the query by a function of ``n`` is the same thing as scaling the
        logits by it; using the real token count rather than the padded sequence
        length keeps the attention temperature independent of how much padding a
        batch happens to carry — which matters when fine-tuning on the short,
        heavily-padded GLUE tasks.
        """
        if self.mode == "none":
            return None
        s = self.s.to(torch.float32)
        n = lengths.to(torch.float32).clamp_min(1.0)
        if self.mode == "ssmax":
            scale = s * torch.log(n)                # published SSMax
        else:                                       # sssmax
            # n >= 1 (clamped above) so s*log n >= 0; base > 0 and relu(b) >= 0
            # make the whole scale strictly positive at every length.
            scale = s * torch.log(n) + self.base + F.relu(self.b.to(torch.float32))
        return scale.reshape(-1, 1, 1, 1)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def build_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """ModernBERT-style padding mask (transformers ``_expand_mask``).

    Turns a ``(B, T)`` 0/1 key-padding mask into a ``(B, 1, 1, T)`` additive mask in
    the compute ``dtype``: ``0`` for real tokens and ``torch.finfo(dtype).min`` for
    padding (matching ModernBERT / ``AttentionMaskConverter._expand_mask`` — a
    dtype-saturating fill instead of a fixed ``-1e4``). It masks padding *keys* only
    and broadcasts over the query axis, so it never restricts attention across
    documents packed into the same sequence — exactly ModernBERT's behaviour.
    """
    inverted = 1.0 - attention_mask[:, None, None, :].to(dtype)
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@use_kernelized_func(apply_rotary_emb)
class NexteraSelfAttention(nn.Module):
    """Long-context attention.

    Designed to improve comprehension over long-range context by combining three
    ingredients: Stable Scalable-Softmax (SSSMax) query scaling, which keeps the attention
    distribution sharp as the sequence length grows; a per-channel sigmoid output
    gate (Gated Attention); and Separable DyT (tanh) normalisation of the queries/keys in
    place of LayerNorm. Built on rotary position embeddings and grouped-query KV
    sharing (``n_kv_head`` < ``n_head``) as full O(T^2) bidirectional attention.
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        config._attn_implementation = "sdpa"
        self.is_causal = False
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_gqa = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.scalable_factor = ScalableFactor.from_config(config)
        self.q_norm = SeparableDyT(self.head_dim)
        self.k_norm = SeparableDyT(self.head_dim)
        self.enable_gqa = True if self.n_head > self.n_kv_head else False

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, -1, self.head_dim)
        if z is None:
            k = self.c_k(x).view(B, T, -1, self.head_dim)
            v = self.c_v(x).view(B, T, -1, self.head_dim)
        else:
            k = self.c_k(z).view(B, T, -1, self.head_dim)
            v = self.c_v(z).view(B, T, -1, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # SSSMax length = number of real (non-pad) keys. attention_mask here is the
        # additive (B,1,1,T) key mask (real tokens are exactly 0), so count them;
        # fall back to the full length T when no mask is supplied.
        if attention_mask is not None:
            lengths = (attention_mask == 0).reshape(B, -1).sum(-1)
        else:
            lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        # Cast the (B,1,1,1) fp32 scale down BEFORE the broadcast multiply, so the
        # (B, n_head, T, head_dim) query is never promoted to fp32.
        # None under ssmax_mode="none" — the plain-softmax ablation arm.
        sf = self.scalable_factor(lengths)
        if sf is not None:
            q = q * sf.to(q.dtype)
        k = k.repeat_interleave(self.n_gqa, dim=1)
        v = v.repeat_interleave(self.n_gqa, dim=1)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.head_dim**-0.5)
        y = attn_output.view(B, T, -1).contiguous()
        g = F.sigmoid(self.c_gate(y))
        y = self.c_proj(g*y)
        return y

    def forward_packed(self, x: torch.Tensor, packed: "PackedBatch",
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream (see ``PackedBatch``)."""
        _, N, _ = x.size()
        q = self.q_norm(self.c_q(x).view(1, N, -1, self.head_dim))
        k = self.k_norm(self.c_k(x).view(1, N, -1, self.head_dim))
        v = self.c_v(x).view(1, N, -1, self.head_dim)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # SSSMax per token, from the length of the token's own document.
        sf = self.scalable_factor(packed.seqlen)
        if sf is not None:
            q = q * sf.reshape(1, 1, -1, 1).to(q.dtype)
        y = _packed_attention(q, k, v, packed, "doc", self.head_dim ** -0.5)
        y = y.reshape(1, N, -1)
        g = F.sigmoid(self.c_gate(y))
        return self.c_proj(g * y)


_SWA_MASK_CACHE: dict = {}

# Resolved once at import: probing inside forward() would graph-break under
# torch.compile, and the answer cannot change at runtime anyway.
try:
    from torch.nn.attention.flex_attention import (  # noqa: F401
        create_block_mask as _create_block_mask,
        flex_attention as _flex_attention_raw,
    )
    HAS_FLEX_ATTENTION = True
except Exception:  # pragma: no cover - older torch
    HAS_FLEX_ATTENTION = False


@torch.compiler.disable(recursive=True)
def _build_swa_block_mask(T: int, half: int, device):
    """Build the band ``BlockMask``, always OUTSIDE the compiled region.

    ``torch.compiler.disable`` is load-bearing, not a hint. ``create_block_mask``
    allocates tensors, and when it runs inside a graph compiled with
    ``mode="reduce-overhead"`` those tensors belong to the CUDA-graph memory pool.
    Caching them across invocations then hands the next step a buffer the graph
    replay has already overwritten — "accessing tensor output of CUDAGraphs that
    has been overwritten by a subsequent run". Building it eagerly gives ordinary
    tensors that are safe to keep, and it is a graph break that happens once per
    (T, half) rather than once per step, because the caller memoises the result.

    ``inference_mode(False)`` is the second half of the same rule: a table that
    outlives the call must not capture the ambient autograd mode either. Built
    under ``torch.inference_mode`` — which ``mteb_encoder.py`` and
    ``model_pplbench.py`` both use — the tensors are *inference tensors*, and the
    next training step at that sequence length dies with "Inference tensors cannot
    be saved for backward".
    """

    def in_window(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= half

    with torch.inference_mode(False), torch.no_grad():
        mask = _band_block_mask(in_window, T, half, device)
        if mask is not None:
            return mask
        return _create_block_mask(in_window, B=None, H=None, Q_LEN=T, KV_LEN=T,
                                  device=device)


def _band_block_mask(mask_mod, T: int, half: int, device):
    """The band's ``BlockMask`` written down directly, at block resolution.

    ``create_block_mask`` finds the block structure by *evaluating the mask on every
    (q, kv) pair*: a dense ``(T, T)`` bool plus the int64 ``q_idx - kv_idx`` and
    ``abs`` intermediates behind it -- about 17 bytes per pair, i.e. ~4 GiB at
    T=16384 and ~68 GiB at T=65536, to describe a mask whose whole point is that it is
    *not* T x T. It is a one-off per ``(T, half)`` (the caller memoises), which is
    exactly how it shows up: the first forward at a new length spikes or OOMs, every
    later one is small.

    For a band the answer is closed-form. Blocks ``i`` and ``j`` of size ``S`` that are
    ``d = |i - j|`` apart hold pairs at distances ``(d-1)*S + 1 .. (d+1)*S - 1`` (``0 ..
    S - 1`` for ``d = 0``), so a block is non-empty iff its smallest distance is
    ``<= half`` and full iff its largest is -- except that a block touching the ragged
    tail (``T % S != 0``) is never full, because ``create_block_mask`` pads the mask
    with zeros there. That reproduces its output exactly, from ``(T/S)^2`` booleans.
    Returns ``None`` when this torch has no ``BlockMask.from_kv_blocks`` to build from.
    """
    try:
        from torch.nn.attention.flex_attention import (
            _DEFAULT_SPARSE_BLOCK_SIZE as S,
            BlockMask,
        )
    except ImportError:
        return None
    n_blocks = -(-T // S)
    idx = torch.arange(n_blocks, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    nonempty = ((d - 1) * S + 1 <= half) | (d == 0)
    full = (d + 1) * S - 1 <= half
    if T % S:
        full[-1, :] = False
        full[:, -1] = False
    partial = nonempty & ~full

    def ordered(dense):
        # BlockMask's layout: per query block, the selected kv blocks first.
        dense = dense[None, None].to(torch.int32)
        return (dense.sum(-1).to(torch.int32),
                torch.argsort(dense, dim=-1, descending=True, stable=True).to(torch.int32))

    kv_num, kv_idx = ordered(partial)
    full_num, full_idx = ordered(full)
    try:
        return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx,
                                        BLOCK_SIZE=S, mask_mod=mask_mod,
                                        seq_lengths=(T, T))
    except TypeError:       # an older signature: let create_block_mask handle it
        return None


def _swa_block_mask(T: int, half: int, device):
    """Cached FlexAttention ``BlockMask`` for the band ``|q - kv| <= half``.

    This is what makes SWA actually cheaper rather than merely differently masked:
    the block mask lets FlexAttention *skip* the score blocks that lie entirely
    outside the band, so the cost is O(T * window) instead of O(T^2). A dense
    additive mask handed to SDPA would compute every score and then throw most of
    them away — no saving at all, and a (B, 1, T, T) mask that is 537 MB at
    T=8192.

    Depends only on ``(T, half, device)`` — not on the batch's padding, which is
    applied per-call through ``score_mod`` — so it is built once and reused.
    """
    key = (T, half, str(device))
    cached = _SWA_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = _build_swa_block_mask(T, half, device)
    _SWA_MASK_CACHE[key] = mask
    return mask


@torch.compiler.disable(recursive=True)
def _build_swa_additive_mask(T: int, half: int, dtype: torch.dtype, device):
    """Dense band mask, built eagerly — see :func:`_build_swa_block_mask` for why."""
    with torch.inference_mode(False), torch.no_grad():
        idx = torch.arange(T, device=device)
        blocked = (idx[:, None] - idx[None, :]).abs() > half
        bias = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(
            blocked, torch.finfo(dtype).min)
        return bias[None, None]


def _swa_additive_mask(T: int, half: int, dtype: torch.dtype, device):
    """Cached dense ``(1, 1, T, T)`` additive band mask — the SDPA fallback."""
    key = ("dense", T, half, dtype, str(device))
    cached = _SWA_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    bias = _build_swa_additive_mask(T, half, dtype, device)
    _SWA_MASK_CACHE[key] = bias
    return bias


_FLEX_ATTENTION = None
#: Recompile budget granted to the compiled FlexAttention. Dynamo's default (8)
#: is exhausted by an ordinary fine-tune + evaluate run -- train/no_grad, bf16
#: autocast on/off, ragged batches, evaluator batches -- after which dynamo stops
#: compiling *silently* and every call takes the unfused dense path (O(T^2)
#: memory: 64 GiB of scores for 16 x 8192 tokens).
FLEX_RECOMPILE_LIMIT = 256


def ensure_dynamo_recompile_budget(limit: int = FLEX_RECOMPILE_LIMIT) -> None:
    """Raise dynamo's per-function and accumulated recompile limits to ``limit``.

    Process-global and idempotent; only ever raises, never lowers. Called before
    the first FlexAttention compile, and exposed so entry points that load the
    model through remote code (a bundled copy of this file) can call it too.
    """
    try:
        import torch._dynamo.config as dynamo_config
    except Exception:  # noqa: BLE001 - dynamo missing / renamed: nothing to do
        return
    for name in ("recompile_limit", "cache_size_limit"):          # per code object
        if hasattr(dynamo_config, name) and getattr(dynamo_config, name) < limit:
            setattr(dynamo_config, name, limit)
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(dynamo_config, name) and getattr(dynamo_config, name) < limit * 4:
            setattr(dynamo_config, name, limit * 4)


def _flex_attention_fn():
    """Lazily ``torch.compile`` FlexAttention — it MUST be compiled.

    Uncompiled, ``flex_attention`` falls back to an implementation that
    materialises the whole ``(B, H, T, T)`` score matrix, which throws away both
    reasons the band exists: at T=8192 that is 4.3 GB of scores per call, and the
    block mask saves nothing.

    Do **not** pass ``dynamic=False`` here. It specialises the graph on the exact
    ``(B, T)`` shape, so a run that varies either — a short/long curriculum, a
    ragged last batch, uneven DDP shards — burns one recompile per shape and hits
    dynamo's ``recompile_limit`` (8 by default; raised to ``FLEX_RECOMPILE_LIMIT``
    below, which only postpones it). Past that dynamo stops compiling this
    function *silently* and every later call takes the unfused path; measured, the
    9th distinct sequence length was enough to trigger it. Letting dynamo mark the
    sequence dimension dynamic keeps one graph for every shape.
    """
    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is None:
        ensure_dynamo_recompile_budget()
        _FLEX_ATTENTION = torch.compile(_flex_attention_raw)
    return _FLEX_ATTENTION


# ---------------------------------------------------------------------------
# Unpadding: run a padded batch as one stream of its real tokens
# ---------------------------------------------------------------------------
#
# A padded (B, T) batch spends every GEMM, conv and attention score on its pad slots.
# With unpadding on (``config.unpadding`` / ``NexteraBERT.set_unpadding``) the encoder
# gathers the real tokens of every row back to back into a single (1, N) stream --
# ModernBERT's unpadded layout, N = the number of real tokens -- runs the stack on it
# and scatters the result back to (B, T) at the end. Each row is a *document* of the
# stream, and every mixer keeps the documents apart:
#
#   * attn / swa / hra attend within their document only: FlexAttention with a
#     document BlockMask built at block resolution (``_doc_block_mask``; the band is
#     intersected with it for swa), so a query block visits only the kv blocks of its
#     own documents and the cost is ~sum(L_d^2), not N^2. Without FlexAttention (CPU)
#     q/k/v are scattered back to a (docs, L_max) layout for SDPA instead -- exact, and
#     the GEMMs still see only the real tokens.
#   * RoPE positions, the SSSMax length and HRA's blocks restart at every document.
#   * the depthwise convs of SnowLily and HRA zero-pad at document edges
#     (``_packed_depthwise_conv``), so no tap reads a neighbouring document.
#
# Every row is therefore encoded exactly as if it were alone in the batch. That is
# *more* faithful than the padded path, whose HRA convs read the pad slots next to
# the last real tokens (the padding leak); the other mixers are pad-invariant there.


class PackedBatch(NamedTuple):
    """What the mixers need to run on an unpadded ``(1, N)`` stream.

    Built once per forward by :meth:`NexteraBERT._build_packed_batch`. ``N`` is the
    number of real tokens plus an optional filler tail (``pad_to_multiple_of`` of
    :meth:`NexteraBERT.set_unpadding`), which forms a document of its own and is
    dropped when the output is scattered back. The rotary tables themselves are
    computed by the encoder from ``pos`` / ``blk_pos`` and the fp32 inverse
    frequencies here, inside the (possibly compiled) forward where they fuse.
    """

    indices: torch.Tensor                  # (n_real,) flat (B*T) index of each real token
    gather: torch.Tensor                   # (N,) indices + the filler tail (reads index 0)
    pos: torch.Tensor                      # (N,) position inside the token's document
    seqlen: torch.Tensor                   # (N,) length of the token's document
    rope_inv: torch.Tensor | None          # (hd/2,) attn inverse frequencies ((N, hd/2) if NTK)
    swa_inv: torch.Tensor | None           # (hd/2,) swa inverse frequencies (swa_rope_base)
    blk_inv: torch.Tensor | None           # (hd/2,) hra inverse frequencies ((NB, hd/2) if NTK)
    blk_pos: torch.Tensor | None           # (NB,) position of each HRA block in its document
    blk_tok: torch.Tensor | None           # (NB, block_size) token index of each block slot
    blk_valid: torch.Tensor | None         # (NB, block_size) the slot holds a real token
    tok_blk: torch.Tensor | None           # (N,) HRA block of each token
    blk_len: torch.Tensor | None           # (NB,) blocks in the block's document (SSSMax length)
    doc_mask: object | None                # FlexAttention BlockMask: same document (attn)
    swa_mask: object | None                # ... and |i - j| <= swa_window // 2 (swa)
    blk_mask: object | None                # same document at block resolution (hra)
    tok_slot: torch.Tensor | None          # (N,) slot in the (docs, L_max) layout (SDPA path)
    tok_keys: torch.Tensor | None          # (docs, L_max) the slot holds a real key
    blk_slot: torch.Tensor | None          # (NB,) / (docs, NB_max): the same for HRA blocks
    blk_keys: torch.Tensor | None
    gap_idx: torch.Tensor                  # (N,) row in a layout with `gap` zeros between docs
    gap: int                               # widest one-sided conv reach in the model
    gap_len: int                           # rows of that layout


def _packed_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, packed: PackedBatch,
                           left: int) -> torch.Tensor:
    """Depthwise 1-D convolution of a packed ``(1, N, C)`` stream, per document.

    ``weight`` is the ``(C, k)`` kernel of a ``groups=C`` convolution whose output ``t``
    reads inputs ``t - left .. t - left + k - 1`` (``left = k // 2`` centred, ``k - 1``
    causal). A tap survives only when the input it reads lies in the same document --
    exactly the zero padding a conv applies at both ends of a sequence it sees alone.

    Two exact forms. Eager, the stream is scattered into a layout with ``packed.gap``
    zero rows between documents and convolved there by the same ``conv1d`` the padded
    path runs -- a handful of launches. Under ``torch.compile`` it is a shift-and-add
    over the ``k`` taps, accumulated in fp32 like the conv, which inductor fuses into
    one kernel over the ``(N, C)`` layout (no transposes around a cuDNN call).
    """
    k = weight.shape[-1]
    N = x.size(1)
    if not torch.compiler.is_compiling() and max(left, k - 1 - left) <= packed.gap:
        buf = x.new_zeros(packed.gap_len, x.size(-1)).index_copy(0, packed.gap_idx, x[0])
        y = F.conv1d(F.pad(buf.t()[None], (left, k - 1 - left)), weight[:, None, :],
                     groups=weight.size(0))
        return y[0].t().index_select(0, packed.gap_idx)[None]
    acc = torch.promote_types(x.dtype, torch.float32)
    xp = F.pad(x, (0, 0, left, k - 1 - left))
    w = weight.to(acc)
    out = None
    for j in range(k):
        off = j - left
        tap = xp[:, j:j + N].to(acc) * w[:, j]
        if off != 0:
            keep = (packed.pos + off >= 0) & (packed.pos + off < packed.seqlen)
            tap = tap * keep.to(acc)[None, :, None]
        out = tap if out is None else out + tap
    return out.to(x.dtype)


def _packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      packed: PackedBatch, kind: str, scale: float,
                      half: int | None = None) -> torch.Tensor:
    """Attention of a packed stream within each document. Returns ``(1, L, H, D)``.

    ``q`` is ``(1, H, L, D)`` and ``k``/``v`` ``(1, H_kv, L, D)``, where ``L`` counts
    tokens, or HRA blocks for ``kind="blk"``. ``kind`` selects the mask: ``"doc"``
    (same document), ``"swa"`` (same document and ``|i - j| <= half``) or ``"blk"``.
    """
    if q.dtype != v.dtype:
        # The DyT q/k norms return fp32 under autocast (fp32 gains); SDPA's autocast
        # would cast all three to the compute dtype, so do the same explicitly.
        q, k = q.to(v.dtype), k.to(v.dtype)
    block_mask = {"doc": packed.doc_mask, "swa": packed.swa_mask,
                  "blk": packed.blk_mask}[kind]
    if block_mask is not None:
        if k.size(1) != q.size(1):
            # Expanded explicitly: FlexAttention's enable_gqa=True has no valid kernel
            # config on this stack (NoValidChoicesError at head_dim 64, sm_120).
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
            v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
        out = _flex_attention_fn()(q, k, v, block_mask=block_mask, scale=scale)
        return out.transpose(1, 2)
    # SDPA fallback: scatter to a (docs, L_max) layout, mask the pad keys (and the band
    # for swa), gather the real rows back. The attention itself is then as padded as
    # before; everything around it still runs on the real tokens only.
    slot, keys = ((packed.blk_slot, packed.blk_keys) if kind == "blk"
                  else (packed.tok_slot, packed.tok_keys))
    rows, lmax = keys.shape

    def to_rows(t):
        buf = t.new_zeros(t.size(1), rows * lmax, t.size(-1))
        buf = buf.index_copy(1, slot, t[0])
        return buf.view(t.size(1), rows, lmax, t.size(-1)).transpose(0, 1)

    qr, kr, vr = to_rows(q), to_rows(k), to_rows(v)
    if kr.size(1) != qr.size(1):
        kr = kr.repeat_interleave(qr.size(1) // kr.size(1), dim=1)
        vr = vr.repeat_interleave(qr.size(1) // vr.size(1), dim=1)
    mask = keys[:, None, None, :]
    if kind == "swa":
        idx = torch.arange(lmax, device=q.device)
        mask = mask & ((idx[:, None] - idx[None, :]).abs() <= half)
    out = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=mask, scale=scale)
    out = out.transpose(0, 1).reshape(out.size(1), rows * lmax, out.size(-1))
    return out.index_select(1, slot).transpose(0, 1)[None]


def _same_doc(doc: torch.Tensor):
    def mask_mod(b, h, q_idx, kv_idx):
        return doc[q_idx] == doc[kv_idx]
    return mask_mod


def _same_doc_band(doc: torch.Tensor, half: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return (doc[q_idx] == doc[kv_idx]) & ((q_idx - kv_idx).abs() <= half)
    return mask_mod


def _flex_block_size() -> int:
    try:
        from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
        return int(_DEFAULT_SPARSE_BLOCK_SIZE)
    except ImportError:
        return 128


def _doc_block_layout(doc: torch.Tensor, cu: torch.Tensor, half: int | None = None,
                      S: int = 128):
    """Block layout of the "same document" (and ``|i - j| <= half``) FlexAttention mask.

    ``doc`` is the non-decreasing ``(L,)`` document id of every position and ``cu`` the
    ``(n_docs + 1,)`` start offsets of the documents. Returns ``(kv_num, kv_idx,
    full_num, full_idx)`` -- int32 ``(n,)`` / ``(n, n)``, ``n = ceil(L / S)`` -- in
    ``BlockMask``'s layout (per query block, its kv blocks first, ascending).

    Closed form, O(n^2) tiny integer ops and no sort: a query block holding documents
    ``first .. last`` sees exactly the kv blocks spanning ``cu[first] .. cu[last+1]-1``
    (a contiguous run), and a pair is *full* (mask_mod skipped) when the query block
    lies inside one document and the kv block inside the same one (another run,
    nested in the first). The band intersects both runs with the block distances of
    :func:`_band_block_mask`, so the partial blocks are at most two runs. Built per
    batch -- ``create_block_mask`` would evaluate the mask on all ``L x L`` pairs.
    """
    L = doc.numel()
    n = -(-L // S)
    i = torch.arange(n, device=doc.device)
    starts = i * S
    first = doc[starts]
    last = doc[(starts + S).clamp(max=L) - 1]
    vis_lo = cu[first] // S
    vis_hi = (cu[last + 1] - 1) // S
    full_lo = (cu[first] + S - 1) // S
    full_hi = cu[first + 1] // S - 1
    no_full = first != last
    if L % S:  # like create_block_mask: nothing touching the ragged tail is full
        full_hi = full_hi.clamp(max=n - 2)
        no_full[-1] = True
    if half is not None:
        reach, reach_full = (half - 1) // S + 1, (half + 1) // S - 1
        vis_lo = torch.maximum(vis_lo, i - reach)
        vis_hi = torch.minimum(vis_hi, i + reach)
        full_lo = torch.maximum(full_lo, i - reach_full)
        full_hi = torch.minimum(full_hi, i + reach_full)
    no_full |= full_hi < full_lo
    # Partial = the visible run minus the full run: a left piece and a right piece.
    full_lo = torch.where(no_full, vis_hi + 1, full_lo)
    full_hi = torch.where(no_full, vis_hi, full_hi)
    left = full_lo - vis_lo
    j = torch.arange(n, device=doc.device)[None, :]
    kv_idx = torch.where(j < left[:, None], vis_lo[:, None] + j,
                         full_hi[:, None] + 1 + j - left[:, None]).clamp(0, n - 1)
    full_idx = (full_lo[:, None] + j).clamp(0, n - 1)
    return ((left + vis_hi - full_hi).to(torch.int32), kv_idx.to(torch.int32),
            (full_hi - full_lo + 1).to(torch.int32), full_idx.to(torch.int32))


def _block_mask_from_layout(layout, L: int, mask_mod, S: int = 128):
    """A ``BlockMask`` over ``L`` positions from a :func:`_doc_block_layout`."""
    from torch.nn.attention.flex_attention import BlockMask

    kv_num, kv_idx, full_num, full_idx = (t[None, None] for t in layout)
    kwargs = dict(BLOCK_SIZE=S, mask_mod=mask_mod, seq_lengths=(L, L))
    try:
        # The transposed (q-major) indices are only read by the backward pass.
        return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, **kwargs,
                                        compute_q_blocks=torch.is_grad_enabled())
    except TypeError:  # older signature without compute_q_blocks
        return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, **kwargs)


def _doc_block_mask(doc: torch.Tensor, cu: torch.Tensor, mask_mod, half: int | None = None):
    """FlexAttention ``BlockMask`` for "same document" (and ``|i - j| <= half``)."""
    S = _flex_block_size()
    return _block_mask_from_layout(_doc_block_layout(doc, cu, half, S), doc.numel(),
                                   mask_mod, S)


_INV_FREQ_CACHE: dict = {}


def _rope_inv_freq(base, head_dim: int, device) -> torch.Tensor:
    """``1 / base^(2i/d)`` exactly as ``_precompute_rotary_embeddings`` computes it,
    memoised per ``(base, head_dim, device)``. Called from the eager packing builder,
    and built with inference mode off, per the cross-call cache rule."""
    key = (float(base), int(head_dim), str(device))
    inv = _INV_FREQ_CACHE.get(key)
    if inv is None:
        with torch.inference_mode(False), torch.no_grad():
            ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            inv = 1.0 / (base ** (ch / head_dim))
        _INV_FREQ_CACHE[key] = inv
    return inv


def _to_device_views(tensors: dict, device, dtype) -> dict:
    """Move a dict of small host tensors to ``device`` in ONE copy: they are
    concatenated into a single ``dtype`` buffer and handed back as views of it."""
    flat = [t.reshape(-1).to(dtype) for t in tensors.values()]
    buf = torch.cat(flat).to(device)
    out, offset = {}, 0
    for (name, t), f in zip(tensors.items(), flat):
        out[name] = buf[offset:offset + f.numel()].view(t.shape)
        offset += f.numel()
    return out


@use_kernelized_func(apply_rotary_emb)
class NexteraSlidingWindowAttention(nn.Module):
    """Gated sliding-window attention with rotary position embeddings.

    ``block_types`` entry ``"swa"``. This is :class:`NexteraSelfAttention`'s local
    counterpart: the same rotary position embeddings, the same per-channel sigmoid
    **output gate**, the same Separable DyT normalisation of q/k and the same
    grouped-query KV sharing — only the receptive field changes to a symmetric
    band ``|i - j| <= swa_window // 2``. Scalable-Softmax is deliberately absent:
    SSSMax exists to keep the distribution sharp as the *number of attended keys*
    grows, and a band attends at most ``swa_window + 1`` of them however long the
    sequence is.

    Position is carried entirely by RoPE, so the layer holds no positional
    parameters at all — the score is just::

        A[i,j] = (rope(q)_i . rope(k)_j) / sqrt(head_dim),   |i - j| <= half

    Cost. On CUDA the band is a FlexAttention ``BlockMask``, so the blocks outside
    it are skipped and the attention is O(T * window) instead of O(T^2); with no
    positional bias to carry, ``score_mod`` is needed only to apply padding, and
    drops out entirely when there is none. Elsewhere (notably CPU) it falls back
    to the shared ``ALL_ATTENTION_FUNCTIONS`` interface — the same call ``attn``
    and ``hra`` make — with the band folded into the additive mask. That mask is
    head-independent here, so it stays ``(B, 1, T, T)`` and broadcasts.

    NOTE on the RoPE base. The encoder feeds ``swa`` layers their own rotary
    table built from ``config.swa_rope_base`` (default 10000) instead of the
    global ``config.rope_base`` (default 100000, sized for full attention at
    8192 context): over a band of at most ``swa_window + 1`` keys the global
    base's low-frequency channels barely rotate, so a matched small base
    restores the positional resolution — the same reason ModernBERT gives its
    local layers their own much smaller base (10000). The swa table is never
    NTK-rescaled: within-band relative distances are bounded by the window
    however long the sequence grows (see ``_precompute_rotary_embeddings``).
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        config._attn_implementation = "sdpa"
        self.is_causal = False
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_gqa = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Total window width; the band reaches half of it on each side, so a query
        # attends up to window + 1 keys.
        self.window = int(getattr(config, "swa_window", 256))
        self.half = max(1, self.window // 2)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.q_norm = SeparableDyT(self.head_dim)
        self.k_norm = SeparableDyT(self.head_dim)
        self.attn_scale = self.head_dim ** -0.5
        self.enable_gqa = True if self.n_head > self.n_kv_head else False

    def extra_repr(self) -> str:
        return f"window={self.window} (+-{self.half})"

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        src = x if z is None else z
        q = self.q_norm(self.c_q(x).view(B, T, -1, self.head_dim))
        k = self.k_norm(self.c_k(src).view(B, T, -1, self.head_dim))
        # Normalise, then rotate — the same order NexteraSelfAttention uses, so the
        # DyT curve sees unrotated channels.
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        v = self.c_v(src).view(B, T, -1, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = k.repeat_interleave(self.n_gqa, dim=1)
        v = v.repeat_interleave(self.n_gqa, dim=1)

        if HAS_FLEX_ATTENTION and q.is_cuda:
            block_mask = _swa_block_mask(T, self.half, q.device)
            score_mod = None
            if attention_mask is not None:
                # Padding rides along as a score bias rather than in the block mask:
                # it varies per batch, and rebuilding the block mask every step would
                # cost more than the extra blocks it could skip.
                pad_bias = attention_mask.to(q.dtype).reshape(B, T)

                def score_mod(score, b, h, q_idx, kv_idx):
                    return score + pad_bias[b, kv_idx]

            if q.dtype != v.dtype:
                # The DyT q/k norms return fp32 under autocast while v is bf16 — the
                # mismatch _packed_attention resolves the same way. Left to
                # flex_attention's autocast rule, the cast happens inside the op, and
                # on queries shorter than 128 tokens (where inductor picks the
                # flex_decoding kernel) that gives wrong attention: cos ~0 against
                # fp32 on torch 2.12.1. Cast after pad_bias is built, because
                # finfo(fp32).min is -inf in bf16.
                q, k = q.to(v.dtype), k.to(v.dtype)
            attn_output = _flex_attention_fn()(
                q, k, v, score_mod=score_mod, block_mask=block_mask,
                scale=self.attn_scale).transpose(1, 2)
        else:
            # No per-head term any more, so the band mask stays head-independent
            # and this can go through the shared attention interface.
            mask = _swa_additive_mask(T, self.half, q.dtype, q.device)
            if attention_mask is not None:
                mask = mask + attention_mask
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attention_interface(
                self, q, k, v, mask, dropout=0.0, scaling=self.attn_scale)

        y = attn_output.reshape(B, T, -1).contiguous()
        g = F.sigmoid(self.c_gate(y))
        y = self.c_proj(g*y)
        return y

    def forward_packed(self, x: torch.Tensor, packed: PackedBatch,
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream: the band, cut at
        document edges (the packed stream is contiguous, so ``|i - j|`` inside one
        document is the in-document distance)."""
        _, N, _ = x.size()
        q = self.q_norm(self.c_q(x).view(1, N, -1, self.head_dim))
        k = self.k_norm(self.c_k(x).view(1, N, -1, self.head_dim))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        v = self.c_v(x).view(1, N, -1, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = _packed_attention(q, k, v, packed, "swa", self.attn_scale, half=self.half)
        y = y.reshape(1, N, -1)
        g = F.sigmoid(self.c_gate(y))
        return self.c_proj(g * y)


@use_kernelized_func(apply_rotary_emb)
class NexteraHRA(nn.Module):
    """Hierarchical Recurrent Attention (HRM-inspired).

    Borrows the Hierarchical Reasoning Model (HRM) idea to *pseudo*-generate a
    hierarchical recurrent structure for efficient computation: a linear
    projection, a depthwise conv (DSConv) and a multiplicative linear gate produce
    per-token features, which are mean-pooled into a per-block hidden state (the "block
    mean"); queries/keys/values are then generated from that block-level state.
    Attention runs at the (much shorter) block level and the result is broadcast
    back to token resolution and fused with a token-wise residual, emulating
    multi-scale recurrence at O(T * n_block) instead of O(T^2). Shares
    NexteraSelfAttention's SSSMax scaling, gated output and Separable DyT norm, with
    rotary embeddings.
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        config._attn_implementation = "sdpa"
        self.is_causal = False
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_gqa = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.block_size = config.block_size
        # Depthwise conv (DSConv) + a linear gate produce the
        # per-block hidden state (block mean); q/k/v are generated from that
        # block-level state rather than pooled after being generated per token.
        self.qkv_conv = DSConv(
            channels=self.n_embd,
            kernel_size=config.block_size//2+1
        )
        self.qkv_in = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.qkv_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_C = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_o = DSConv(
            channels=self.n_embd,
            kernel_size=config.block_size//2+1
        )
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.scalable_factor = ScalableFactor.from_config(config)
        self.q_norm = SeparableDyT(self.head_dim)
        self.k_norm = SeparableDyT(self.head_dim)
        self.enable_gqa = True if self.n_head > self.n_kv_head else False

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        pad = (
            self.block_size - T % self.block_size
        ) % self.block_size
        n_block = (T+pad)//self.block_size
        # Per-token validity for the masked block mean. attention_mask is the additive
        # (B,1,1,T) key mask (real tokens are exactly 0); pooling over real tokens only
        # keeps a partially filled block from being diluted by its pad positions (a
        # plain mean divides by the full block_size), so a real token's block
        # representation does not depend on how much padding the batch carries.
        if attention_mask is not None:
            valid = (attention_mask == 0).reshape(B, T)
            if pad:
                valid = F.pad(valid, (0, pad), value=False)
            vm = valid.reshape(B, n_block, self.block_size, 1).to(x.dtype)
            denom = vm.sum(dim=2).clamp_min(1.0)
        else:
            vm = None

        def block_mean(src: torch.Tensor) -> torch.Tensor:
            # Depthwise conv (DSConv) + a linear gate produce the
            # per-token features; mean-pool them within each block to get the
            # block-level hidden state that q/k/v are generated from. F.pad always
            # materialises a new tensor, so skip it when T already tiles block_size.
            i = self.qkv_in(src)
            h = self.qkv_conv(i) * self.qkv_gate(src)
            if pad:
                h = F.pad(h, (0, 0, 0, pad))
            h = h.reshape(B, n_block, self.block_size, -1)
            if vm is not None:
                return (h * vm).sum(dim=2) / denom
            return h.mean(dim=2)

        # A single block mean feeds all three projections (self-attention); for cross
        # attention (z given) the keys/values pool the cross sequence instead.
        hq = block_mean(x)
        hkv = hq if z is None else block_mean(z)
        qb = self.c_q(hq).view(B, n_block, self.n_head, self.head_dim)
        kb = self.c_k(hkv).view(B, n_block, self.n_kv_head, self.head_dim)
        vb = self.c_v(hkv).view(B, n_block, self.n_kv_head, self.head_dim)
        cos, sin = cos[:, :n_block], sin[:, :n_block]
        # Normalise before RoPE, so DyT acts on the channels before rotation.
        qb = self.q_norm(qb)
        kb = self.k_norm(kb)
        qb, kb = apply_rotary_emb(qb, cos, sin), apply_rotary_emb(kb, cos, sin)
        qb, kb, vb = qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)
        # Build a proper block-level mask by aggregating per-block token validity.
        # attention_mask is (B,1,1,T) additive; pad the alignment tail with the same
        # masked (finfo.min) value, then take max per block (a block is "visible" if
        # any token in it is unmasked). Built before SSSMax so the number of *valid*
        # blocks (not the padded block count) sets the SSSMax length.
        if attention_mask is not None:
            if pad:
                attention_mask = F.pad(
                    attention_mask, (0, pad),
                    value=torch.finfo(attention_mask.dtype).min)
            block_mask = attention_mask.reshape(B, 1, 1, n_block, self.block_size).max(dim=-1).values
            lengths = (block_mask == 0).reshape(B, -1).sum(-1)
        else:
            block_mask = None
            lengths = torch.full((B,), n_block, device=x.device, dtype=torch.long)
        sf = self.scalable_factor(lengths)
        if sf is not None:
            qb = qb * sf.to(qb.dtype)
        kb = kb.repeat_interleave(self.n_gqa, dim=1)
        vb = vb.repeat_interleave(self.n_gqa, dim=1)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            qb,
            kb,
            vb,
            block_mask,
            dropout=0.0,
            scaling=self.head_dim**-0.5)
        C = self.c_C(x)
        y = attn_output.view(B, n_block, -1).contiguous()
        # Broadcast each block output to the block_size tokens of its block
        # (expand + reshape, which avoids building an index tensor).
        y = (
            y.unsqueeze(2)
            .expand(B, n_block, self.block_size, y.size(-1))
            .reshape(B, n_block * self.block_size, -1)[:, :T]
            + C
        )
        g = F.sigmoid(self.c_gate(y))
        y = self.c_o(y)
        y = self.c_proj(g*y)
        return y

    def forward_packed(self, x: torch.Tensor, packed: PackedBatch,
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream.

        Blocks restart at every document (``packed.blk_tok`` lists each block's
        tokens), so a block never pools two documents and the block attention,
        its RoPE positions and its SSSMax length are per document -- what
        :meth:`forward` computes for the same text on its own. ``cos``/``sin`` are
        the global table at *block* positions (``packed.blk_pos``).
        """
        _, N, _ = x.size()
        h = self.qkv_conv.forward_packed(self.qkv_in(x), packed) * self.qkv_gate(x)
        # Masked block mean, in the same dtype arithmetic as forward().
        vm = packed.blk_valid.to(x.dtype)[..., None]                   # (NB, bs, 1)
        hb = (h[0][packed.blk_tok] * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        hb = hb[None]                                                   # (1, NB, C)
        n_blk = hb.size(1)
        qb = self.q_norm(self.c_q(hb).view(1, n_blk, self.n_head, self.head_dim))
        kb = self.k_norm(self.c_k(hb).view(1, n_blk, self.n_kv_head, self.head_dim))
        vb = self.c_v(hb).view(1, n_blk, self.n_kv_head, self.head_dim)
        qb, kb = apply_rotary_emb(qb, cos, sin), apply_rotary_emb(kb, cos, sin)
        qb, kb, vb = qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)
        sf = self.scalable_factor(packed.blk_len)
        if sf is not None:
            qb = qb * sf.reshape(1, 1, -1, 1).to(qb.dtype)
        y = _packed_attention(qb, kb, vb, packed, "blk", self.head_dim ** -0.5)
        # Broadcast each block's output back to its tokens.
        y = y.reshape(1, n_blk, -1)[:, packed.tok_blk] + self.c_C(x)
        g = F.sigmoid(self.c_gate(y))
        y = self.c_o.forward_packed(y, packed)
        return self.c_proj(g * y)


class SnowstormConv(nn.Module):
    """State-space 1D convolution (SSM) as a *separable* low-rank operator.
 
    Optimised version. The original realised the temporal mixing by unfolding
    the low-rank features to (B, T, r*k) and applying a single dense
    out_proj of shape (C, r*k). With r=128, k=5 that is r*k = 640 ~= C = 768,
    so out_proj was effectively a full (C, C) GEMM and the op cost
    ~= B*T*C*r*(k+1) MACs -- on par with a depthwise-separable conv, hence
    "no faster than DS conv".
 
    Here the temporal mixing is factored into a *depthwise* SSM kernel applied
    in the r-dimensional latent space, followed by a single up-projection:
 
        z = in_proj(x)                              # C -> r   (B*T*C*r)
        a = A(x)                                    # C -> r   (B*T*C*r)
        s[t] = sum_j kernel[:, j] * z[t+j-k//2]     # depthwise (B*T*r*k, cheap;
                                                    #   causal: z[t+j-k+1])
        y = out_proj(s + a)                         # r -> C   (B*T*r*C)
 
    Total ~= 3*B*T*C*r MACs -- roughly (k+1)/3 = 2x fewer than the original for
    k=5, while keeping the same down/up low-rank structure and an explicit,
    per-latent-channel temporal (SSM) filter. The depthwise mixing is run as a
    single fused depthwise conv1d (groups=r) so it is one kernel launch -- a
    naive Python shift-and-add over the k taps would instead issue k separate
    memory-bound elementwise kernels that dominate on GPU once the GEMMs shrink.
 
    NOTE: this changes the parameterisation (out_proj is now (C, r) instead of
    (C, r*k), plus a (r, k) kernel), so weights are not interchangeable with the
    original -- the module must be (re)trained. Output shape is unchanged.
    When non-causal, kernel_size must be odd to preserve T.
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        low_rank: int,
        kernel_size: int,
        bias: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.k = kernel_size
        self.r = low_rank
        self.in_proj = nn.Linear(in_channels, low_rank, bias=bias)
        # up-projection now maps r -> C (was r*k -> C)
        self.out_proj = nn.Linear(low_rank, out_channels, bias=bias)
        # depthwise temporal kernel in the latent space: one length-k filter
        # per low-rank channel. Initialised as a centred impulse (last tap when
        # causal) so the conv path starts as a plain low-rank pointwise map.
        kernel = torch.zeros(low_rank, kernel_size)
        kernel[:, (kernel_size - 1) if causal else (kernel_size // 2)] = 1.0
        kernel += 0.02 * torch.randn(low_rank, kernel_size)
        self.kernel = nn.Parameter(kernel)
        self.causal = causal
        self.register_buffer("cache", None)
        self.A = nn.Linear(in_channels, low_rank, bias=bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.A(x)
        z = self.in_proj(x).transpose(1, 2)                   # (B, r, T)
        # depthwise temporal mixing as a SINGLE fused conv kernel.
        # (An equivalent Python shift-and-add over the k taps issues k separate
        # memory-bound elementwise kernels, which dominate on GPU once the GEMMs
        # are small -- conv1d collapses them into one launch.)
        w = self.kernel.unsqueeze(1)                          # (r, 1, k)
        if self.causal:
            z = F.pad(z, (self.k - 1, 0))                     # left-pad T: reads z[t-k+1..t]
            s = F.conv1d(z, w, groups=self.r)
        else:
            s = F.conv1d(z, w, groups=self.r, padding=self.k // 2)
        s = s.transpose(1, 2)                                 # (B, T, r)
        y = self.out_proj(s + residual)                                  # (B, T, C)
        return y

    def forward_packed(self, x: torch.Tensor, packed: PackedBatch) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream, per document."""
        residual = self.A(x)
        left = self.k - 1 if self.causal else self.k // 2
        s = _packed_depthwise_conv(self.in_proj(x), self.kernel, packed, left)
        return self.out_proj(s + residual)


class DSConv(nn.Module):
    """Depthwise conv (groups=C), optionally followed by a pointwise 1x1 (``point=True``)."""

    def __init__(self, channels: int, kernel_size: int, point: bool = False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False,
        )
        if point:
            self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        else:
            self.pointwise = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.depthwise(h)
        if self.pointwise is not None:
            h = self.pointwise(h)
        return h.transpose(1, 2)

    def forward_packed(self, x: torch.Tensor, packed: PackedBatch) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream, per document."""
        h = _packed_depthwise_conv(x, self.depthwise.weight[:, 0], packed,
                                   self.depthwise.padding[0])
        if self.pointwise is not None:
            h = F.linear(h, self.pointwise.weight[:, :, 0])
        return h


# SnowLily implementation
# Copyright 2026 Rikka Botan. All rights reserved
class SnowLily(nn.Module):
    def __init__(
        self,
        config
    ):
        """Dynamic liquid (LTC) convolution layer.

        SnowStormConv + Liquid Layer by time-enhanced dYnamical adaptation.
        Dynamicises the coefficients of a Liquid Time-Constant (LTC) system so the
        per-token mixing adapts to the input — raising expressiveness — while
        SnowStormConv supplies the efficient (GEMM-based) state operator. A
        bidirectional liquid convolution module inspired by LFM2.LFM2ConvBlock.
        ```
        Formulation:

        x ∈ ℝ^{B×S×E}
        y ∈ ℝ^{B×S×E}

        y = B ⋅ ∏ᵢ₌ⱼ⁽ʲ⁺ᵏ⁾ Aᵢ ⋅ xᵢ

        ----------------------------------------
        Algorithm: SnowLily
        ----------------------------------------
        Input: x: (B, S, E)
        Output: y: (B, S, E)
            1: A <- SnowstormConv(x); B <- Linear(x)
            2: x₁ <- Linear(x) (+ Linear(z) if z is given)
            3: x₂: (B, S, E) <- DepthwiseConv1D(A*x₁)
            4: x₃: (B, S, E) <- B*x₂
            5: y: (B, S, E) <- Linear(x₃)
            6: return y
        ----------------------------------------
        ```
        """
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.d_head = config.expand*config.n_embd//config.n_head
        self.n_kernel = config.n_kernel
        self.ltc_conv = SnowstormConv(
            in_channels=self.n_embd,
            out_channels=self.n_embd,
            low_rank=self.d_head,
            kernel_size=self.n_kernel
        )
        self.x_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_embd,
            bias=False
        )
        self.B_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_embd,
            bias=False
        )
        self.state_conv = DSConv(channels=self.n_embd, kernel_size=self.n_kernel)
        self.c_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_embd,
            bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seql, _ = hidden_states.size()
        # This is a 1D-convolution (state-space) mixer, so a pad token's embedding
        # would otherwise leak into nearby real tokens through the conv window. Zero
        # the pad positions first (attention_mask is the additive (B,1,1,T) key mask,
        # real tokens exactly 0) so they contribute nothing to their neighbours.
        if attention_mask is not None:
            tok_mask = (attention_mask == 0).reshape(bsz, seql, 1).to(hidden_states.dtype)
            hidden_states = hidden_states * tok_mask
        A = self.ltc_conv(hidden_states)
        x = self.x_proj(hidden_states)
        B = self.B_proj(hidden_states)
        if z is not None:
            x = x + self.x_proj(z)
        xA = self.state_conv(A*x)
        xAB = B * xA
        y = self.c_proj(xAB)
        return y

    def forward_packed(self, hidden_states: torch.Tensor, packed: PackedBatch,
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream. There are no pad
        slots to zero: both convolutions stop at document edges instead."""
        A = self.ltc_conv.forward_packed(hidden_states, packed)
        x = self.x_proj(hidden_states)
        B = self.B_proj(hidden_states)
        xA = self.state_conv.forward_packed(A * x, packed)
        return self.c_proj(B * xA)


class LFM2Conv(nn.Module):
    """LFM2's gated short-convolution block, made bidirectional.

    Liquid AI's LFM2 replaces most of its attention layers with this block
    (``LFM2ShortConv`` / ``LFM2ConvBlock``): one input projection produces three
    streams, the first gates the input, a short **depthwise** convolution mixes it
    over time, and the third gates the result::

        B, C, x = in_proj(u).chunk(3, dim=-1)
        y = C * conv(B * x)
        out = c_proj(y)

    LFM2 is a decoder, so its convolution is causal; an encoder sees the whole
    sequence, so the same kernel is applied CENTRED (``padding = k//2``), exactly
    as SnowLily applies its own convolutions. That is the only departure from the
    published block — the gating, the single fused ``in_proj`` and the depthwise
    kernel are unchanged.

    Included as the middle rung of the local-mixer ablation: SnowLily adds a
    liquid (input-dependent, low-rank state-space) operator on top of gating,
    LFM2Conv keeps the gating but not the state operator, and
    :class:`DSConvMixer` keeps neither.
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.n_embd = config.n_embd
        self.k = config.n_kernel
        if self.k % 2 == 0:
            raise ValueError(
                f"n_kernel must be odd for a centred (non-causal) conv, got {self.k}")
        self.in_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.conv = nn.Conv1d(
            self.n_embd, self.n_embd, self.k,
            padding=self.k // 2, groups=self.n_embd, bias=False,
        )
        # Named c_proj because NexteraBERT.init_weights gives every mixer's output
        # projection the GPT-2 scaled-residual init through that attribute.
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seql, _ = hidden_states.size()
        # A convolution would otherwise smear a pad token's embedding into its real
        # neighbours; zero the pad positions first (attention_mask is the additive
        # (B,1,1,T) key mask, real tokens exactly 0), as SnowLily does.
        if attention_mask is not None:
            tok_mask = (attention_mask == 0).reshape(bsz, seql, 1).to(hidden_states.dtype)
            hidden_states = hidden_states * tok_mask
        gate_b, gate_c, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        if z is not None:
            x = x + self.in_proj(z).chunk(3, dim=-1)[2]
        x = gate_b * x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.c_proj(gate_c * x)

    def forward_packed(self, hidden_states: torch.Tensor, packed: PackedBatch,
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream, per document."""
        gate_b, gate_c, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        x = _packed_depthwise_conv(gate_b * x, self.conv.weight[:, 0], packed, self.k // 2)
        return self.c_proj(gate_c * x)


class DSConvMixer(nn.Module):
    """Plain depthwise-separable convolution as the sequence mixer — the control
    condition for :class:`SnowLily` and :class:`LFM2Conv`::

        y = c_proj( SiLU( pointwise( depthwise( in_proj(u) ) ) ) )

    No gating, no input-dependent (liquid) coefficients, no state-space
    parameterisation: what is left is the convolution itself. It reuses the
    :class:`DSConv` helper that HRA and SnowLily already contain, so the ablation
    changes what surrounds the convolution rather than the convolution.

    Parameters are ``3·n_embd²`` (+ the depthwise kernel), within a few percent of
    SnowLily's ``3·n_embd² + 3·n_embd·d_head``; LFM2Conv's fused 3-way projection
    makes it ``4·n_embd²``. Report the counts alongside the scores — these blocks
    are matched in *shape*, not exactly in parameter count.
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.n_embd = config.n_embd
        self.k = config.n_kernel
        if self.k % 2 == 0:
            raise ValueError(
                f"n_kernel must be odd for a centred (non-causal) conv, got {self.k}")
        self.in_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.conv = DSConv(channels=self.n_embd, kernel_size=self.k, point=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seql, _ = hidden_states.size()
        if attention_mask is not None:
            tok_mask = (attention_mask == 0).reshape(bsz, seql, 1).to(hidden_states.dtype)
            hidden_states = hidden_states * tok_mask
        x = self.in_proj(hidden_states)
        if z is not None:
            x = x + self.in_proj(z)
        x = self.conv(x)
        return self.c_proj(F.silu(x))

    def forward_packed(self, hidden_states: torch.Tensor, packed: PackedBatch,
                       cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream, per document."""
        x = self.conv.forward_packed(self.in_proj(hidden_states), packed)
        return self.c_proj(F.silu(x))


class MLP(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inter, bias=False)
        self.c_proj = nn.Linear(config.n_inter, config.n_embd, bias=False)
        self.n_inter = config.n_inter

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.c_fc(x)
        x = x.relu().square()
        x = self.c_proj(x)
        return x


class RippleBloomUGM(nn.Module):
    def __init__(self, config):
        """Unified Granularity Module (per-expert squared-ReLU MLP experts).
 
        Each expert is a squared-ReLU MLP, per expert and routed:
 
            u  = x · up                        # (per-expert up,  like c_fc)
            h  = u.relu().square()             # squared-ReLU
            o  = (coef * h) · down             # (per-expert down, like c_proj)
 
        Two-stage selection:
          (1) coarse, expert-level: the router (`c_proj`) picks top-`k_routed` routed
              experts per *sequence* (selection on the seq-MEAN logits, so the weight
              gather stays (B, topk, ...)).
          (2) fine, neuron-level: squared-ReLU decides which of the `lora_rank`
              neurons fire inside each selected expert.
 
        Coefficient: computed from the *pre-mean* (per-token) router logits -- only
        the *selection* uses the sequence mean. The shared and selected-routed experts
        are softmaxed *jointly*, so the shared experts also carry a (token-wise)
        coefficient instead of a constant 1. At init (c_proj small-random) the joint
        softmax is near-uniform, so the block stays alive while allowing diverse
        expert selection across different inputs from step 0.

        Trainer compatibility:
          * router named `c_proj` (nn.Linear) -> small random init (std=0.02) in
            both ``RippleBloomUGM.__init__`` and ``NexteraBERT.init_weights``.
          * up / down are raw nn.Parameters -> inited here.
          * ``router_z_loss`` and ``load_balance_loss`` set every forward.
 
        `perturbation_element` is unused here.
        """
        super().__init__()
        self.n_embd = config.n_embd
        self.lora_rank = config.lora_rank
        self.n_group = config.n_inter // self.lora_rank
        self.topk = config.topk                          # total active = shared + routed
        self.n_shared = getattr(config, "n_shared", 1)   # always-on experts (group 0..)
        self.n_routed = self.n_group - self.n_shared
        self.k_routed = self.topk - self.n_shared
        assert self.k_routed >= 0, "topk must be >= n_shared"
        assert self.n_routed >= self.k_routed, "n_routed must be >= k_routed"
 
        # Per-expert weights. Index [0:n_shared] = shared, [n_shared:] = routed.
        self.up = nn.Parameter(torch.empty(self.n_group, self.lora_rank, self.n_embd))  # like c_fc
        self.down = nn.Parameter(torch.empty(self.n_group, self.n_embd, self.lora_rank))  # like c_proj
 
        # Router over ALL experts (shared + routed): the first n_shared columns are
        # the shared experts' coefficient logits, the rest are the routed logits used
        # for both selection and coefficients. Small random init (std 0.02).
        self.c_proj = nn.Linear(self.n_embd, self.n_group, bias=False)
  
        # Raw Parameters bypass _init_weights -> init here.
        nn.init.normal_(self.up, mean=0.0, std=self.n_embd ** -0.5)
        nn.init.normal_(self.down, mean=0.0, std=(self.topk * self.lora_rank) ** -0.5)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
 
        self.router_z_loss = None
        self.load_balance_loss = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.size()
        ns, kr = self.n_shared, self.k_routed

        # Per-token validity (1 real / 0 pad). attention_mask arrives as the additive
        # (B,1,1,T) key mask (real tokens are exactly 0); collapse it to (B,T,1) and
        # use it for every sequence reduction below so padding positions never leak
        # into the router — critical on the short, heavily-padded GLUE tasks.
        tok_mask = (attention_mask == 0).reshape(B, T, 1).to(x.dtype) if attention_mask is not None else None

        # --- router logits for ALL experts (PRE-MEAN, token-wise) --------------
        all_logits = self.c_proj(x)                          # (B,T,n_group)
        shared_logits = all_logits[..., :ns]                 # (B,T,ns)
        routed_logits = all_logits[..., ns:]                 # (B,T,n_routed)
        z_per_token = torch.logsumexp(routed_logits.float(), dim=-1).square()   # (B,T)
        if tok_mask is not None:
            m = tok_mask[..., 0].float()                     # (B,T)
            self.router_z_loss = (z_per_token * m).sum().div(m.sum().clamp_min(1.0)).clamp(max=100.0)
        else:
            self.router_z_loss = z_per_token.mean().clamp(max=100.0)

        # --- stage 1: SELECTION on the sequence MEAN (routed only) -------------
        # Masked mean so padding positions never bias which experts are selected.
        if tok_mask is not None:
            seq_logits = (routed_logits * tok_mask).sum(dim=1) / tok_mask.sum(dim=1).clamp_min(1.0)
        else:
            seq_logits = routed_logits.mean(dim=1)           # (B,n_routed)  <- mean only for selection
        routed_idx = torch.topk(seq_logits, kr, dim=-1).indices   # (B,kr)

        # Router balance regularisation: penalise the variance of per-sample
        # routing logits.  High variance ⇒ the softmax is sharp ⇒ a few experts
        # grab most of the coefficient mass ⇒ expert collapse risk.  Unlike the
        # Switch-Transformer f·P formulation (which is a mathematical constant
        # ≡ kr when the softmax is near-uniform), logit variance is always
        # responsive to changes in c_proj and provides a nonzero gradient.
        self.load_balance_loss = seq_logits.float().var(dim=-1).mean().clamp(max=100.0)
        shared_idx = torch.arange(ns, device=x.device).expand(B, ns)
        expert_idx = torch.cat([shared_idx, routed_idx + ns], dim=1)   # (B,topk)
 
        # --- coefficient from the PRE-MEAN (token-wise) logits -----------------
        # shared + selected-routed are softmaxed jointly, so the shared experts now
        # carry a (token-wise) coefficient too instead of a constant 1.
        sel = routed_idx.unsqueeze(1).expand(B, T, kr)                 # (B,T,kr)
        routed_sel_logits = torch.gather(routed_logits, 2, sel)        # (B,T,kr)
        active_logits = torch.cat([shared_logits, routed_sel_logits], dim=-1)  # (B,T,topk)
        coef = F.softmax(active_logits, dim=-1)                        # (B,T,topk)
 
        # --- gather the active experts (sequence-level) ------------------------
        w_up = self.up[expert_idx]                           # (B,topk,lr,E)
        w_down = self.down[expert_idx]                       # (B,topk,E,lr)
 
        # --- stage 2: per-expert squared-ReLU MLP -------------------------------
        u = torch.einsum("bte, bkre -> btkr", x, w_up)       # (B,T,topk,lr)
        h = u.relu().square()                                # squared-ReLU
        h = h * coef.unsqueeze(-1)                           # token-wise expert coefficient
 
        o = torch.einsum("btkr, bker -> bte", h, w_down)     # sum over experts + lr
        return o


MIXERS = {
    "attn": NexteraSelfAttention,
    "swa": NexteraSlidingWindowAttention,
    "hra": NexteraHRA,
    "lily": SnowLily,
    # Ablation-only local mixers (the paper's local-mixer ablation): drop-in replacements for "lily"
    # that keep the block wiring and change only the local operator.
    "lfm2conv": LFM2Conv,
    "dsconv": DSConvMixer,
}


def build_mixer(config, layer_idx: int) -> nn.Module:
    """Per-layer sequence mixer, selected by ``config.block_types[layer_idx]``."""
    lt = config.block_types[layer_idx]
    if lt not in MIXERS:
        raise ValueError(
            f"Unknown block_types[{layer_idx}] = '{lt}'. "
            f"Choose one of {tuple(MIXERS)}.")
    return MIXERS[lt](config)


def build_mlp(config) -> nn.Module:
    """Feed-forward, selected by ``config.mlp_class`` ("mlp" | "ugm").

    ``getattr`` with a default so a ``config.json`` written before ``mlp_class``
    existed still loads as the dense MLP it was trained with.
    """
    mlp_class = str(getattr(config, "mlp_class", "mlp")).lower()
    if mlp_class == "mlp":
        return MLP(config)
    if mlp_class == "ugm":
        return RippleBloomUGM(config)
    raise ValueError(f"Unknown mlp_class '{mlp_class}'. Choose 'mlp' or 'ugm'.")


class Block(nn.Module):
    """Standard pre-norm residual block: ``x = x + f(norm(x))`` at both sites.

    The mixer ``f`` is whatever ``config.block_types[layer_idx]`` names (attn /
    swa / hra / lily, or the ablation-only lfm2conv / dsconv) and the feed-forward is whatever ``config.mlp_class`` names; the
    residual wiring is the same for every mixer, so there is one path here.
    """

    def __init__(
        self,
        config,
        layer_idx
    ):
        super().__init__()
        self.attn = build_mixer(config, layer_idx)
        self.mlp = build_mlp(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.lt = config.block_types[layer_idx]

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), z, cos, sin, attention_mask)
        x = x + self.mlp(self.mlp_norm(x), attention_mask)
        return x

    def forward_packed(
        self,
        x: torch.Tensor,
        packed: PackedBatch,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """:meth:`forward` on an unpadded ``(1, N, C)`` stream (see ``PackedBatch``).
        The dense MLP is token-wise, so only the mixer needs a packed variant."""
        x = x + self.attn.forward_packed(self.attn_norm(x), packed, cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def masked_mean_pool(hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over the real (non-pad) tokens of a ``(B, T, C)`` sequence.

    ``attention_mask`` is the raw ``(B, T)`` 0/1 key-padding mask (1 = real token);
    pad positions drop out of both the sum and the count. Falls back to a plain mean
    over ``T`` when no mask is given. Shared by the sequence-classification pooler, the
    embedding/retrieval ``feature_extraction`` and COCO-LM's SCL projection so all
    single-vector representations pool the same way.
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)      # (B, T, 1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class NexteraBERTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ModernBERT-style head transform over the pooled vector: dense (no bias,
        # ``classifier_bias=False``) -> GELU -> LayerNorm with a learnable gain
        # (``norm_bias=False``). The affine gain sits AFTER the normalisation, so
        # the overall scale of the pooled vector stays learnable — the STS-B
        # regression head reads it.
        self.dense = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(config.n_embd, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked mean pooling over the real (non-pad) tokens (ModernBERT's
        # ``classifier_pooling="mean"``). The conv-heavy encoder (few attention
        # layers, a conv final layer) makes [CLS] a weak sequence aggregator, and
        # MLM/ELECTRA pretraining never trains [CLS] as a summary, so averaging
        # every token transfers to the GLUE classification tasks better than
        # reading [CLS] alone.
        pooled = masked_mean_pool(hidden_states, attention_mask)
        return self.norm(self.activation(self.dense(pooled)))


class NexteraBERT(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Segment (token-type) embeddings, added to the token embeddings for
        # sentence-pair inputs. Zero-initialised (see init_weights) so the layer is
        # a no-op on the single-segment-pretrained backbone until fine-tuning.
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd)
        self.pooler = NexteraBERTPooler(config)
        self.blocks = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        # Final norm only: kept as LayerNorm (the blocks use RMSNorm) so the
        # exported hidden states are centred as well as scaled.
        self.out_norm = nn.LayerNorm(config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self._emb_proj = None
        # (key, cos, sin) memos for the rotary tables — slot 0 = the global table
        # (attn/hra, config.rope_base), slot 1 = the "swa" layers' table
        # (config.swa_rope_base). Plain attributes, not buffers, so the state_dict
        # is unchanged and the caches never travel with a checkpoint.
        self._rope_cache = [None, None]
        self._has_swa = any(t == "swa" for t in config.block_types)
        # Opt-in activation checkpointing; see set_gradient_checkpointing.
        self.gradient_checkpointing = False
        # Opt-in unpadded execution; see set_unpadding.
        self.unpadding = bool(getattr(config, "unpadding", False))
        self.unpad_multiple = 1
        self.unpad_attention = "auto"
        self._conv_reach_cache = None
        self.init_weights()

    def set_gradient_checkpointing(self, enable: bool = True):
        """Recompute each block in backward instead of storing its activations.

        Trades roughly a third more compute for a large drop in activation memory —
        the encoder keeps one block's intermediates instead of ``n_layer``'s. Worth it
        exactly where memory, not arithmetic, caps throughput: the 8192-token phase-2
        run, where the per-example activation footprint otherwise forces the batch so
        small that the GPU never fills. At the 1024-token phase-1 length, where a
        useful batch already fits, leave it off — there it is pure overhead.

        Only active when ``self.training`` is True; inference builds no graph to
        trade away.
        """
        self.gradient_checkpointing = bool(enable)
        return self

    def set_unpadding(self, enable: bool = True, pad_to_multiple_of: int | None = None,
                      attention: str | None = None):
        """Run padded batches unpadded (ModernBERT-style): the real tokens of every row
        are packed into one stream, so no GEMM, conv or attention is spent on pad
        slots. See ``PackedBatch`` for how each mixer keeps the rows apart.

        Every row then comes out as a B=1 run of the same text would. That includes
        HRA, whose convs otherwise read the pad slots next to a row's last tokens, so
        a head fine-tuned on padded batches sees slightly different features. Calls
        without an ``attention_mask``, cross-attention calls (``cross_ids``) and
        ``mlp_class="ugm"`` (routing on per-sequence means) keep the padded path.

        ``pad_to_multiple_of`` rounds the stream length up with a filler document,
        which bounds the number of distinct shapes a compiled model sees.
        ``attention="sdpa"`` forces the SDPA fallback where FlexAttention would be
        used; ``"auto"`` restores the default.
        """
        self.unpadding = bool(enable)
        if pad_to_multiple_of is not None:
            self.unpad_multiple = max(1, int(pad_to_multiple_of))
        if attention is not None:
            if attention not in ("auto", "sdpa"):
                raise ValueError(f"attention must be 'auto' or 'sdpa', got {attention!r}")
            self.unpad_attention = attention
        return self

    def _unpad_this_call(self, attention_mask, cross_ids) -> bool:
        if not self.unpadding or attention_mask is None or cross_ids is not None:
            return False
        return all(hasattr(b.attn, "forward_packed") and isinstance(b.mlp, MLP)
                   for b in self.blocks)

    @torch.compiler.disable(recursive=True)
    def _build_packed_batch(self, attention_mask: torch.Tensor) -> PackedBatch:
        """Pack the real tokens of a ``(B, T)`` batch -- see ``PackedBatch``.

        Eager on purpose (``torch.compiler.disable``): the stream length is the number
        of real tokens, a data-dependent size. So the ``(B, T)`` mask is copied to the
        host once -- the forward's one sync -- and the whole layout (positions, HRA
        blocks, the BlockMasks' kv-block runs) is worked out there with a few dozen
        tiny host ops, then shipped back in one copy per dtype. Launching the same
        arithmetic as GPU kernels cost ~3x more, which short batches cannot hide.
        Rows are packed in order and each row's tokens in column order, so any mask
        works (right- or left-padded).
        """
        cfg = self.config
        device = attention_mask.device
        types = set(cfg.block_types)
        mask = attention_mask.detach().to("cpu").to(torch.bool)
        B = mask.size(0)
        lens = mask.sum(dim=1)
        n_real = int(lens.sum())
        mult = max(1, int(self.unpad_multiple))
        n = max(-(-n_real // mult) * mult, 1)
        # The filler tail (if any) is one more document, B, of its own.
        lens = torch.cat([lens, lens.new_tensor([n - n_real])])
        cu = F.pad(lens.cumsum(0), (1, 0))
        doc = torch.repeat_interleave(torch.arange(B + 1), lens)
        pos = torch.arange(n) - cu[doc]
        indices = mask.flatten().nonzero().flatten()
        ints = dict(indices=indices, gather=F.pad(indices, (0, n - n_real)), pos=pos,
                    seqlen=lens[doc], doc=doc, lens=lens)
        if "hra" in types:
            bs = int(cfg.block_size)
            nbl = (lens + bs - 1) // bs
            cub = F.pad(nbl.cumsum(0), (1, 0))
            blk_doc = torch.repeat_interleave(torch.arange(B + 1), nbl)
            blk_pos = torch.arange(blk_doc.numel()) - cub[blk_doc]
            off = blk_pos[:, None] * bs + torch.arange(bs)
            blk_valid = off < lens[blk_doc][:, None]
            ints.update(blk_doc=blk_doc, blk_pos=blk_pos, blk_len=nbl[blk_doc],
                        blk_tok=torch.where(blk_valid, cu[blk_doc][:, None] + off, 0),
                        blk_valid=blk_valid, tok_blk=cub[doc] + pos // bs)

        # The eager conv layout: `gap` zero rows between consecutive documents.
        gap = self._conv_reach()
        ints["gap_idx"] = torch.arange(n) + gap * doc

        use_flex = (HAS_FLEX_ATTENTION and device.type == "cuda"
                    and self.unpad_attention != "sdpa")
        S = _flex_block_size()
        half = max(1, int(getattr(cfg, "swa_window", 256)) // 2)
        layouts = {}
        if use_flex:
            if "attn" in types:
                layouts["doc"] = _doc_block_layout(doc, cu, None, S)
            if "swa" in types:
                layouts["swa"] = _doc_block_layout(doc, cu, half, S)
            if "hra" in types:
                layouts["blk"] = _doc_block_layout(blk_doc, cub, None, S)
        else:
            # key 0 of every row stays on, so an empty row never softmaxes over nothing
            lmax = max(int(lens.max()), 1)
            keys = torch.arange(lmax)
            ints.update(tok_slot=doc * lmax + pos,
                        tok_keys=(keys < lens[:, None]) | (keys == 0))
            if "hra" in types:
                lmax_b = max(int(nbl.max()), 1)
                keys = torch.arange(lmax_b)
                ints.update(blk_slot=blk_doc * lmax_b + blk_pos,
                            blk_keys=(keys < nbl[:, None]) | (keys == 0))

        dev = _to_device_views(ints, device, torch.int64)
        masks = dict.fromkeys(("doc", "swa", "blk"))
        if layouts:
            flat = {f"{k}{i}": t for k, lay in layouts.items() for i, t in enumerate(lay)}
            lay = _to_device_views(flat, device, torch.int32)
            for k in layouts:
                if k == "blk":
                    L, mask_mod = dev["blk_doc"].numel(), _same_doc(dev["blk_doc"])
                elif k == "swa":
                    L, mask_mod = n, _same_doc_band(dev["doc"], half)
                else:
                    L, mask_mod = n, _same_doc(dev["doc"])
                masks[k] = _block_mask_from_layout(
                    tuple(lay[f"{k}{i}"] for i in range(4)), L, mask_mod, S)

        hd = self.head_dim
        base = getattr(cfg, "rope_base", 100000)
        rope_inv = blk_inv = swa_inv = None
        if types & {"attn", "hra"}:
            rope_inv = blk_inv = _rope_inv_freq(base, hd, device)
            train_len = getattr(cfg, "rope_ntk_train_len", 0) or 0
            if train_len > 0 and int(lens.max()) > train_len:
                # Dynamic NTK per document: each is stretched by its own length, as
                # _precompute_rotary_embeddings stretches a sequence encoded alone.
                ch = torch.arange(0, hd, 2, dtype=torch.float32, device=device)
                grow = (dev["lens"].to(torch.float64) / train_len) ** (hd / (hd - 2))
                long_inv = 1.0 / ((base * grow).to(torch.float32)[:, None] ** (ch / hd))
                per_doc = torch.where((dev["lens"] > train_len)[:, None], long_inv, rope_inv)
                rope_inv = per_doc[dev["doc"]]
                blk_inv = per_doc[dev["blk_doc"]] if "hra" in types else None
        if "swa" in types:
            swa_inv = _rope_inv_freq(getattr(cfg, "swa_rope_base", 10000), hd, device)

        return PackedBatch(
            indices=dev["indices"], gather=dev["gather"], pos=dev["pos"],
            seqlen=dev["seqlen"], rope_inv=rope_inv, swa_inv=swa_inv, blk_inv=blk_inv,
            blk_pos=dev.get("blk_pos"), blk_tok=dev.get("blk_tok"),
            blk_valid=dev.get("blk_valid"), tok_blk=dev.get("tok_blk"),
            blk_len=dev.get("blk_len"),
            doc_mask=masks["doc"], swa_mask=masks["swa"], blk_mask=masks["blk"],
            tok_slot=dev.get("tok_slot"),
            tok_keys=None if "tok_keys" not in dev else dev["tok_keys"].bool(),
            blk_slot=dev.get("blk_slot"),
            blk_keys=None if "blk_keys" not in dev else dev["blk_keys"].bool(),
            gap_idx=dev["gap_idx"], gap=gap, gap_len=n + gap * B,
        )

    def _conv_reach(self) -> int:
        """Widest one-sided reach of any token-level convolution in the stack -- the
        zero gap the eager packed conv needs between documents."""
        if self._conv_reach_cache is None:
            reach = 0
            for m in self.modules():
                if isinstance(m, SnowstormConv):
                    reach = max(reach, m.k - 1 if m.causal else m.k // 2)
                elif isinstance(m, nn.Conv1d):
                    k, p = m.kernel_size[0], m.padding[0]
                    reach = max(reach, p, k - 1 - p)
            self._conv_reach_cache = reach
        return self._conv_reach_cache

    def _encode_packed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The unpadded encoder: ``(B, T, C)`` hidden states after ``out_norm``, with
        zeros at the pad positions (the padded path leaves meaningless values there)."""
        B, T = input_ids.shape
        packed = self._build_packed_batch(attention_mask)
        x = self.embedding(input_ids.reshape(-1)[packed.gather])[None]      # (1, N, C)
        if self._emb_proj is not None:
            x = self._emb_proj(x)
        if token_type_ids is not None:
            x = x + self.token_type_embeddings(token_type_ids.reshape(-1)[packed.gather])[None]
        # Rotary tables at per-document positions, with _precompute_rotary_embeddings'
        # arithmetic (fp32 outer product, then cos/sin, then the activation dtype).
        ropes = {}
        for lt, positions, inv in (("attn", packed.pos, packed.rope_inv),
                                   ("swa", packed.pos, packed.swa_inv),
                                   ("hra", packed.blk_pos, packed.blk_inv)):
            if inv is not None:
                freqs = positions.to(torch.float32)[:, None] * inv
                ropes[lt] = (freqs.cos().to(x.dtype)[None, :, None, :],
                             freqs.sin().to(x.dtype)[None, :, None, :])
        for block in self.blocks:
            cos, sin = ropes.get(block.lt, (None, None))
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block.forward_packed, x, packed, cos, sin, use_reentrant=False)
            else:
                x = block.forward_packed(x, packed, cos, sin)
        x = self.out_norm(x)
        out = x.new_zeros(B * T, x.size(-1)).index_copy(
            0, packed.indices, x[0, :packed.indices.numel()])
        return out.view(B, T, -1)

    def init_weights(
        self
    ):
        self.apply(self._init_weights)
        # Segment embeddings start at zero: adding this layer to a backbone that was
        # pretrained on single-segment inputs is then a no-op at fine-tune step 0,
        # and the segment signal is learned from the pair-task data. The two rows
        # break symmetry naturally from their differing token contexts.
        nn.init.zeros_(self.token_type_embeddings.weight)
        n_layer = len(self.blocks)
        for block in self.blocks:
            # MLP router: small random init so different inputs produce
            # different expert selections from step 0.  Zero-init makes
            # every input yield identical logits → the softmax stays flat →
            # the logit-variance balance loss sits at 0 with no useful gradient.
            # (Under mlp_class="mlp" this same tensor is the dense down
            # projection; the small std is a fine init for it too.)
            if hasattr(block.mlp, "c_proj"):
                torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02)
            # Every mixer's output projection (c_proj): scaled residual init
            # (GPT-2 style) so inter-position communication is alive from step 0.
            # Zero-init here leaves these branches dead until c_proj grows from
            # zero, which stalls the loss at the unigram baseline when LR is low.
            fan_in = block.attn.c_proj.weight.size(1)
            std = (1.0 / math.sqrt(fan_in)) / math.sqrt(2 * n_layer)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=std)

    def _init_weights(
        self,
        module
    ):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Small init (standard transformer 0.02). The embedding is tied to the
            # generator's LM head, so a large std blows up the output logits
            # (std ~ sqrt(n_embd)) and the initial MLM loss to ~80 instead of
            # ln(vocab) ~ 10.8. The encoder input path is normalised (pre-norm
            # RMSNorm in each block), so the small scale is harmless there.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _precompute_rotary_embeddings(
        self,
        x: torch.Tensor,
        base: int | None = None,
        ntk: bool = True,
        slot: int = 0,
    ) -> List[torch.Tensor]:
        device = x.device
        if base is None:
            base = getattr(self.config, "rope_base", 100000)
        # Dynamic NTK-aware scaling (train-free context extension, as used by
        # nomic-bert-2048 and validated for RoPE encoders by LongEmbed): when the
        # input is longer than the trained context (config.rope_ntk_train_len > 0),
        # stretch the base so every channel's rotation angle at position T stays
        # within the range covered during training, instead of extrapolating the
        # mid/low-frequency channels into angles the model has never seen. The
        # scale is identity for T <= rope_ntk_train_len, so enabling this cannot
        # change short-sequence behaviour. NexteraHRA slices this same table at
        # block resolution (T/block_size positions vs a trained T_train/block_size
        # range) — token and block positions extend by the same T/T_train factor,
        # so the single shared rescaled table stays consistent for both.
        # ``ntk=False`` (the "swa" table) skips the rescaling entirely: RoPE scores
        # depend only on the RELATIVE rotation q-k, and a band layer never sees a
        # relative distance beyond swa_window // 2 however long the sequence grows,
        # so its trained angle range is never exceeded — stretching its base would
        # change within-band behaviour for no benefit.
        train_len = (getattr(self.config, "rope_ntk_train_len", 0) or 0) if ntk else 0
        T = x.size(1)
        # The table depends only on (T, base, train_len, dtype, device) — none of which
        # change within a step — so in EAGER we memoise it instead of rebuilding
        # arange/outer/cos/sin on every forward.
        #
        # The memo is deliberately skipped while tracing: comparing the key would make
        # dynamo guard on the concrete T, which specialises the graph per sequence
        # length and blows the recompile limit on a dynamic-padding loader (after which
        # dynamo gives up and runs the model in eager for the rest of the run).
        # ``is_compiling()`` folds to a constant at trace time, so the branch — and the
        # guard with it — disappears from the compiled graph, where inductor hoists and
        # fuses the table construction anyway.
        compiling = torch.compiler.is_compiling()
        key = None
        if not compiling:
            key = (T, base, train_len, x.dtype, device)
            cached = self._rope_cache[slot]
            if cached is not None and cached[0] == key:
                return cached[1], cached[2]
        if train_len > 0 and T > train_len:
            base = base * (T / train_len) ** (self.head_dim / (self.head_dim - 2))
        # Built with inference mode explicitly OFF: this table is memoised above and
        # therefore outlives the call. Created under torch.inference_mode — which
        # mteb_encoder.py and model_pplbench.py both use — cos/sin would be
        # *inference tensors*, and the next training step at that sequence length
        # would die in apply_rotary_emb with "Inference tensors cannot be saved for
        # backward". They never need grad, so no_grad costs nothing.
        with torch.inference_mode(False), torch.no_grad():
            channel_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                                         device=device)
            inv_freq = 1.0 / (base ** (channel_range / self.head_dim))
            t = torch.arange(x.size(1), dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)
            cos, sin = freqs.cos(), freqs.sin()
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)
            cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        if not compiling:
            self._rope_cache[slot] = (key, cos, sin)
        return cos, sin

    def _swa_rotary_embeddings(self, x: torch.Tensor):
        """The "swa" layers' own rotary table (``config.swa_rope_base``, default
        10000), or ``None`` when the stack has no swa layer. Kept separate from the
        global table because a band of at most ``swa_window + 1`` keys wants a much
        smaller base (see the NexteraSlidingWindowAttention docstring), and never
        NTK-rescaled (relative distances inside the band are length-invariant).
        """
        if not self._has_swa:
            return None
        return self._precompute_rotary_embeddings(
            x, base=getattr(self.config, "swa_rope_base", 10000), ntk=False, slot=1)

    def feature_extraction(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._unpad_this_call(attention_mask, cross_ids):
            return masked_mean_pool(self._encode_packed(input_ids, attention_mask),
                                    attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.embedding(input_ids)
        if self._emb_proj is not None:
            x = self._emb_proj(x)
        # ModernBERT-style padding mask (finfo(dtype).min) in the compute dtype.
        additive_mask = build_additive_mask(attention_mask, x.dtype)
        cos, sin = self._precompute_rotary_embeddings(x)
        swa_rope = self._swa_rotary_embeddings(x)
        if cross_ids is not None:
            z = self.embedding(cross_ids)
            if self._emb_proj is not None:
                z = self._emb_proj(z)
        else:
            z = None
        for block in self.blocks:
            bcos, bsin = swa_rope if block.lt == "swa" else (cos, sin)
            x = block(x, z, bcos, bsin, additive_mask)
        # Masked-mean pooling over the real tokens (out_norm applied per token first),
        # matching NexteraBERTPooler so the embedding/retrieval representation pools
        # the same way as the sequence-classification path.
        x = self.out_norm(x)
        return masked_mean_pool(x, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pad_mask = attention_mask  # (B, T) 0/1 mask before conversion
        if self._unpad_this_call(attention_mask, cross_ids):
            x = self._encode_packed(input_ids, attention_mask, token_type_ids)
            return x, self.pooler(x, pad_mask)
        x = self.embedding(input_ids)
        if self._emb_proj is not None:
            x = self._emb_proj(x)
        # Add segment embeddings (BERT-style word + segment). Skipped
        # when token_type_ids is None so single-segment / pretraining paths are unchanged.
        if token_type_ids is not None:
            x = x + self.token_type_embeddings(token_type_ids)
        # ModernBERT-style padding mask (finfo(dtype).min) in the compute dtype.
        if attention_mask is not None:
            attention_mask = build_additive_mask(attention_mask, x.dtype)
        cos, sin = self._precompute_rotary_embeddings(x)
        swa_rope = self._swa_rotary_embeddings(x)
        if cross_ids is not None:
            z = self.embedding(cross_ids)
            if self._emb_proj is not None:
                z = self._emb_proj(z)
        else:
            z = None
        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                bcos, bsin = swa_rope if block.lt == "swa" else (cos, sin)
                # use_reentrant=False is the variant that keeps working under
                # torch.compile and with inputs that do not require grad.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, z, bcos, bsin, attention_mask, use_reentrant=False)
        else:
            for block in self.blocks:
                bcos, bsin = swa_rope if block.lt == "swa" else (cos, sin)
                x = block(x, z, bcos, bsin, attention_mask)
        x = self.out_norm(x)
        pooler_output = self.pooler(x, pad_mask)
        return x, pooler_output


class NexteraBERTPredictionHead(nn.Module):
    """ModernBERT-style transform between the encoder and an output projection:
    dense (no bias) -> GELU -> LayerNorm (no bias) — ``ModernBertPredictionHead``.

    Under ``tie_word_embeddings`` the vocab decoder reuses the input embedding
    matrix; this transform lets the final hidden states differ from the embedding
    geometry, so the tied matrix is not forced to serve as both input lookup and
    output projection.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.n_embd, bias=False)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = NexteraBERTPredictionHead(config)
        # Decoder bias (ModernBERT ``decoder_bias=True``), never tied: it absorbs
        # the per-token frequency prior so a tied embedding matrix doesn't have to.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_head(self.head(x))  # (B,L,V)


class NexteraBERTForEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encoder.feature_extraction(input_ids, cross_ids, attention_mask)
        return hidden  # (B, hidden_size) pooled, normalised


class NexteraBERTForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        self.head = NexteraBERTPredictionHead(config)
        # Decoder bias (ModernBERT ``decoder_bias=True``) stays untied even when
        # the weight is tied: it absorbs the per-token frequency prior so the
        # embedding matrix doesn't have to.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        nn.init.zeros_(self.lm_head.bias)
        if config.tie_word_embeddings:
            # tie the LM head weight with the input embedding matrix
            self.lm_head.weight = self.encoder.embedding.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        select_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``(B, L, V)`` logits, or ``(N, V)`` when ``select_mask`` is given.

        ``select_mask`` is a ``(B, L)`` boolean mask of the positions whose logits the
        caller actually consumes — the masked positions, under every MLM objective
        here. Gathering the hidden states *before* the head transform and vocab
        projection is mathematically identical to projecting everything and gathering
        after (both are position-wise), but at the usual 15-25% mask ratio it cuts
        the projection's FLOPs and its ``(B, L, V)`` activation — which backward has
        to keep alive, and which is the single largest tensor in the step — by 4-6x.
        """
        hidden, _ = self.encoder(input_ids, cross_ids, attention_mask)
        if select_mask is not None:
            hidden = hidden[select_mask]                # (N, C)
        return self.lm_head(self.head(hidden))


class NexteraBERTForDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        # ELECTRA's discriminator head: dense -> GELU -> linear probe. A bare
        # linear probe would force the encoder to linearise the RTD decision in
        # its final hidden states — the same states downstream fine-tuning reads;
        # the dense layer takes that pressure instead.
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.classifier = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, _ = self.encoder(input_ids, cross_ids, attention_mask)
        logits = self.classifier(F.gelu(self.dense(hidden))).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            
        return logits, loss


def _ugm_modules(parent):
    """The RippleBloomUGM submodules of ``parent``, found once and memoised.

    The two collectors below run on every forward, and walking ``parent.modules()``
    each time is a full Python traversal of the encoder per micro-step — pure
    overhead, and entirely wasted under the default ``mlp_class="mlp"`` where there
    are no UGMs at all. Stored straight into ``__dict__`` so ``nn.Module.__setattr__``
    does not try to interpret the list, and so it stays out of the state_dict.
    """
    cache = parent.__dict__.get("_ugm_cache")
    if cache is None:
        cache = tuple(m for m in parent.modules() if isinstance(m, RippleBloomUGM))
        parent.__dict__["_ugm_cache"] = cache
    return cache


def _collect_load_balance_loss(modules):
    """Sum the most-recent load-balance auxiliary loss over every RippleBloomUGM."""
    losses = [m.load_balance_loss
              for parent in modules
              for m in _ugm_modules(parent)
              if m.load_balance_loss is not None]
    if not losses:
        return None
    return torch.stack(losses).sum()


def _collect_router_z_loss(modules):
    """Sum the most-recent ST-MoE router z-loss over every RippleBloomUGM found in
    ``modules`` (returns ``None`` when there are none)."""
    losses = [m.router_z_loss
              for parent in modules
              for m in _ugm_modules(parent)
              if m.router_z_loss is not None]
    if not losses:
        return None
    return torch.stack(losses).sum()


def freeze_pretraining_unused(*encoders: nn.Module) -> list:
    """Freeze the encoder parameters no pretraining objective can ever reach.

    Two groups are unreachable *by construction*, not by chance:

    * ``token_type_embeddings`` — pretraining never passes ``token_type_ids``, so
      the layer is skipped entirely. It is zero-initialised on purpose and the
      segment signal is meant to be learned during paired fine-tuning.
    * ``pooler`` — every pretraining head reads the token-level hidden states and
      throws ``pooler_output`` away; only the sequence-classification head uses it.

    Left trainable they receive no gradient, and DDP with the default
    ``find_unused_parameters=False`` then aborts on the *second* iteration with
    "Expected to have finished reduction in the prior iteration". Freezing is
    better than turning that flag on: DDP simply skips non-``requires_grad``
    parameters, so there is no per-iteration graph traversal, and the optimizer
    (which filters on ``requires_grad``) drops them too. They are still saved in
    the checkpoint and become trainable again in the downstream heads.

    Returns the names frozen, for logging.
    """
    frozen = []
    for enc in encoders:
        for name, module in (("token_type_embeddings", getattr(enc, "token_type_embeddings", None)),
                             ("pooler", getattr(enc, "pooler", None))):
            if module is None:
                continue
            for pname, p in module.named_parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen.append(f"{name}.{pname}")
    return frozen


class NexteraBERTForElectraTrainer(nn.Module):
    def __init__(
        self,
        generator_config,
        discriminator_config,
        mask_id: int | None = None,
        pad_token_id: int | None = None,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        mask_ratio: float = 0.15,
        disc_lambda: float = 50.0,
        gen_temperature: float = 1.0,
        router_z_loss_coef: float = 1e-3,
        load_balance_loss_coef: float = 1e-2,
        tie_embeddings: bool = True,
        true_vocab_size: int | None = None,
    ):
        """
        ## NexteraBERT class for electra like training

        ``mask_ratio`` follows ELECTRA (~15%): too high a value destroys the
        generator's context, so its replacements become trivially detectable and
        the discriminator — the backbone you keep — stops receiving useful
        gradient. ``gen_temperature`` divides the generator's sampling logits; a
        value > 1 flattens the distribution (less plausible, easier-to-detect
        replacements), < 1 sharpens it (more often the original token). 1.0 is
        the ELECTRA default.

        ``router_z_loss_coef`` weights the ST-MoE router z-loss summed over every
        RippleBloomUGM router (generator + discriminator); 0 disables it.
        """
        # special-token ids default to the config's (ModernBERT tokenizer)
        mask_id = discriminator_config.mask_token_id if mask_id is None else mask_id
        pad_token_id = discriminator_config.pad_token_id if pad_token_id is None else pad_token_id
        cls_token_id = discriminator_config.cls_token_id if cls_token_id is None else cls_token_id
        sep_token_id = discriminator_config.sep_token_id if sep_token_id is None else sep_token_id
        super().__init__()
        self.generator = NexteraBERTForMaskedLM(generator_config)
        self.discriminator = NexteraBERTForDiscriminator(discriminator_config)
        self.disc_lambda = disc_lambda
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio
        self.gen_temperature = gen_temperature
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.pad_token_id = pad_token_id
        self.special_token_ids = {pad_token_id, cls_token_id, sep_token_id}
        self.vocab_size = generator_config.vocab_size
        # config.vocab_size may be padded past the real tokenizer vocab (multiple
        # of 64 for GPU tile alignment); sampling must never emit the padded ids.
        self.true_vocab_size = true_vocab_size or generator_config.vocab_size

        # share token embeddings between generator and discriminator
        if tie_embeddings:
            gen_n_embd = generator_config.n_embd
            disc_n_embd = discriminator_config.n_embd
            disc_emb = self.discriminator.encoder.embedding
            self.generator.encoder.embedding = disc_emb
            if gen_n_embd == disc_n_embd:
                if hasattr(self.generator, 'lm_head'):
                    self.generator.lm_head.weight = disc_emb.weight
            else:
                self.generator.encoder._emb_proj = nn.Linear(
                    disc_n_embd, gen_n_embd, bias=False)

        # Neither encoder's pooler / segment embeddings can receive gradient from
        # the ELECTRA objective — see freeze_pretraining_unused.
        freeze_pretraining_unused(self.generator.encoder, self.discriminator.encoder)

        self.eval_metrics = {
            "disc_accuracy": DiscriminatorAccuracy(pad_token_id),
            "mlm_accuracy": TokenAccuracy(pad_token_id)
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        run_discriminator: bool = True,
    ) -> dict:
        """ELECTRA forward.

        ``run_discriminator=False`` runs only the generator MLM path (no sampling,
        no discriminator forward) — used by the generator-warmup phase so the
        discriminator is neither run nor updated until the generator is competent.
        """
        rand = torch.rand_like(input_ids, dtype=torch.float)
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            valid_mask &= (input_ids != token_id)
        replace_mask = (rand < self.mask_ratio) & valid_mask

        original_token_ids = input_ids.clone()

        # ELECTRA: every selected position is shown to the generator as [MASK].
        # (No BERT-style 80/10/10 — injecting random tokens here produces
        #  obviously-wrong fakes that make the discriminator's job trivial.)
        masked_input_ids = input_ids.clone()
        masked_input_ids[replace_mask] = self.mask_id

        # Only the ``replace_mask`` positions are ever read — by the sampler just below
        # and by the MLM cross-entropy in ``loss`` — so the generator projects only
        # those, returning a compact (N, V) instead of (B, T, V).
        gen_logits = self.generator(input_ids=masked_input_ids,
                                    attention_mask=attention_mask,
                                    select_mask=replace_mask)

        disc_input_ids = None
        disc_logits = None
        if run_discriminator:
            # Sample replacements only at the masked positions. Sampling is
            # non-differentiable (ELECTRA does not backprop the discriminator into
            # the generator), so it runs under no_grad.
            disc_input_ids = input_ids.clone()
            with torch.no_grad():
                # slice off padded (fake) vocab ids before sampling
                masked_logits = gen_logits[:, :self.true_vocab_size]
                if masked_logits.numel() > 0:
                    probs = F.softmax(masked_logits / self.gen_temperature, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    disc_input_ids[replace_mask] = sampled

            disc_logits, _ = self.discriminator(disc_input_ids, attention_mask=attention_mask)

        # aggregate the ST-MoE router z-loss over the modules that actually ran
        modules = [self.generator]
        if run_discriminator:
            modules.append(self.discriminator)
        router_z_loss = _collect_router_z_loss(modules)
        load_balance_loss = _collect_load_balance_loss(modules)

        return {
            "gen_logits": gen_logits,
            "original_token_ids": original_token_ids,
            "disc_logits": disc_logits,
            "replace_mask": replace_mask,
            "disc_input_ids": disc_input_ids,
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
        }

    
    def loss(
        self,
        outputs: dict,
        return_components: bool = False,
    ):
        """Combined ELECTRA loss ``gen_loss + disc_lambda * disc_loss
        + router_z_loss_coef * router_z_loss + load_balance_loss_coef * load_balance_loss``.

        With ``return_components=True`` also returns the raw (unweighted) terms so
        they can be logged: ``total, {"generator_loss": ..., "discriminator_loss": ...,
        "router_z_loss": ..., "load_balance_loss": ...}``.
        """
        gen_logits = outputs['gen_logits']
        original_token_ids = outputs['original_token_ids']
        replace_mask = outputs['replace_mask']
        disc_input_ids = outputs['disc_input_ids']  # discriminator input
        disc_logits = outputs['disc_logits']

        # gen_logits is already the compact (N, V) gathered at the masked positions
        # (see the generator call in forward), so the targets are gathered to match.
        # Guard against a batch where no token was selected (small batch / low
        # mask_ratio): cross_entropy over an empty tensor would be NaN. ``numel()``
        # reads the shape, so the check needs no device-to-host sync.
        if gen_logits.numel() > 0:
            gen_loss = F.cross_entropy(
                gen_logits, original_token_ids[replace_mask]
            )
        else:
            gen_loss = gen_logits.sum() * 0.0

        # Generator-warmup phase: the discriminator was not run, so the loss is the
        # MLM objective alone.
        if disc_logits is None:
            disc_loss = gen_logits.new_zeros(())
        else:
            if disc_logits.dim() == 3:
                disc_logits = disc_logits.squeeze(-1)
            # Discriminator target: was the token actually changed from the
            # original? (A masked position whose generator sample happens to equal
            #  the original is correctly labelled "not replaced".)
            actual_replace_mask = (disc_input_ids != original_token_ids).float()
            valid_mask = (original_token_ids != self.pad_token_id)
            # Masked mean instead of a boolean gather (no device sync, no gathered
            # temporaries). The clamp keeps the pad positions finite so their zero
            # weight is well defined, and clamp_min on the count covers an all-pad
            # batch.
            w = valid_mask.float()
            per_token = F.binary_cross_entropy_with_logits(
                disc_logits.float().clamp(-20.0, 20.0),
                actual_replace_mask,
                reduction="none",
            )
            disc_loss = (per_token * w).sum() / w.sum().clamp_min(1.0)

        total = gen_loss + self.disc_lambda * disc_loss

        # ST-MoE router z-loss (summed over the UGM routers that ran this forward)
        router_z_loss = outputs.get("router_z_loss")
        if router_z_loss is None:
            router_z_loss = gen_logits.new_zeros(())
        if self.router_z_loss_coef:
            total = total + self.router_z_loss_coef * router_z_loss

        # Load-balance auxiliary loss
        load_balance_loss = outputs.get("load_balance_loss")
        if load_balance_loss is None:
            load_balance_loss = gen_logits.new_zeros(())
        if self.load_balance_loss_coef:
            total = total + self.load_balance_loss_coef * load_balance_loss

        if return_components:
            return total, {
                "generator_loss": gen_loss.detach(),
                "discriminator_loss": disc_loss.detach(),
                "router_z_loss": router_z_loss.detach(),
                "load_balance_loss": load_balance_loss.detach(),
            }
        return total

    def eval_forward(self, batch, outputs=None, attention_mask: torch.Tensor | None = None,):
        outputs = self.forward(batch, attention_mask=attention_mask)
        return outputs

    def get_metrics(self, is_train=False):
        return {} if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, DiscriminatorAccuracy):
            metric.update(outputs, batch)
        elif isinstance(metric, TokenAccuracy):
            metric.update(outputs, batch)


class NexteraBERTForBertTrainer(nn.Module):
    """Standard BERT-style MLM pretraining (no discriminator).

    Uses 80/10/10 masking: 80% [MASK], 10% random token, 10% unchanged.
    The interface mirrors ``NexteraBERTForElectraTrainer`` so the training
    loop in ``pretrain.py`` works with either mode.
    """

    def __init__(
        self,
        config,
        mask_id: int | None = None,
        pad_token_id: int | None = None,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        mask_ratio: float = 0.15,
        router_z_loss_coef: float = 1e-3,
        load_balance_loss_coef: float = 1e-2,
        true_vocab_size: int | None = None,
    ):
        # special-token ids default to the config's (ModernBERT tokenizer)
        mask_id = config.mask_token_id if mask_id is None else mask_id
        pad_token_id = config.pad_token_id if pad_token_id is None else pad_token_id
        cls_token_id = config.cls_token_id if cls_token_id is None else cls_token_id
        sep_token_id = config.sep_token_id if sep_token_id is None else sep_token_id
        super().__init__()
        self.model = NexteraBERTForMaskedLM(config)
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.pad_token_id = pad_token_id
        self.special_token_ids = {pad_token_id, cls_token_id, sep_token_id}
        self.vocab_size = config.vocab_size
        # config.vocab_size may be padded past the real tokenizer vocab (multiple
        # of 64 for GPU tile alignment); random replacement must never emit the
        # padded ids — the tokenizer can never produce them.
        self.true_vocab_size = true_vocab_size or config.vocab_size

        # The pooler / segment embeddings cannot receive gradient from MLM — see
        # freeze_pretraining_unused. BERT mode runs DDP with
        # find_unused_parameters=False, so leaving them trainable aborts training
        # on the second iteration.
        freeze_pretraining_unused(self.model.encoder)

        self.eval_metrics = {
            "mlm_accuracy": TokenAccuracy(pad_token_id),
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        run_discriminator: bool = True,
    ) -> dict:
        rand = torch.rand_like(input_ids, dtype=torch.float)
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            valid_mask &= (input_ids != token_id)
        replace_mask = (rand < self.mask_ratio) & valid_mask

        original_token_ids = input_ids.clone()

        # BERT 80/10/10 masking, fully vectorised over the (B, T) grid: one uniform
        # per position selected with where/masked_fill, so no device-to-host sync.
        choice = torch.rand_like(rand)
        to_mask = replace_mask & (choice < 0.8)                     # 80% -> [MASK]
        to_random = replace_mask & (choice >= 0.8) & (choice < 0.9)  # 10% -> random
        # remaining 10% -> unchanged
        random_ids = torch.randint_like(input_ids, 0, self.true_vocab_size)
        masked_input_ids = torch.where(to_random, random_ids, input_ids)
        masked_input_ids = masked_input_ids.masked_fill(to_mask, self.mask_id)

        # Only the masked positions' logits are consumed (by the loss and the
        # accuracy metric), so project just those: (N, V) instead of (B, T, V).
        gen_logits = self.model(input_ids=masked_input_ids,
                                attention_mask=attention_mask,
                                select_mask=replace_mask)

        router_z_loss = _collect_router_z_loss([self.model])
        load_balance_loss = _collect_load_balance_loss([self.model])

        return {
            "gen_logits": gen_logits,
            "original_token_ids": original_token_ids,
            "disc_logits": None,
            "replace_mask": replace_mask,
            "disc_input_ids": None,
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
        }

    def loss(self, outputs, return_components=False):
        gen_logits = outputs["gen_logits"]
        original_token_ids = outputs["original_token_ids"]
        replace_mask = outputs["replace_mask"]

        # gen_logits is the compact (N, V) already gathered at the masked positions;
        # the shape test replaces an ``.any()`` device sync.
        if gen_logits.numel() > 0:
            gen_loss = F.cross_entropy(
                gen_logits, original_token_ids[replace_mask]
            )
        else:
            gen_loss = gen_logits.sum() * 0.0

        total = gen_loss

        router_z_loss = outputs.get("router_z_loss")
        if router_z_loss is None:
            router_z_loss = gen_logits.new_zeros(())
        if self.router_z_loss_coef:
            total = total + self.router_z_loss_coef * router_z_loss

        load_balance_loss = outputs.get("load_balance_loss")
        if load_balance_loss is None:
            load_balance_loss = gen_logits.new_zeros(())
        if self.load_balance_loss_coef:
            total = total + self.load_balance_loss_coef * load_balance_loss

        if return_components:
            return total, {
                "generator_loss": gen_loss.detach(),
                "discriminator_loss": gen_logits.new_zeros(()),
                "router_z_loss": router_z_loss.detach(),
                "load_balance_loss": load_balance_loss.detach(),
            }
        return total

    def eval_forward(self, batch, outputs=None, attention_mask=None):
        return self.forward(batch, attention_mask=attention_mask)

    def get_metrics(self, is_train=False):
        return {} if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, TokenAccuracy):
            metric.update(outputs, batch)


def matryoshka_info_nce(z1, z2, dims, temperature: float = 1.0):
    """Matryoshka sequence-contrastive loss over nested embedding prefixes.

    ``z1`` / ``z2`` are the two views' projected representations ``(B, P)``. For
    every nested dimension ``d`` in ``dims`` the views are truncated to their first
    ``d`` channels, L2-normalised and scored with a *symmetric* in-batch InfoNCE
    (NT-Xent): each row's positive is its paired view, negatives are the other
    sequences in the batch. The per-dimension losses are **averaged**, so a single
    embedding is trained to stay discriminative at every prefix width — the same
    recipe as sentence-transformers' ``MatryoshkaLoss`` wrapping
    ``MultipleNegativesRankingLoss`` (used by the reference Stable-Static-Embedding
    ``train.py``), applied here to COCO-LM's Sequence Contrastive Learning task.

    ``temperature`` follows the COCO-LM paper default of 1.0 (the reference
    MNRL uses ~0.05); the caller exposes it as a hyper-parameter.
    """
    B = z1.size(0)
    labels = torch.arange(B, device=z1.device)
    total = z1.new_zeros(())
    n = 0
    for d in dims:
        a = F.normalize(z1[:, :d].float(), dim=-1)
        b = F.normalize(z2[:, :d].float(), dim=-1)
        logits = (a @ b.t()) / temperature
        total = total + 0.5 * (F.cross_entropy(logits, labels)
                               + F.cross_entropy(logits.t(), labels))
        n += 1
    return total / max(n, 1)


def random_span_crop(input_ids, attention_mask, crop_ratio, cls_id, pad_id):
    """Random contiguous crop of each sequence, keeping ``[CLS]`` at position 0.

    Builds COCO-LM's second SCL view: for every row the real (non-pad) body is
    cropped to a random contiguous span of ``crop_ratio`` of its length; ``[CLS]``
    is kept and the result is re-padded to the original width. Returns
    ``(crop_ids, crop_mask)`` with the same shape/dtype as ``input_ids``. Fully
    vectorised (a per-row random start + a single gather), so it adds negligible
    cost next to the encoder forward.
    """
    B, T = input_ids.shape
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    real_len = attention_mask.long().sum(dim=1)                 # (B,) includes [CLS]
    body_len = (real_len - 1).clamp(min=1)                      # excludes [CLS]
    keep = (body_len.float() * crop_ratio).round().long().clamp(min=1)
    keep = torch.minimum(keep, body_len)
    max_start = (body_len - keep).clamp(min=0)                  # (B,)
    r = torch.rand(B, device=device)
    start = (r * (max_start + 1).float()).long()
    start = torch.minimum(start, max_start)                     # in [0, max_start]

    col = torch.arange(T - 1, device=device)                    # output body columns
    src = 1 + start.unsqueeze(1) + col.unsqueeze(0)            # (B, T-1) source indices
    valid = (col.unsqueeze(0) < keep.unsqueeze(1)) & (src < real_len.unsqueeze(1))
    src = src.clamp(max=T - 1)
    gathered = torch.gather(input_ids, 1, src)                 # (B, T-1)

    crop_ids = torch.full((B, T), pad_id, dtype=input_ids.dtype, device=device)
    crop_ids[:, 0] = cls_id
    crop_ids[:, 1:] = torch.where(valid, gathered, torch.full_like(gathered, pad_id))
    crop_mask = torch.zeros((B, T), dtype=attention_mask.dtype, device=device)
    crop_mask[:, 0] = 1
    crop_mask[:, 1:] = valid.to(attention_mask.dtype)
    return crop_ids, crop_mask


class NexteraBERTForCocoLM(nn.Module):
    """COCO-LM main model — the backbone you keep.

    A single :class:`NexteraBERT` encoder shared by three heads:

      * ``lm_head``   — corrective all-token MLM (predict the *original* token at
                        every position); tied to the input embedding when
                        ``config.tie_word_embeddings`` is set.
      * ``copy_head`` — binary copy / replaced-token-detection gate (was this
                        token replaced by the auxiliary generator?).
      * ``scl_proj``  — projects the mean-pooled representation for Sequence
                        Contrastive Learning (Matryoshka InfoNCE).

    See :class:`NexteraBERTForCocoLMTrainer` for how the heads are combined.
    """

    def __init__(self, config, scl_proj_dim: int = 512):
        super().__init__()
        self.encoder = NexteraBERT(config)
        self.head = NexteraBERTPredictionHead(config)
        # Decoder bias (ModernBERT ``decoder_bias=True``) stays untied — see
        # NexteraBERTForMaskedLM.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        nn.init.zeros_(self.lm_head.bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.encoder.embedding.weight
        # ELECTRA-style transform for the binary copy/RTD gate (dense -> GELU ->
        # linear) — see NexteraBERTForDiscriminator.
        self.copy_dense = nn.Linear(config.n_embd, config.n_embd)
        self.copy_head = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.copy_dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.copy_dense.bias)
        self.scl_proj = nn.Sequential(
            nn.Linear(config.n_embd, scl_proj_dim),
            SeparableDyT(scl_proj_dim)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden, _ = self.encoder(input_ids, cross_ids, attention_mask)   # (B,T,C), out_norm'd
        lm_logits = self.lm_head(self.head(hidden))                      # (B,T,V)
        copy_logits = self.copy_head(
            F.gelu(self.copy_dense(hidden))).squeeze(-1)                 # (B,T)
        return hidden, lm_logits, copy_logits

    def project(self, hidden: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Project the (already normalised) mean-pooled hidden state for SCL."""
        return norm(self.scl_proj(masked_mean_pool(hidden, attention_mask)))

    def encode_cls(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SCL projection only — skips the (expensive) vocab / copy heads."""
        hidden, _ = self.encoder(input_ids, cross_ids, attention_mask)
        return norm(self.scl_proj(masked_mean_pool(hidden, attention_mask)))


class NexteraBERTForCocoLMTrainer(nn.Module):
    """COCO-LM pretraining (Meng et al., arXiv:2102.08473).

    Extends the ELECTRA setup with two changes to the main model's objective:

      1. **Corrective Language Modeling (CLM)** replaces ELECTRA's plain binary
         RTD. The main model both *detects* replaced tokens (``copy_head``, a
         binary gate) and *corrects* them via an **All-Token MLM with a copy
         mechanism**: the predicted probability of the original token mixes a
         copy of the input token (weighted by the stop-gradient copy gate) with
         the LM head's softmax, and the cross-entropy is taken over *all* tokens.
      2. **Sequence Contrastive Learning (SCL)** aligns two views of the same
         sequence — the corrupted sequence ``X^MLM`` and a random contiguous crop
         of the original — via a **Matryoshka InfoNCE** on the projection
         of the masked-mean-pooled hidden state (see :func:`matryoshka_info_nce`).

    The auxiliary generator is the same MLM model ELECTRA uses. The interface
    (``forward`` / ``loss`` / ``eval_forward`` / ``get_metrics`` /
    ``update_metric``) mirrors :class:`NexteraBERTForElectraTrainer` so the
    ``scripts/pretrain.py`` training loop drives it unchanged.
    """

    def __init__(
        self,
        generator_config,
        main_config,
        mask_id: int | None = None,
        pad_token_id: int | None = None,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        mask_ratio: float = 0.15,
        copy_lambda: float = 50.0,
        scl_lambda: float = 1.0,
        scl_temperature: float = 1.0,
        scl_proj_dim: int = 512,
        scl_crop_ratio: float = 0.9,
        matryoshka_dims=(512, 256, 128, 64, 32),
        gen_temperature: float = 1.0,
        router_z_loss_coef: float = 1e-3,
        load_balance_loss_coef: float = 1e-2,
        tie_embeddings: bool = True,
        true_vocab_size: int | None = None,
    ):
        # special-token ids default to the config's (ModernBERT tokenizer)
        mask_id = main_config.mask_token_id if mask_id is None else mask_id
        pad_token_id = main_config.pad_token_id if pad_token_id is None else pad_token_id
        cls_token_id = main_config.cls_token_id if cls_token_id is None else cls_token_id
        sep_token_id = main_config.sep_token_id if sep_token_id is None else sep_token_id
        super().__init__()
        self.generator = NexteraBERTForMaskedLM(generator_config)
        self.model = NexteraBERTForCocoLM(main_config, scl_proj_dim=scl_proj_dim)
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio
        self.copy_lambda = copy_lambda
        self.scl_lambda = scl_lambda
        self.scl_temperature = scl_temperature
        self.scl_proj_dim = scl_proj_dim
        self.scl_crop_ratio = scl_crop_ratio
        # keep only nested dims that fit the projection width (largest first)
        dims = sorted({int(d) for d in matryoshka_dims if 0 < int(d) <= scl_proj_dim},
                      reverse=True)
        self.matryoshka_dims = dims or [scl_proj_dim]
        self.gen_temperature = gen_temperature
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.special_token_ids = {pad_token_id, cls_token_id, sep_token_id}
        self.vocab_size = generator_config.vocab_size
        # config.vocab_size may be padded past the real tokenizer vocab (multiple
        # of 64 for GPU tile alignment); sampling must never emit the padded ids.
        self.true_vocab_size = true_vocab_size or generator_config.vocab_size

        # Share token embeddings between generator and main model (ELECTRA style);
        # when n_embd differs a projection layer is added to the generator input.
        if tie_embeddings:
            gen_n_embd = generator_config.n_embd
            main_n_embd = main_config.n_embd
            main_emb = self.model.encoder.embedding
            self.generator.encoder.embedding = main_emb
            if gen_n_embd == main_n_embd:
                if hasattr(self.generator, "lm_head"):
                    self.generator.lm_head.weight = main_emb.weight
            else:
                self.generator.encoder._emb_proj = nn.Linear(
                    main_n_embd, gen_n_embd, bias=False)

        # COCO-LM reads token-level states and the SCL projection, never the
        # pooler — see freeze_pretraining_unused.
        freeze_pretraining_unused(self.generator.encoder, self.model.encoder)

        self.eval_metrics = {
            "copy_accuracy": DiscriminatorAccuracy(pad_token_id),
            "mlm_accuracy": TokenAccuracy(pad_token_id),            # main model corrective LM
            "gen_mlm_accuracy": TokenAccuracy(pad_token_id,         # auxiliary generator MLM
                                              logits_key="gen_mlm_logits"),
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        run_discriminator: bool = True,
    ) -> dict:
        """COCO-LM forward.

        ``run_discriminator=False`` runs only the generator MLM path (no
        sampling, no main-model forward) — used by the generator-warmup phase, so
        the main model / SCL are neither run nor updated until the generator is
        competent (mirrors the ELECTRA trainer).
        """
        rand = torch.rand_like(input_ids, dtype=torch.float)
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            valid_mask &= (input_ids != token_id)
        replace_mask = (rand < self.mask_ratio) & valid_mask

        original_token_ids = input_ids.clone()
        masked_input_ids = input_ids.clone()
        masked_input_ids[replace_mask] = self.mask_id

        # Auxiliary generator: only the masked positions' logits are read (sampling +
        # the L_gen cross-entropy), so project just those -> (N, V). The MAIN model's
        # lm_logits below stay (B, T, V): COCO-LM's corrective LM is an all-token
        # objective and genuinely needs every position.
        gen_logits = self.generator(input_ids=masked_input_ids,
                                    attention_mask=attention_mask,
                                    select_mask=replace_mask)

        if not run_discriminator:
            return {
                "gen_logits": gen_logits,
                "original_token_ids": original_token_ids,
                "corrupted_ids": None,
                "replace_mask": replace_mask,
                "lm_logits": None,
                "copy_logits": None,
                "z1": None,
                "z2": None,
                "router_z_loss": _collect_router_z_loss([self.generator]),
                "load_balance_loss": _collect_load_balance_loss([self.generator]),
            }

        # Build the corrupted sequence X^MLM by sampling the generator at the
        # masked positions (non-differentiable — no backprop into the generator).
        corrupted_ids = input_ids.clone()
        with torch.no_grad():
            # slice off padded (fake) vocab ids before sampling
            masked_logits = gen_logits[:, :self.true_vocab_size]
            if masked_logits.numel() > 0:
                probs = F.softmax(masked_logits / self.gen_temperature, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                corrupted_ids[replace_mask] = sampled

        # Main model on the corrupted view: CLM heads + SCL view 1.
        hidden_c, lm_logits, copy_logits = self.model(
            corrupted_ids, attention_mask=attention_mask)
        z1 = self.model.project(hidden_c, attention_mask=attention_mask)

        # Main model on a random crop of the ORIGINAL sequence: SCL view 2.
        crop_ids, crop_mask = random_span_crop(
            input_ids, attention_mask, self.scl_crop_ratio,
            self.cls_token_id, self.pad_token_id)
        z2 = self.model.encode_cls(crop_ids, attention_mask=crop_mask)

        return {
            "gen_logits": gen_logits,
            "original_token_ids": original_token_ids,
            "corrupted_ids": corrupted_ids,
            "replace_mask": replace_mask,
            "lm_logits": lm_logits,
            "copy_logits": copy_logits,
            "z1": z1,
            "z2": z2,
            "router_z_loss": _collect_router_z_loss([self.generator, self.model]),
            "load_balance_loss": _collect_load_balance_loss([self.generator, self.model]),
        }

    def loss(self, outputs, return_components: bool = False):
        """COCO-LM loss ``L_gen + copy_lambda*L_copy + L_corr + scl_lambda*L_scl``.

        ``L_gen``  auxiliary generator MLM (masked positions).
        ``L_copy`` binary copy/RTD detection over all non-pad tokens.
        ``L_corr`` copy-mechanism all-token corrective MLM (predict the original).
        ``L_scl``  Matryoshka sequence-contrastive loss on the mean-pooled projections.
        """
        gen_logits = outputs["gen_logits"]
        original_token_ids = outputs["original_token_ids"]
        replace_mask = outputs["replace_mask"]
        zero = gen_logits.new_zeros(())

        # --- auxiliary generator MLM ------------------------------------------
        # gen_logits is the compact (N, V) gathered at the masked positions.
        if gen_logits.numel() > 0:
            gen_loss = F.cross_entropy(
                gen_logits, original_token_ids[replace_mask]
            )
        else:
            gen_loss = gen_logits.sum() * 0.0

        lm_logits = outputs["lm_logits"]
        copy_logits = outputs["copy_logits"]
        corrupted_ids = outputs["corrupted_ids"]
        z1, z2 = outputs["z1"], outputs["z2"]

        if lm_logits is None:
            # generator-warmup phase: only the MLM objective is active
            copy_loss = corr_loss = scl_loss = zero
        else:
            nonpad = original_token_ids != self.pad_token_id
            replaced = corrupted_ids != original_token_ids

            # binary copy / replaced-token detection over all non-pad tokens
            cl = copy_logits.float().clamp(-20.0, 20.0)
            if nonpad.any():
                copy_loss = F.binary_cross_entropy_with_logits(
                    cl[nonpad], replaced[nonpad].float())
            else:
                copy_loss = zero

            # copy-mechanism All-Token MLM: probability of the original token is a
            # mixture of copying the input (weight = stop-grad copy gate) and the
            # LM head's softmax. The gate is detached so the copy head is trained
            # only by L_copy (matches the paper's stop-gradient on p_copy).
            gate = torch.sigmoid(-cl).detach()               # P(not replaced)
            logp = F.log_softmax(lm_logits.float(), dim=-1)
            q = logp.gather(-1, original_token_ids.unsqueeze(-1)).squeeze(-1).exp()
            c = (original_token_ids == corrupted_ids).float()
            p = gate * c + (1.0 - gate) * q
            if nonpad.any():
                corr_loss = -(torch.log(p[nonpad] + 1e-8)).mean()
            else:
                corr_loss = zero

            # Matryoshka sequence contrastive learning (needs B >= 2 for a negative)
            if z1 is not None and z1.size(0) > 1:
                scl_loss = matryoshka_info_nce(
                    z1, z2, self.matryoshka_dims, self.scl_temperature)
            else:
                scl_loss = zero

        total = gen_loss + self.copy_lambda * copy_loss + corr_loss \
            + self.scl_lambda * scl_loss

        router_z_loss = outputs.get("router_z_loss")
        if router_z_loss is None:
            router_z_loss = zero
        if self.router_z_loss_coef:
            total = total + self.router_z_loss_coef * router_z_loss

        load_balance_loss = outputs.get("load_balance_loss")
        if load_balance_loss is None:
            load_balance_loss = zero
        if self.load_balance_loss_coef:
            total = total + self.load_balance_loss_coef * load_balance_loss

        if return_components:
            return total, {
                "generator_loss": gen_loss.detach(),
                "discriminator_loss": copy_loss.detach(),  # copy/RTD (kept for logs)
                "correction_loss": corr_loss.detach(),
                "scl_loss": scl_loss.detach(),
                "router_z_loss": router_z_loss.detach(),
                "load_balance_loss": load_balance_loss.detach(),
            }
        return total

    def eval_forward(self, batch, outputs=None, attention_mask: torch.Tensor | None = None):
        outputs = self.forward(batch, attention_mask=attention_mask, run_discriminator=True)
        # Reshape into the keys the shared metrics read, pointed at the MAIN model:
        #   copy_accuracy <- copy_logits vs (corrupted != original)
        #   mlm_accuracy  <- main lm_logits (corrective) at masked positions
        lm_logits = outputs["lm_logits"]
        return {
            "gen_logits": lm_logits if lm_logits is not None else outputs["gen_logits"],
            "gen_mlm_logits": outputs["gen_logits"],   # the auxiliary generator's MLM logits
            "original_token_ids": outputs["original_token_ids"],
            "replace_mask": outputs["replace_mask"],
            "disc_logits": outputs["copy_logits"],
            "disc_input_ids": outputs["corrupted_ids"],
        }

    def get_metrics(self, is_train=False):
        return {} if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, (DiscriminatorAccuracy, TokenAccuracy)):
            metric.update(outputs, batch)


class NexteraBERTForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        # Small-std head init (plus config.classifier_dropout — 0.0 by default,
        # matching ModernBERT) regularises the tiny GLUE tasks (MRPC, STS-B),
        # where a bare default-initialised classifier over-fits quickly.
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.num_labels = config.num_labels
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        _, pooler_output = self.encoder(input_ids, cross_ids, attention_mask, token_type_ids)
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # squeeze both sides so (B,1)-vs-(B,) label conventions never
                # silently broadcast MSE to (B,B)
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))
            elif self.num_labels > 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits


class NexteraBERTForMultipleChoice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, 1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # (B, num_choices, T) -> (B*num_choices, T): the encoder is strictly 2-D.
        num_choices = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if cross_ids is not None:
            cross_ids = cross_ids.view(-1, cross_ids.size(-1))
        _, pooler_output = self.encoder(input_ids, cross_ids, attention_mask)
        logits = self.classifier(self.dropout(pooler_output))
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        return loss, reshaped_logits


class NexteraBERTForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        # ModernBERT applies the head transform + dropout before the per-token
        # classifier (ModernBertForTokenClassification).
        self.head = NexteraBERTPredictionHead(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.num_labels = config.num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        output, _ = self.encoder(input_ids, cross_ids, attention_mask)
        logits = self.classifier(self.dropout(self.head(output)))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits


class NexteraBERTForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = NexteraBERT(config)
        # ModernBERT applies the head transform + dropout before the span
        # classifier (ModernBertForQuestionAnswering).
        self.head = NexteraBERTPredictionHead(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.num_labels = config.num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        cross_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        start_positions: torch.LongTensor = None,
        end_positions: torch.LongTensor = None
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        output, _ = self.encoder(input_ids, cross_ids, attention_mask)
        logits = self.classifier(self.dropout(self.head(output)))
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits


__all__ = [
    "ScalableFactor",
    "NexteraSelfAttention",
    "NexteraSlidingWindowAttention",
    "NexteraHRA",
    "SnowLily",
    "LFM2Conv",
    "DSConvMixer",
    "MLP",
    "RippleBloomUGM",
    "Block",
    "build_mixer",
    "build_mlp",
    "MLMHead",
    "NexteraBERTPredictionHead",
    "NexteraBERTPooler",
    "NexteraBERTForMaskedLM",
    "NexteraBERTForDiscriminator",
    "NexteraBERTForElectraTrainer",
    "NexteraBERTForBertTrainer",
    "NexteraBERTForCocoLM",
    "NexteraBERTForCocoLMTrainer",
    "NexteraBERTForEmbedding",
    "NexteraBERTForMultipleChoice",
    "NexteraBERTForQuestionAnswering",
    "NexteraBERTForSequenceClassification",
    "NexteraBERTForTokenClassification",
    "DiscriminatorAccuracy",
    "TokenAccuracy",
    "is_ddp",
    "get_dist_info",
    "NexteraBERT",
]