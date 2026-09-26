"""
coding = utf-8
Licensed under "MIT License"

NexteraBERT configuration: a ``transformers.PretrainedConfig`` subclass with named
size presets, serialised to ``config.json`` and loadable from the Hugging Face Hub.
"""

from __future__ import annotations

from typing import List

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover - older alias
    from transformers import PreTrainedConfig as PretrainedConfig


# Token mixers: "lily" (SnowLily), "swa" (sliding-window attention), "attn" (full
# attention) and "hra" (attention over bands of block_size tokens).
#
#   pianissimo, piano       n14:  3 attn / 2 hra / 3 swa /  6 lily
#   mezzopiano, mezzoforte  n18:  3 attn / 2 hra / 5 swa /  8 lily
#   forte, fortissimo       n24:  4 attn / 3 hra / 7 swa / 10 lily
_PRESETS = {
    "pianissimo": dict(
        n_layer=14, n_head=4, n_kv_head=4, n_embd=256, n_inter=1024, topk=5,
        layer_types=["lily", "lily", "swa", "lily", "attn",
                     "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
    "piano": dict(
        n_layer=14, n_head=6, n_kv_head=3, n_embd=384, n_inter=1280, topk=5,
        layer_types=["lily", "lily", "swa", "lily", "attn",
                     "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
    "mezzopiano": dict(
        n_layer=18, n_head=8, n_kv_head=4, n_embd=512, n_inter=1536, topk=7,
        layer_types=["lily", "lily", "swa", "lily", "swa", "lily", "attn",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
    "mezzoforte": dict(
        n_layer=18, n_head=16, n_kv_head=8, n_embd=1024, n_inter=2304, topk=9,
        layer_types=["lily", "lily", "swa", "lily", "swa", "lily", "attn",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
    "forte": dict(
        n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, n_inter=2560, topk=11,
        layer_types=["lily", "lily", "swa", "lily", "swa", "lily", "attn",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
    "fortissimo": dict(
        n_layer=24, n_head=24, n_kv_head=6, n_embd=1536, n_inter=3072, topk=13,
        layer_types=["lily", "lily", "swa", "lily", "swa", "lily", "attn",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "lily", "swa", "attn", "hra",
                     "lily", "swa", "attn", "hra", "lily"],
    ),
}

PRESET_NAMES = tuple(_PRESETS.keys())

# Feed-forward inside a block (``mlp_class``):
#   "mlp"  dense squared-ReLU MLP
#   "ugm"  routed Unified Granularity Module (adds router auxiliary losses)
MLP_CLASSES = ("mlp", "ugm")

# Length-dependent query scaling in the "attn" / "hra" softmax (``ScalableFactor``):
#   "sssmax"  s*log(T) + ssmax_base + relu(b)   Stable Scalable-Softmax (default)
#   "ssmax"   s*log(T)                          Scalable-Softmax (Nakanishi 2025)
#   "none"    plain softmax, no parameters
SSMAX_MODES = ("sssmax", "ssmax", "none")


class NexteraBERTConfig(PretrainedConfig):
    """Configuration for every NexteraBERT head.

    Use :meth:`from_preset` to obtain one of the named sizes, or instantiate
    directly to override individual hyper-parameters.
    """

    model_type = "nexterabert"

    def __init__(
        self,
        vocab_size: int = 50368,
        n_layer: int = 22,
        n_head: int = 12,
        n_kv_head: int = 4,
        n_embd: int = 768,
        n_inter: int = 128 + 4096,
        n_kernel: int = 5,
        expand: int = 1,
        block_size: int = 4,
        swa_window: int = 256,
        swa_rope_base: int = 10000,
        lora_rank: int = 256,
        topk: int = 4,
        perturbation_element: int = 8,
        rope_base: int = 100000,
        rope_ntk_train_len: int = 0,
        ssmax_mode: str = "sssmax",
        ssmax_s: float = 0.43,
        ssmax_b: float = 0.1,
        ssmax_base: float = 0.1,
        tie_word_embeddings: bool = True,
        num_labels: int = 2,
        classifier_dropout: float = 0.0,
        type_vocab_size: int = 2,
        block_types: List[str] | None = None,
        mlp_class: str = "mlp",
        unpadding: bool = False,
        # special tokens of the answerdotai/ModernBERT-base tokenizer
        pad_token_id: int = 50283,
        cls_token_id: int = 50281,
        sep_token_id: int = 50282,
        mask_token_id: int = 50284,
        **kwargs,
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.n_inter = n_inter
        self.n_kernel = n_kernel
        self.expand = expand
        self.block_size = block_size
        # "swa" band width: query i attends keys j with |i - j| <= swa_window // 2.
        if swa_window < 1:
            raise ValueError(f"swa_window must be >= 1, got {swa_window}")
        self.swa_window = int(swa_window)
        # RoPE base of the "swa" layers, smaller than rope_base to match the band.
        if swa_rope_base < 1:
            raise ValueError(f"swa_rope_base must be >= 1, got {swa_rope_base}")
        self.swa_rope_base = int(swa_rope_base)
        self.lora_rank = lora_rank
        self.topk = topk
        self.perturbation_element = perturbation_element
        self.rope_base = rope_base
        # Dynamic NTK scaling: sequences longer than this trained length rescale the
        # RoPE base; shorter ones are unchanged. 0 disables it.
        self.rope_ntk_train_len = rope_ntk_train_len
        ssmax_mode = str(ssmax_mode).lower()
        if ssmax_mode not in SSMAX_MODES:
            raise ValueError(
                f"Unknown ssmax_mode '{ssmax_mode}'. Choose one of {SSMAX_MODES}.")
        self.ssmax_mode = ssmax_mode
        # Initial s and b (b > 0 so relu(b) has a gradient); ssmax_base is the
        # constant floor that keeps the "sssmax" scale positive.
        self.ssmax_s = float(ssmax_s)
        self.ssmax_b = float(ssmax_b)
        if float(ssmax_base) <= 0:
            raise ValueError(
                f"ssmax_base must be positive (it is the scale's floor), got {ssmax_base}")
        self.ssmax_base = float(ssmax_base)
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.classifier_dropout = classifier_dropout
        self.type_vocab_size = type_vocab_size
        if block_types is None:
            block_types = list(_PRESETS["mezzoforte"]["layer_types"])
        self.block_types = block_types
        mlp_class = str(mlp_class).lower()
        if mlp_class not in MLP_CLASSES:
            raise ValueError(
                f"Unknown mlp_class '{mlp_class}'. Choose one of {MLP_CLASSES}.")
        self.mlp_class = mlp_class
        # Run padded batches unpadded (NexteraBERT.set_unpadding). A runtime switch,
        # not an architecture key: the weights are the same either way.
        self.unpadding = bool(unpadding)
        kwargs.pop("arch_version", None)  # stale key in older config.json files
        # `hidden_size` / `num_hidden_layers` / `num_attention_heads` aliases for HF utilities
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "NexteraBERTConfig":
        name = name.lower()
        if name not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. Choose one of {PRESET_NAMES}."
            )
        params = {k: (list(v) if isinstance(v, list) else v)
                  for k, v in _PRESETS[name].items()}
        params["block_types"] = params.pop("layer_types")
        params.update(overrides)
        return cls(**params)


__all__ = ["NexteraBERTConfig", "PRESET_NAMES", "MLP_CLASSES", "SSMAX_MODES"]
