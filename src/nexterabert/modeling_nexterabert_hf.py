"""``transformers``-native wrappers so a NexteraBERT repo loads with ``trust_remote_code``.

The classes in ``modeling_nexterabert.py`` are plain ``nn.Module``s — deliberately, so
the training loop owns its own checkpoint format — which means ``AutoModel.from_pretrained``
cannot instantiate them. This module adds thin ``PreTrainedModel`` subclasses over the
same encoder, so an uploaded backbone works with the ordinary 🤗 entry points::

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)

They are wrappers, not a reimplementation: every parameter is the same
:class:`~nexterabert.modeling_nexterabert.NexteraBERT` encoder and the same
prediction head, so the numbers match ``nexterabert.loading.load_masked_lm`` exactly.

Two details make this work against the **existing** export format, so already-uploaded
repos only need a ``config.json`` touch-up rather than a re-upload of the weights:

  * ``model.safetensors`` holds the encoder's ``state_dict`` with *bare* keys
    (``embedding.weight``, ``blocks.0...``), not ``encoder.``-prefixed ones. Declaring
    ``base_model_prefix = "encoder"`` is what lets ``from_pretrained`` recognise that
    and add the prefix while loading.
  * The masked-LM output head lives in a separate ``mlm_head.safetensors`` (it is not
    part of the encoder), which ``from_pretrained`` knows nothing about.
    :class:`NexteraBERTForMaskedLM` therefore loads it itself afterwards — see
    :meth:`NexteraBERTForMaskedLM.from_pretrained`. Without it the prediction
    transform would stay randomly initialised and the model would silently emit noise.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .configuration_nexterabert import NexteraBERTConfig
from .modeling_nexterabert import (
    NexteraBERT,
    NexteraBERTPredictionHead,
)

MLM_HEAD_FILE = "mlm_head.safetensors"

# What a Hub config.json needs so `trust_remote_code=True` can find these classes.
AUTO_MAP = {
    "AutoConfig": "configuration_nexterabert.NexteraBERTConfig",
    "AutoModel": "modeling_nexterabert_hf.NexteraBERTModel",
    "AutoModelForMaskedLM": "modeling_nexterabert_hf.NexteraBERTForMaskedLM",
    "AutoModelForSequenceClassification":
        "modeling_nexterabert_hf.NexteraBERTForSequenceClassification",
}


class NexteraBERTPreTrainedModel(PreTrainedModel):
    config_class = NexteraBERTConfig
    # The exported weights are the bare encoder state_dict; this is what tells
    # from_pretrained to prefix them with "encoder." when filling a derived head.
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    # The task heads the classes below put on top of the encoder.
    _head_attrs = ("classifier", "head", "lm_head")

    def _init_weights(self, module):
        """Initialise the task heads; leave the encoder alone.

        transformers >= 5 builds a ``from_pretrained`` model on the meta device, gives
        every parameter the checkpoint lacks an uninitialised ``torch.empty`` tensor,
        and relies on this method to fill it in. A no-op here left the freshly added
        ``classifier`` of :class:`NexteraBERTForSequenceClassification` holding
        whatever that memory contained (garbage on one load, zeros on the next), and
        fine-tuning from it never got past chance accuracy. The heads get the
        initialisation of their counterparts in ``modeling_nexterabert.py``: normal
        std-0.02 weights, zero biases, unit LayerNorm gains.

        The encoder is skipped on purpose. ``NexteraBERT`` initialises itself in
        ``__init__`` (depth-scaled residual projections, zeroed segment embeddings),
        which a model built from a config must keep, and a checkpoint supplies all of
        its weights. Loaded head weights are not overwritten either: transformers runs
        this under ``guard_torch_init_functions``, so ``nn.init`` skips every tensor
        the checkpoint filled.
        """
        if not self._is_head_module(module):
            return
        if isinstance(module, nn.Linear):
            # A tied decoder weight is the input embedding, which is the encoder's.
            tied = (module is getattr(self, "lm_head", None)
                    and self.config.tie_word_embeddings)
            if not tied:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _is_head_module(self, module) -> bool:
        """True when ``module`` is one of the task heads or lives inside one."""
        for name in self._head_attrs:
            head = getattr(self, name, None)
            if head is not None and any(module is m for m in head.modules()):
                return True
        return False


class NexteraBERTModel(NexteraBERTPreTrainedModel):
    """The bare encoder: ``last_hidden_state`` + the pooled ``[CLS]``-style vector."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = NexteraBERT(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embedding

    def set_input_embeddings(self, value):
        self.encoder.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        hidden, pooled = self.encoder(input_ids, None, attention_mask, token_type_ids)
        if return_dict is False:
            return (hidden, pooled)
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=pooled)


class NexteraBERTForMaskedLM(NexteraBERTPreTrainedModel):
    """Masked-LM head over the encoder.

    ``from_pretrained`` is extended to also restore ``mlm_head.safetensors``, which
    lives outside ``model.safetensors`` and is therefore invisible to the base
    implementation.
    """

    _tied_weights_keys = {"lm_head.weight": "encoder.embedding.weight"}
    # model.safetensors is the encoder alone; the head arrives from
    # mlm_head.safetensors in from_pretrained below, so HF's load report should not
    # advertise it as missing.
    _keys_to_ignore_on_load_missing = [r"^head\.", r"^lm_head\.bias$"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = NexteraBERT(config)
        self.head = NexteraBERTPredictionHead(config)
        # Decoder bias stays untied even when the weight is tied: it absorbs the
        # per-token frequency prior so the embedding matrix does not have to.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        nn.init.zeros_(self.lm_head.bias)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embedding

    def set_input_embeddings(self, value):
        self.encoder.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        state = _fetch_mlm_head(pretrained_model_name_or_path,
                                token=kwargs.get("token"),
                                revision=kwargs.get("revision"))
        if state:
            # lm_head.weight is tied to the embedding the encoder already restored;
            # loading a copy would break the tie.
            state = {k: v for k, v in state.items() if k != "lm_head.weight"} \
                if model.config.tie_word_embeddings else state
            model.load_state_dict(state, strict=False)
        else:
            print(f"[NexteraBERT] WARNING: no {MLM_HEAD_FILE} in "
                  f"{pretrained_model_name_or_path} -- the masked-LM output head "
                  f"(prediction transform + decoder bias) stays randomly initialised "
                  f"and predictions will be meaningless. Add it with "
                  f"scripts/upload_mlm_head.py.")
        return model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        hidden, _ = self.encoder(input_ids, None, attention_mask, token_type_ids)
        logits = self.lm_head(self.head(hidden))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), labels.view(-1))
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return MaskedLMOutput(loss=loss, logits=logits)


class NexteraBERTForSequenceClassification(NexteraBERTPreTrainedModel):
    """Pooled-representation classifier — the fine-tuning entry point."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        self.encoder = NexteraBERT(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.0) or 0.0)
        self.classifier = nn.Linear(config.n_embd, self.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embedding

    def set_input_embeddings(self, value):
        self.encoder.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        _, pooled = self.encoder(input_ids, None, attention_mask, token_type_ids)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(logits.squeeze(-1).float(), labels.float())
            else:
                loss = nn.functional.cross_entropy(logits.float(), labels.view(-1))
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _fetch_mlm_head(path, token=None, revision=None):
    """Read ``mlm_head.safetensors`` from a local directory or a Hub repo id."""
    if os.path.isdir(path):
        local = os.path.join(path, MLM_HEAD_FILE)
        if not os.path.exists(local):
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(str(path), MLM_HEAD_FILE,
                                    token=token, revision=revision)
        except Exception:  # noqa: BLE001 - absent file, offline, or no access
            return None

    from safetensors.torch import load_file

    return load_file(local, device="cpu")


__all__ = [
    "AUTO_MAP",
    "NexteraBERTPreTrainedModel",
    "NexteraBERTModel",
    "NexteraBERTForMaskedLM",
    "NexteraBERTForSequenceClassification",
]
