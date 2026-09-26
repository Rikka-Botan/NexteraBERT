"""Sentence-embedding heads + the InfoNCE loss both published protocols use.

MTEB and BEIR score a single vector per text, but an MLM/RTD-pretrained encoder
emits one vector per token and was never trained to place related texts near each
other. Every published number on either benchmark therefore comes from a model
that first learned to embed, under one of two recipes:

* **MTEB** -- OptiBERT (Dervishi et al., EMNLP 2025, App. D.2): an attentive
  pooling head, fine-tuned end-to-end on MNLI+SNLI triplets with supervised-SimCSE
  InfoNCE for 3 epochs (Gao et al., 2021). Their Table 5 follows this.
* **Retrieval** -- ModernBERT (Warner et al., 2024, 3.1.2): plain mean pooling,
  fine-tuned on 1.25M MS MARCO pairs with mined hard negatives. Their Table 7 BEIR
  scores (BERT-base 38.9, ModernBERT-base 41.6) follow this. sentence-transformers'
  MultipleNegativesRankingLoss is the same InfoNCE at scale 20 = temperature 0.05.

Comparing a raw backbone against either table compares protocols, not models.

`scripts/finetune_contrastive.py` runs both; `nexterabert.mteb_encoder` picks the
result up automatically -- an attentive run leaves ``pooling_head.pt`` beside the
backbone, and either run records what it did in ``contrastive_run.json``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

POOLING_HEAD_FILE = "pooling_head.pt"
#: Marker naming the contrastive protocol a model directory went through.
#: Its absence means the backbone is raw -- which the eval scripts warn about,
#: because no published MTEB/BEIR number is measured that way.
CONTRASTIVE_RUN_FILE = "contrastive_run.json"

# SimCSE's supervised temperature (Gao et al., 2021, Sec. 6.1 / their released
# run_sup_example.sh). The loss is scale-free otherwise, so this is the one knob
# that sets how hard the negatives push.
DEFAULT_TEMPERATURE = 0.05


class AttentivePooling(nn.Module):
    """Additive attention over the token states with a single learned query.

    ``score_t = q^T tanh(W h_t)`` softmaxed over the real tokens, then a weighted
    sum. Unlike mean pooling this can down-weight the tokens that carry no
    sentence meaning — padding is masked out, and [CLS]/[SEP] stop dominating the
    average of a short text purely because they are 2 of its 12 positions.
    """

    def __init__(self, n_embd: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or n_embd
        self.proj = nn.Linear(n_embd, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.query(torch.tanh(self.proj(hidden_states))).squeeze(-1)  # (B, T)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0,
                                        torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).unsqueeze(-1)                         # (B, T, 1)
        return (hidden_states * weights).sum(dim=1)                            # (B, C)


class NexteraBERTForSentenceEmbedding(nn.Module):
    """Encoder + attentive pooling head: text -> one vector.

    The encoder's own ``out_norm`` is already applied to the token states it
    returns, so the head pools exactly the representation the other heads read.
    """

    def __init__(self, config):
        super().__init__()
        from .modeling_nexterabert import NexteraBERT

        self.encoder = NexteraBERT(config)
        self.pooling = AttentivePooling(config.n_embd)

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _ = self.encoder(input_ids, None, attention_mask)
        return self.pooling(hidden, attention_mask)

    def freeze_unused(self) -> list:
        """Freeze the parameters this objective can never reach.

        The sequence-classification pooler and the segment embeddings get no
        gradient here (nothing calls them), exactly as in pretraining — see
        ``freeze_pretraining_unused``.
        """
        from .modeling_nexterabert import freeze_pretraining_unused

        return freeze_pretraining_unused(self.encoder)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                  negative: torch.Tensor | None = None,
                  temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """Supervised-SimCSE InfoNCE over cosine similarities.

    Columns are every positive in the batch followed by every hard negative, so
    anchor ``i`` is pushed towards ``positive[i]`` and away from both the other
    anchors' positives (in-batch negatives) and all the contradiction hypotheses.
    Raising the batch size therefore strengthens the objective itself, not just
    the gradient estimate -- and gradient accumulation does NOT substitute for it,
    since negatives only come from tensors that share a forward pass.

    Similarities are computed in fp32: under bf16 autocast a 1024-d dot product of
    unit vectors loses ~3 decimal digits, which at 1/0.05 = 20x logit scaling is
    enough to flatten the softmax.
    """
    a = F.normalize(anchor.float(), dim=-1)
    columns = F.normalize(positive.float(), dim=-1)
    if negative is not None:
        columns = torch.cat([columns, F.normalize(negative.float(), dim=-1)], dim=0)
    logits = (a @ columns.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def freeze_unused(model) -> list:
    """Freeze the parameters a contrastive objective can never reach.

    The sequence-classification pooler and the segment embeddings get no gradient
    here (nothing calls them), exactly as in pretraining -- see
    ``freeze_pretraining_unused``. Works for either pooling variant.
    """
    from .modeling_nexterabert import freeze_pretraining_unused

    return freeze_pretraining_unused(model.encoder)


def save_contrastive_model(model, output_dir, *, config=None, tokenizer=None,
                           run_summary: dict | None = None) -> str:
    """Write the fine-tuned model out as an ordinary model directory.

    The backbone is saved in the repo's usual format, so every other script can
    still read it. A trained attentive head goes beside it in ``pooling_head.pt``
    (a mean-pooling run has none -- the pooling is the backbone's own), and
    ``run_summary`` is recorded in ``contrastive_run.json`` so the eval scripts can
    report which protocol produced the numbers.
    """
    import json

    from .export import save_pretrained

    out = save_pretrained(model.encoder, output_dir, config=config, tokenizer=tokenizer)
    head = getattr(model, "pooling", None)
    if isinstance(head, nn.Module):
        torch.save(head.state_dict(), Path(out) / POOLING_HEAD_FILE)
    if run_summary is not None:
        with open(Path(out) / CONTRASTIVE_RUN_FILE, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
    return out


def has_pooling_head(path) -> bool:
    """True when ``path`` is a model directory carrying a trained pooling head."""
    return (Path(path) / POOLING_HEAD_FILE).exists()


def contrastive_run(path) -> dict | None:
    """The recorded contrastive stage for a model directory, or None if it is raw."""
    import json

    marker = Path(path) / CONTRASTIVE_RUN_FILE
    if not marker.exists():
        return None
    try:
        with open(marker, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_contrastive_model(pretrained_path: str, pooling: str = "attentive",
                           **config_overrides):
    """Load a model in the pooling variant a protocol trains.

    ``attentive`` builds ``NexteraBERTForSentenceEmbedding`` (encoder + trained
    head); ``mean`` builds ``NexteraBERTForEmbedding``, whose pooling is the
    backbone's own masked mean -- ModernBERT's retrieval protocol tunes the
    encoder itself and adds no parameters.
    """
    if pooling == "mean":
        from .loading import load_for_task

        model, config, info = load_for_task(pretrained_path, "embedding")
        return model, config, {**info, "pooling_head": False}
    return load_for_sentence_embedding(pretrained_path, **config_overrides)


def load_for_sentence_embedding(pretrained_path: str, **config_overrides):
    """Rebuild ``NexteraBERTForSentenceEmbedding`` from a model directory.

    Mirrors ``loading.load_for_task``: the backbone is loaded non-strictly and the
    keys that are legitimately absent from a pretraining checkpoint are filtered
    out of ``info["missing"]``. A missing pooling head is reported in
    ``info["pooling_head"]`` rather than raised -- an untrained head is the
    'before' side of the comparison this module exists for.
    """
    from .loading import load_backbone_state, load_config

    config = load_config(pretrained_path, **config_overrides)
    model = NexteraBERTForSentenceEmbedding(config)
    missing, unexpected = model.encoder.load_state_dict(
        load_backbone_state(pretrained_path), strict=False)

    head_path = Path(pretrained_path) / POOLING_HEAD_FILE
    trained_head = head_path.exists()
    if trained_head:
        model.pooling.load_state_dict(torch.load(head_path, map_location="cpu"))

    info = {
        "missing": [m for m in missing
                    if not m.startswith(("pooler", "token_type_embeddings"))],
        "unexpected": unexpected,
        "pooling_head": trained_head,
    }
    return model, config, info
