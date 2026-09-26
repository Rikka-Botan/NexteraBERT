"""Inference tokenizer for NexteraBERT — unpadded (dynamically padded) batches.

At training time ``scripts/prepare_data.py --pack_mode docs`` stores one document per
example with NO baked padding, and the dynamic-padding collate
(:func:`nexterabert.data.make_dynamic_pad_collate`) pads each *batch* to its longest
example rounded up to a multiple of NexteraHRA's block granularity. This class does
the SAME thing at inference: given one string or a list of strings it returns
``input_ids`` / ``attention_mask`` padded only to the batch's longest sequence
(rounded to ``pad_multiple_of``), never to a fixed maximum length — so inference is
"unpadded" and the block-pooling NexteraHRA layers see exactly one document per
sequence, just like during pretraining.
"""

from __future__ import annotations


class NexteraBERTTokenizer:
    """Thin wrapper over the ModernBERT BPE tokenizer that emits unpadded batches.

    Parameters
    ----------
    tokenizer_name : str
        HF tokenizer id (default the ModernBERT BPE tokenizer NexteraBERT trains with).
    tokenizer : optional
        A pre-loaded ``transformers`` tokenizer to wrap instead of loading one.
    pad_multiple_of : int
        Pad each batch's length up to a multiple of this (>= the model's HRA
        ``block_size``, and ideally a multiple of it). Default 8, matching
        ``pretrain.py --pad_multiple_of``.
    max_length : int | None
        Optional hard cap; longer inputs are truncated. ``None`` = no truncation.

    Examples
    --------
    >>> tok = NexteraBERTTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    >>> batch = tok(["hello world", "a longer example sentence here"])
    >>> out = model(**batch)                     # input_ids / attention_mask
    """

    def __init__(self, tokenizer_name: str = "answerdotai/ModernBERT-base",
                 tokenizer=None, pad_multiple_of: int = 8, max_length: int | None = None):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = tokenizer
        self.pad_multiple_of = pad_multiple_of
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.mask_token_id = tokenizer.mask_token_id

    @classmethod
    def from_pretrained(cls, tokenizer_name: str = "answerdotai/ModernBERT-base", **kwargs):
        return cls(tokenizer_name=tokenizer_name, **kwargs)

    def __call__(self, texts, return_tensors: str = "pt", truncation: bool | None = None,
                 max_length: int | None = None):
        """Tokenise ``texts`` (str or list[str]) into an unpadded batch.

        Adds ``[CLS] … [SEP]`` and pads with ``padding="longest"``
        + ``pad_to_multiple_of`` — i.e. only to the batch's longest example, rounded to
        ``pad_multiple_of``. No fixed-length padding, so it is genuinely unpadded.
        """
        if isinstance(texts, str):
            texts = [texts]
        ml = max_length if max_length is not None else self.max_length
        do_trunc = truncation if truncation is not None else (ml is not None)
        enc = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=do_trunc,
            max_length=ml,
            padding="longest",
            pad_to_multiple_of=self.pad_multiple_of,
            return_tensors=return_tensors,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    # convenience pass-throughs
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __len__(self):
        return len(self.tokenizer)


__all__ = ["NexteraBERTTokenizer"]
