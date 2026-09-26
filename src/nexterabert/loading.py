"""Load a pretrained NexteraBERT backbone into downstream heads.

A pretrained model produced by ``scripts/pretrain.py`` is exported as a directory
containing ``config.json`` + ``pytorch_model.bin`` (the discriminator *encoder*
weights) + the tokenizer.  These helpers reconstruct any head and transfer the
encoder weights.
"""

from __future__ import annotations

import os

import torch

from .configuration_nexterabert import NexteraBERTConfig
from .modeling_nexterabert import (
    NexteraBERTForEmbedding,
    NexteraBERTForMaskedLM,
    NexteraBERTForMultipleChoice,
    NexteraBERTForQuestionAnswering,
    NexteraBERTForSequenceClassification,
    NexteraBERTForTokenClassification,
)

MLM_HEAD_FILE = "mlm_head.safetensors"

_HEADS = {
    "embedding": NexteraBERTForEmbedding,
    "sequence-classification": NexteraBERTForSequenceClassification,
    "token-classification": NexteraBERTForTokenClassification,
    "multiple-choice": NexteraBERTForMultipleChoice,
    "question-answering": NexteraBERTForQuestionAnswering,
}


def load_config(path: str, **overrides) -> NexteraBERTConfig:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        config = NexteraBERTConfig.from_pretrained(path)
        for k, v in overrides.items():
            setattr(config, k, v)
        return config
    return NexteraBERTConfig(**overrides)


def load_backbone_state(path: str) -> dict:
    """Return the encoder ``state_dict`` from a pretrained directory or .bin/.pt.

    Inside a directory a ``model.safetensors`` file is preferred when present,
    falling back to ``pytorch_model.bin``. A ``.safetensors`` file path is also
    accepted directly.
    """
    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            return load_file(safetensors_path, device="cpu")
        bin_path = os.path.join(path, "pytorch_model.bin")
    elif path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    else:
        bin_path = path
    state = torch.load(bin_path, map_location="cpu")
    # A full training checkpoint -> pull out the reusable encoder. The prefix names
    # the trainer's submodel: ``discriminator.`` for ELECTRA (the backbone you keep),
    # ``model.`` for the BERT and COCO-LM trainers. Without the latter, a BERT
    # checkpoint fell through and this returned the whole payload
    # ({"model", "optimizer", "step", "config"}), which downstream code then tried to
    # treat as a state_dict.
    if "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
        for prefix in ("discriminator.encoder.", "model.encoder.", "encoder."):
            enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if enc:
                return enc
    return state


def mlm_head_from_trainer_state(sd: dict, submodel: str = "auto") -> dict | None:
    """Pull the MLM output head out of a full trainer ``state_dict``.

    ``training_utils.save_checkpoint`` stores the *trainer's* state dict, so the head
    sits under a submodel prefix: ``model.`` for the BERT and COCO-LM trainers,
    ``generator.`` for the ELECTRA trainer (whose discriminator has no MLM head at
    all — its objective is replaced-token detection). Returns a dict keyed for
    :class:`NexteraBERTForMaskedLM` (``head.*`` / ``lm_head.*``), or ``None``.
    """
    prefixes = {
        "auto": ("model.", "generator.", ""),
        "model": ("model.",),
        "generator": ("generator.",),
        "none": ("",),
    }[submodel]
    for prefix in prefixes:
        keys = [k for k in sd
                if k.startswith(prefix + "head.") or k.startswith(prefix + "lm_head.")]
        # The transform is what identifies a real MLM head; a tied lm_head.weight on
        # its own (an ELECTRA discriminator) is not one.
        if any(k.startswith(prefix + "head.") for k in keys):
            return {k[len(prefix):]: sd[k] for k in keys}
    return None


def _submodel_encoder_matches(sd: dict, prefix: str, reference: dict) -> bool:
    """True when ``sd[prefix + 'encoder.*']`` holds the same weights as ``reference``.

    A checkpoint can contain more than one encoder — an ELECTRA run has both the
    discriminator's and the (smaller) generator's — and only the generator carries an
    MLM head. Taking that head for a *discriminator* backbone would pair two
    different models: no error, just a wrong model. So a head is only accepted when
    the encoder it was trained with is the one in the directory.
    """
    enc_prefix = prefix + "encoder."
    keys = [k for k in sd if k.startswith(enc_prefix)]
    if not keys:
        return False
    shared = sorted(k for k in keys if k[len(enc_prefix):] in reference)
    if not shared:
        return False
    # A few tensors spread through the stack settle it: two runs that agree here
    # agree everywhere, and two that differ, differ here.
    probes = {shared[0], shared[len(shared) // 2], shared[-1]}
    for key in probes:
        a, b = sd[key], reference[key[len(enc_prefix):]]
        if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-5):
            return False
    return True


def find_mlm_head(backbone_dir: str, ckpt: str | None = None,
                  submodel: str = "auto",
                  match_encoder: dict | None = None
                  ) -> tuple[dict | None, str | None]:
    """Locate the trained MLM head for a backbone directory. ``(state, source)``.

    The head is not part of the encoder, so ``save_pretrained`` does not write it.
    It is looked for in three places, in order:

      1. ``mlm_head.safetensors`` inside ``backbone_dir`` (what current exports write),
      2. ``ckpt``, when one is named explicitly,
      3. ``final.pt`` — else the highest-numbered ``step_*.pt`` — in the backbone's
         parent run directory, which is where ``scripts/pretrain.py`` puts them.

    Step 3 is what makes a *pre-fix* backbone directory usable without retraining:
    the head was never exported, but the training checkpoint beside it still has it.
    Every consumer must search the same way, or one of them (the exporter, say) sees
    "no head" while another (the perplexity benchmark) happily finds one.
    """
    import glob

    side = os.path.join(backbone_dir, MLM_HEAD_FILE)
    if os.path.isfile(side):
        from safetensors.torch import load_file

        return load_file(side, device="cpu"), side

    candidates = []
    if ckpt:
        candidates.append(ckpt)
    else:
        run_dir = os.path.dirname(os.path.abspath(backbone_dir.rstrip("/\\")))
        final = os.path.join(run_dir, "final.pt")
        if os.path.exists(final):
            candidates.append(final)

        def _step_no(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return -1

        # Numeric order, not lexicographic: step_1000 must beat step_900.
        candidates.extend(sorted(glob.glob(os.path.join(run_dir, "step_*.pt")),
                                 key=_step_no, reverse=True))

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as err:  # noqa: BLE001 - a broken checkpoint must not abort
            print(f"[NexteraBERT] could not read {path} ({type(err).__name__}: {err})")
            continue
        sd = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(sd, dict):
            continue
        head = mlm_head_from_trainer_state(sd, submodel)
        if not head:
            continue
        if match_encoder is not None:
            prefix = next((pre for pre in ("model.", "generator.", "")
                           if any(k.startswith(pre + "head.") for k in sd)), "")
            if not _submodel_encoder_matches(sd, prefix, match_encoder):
                print(f"[NexteraBERT] {path}: the MLM head found under "
                      f"{prefix or '<root>'!r} belongs to a different encoder than "
                      f"{backbone_dir} -- ignoring it (an ELECTRA discriminator has no "
                      f"MLM head of its own).")
                continue
        return head, path
    return None, None


def write_mlm_head(state: dict, out_dir: str, tie_word_embeddings: bool = True) -> str:
    """Write ``state`` as ``out_dir/mlm_head.safetensors`` and return the path.

    ``lm_head.weight`` is dropped when it is tied to the input embedding: the encoder
    weights already carry that matrix, and a second copy would only risk drifting
    from the tie.
    """
    from safetensors.torch import save_file

    if tie_word_embeddings:
        state = {k: v for k, v in state.items() if k != "lm_head.weight"}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, MLM_HEAD_FILE)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in state.items()},
              path, metadata={"format": "pt"})
    return path


def load_mlm_head_state(path: str) -> dict | None:
    """Return the MLM output-head ``state_dict`` bundled with a pretrained
    directory, or ``None`` when the directory carries encoder weights only.

    ``save_pretrained`` writes the encoder; the masked-LM output head (the
    prediction transform + the untied decoder bias) is written separately as
    ``mlm_head.safetensors`` by :func:`nexterabert.export.save_mlm_head`.
    Accepts a local directory or a Hub repo id (downloaded on demand).
    """
    if os.path.isdir(path):
        head_path = os.path.join(path, MLM_HEAD_FILE)
        if not os.path.exists(head_path):
            return None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            head_path = hf_hub_download(path, MLM_HEAD_FILE)
        except (EntryNotFoundError, OSError):
            return None

    from safetensors.torch import load_file

    return load_file(head_path, device="cpu")


def load_masked_lm(pretrained_path: str, **config_overrides):
    """Rebuild the full masked-LM (encoder + output head) from a pretrained directory.

    The encoder comes from ``model.safetensors`` / ``pytorch_model.bin``; the output
    head — the prediction transform (dense + LayerNorm) and the decoder bias — comes
    from ``mlm_head.safetensors``. **Both are required for meaningful predictions**:
    the head is not part of the encoder, so without it the vocab projection runs on a
    randomly initialised transform and the model outputs noise (an inflated
    perplexity rather than an obvious crash). ``info["has_mlm_head"]`` says whether it
    was found, and a warning is printed when it was not.

    ``pretrained_path`` may be a local directory or a Hub repo id (fetched with
    ``snapshot_download``). Returns ``(model, config, info)``, mirroring
    :func:`load_for_task`.
    """
    if not os.path.isdir(pretrained_path):
        from huggingface_hub import snapshot_download

        pretrained_path = snapshot_download(pretrained_path)
    config = load_config(pretrained_path, **config_overrides)
    model = NexteraBERTForMaskedLM(config)

    enc_state = load_backbone_state(pretrained_path)
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)

    head_state = load_mlm_head_state(pretrained_path)
    if head_state:
        # lm_head.weight is tied to the input embedding the encoder already restored;
        # save_mlm_head omits it, and a stale copy must not break the tie.
        head_state = {k: v for k, v in head_state.items() if k != "lm_head.weight"} \
            if config.tie_word_embeddings else head_state
        h_missing, h_unexpected = model.load_state_dict(head_state, strict=False)
        unexpected = list(unexpected) + list(h_unexpected)
    else:
        print(f"[NexteraBERT] warning: no {MLM_HEAD_FILE} in {pretrained_path} -- the "
              f"MLM output head stays randomly initialised and predictions will be "
              f"meaningless. Re-export the backbone with "
              f"nexterabert.export.save_mlm_head().")

    info = {
        "missing": [m for m in missing
                    if not m.startswith(("pooler", "token_type_embeddings"))],
        "unexpected": unexpected,
        "has_mlm_head": bool(head_state),
    }
    return model, config, info


def load_for_task(
    pretrained_path: str,
    task: str,
    num_labels: int | None = None,
    **config_overrides,
):
    if task not in _HEADS:
        raise ValueError(f"task must be one of {list(_HEADS)}")
    if num_labels is not None:
        config_overrides["num_labels"] = num_labels
    config = load_config(pretrained_path, **config_overrides)
    model = _HEADS[task](config)

    enc_state = load_backbone_state(pretrained_path)
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
    info = {
        # pooler and token_type_embeddings are legitimately absent from an
        # ELECTRA discriminator backbone (the RTD head never uses the pooler, and
        # pretraining is single-segment), so a fresh zero/random init here is
        # expected -- don't surface them as a missing-weights warning.
        "missing": [m for m in missing
                    if not m.startswith(("pooler", "token_type_embeddings"))],
        "unexpected": unexpected,
    }
    return model, config, info
