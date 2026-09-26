"""Save a NexteraBERT backbone to disk and push it to the Hugging Face Hub.

These are the library-level counterparts of ``scripts/save_model.py`` and
``scripts/upload_to_hub.py`` — import them to export / upload a model
programmatically (e.g. from a training loop or a notebook)::

    from nexterabert.export import save_pretrained, push_to_hub

    save_pretrained(model, "checkpoints/discriminator", tokenizer=tok)
    push_to_hub("checkpoints/discriminator", "user/NexteraBERT-mezzoforte")

``save_pretrained`` writes a self-contained backbone directory:

    config.json            NexteraBERTConfig
    pytorch_model.bin      encoder weights (torch binary)            [default]
    model.safetensors      encoder weights (safetensors)            [optional]
    tokenizer files        when a tokenizer is supplied
    *.py source modules    when ``bundle_source=True`` (needed on the Hub), plus an
                           ``auto_map`` stamped into config.json so the repo loads
                           with ``trust_remote_code=True``

The weights are the **encoder (backbone)** ``state_dict`` — the reusable part of
an ELECTRA discriminator — matching what :func:`nexterabert.loading.load_for_task`
expects to read back.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .configuration_nexterabert import NexteraBERTConfig

# source modules copied into an uploaded repo so the model can be reconstructed
_SOURCE_FILES = (
    "modeling_nexterabert.py",
    "modeling_nexterabert_hf.py",
    "configuration_nexterabert.py",
    "loading.py",
    "export.py",
    "__init__.py",
)
_SRC_DIR = Path(__file__).resolve().parent

MODEL_CARD = """---
license: mit
library_name: nexterabert
tags:
  - nexterabert
  - encoder
  - fill-mask
pipeline_tag: feature-extraction
---

# {repo_id}

NexteraBERT ({size}) — a bidirectional encoder that interleaves SnowLily liquid
mixers with sliding-window attention, full attention (SSSMax) and NexteraHRA.

## Usage

```python
import torch
from huggingface_hub import snapshot_download
from nexterabert.loading import load_for_task   # pip install nexterabert (this repo)

path = snapshot_download("{repo_id}")
model, config, _ = load_for_task(path, "embedding")
model.eval()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
ids = tok(["hello world"], return_tensors="pt").input_ids
with torch.no_grad():
    embedding = model(input_ids=ids)   # (1, hidden_size), mean-pooled + normalised
```

For fine-tuning use `load_for_task(path, "sequence-classification", num_labels=N)`
etc. See the training repository for full details.

## Masked-LM usage

```python
from nexterabert.loading import load_masked_lm

model, config, info = load_masked_lm("{repo_id}")   # encoder + MLM output head
```

`load_for_task` restores the encoder only, which is all a fine-tuning head needs.
Masked-LM inference additionally needs `mlm_head.safetensors` (the prediction
transform + decoder bias); `load_masked_lm` loads both and warns if the head is
absent.

## Files
- `config.json` — `NexteraBERTConfig`
- `pytorch_model.bin` / `model.safetensors` — encoder (backbone) weights
- `mlm_head.safetensors` — masked-LM output head (present on MLM-trained backbones)
- `modeling_nexterabert.py`, `configuration_nexterabert.py`, `loading.py` — source
"""


def _encoder_module(model) -> nn.Module:
    """Resolve the reusable encoder ``nn.Module`` from whatever was passed.

    Accepts an encoder module, any head with an ``.encoder`` attribute, or an
    ELECTRA trainer (``.discriminator.encoder``); DDP / ``torch.compile`` wrappers
    are unwrapped first.
    """
    raw = model.module if hasattr(model, "module") else model          # unwrap DDP
    raw = getattr(raw, "_orig_mod", raw)                               # unwrap compile
    if hasattr(raw, "discriminator"):                                 # electra trainer
        raw = raw.discriminator
    if hasattr(raw, "encoder"):                                       # a task head
        raw = raw.encoder
    if not isinstance(raw, nn.Module):
        raise TypeError(
            "save_pretrained expects a model / encoder / state_dict, "
            f"got {type(model)!r}"
        )
    return raw


def _encoder_state_dict(model) -> dict[str, torch.Tensor]:
    """Pull the reusable encoder ``state_dict`` out of whatever was passed.

    Accepts an already-extracted ``state_dict`` in addition to the module forms
    handled by :func:`_encoder_module`.
    """
    if isinstance(model, Mapping):
        return {k: v for k, v in model.items()}
    return _encoder_module(model).state_dict()


def _resolve_config(model, config) -> NexteraBERTConfig:
    if config is not None:
        return config
    if not isinstance(model, Mapping):
        cfg = getattr(_encoder_module(model), "config", None)
        if isinstance(cfg, NexteraBERTConfig):
            return cfg
    raise ValueError(
        "could not infer the config — pass config=... explicitly"
    )


def save_mlm_head(model, output_dir: str | os.PathLike) -> str | None:
    """Write the MLM **output head** next to an exported backbone.

    :func:`save_pretrained` deliberately stores only the encoder, but the masked-LM
    output head — the ModernBERT-style prediction transform (dense + LayerNorm) and
    the untied decoder bias — is a trained part of the model that lives *outside* it.
    Without these, rebuilding ``NexteraBERTForMaskedLM`` from a backbone directory
    leaves the transform randomly initialised and the vocab projection is nonsense
    (``model_pplbench.py`` then reports a pseudo-perplexity in the thousands).

    Written as ``mlm_head.safetensors`` with keys relative to
    ``NexteraBERTForMaskedLM`` (``head.*`` / ``lm_head.*``). The decoder weight is
    skipped when it is tied to the input embedding, which the backbone already
    stores. Returns the file path, or ``None`` when ``model`` has no MLM head (an
    ELECTRA discriminator).
    """
    raw = model.module if hasattr(model, "module") else model          # unwrap DDP
    raw = getattr(raw, "_orig_mod", raw)                               # unwrap compile
    if hasattr(raw, "generator"):                                      # electra trainer
        raw = raw.generator
    elif hasattr(raw, "model"):                                        # bert/coco trainer
        raw = raw.model

    head = getattr(raw, "head", None)
    lm_head = getattr(raw, "lm_head", None)
    if head is None or lm_head is None:
        return None

    state = {f"head.{k}": v for k, v in head.state_dict().items()}
    if lm_head.bias is not None:
        state["lm_head.bias"] = lm_head.bias
    # Only store the decoder matrix when it is NOT the (already-saved) input embedding.
    emb = getattr(getattr(raw, "encoder", None), "embedding", None)
    tied = emb is not None and lm_head.weight.data_ptr() == emb.weight.data_ptr()
    if not tied:
        state["lm_head.weight"] = lm_head.weight

    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mlm_head.safetensors"
    save_file({k: v.detach().to("cpu").contiguous() for k, v in state.items()},
              str(path), metadata={"format": "pt"})
    return str(path)


def save_pretrained(
    model,
    output_dir: str | os.PathLike,
    *,
    config: NexteraBERTConfig | None = None,
    tokenizer=None,
    save_safetensors: bool = True,
    save_bin: bool = True,
    bundle_source: bool = False,
) -> str:
    """Export an encoder backbone to ``output_dir`` as a reusable model directory.

    Parameters
    ----------
    model
        A task head, an :class:`NexteraBERT` encoder, an ELECTRA trainer, or a raw
        encoder ``state_dict``.  The encoder weights are what gets saved.
    output_dir
        Destination directory (created if missing).
    config
        The :class:`NexteraBERTConfig`; inferred from ``model`` when omitted.
    tokenizer
        Optional tokenizer; ``tokenizer.save_pretrained`` is called when given.
    save_safetensors / save_bin
        Which binary formats to write. ``model.safetensors`` and/or
        ``pytorch_model.bin``. At least one must be true.
    bundle_source
        Copy the NexteraBERT ``*.py`` source modules into the directory so the
        model can be rebuilt from the Hub. Enabled automatically by
        :func:`push_to_hub`.

    Returns
    -------
    str
        The output directory path.
    """
    if not (save_safetensors or save_bin):
        raise ValueError("at least one of save_safetensors / save_bin must be True")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = _encoder_state_dict(model)
    # move to CPU & make contiguous so the files are portable
    state = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}

    cfg = _resolve_config(model, config)
    cfg.save_pretrained(out)

    if save_bin:
        torch.save(state, out / "pytorch_model.bin")
    if save_safetensors:
        from safetensors.torch import save_file

        save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})

    if tokenizer is not None:
        tokenizer.save_pretrained(out)

    if bundle_source:
        _bundle_source(out)

    return str(out)


def _bundle_source(out: Path) -> None:
    for fname in _SOURCE_FILES:
        src = _SRC_DIR / fname
        if src.exists():
            shutil.copy(src, out / fname)
    stamp_auto_map(out)


def stamp_auto_map(out: str | os.PathLike) -> bool:
    """Write ``auto_map`` into ``out/config.json`` so ``trust_remote_code`` works.

    ``AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)`` finds the
    remote classes only through this mapping; without it transformers has the source
    files but no idea which class implements which auto-class. Paired with
    :func:`_bundle_source`, which puts ``modeling_nexterabert_hf.py`` in the repo.

    Returns True when the file was changed.
    """
    import json

    cfg_path = Path(out) / "config.json"
    if not cfg_path.exists():
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    from .modeling_nexterabert_hf import AUTO_MAP

    if cfg.get("auto_map") == AUTO_MAP:
        return False
    cfg["auto_map"] = dict(AUTO_MAP)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return True


def ensure_mlm_head(backbone_dir: str | os.PathLike,
                    ckpt: str | None = None) -> str | None:
    """Make sure ``backbone_dir`` carries its ``mlm_head.safetensors``, if one exists.

    A backbone exported before ``save_mlm_head`` existed has no head file, but the
    training checkpoint sitting next to it still holds the trained head. This is
    exactly the fallback ``model_pplbench.py`` already does at load time — so the
    benchmark scores such a backbone happily while an upload of the *same directory*
    ships no head at all, and the published model predicts noise.

    Recovers the head into the directory so the export, the upload and the benchmark
    all agree. Returns the path written, the existing path if there was already one,
    or ``None`` when the run genuinely has no MLM head (an ELECTRA discriminator).
    """
    from .loading import MLM_HEAD_FILE, find_mlm_head, write_mlm_head

    out = Path(backbone_dir)
    existing = out / MLM_HEAD_FILE
    if existing.is_file():
        return str(existing)

    # Match against this directory's own encoder: an ELECTRA checkpoint holds both a
    # discriminator and a generator, and only the generator has an MLM head. Handing
    # that head to a discriminator backbone would pair two different models silently.
    from .loading import load_backbone_state

    try:
        reference = load_backbone_state(str(out))
    except Exception:  # noqa: BLE001 - unreadable weights; fall back to no matching
        reference = None
    state, source = find_mlm_head(str(out), ckpt, match_encoder=reference)
    if not state:
        return None

    tie = True
    cfg_path = out / "config.json"
    if cfg_path.exists():
        import json

        tie = json.loads(cfg_path.read_text(encoding="utf-8")).get(
            "tie_word_embeddings", True)
    path = write_mlm_head(state, str(out), tie_word_embeddings=tie)
    print(f"[NexteraBERT] recovered the MLM head from {source} -> {path}")
    return path


def _warn_stale_mlm_head(api, repo_id: str, model_dir: Path) -> None:
    """Warn when uploading a head-less backbone over a repo that already has a head.

    ``upload_folder`` only adds and overwrites — it never deletes what is already in
    the repo. So pushing an encoder-only export (an ELECTRA discriminator, or a
    directory made before ``save_mlm_head`` existed) into a repo that a previous run
    gave an ``mlm_head.safetensors`` leaves that head behind, now paired with a
    *different* encoder. When the two happen to share a shape nothing errors and the
    mismatch is silent, so say so loudly. Deleting is the user's call, not ours.
    """
    if (model_dir / "mlm_head.safetensors").exists():
        return
    try:
        remote = api.list_repo_files(repo_id, repo_type="model")
    except Exception:  # noqa: BLE001 - a brand-new repo, or an offline/permissions hiccup
        return
    if "mlm_head.safetensors" in remote:
        print(
            f"[NexteraBERT] WARNING: {repo_id} already contains mlm_head.safetensors, "
            f"but {model_dir} has none, and upload_folder does not delete. The old "
            f"head will survive next to the new encoder and load_masked_lm() will "
            f"pair them -- silently wrong if the shapes happen to match. Delete it "
            f"with:\n"
            f"    python -c \"from huggingface_hub import HfApi; "
            f"HfApi().delete_file('mlm_head.safetensors', '{repo_id}')\"\n"
            f"or upload this backbone to its own repo."
        )


def push_to_hub(
    model,
    repo_id: str,
    *,
    config: NexteraBERTConfig | None = None,
    tokenizer=None,
    size: str = "",
    private: bool = False,
    commit_message: str = "Upload NexteraBERT backbone",
    token: str | None = None,
    save_safetensors: bool = True,
    create_model_card: bool = True,
) -> str:
    """Save (if needed) and upload a NexteraBERT backbone to the Hugging Face Hub.

    ``model`` may be a path to an already-exported directory or any object that
    :func:`save_pretrained` accepts. When it is not a directory, the model is
    exported to a temporary folder first. Authentication uses ``token`` (or a
    cached ``huggingface-cli login`` / ``HF_TOKEN``).

    Returns the ``https://huggingface.co/<repo_id>`` URL.
    """
    from huggingface_hub import HfApi, create_repo

    tmp_dir: Path | None = None
    model_dir = Path(model) if isinstance(model, (str, os.PathLike)) else None

    if model_dir is not None and model_dir.is_dir():
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"{model_dir}/config.json not found — run save_pretrained / pretrain export first"
            )
        # A directory exported before save_mlm_head existed has no head file, but the
        # training checkpoint beside it does. Recover it now rather than publishing a
        # backbone whose masked-LM output is noise.
        ensure_mlm_head(model_dir)
        _bundle_source(model_dir)
    else:
        # in-memory model (or a state_dict): export to a temp dir next to nothing
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="nexterabert_hub_"))
        save_pretrained(
            model, tmp_dir, config=config, tokenizer=tokenizer,
            save_safetensors=save_safetensors, bundle_source=True,
        )
        # The MLM output head lives outside the encoder, so save_pretrained skips
        # it; ship it too when the model has one, or the uploaded repo cannot be
        # used as a masked LM (a randomly initialised head predicts noise).
        save_mlm_head(model, tmp_dir)
        model_dir = tmp_dir

    try:
        if create_model_card:
            card = MODEL_CARD.format(repo_id=repo_id, size=size or "custom")
            (model_dir / "README.md").write_text(card, encoding="utf-8")

        create_repo(repo_id, token=token, private=private,
                    exist_ok=True, repo_type="model")
        api = HfApi(token=token)
        _warn_stale_mlm_head(api, repo_id, model_dir)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    url = f"https://huggingface.co/{repo_id}"
    print(f"uploaded -> {url}")
    return url


__all__ = ["save_pretrained", "save_mlm_head", "stamp_auto_map", "push_to_hub",
           "MODEL_CARD"]
