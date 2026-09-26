#!/usr/bin/env python
"""Export a NexteraBERT backbone to a binary model directory.

Convert a training checkpoint (``final.pt`` / ``step_N.pt`` produced by
``scripts/pretrain.py``) into a clean, reusable backbone directory holding the
**discriminator encoder** weights in binary form:

    python scripts/save_model.py --checkpoint checkpoints/final.pt \
        --output checkpoints/discriminator --tokenizer answerdotai/ModernBERT-base

Output directory contents:

    config.json            NexteraBERTConfig
    pytorch_model.bin      encoder weights (torch binary)
    model.safetensors      encoder weights (safetensors)   [unless --no_safetensors]
    mlm_head.safetensors   masked-LM output head, when the checkpoint has one
    tokenizer files        when --tokenizer is given

The result is what ``nexterabert.loading.load_for_task`` and
``scripts/upload_to_hub.py`` consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert import NexteraBERTConfig  # noqa: E402
from nexterabert.export import save_pretrained  # noqa: E402
from nexterabert.loading import (  # noqa: E402
    MLM_HEAD_FILE,
    load_backbone_state,
    mlm_head_from_trainer_state,
)


def _mlm_head_state(ckpt_path: Path) -> dict | None:
    """The MLM output head stored in a training checkpoint, or ``None``.

    ``save_checkpoint`` writes the *trainer's* state dict, so the head sits under a
    submodel prefix: ``model.`` for the BERT / COCO-LM trainers, ``generator.`` for
    ELECTRA (whose discriminator has no MLM head by design).
    """
    if ckpt_path.is_dir():
        from nexterabert.loading import load_mlm_head_state

        return load_mlm_head_state(str(ckpt_path))
    import torch

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(sd, dict):
        return None
    if any(k.startswith("discriminator.encoder.") for k in sd):
        # ELECTRA: load_backbone_state exports the DISCRIMINATOR encoder, and the
        # only MLM head in the checkpoint belongs to the generator — a different
        # (usually smaller) encoder. Pairing them would produce a wrong model, so
        # export the discriminator head-less, as ELECTRA intends.
        return None
    return mlm_head_from_trainer_state(sd)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="training checkpoint (.pt) or existing backbone dir / .bin")
    p.add_argument("--output", required=True, help="destination backbone directory")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer name/path to bundle (e.g. answerdotai/ModernBERT-base)")
    p.add_argument("--no_safetensors", action="store_true",
                   help="skip writing model.safetensors")
    p.add_argument("--no_bin", action="store_true",
                   help="skip writing pytorch_model.bin")
    p.add_argument("--bundle_source", action="store_true",
                   help="also copy the NexteraBERT source modules into the directory")
    return p.parse_args()


def _load_config(ckpt_path: Path) -> NexteraBERTConfig:
    """Recover a NexteraBERTConfig from a checkpoint / directory."""
    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        return NexteraBERTConfig.from_pretrained(ckpt_path)
    import torch

    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        return NexteraBERTConfig(**obj["config"])
    raise SystemExit(
        f"error: no config found in {ckpt_path}; pass an exported directory or a "
        "checkpoint saved by pretrain.py (which embeds the config)."
    )


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"error: {ckpt} not found")

    config = _load_config(ckpt)
    state = load_backbone_state(str(ckpt))
    head_state = _mlm_head_state(ckpt)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = save_pretrained(
        state,
        args.output,
        config=config,
        tokenizer=tokenizer,
        save_safetensors=not args.no_safetensors,
        save_bin=not args.no_bin,
        bundle_source=args.bundle_source,
    )
    print(f"saved backbone -> {out}")

    # The MLM output head lives outside the encoder, so save_pretrained never sees
    # it. Carry it across when the checkpoint has one, or the exported directory
    # cannot be used as a masked LM (a randomly initialised head predicts noise).
    if head_state:
        from safetensors.torch import save_file

        head_state.pop("lm_head.weight", None)      # tied to the saved embedding
        path = Path(out) / MLM_HEAD_FILE
        save_file({k: v.detach().to("cpu").contiguous()
                   for k, v in head_state.items()},
                  str(path), metadata={"format": "pt"})
        print(f"saved MLM head -> {path}")
    else:
        print("no MLM head in this checkpoint (expected for an ELECTRA discriminator);"
              " the directory is encoder-only and cannot do masked-LM inference.")


if __name__ == "__main__":
    main()
