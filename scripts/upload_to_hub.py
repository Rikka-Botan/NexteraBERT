#!/usr/bin/env python
"""Push a trained NexteraBERT backbone to the Hugging Face Hub.

    python scripts/upload_to_hub.py --model checkpoints/discriminator \
        --repo_id <user>/NexteraBERT-mezzoforte --private

Requires `huggingface-cli login` (or the HF_TOKEN env var / --token) beforehand.

The uploaded repo contains: config.json, pytorch_model.bin + model.safetensors
(encoder weights), the tokenizer, the NexteraBERT source modules (so the model can
be rebuilt) and a generated model card.

This is a thin CLI around ``nexterabert.export.push_to_hub``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert.export import push_to_hub  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="exported backbone directory")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--size", default="", help="size tag for the model card")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (else cached login / HF_TOKEN)")
    p.add_argument("--no_model_card", action="store_true",
                   help="do not generate/overwrite README.md")
    p.add_argument("--commit_message", default="Upload NexteraBERT backbone")
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        sys.exit(f"error: {model_dir}/config.json not found — run pretrain.py export first")

    push_to_hub(
        str(model_dir),
        repo_id=args.repo_id,
        size=args.size,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
        create_model_card=not args.no_model_card,
    )


if __name__ == "__main__":
    main()
