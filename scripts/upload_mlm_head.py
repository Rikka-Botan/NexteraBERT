#!/usr/bin/env python
"""Add the missing MLM output head to a NexteraBERT repo already on the Hub.

Backbones exported before this fix contain the **encoder only** — ``save_pretrained``
writes ``_encoder_state_dict``, so the masked-LM output head (the ModernBERT-style
prediction transform ``head.dense`` + ``head.norm``, and the untied decoder bias
``lm_head.bias``) never made it into the repo. Rebuilding ``NexteraBERTForMaskedLM``
from such a repo leaves that transform randomly initialised: the model does not
crash, it just emits noise, which is why ``model_pplbench.py`` reported a
pseudo-perplexity in the thousands.

This script recovers the head from the training checkpoint and uploads it as a
single extra file, ``mlm_head.safetensors``. Nothing else in the repo is touched —
the encoder weights, config and tokenizer already there are left exactly as they
are, so the upload is additive and existing downloads keep working.

    # from the full training checkpoint
    python scripts/upload_mlm_head.py \
        --repo_id <user>/NexteraBERT-mezzoforte \
        --ckpt checkpoints/mezzoforte_bert/phase2/final.pt

    # or from a locally re-exported backbone that already has the sidecar
    python scripts/upload_mlm_head.py \
        --repo_id <user>/NexteraBERT-mezzoforte \
        --backbone checkpoints/mezzoforte_bert/phase2/backbone

    # look before you leap
    python scripts/upload_mlm_head.py --repo_id <user>/... --ckpt ... --dry_run

By default the NexteraBERT source modules and an ``auto_map`` in config.json go up
alongside the head, and ``scripts/verify_hub_model.py`` then confirms the result
really loads with ``AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)``
— checked in a subprocess that cannot import the local package, so the repo is proven
self-contained. ``--no_auto_class`` uploads the head alone; ``--no_verify`` skips the
check.

For an ELECTRA run the MLM head belongs to the **generator** repo — the
discriminator was never trained for MLM and correctly has no head. Pass
``--submodel generator`` (the default is auto-detection).

Requires ``huggingface-cli login`` (or ``HF_TOKEN`` / ``--token``).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexterabert.loading import (  # noqa: E402
    MLM_HEAD_FILE,
    mlm_head_from_trainer_state,
)

# Force UTF-8 stdout so progress logs survive a legacy Windows (cp932) console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def head_from_checkpoint(ckpt_path: str, submodel: str = "auto") -> tuple[dict, str]:
    """Extract the MLM head from a ``scripts/pretrain.py`` checkpoint.

    ``save_checkpoint`` stores the *trainer's* state dict, so the head sits under a
    submodel prefix: ``model.`` for the BERT and COCO-LM trainers, ``generator.``
    for the ELECTRA trainer. Returns ``(state, prefix_used)``.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(sd, dict):
        raise SystemExit(f"error: {ckpt_path} does not look like a training checkpoint")

    head = mlm_head_from_trainer_state(sd, submodel)
    if head is None:
        raise SystemExit(
            f"error: no MLM head found in {ckpt_path}. An ELECTRA *discriminator* has "
            f"none by design — use the generator checkpoint, or check --submodel.")
    # Report which submodel it came from, for the log line in main().
    prefix = next(pre for pre in ("model.", "generator.", "")
                  if any(k.startswith(pre + "head.") for k in sd))
    return head, prefix


def head_from_backbone(backbone_dir: str) -> dict:
    from safetensors.torch import load_file

    path = Path(backbone_dir) / MLM_HEAD_FILE
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found. Re-export with a current scripts/pretrain.py, "
            f"or pass --ckpt <run_dir>/final.pt instead.")
    return load_file(str(path), device="cpu")


def verify_against_repo(state: dict, repo_id: str, token: str | None) -> None:
    """Refuse to upload a head whose shapes contradict the repo's own config."""
    import json

    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(repo_id, "config.json", token=token))
                     .read_text(encoding="utf-8"))
    n_embd, vocab = cfg.get("n_embd"), cfg.get("vocab_size")
    checks = {
        "head.dense.weight": (n_embd, n_embd),
        "head.norm.weight": (n_embd,),
        "lm_head.bias": (vocab,),
    }
    for key, want in checks.items():
        if key not in state:
            raise SystemExit(f"error: the extracted head has no {key!r} "
                             f"(got {sorted(state)})")
        got = tuple(state[key].shape)
        if None not in want and got != want:
            raise SystemExit(
                f"error: {key} is {got} but {repo_id}/config.json implies {want} — "
                f"this checkpoint is not the one this repo was exported from.")
    print(f"[head] shapes match {repo_id}/config.json "
          f"(n_embd={n_embd}, vocab_size={vocab})")


def verify_same_run(ckpt_path: str, repo_id: str, token: str | None,
                    submodel: str = "auto") -> None:
    """Refuse to upload a head that came from a *different training run*.

    Shapes are a weak identity check: every NexteraBERT pipeline here writes to the
    same ``HF_REPO_ID`` by default, so a BERT, a COCO-LM and an ELECTRA run all end
    up as candidates for the same repo. Built from the same preset they have
    identical shapes, and the head from the wrong run would attach without complaint
    and quietly produce a wrong model.

    So compare the *values*: the encoder weights in the repo must be the ones this
    checkpoint holds. The head is only meaningful on top of the encoder it was
    trained with.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        weights = hf_hub_download(repo_id, "model.safetensors", token=token)
        repo_state = load_file(weights, device="cpu")
    except Exception as e:  # noqa: BLE001 - only pytorch_model.bin, or no access
        print(f"[head] [warn] could not read {repo_id}/model.safetensors "
              f"({type(e).__name__}); skipping the run-identity check.")
        return

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload) if isinstance(payload, dict) else payload
    prefixes = {"auto": ("model.encoder.", "generator.encoder.",
                         "discriminator.encoder.", "encoder."),
                "model": ("model.encoder.",), "generator": ("generator.encoder.",),
                "none": ("encoder.",)}[submodel]
    ckpt_enc = {}
    for prefix in prefixes:
        ckpt_enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if ckpt_enc:
            break
    if not ckpt_enc:
        print("[head] [warn] no encoder weights in the checkpoint; skipping the "
              "run-identity check.")
        return

    # A handful of tensors from different depths is plenty -- two runs that agree on
    # these agree everywhere; two that differ, differ here.
    shared = sorted(set(repo_state) & set(ckpt_enc))
    if not shared:
        raise SystemExit(
            f"error: {repo_id}/model.safetensors shares no parameter names with "
            f"{ckpt_path}. These are not the same architecture.")
    probes = [shared[0], shared[len(shared) // 3], shared[2 * len(shared) // 3],
              shared[-1]]
    mismatched = []
    for key in dict.fromkeys(probes):
        a, b = repo_state[key], ckpt_enc[key]
        if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-5):
            mismatched.append(key)

    if mismatched:
        raise SystemExit(
            f"error: the encoder weights in {repo_id} do NOT match {ckpt_path}\n"
            f"       (differ at {mismatched}).\n"
            f"\n"
            f"       This head belongs to a different training run, so the repo may\n"
            f"       hold a different model than you think. Check which run last\n"
            f"       uploaded to it, and pass that run's final.pt, e.g.\n"
            f"         checkpoints/mezzoforte_bert/phase2/final.pt\n"
            f"       Uploading the wrong head would not error -- it would just make\n"
            f"       the model predict noise, which is the bug this script exists\n"
            f"       to fix.")
    print(f"[head] encoder weights in {repo_id} match {ckpt_path} "
          f"(checked {len(set(probes))} tensors) -- same training run")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="the Hub repo to add the head to")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="training checkpoint (.pt) holding the MLM head")
    src.add_argument("--backbone", help="local backbone dir already holding "
                                        f"{MLM_HEAD_FILE}")
    p.add_argument("--submodel", default="auto",
                   choices=["auto", "model", "generator", "none"],
                   help="which submodel of the trainer checkpoint holds the MLM head "
                        "('model' = BERT/COCO-LM, 'generator' = ELECTRA)")
    p.add_argument("--token", default=None, help="HF token (else cached login / HF_TOKEN)")
    p.add_argument("--commit_message",
                   default="Add the MLM output head (mlm_head.safetensors)")
    p.add_argument("--no_auto_class", dest="auto_class", action="store_false",
                   help="upload ONLY mlm_head.safetensors. By default the NexteraBERT "
                        "source modules and an 'auto_map' in config.json go up too, "
                        "which is what makes the repo loadable with "
                        "AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True).")
    p.set_defaults(auto_class=True)
    p.add_argument("--no_check_run", dest="check_run", action="store_false",
                   help="skip the check that the repo's encoder weights actually come "
                        "from this checkpoint. That check is what stops a head from a "
                        "different run being attached -- all three pipelines default "
                        "to the same HF_REPO_ID, and same-preset runs have identical "
                        "shapes, so the shape check alone cannot tell them apart.")
    p.set_defaults(check_run=True)
    p.add_argument("--no_verify", dest="verify", action="store_false",
                   help="skip the post-upload trust_remote_code check "
                        "(scripts/verify_hub_model.py)")
    p.set_defaults(verify=True)
    p.add_argument("--dry_run", action="store_true",
                   help="extract, verify and report — upload nothing")
    args = p.parse_args()

    if args.ckpt:
        state, prefix = head_from_checkpoint(args.ckpt, args.submodel)
        print(f"[head] extracted from {args.ckpt} (prefix {prefix!r})")
    else:
        state = head_from_backbone(args.backbone)
        print(f"[head] read from {Path(args.backbone) / MLM_HEAD_FILE}")

    # The decoder weight is tied to the input embedding, which the repo's encoder
    # weights already carry; shipping a second copy would double the file for
    # nothing and risk drifting from the tie.
    state.pop("lm_head.weight", None)
    state = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}
    for k, v in sorted(state.items()):
        print(f"[head]   {k:24s} {tuple(v.shape)}  {v.dtype}")

    verify_against_repo(state, args.repo_id, args.token)
    if args.ckpt and args.check_run:
        verify_same_run(args.ckpt, args.repo_id, args.token, args.submodel)

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / MLM_HEAD_FILE
        save_file(state, str(local), metadata={"format": "pt"})
        size_mb = local.stat().st_size / 1e6
        if args.dry_run:
            print(f"[head] dry run — would upload {MLM_HEAD_FILE} ({size_mb:.1f} MB) "
                  f"to {args.repo_id}"
                  + (", plus the source modules and an auto_map in config.json"
                     if args.auto_class else "")
                  + "; nothing was sent.")
            return

        from huggingface_hub import HfApi

        HfApi(token=args.token).upload_file(
            path_or_fileobj=str(local),
            path_in_repo=MLM_HEAD_FILE,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    print(f"[head] uploaded {MLM_HEAD_FILE} ({size_mb:.1f} MB) -> "
          f"https://huggingface.co/{args.repo_id}")
    print("[head] load it with nexterabert.loading.load_masked_lm(repo_id)")

    if args.auto_class:
        upload_auto_class(args.repo_id, args.token)

    if args.verify:
        print()
        from verify_hub_model import verify

        verify(args.repo_id, args.token)


def upload_auto_class(repo_id: str, token: str | None) -> None:
    """Ship the source modules + an ``auto_map`` so ``trust_remote_code`` works.

    transformers needs both halves: the ``*.py`` files that define the classes, and
    the ``auto_map`` in config.json saying which class backs which auto-class. A repo
    uploaded before ``modeling_nexterabert_hf.py`` existed has neither, so both are
    (re)written here. Existing weights and the tokenizer are untouched.
    """
    import shutil
    import tempfile as _tempfile

    from huggingface_hub import HfApi, hf_hub_download
    from nexterabert.export import _SOURCE_FILES, _SRC_DIR, stamp_auto_map

    with _tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        names = []
        for fname in _SOURCE_FILES:
            src = _SRC_DIR / fname
            if src.exists():
                shutil.copy(src, staged / fname)
                names.append(fname)
        # Stamp auto_map onto the repo's OWN config.json so nothing else in it moves.
        shutil.copy(hf_hub_download(repo_id, "config.json", token=token),
                    staged / "config.json")
        if stamp_auto_map(staged):
            names.append("config.json")
        else:
            (staged / "config.json").unlink()
            print("[auto] config.json already has the right auto_map")

        print(f"[auto] uploading {len(names)} files: {', '.join(sorted(names))}")
        HfApi(token=token).upload_folder(
            folder_path=str(staged),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add transformers auto_map + source modules "
                           "(trust_remote_code)",
        )
    print(f"[auto] {repo_id} is now loadable with "
          f"AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)")


if __name__ == "__main__":
    main()
