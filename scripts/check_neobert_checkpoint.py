"""Was this fine-tuned NeoBERT checkpoint trained with a zero-filled RoPE table?

transformers >= 5 leaves NeoBERT's computed RoPE table (``freqs_cis``) ZERO-FILLED when
the model is opened by a plain ``AutoModel.from_pretrained`` -- which is what
sentence-transformers and PyLate do. Until ``hf_baselines.repair_wrapped_neobert`` was
wired into the trainers (2026-09-20), every NeoBERT DPR / ColBERT run here therefore
trained with *uniform attention*: rotating by a zero table zeroes every query and key.

That leaves a fingerprint in the weights, so no retraining is needed to find out. With
``q = k = 0`` the loss does not depend on the query / key projections at all: their
gradient is exactly zero, and without weight decay (the trainers' default) those rows
of every ``qkv`` matrix are still **bit-identical to the base model**, while the value
rows right next to them -- same matrix, same optimiser -- have moved. A healthy run
moves all three.

Such a checkpoint scores ~0 once the table is repaired at evaluation time (the rest of
the network was tuned for uniform-attention inputs and now receives real attention),
and its uniform-attention score was never a NeoBERT number either. Retrain it.

    python scripts/check_neobert_checkpoint.py eval_results/NeoBERT/dpr/lr8e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_state(path):
    """``state_dict`` of a Hub id or of a model / sentence-transformers directory."""
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.safetensors"))
        files = [f for f in files if "checkpoint-" not in str(f)] or files
        if not files:
            raise SystemExit(f"no .safetensors file under {p}")
        state = {}
        for f in files:
            state.update(load_file(str(f), device="cpu"))
        return state
    from huggingface_hub import hf_hub_download

    return load_file(hf_hub_download(str(path), "model.safetensors"), device="cpu")


def qkv_rows(state, n_heads):
    """{layer key: (q, k, v) row blocks}. NeoBERT packs each head as [q | k | v]."""
    out = {}
    for key, w in state.items():
        if not key.endswith("qkv.weight"):
            continue
        per_head = w.float().view(n_heads, 3, w.size(0) // (3 * n_heads), w.size(1))
        layer = key[: -len("qkv.weight")].rstrip(".").split("transformer_encoder.")[-1]
        out[layer] = (per_head[:, 0], per_head[:, 1], per_head[:, 2])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="fine-tuned NeoBERT: model or "
                                       "sentence-transformers / PyLate directory")
    ap.add_argument("--base", default="chandar-lab/NeoBERT",
                    help="the checkpoint it was fine-tuned from")
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--finite_only", action="store_true",
                    help="only test for NaN/inf weights (exit 3 when found, else 0); "
                         "works for any model, and is what eval_common.sh asks before "
                         "it reuses a checkpoint")
    args = ap.parse_args()

    tuned_state = load_state(args.checkpoint)
    import torch

    dead = [k for k, v in tuned_state.items()
            if (v.is_floating_point() or v.is_complex())
            and not bool(torch.isfinite(v).all())]
    if dead:
        print(f"VERDICT: DIVERGED -- {len(dead)} of {len(tuned_state)} tensors hold "
              f"NaN/inf (e.g. {dead[:3]}). Training blew up and the checkpoint was "
              f"saved anyway; every retrieval metric reads 0.0 because the embeddings "
              f"are NaN. Look for the first 'nan' in the training log, then retrain "
              f"with a lower learning rate.")
        return 3
    if args.finite_only:
        print(f"finite: all {len(tuned_state)} tensors")
        return 0
    tuned = qkv_rows(tuned_state, args.num_heads)
    base = qkv_rows(load_state(args.base), args.num_heads)
    layers = sorted(set(tuned) & set(base), key=lambda s: (len(s), s))
    if not layers:
        raise SystemExit("no matching '*qkv.weight' tensors -- is this a NeoBERT?")

    print(f"{'layer':<8}{'max|dq|':>12}{'max|dk|':>12}{'max|dv|':>12}")
    dq = dk = dv = 0.0
    for layer in layers:
        d = [float((t - b).abs().max()) for t, b in zip(tuned[layer], base[layer])]
        dq, dk, dv = max(dq, d[0]), max(dk, d[1]), max(dv, d[2])
        print(f"{layer:<8}{d[0]:>12.3e}{d[1]:>12.3e}{d[2]:>12.3e}")

    print()
    if dv == 0.0:
        print("VERDICT: identical to the base model -- this checkpoint was not "
              "fine-tuned at all.")
        return 2
    if max(dq, dk) <= 1e-6 * dv:
        print("VERDICT: trained with a ZERO RoPE table (uniform attention). The query "
              "and key projections never received a gradient -- they are unchanged "
              "from the base model in every layer -- while the value projections "
              "moved. Retrain this checkpoint; none of its scores are a NeoBERT "
              "result.")
        return 1
    print("VERDICT: healthy -- query, key and value projections all moved, so "
          "attention was live during training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
