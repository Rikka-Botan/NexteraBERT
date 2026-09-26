#!/usr/bin/env python
"""Verify that an uploaded NexteraBERT repo really loads with ``trust_remote_code``.

    python scripts/verify_hub_model.py --repo_id <user>/NexteraBERT-Mezzoforte-220M-en

Downloading a repo and calling ``AutoModelForMaskedLM.from_pretrained(...,
trust_remote_code=True)`` in the same process that has ``nexterabert`` installed
proves very little: the local package can satisfy imports the repo does not actually
ship. So the Auto* checks run in a **subprocess with the local package removed from
the import path** -- if it passes there, it passes for a stranger with nothing but
``pip install transformers``.

Checks, in order:

  1. the repo carries config.json, encoder weights, ``mlm_head.safetensors`` and the
     bundled ``*.py`` source modules;
  2. config.json has an ``auto_map`` (without it ``trust_remote_code`` has the source
     but no idea which class implements which auto-class);
  3. ``AutoConfig`` / ``AutoTokenizer`` / ``AutoModel`` / ``AutoModelForMaskedLM``
     all instantiate from the remote code;
  4. the masked-LM logits match ``nexterabert.loading.load_masked_lm`` on the same
     input -- this is what catches a *silently* missing output head, which produces a
     perfectly shaped tensor full of noise rather than an error;
  5. a real ``fill-mask`` pipeline call, printed so the predictions can be eyeballed.

``--skip_parity`` drops step 4 (it is the only step needing the local package, so use
it to verify a repo from a machine that does not have this repository checked out).

``--expect encoder`` verifies a backbone that has no masked-LM head *by design* - an
ELECTRA discriminator, whose objective is replaced-token detection. The head stops
being required, the model check ends at ``AutoModel``, and parity is measured on the
encoder's hidden states instead of MLM logits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = ("config.json", "modeling_nexterabert.py",
                  "modeling_nexterabert_hf.py", "configuration_nexterabert.py")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

# Sentences used for the fill-mask smoke test. Kept ASCII so a legacy Windows
# console can print the predictions back.
SENTENCES = [
    "The capital of France is {mask}.",
    "Photosynthesis converts sunlight into chemical {mask}.",
]


def check_files(repo_id: str, token: str | None, expect: str = "mlm",
                local_dir: str | None = None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    files = set(HfApi(token=token).list_repo_files(repo_id, repo_type="model"))
    print(f"[verify] {repo_id}: {len(files)} files")

    problems = []
    for name in REQUIRED_FILES:
        ok = name in files
        print(f"[verify]   {'OK  ' if ok else 'MISS'} {name}")
        if not ok:
            problems.append(name)
    if not (set(WEIGHT_FILES) & files):
        print(f"[verify]   MISS {' / '.join(WEIGHT_FILES)}")
        problems.append("weights")
    else:
        print(f"[verify]   OK   encoder weights")

    has_head = "mlm_head.safetensors" in files
    if expect == "mlm":
        print(f"[verify]   {'OK  ' if has_head else 'MISS'} mlm_head.safetensors"
              f"{'' if has_head else '  <- masked-LM output would be noise'}")
        if not has_head:
            problems.append("mlm_head.safetensors")
    else:
        # An ELECTRA discriminator is trained on replaced-token detection and has no
        # MLM head at all, so its absence is correct rather than a defect.
        print(f"[verify]   n/a  mlm_head.safetensors (encoder-only repo"
              f"{'; one is present anyway' if has_head else ''})")

    cfg = json.loads(Path(hf_hub_download(repo_id, "config.json", token=token))
                     .read_text(encoding="utf-8"))
    auto_map = cfg.get("auto_map") or {}
    print(f"[verify]   {'OK  ' if auto_map else 'MISS'} auto_map in config.json")
    for k, v in sorted(auto_map.items()):
        print(f"[verify]        {k} -> {v}")
    if not auto_map:
        problems.append("auto_map")

    if problems:
        raise SystemExit(_missing_files_help(repo_id, problems, local_dir))
    return {"has_head": has_head, "config": cfg}


def _local_evidence(local_dir: str | None, problems: list) -> list:
    """Say which side is at fault by looking at the directory that was uploaded.

    Called from a pipeline, ``local_dir`` is the backbone that ``upload_to_hub.py``
    just pushed. Comparing it against the repo turns "something is missing" into one
    of two very different diagnoses: the export never wrote the file, or the export
    wrote it and the upload did not carry it across.
    """
    if not local_dir:
        return []
    import os as _os

    if not _os.path.isdir(local_dir):
        return [f"The local backbone {local_dir} does not exist.", ""]

    have = sorted(_os.listdir(local_dir))
    lines = [f"Local backbone {local_dir} contains:"]
    lines += [f"    {f}" for f in have] or ["    (empty)"]
    lines.append("")

    if "mlm_head.safetensors" in problems:
        if "mlm_head.safetensors" in have:
            lines += [
                ">> The head IS present locally but did NOT reach the repo, so the",
                "   EXPORT is fine and the UPLOAD is the problem. Push just that file:",
                "",
                "     python scripts/upload_mlm_head.py \\",
                "         --repo_id REPO --backbone " + local_dir,
                "",
                "   (case (b) below does not apply -- do not re-export.)",
                "",
            ]
        else:
            lines += [
                ">> The head is missing LOCALLY too, so the export never wrote it:",
                "   this backbone was produced by a scripts/pretrain.py that predates",
                "   the fix. Case (b) below is the one that applies -- re-export from",
                "   the checkpoint, no retraining needed.",
                "",
            ]
    return lines


def _missing_files_help(repo_id: str, problems: list,
                        local_dir: str | None = None) -> str:
    """Explain what to do about each missing file, not just that it is missing.

    "the repo is missing X" is only half an answer: the same missing file means
    different things depending on how the repo got there, and the fix differs. Spell
    out the branches so the message is actionable on its own, and when a local
    backbone is known, say which branch it actually is.
    """
    lines = [f"[verify] FAIL: {repo_id} is missing {problems}.", ""]
    lines += _local_evidence(local_dir, problems)

    if "mlm_head.safetensors" in problems:
        lines += [
            "The masked-LM output head is not in the repo. Three ways that happens,",
            "with a different fix each:",
            "",
            "  a) The repo was uploaded before the head was ever exported (most",
            "     likely). Recover it from the training checkpoint:",
            "",
            "       python scripts/upload_mlm_head.py \\",
            f"           --repo_id {repo_id} \\",
            "           --ckpt <output_dir>/final.pt",
            "",
            "     <output_dir> is the one from your configs/*.yaml, e.g.",
            "     checkpoints/mezzoforte_bert/phase2. Add --dry_run to look first.",
            "",
            "  b) A pipeline uploaded it just now, from a backbone directory that",
            "     predates the fix. Check with:",
            "",
            "       ls <output_dir>/backbone/mlm_head.safetensors",
            "",
            "     If it is absent, re-export from the checkpoint (no retraining):",
            "",
            "       python scripts/save_model.py \\",
            "           --checkpoint <output_dir>/final.pt \\",
            "           --output <output_dir>/backbone \\",
            "           --tokenizer answerdotai/ModernBERT-base",
            "",
            "     then re-run scripts/upload_to_hub.py.",
            "",
            "  c) This repo is an ELECTRA *discriminator*, which has no MLM head by",
            "     design (its objective is replaced-token detection). Verify it as",
            "     an encoder instead:",
            "",
            f"       python scripts/verify_hub_model.py --repo_id {repo_id} \\",
            "           --expect encoder",
            "",
            "     An ELECTRA run keeps its MLM head in the generator backbone.",
            "",
            "If final.pt is gone the head cannot be recovered: it is trained weights,",
            "not something that can be recomputed.",
            "",
        ]

    other = [p for p in problems if p != "mlm_head.safetensors"]
    if other:
        lines += [
            f"Also missing: {other}. Without the source modules and the auto_map,",
            "trust_remote_code has nothing to load. Both go up with:",
            "",
            f"    python scripts/upload_mlm_head.py --repo_id {repo_id} \\",
            "        --ckpt <output_dir>/final.pt",
            "",
            "(or with any upload made by a current scripts/upload_to_hub.py).",
        ]
    return "\n".join(lines)


def reference_logits(repo_id: str, out_path: str, expect: str = "mlm") -> None:
    """Reference activations from the in-repo API, for the remote-code model to match.

    ``expect="mlm"`` compares masked-LM logits (``load_masked_lm``); ``"encoder"``
    compares the encoder's ``last_hidden_state``, which is all an RTD discriminator
    exposes.
    """
    import torch

    sys.path.insert(0, str(_REPO_ROOT / "src"))

    ids = torch.arange(1000, 1000 + 64).unsqueeze(0) % 30000 + 100
    if expect == "mlm":
        from nexterabert.loading import load_masked_lm

        model, _, info = load_masked_lm(repo_id)
        model.eval()
        if not info["has_mlm_head"]:
            raise SystemExit("[verify] FAIL: load_masked_lm found no MLM head")
        with torch.no_grad():
            out = model(ids, attention_mask=torch.ones_like(ids))
        source = "load_masked_lm"
    else:
        from nexterabert.loading import load_backbone_state, load_config
        from nexterabert.modeling_nexterabert import NexteraBERT

        encoder = NexteraBERT(load_config(repo_id))
        encoder.load_state_dict(load_backbone_state(repo_id), strict=False)
        encoder.eval()
        with torch.no_grad():
            out, _ = encoder(ids, attention_mask=torch.ones_like(ids))
        source = "NexteraBERT encoder"
    torch.save({"ids": ids, "logits": out}, out_path)
    print(f"[verify] reference activations from {source}: {tuple(out.shape)}")


CHILD = '''
import sys, torch
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

REPO = {repo!r}
REF = {ref!r}
TOKEN = {token!r}
SENTENCES = {sentences!r}
EXPECT = {expect!r}

# The point of the subprocess: the local package must be unreachable, so the only
# possible source of the model classes is the code inside the repo. Dropping it from
# the path is not enough -- it is usually pip-installed on a training machine -- so
# block the import outright. transformers loads remote code under
# `transformers_modules.*`, which this does not touch.
import importlib.abc


class _BlockLocalPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nexterabert" or fullname.startswith("nexterabert."):
            raise ImportError(f"{{fullname}} is blocked by verify_hub_model isolation")
        return None


sys.meta_path.insert(0, _BlockLocalPackage())
try:
    import nexterabert  # noqa: F401
    raise SystemExit("[verify] FAIL: could not isolate the local package")
except ImportError:
    print("[verify] isolation: the local nexterabert package is unreachable")

from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, pipeline)

kw = dict(trust_remote_code=True)
if TOKEN:
    kw["token"] = TOKEN

cfg = AutoConfig.from_pretrained(REPO, **kw)
print(f"[verify] AutoConfig               -> {{type(cfg).__name__}} "
      f"(model_type={{cfg.model_type}})")

tok = AutoTokenizer.from_pretrained(REPO, **({{"token": TOKEN}} if TOKEN else {{}}))
print(f"[verify] AutoTokenizer            -> {{type(tok).__name__}} (vocab {{len(tok)}})")

base = AutoModel.from_pretrained(REPO, **kw).eval()
print(f"[verify] AutoModel                -> {{type(base).__name__}}")

model = None
if EXPECT == "mlm":
    model = AutoModelForMaskedLM.from_pretrained(REPO, **kw).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[verify] AutoModelForMaskedLM     -> {{type(model).__name__}} "
          f"({{n/1e6:.1f}}M params)")
else:
    print("[verify] AutoModelForMaskedLM     -> skipped (encoder-only repo)")

if REF:
    ref = torch.load(REF, weights_only=False)
    with torch.no_grad():
        if EXPECT == "mlm":
            got = model(input_ids=ref["ids"],
                        attention_mask=torch.ones_like(ref["ids"])).logits
            source = "load_masked_lm"
        else:
            got = base(input_ids=ref["ids"],
                       attention_mask=torch.ones_like(ref["ids"])).last_hidden_state
            source = "the NexteraBERT encoder"
    print(f"[verify] activations              -> {{tuple(got.shape)}}")
    if got.shape != ref["logits"].shape:
        raise SystemExit(f"[verify] FAIL: shape {{tuple(got.shape)}} != "
                         f"reference {{tuple(ref['logits'].shape)}}")
    delta = (got - ref["logits"]).abs().max().item()
    print(f"[verify] max |delta| vs {{source}}: {{delta:.2e}}")
    if delta > 1e-4:
        raise SystemExit(
            f"[verify] FAIL: the trust_remote_code model disagrees with {{source}}."
            + (" The most likely cause is that mlm_head.safetensors was not applied, "
               "leaving the prediction head randomly initialised."
               if EXPECT == "mlm" else ""))

if EXPECT == "mlm":
    fm = pipeline("fill-mask", model=model, tokenizer=tok)
    for s in SENTENCES:
        text = s.format(mask=tok.mask_token)
        preds = fm(text, top_k=5)
        shown = ", ".join(f"{{p['token_str'].strip()!r}} {{p['score']:.2%}}"
                          for p in preds)
        print(f"[verify] fill-mask: {{text}}")
        print(f"[verify]         -> {{shown}}")

print("[verify] TRUST_REMOTE_CODE_OK")
'''


def run_remote_checks(repo_id, ref_path, token, expect="mlm") -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "child.py")
        Path(script).write_text(
            CHILD.format(repo=repo_id, ref=ref_path, token=token,
                         sentences=SENTENCES, expect=expect),
            encoding="utf-8")
        env = dict(os.environ)
        # Strip anything that could put the local checkout back on the path.
        env.pop("PYTHONPATH", None)
        proc = subprocess.run([sys.executable, script], cwd=tmp, env=env,
                              capture_output=True, text=True, encoding="utf-8")
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0 or "TRUST_REMOTE_CODE_OK" not in proc.stdout:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"[verify] FAIL: remote-code load failed "
                         f"(exit {proc.returncode})")


def verify(repo_id: str, token: str | None = None, skip_parity: bool = False,
           expect: str = "mlm", local_dir: str | None = None) -> None:
    """Run every check; raises SystemExit on the first failure.

    ``expect="encoder"`` verifies a backbone that has no masked-LM head by design
    (an ELECTRA discriminator): the head is no longer required, the Auto* check stops
    at ``AutoModel``, and parity is measured on the encoder's hidden states.
    """
    if expect not in ("mlm", "encoder"):
        raise SystemExit(f"expect must be 'mlm' or 'encoder', got {expect!r}")
    print(f"[verify] mode: {expect}")
    check_files(repo_id, token, expect, local_dir)

    ref_path = None
    tmp_dir = None
    if not skip_parity:
        tmp_dir = tempfile.mkdtemp()
        ref_path = os.path.join(tmp_dir, "ref.pt")
        reference_logits(repo_id, ref_path, expect)
    else:
        print("[verify] --skip_parity: not comparing against the local implementation")

    try:
        run_remote_checks(repo_id, ref_path, token, expect)
    finally:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    entry = "AutoModelForMaskedLM" if expect == "mlm" else "AutoModel"
    print(f"\n[verify] PASS - https://huggingface.co/{repo_id} loads with "
          f"{entry}.from_pretrained(..., trust_remote_code=True)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--token", default=None,
                   help="HF token (else cached login / HF_TOKEN); needed for a private repo")
    p.add_argument("--skip_parity", action="store_true",
                   help="skip the comparison against the local implementation (the "
                        "only check that needs this repository checked out locally)")
    p.add_argument("--expect", default="mlm", choices=["mlm", "encoder"],
                   help="'mlm' (default) requires a masked-LM head and checks it; "
                        "'encoder' is for a backbone that has none by design - an "
                        "ELECTRA discriminator - and verifies AutoModel instead")
    p.add_argument("--local", default=None,
                   help="the backbone directory that was uploaded. When a check fails, "
                        "this is compared against the repo to say whether the export "
                        "or the upload is at fault. Pipelines pass ${BACKBONE}.")
    args = p.parse_args()
    verify(args.repo_id, args.token, args.skip_parity, args.expect, args.local)


if __name__ == "__main__":
    main()
