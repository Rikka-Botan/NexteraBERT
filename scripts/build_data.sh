#!/usr/bin/env bash
# Build the pretokenized pretraining shards (1024 + 8192 tokens) plus held-out eval
# shards, and upload them to Hugging Face. OptiBERT-style recipe
# (aclanthology.org/2025.emnlp-main.1804): ModernBERT BPE tokenizer, FineWeb-Edu data,
# a 1024-token main phase, followed by a brief 8192 long-context extension slice
# (ModernBERT-style phase 2).
#
# Corpus (~121 B tokens, NO duplication between sources or the eval split):
#
#     FineWeb-Edu    90.25%  (~109.4 B)     DCLM baseline   4.75%  (~5.76 B)
#     StarCoderData code 5%  (~6.05 B: python/java/cpp/javascript)
#
# Packing: ONE DOCUMENT PER EXAMPLE, variable length (`--pack_mode docs`). Each shard
# is a FLAT token stream (documents back-to-back, each [CLS] doc [SEP], NO padding)
# plus a <shard>.lens sidecar of per-document lengths. Padding is NOT baked in: the
# training loader (VarLengthDataset + dynamic-pad collate) pads each BATCH to its
# longest example at model-input time, so NexteraHRA sees exactly one document per
# sequence (block-pooling never straddles documents) with near-zero wasted padding.
#
# Disjoint slices, carved from each source's own document stream by round-robin
# bucket (`--holdout_mod 100`, deterministic by doc index, so no source is shared):
#     train (1024):  buckets [0,99)   -> ~99% of the budget (~120 B tokens)
#     eval  (1024):  bucket  [99,100) -> tiny held-out set, SAME 90.25/4.75/5 distribution
#
# Phase 2 (long-context) slices use their OWN source list - the SAME ~95% text +
# 5% code recipe re-read from the same sources (StarCoderData is the
# permissively-licensed corpus Stack-Edu is filtered from; Stack-Edu itself ships
# only blob_ids, not code text, so it cannot be streamed here) - but packed as
# FIXED 8192-token blocks with `--pack_mode unpad`
# (documents packed densely, each [CLS] doc [SEP] segment aligned to 8 tokens so
# NexteraHRA block-pooling never straddles two documents). Fixed full-length blocks
# are what give the model real long-range attention distances; code files are also
# naturally long, which suits this phase.
#     train (8192):  buckets [0,99)   -> ~10 B tokens (90.25/4.75/5 web-edu/dclm/code)
#     eval  (8192):  bucket  [99,100) -> tiny held-out set, same mix
#
# Run this ONCE on a machine with good network. run_pipeline_bert.sh downloads the
# finished shards from HF instead of building them; the shards used for the
# released models are published as RikkaBotan/NexteraBERT-data-mix.
#
# Usage:
#   bash scripts/build_data.sh
#   DATA_REPO_ID=<user>/NexteraBERT-data-mix bash scripts/build_data.sh   # + upload
set -euo pipefail

###############################################################################
# Settings
###############################################################################

HF_TOKEN="${HF_TOKEN:-}"                            # export HF_TOKEN in your shell/CI (or huggingface-cli login)
DATA_REPO_ID="${DATA_REPO_ID:-}"                    # HF dataset repo to upload the shards to; empty = keep them in data/
TOKENIZER="answerdotai/ModernBERT-base"             # ModernBERT BPE (vocab 50,368)

# Corpus: "name|config|budget_B|data_dir|extra_json". budget_B is the source's
# TOTAL token budget across the slices that use the list (billions). data_dir is
# for repos organised by subdirectory instead of configs (e.g. starcoderdata's
# per-language dirs). All sources are script-free (Parquet).
# Phase-1 (1024) mix: 90.25% FineWeb-Edu + 4.75% DCLM + 5% code from
# bigcode/starcoderdata ("content" column -> renamed to "text"). Budgets total
# 121.21 B so the 0.99 train slice below is exactly 120 B. FineWeb-Edu needs the
# sample-350BT config (not 100BT): its budget alone is ~109 B tokens.
# NOTE: starcoderdata is GATED - accept the terms once with the HF account behind
# HF_TOKEN: https://huggingface.co/datasets/bigcode/starcoderdata
DATASETS=(
  "HuggingFaceFW/fineweb-edu|sample-350BT|109.4||"
  "mlfoundations/dclm-baseline-1.0-parquet||5.76||"
  'bigcode/starcoderdata||2.42|python|,"text_column":"content"'
  'bigcode/starcoderdata||1.21|java|,"text_column":"content"'
  'bigcode/starcoderdata||1.21|cpp|,"text_column":"content"'
  'bigcode/starcoderdata||1.21|javascript|,"text_column":"content"'
)

# Phase-2 (8192) mix: the SAME 90.25/4.75/5 recipe as phase 1, re-read from the
# same sources. Budgets total 10.10 B so the 0.99 train slice is exactly 10 B.
DATASETS_LONG=(
  "HuggingFaceFW/fineweb-edu|sample-350BT|9.12||"
  "mlfoundations/dclm-baseline-1.0-parquet||0.48||"
  'bigcode/starcoderdata||0.20|python|,"text_column":"content"'
  'bigcode/starcoderdata||0.10|java|,"text_column":"content"'
  'bigcode/starcoderdata||0.10|cpp|,"text_column":"content"'
  'bigcode/starcoderdata||0.10|javascript|,"text_column":"content"'
)

# Slices: "seq|frac|keep_buckets|outfile|pack_mode|srcset". frac = fraction of each
# source's budget (in blocks) this slice builds; keep_buckets = 'lo:hi' half-open
# bucket range; pack_mode empty = the PACK_MODE default below; srcset 'long' uses
# DATASETS_LONG (the phase-2 mix), anything else uses DATASETS.
PHASES=(
  "1024|0.99|0:99|data/mix_1024.bin||"
  "1024|0.001|99:100|data/mix_eval_1024.bin||"
  "8192|0.99|0:99|data/mix_8192.bin|unpad|long"
  "8192|0.001|99:100|data/mix_eval_8192.bin|unpad|long"
)
HOLDOUT_MOD=100
PACK_MODE=docs          # one document per example, variable length (dynamic-pad at collate)
UNPAD_ALIGN=8           # used by the 8192 unpad slices (2x NexteraHRA block_size 4)

PARTS_DIR="data/parts"  # per-source packed sub-shards, concatenated per slice

# Existing outputs are trusted BY FILENAME (a finished part / concatenated shard is
# never rebuilt). After changing budgets or the mix above, delete data/parts and the
# data/mix_*.bin shards first, or stale token counts will be reused silently.

###############################################################################

if command -v python3 &>/dev/null; then PY=python3
elif command -v python &>/dev/null;  then PY=python
else echo "ERROR: python not found" >&2; exit 1
fi
PIP="${PY} -m pip"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

$PIP install -e ".[train]" zstandard --quiet
# gigatoken: Rust SIMD BPE, ~50x the tokenizers engine. Best-effort - if the
# install fails, prepare_data's 'auto' backend falls back to tokenizers; when
# present it is only used after passing an exact-parity self-test, so shards
# are byte-identical either way.
$PIP install gigatoken --quiet || true

if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
  # older datasets/huggingface_hub versions only honour the legacy variable name
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# -- Pre-flight: HF auth + gated-dataset access -------------------------------
# The phase-2 mix streams bigcode/starcoderdata, which is GATED. Verify BOTH
# conditions up front - (1) a token is available, (2) the terms were accepted
# with that account - so the build fails here in seconds with instructions,
# not hours in with a DatasetNotFoundError mid-stream.
log "Checking Hugging Face authentication and gated-dataset access ..."
$PY - <<'PYEOF'
import sys
from huggingface_hub import HfApi

GATED = "bigcode/starcoderdata"
api = HfApi()
try:
    user = api.whoami()["name"]
except Exception:
    sys.exit(
        "ERROR: not authenticated to Hugging Face - no token found.\n"
        "  export HF_TOKEN=<read token from https://huggingface.co/settings/tokens>\n"
        "  (or run: huggingface-cli login) and re-run this script.")
try:
    api.dataset_info(GATED)
except Exception as e:
    sys.exit(
        f"ERROR: account '{user}' has no access to the gated dataset {GATED} "
        f"({type(e).__name__}).\n"
        f"  Accept the terms ONCE while logged in as '{user}':\n"
        f"  https://huggingface.co/datasets/{GATED}\n"
        f"  then re-run this script.")
print(f"HF auth OK (user: {user}); access to {GATED} OK")
PYEOF

mkdir -p data "$PARTS_DIR"

# -- Build one per-source packed sub-shard -----------------------------------
build_part() {
  local out="$1" seq="$2" blocks="$3" keep="$4" mix="$5" label="$6" pm="$7"
  if [ -f "$out" ] && [ -f "${out}.meta" ]; then
    log "  ${label}: already exists (${out})"
    return
  fi
  # Do NOT wipe a partial <out>/<out>.progress: prepare_data resumes from its
  # checkpoint (or starts fresh if params changed), so a crash continues.
  if [ -f "${out}.progress" ]; then
    log "  ${label}: resuming from checkpoint ..."
  else
    log "  ${label}: building (seq=${seq}, <=${blocks} blocks, keep=${keep}) ..."
  fi
  # Pre-tokenisation is pure CPU: hide the GPU so torch never inits a CUDA context.
  CUDA_VISIBLE_DEVICES="" $PY scripts/prepare_data.py \
        --dataset_mix "$mix" --tokenizer "$TOKENIZER" \
        --holdout_mod "$HOLDOUT_MOD" --holdout_keep "$keep" \
        --pack_mode "$pm" --unpad_align "$UNPAD_ALIGN" \
        --max_seq_len "$seq" --max_blocks "$blocks" \
        --output "$out" || true
  # prepare_data writes <out>.meta last and removes <out>.progress only on full
  # completion, so (.meta present AND no .progress) is an authoritative success
  # marker - trust it even if the interpreter crashed during shutdown.
  if [ -f "$out" ] && [ -f "${out}.meta" ] && [ ! -f "${out}.progress" ]; then
    return
  fi
  # An interrupted or failed build is RESUMABLE state, not junk: keep the partial
  # shard + .progress so the next invocation continues from the checkpoint.
  # (prepare_data itself ignores a checkpoint whose build parameters changed and
  # starts over cleanly, so stale state never needs deleting here. A previous
  # version of this script rm -f'd the shard AND the checkpoint on any non-zero
  # exit - including Ctrl-C / kill - throwing away days of packing.)
  log "ERROR: ${label} build did not complete; partial state kept - re-run to resume"
  exit 1
}

# -- Build every (slice x source) sub-shard, then concatenate per slice -------
for phase in "${PHASES[@]}"; do
  IFS='|' read -r seq frac keep outfile pm srcset <<< "$phase"
  pm="${pm:-$PACK_MODE}"
  if [ "$srcset" = "long" ]; then
    srcs=("${DATASETS_LONG[@]}")
  else
    srcs=("${DATASETS[@]}")
  fi

  parts=()
  for spec in "${srcs[@]}"; do
    IFS='|' read -r name config budget datadir extra <<< "$spec"

    # per-source block budget for this slice: budget_B * 1e9 * frac / seq
    blocks=$($PY -c "import sys;print(int(float(sys.argv[1])*1e9*float(sys.argv[2])/int(sys.argv[3])))" \
             "$budget" "$frac" "$seq")

    # single-entry mix JSON (omit "config"/"data_dir" when unset)
    fields="\"name\":\"${name}\",\"weight\":1"
    if [ -n "$config" ]; then
      fields="${fields},\"config\":\"${config}\""
    fi
    if [ -n "$datadir" ]; then
      fields="${fields},\"data_dir\":\"${datadir}\""
    fi
    mix="[{${fields}${extra}}]"

    slug=$(echo "${name}${datadir:+_$datadir}" | tr '/' '_')
    part="${PARTS_DIR}/$(basename "${outfile%.bin}")__${slug}.bin"
    build_part "$part" "$seq" "$blocks" "$keep" "$mix" "${name}${datadir:+/$datadir} (seq=${seq})" "$pm"
    parts+=("$part")
  done

  # Concatenate the per-source sub-shards into the slice shard.
  if [ -f "$outfile" ] && [ -f "${outfile}.meta" ]; then
    log "${outfile}: already concatenated"
  elif [ "$pm" = "docs" ]; then
    # variable-length: concat the flat token streams AND the .lens sidecars, then
    # write a merged .meta (summed n_docs / total_tokens). Offsets are derived from
    # .lens at load time, so a byte-wise concat of both files stays consistent.
    log "Concatenating ${#parts[@]} varlen sources -> ${outfile} (max_seq_len=${seq}) ..."
    cat "${parts[@]}" > "$outfile"
    cat "${parts[@]/%/.lens}" > "${outfile}.lens"
    $PY - "$outfile" "$seq" "${parts[@]}" <<'PYEOF'
import json, sys
out, seq = sys.argv[1], int(sys.argv[2])
parts = sys.argv[3:]
n_docs = total = 0
pad = dtype = None
for p in parts:
    m = json.load(open(p + ".meta", encoding="utf-8"))
    n_docs += m["n_docs"]; total += m["total_tokens"]
    pad, dtype = m["pad_token_id"], m["dtype"]
json.dump({"format": "varlen", "max_seq_len": seq, "pad_token_id": pad,
           "dtype": dtype, "n_docs": n_docs, "total_tokens": total},
          open(out + ".meta", "w", encoding="utf-8"))
print(f"  merged: {n_docs:,} docs / {total:,} tokens")
PYEOF
    log "${outfile}: $(( $(wc -c < "$outfile") / 2 )) tokens (varlen, one doc per example)"
  else
    # fixed-block: byte-wise concat (identical seq_len/dtype/pad); reuse one .meta.
    log "Concatenating ${#parts[@]} sources -> ${outfile} (seq=${seq}) ..."
    cat "${parts[@]}" > "$outfile"
    cp "${parts[0]}.meta" "${outfile}.meta"
    log "${outfile}: $(( $(wc -c < "$outfile") / seq / 2 )) blocks of ${seq} tokens"
  fi
done

# -- Upload to HF ------------------------------------------------------------
if [ -z "$DATA_REPO_ID" ]; then
  log "DATA_REPO_ID is empty - skipping upload (shards are in data/)"
  exit 0
fi

log "Uploading shards to ${DATA_REPO_ID} ..."
$PY - "$DATA_REPO_ID" <<'PYEOF'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

import os
files = []
for shard in ["data/mix_1024.bin", "data/mix_eval_1024.bin",
              "data/mix_8192.bin", "data/mix_eval_8192.bin"]:
    files += [shard, shard + ".meta"]
    if os.path.exists(shard + ".lens"):     # variable-length shards carry a .lens sidecar
        files.append(shard + ".lens")
for name in files:
    print(f"  uploading {name} ...")
    api.upload_file(path_or_fileobj=name, path_in_repo=name,
                    repo_id=repo_id, repo_type="dataset")
print("done")
PYEOF

log "Upload complete: https://huggingface.co/datasets/${DATA_REPO_ID}"
