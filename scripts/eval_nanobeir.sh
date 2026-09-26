#!/usr/bin/env bash
# NanoBEIR (Zeta Alpha's 13 BEIR subsets: 50 queries and at most 10k documents
# each, nDCG@10) after a shorter MS MARCO stage: the recipe of eval_dpr.sh on the
# first 250k triplets (DPR_BUDGET=lite, see eval_common.sh), scored with the
# settings of the BEIR column of eval_dpr.sh: the same scoring length
# (BEIR_MAX_LEN, 512), batch size (RETRIEVAL_BATCH_SIZE), precision
# (RETRIEVAL_DTYPE) and scorer (scripts/evaluate_retrieval.py through mteb).
#
# The 13 subsets cover every BEIR dataset of BEIR_TASKS_ALL except TREC-COVID and
# CQADupstack (NanoBEIR has none), MSMARCO included; mteb keeps them in a "train"
# split, which is just where the 50 sampled test/dev queries live.
#
# Usage:
#   bash scripts/eval_nanobeir.sh                                     # NexteraBERT-Mezzoforte-220M-en
#   MODEL=answerdotai/ModernBERT-base bash scripts/eval_nanobeir.sh   # a baseline
#   bash scripts/eval_nanobeir_baselines.sh                           # NexteraBERT + the three baselines, one table
#   DPR_MODEL=checkpoints/mezzoforte/dpr_lite bash scripts/eval_nanobeir.sh   # an existing checkpoint
#
# Output: ${OUT_DIR}/nanobeir_dpr_lite.json (checkpoint in ${OUT_DIR}/dpr_lite)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# the 250k-triplet MS MARCO stage of eval_common.sh
export DPR_BUDGET=lite ALLOW_DPR_LITE=1
source scripts/eval_common.sh

NANOBEIR_TASKS="${NANOBEIR_TASKS:-NanoMSMARCORetrieval,NanoFEVERRetrieval,NanoClimateFeverRetrieval,NanoHotpotQARetrieval,NanoDBPediaRetrieval,NanoNQRetrieval,NanoQuoraRetrieval,NanoTouche2020Retrieval,NanoFiQA2018Retrieval,NanoSCIDOCSRetrieval,NanoArguAnaRetrieval,NanoSciFactRetrieval,NanoNFCorpusRetrieval}"
DPR_MODEL="${DPR_MODEL:-}"
# DPR_TAG/DPR_SUBDIR: "_lite" / dpr_lite (eval_common.sh)
NANOBEIR_OUT="${OUT_DIR}/nanobeir_dpr${DPR_TAG}.json"
DPR_ROOT="${OUT_DIR}/${DPR_SUBDIR}"
SELECTED="${DPR_ROOT}/selected.json"

START=$(date +%s)
mkdir -p "$OUT_DIR" "$DPR_ROOT"

# -- the single-vector checkpoint ----------------------------------------------
# DPR_MODEL (explicit) > RETRIEVAL_MODEL (reused checkpoint) > the one recorded in
# ${DPR_ROOT}/selected.json > train one here.
if [ -z "$DPR_MODEL" ]; then
    DPR_MODEL=$(resolve_retrieval_model)
    [ -n "$DPR_MODEL" ] && log "using the contrastive checkpoint: ${DPR_MODEL}"
fi
if [ -z "$DPR_MODEL" ]; then
    BACKBONE=$(resolve_model "$MODEL")
    DPR_MODEL="${DPR_ROOT}/lr${DPR_LR}"
    log "no contrastive checkpoint yet - training one (${DPR_LITE_SAMPLES} triplets, lr=${DPR_LR})"
    train_dpr "$BACKBONE" "$DPR_LR" "$DPR_MODEL"
    [ -f "$SELECTED" ] || record_selected "$SELECTED" "$DPR_LR" "$DPR_MODEL"
fi
if [ ! -f "${DPR_MODEL}/contrastive_run.json" ]; then
    log "WARNING: ${DPR_MODEL} records no contrastive stage - it is not an MS MARCO stage output"
fi

# -- NanoBEIR ------------------------------------------------------------------
# One subset per GPU (NPROC at a time); with a single GPU this is one mteb call.
eval_nanobeir_tasks() {   # <model> <out> [evaluate_retrieval args...]
    local model="$1" out="$2"; shift 2
    eval_retrieval "$model" "$out" "$BEIR_MAX_LEN" "$@"
}
N_TASKS=$(awk -F',' '{print NF}' <<< "$NANOBEIR_TASKS")
log "NanoBEIR (${N_TASKS} subsets, ${RETRIEVAL_DTYPE}, max_len ${BEIR_MAX_LEN}, bs ${RETRIEVAL_BATCH_SIZE}, ${NPROC} GPU(s)) on ${DPR_MODEL} -> ${NANOBEIR_OUT}"
parallel_tasks eval_nanobeir_tasks "$DPR_MODEL" "$NANOBEIR_OUT" "$NANOBEIR_TASKS"

log "NanoBEIR done in $(fmt_elapsed "$START")"
"$PY" scripts/summarize_nanobeir.py "$NANOBEIR_OUT"
