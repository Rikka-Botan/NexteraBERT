#!/usr/bin/env bash
# NanoBEIR (Zeta Alpha's 13 BEIR subsets: 50 queries and at most 10k documents
# each, nDCG@10), scored with the settings of the BEIR column of eval_dpr.sh: the
# same scoring length (BEIR_MAX_LEN, 512), batch size (RETRIEVAL_BATCH_SIZE) and
# scorer (scripts/evaluate_retrieval.py through mteb).
#
# NANOBEIR_MODE picks WHAT is scored:
#   zeroshot (default)  the raw pretrained backbone, masked mean pooling, L2-normalised
#                       -- no contrastive stage at all, nothing is trained. Runs in
#                       minutes and ranks the pretrained representations themselves,
#                       through the repo's own encoders (NexteraEncoder / HFEncoder,
#                       bf16 autocast, the NeoBERT / LFM2.5 load repairs included).
#                       An MLM backbone was never taught that a query and its passage
#                       belong together, so these numbers sit far below BEIR-after-DPR
#                       ones: compare them across models, not to published BEIR rows.
#   dpr                 the single-vector checkpoint eval_dpr.sh selected under
#                       RETRIEVAL_PROTOCOL (default the st-msmarco DPR stage; one is
#                       trained at DPR_LR if none exists), in RETRIEVAL_DTYPE. A
#                       minutes-long proxy of that model's beir_dpr.json. Zero-shot only
#                       in BEIR's sense (MS MARCO-trained; NanoMSMARCO is in-domain).
#
# The 13 subsets cover every BEIR dataset of BEIR_TASKS_ALL except TREC-COVID and
# CQADupstack (NanoBEIR has none), MSMARCO included; mteb keeps them in a "train"
# split, which is just where the 50 sampled test/dev queries live.
#
# Usage:
#   bash scripts/eval_nanobeir.sh                                     # zero-shot, NexteraBERT-Mezzoforte-220M-en
#   MODEL=answerdotai/ModernBERT-base bash scripts/eval_nanobeir.sh   # zero-shot, a baseline
#   bash scripts/eval_nanobeir_baselines.sh                           # NexteraBERT + the three baselines, one table
#   NANOBEIR_MODE=dpr bash scripts/eval_nanobeir.sh                   # the DPR checkpoint instead
#   NANOBEIR_MODE=dpr DPR_MODEL=checkpoints/mezzoforte/dpr bash scripts/eval_nanobeir.sh
#   NANOBEIR_MODE=dpr DPR_BUDGET=lite bash scripts/eval_nanobeir.sh   # 250k-triplet stage, 1/5 the training
#       (own checkpoint dpr_lite/, results nanobeir_dpr_lite.json -- see eval_common.sh)
#
# Output: ${OUT_DIR}/nanobeir_zeroshot.json  (NANOBEIR_MODE=dpr: nanobeir_dpr.json)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NANOBEIR_MODE="${NANOBEIR_MODE:-zeroshot}"
case "$NANOBEIR_MODE" in
    zeroshot) export NO_CONTRASTIVE_STAGE=1 ALLOW_DPR_LITE=1 ;;   # eval_common.sh: no trainer needed
    dpr) export ALLOW_DPR_LITE=1 ;;              # DPR_BUDGET=lite is accepted here
    *) echo "ERROR: NANOBEIR_MODE must be zeroshot or dpr (got ${NANOBEIR_MODE})" >&2; exit 1 ;;
esac
source scripts/eval_common.sh

NANOBEIR_TASKS="${NANOBEIR_TASKS:-NanoMSMARCORetrieval,NanoFEVERRetrieval,NanoClimateFeverRetrieval,NanoHotpotQARetrieval,NanoDBPediaRetrieval,NanoNQRetrieval,NanoQuoraRetrieval,NanoTouche2020Retrieval,NanoFiQA2018Retrieval,NanoSCIDOCSRetrieval,NanoArguAnaRetrieval,NanoSciFactRetrieval,NanoNFCorpusRetrieval}"
DPR_MODEL="${DPR_MODEL:-}"
# DPR_TAG/DPR_SUBDIR: "_lite" / dpr_lite under DPR_BUDGET=lite (dpr mode only)
if [ "$NANOBEIR_MODE" = "dpr" ]; then NANOBEIR_OUT="${OUT_DIR}/nanobeir_dpr${DPR_TAG}.json"
else NANOBEIR_OUT="${OUT_DIR}/nanobeir_${NANOBEIR_MODE}.json"; fi
DPR_ROOT="${OUT_DIR}/${DPR_SUBDIR}"
SELECTED="${DPR_ROOT}/selected.json"

START=$(date +%s)
mkdir -p "$OUT_DIR"
EXTRA_ARGS=()

if [ "$NANOBEIR_MODE" = "zeroshot" ]; then
    # -- the raw backbone ------------------------------------------------------
    # --pooling mean even if the directory carries a SimCSE head: zero-shot means
    # nothing but the pretrained weights takes part.
    SCORED=$(resolve_model "$MODEL")
    EXTRA_ARGS=(--pooling mean)
    DESC="zero-shot, raw backbone, mean pooling"
else
    # -- the single-vector checkpoint ------------------------------------------
    # DPR_MODEL (explicit) > RETRIEVAL_MODEL (reused checkpoint) > the one eval_dpr.sh
    # selected > train one here under RETRIEVAL_PROTOCOL. Same order as eval_code.sh,
    # so NanoBEIR, BEIR, MLDR and CoIR all score one checkpoint.
    mkdir -p "$DPR_ROOT"
    if [ -z "$DPR_MODEL" ]; then
        DPR_MODEL=$(resolve_retrieval_model)
        [ -n "$DPR_MODEL" ] && log "using the contrastive checkpoint: ${DPR_MODEL}"
    fi
    if [ -z "$DPR_MODEL" ]; then
        BACKBONE=$(resolve_model "$MODEL")
        DPR_MODEL="${DPR_ROOT}/lr${DPR_LR}"
        log "no contrastive checkpoint yet - training one (${RETRIEVAL_PROTOCOL}, lr=${DPR_LR}); eval_dpr.sh runs the sweep"
        train_dpr "$BACKBONE" "$DPR_LR" "$DPR_MODEL"
        [ -f "$SELECTED" ] || record_selected "$SELECTED" "$DPR_LR" "$DPR_MODEL" "$DPR_LR"
    fi
    if [ ! -f "${DPR_MODEL}/contrastive_run.json" ]; then
        log "WARNING: ${DPR_MODEL} records no contrastive stage - this is a zero-shot score filed as dpr"
    fi
    SCORED="$DPR_MODEL"
    DESC="${RETRIEVAL_PROTOCOL} checkpoint, ${RETRIEVAL_DTYPE}"
fi

# -- NanoBEIR ------------------------------------------------------------------
# One subset per GPU (NPROC at a time); with a single GPU this is one mteb call.
eval_nanobeir_tasks() {   # <model> <out> [evaluate_retrieval args...]
    local model="$1" out="$2"; shift 2
    eval_retrieval "$model" "$out" "$BEIR_MAX_LEN" "$@"
}
N_TASKS=$(awk -F',' '{print NF}' <<< "$NANOBEIR_TASKS")
log "NanoBEIR (${N_TASKS} subsets, ${DESC}, max_len ${BEIR_MAX_LEN}, bs ${RETRIEVAL_BATCH_SIZE}, ${NPROC} GPU(s)) on ${SCORED} -> ${NANOBEIR_OUT}"
parallel_tasks eval_nanobeir_tasks "$SCORED" "$NANOBEIR_OUT" "$NANOBEIR_TASKS" "${EXTRA_ARGS[@]}"

log "NanoBEIR done in $(fmt_elapsed "$START")"
"$PY" scripts/summarize_nanobeir.py "$NANOBEIR_OUT"
