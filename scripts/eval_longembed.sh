#!/usr/bin/env bash
# LongEmbed (Zhu et al., 2024, arXiv:2404.12096; mteb benchmark "LongEmbed") under
# the settings of the long-context column of eval_dpr.sh (MLDR out-of-domain): the
# same single-vector checkpoint (the one eval_dpr.sh selected under
# RETRIEVAL_PROTOCOL, default the st-msmarco DPR stage), no further training, read
# at LONGEMBED_MAX_LEN tokens (default MLDR_MAX_LEN = 8192), the same batch size
# (LONG_RETRIEVAL_BATCH_SIZE, else RETRIEVAL_BATCH_SIZE), precision
# (RETRIEVAL_DTYPE) and scorer (scripts/evaluate_retrieval.py through mteb). Like
# BEIR / MLDR_OOD it is zero-shot: nothing of LongEmbed is trained on.
#
# Six tasks, scored as in the paper and on the mteb leaderboard:
#   NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA   real tasks, nDCG@10
#   Needle, Passkey                                      synthetic, nDCG@1 (= accuracy:
#       one relevant document per query), 50 queries x 100 candidate documents at each
#       context length 256 .. 32768, averaged over the 8 lengths
# and the headline number is the mean of the six. A document longer than
# LONGEMBED_MAX_LEN is truncated, exactly as for MLDR -- so the 16384 / 32768
# splits measure what survives in the first LONGEMBED_MAX_LEN tokens and sit near
# chance when the needle lies beyond them. scripts/summarize_longembed.py
# therefore prints the per-length curve and the mean over the lengths that fit
# next to the paper's 8-length mean. NarrativeQA documents average ~50k words,
# QMSum ~10k, SummScreenFD and 2WikimQA ~6k (paper, Table 1): at 8192 tokens the
# first two are heavily truncated and the last two roughly fit, for every model
# of the comparison alike.
#
# Usage:
#   bash scripts/eval_dpr.sh && bash scripts/eval_longembed.sh   # reuse the selected DPR checkpoint
#   bash scripts/eval_longembed.sh                                # trains one at DPR_LR if none exists
#   MODEL=answerdotai/ModernBERT-base bash scripts/eval_longembed.sh   # a baseline
#   MODEL=LiquidAI/LFM2.5-Encoder-230M LONG_RETRIEVAL_BATCH_SIZE=8 bash scripts/eval_longembed.sh
#   DPR_MODEL=checkpoints/mezzoforte/dpr bash scripts/eval_longembed.sh
#   LONGEMBED_SUBSET=synthetic bash scripts/eval_longembed.sh    # needle + passkey only
#   DPR_BUDGET=lite bash scripts/eval_longembed.sh               # 250k-triplet MS MARCO stage (1/5 the
#       training; own checkpoint dpr_lite/ and results *_dpr_lite.json -- see eval_common.sh)
#   LONGEMBED_MAX_LEN=32768 LONG_RETRIEVAL_BATCH_SIZE=2 bash scripts/eval_longembed.sh   # length extrapolation
#   LONGEMBED_MODE=zeroshot LONGEMBED_SUBSET=synthetic LONGEMBED_MAX_LEN=32768 LONG_RETRIEVAL_BATCH_SIZE=2 \
#       bash scripts/eval_longembed.sh    # the RAW backbone past 8192: no short-text stage, no truncation at 8192
#   bash scripts/eval_longembed_baselines.sh                      # NexteraBERT + the three baselines, one table
#
# Output: ${OUT_DIR}/longembed_dpr.json  (longembed_{synthetic,real}_dpr.json for a
#         LONGEMBED_SUBSET, and a _len<N> suffix when LONGEMBED_MAX_LEN is not
#         8192, so two subsets / lengths never overwrite each other)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# LONGEMBED_MODE picks WHAT is scored (as NANOBEIR_MODE does):
#   dpr (default)  the single-vector checkpoint of the MS MARCO stage. That stage updates
#                  every weight on ~80-token passages only, so what it scores is "the
#                  backbone after a short-text fine-tuning" -- a long-context property of
#                  the pretrained model (its MLM loss at 8192+) may or may not survive it.
#   zeroshot       the raw pretrained backbone, masked mean pooling, nothing trained:
#                  the long-context behaviour of the pretraining itself, through the
#                  repo's own encoders (bf16 autocast, NeoBERT / LFM2.5 load repairs).
#                  Needle / passkey stay meaningful without a contrastive stage: the
#                  candidates of one length share their filler, so even a context-free
#                  mean of token vectors solves passkey (scripts/diagnose_longembed.py
#                  --model static-mean) -- a backbone can only lose by breaking at length.
#                  The four real tasks do need the stage; expect floor-level nDCG there.
# Combine with LONGEMBED_MAX_LEN=32768 to see anything past 8192: at the default 8192
# every model is truncated alike and the 16384 / 32768 splits cannot separate them.
LONGEMBED_MODE="${LONGEMBED_MODE:-dpr}"
case "$LONGEMBED_MODE" in
    zeroshot) export NO_CONTRASTIVE_STAGE=1 ;;   # eval_common.sh: no trainer needed
    dpr) ;;
    *) echo "ERROR: LONGEMBED_MODE must be dpr or zeroshot (got ${LONGEMBED_MODE})" >&2; exit 1 ;;
esac
export ALLOW_DPR_LITE=1     # DPR_BUDGET=lite is accepted here (eval_common.sh)
source scripts/eval_common.sh

# LONGEMBED_SUBSET: all (default) | synthetic (needle + passkey only: 2 x 8 lengths x
# 100 short-to-32k documents, minutes) | real (the four QA / summarisation tasks).
# A subset writes its own results file, so it never blocks or shadows a full run;
# the mteb cache is per task, so a later full run reuses what a subset scored.
LONGEMBED_SUBSET="${LONGEMBED_SUBSET:-all}"
# largest first: NarrativeQA (355 documents of ~50k words) dominates the wall clock
_LEMB_REAL="LEMBNarrativeQARetrieval,LEMBQMSumRetrieval,LEMBSummScreenFDRetrieval,LEMBWikimQARetrieval"
_LEMB_SYNTHETIC="LEMBNeedleRetrieval,LEMBPasskeyRetrieval"
case "$LONGEMBED_SUBSET" in
    all)       _LEMB_DEFAULT="${_LEMB_REAL},${_LEMB_SYNTHETIC}"; _LEMB_STEM="longembed_dpr" ;;
    synthetic) _LEMB_DEFAULT="$_LEMB_SYNTHETIC"; _LEMB_STEM="longembed_synthetic_dpr" ;;
    real)      _LEMB_DEFAULT="$_LEMB_REAL"; _LEMB_STEM="longembed_real_dpr" ;;
    *) echo "ERROR: LONGEMBED_SUBSET must be all, synthetic or real (got ${LONGEMBED_SUBSET})" >&2; exit 1 ;;
esac
LONGEMBED_TASKS="${LONGEMBED_TASKS:-$_LEMB_DEFAULT}"
LONGEMBED_MAX_LEN="${LONGEMBED_MAX_LEN:-$MLDR_MAX_LEN}"
DPR_MODEL="${DPR_MODEL:-}"
DPR_ROOT="${OUT_DIR}/${DPR_SUBDIR}"     # dpr, or dpr_lite under DPR_BUDGET=lite
SELECTED="${DPR_ROOT}/selected.json"
if [ "$LONGEMBED_MODE" = "zeroshot" ]; then
    _LEMB_STEM="${_LEMB_STEM%_dpr}_zeroshot"   # longembed[_subset]_zeroshot
else
    _LEMB_STEM="${_LEMB_STEM}${DPR_TAG}"     # ..._dpr_lite: a lite checkpoint's scores stay apart
fi
if [ "$LONGEMBED_MAX_LEN" = "8192" ]; then
    LONGEMBED_OUT="${OUT_DIR}/${_LEMB_STEM}.json"
else
    LONGEMBED_OUT="${OUT_DIR}/${_LEMB_STEM}_len${LONGEMBED_MAX_LEN}.json"
fi
# mteb keys its result cache by model name + task, NOT by the scoring length: a
# second run at another LONGEMBED_MAX_LEN would be served the first one's scores.
MTEB_CACHE="${MTEB_CACHE}/longembed_len${LONGEMBED_MAX_LEN}"

START=$(date +%s)
mkdir -p "$OUT_DIR"
EXTRA_ARGS=()

if [ "$LONGEMBED_MODE" = "zeroshot" ]; then
    # -- the raw backbone --------------------------------------------------------
    # --pooling mean even if the directory carries a SimCSE head: nothing but the
    # pretrained weights takes part.
    DPR_MODEL=$(resolve_model "$MODEL")
    EXTRA_ARGS=(--pooling mean)
    log "zero-shot: raw backbone ${DPR_MODEL}, mean pooling, no contrastive stage"
else
mkdir -p "$DPR_ROOT"
# -- the single-vector checkpoint --------------------------------------------
# DPR_MODEL (explicit) > RETRIEVAL_MODEL (reused checkpoint) > the one eval_dpr.sh
# selected > train one here under RETRIEVAL_PROTOCOL. Same order as eval_code.sh /
# eval_nanobeir.sh, so BEIR, MLDR, CoIR and LongEmbed all score one checkpoint.
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
    log "WARNING: ${DPR_MODEL} records no contrastive stage - scores are a raw-backbone floor, not comparable to published LongEmbed numbers"
fi
fi   # LONGEMBED_MODE

# -- LongEmbed -----------------------------------------------------------------
# One task per GPU (NPROC at a time); with a single GPU this is one mteb call.
# --metric main_score: nDCG@10 for the real tasks, nDCG@1 for needle / passkey.
eval_longembed_tasks() {   # <model> <out> [evaluate_retrieval args...]
    local model="$1" out="$2"; shift 2
    eval_retrieval "$model" "$out" "$LONGEMBED_MAX_LEN" --metric main_score "$@"
}
N_TASKS=$(awk -F',' '{print NF}' <<< "$LONGEMBED_TASKS")
log "LongEmbed (${N_TASKS} tasks, max_len ${LONGEMBED_MAX_LEN}, bs ${LONG_RETRIEVAL_BATCH_SIZE:-$RETRIEVAL_BATCH_SIZE}, ${RETRIEVAL_DTYPE}, ${NPROC} GPU(s)) on ${DPR_MODEL} -> ${LONGEMBED_OUT}"
parallel_tasks eval_longembed_tasks "$DPR_MODEL" "$LONGEMBED_OUT" "$LONGEMBED_TASKS" "${EXTRA_ARGS[@]}"

log "LongEmbed done in $(fmt_elapsed "$START")"
"$PY" scripts/summarize_longembed.py "$LONGEMBED_OUT"
