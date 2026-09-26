#!/usr/bin/env bash
# Single-vector retrieval after the MS MARCO stage: BEIR and MultiLongDocRetrieval
# (MLDR).
#
# 1. The contrastive stage (scripts/eval_common.sh): ModernBERT's DPR stage, run with
#    sentence-transformers as a port of their public examples/train_st.py
#    (scripts/train_st_dpr.py): mean pooling, msmarco-co-condenser triplet-hard
#    1.25M, Cached MNRL at per-device batch 512, 5% warmup, 1 epoch, lr 8e-5.
#    RETRIEVAL_MODEL=<dir> reuses an existing checkpoint and skips this stage.
# 2. BEIR (15 datasets, nDCG@10; MSMARCO scored on dev as in the original
#    suite) on that checkpoint.                            -> beir_dpr.json
# 3. MLDR: the same checkpoint, no further training, on the English MLDR test
#    split read at 8192 tokens.                            -> mldr_ood_dpr.json
#
# Usage:
#   bash scripts/eval_dpr.sh                                    # lr 8e-5
#   MODEL=LiquidAI/LFM2.5-Encoder-230M LONG_RETRIEVAL_BATCH_SIZE=16 \
#       bash scripts/eval_dpr.sh                                # a baseline: BEIR + MLDR
#       (train_st_dpr.py repairs LFM's checkpoint layout first -- a plain AutoModel
#        load, which is what sentence-transformers does, yields RANDOM weights)
#   DPR_EXTRA_ARGS="--max_samples 100000" bash scripts/eval_dpr.sh   # smoke run
#
# Every stage skips work whose output already exists (FORCE=1 to redo), so an
# interrupted run resumes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

DPR_ROOT="${OUT_DIR}/dpr"
SELECTED="${DPR_ROOT}/selected.json"
BEIR_OUT="${OUT_DIR}/beir_dpr.json"
MLDR_OOD_OUT="${OUT_DIR}/mldr_ood_dpr.json"

START=$(date +%s)
mkdir -p "$DPR_ROOT"
BACKBONE=$(resolve_model "$MODEL")
log "IR (DPR): model=${BACKBONE} lr=${DPR_LR} bs=${DPR_BATCH_SIZE:-512} gpus=${NPROC}"

# -- 1. train -------------------------------------------------------------------
if [ -n "$RETRIEVAL_MODEL" ]; then
    BEST=$(resolve_retrieval_model)
    BEST_LR=$(run_json_field "${BEST}/contrastive_run.json" lr)
    log "reusing contrastive checkpoint ${BEST} (protocol $(run_json_field "${BEST}/contrastive_run.json" protocol), lr ${BEST_LR})"
else
    BEST_LR="$DPR_LR"
    BEST="${DPR_ROOT}/lr${BEST_LR}"
    train_dpr "$BACKBONE" "$BEST_LR" "$BEST"
fi
record_selected "$SELECTED" "$BEST_LR" "$BEST"
log "single-vector checkpoint: ${BEST} (recorded in ${SELECTED})"

# -- 2. BEIR -------------------------------------------------------------------
# One dataset per GPU (NPROC at a time); with a single GPU this is one mteb call.
eval_retrieval_tasks() {   # <model> <out> [evaluate_retrieval args...]
    local model="$1" out="$2"; shift 2
    eval_retrieval "$model" "$out" "$BEIR_MAX_LEN" "$@"
}
log "BEIR (15 datasets, max_len ${BEIR_MAX_LEN}, ${NPROC} GPU(s)) -> ${BEIR_OUT}"
parallel_tasks eval_retrieval_tasks "$BEST" "$BEIR_OUT" "$BEIR_TASKS_ALL"

# -- 3. MLDR -------------------------------------------------------------------
log "MLDR (English test, max_len ${MLDR_MAX_LEN}) -> ${MLDR_OOD_OUT}"
eval_retrieval "$BEST" "$MLDR_OOD_OUT" "$MLDR_MAX_LEN" "${MLDR_TASK_ARGS[@]}"

log "IR (DPR) done in $(fmt_elapsed "$START")"
for f in "$BEIR_OUT" "$MLDR_OOD_OUT"; do
    [ -f "$f" ] && log "  $(basename "$f"): $("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d.get('average_ndcg@10', float('nan'))*100:.2f} nDCG@10\")" "$f")"
done
exit 0
