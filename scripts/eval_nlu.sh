#!/usr/bin/env bash
# GLUE: fine-tunes the pretrained backbone on the 8 standard GLUE tasks with the
# per-task recipe in scripts/evaluate_glue.py (layer-wise LR decay 0.9, MNLI ->
# RTE/MRPC/STS-B/QNLI transfer, early stopping, the MosaicBERT/ModernBERT 5-seed set
# with per-task seed caps) and reports the dev average (WNLI excluded).
#
# Usage:
#   bash scripts/eval_nlu.sh                                  # RikkaBotan/NexteraBERT-Mezzoforte-220M-en
#   MODEL=RikkaBotan/NexteraBERT-Mezzoforte-220M-en NPROC=4 bash scripts/eval_nlu.sh
#   GLUE_TASKS="rte mrpc" GLUE_SEEDS="19" bash scripts/eval_nlu.sh     # quick check
#   MODEL=LiquidAI/LFM2.5-Encoder-230M bash scripts/eval_nlu.sh   # a Hugging Face baseline:
#       same recipe, optimiser and head over its AutoModel backbone (hf_baselines.py)
#
# Output: ${OUT_DIR}/glue_results.json  (glue_avg + per-task mean/std over seeds)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

GLUE_TASKS="${GLUE_TASKS:-all}"
GLUE_SEEDS="${GLUE_SEEDS:-19 8364 717 10536 90166}"
GLUE_EXTRA_ARGS="${GLUE_EXTRA_ARGS:-}"
GLUE_OUT="${OUT_DIR}/glue_results.json"

START=$(date +%s)
mkdir -p "$OUT_DIR"
BACKBONE=$(resolve_model "$MODEL")
# a Hugging Face baseline (MODEL=LiquidAI/LFM2.5-Encoder-230M, ...): same recipe
# and head, its own tokenizer unless the caller chose one
TOKENIZER=$(tokenizer_for "$BACKBONE")
log "NLU / GLUE: model=${BACKBONE} tasks=[${GLUE_TASKS}] seeds=[${GLUE_SEEDS}] gpus=${NPROC}"

if done_or_skip "$GLUE_OUT"; then exit 0; fi

# One task per GPU, tasks concurrent (NOT DDP-sharded): the recipe's batch sizes
# are totals and DDP would multiply them.
PARALLEL_ARGS=()
if [ "$NPROC" -gt 1 ]; then
    PARALLEL_ARGS=(--task_parallel --num_gpus "$NPROC")
fi

# shellcheck disable=SC2086
"$PY" scripts/evaluate_glue.py \
    --model "$BACKBONE" --tokenizer "$TOKENIZER" \
    --tasks ${GLUE_TASKS} \
    --seeds ${GLUE_SEEDS} \
    "${PARALLEL_ARGS[@]}" \
    --output "$GLUE_OUT" \
    ${GLUE_EXTRA_ARGS}

log "GLUE done in $(fmt_elapsed "$START"); results -> ${GLUE_OUT}"
