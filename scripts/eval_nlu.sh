#!/usr/bin/env bash
# NLU column of ModernBERT's Table 1 (Warner et al., 2024, 3.1.1 + App. E.1): GLUE.
#
# Fine-tunes the pretrained backbone on the 8 standard GLUE tasks with the
# ModernBERT-base recipe in scripts/evaluate_glue.py (per-task lr/wd/epochs from
# their Table 6, MNLI -> RTE/MRPC/STS-B transfer, early stopping, the
# MosaicBERT/ModernBERT 5-seed set with per-task seed caps) and reports the dev
# average that Table 1 calls GLUE (WNLI excluded). ModernBERT-base: 88.4.
#
# Usage:
#   bash scripts/eval_nlu.sh                                  # RikkaBotan/NexteraBERT-Mezzoforte-220M-en
#   MODEL=RikkaBotan/NexteraBERT-Mezzoforte-220M-en NPROC=4 bash scripts/eval_nlu.sh
#   GLUE_TASKS="rte mrpc" GLUE_SEEDS="19" bash scripts/eval_nlu.sh     # quick check
#   GLUE_EXTRA_ARGS="--no_llrd" bash scripts/eval_nlu.sh    # exact published optimiser
#   GLUE_SEARCH=1 bash scripts/eval_nlu.sh                   # their per-task hparam sweep first
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
# ModernBERT searches lr/wd/epochs per task (App. E.1) and reports the winner;
# GLUE_SEARCH=1 runs scripts/search_glue.py (coordinate descent over that grid)
# and feeds its winners into the final multi-seed run. 0 = use their Table 6
# values directly (much cheaper; those values are tuned for ModernBERT-base).
GLUE_SEARCH="${GLUE_SEARCH:-0}"
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

if [ "$GLUE_SEARCH" = "1" ]; then
    SEARCH_OUT="${OUT_DIR}/glue_search.json"
    log "GLUE hyperparameter search (App. E.1 grid) -> ${SEARCH_OUT}"
    # shellcheck disable=SC2086
    "$PY" scripts/search_glue.py \
        --model "$BACKBONE" --tokenizer "$TOKENIZER" \
        --num_gpus "$NPROC" \
        --output "$SEARCH_OUT" --no_final ${GLUE_EXTRA_ARGS}
    GLUE_EXTRA_ARGS="${GLUE_EXTRA_ARGS} --hparams ${SEARCH_OUT}"
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
