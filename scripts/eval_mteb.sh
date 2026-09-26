#!/usr/bin/env bash
# MTEB(eng, v2) under the OptiBERT protocol (Dervishi et al., EMNLP 2025, App.
# D.2) for ONE model: attentive pooling + supervised SimCSE on MNLI+SNLI
# (scripts/finetune_contrastive.py --protocol mteb-nli), then the full English
# MTEB v2 suite (scripts/evaluate_mteb.py). Works for a NexteraBERT checkpoint
# AND for a Hugging Face baseline -- ModernBERT, NeoBERT, LFM2.5-Encoder -- so every row
# of the comparison goes through the identical stage and scorer (see
# src/nexterabert/hf_baselines.py for why that matters).
#
# Usage:
#   bash scripts/eval_mteb.sh                                         # RikkaBotan/NexteraBERT-Mezzoforte-220M-en
#   MODEL=answerdotai/ModernBERT-base bash scripts/eval_mteb.sh       # a baseline (own tokenizer)
#   MODEL=chandar-lab/NeoBERT bash scripts/eval_mteb.sh
#   MODEL=LiquidAI/LFM2.5-Encoder-230M bash scripts/eval_mteb.sh
#   MODEL=checkpoints/mezzoforte_bert/phase2/backbone bash scripts/eval_mteb.sh
#   MTEB_TASK_TYPES=STS,PairClassification MTEB_MAX_TASKS=4 bash scripts/eval_mteb.sh   # smoke
#   scripts/eval_mteb_baselines.sh runs this for several models at once.
#
# Knobs (all optional; see scripts/eval_common.sh for MODEL / OUT_DIR / NPROC):
#   SIMCSE_DIR         where the tuned model goes    (default ${OUT_DIR}/simcse)
#   SIMCSE_LR          5e-5 (OptiBERT / SimCSE)      SIMCSE_BATCH_SIZE 512 (halve on OOM)
#   SIMCSE_EXTRA_ARGS  more finetune_contrastive.py args, e.g. "--no_llrd": LLRD factor
#                      1.0, the other side of the LLRD comparison (use its own OUT_DIR)
#   TRIPLETS_CACHE     NLI triplets JSON, shared between models
#                      (default ${HUB_CACHE_DIR}/nli_triplets.json)
#   MTEB_BENCHMARK     "MTEB(eng, v2)"               MTEB_MAX_LEN 512   MTEB_BATCH_SIZE 64
#   MTEB_TASK_TYPES / MTEB_TASKS / MTEB_EXCLUDE_TASKS / MTEB_MAX_TASKS  -> evaluate_mteb.py
#   MTEB_EXTRA_ARGS    anything else for evaluate_mteb.py
#
# Output: ${OUT_DIR}/mteb_simcse.json
#         per-task cache under ${MTEB_CACHE}, so a killed run resumes.
# Both stages are single-process (InfoNCE negatives come from one forward pass);
# pin a GPU with CUDA_VISIBLE_DEVICES. Rough cost on one H100: SimCSE ~1 h at
# batch 512, MTEB(eng, v2) 3-6 h (MindSmallReranking alone is a large share --
# MTEB_EXCLUDE_TASKS=MindSmallReranking for a quicker first pass).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

SIMCSE_DIR="${SIMCSE_DIR:-${OUT_DIR}/simcse}"
SIMCSE_LR="${SIMCSE_LR:-5e-5}"
SIMCSE_BATCH_SIZE="${SIMCSE_BATCH_SIZE:-512}"
SIMCSE_EXTRA_ARGS="${SIMCSE_EXTRA_ARGS:-}"
TRIPLETS_CACHE="${TRIPLETS_CACHE:-${HUB_CACHE_DIR}/nli_triplets.json}"
MTEB_BENCHMARK="${MTEB_BENCHMARK:-MTEB(eng, v2)}"
MTEB_MAX_LEN="${MTEB_MAX_LEN:-512}"
MTEB_BATCH_SIZE="${MTEB_BATCH_SIZE:-64}"
MTEB_TASK_TYPES="${MTEB_TASK_TYPES:-}"
MTEB_TASKS="${MTEB_TASKS:-}"
MTEB_EXCLUDE_TASKS="${MTEB_EXCLUDE_TASKS:-}"
MTEB_MAX_TASKS="${MTEB_MAX_TASKS:-0}"
MTEB_EXTRA_ARGS="${MTEB_EXTRA_ARGS:-}"
MTEB_OUT="${OUT_DIR}/mteb_simcse.json"

START=$(date +%s)
mkdir -p "$OUT_DIR" "$(dirname "$TRIPLETS_CACHE")"
BACKBONE=$(resolve_model "$MODEL")
MODEL_TYPE=$(model_type "$BACKBONE")
if [ "$MODEL_TYPE" = "nexterabert" ]; then
    KIND="nexterabert"
else
    KIND="hf"
fi
# a baseline is tokenised by its own tokenizer (bundled in its snapshot) unless
# the caller chose one -- eval_common.sh's tokenizer_for
TOKENIZER=$(tokenizer_for "$BACKBONE")
log "MTEB: model=${BACKBONE} (${KIND}: model_type=${MODEL_TYPE:-?}) tokenizer=${TOKENIZER} out=${OUT_DIR}"

MTEB_ARGS=(--benchmark "$MTEB_BENCHMARK" --max_len "$MTEB_MAX_LEN" --batch_size "$MTEB_BATCH_SIZE"
           --cache_dir "$MTEB_CACHE" --skip_errors)
[ -n "$MTEB_TASK_TYPES" ]    && MTEB_ARGS+=(--task_types "$MTEB_TASK_TYPES")
[ -n "$MTEB_TASKS" ]         && MTEB_ARGS+=(--tasks "$MTEB_TASKS")
[ -n "$MTEB_EXCLUDE_TASKS" ] && MTEB_ARGS+=(--exclude_tasks "$MTEB_EXCLUDE_TASKS")
[ "$MTEB_MAX_TASKS" != "0" ] && MTEB_ARGS+=(--max_tasks "$MTEB_MAX_TASKS")

if done_or_skip "$MTEB_OUT"; then exit 0; fi

# -- stage 1: SimCSE (attentive pooling, NLI) ----------------------------------
if [ -f "${SIMCSE_DIR}/contrastive_run.json" ] && [ "$FORCE" != "1" ]; then
    log "SimCSE checkpoint exists, skipping training: ${SIMCSE_DIR}"
else
    log "SimCSE (mteb-nli): lr=${SIMCSE_LR} bs=${SIMCSE_BATCH_SIZE} -> ${SIMCSE_DIR}"
    # --model opens a NexteraBERT export or a Hugging Face snapshot alike (the
    # loader dispatches on config.json's model_type).
    # shellcheck disable=SC2086
    "$PY" scripts/finetune_contrastive.py \
        --protocol mteb-nli \
        --model "$BACKBONE" --tokenizer "$TOKENIZER" \
        --output_dir "$SIMCSE_DIR" \
        --lr "$SIMCSE_LR" --batch_size "$SIMCSE_BATCH_SIZE" \
        --triplets_cache "$TRIPLETS_CACHE" \
        ${SIMCSE_EXTRA_ARGS}
fi

# -- stage 2: MTEB(eng, v2) over the tuned model --------------------------------
log "MTEB ${MTEB_BENCHMARK} -> ${MTEB_OUT}"
# shellcheck disable=SC2086
"$PY" scripts/evaluate_mteb.py \
    --model "$SIMCSE_DIR" --tokenizer "$TOKENIZER" --pooling auto \
    "${MTEB_ARGS[@]}" --output "$MTEB_OUT" ${MTEB_EXTRA_ARGS}

log "MTEB done in $(fmt_elapsed "$START"); results -> ${MTEB_OUT}"
log "compare rows with: $PY scripts/summarize_mteb.py eval_results/*/mteb_simcse.json"
