#!/usr/bin/env bash
# NanoBEIR for NexteraBERT next to its baselines -- ModernBERT-base, NeoBERT and
# LFM2.5-Encoder-230M -- every model through scripts/eval_nanobeir.sh, then one
# comparison table. Default NANOBEIR_MODE=zeroshot: the raw pretrained backbones,
# masked mean pooling, nothing trained, scored with the BEIR column's settings
# (BEIR_MAX_LEN 512, RETRIEVAL_BATCH_SIZE, the same evaluate_retrieval.py).
# NANOBEIR_MODE=dpr scores each model's DPR checkpoint instead (the one eval_dpr.sh
# selected, else one trained under RETRIEVAL_PROTOCOL at DPR_LR).
#
# Usage:
#   bash scripts/eval_nanobeir_baselines.sh
#   MODELS="answerdotai/ModernBERT-base chandar-lab/NeoBERT" bash scripts/eval_nanobeir_baselines.sh
#   MODELS="checkpoints/mezzoforte_bert/phase2/backbone chandar-lab/NeoBERT" NPROC=2 \
#       bash scripts/eval_nanobeir_baselines.sh
#   NANOBEIR_MODE=dpr bash scripts/eval_nanobeir_baselines.sh             # after the MS MARCO stage
#   NANOBEIR_TASKS=NanoSciFactRetrieval,NanoNFCorpusRetrieval RESULTS_ROOT=eval_results_smoke \
#       bash scripts/eval_nanobeir_baselines.sh                           # smoke
#
# MODELS      space-separated Hub ids / local dirs (default: NexteraBERT-Mezzoforte-220M-en +
#             the three baselines). Each gets eval_results/<basename>/, the same
#             directory eval_dpr.sh / eval_code.sh use (so NANOBEIR_MODE=dpr reuses
#             an existing dpr/selected.json and retrains nothing).
# NPROC       GPUs: models run concurrently, one per GPU (scoring, and the DPR stage
#             of NANOBEIR_MODE=dpr, are single-process), the rest queue. Default: all
#             visible GPUs.
# Every eval_common.sh knob (BEIR_MAX_LEN, RETRIEVAL_*, FORCE; RETRIEVAL_PROTOCOL,
# DPR_*, ATTN_IMPLEMENTATION for dpr) passes through. TOKENIZER, RETRIEVAL_MODEL and
# DPR_MODEL are NOT forwarded on purpose: they name one model's tokenizer /
# checkpoint; set them per run of scripts/eval_nanobeir.sh instead.
#
# Output: eval_results/<model>/nanobeir_<mode>.json per model and
#         eval_results/nanobeir_<mode>_comparison.{md,json} from scripts/summarize_nanobeir.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS="${MODELS:-RikkaBotan/NexteraBERT-Mezzoforte-220M-en answerdotai/ModernBERT-base chandar-lab/NeoBERT LiquidAI/LFM2.5-Encoder-230M}"
RESULTS_ROOT="${RESULTS_ROOT:-eval_results}"
NANOBEIR_MODE="${NANOBEIR_MODE:-zeroshot}"
export NANOBEIR_MODE
# DPR_BUDGET=lite (dpr mode only): the 250k-triplet MS MARCO stage of eval_common.sh,
# 1/5 the training, own dpr_lite/ checkpoints and nanobeir_dpr_lite* files
DPR_BUDGET="${DPR_BUDGET:-full}"
export DPR_BUDGET
RUN_TAG="$NANOBEIR_MODE"
if [ "$NANOBEIR_MODE" = "dpr" ] && [ "$DPR_BUDGET" = "lite" ]; then RUN_TAG="dpr_lite"; fi

if [ -z "${PY:-}" ]; then
    if command -v python3 &>/dev/null; then PY=python3; else PY=python; fi
fi
log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
NPROC="${NPROC:-$("$PY" -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)}"
export PY
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi
unset TOKENIZER TOKENIZER_EXPLICIT RETRIEVAL_MODEL DPR_MODEL

START=$(date +%s)
declare -a PIDS=() NAMES=() LOGS=()
FAILED=()
gpu=0
for model in $MODELS; do
    tag=$(basename "${model%/}")
    out="${RESULTS_ROOT}/${tag}"
    mkdir -p "$out"
    logfile="${out}/eval_nanobeir_${RUN_TAG}.log"
    log "===== ${model} (${NANOBEIR_MODE}) -> ${out} (gpu ${gpu}) ====="
    if [ "$NPROC" -gt 1 ]; then
        # NPROC=1 inside the job: it sees exactly one GPU
        CUDA_VISIBLE_DEVICES="$gpu" NPROC=1 MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" \
            bash scripts/eval_nanobeir.sh >"$logfile" 2>&1 &
        PIDS+=("$!"); NAMES+=("$model"); LOGS+=("$logfile")
        gpu=$(( gpu + 1 ))
        if [ "$gpu" -ge "$NPROC" ]; then
            # all GPUs busy: wait for this wave before starting the next
            for i in "${!PIDS[@]}"; do
                if ! wait "${PIDS[$i]}"; then
                    log "FAILED: ${NAMES[$i]} (see ${LOGS[$i]})"; FAILED+=("${NAMES[$i]}")
                fi
            done
            PIDS=(); NAMES=(); LOGS=(); gpu=0
        fi
    else
        if ! MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" bash scripts/eval_nanobeir.sh 2>&1 | tee "$logfile"; then
            log "FAILED: ${model}"; FAILED+=("$model")
        fi
    fi
done
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        log "FAILED: ${NAMES[$i]} (see ${LOGS[$i]})"; FAILED+=("${NAMES[$i]}")
    fi
done

log "===== comparison ====="
results=()
for model in $MODELS; do
    f="${RESULTS_ROOT}/$(basename "${model%/}")/nanobeir_${RUN_TAG}.json"
    [ -f "$f" ] && results+=("$f")
done
if [ "${#results[@]}" -gt 0 ]; then
    "$PY" scripts/summarize_nanobeir.py "${results[@]}" \
        --output "${RESULTS_ROOT}/nanobeir_${RUN_TAG}_comparison.json"
fi

log "finished in $(( ($(date +%s) - START) / 60 ))m"
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "models that failed: ${FAILED[*]} (rerun the same command; finished stages are skipped)"
    exit 1
fi
