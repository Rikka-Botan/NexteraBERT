#!/usr/bin/env bash
# NanoBEIR for NexteraBERT next to its baselines -- ModernBERT-base, NeoBERT and
# LFM2.5-Encoder-230M -- every model through scripts/eval_nanobeir.sh (the 250k-
# triplet MS MARCO stage, then the 13 subsets with the BEIR column's settings:
# BEIR_MAX_LEN 512, RETRIEVAL_BATCH_SIZE, the same evaluate_retrieval.py), then one
# comparison table.
#
# Usage:
#   bash scripts/eval_nanobeir_baselines.sh
#   MODELS="answerdotai/ModernBERT-base chandar-lab/NeoBERT" bash scripts/eval_nanobeir_baselines.sh
#   MODELS="checkpoints/mezzoforte_bert/phase2/backbone chandar-lab/NeoBERT" NPROC=2 \
#       bash scripts/eval_nanobeir_baselines.sh
#   NANOBEIR_TASKS=NanoSciFactRetrieval,NanoNFCorpusRetrieval RESULTS_ROOT=eval_results_smoke \
#       bash scripts/eval_nanobeir_baselines.sh                           # smoke
#
# MODELS      space-separated Hub ids / local dirs (default: NexteraBERT-Mezzoforte-220M-en +
#             the three baselines). Each gets eval_results/<basename>/, the same
#             directory eval_dpr.sh / eval_code.sh use.
# NPROC       GPUs: models run concurrently, one per GPU (the MS MARCO stage and the
#             scoring are single-process), the rest queue. Default: all visible GPUs.
# Every eval_common.sh knob (BEIR_MAX_LEN, RETRIEVAL_*, DPR_*, ATTN_IMPLEMENTATION,
# FORCE) passes through. TOKENIZER, RETRIEVAL_MODEL and DPR_MODEL are NOT forwarded
# on purpose: they name one model's tokenizer / checkpoint; set them per run of
# scripts/eval_nanobeir.sh instead.
#
# Output: eval_results/<model>/nanobeir_dpr_lite.json per model and
#         eval_results/nanobeir_dpr_lite_comparison.{md,json} from scripts/summarize_nanobeir.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS="${MODELS:-RikkaBotan/NexteraBERT-Mezzoforte-220M-en answerdotai/ModernBERT-base chandar-lab/NeoBERT LiquidAI/LFM2.5-Encoder-230M}"
RESULTS_ROOT="${RESULTS_ROOT:-eval_results}"
RUN_TAG="dpr_lite"

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
    log "===== ${model} -> ${out} (gpu ${gpu}) ====="
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
