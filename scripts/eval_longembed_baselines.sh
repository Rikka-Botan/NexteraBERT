#!/usr/bin/env bash
# LongEmbed for NexteraBERT next to its baselines -- ModernBERT-base, NeoBERT and
# LFM2.5-Encoder-230M -- every model through the SAME single-vector stage and
# scorer as the MLDR out-of-domain column (scripts/eval_longembed.sh per model: the
# DPR checkpoint eval_dpr.sh selected, else one trained under RETRIEVAL_PROTOCOL at
# DPR_LR; zero-shot on LongEmbed, LONGEMBED_MAX_LEN 8192, RETRIEVAL_DTYPE), then
# one comparison table.
#
# Context lengths: NexteraBERT and ModernBERT are native at 8192; NeoBERT was
# trained to 4096 and its RoPE table is recomputed to LONGEMBED_MAX_LEN on load
# (hf_baselines.repair_wrapped_neobert), so its 4096..8192 range is extrapolation
# -- the same treatment it gets on MLDR. LONGEMBED_MAX_LEN=4096 scores all four at
# a length every model was trained on.
#
# Usage:
#   bash scripts/eval_longembed_baselines.sh
#   MODELS="answerdotai/ModernBERT-base chandar-lab/NeoBERT" bash scripts/eval_longembed_baselines.sh
#   MODELS="checkpoints/mezzoforte_bert/phase2/backbone chandar-lab/NeoBERT" NPROC=2 \
#       bash scripts/eval_longembed_baselines.sh
#   LONGEMBED_SUBSET=synthetic bash scripts/eval_longembed_baselines.sh   # needle + passkey only
#       -> eval_results/<model>/longembed_synthetic_dpr.json, longembed_synthetic_comparison.{md,json}
#   LONGEMBED_MODE=zeroshot LONGEMBED_SUBSET=synthetic LONGEMBED_MAX_LEN=32768 LONG_RETRIEVAL_BATCH_SIZE=2 \
#       bash scripts/eval_longembed_baselines.sh   # RAW backbones read at 32768: what pretraining gives
#       past 8192, without the short-text MS MARCO stage -> longembed_synthetic_comparison_zeroshot_len32768.*
#   DPR_BUDGET=lite bash scripts/eval_longembed_baselines.sh   # 1/5-cost MS MARCO stage for every model
#       (250k triplets, own dpr_lite/ checkpoints, *_lite result files; not paper-comparable)
#   RETRIEVAL_DTYPE=bf16 LONG_RETRIEVAL_BATCH_SIZE=8 bash scripts/eval_longembed_baselines.sh   # ONE value for all models
#   DPR_EXTRA_ARGS="--max_samples 100000" LONGEMBED_TASKS=LEMBPasskeyRetrieval,LEMBWikimQARetrieval \
#       RESULTS_ROOT=eval_results_smoke bash scripts/eval_longembed_baselines.sh   # smoke
#
# MODELS      space-separated Hub ids / local dirs (default: NexteraBERT-Mezzoforte-220M-en +
#             the three baselines). Each gets eval_results/<basename>/, the same
#             directory eval_dpr.sh / eval_code.sh use, so an existing
#             dpr/selected.json is reused and nothing is retrained.
# NPROC       GPUs: models run concurrently, one per GPU (the DPR stage and the
#             scoring are single-process), the rest queue. Default: all visible GPUs.
# Every eval_common.sh knob (RETRIEVAL_PROTOCOL, DPR_*, LONGEMBED_MAX_LEN,
# RETRIEVAL_*, LONG_RETRIEVAL_BATCH_SIZE, ATTN_IMPLEMENTATION, FORCE) passes
# through. LFM2.5-Encoder's default attention builds a dense (B,1,T,T) mask
# (~134 MB per sample at 8192 in bf16): give LONG_RETRIEVAL_BATCH_SIZE=8 or
# ATTN_IMPLEMENTATION=flash_attention_2. TOKENIZER, RETRIEVAL_MODEL and DPR_MODEL
# are NOT forwarded on purpose: they name one model's tokenizer / checkpoint; set
# them per run of scripts/eval_longembed.sh instead.
#
# Output: eval_results/<model>/longembed_dpr.json per model and
#         eval_results/longembed_comparison.{md,json} from scripts/summarize_longembed.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS="${MODELS:-RikkaBotan/NexteraBERT-Mezzoforte-220M-en answerdotai/ModernBERT-base chandar-lab/NeoBERT LiquidAI/LFM2.5-Encoder-230M}"
RESULTS_ROOT="${RESULTS_ROOT:-eval_results}"
LONGEMBED_MAX_LEN="${LONGEMBED_MAX_LEN:-${MLDR_MAX_LEN:-8192}}"
LONGEMBED_SUBSET="${LONGEMBED_SUBSET:-all}"     # all | synthetic (needle + passkey) | real
export LONGEMBED_MAX_LEN LONGEMBED_SUBSET
# the file names of scripts/eval_longembed.sh
case "$LONGEMBED_SUBSET" in
    all) STEM="longembed_dpr"; COMPARISON="longembed_comparison" ;;
    synthetic|real) STEM="longembed_${LONGEMBED_SUBSET}_dpr"; COMPARISON="longembed_${LONGEMBED_SUBSET}_comparison" ;;
    *) echo "ERROR: LONGEMBED_SUBSET must be all, synthetic or real (got ${LONGEMBED_SUBSET})" >&2; exit 1 ;;
esac
DPR_BUDGET="${DPR_BUDGET:-full}"                # full | lite (250k-triplet MS MARCO stage, eval_common.sh)
LONGEMBED_MODE="${LONGEMBED_MODE:-dpr}"         # dpr | zeroshot (raw backbone, mean pooling, nothing trained)
export DPR_BUDGET LONGEMBED_MODE
if [ "$LONGEMBED_MODE" = "zeroshot" ]; then
    STEM="${STEM%_dpr}_zeroshot"; COMPARISON="${COMPARISON}_zeroshot"
elif [ "$DPR_BUDGET" = "lite" ]; then STEM="${STEM}_lite"; COMPARISON="${COMPARISON}_lite"; fi
if [ "$LONGEMBED_MAX_LEN" != "8192" ]; then
    STEM="${STEM}_len${LONGEMBED_MAX_LEN}"; COMPARISON="${COMPARISON}_len${LONGEMBED_MAX_LEN}"
fi
RESULT_NAME="${STEM}.json"; COMPARISON="${COMPARISON}.json"

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
    logfile="${out}/eval_longembed_${LONGEMBED_MODE}.log"
    log "===== ${model} -> ${out} (gpu ${gpu}) ====="
    if [ "$NPROC" -gt 1 ]; then
        # NPROC=1 inside the job: it sees exactly one GPU
        CUDA_VISIBLE_DEVICES="$gpu" NPROC=1 MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" \
            bash scripts/eval_longembed.sh >"$logfile" 2>&1 &
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
        if ! MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" bash scripts/eval_longembed.sh 2>&1 | tee "$logfile"; then
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
    f="${RESULTS_ROOT}/$(basename "${model%/}")/${RESULT_NAME}"
    [ -f "$f" ] && results+=("$f")
done
if [ "${#results[@]}" -gt 0 ]; then
    "$PY" scripts/summarize_longembed.py "${results[@]}" --output "${RESULTS_ROOT}/${COMPARISON}"
fi

log "finished in $(( ($(date +%s) - START) / 60 ))m"
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "models that failed: ${FAILED[*]} (rerun the same command; finished stages are skipped)"
    exit 1
fi
