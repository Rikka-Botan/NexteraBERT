#!/usr/bin/env bash
# MTEB(eng, v2) for NexteraBERT next to its baselines -- ModernBERT-base,
# NeoBERT and LFM2.5-Encoder-230M -- all through the SAME contrastive stage and scorer
# (scripts/eval_mteb.sh per model), then one comparison table.
#
# Published MTEB rows for these encoders each come from that paper's own
# pipeline (OptiBERT: attentive pooling + SimCSE; ModernBERT / NeoBERT: heavier
# multi-stage recipes), so quoting them next to a NexteraBERT number compares
# pipelines. This script removes that variable: every model gets OptiBERT's
# App. D.2 stage, then MTEB(eng, v2).
#
# Usage:
#   bash scripts/eval_mteb_baselines.sh
#   MODELS="answerdotai/ModernBERT-base chandar-lab/NeoBERT" bash scripts/eval_mteb_baselines.sh
#   MODELS="checkpoints/mezzoforte_bert/phase2/backbone chandar-lab/NeoBERT" NPROC=2 \
#       bash scripts/eval_mteb_baselines.sh
#   MTEB_EXCLUDE_TASKS=MindSmallReranking bash scripts/eval_mteb_baselines.sh   # quicker pass
#   MTEB_TASK_TYPES=STS MTEB_MAX_TASKS=2 SIMCSE_EXTRA_ARGS="--max_samples 2048" \
#       bash scripts/eval_mteb_baselines.sh                                      # smoke
#
# MODELS      space-separated Hub ids / local dirs (default: NexteraBERT-Mezzoforte-220M-en +
#             the three baselines). Each gets eval_results/<basename>/.
# NPROC       GPUs: models run concurrently, one per GPU (both stages are
#             single-process), the rest queue. Default: all visible GPUs.
# Every eval_mteb.sh knob (SIMCSE_*, MTEB_*, TRIPLETS_CACHE, FORCE) passes through.
# TOKENIZER is NOT forwarded on purpose: each baseline keeps its own; set it per
# run of scripts/eval_mteb.sh if a NexteraBERT checkpoint needs a non-default one.
#
# Output: eval_results/<model>/mteb_simcse.json per model and
#         eval_results/mteb_comparison.{md,json} from scripts/summarize_mteb.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS="${MODELS:-RikkaBotan/NexteraBERT-Mezzoforte-220M-en answerdotai/ModernBERT-base chandar-lab/NeoBERT LiquidAI/LFM2.5-Encoder-230M}"
RESULTS_ROOT="${RESULTS_ROOT:-eval_results}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [ -z "${PY:-}" ]; then
    if command -v python3 &>/dev/null; then PY=python3; else PY=python; fi
fi
log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
NPROC="${NPROC:-$("$PY" -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)}"
export PY

if [ "$INSTALL_DEPS" = "1" ]; then
    log "installing dependencies (nexterabert[eval])"
    "$PY" -m pip install -e ".[eval]" --quiet
fi
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi

# The NLI triplets are model-independent: build them once, up front, so the
# concurrent jobs below never race on the cache file.
TRIPLETS_CACHE="${TRIPLETS_CACHE:-checkpoints/hub/nli_triplets.json}"
export TRIPLETS_CACHE
if [ ! -f "$TRIPLETS_CACHE" ]; then
    log "building the NLI triplets once -> ${TRIPLETS_CACHE}"
    mkdir -p "$(dirname "$TRIPLETS_CACHE")"
    "$PY" - "$TRIPLETS_CACHE" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import finetune_contrastive as fc
fc.build_triplets(cache=sys.argv[1])
EOF
fi

START=$(date +%s)
declare -a PIDS=() NAMES=() LOGS=()
FAILED=()
gpu=0
for model in $MODELS; do
    tag=$(basename "${model%/}")
    out="${RESULTS_ROOT}/${tag}"
    mkdir -p "$out"
    logfile="${out}/eval_mteb.log"
    log "===== ${model} -> ${out} (gpu ${gpu}) ====="
    if [ "$NPROC" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES="$gpu" MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" \
            bash scripts/eval_mteb.sh >"$logfile" 2>&1 &
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
        if ! MODEL="$model" OUT_DIR="$out" MODEL_TAG="$tag" bash scripts/eval_mteb.sh 2>&1 | tee "$logfile"; then
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
    f="${RESULTS_ROOT}/$(basename "${model%/}")/mteb_simcse.json"
    [ -f "$f" ] && results+=("$f")
done
if [ "${#results[@]}" -gt 0 ]; then
    "$PY" scripts/summarize_mteb.py "${results[@]}" --output "${RESULTS_ROOT}/mteb_comparison.json"
fi

log "finished in $(( ($(date +%s) - START) / 60 ))m"
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "models that failed: ${FAILED[*]} (rerun the same command; finished stages are skipped)"
    exit 1
fi
