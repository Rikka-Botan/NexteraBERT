#!/usr/bin/env bash
# Code retrieval: CodeSearchNet (CSN) and StackOverflow-QA (SQA), scored as
# single-vector retrieval tasks of the CoIR framework (nDCG@10) on the MS MARCO
# checkpoint eval_dpr.sh trains (see eval_common.sh):
#
#   * CSN: CoIR's CodeSearchNet, code -> docstring ("identify relevant docstring
#     or comments for code blocks"); mteb task COIRCodeSearchNetRetrieval, the
#     mean over its six language subsets.
#   * SQA: StackOverflow-QA, hybrid text+code, ~2000 tokens per query/document
#     on average, so it is read at the full 8192 context; mteb task StackOverflowQA.
#
# Usage:
#   bash scripts/eval_dpr.sh && bash scripts/eval_code.sh   # reuse the DPR checkpoint
#   bash scripts/eval_code.sh                                # trains one at DPR_LR if none exists
#   DPR_MODEL=checkpoints/mezzoforte/dpr bash scripts/eval_code.sh
#   MODEL=LiquidAI/LFM2.5-Encoder-230M LONG_RETRIEVAL_BATCH_SIZE=16 bash scripts/eval_code.sh
#       # a baseline: CSN + SQA on the checkpoint its eval_dpr.sh run trained
#
# Output: ${OUT_DIR}/code_results.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

CODE_TASKS="${CODE_TASKS:-COIRCodeSearchNetRetrieval,StackOverflowQA}"
CODE_MAX_LEN="${CODE_MAX_LEN:-8192}"
DPR_MODEL="${DPR_MODEL:-}"
CODE_OUT="${OUT_DIR}/code_results.json"
DPR_ROOT="${OUT_DIR}/dpr"
SELECTED="${DPR_ROOT}/selected.json"

START=$(date +%s)
mkdir -p "$DPR_ROOT"
# A results file written by an evaluate_retrieval.py that ignored --tasks (the
# pre-2026-09-13 main() scored its default NanoBEIR benchmark instead) must not
# block the real CoIR run: keep it only if it holds every requested task.
has_requested_tasks() {   # has_requested_tasks <results json> <comma-separated tasks>
    "$PY" -c 'import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if all(t in d for t in sys.argv[2].split(",")) else 1)' "$1" "$2"
}
if [ -f "$CODE_OUT" ] && ! has_requested_tasks "$CODE_OUT" "$CODE_TASKS"; then
    log "stale ${CODE_OUT} (does not contain ${CODE_TASKS}) - moving it aside"
    mv -f "$CODE_OUT" "${CODE_OUT%.json}.stale.json"
fi
# ... and neither must one that scored a PREVIOUS training run of the checkpoint
# (retrained after a divergence or a fix): older than its contrastive_run.json.
_CKPT="${DPR_MODEL:-$(resolve_retrieval_model || true)}"
[ -n "$_CKPT" ] && drop_stale_result "$CODE_OUT" "$_CKPT"
if done_or_skip "$CODE_OUT"; then exit 0; fi

# -- the single-vector checkpoint --------------------------------------------
# DPR_MODEL (explicit) > RETRIEVAL_MODEL (reused checkpoint) > the one eval_dpr.sh
# recorded > train one here.
if [ -z "$DPR_MODEL" ]; then
    DPR_MODEL=$(resolve_retrieval_model)
    [ -n "$DPR_MODEL" ] && log "using the contrastive checkpoint: ${DPR_MODEL}"
fi
if [ -z "$DPR_MODEL" ]; then
    BACKBONE=$(resolve_model "$MODEL")
    DPR_MODEL="${DPR_ROOT}/lr${DPR_LR}"
    log "no contrastive checkpoint yet - training one (lr=${DPR_LR})"
    train_dpr "$BACKBONE" "$DPR_LR" "$DPR_MODEL"
    [ -f "$SELECTED" ] || record_selected "$SELECTED" "$DPR_LR" "$DPR_MODEL"
fi
if [ ! -f "${DPR_MODEL}/contrastive_run.json" ]; then
    log "WARNING: ${DPR_MODEL} records no contrastive stage - it is not an MS MARCO stage output"
fi

# -- CoIR tasks ----------------------------------------------------------------
log "Code (CoIR): ${CODE_TASKS} on ${DPR_MODEL}, max_len ${CODE_MAX_LEN} -> ${CODE_OUT}"
eval_retrieval "$DPR_MODEL" "$CODE_OUT" "$CODE_MAX_LEN" --tasks "$CODE_TASKS"

log "Code done in $(fmt_elapsed "$START")"
"$PY" - "$CODE_OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("COIRCodeSearchNetRetrieval", "StackOverflowQA"):
    if k in d:
        print(f"  {k:<28} {d[k] * 100:6.2f} nDCG@10")
EOF
