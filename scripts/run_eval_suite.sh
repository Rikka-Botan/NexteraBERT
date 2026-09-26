#!/usr/bin/env bash
# ModernBERT evaluation suite (Warner et al., ACL 2025, Table 1) for a trained
# NexteraBERT read from the Hugging Face Hub:
#
#   IR (DPR)      BEIR, MLDR out-of-domain, MLDR in-domain    scripts/eval_dpr.sh
#   IR (ColBERT)  BEIR, MLDR out-of-domain                    scripts/eval_colbert.sh
#   NLU           GLUE                                        scripts/eval_nlu.sh
#   Code          CodeSearchNet (CSN), StackOverflow-QA (SQA) scripts/eval_code.sh
#
# then scripts/summarize_eval_suite.py folds the results into one Table-1 row
# next to the paper's BERT-base / ModernBERT-base rows.
#
# Usage:
#   bash scripts/run_eval_suite.sh                                   # RikkaBotan/NexteraBERT-Mezzoforte-220M-en
#   MODEL=RikkaBotan/NexteraBERT-Mezzoforte-220M-en NPROC=8 bash scripts/run_eval_suite.sh
#   MODEL=checkpoints/mezzoforte_bert/phase2/backbone bash scripts/run_eval_suite.sh
#   RUN_COLBERT=0 bash scripts/run_eval_suite.sh                     # skip a column group
#   RETRIEVAL_MODEL=checkpoints/nextera-130B-simcse bash scripts/run_eval_suite.sh   # reuse the MTEB checkpoint
#   RETRIEVAL_PROTOCOL=retrieval-msmarco DPR_LRS="1e-5 2e-5 3e-5 5e-5 8e-5 1e-4" \
#       COLBERT_LRS="1e-5 2e-5 3e-5 5e-5 8e-5 1e-4" bash scripts/run_eval_suite.sh  # the paper's DPR stage + lr sweeps
#
# The single-vector stage (BEIR / MLDR / CoIR) defaults to RETRIEVAL_PROTOCOL=mteb-nli,
# the same attentive-pooling SimCSE recipe the MTEB numbers come from, so every
# single-vector number in the repo shares one embedding recipe; ModernBERT's own
# MS MARCO DPR stage is RETRIEVAL_PROTOCOL=retrieval-msmarco (see eval_common.sh).
#
# Every stage script can also be run on its own with the same environment
# variables (they are documented in scripts/eval_common.sh and in each script's
# header). Results land in ${OUT_DIR} (default eval_results/<model name>); a
# stage whose results file exists is skipped, so rerunning after a failure
# resumes where it stopped (FORCE=1 redoes everything).
#
# Rough cost on 8 GPUs, no lr sweeps: GLUE a few hours; DPR training ~2-3 h then
# BEIR ~1-2 h; MLDR at 8192 tokens is dominated by encoding the 200k-document
# corpus; ColBERT training ~4-6 h, BEIR with PLAID indexes ~2-3 h, and ColBERT
# MLDR the largest single step (see eval_colbert.sh).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NLU="${RUN_NLU:-1}"
RUN_DPR="${RUN_DPR:-1}"
RUN_CODE="${RUN_CODE:-1}"
RUN_COLBERT="${RUN_COLBERT:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

# eval_common.sh resolves MODEL / OUT_DIR / NPROC the same way for every stage;
# sourcing it here only fixes those values and prints the plan.
source scripts/eval_common.sh
export MODEL TOKENIZER OUT_DIR NPROC FORCE HUB_CACHE_DIR MODEL_TAG MTEB_CACHE PY
export RETRIEVAL_PROTOCOL RETRIEVAL_MODEL

START=$(date +%s)
mkdir -p "$OUT_DIR"
log "ModernBERT evaluation suite: model=${MODEL} out=${OUT_DIR} gpus=${NPROC}"
log "stages: nlu=${RUN_NLU} dpr=${RUN_DPR} code=${RUN_CODE} colbert=${RUN_COLBERT}  single-vector stage: ${RETRIEVAL_PROTOCOL}${RETRIEVAL_MODEL:+ (reusing ${RETRIEVAL_MODEL})}"

if [ "$INSTALL_DEPS" = "1" ]; then
    log "installing dependencies (nexterabert[eval])"
    "$PY" -m pip install -e ".[eval]" --quiet
fi
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi

# Download once so the stages (and any parallel sweep jobs) never race on it.
BACKBONE=$(resolve_model "$MODEL")
log "backbone snapshot: ${BACKBONE}"

FAILED=()
run_stage() {   # run_stage <name> <script>
    local name="$1" script="$2" t0
    t0=$(date +%s)
    log "===== ${name}: ${script} ====="
    if bash "$script"; then
        log "===== ${name} finished in $(fmt_elapsed "$t0") ====="
    else
        log "===== ${name} FAILED (exit $?) after $(fmt_elapsed "$t0") - continuing ====="
        FAILED+=("$name")
    fi
}

# DPR first: eval_code.sh reuses the checkpoint it selects.
[ "$RUN_DPR" = "1" ]     && run_stage "IR (DPR)"     scripts/eval_dpr.sh
[ "$RUN_CODE" = "1" ]    && run_stage "Code"         scripts/eval_code.sh
[ "$RUN_NLU" = "1" ]     && run_stage "NLU (GLUE)"   scripts/eval_nlu.sh
[ "$RUN_COLBERT" = "1" ] && run_stage "IR (ColBERT)" scripts/eval_colbert.sh

log "===== summary ====="
"$PY" scripts/summarize_eval_suite.py "$OUT_DIR" --model_name "$MODEL_TAG"

log "suite finished in $(fmt_elapsed "$START")"
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "stages that failed: ${FAILED[*]} (rerun the same command; finished stages are skipped)"
    exit 1
fi
