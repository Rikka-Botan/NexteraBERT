#!/usr/bin/env bash
# IR (DPR) columns of ModernBERT's Table 1 (Warner et al., 2024, 3.1.2 / 3.1.3,
# App. E.2): single-vector retrieval -- BEIR, MLDR out-of-domain, MLDR in-domain.
#
# Stage 1 is a contrastive fine-tuning whose recipe is RETRIEVAL_PROTOCOL
# (scripts/eval_common.sh):
#   st-msmarco (default) the paper's DPR stage, run with sentence-transformers as a
#                        port of ModernBERT's public examples/train_st.py: mean
#                        pooling, msmarco-co-condenser triplet-hard 1.25M, Cached
#                        MNRL at per-device batch 512, 5% warmup, 1 epoch, lr 8e-5
#                        (Table 9) or the sweep over [1e-5 .. 1e-4] selected on
#                        NFCorpus/SciFact/TREC-COVID/FiQA (their evaluate_st.py).
#                        Comparable to the paper's IR (DPR) / Code columns; the same
#                        script on MODEL=bert-base-uncased reproduces their control.
#   mteb-nli             the same stage the repo's MTEB numbers come from --
#                        attentive pooling head + supervised-SimCSE InfoNCE on
#                        MNLI+SNLI, batch 512, max_len 64, 3 epochs, STS-B dev
#                        checkpoint selection (OptiBERT App. D.2). Retrieval is
#                        then scored with that head, so BEIR / MLDR / CoIR share
#                        one embedding recipe with MTEB. NOT the paper's DPR
#                        protocol: comparable across this repo's runs, not to the
#                        paper's IR (DPR) row.
#   retrieval-msmarco    this repo's own re-implementation of the DPR stage
#                        (finetune_contrastive.py, batch 64, LLRD); reference only.
#   RETRIEVAL_MODEL=<dir> reuses an existing checkpoint (e.g. the simcse directory
#                        an MTEB run produced) and skips stage 1.
# Then:
#   2. BEIR (15 datasets, nDCG@10; MSMARCO scored on dev as in the original
#      suite) on the selected checkpoint.                  -> beir_dpr.json
#   3. MLDR out-of-domain: the same checkpoint, no further training, on the
#      English MLDR test split read at 8192 tokens.        -> mldr_ood_dpr.json
#   4. MLDR in-domain: the checkpoint is further fine-tuned on the MLDR-en
#      training split at 8192 tokens (one sampled negative per query = ~10k
#      triplets, batch 32, one epoch, lr 2e-5; the paper gives no hyperparameters
#      or script for this stage. Measured on nextera-130B: the DPR lr 8e-5 was
#      destructive (29.5 < 36.3 OOD), one negative reached 41.6, all 20 mined
#      negatives over-trained to 37.8. MLDR_ID_EXTRA_ARGS="--negatives k" samples
#      k per query; MLDR_ID_LRS="1e-5 2e-5 5e-5" sweeps the lr, selected on
#      held-out MLDR train queries; pick further variants on the MLDR dev split
#      with scripts/diagnose_retrieval.py --split dev, never on test), then
#      re-scored.
#                                                          -> mldr_id_dpr.json
# Table 1 reference (base): BERT 38.9 / 23.9 / 32.2, ModernBERT 41.6 / 27.4 / 44.0.
#
# Usage:
#   bash scripts/eval_dpr.sh                                    # st-msmarco, lr 8e-5
#   DPR_LRS="1e-5 2e-5 3e-5 5e-5 8e-5 1e-4" bash scripts/eval_dpr.sh   # the paper's lr sweep
#   MODEL=bert-base-uncased bash scripts/eval_dpr.sh            # the paper's BERT-base control (38.9)
#   MODEL=LiquidAI/LFM2.5-Encoder-230M RUN_MLDR_ID=0 LONG_RETRIEVAL_BATCH_SIZE=16 \
#       bash scripts/eval_dpr.sh                                # a baseline: BEIR + MLDR_OOD
#       (train_st_dpr.py repairs LFM's checkpoint layout first -- a plain AutoModel
#        load, which is what sentence-transformers does, yields RANDOM weights)
#   RETRIEVAL_PROTOCOL=mteb-nli RETRIEVAL_MODEL=checkpoints/nextera-130B-simcse bash scripts/eval_dpr.sh
#   RUN_MLDR_ID=0 bash scripts/eval_dpr.sh                      # skip the in-domain stage
#   MLDR_ID_LRS="1e-5 2e-5 5e-5" bash scripts/eval_dpr.sh       # sweep the in-domain lr instead of 2e-5
#   DPR_EXTRA_ARGS="--no_llrd" bash scripts/eval_dpr.sh         # flat published optimiser
#   DPR_EXTRA_ARGS="--max_samples 100000" bash scripts/eval_dpr.sh   # smoke run
#
# Sweep points train one per GPU (NPROC at a time). Every stage skips work whose
# output already exists (FORCE=1 to redo), so an interrupted run resumes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

RUN_MLDR_ID="${RUN_MLDR_ID:-1}"
MLDR_ID_LR="${MLDR_ID_LR:-}"                # empty = the lr selected above
MLDR_ID_EXTRA_ARGS="${MLDR_ID_EXTRA_ARGS:-}"

DPR_ROOT="${OUT_DIR}/dpr"
SELECTED="${DPR_ROOT}/selected.json"
BEIR_OUT="${OUT_DIR}/beir_dpr.json"
MLDR_OOD_OUT="${OUT_DIR}/mldr_ood_dpr.json"
MLDR_ID_OUT="${OUT_DIR}/mldr_id_dpr.json"

START=$(date +%s)
mkdir -p "$DPR_ROOT"
BACKBONE=$(resolve_model "$MODEL")
LRS="${DPR_LRS:-$DPR_LR}"
log "IR (DPR): model=${BACKBONE} protocol=${RETRIEVAL_PROTOCOL} lrs=[${LRS}] bs=${DPR_BATCH_SIZE:-protocol} gpus=${NPROC}"

# -- helpers: run <fn lr gpu> for every lr, NPROC jobs at a time ---------------
sweep_train() {   # sweep_train <lr> <gpu>
    with_gpu "$2" train_dpr "$BACKBONE" "$1" "${DPR_ROOT}/lr$1" \
        > "${DPR_ROOT}/train_lr$1.log" 2>&1
}
sweep_select() {  # sweep_select <lr> <gpu>: score the App. E.2 selection subset
    with_gpu "$2" eval_retrieval "${DPR_ROOT}/lr$1" "${DPR_ROOT}/select_lr$1.json" \
        "$BEIR_MAX_LEN" --tasks "$SELECTION_TASKS" \
        > "${DPR_ROOT}/select_lr$1.log" 2>&1
}
run_over_lrs() {  # run_over_lrs <fn> : parallel over GPUs, fails if any job fails
    local fn="$1" i=0 pids=() lr
    local lrs="${LRS}"
    for lr in $lrs; do
        "$fn" "$lr" "$(( i % NPROC ))" &
        pids+=("$!")
        i=$(( i + 1 ))
        if [ $(( i % NPROC )) -eq 0 ]; then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

# -- 1. train (and select the learning rate) ----------------------------------
N_LRS=$(wc -w <<< "$LRS")
if [ -n "$RETRIEVAL_MODEL" ]; then
    BEST=$(resolve_retrieval_model)
    BEST_LR=$(run_json_field "${BEST}/contrastive_run.json" lr)
    LRS="$BEST_LR"
    log "reusing contrastive checkpoint ${BEST} (protocol $(run_json_field "${BEST}/contrastive_run.json" protocol), lr ${BEST_LR})"
elif [ "$N_LRS" -eq 1 ]; then
    BEST_LR="$LRS"
    BEST="${DPR_ROOT}/lr${BEST_LR}"
    train_dpr "$BACKBONE" "$BEST_LR" "$BEST"
else
    log "lr sweep over ${N_LRS} points, ${NPROC} at a time (logs: ${DPR_ROOT}/train_lr*.log)"
    run_over_lrs sweep_train
    log "scoring the selection subset [${SELECTION_TASKS}] for every point"
    run_over_lrs sweep_select
    PAIRS=()
    for lr in $LRS; do PAIRS+=("${lr}=${DPR_ROOT}/select_lr${lr}.json"); done
    BEST_LR=$(pick_lr "${PAIRS[@]}")
    BEST="${DPR_ROOT}/lr${BEST_LR}"
    log "selected lr=${BEST_LR}"
fi
record_selected "$SELECTED" "$BEST_LR" "$BEST" "$LRS"
log "single-vector checkpoint: ${BEST} (recorded in ${SELECTED})"

# -- 2. BEIR -------------------------------------------------------------------
# One dataset per GPU (NPROC at a time); with a single GPU this is one mteb call.
eval_retrieval_tasks() {   # <model> <out> [evaluate_retrieval args...]
    local model="$1" out="$2"; shift 2
    eval_retrieval "$model" "$out" "$BEIR_MAX_LEN" "$@"
}
log "BEIR (15 datasets, max_len ${BEIR_MAX_LEN}, ${NPROC} GPU(s)) -> ${BEIR_OUT}"
parallel_tasks eval_retrieval_tasks "$BEST" "$BEIR_OUT" "$BEIR_TASKS_ALL"

# -- 3. MLDR out-of-domain -----------------------------------------------------
log "MLDR out-of-domain (English test, max_len ${MLDR_MAX_LEN}) -> ${MLDR_OOD_OUT}"
eval_retrieval "$BEST" "$MLDR_OOD_OUT" "$MLDR_MAX_LEN" "${MLDR_TASK_ARGS[@]}"

# -- 4. MLDR in-domain ---------------------------------------------------------
if [ "$RUN_MLDR_ID" = "1" ]; then
    ID_LRS="${MLDR_ID_LR:-$MLDR_ID_LRS}"
    N_ID=$(wc -w <<< "$ID_LRS")
    if [ "$N_ID" -eq 1 ]; then
        ID_LR="$ID_LRS"
        MLDR_MODEL="${DPR_ROOT}/mldr_id_lr${ID_LR}"
        train_mldr_id "$BEST" "$ID_LR" "$MLDR_MODEL"
    else
        log "MLDR in-domain lr sweep over [${ID_LRS}], ${NPROC} at a time, selected on held-out MLDR train queries"
        id_train() { with_gpu "$2" train_mldr_id "$BEST" "$1" "${DPR_ROOT}/mldr_id_lr$1" > "${DPR_ROOT}/train_mldr_id_lr$1.log" 2>&1; }
        LRS="$ID_LRS" run_over_lrs id_train
        PAIRS=()
        for lr in $ID_LRS; do PAIRS+=("${lr}=${DPR_ROOT}/mldr_id_lr${lr}"); done
        ID_LR=$("$PY" scripts/summarize_eval_suite.py pick-triplet "${PAIRS[@]}")
        MLDR_MODEL="${DPR_ROOT}/mldr_id_lr${ID_LR}"
        log "selected MLDR in-domain lr=${ID_LR}"
    fi
    "$PY" -c 'import json, sys
json.dump({"protocol": "st-mldr", "lr": float(sys.argv[2]), "model": sys.argv[3],
           "sweep": [float(x) for x in sys.argv[4].split()], "selection": "held-out MLDR train triplet accuracy"},
          open(sys.argv[1], "w"), indent=2)' "${DPR_ROOT}/mldr_id_selected.json" "$ID_LR" "$MLDR_MODEL" "$ID_LRS"
    log "MLDR in-domain (English test, max_len ${MLDR_MAX_LEN}) -> ${MLDR_ID_OUT}"
    eval_retrieval "$MLDR_MODEL" "$MLDR_ID_OUT" "$MLDR_MAX_LEN" "${MLDR_TASK_ARGS[@]}"
else
    log "RUN_MLDR_ID=0 - skipping the MLDR in-domain stage"
fi

log "IR (DPR) done in $(fmt_elapsed "$START")"
for f in "$BEIR_OUT" "$MLDR_OOD_OUT" "$MLDR_ID_OUT"; do
    [ -f "$f" ] && log "  $(basename "$f"): $("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d.get('average_ndcg@10', float('nan'))*100:.2f} nDCG@10\")" "$f")"
done
exit 0
