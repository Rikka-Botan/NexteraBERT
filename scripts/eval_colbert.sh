#!/usr/bin/env bash
# IR (ColBERT) columns of ModernBERT's Table 1 (Warner et al., 2024, 3.1.2 /
# 3.1.3, App. E.2): multi-vector retrieval -- BEIR and MLDR out-of-domain.
#
# Protocol:
#   1. JaColBERTv2.5-style knowledge distillation with PyLate: KL-divergence
#      between normalised BGE-M3 teacher scores and the student's MaxSim scores,
#      810k MS MARCO queries x 32 scored passages (lightonai/ms-marco-en-bge),
#      batch 16 (their examples/train_pylate.py; App. E.2 writes it as 8 x 2),
#      5% warmup, one epoch
#      (scripts/train_colbert.py). The learning rate is swept over
#      [1e-5 .. 1e-4] and selected on NFCorpus / SciFact / TREC-COVID / FiQA;
#      ModernBERT-base landed on 1e-4 (Table 9), the default without a sweep.
#   2. BEIR nDCG@10 through mteb's PyLate wrapper (PLAID index + MaxSim),
#      documents read at 510 tokens as in their examples/evaluate_pylate.py.
#                                                        -> beir_colbert.json
#   3. MLDR out-of-domain: the BEIR checkpoint scored on the English MLDR test
#      split "without any further fine-tuning", documents read at 8192 tokens.
#                                                        -> mldr_ood_colbert.json
# Table 1 reference (base): BERT 49.0 / 28.1, ModernBERT 51.3 / 80.2.
#
# Environment: PyLate pins sentence-transformers / transformers below this
# repo's lock, so the stage runs in its own virtualenv (COLBERT_VENV, created on
# first use with the repo's eval extras + pylate). The backbone is read from the
# same Hub snapshot the other stages use, through AutoModel + trust_remote_code.
#
# Usage:
#   bash scripts/eval_colbert.sh                                     # lr 1e-4
#   COLBERT_LRS="1e-5 2e-5 3e-5 5e-5 8e-5 1e-4" bash scripts/eval_colbert.sh
#   COLBERT_EXTRA_ARGS="--max_samples 50000" bash scripts/eval_colbert.sh  # smoke run
#   RUN_COLBERT_MLDR=0 bash scripts/eval_colbert.sh                  # BEIR only
#   NPROC=8 COLBERT_DDP=1 bash scripts/eval_colbert.sh                # 8 GPUs: DDP training, one BEIR dataset per GPU
#   COLBERT_BEIR_TASKS="NFCorpus,SciFact,TRECCOVID,FiQA2018" bash scripts/eval_colbert.sh   # the 4 of their evaluate_pylate.py
#
# MLDR with a multi-vector index is the heaviest step of the whole suite: the
# 200k-document English corpus is indexed token by token at up to 8192 tokens
# per document. COLBERT_MLDR_DOC_LEN lowers that cap if it does not fit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/eval_common.sh

COLBERT_LR="${COLBERT_LR:-1e-4}"             # Table 9: ModernBERT-base multi-vector
COLBERT_LRS="${COLBERT_LRS:-}"               # sweep list; empty = COLBERT_LR only
COLBERT_BATCH_SIZE="${COLBERT_BATCH_SIZE:-16}"  # ModernBERT examples/train_pylate.py (App. E.2's 8 x 2 = the same 16)
COLBERT_GRAD_ACCUM="${COLBERT_GRAD_ACCUM:-1}"
COLBERT_QUERY_LEN="${COLBERT_QUERY_LEN:-32}"
COLBERT_DOC_LEN="${COLBERT_DOC_LEN:-180}"       # training truncation (PyLate default; the reference sets none)
COLBERT_EVAL_DOC_LEN="${COLBERT_EVAL_DOC_LEN:-510}"   # BEIR scoring: ModernBERT examples/evaluate_pylate.py uses document_length=510
COLBERT_MLDR_DOC_LEN="${COLBERT_MLDR_DOC_LEN:-8192}"
COLBERT_EVAL_BATCH_SIZE="${COLBERT_EVAL_BATCH_SIZE:-32}"
COLBERT_NBITS="${COLBERT_NBITS:-4}"
COLBERT_EXTRA_ARGS="${COLBERT_EXTRA_ARGS:-}"
COLBERT_EVAL_EXTRA_ARGS="${COLBERT_EVAL_EXTRA_ARGS:-}"
RUN_COLBERT_MLDR="${RUN_COLBERT_MLDR:-1}"
# Multi-vector BEIR is dominated by the six giant corpora (MSMARCO 8.8M, FEVER
# 5.4M, ClimateFEVER 5.4M, HotpotQA 5.2M, DBPedia 4.6M, NQ 2.7M documents; ~32M
# of the suite's ~34M): every document is stored as ~100-500 token vectors and
# PLAID-indexed, so those six cost days and hundreds of GB of RAM on one node.
# ModernBERT's own evaluate_pylate.py scores only NFCorpus/SciFact/TREC-COVID/FiQA.
# COLBERT_BEIR_TASKS restricts the BEIR stage to a comma-separated task list, e.g.
# the nine small/medium corpora (<= 0.5M documents) plus nothing giant:
#   COLBERT_BEIR_TASKS="NFCorpus,SciFact,TRECCOVID,FiQA2018,ArguAna,SCIDOCS,Touche2020,CQADupstackRetrieval,QuoraRetrieval"
# Empty = the full 15-dataset benchmark (MSMARCO on dev), as in Table 8.
COLBERT_BEIR_TASKS="${COLBERT_BEIR_TASKS:-}"
# The BEIR datasets are scored one per GPU, NPROC at a time (largest corpora
# first), and the KD training can be sharded over the GPUs with torchrun:
# COLBERT_DDP=1 keeps the reference's global batch (COLBERT_BATCH_SIZE, 16) by
# using COLBERT_BATCH_SIZE / NPROC per device -- the Distillation loss is
# per-query, so sharding does not change the objective. NPROC must divide it.
COLBERT_DDP="${COLBERT_DDP:-0}"
COLBERT_VENV="${COLBERT_VENV:-.venv-colbert}"
COLBERT_SKIP_INSTALL="${COLBERT_SKIP_INSTALL:-0}"
COLBERT_PIP_EXTRA="${COLBERT_PIP_EXTRA:-}"      # e.g. a CUDA torch index: "--index-url https://download.pytorch.org/whl/cu128"

COLBERT_ROOT="${OUT_DIR}/colbert"
SELECTED="${COLBERT_ROOT}/selected.json"
BEIR_OUT="${OUT_DIR}/beir_colbert.json"
MLDR_OUT="${OUT_DIR}/mldr_ood_colbert.json"
COLBERT_MTEB_CACHE="${COLBERT_ROOT}/mteb_cache"

START=$(date +%s)
mkdir -p "$COLBERT_ROOT"
BACKBONE=$(resolve_model "$MODEL")

# -- PyLate virtualenv ---------------------------------------------------------
if [ -x "${COLBERT_VENV}/bin/python" ]; then
    VPY="${COLBERT_VENV}/bin/python"
elif [ -x "${COLBERT_VENV}/Scripts/python.exe" ]; then
    VPY="${COLBERT_VENV}/Scripts/python.exe"
else
    log "creating the PyLate virtualenv ${COLBERT_VENV}"
    if command -v uv &>/dev/null; then
        uv venv "$COLBERT_VENV"
    else
        "$PY" -m venv "$COLBERT_VENV"
    fi
    if [ -x "${COLBERT_VENV}/bin/python" ]; then VPY="${COLBERT_VENV}/bin/python"
    else VPY="${COLBERT_VENV}/Scripts/python.exe"; fi
    COLBERT_SKIP_INSTALL=0
fi
if [ "$COLBERT_SKIP_INSTALL" != "1" ]; then
    log "installing nexterabert[eval] + pylate into ${COLBERT_VENV}"
    # shellcheck disable=SC2086
    if command -v uv &>/dev/null; then
        uv pip install --python "$VPY" ${COLBERT_PIP_EXTRA} -e ".[eval]" pylate
    else
        "$VPY" -m pip install --quiet ${COLBERT_PIP_EXTRA} -e ".[eval]" pylate
    fi
fi
"$VPY" -c "import pylate, sentence_transformers, transformers, mteb; print(f'[env] pylate {pylate.__version__} sentence-transformers {sentence_transformers.__version__} transformers {transformers.__version__} mteb {mteb.__version__}')" >&2

LRS="${COLBERT_LRS:-$COLBERT_LR}"
log "IR (ColBERT): model=${BACKBONE} lrs=[${LRS}] bs=${COLBERT_BATCH_SIZE}x${COLBERT_GRAD_ACCUM} gpus=${NPROC}"

# -- helpers -------------------------------------------------------------------
train_colbert() {   # train_colbert <lr> <out dir>
    local lr="$1" out="$2"
    if [ -f "${out}/colbert_run.json" ] && [ "$FORCE" != "1" ]; then
        log "ColBERT checkpoint exists, skipping training: ${out}"
        return 0
    fi
    if [ "$COLBERT_DDP" = "1" ] && [ "$NPROC" -gt 1 ]; then
        if [ $(( COLBERT_BATCH_SIZE % NPROC )) -ne 0 ]; then
            log "ERROR: COLBERT_DDP=1 needs NPROC (${NPROC}) to divide COLBERT_BATCH_SIZE (${COLBERT_BATCH_SIZE})"
            return 1
        fi
        log "  DDP over ${NPROC} GPUs: per-device batch $(( COLBERT_BATCH_SIZE / NPROC )) (global ${COLBERT_BATCH_SIZE})"
        # shellcheck disable=SC2086
        "$VPY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
            scripts/train_colbert.py \
            --model "$BACKBONE" --output_dir "$out" --lr "$lr" \
            --batch_size $(( COLBERT_BATCH_SIZE / NPROC )) --grad_accum "$COLBERT_GRAD_ACCUM" \
            --query_length "$COLBERT_QUERY_LEN" --document_length "$COLBERT_DOC_LEN" \
            ${COLBERT_EXTRA_ARGS}
        return
    fi
    # shellcheck disable=SC2086
    "$VPY" scripts/train_colbert.py \
        --model "$BACKBONE" --output_dir "$out" --lr "$lr" \
        --batch_size "$COLBERT_BATCH_SIZE" --grad_accum "$COLBERT_GRAD_ACCUM" \
        --query_length "$COLBERT_QUERY_LEN" --document_length "$COLBERT_DOC_LEN" \
        ${COLBERT_EXTRA_ARGS}
}
eval_colbert() {    # eval_colbert <model dir> <results json> [evaluate_colbert args...]
    local model="$1" out="$2"; shift 2
    done_or_skip "$out" && return 0
    # shellcheck disable=SC2086
    "$VPY" scripts/evaluate_colbert.py \
        --model "$model" --output "$out" \
        --batch_size "$COLBERT_EVAL_BATCH_SIZE" --nbits "$COLBERT_NBITS" \
        --output_folder "$COLBERT_MTEB_CACHE" \
        --index_dir "${COLBERT_ROOT}/plaid" \
        ${COLBERT_EVAL_EXTRA_ARGS} "$@"
}
sweep_train() {  with_gpu "$2" train_colbert "$1" "${COLBERT_ROOT}/lr$1" > "${COLBERT_ROOT}/train_lr$1.log" 2>&1; }
sweep_select() { with_gpu "$2" eval_colbert "${COLBERT_ROOT}/lr$1" "${COLBERT_ROOT}/select_lr$1.json" \
                     --tasks "$SELECTION_TASKS" > "${COLBERT_ROOT}/select_lr$1.log" 2>&1; }
run_over_lrs() {
    local fn="$1" i=0 pids=() lr
    for lr in $LRS; do
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
if [ "$N_LRS" -eq 1 ]; then
    BEST_LR="$LRS"
    train_colbert "$BEST_LR" "${COLBERT_ROOT}/lr${BEST_LR}"
else
    log "lr sweep over ${N_LRS} points, ${NPROC} at a time (logs: ${COLBERT_ROOT}/train_lr*.log)"
    run_over_lrs sweep_train
    log "scoring the selection subset [${SELECTION_TASKS}] for every point"
    run_over_lrs sweep_select
    PAIRS=()
    for lr in $LRS; do PAIRS+=("${lr}=${COLBERT_ROOT}/select_lr${lr}.json"); done
    BEST_LR=$("$VPY" scripts/summarize_eval_suite.py pick-lr "${PAIRS[@]}")
    log "selected lr=${BEST_LR}"
fi
BEST="${COLBERT_ROOT}/lr${BEST_LR}"
"$VPY" - "$SELECTED" "$BEST_LR" "$BEST" "$LRS" <<'EOF'
import json, sys
out, lr, model, lrs = sys.argv[1:5]
json.dump({"protocol": "colbert-msmarco-kd", "lr": float(lr), "model": model,
           "sweep": [float(x) for x in lrs.split()]}, open(out, "w"), indent=2)
EOF
log "ColBERT checkpoint: ${BEST} (recorded in ${SELECTED})"

# -- 2. BEIR -------------------------------------------------------------------
if [ -n "$COLBERT_BEIR_TASKS" ]; then
    log "BEIR subset [${COLBERT_BEIR_TASKS}] (document_length ${COLBERT_EVAL_DOC_LEN}, ${NPROC} GPU(s)) -> ${BEIR_OUT}"
    log "  NOTE: a subset average is not Table 8's 15-dataset average; the summary will say so"
else
    log "BEIR (15 datasets, document_length ${COLBERT_EVAL_DOC_LEN}, ${NPROC} GPU(s)) -> ${BEIR_OUT}"
fi
PY="$VPY" parallel_tasks eval_colbert "$BEST" "$BEIR_OUT" "${COLBERT_BEIR_TASKS:-$BEIR_TASKS_ALL}" \
    --document_length "$COLBERT_EVAL_DOC_LEN"

# -- 3. MLDR out-of-domain -----------------------------------------------------
if [ "$RUN_COLBERT_MLDR" = "1" ]; then
    log "MLDR out-of-domain (English test, document_length ${COLBERT_MLDR_DOC_LEN}) -> ${MLDR_OUT}"
    eval_colbert "$BEST" "$MLDR_OUT" "${MLDR_TASK_ARGS[@]}" --document_length "$COLBERT_MLDR_DOC_LEN"
else
    log "RUN_COLBERT_MLDR=0 - skipping MLDR"
fi

log "IR (ColBERT) done in $(fmt_elapsed "$START")"
for f in "$BEIR_OUT" "$MLDR_OUT"; do
    [ -f "$f" ] && log "  $(basename "$f"): $("$VPY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d.get('average_ndcg@10', float('nan'))*100:.2f} nDCG@10\")" "$f")"
done
exit 0
