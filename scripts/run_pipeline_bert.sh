#!/usr/bin/env bash
# Full NexteraBERT BERT-mode pipeline:
#   data download -> phase 1 (1024) -> phase 2 (8192) -> export -> GLUE eval
#   -> SimCSE contrastive fine-tune -> MTEB eval -> Hub upload
#   -> trust_remote_code verification.
#
# Standard MLM pretraining with 80/10/10 masking (no ELECTRA discriminator).
# Two-phase ModernBERT-style curriculum: the bulk of training runs at 1024
# (120 B tokens, OptiBERT recipe), then a brief long-context extension at the full
# 8192 target length (10 B tokens, weights warm-started from phase 1).
# Both budgets are set as train_tokens in the configs; max_steps is derived.
#
# Dataset: FineWeb-Edu (90.25%) + DCLM (4.75%) + StarCoderData code (5%), ~121 B
# tokens, ModernBERT BPE. Phase 1 uses one document per example (variable length;
# dynamic padding at collate); phase 2 re-reads the same mix as fixed 8192-token
# unpad-packed blocks (~10 B); see scripts/build_data.sh.
#
# Usage:
#   bash scripts/run_pipeline_bert.sh
#
# Before running, fill in the "User settings" section below.
set -euo pipefail

###############################################################################
# User settings - edit these before running
###############################################################################

# Weights & Biases ------------------------------------------------------------
WANDB_API_KEY="${WANDB_API_KEY:-}"                # export WANDB_API_KEY in your shell/CI; empty = interactive login
WANDB_PROJECT="nexterabert"
WANDB_ENTITY=""
WANDB_RUN_PREFIX="mezzoforte-bert"

# Hugging Face Hub ------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"                          # export HF_TOKEN in your shell/CI; empty = interactive login / skip upload
HF_REPO_ID="${HF_REPO_ID:-}"                      # e.g. <user>/NexteraBERT-Mezzoforte-220M-en; empty = skip the upload
HF_PRIVATE="true"

# Hardware ---------------------------------------------------------------------
# GPUs for both pretraining phases and for GLUE's task-parallel runs. The configs'
# batch_size / grad_accum are PER GPU, and the released models were trained on ONE
# GPU: 1024 sequences per optimizer step in phase 1, 128 blocks of 8192 tokens in
# phase 2. More GPUs multiply that effective batch; to keep the released recipe on
# N GPUs, divide grad_accum in both configs by N.
NPROC="${NPROC:-8}"

# Pretokenized data (built by scripts/build_data.sh) ---------------------------
DATA_REPO_ID="${DATA_REPO_ID:-RikkaBotan/NexteraBERT-data-mix}"

# GLUE evaluation --------------------------------------------------------------
# evaluate_glue.py fine-tunes RTE/MRPC/STS-B/QNLI from an MNLI-tuned encoder and
# averages over the seeds below, with its per-task hyperparameters. Each task caps
# how many of these seeds it uses, as ModernBERT does (rte/mrpc/stsb up to 5, cola
# 4, sst2 3, the large stable tasks - MNLI/QQP/QNLI - just one), and runs
# early-stop after 2 epochs without val improvement. torch.compile + TF32 are on
# by default on CUDA. "all" = the 8 standard GLUE tasks (WNLI excluded).
# Override in the shell, e.g.
#   GLUE_TASKS="sst2 rte" GLUE_SEEDS="1234" bash scripts/run_pipeline_bert.sh
GLUE_TASKS="${GLUE_TASKS:-all}"
GLUE_SEEDS="${GLUE_SEEDS:-19 8364 717 10536 90166}"   # MosaicBERT/ModernBERT 5-seed set

# MTEB evaluation --------------------------------------------------------------
# MTEB scores one vector per text, and an MLM backbone was never trained to place
# related texts near each other - published MTEB numbers (OptiBERT Table 5, App.
# D.2) are measured after a supervised-SimCSE stage with an attentive pooling
# head. The stage runs that first (scripts/finetune_contrastive.py --protocol
# mteb-nli, single GPU: InfoNCE negatives come from the batch, so it is
# deliberately not DDP-sharded) and scores its output.
# The default benchmark is the full English suite (41 tasks, several GPU-hours);
# narrow it with MTEB_BENCHMARK / MTEB_TASK_TYPES / MTEB_EXCLUDE_TASKS / MTEB_MAX_TASKS, e.g.
#   MTEB_TASK_TYPES="STS,PairClassification" MTEB_MAX_TASKS=4 bash scripts/run_pipeline_bert.sh
#   MTEB_EXCLUDE_TASKS="MindSmallReranking" bash scripts/run_pipeline_bert.sh
#   RUN_MTEB=0 bash scripts/run_pipeline_bert.sh                 # skip the stage entirely
RUN_MTEB="${RUN_MTEB:-1}"
MTEB_BENCHMARK="${MTEB_BENCHMARK:-MTEB(eng, v2)}"
MTEB_TASK_TYPES="${MTEB_TASK_TYPES:-}"          # e.g. "STS,Retrieval"; empty = all
MTEB_EXCLUDE_TASKS="${MTEB_EXCLUDE_TASKS:-}"    # e.g. "MindSmallReranking"
MTEB_MAX_TASKS="${MTEB_MAX_TASKS:-0}"           # 0 = every task in the benchmark
MTEB_MAX_LEN="${MTEB_MAX_LEN:-512}"
MTEB_BATCH_SIZE="${MTEB_BATCH_SIZE:-64}"

###############################################################################
# Derived (no need to edit below this line)
###############################################################################

if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: python3 or python not found" >&2; exit 1
fi
PIP="${PY} -m pip"

TOKENIZER="answerdotai/ModernBERT-base"
PHASE1_CFG="configs/pretrain_bert_phase1.yaml"
PHASE2_CFG="configs/pretrain_bert_phase2.yaml"
PHASE1_CKPT="checkpoints/mezzoforte_bert/phase1/final.pt"
OUTPUT_DIR="checkpoints/mezzoforte_bert/phase2"
BACKBONE="checkpoints/mezzoforte_bert/phase2/backbone"
# MTEB stage outputs. The mteb result cache is keyed by the encoder's reported
# model name, which comes from the model directory's basename - so the cache, the
# SimCSE output directory and the results file are kept distinct per pipeline.
# Sharing them would let one model's cached task scores be reported for another.
SIMCSE_DIR="checkpoints/mezzoforte_bert/phase2/simcse"
MTEB_OUT="mteb_simcse_bert.json"
MTEB_CACHE="mteb_cache_bert"

WANDB_ARGS="--wandb_project ${WANDB_PROJECT}"
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
fi

START_TS=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -- Pre-flight: install dependencies ---------------------------------------
log "Installing dependencies ..."
$PIP install -e ".[train,eval]" zstandard --quiet

# -- Pre-flight: W&B login --------------------------------------------------
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
    log "W&B: using API key from env"
elif ! $PY -c "import wandb; wandb.login()" 2>/dev/null; then
    log "wandb not logged in. Running 'wandb login' ..."
    $PY -m wandb login
fi
log "W&B: project=${WANDB_PROJECT}  entity=${WANDB_ENTITY:-'(personal)'}"

# -- Pre-flight: HF login (only if upload is requested) ---------------------
if [ -n "$HF_REPO_ID" ]; then
    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
        log "HF: using token from env"
    elif ! $PY -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null; then
        log "Hugging Face not logged in. Set HF_TOKEN or run: huggingface-cli login"
        exit 1
    fi
    log "HF upload target: ${HF_REPO_ID} (private=${HF_PRIVATE})"
fi

# -- 0. Download pretokenized data from HF -----------------------------------
download_shard() {
  local file="$1"
  if [ -f "$file" ] && [ -f "${file}.meta" ]; then
    log "Data shard already exists: ${file}"
    return
  fi
  log "Downloading ${file} from ${DATA_REPO_ID} ..."
  $PY -c "
from huggingface_hub import hf_hub_download
import sys, os
repo, f = sys.argv[1], sys.argv[2]
for name in [f, f + '.meta', f + '.lens']:
    try:
        hf_hub_download(repo_id=repo, filename=name, repo_type='dataset',
                        local_dir='.', local_dir_use_symlinks=False)
        print(f'  downloaded {name}')
    except Exception:
        if not name.endswith('.lens'):
            raise  # .lens is a variable-length sidecar; fixed-block shards lack it
" "$DATA_REPO_ID" "$file"
}

if [ -z "$DATA_REPO_ID" ]; then
  log "ERROR: DATA_REPO_ID is empty. Build shards first with build_data.sh and set the repo ID."
  exit 1
fi
download_shard data/mix_1024.bin
download_shard data/mix_eval_1024.bin
download_shard data/mix_8192.bin
download_shard data/mix_eval_8192.bin

# -- 1. Phase 1: pretraining (1024) -------------------------------------------
P1_START=$(date +%s)
log "Phase 1: pretraining [BERT] (1024), ${NPROC} GPUs"
$PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    scripts/pretrain.py --config "$PHASE1_CFG" \
    --run_name "${WANDB_RUN_PREFIX}-phase1" \
    ${WANDB_ARGS}
P1_END=$(date +%s)
P1_ELAPSED=$((P1_END - P1_START))
P1_MINUTES=$((P1_ELAPSED / 60))
P1_SECONDS=$((P1_ELAPSED % 60))
log "Phase 1 done in ${P1_MINUTES}m ${P1_SECONDS}s"

# -- 2. Phase 2: long-context extension (8192) --------------------------------
# pretrain.py silently skips a missing --resume path, so fail fast here instead
# of burning GPU-hours training the 8192 phase from random weights.
if [ ! -f "$PHASE1_CKPT" ]; then
    log "ERROR: phase 1 checkpoint not found: ${PHASE1_CKPT}"
    exit 1
fi
P2_START=$(date +%s)
log "Phase 2: long-context extension [BERT] (8192), ${NPROC} GPUs"
$PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    scripts/pretrain.py --config "$PHASE2_CFG" \
    --run_name "${WANDB_RUN_PREFIX}-phase2" \
    ${WANDB_ARGS}
P2_END=$(date +%s)
P2_ELAPSED=$((P2_END - P2_START))
P2_MINUTES=$((P2_ELAPSED / 60))
P2_SECONDS=$((P2_ELAPSED % 60))
log "Phase 2 done in ${P2_MINUTES}m ${P2_SECONDS}s"

log "Backbone exported to ${BACKBONE}"

# The MLM output head lives outside the encoder, so save_pretrained does not write
# it; export_bert_backbone / export_cocolm_backbone add it separately. Check now
# rather than after GLUE and the upload -- without it the published model predicts
# noise, and the Hub verification at the end would be the first thing to notice.
if [ ! -f "${BACKBONE}/mlm_head.safetensors" ]; then
    # A backbone exported by an older scripts/pretrain.py has no head file, but the
    # training checkpoint beside it still holds the trained head -- which is exactly
    # how model_pplbench.py manages to score such a backbone. Recover it into the
    # directory so the upload ships the same weights the benchmark reads, instead of
    # publishing a model whose masked-LM output is noise.
    log "MLM head not in ${BACKBONE}; recovering it from ${OUTPUT_DIR} ..."
    $PY -c "import sys; sys.path.insert(0, 'src'); \
from nexterabert.export import ensure_mlm_head; \
sys.exit(0 if ensure_mlm_head('${BACKBONE}') else 1)" || {
        log "ERROR: no MLM head found for ${BACKBONE}."
        log "  Looked in the directory and in ${OUTPUT_DIR}/final.pt (and step_*.pt)."
        log "  If final.pt is gone the head cannot be recovered - it is trained"
        log "  weights, not something that can be recomputed."
        exit 1
    }
fi
log "MLM head present: ${BACKBONE}/mlm_head.safetensors"

# -- 3. GLUE evaluation ------------------------------------------------------
GLUE_WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
    GLUE_WANDB_ARGS="--wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_PREFIX}-glue"
    if [ -n "$WANDB_ENTITY" ]; then
        GLUE_WANDB_ARGS="${GLUE_WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi
# GLUE runs task-parallel: one task per single GPU, tasks concurrent across the
# ${NPROC} GPUs (NOT DDP-sharded). This keeps OptiBERT's per-task batch sizes - which
# are TOTALS - intact; DDP would multiply the effective batch by the GPU count and
# hurt the small, batch-sensitive tasks. The MNLI transfer source trains first and
# its dependents (rte/mrpc/stsb/qnli) wait for it, all handled inside the script.
log "Running GLUE (tasks=[${GLUE_TASKS}] seeds=[${GLUE_SEEDS}], MNLI->{rte,mrpc,stsb,qnli} transfer), task-parallel over ${NPROC} GPUs ..."
$PY scripts/evaluate_glue.py \
    --model "$BACKBONE" \
    --tokenizer "$TOKENIZER" \
    --tasks ${GLUE_TASKS} \
    --seeds ${GLUE_SEEDS} \
    --task_parallel \
    --num_gpus "$NPROC" \
    --output glue_results_bert.json \
    ${GLUE_WANDB_ARGS}
log "GLUE results saved to glue_results_bert.json"

# -- 3b. MTEB evaluation -----------------------------------------------------
if [ "$RUN_MTEB" = "1" ]; then
    MTEB_START=$(date +%s)
    MTEB_READY=1
    log "SimCSE contrastive fine-tuning (MTEB protocol, MNLI+SNLI triplets) -> ${SIMCSE_DIR} ..."
    if ! $PY scripts/finetune_contrastive.py \
            --protocol mteb-nli \
            --model "$BACKBONE" \
            --tokenizer "$TOKENIZER" \
            --output_dir "$SIMCSE_DIR"; then
        # Scoring the raw backbone instead would quietly report a mean-pooled number
        # under the SimCSE protocol's name, so stop the stage.
        log "ERROR: SimCSE fine-tuning failed - skipping MTEB"
        MTEB_READY=0
    fi

    if [ "$MTEB_READY" = "1" ]; then
        # an array, not a string: the benchmark name contains a space ("MTEB(eng, v2)")
        # --pooling auto picks up the trained attentive pooling head
        MTEB_ARGS=(--model "$SIMCSE_DIR" --tokenizer "$TOKENIZER"
                   --pooling auto --benchmark "$MTEB_BENCHMARK"
                   --max_len "$MTEB_MAX_LEN" --batch_size "$MTEB_BATCH_SIZE"
                   --output "$MTEB_OUT" --cache_dir "$MTEB_CACHE" --skip_errors)
        if [ -n "$MTEB_TASK_TYPES" ]; then
            MTEB_ARGS+=(--task_types "$MTEB_TASK_TYPES")
        fi
        if [ -n "$MTEB_EXCLUDE_TASKS" ]; then
            MTEB_ARGS+=(--exclude_tasks "$MTEB_EXCLUDE_TASKS")
        fi
        if [ "$MTEB_MAX_TASKS" != "0" ]; then
            MTEB_ARGS+=(--max_tasks "$MTEB_MAX_TASKS")
        fi
        log "Running MTEB (${MTEB_BENCHMARK}) on ${SIMCSE_DIR} ..."
        $PY scripts/evaluate_mteb.py "${MTEB_ARGS[@]}"
        MTEB_END=$(date +%s)
        MTEB_ELAPSED=$((MTEB_END - MTEB_START))
        MTEB_MINUTES=$((MTEB_ELAPSED / 60))
        MTEB_SECONDS=$((MTEB_ELAPSED % 60))
        log "MTEB done in ${MTEB_MINUTES}m ${MTEB_SECONDS}s; results -> ${MTEB_OUT}"
    fi
else
    log "RUN_MTEB=0 - skipping MTEB evaluation"
fi

# -- 4. Upload to Hub ---------------------------------------------------------
if [ -n "$HF_REPO_ID" ]; then
    HF_ARGS="--model ${BACKBONE} --repo_id ${HF_REPO_ID} --size mezzoforte"
    if [ "$HF_PRIVATE" = "true" ]; then
        HF_ARGS="${HF_ARGS} --private"
    fi
    log "Uploading to ${HF_REPO_ID} ..."
    $PY scripts/upload_to_hub.py ${HF_ARGS}
    log "Upload complete: https://huggingface.co/${HF_REPO_ID}"

    # -- 5. Verify the upload is actually usable ------------------------------
    # push_to_hub ships the encoder, the MLM head, the NexteraBERT source modules
    # and an auto_map. This proves the result loads through the ordinary HF entry
    # points: the Auto* calls run in a subprocess where the local nexterabert
    # package is blocked from importing, so a pass means a stranger with nothing
    # but `pip install transformers` gets the same model. It also compares the
    # remote-code activations against the local implementation, which is the only
    # thing that catches a silently missing head -- that produces a correctly
    # shaped tensor full of noise rather than an error.
    log "Verifying ${HF_REPO_ID} loads with trust_remote_code ..."
    $PY scripts/verify_hub_model.py --repo_id "${HF_REPO_ID}" \
        --expect mlm --local "${BACKBONE}"
    log "Verification passed"
else
    log "HF_REPO_ID is empty - skipping Hub upload"
fi

TOTAL=$(( $(date +%s) - START_TS ))
log "Pipeline complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
