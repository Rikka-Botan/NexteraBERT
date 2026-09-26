#!/usr/bin/env bash
# Shared helpers for the ModernBERT-style evaluation scripts
# (eval_nlu.sh / eval_dpr.sh / eval_colbert.sh / eval_code.sh / run_eval_suite.sh).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
#
# Everything here reads its settings from the environment so the stage scripts
# can be run on their own or from run_eval_suite.sh with identical behaviour:
#
#   MODEL       Hugging Face repo id (RikkaBotan/NexteraBERT-Mezzoforte-220M-en) or a local
#               exported backbone directory. A repo id is snapshot-downloaded
#               once into ${HUB_CACHE_DIR} and every stage then reads that copy.
#   TOKENIZER   tokenizer id/dir (default: the ModernBERT BPE tokenizer NexteraBERT
#               trains with; the Hub repo also bundles it).
#   OUT_DIR     where every results JSON of this model goes
#               (default eval_results/<model basename>).
#   NPROC       GPUs to use (default: all visible). Single-process stages run one
#               job per GPU where a sweep allows it.
#   FORCE=1     rerun stages whose results file already exists.
#   HF_TOKEN    for private repos (huggingface_hub reads it from the environment).
#   PY          python interpreter (default: python3, else python; "uv run python"
#               also works).

# -- interpreter -------------------------------------------------------------
if [ -z "${PY:-}" ]; then
    if command -v python3 &>/dev/null; then PY=python3
    elif command -v python &>/dev/null; then PY=python
    else echo "ERROR: python3 or python not found" >&2; exit 1; fi
fi

# -- settings shared by every stage -----------------------------------------
MODEL="${MODEL:-RikkaBotan/NexteraBERT-Mezzoforte-220M-en}"
# TOKENIZER_EXPLICIT remembers whether the CALLER chose a tokenizer, before the
# default below lands (and survives run_eval_suite.sh exporting TOKENIZER to its
# stages). Without a choice, a Hugging Face baseline (MODEL=LiquidAI/LFM2.5-
# Encoder-230M, answerdotai/ModernBERT-base, ...) is tokenised by its own
# tokenizer -- see tokenizer_for below.
if [ -z "${TOKENIZER_EXPLICIT+x}" ]; then TOKENIZER_EXPLICIT="${TOKENIZER:+1}"; fi
export TOKENIZER_EXPLICIT
TOKENIZER="${TOKENIZER:-answerdotai/ModernBERT-base}"
HUB_CACHE_DIR="${HUB_CACHE_DIR:-checkpoints/hub}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL%/}")}"
OUT_DIR="${OUT_DIR:-eval_results/${MODEL_TAG}}"
FORCE="${FORCE:-0}"
MTEB_CACHE="${MTEB_CACHE:-${OUT_DIR}/mteb_cache}"

# Single-vector contrastive stage that precedes BEIR / MLDR / CoIR scoring.
#   st-msmarco (default) ModernBERT's DPR stage run with sentence-transformers, a
#                        port of their public examples/train_st.py: mean pooling,
#                        msmarco-co-condenser triplet-hard 1.25M, Cached MNRL with
#                        per-device batch 512 (1023 in-batch negatives), 5% warmup,
#                        1 epoch, lr 8e-5 (Table 9) or the paper's sweep. This is
#                        the protocol behind their Table 1 IR (DPR) / Code columns,
#                        so these numbers are comparable to the paper -- and the
#                        same script runs bert-base-uncased as a control
#                        (MODEL=bert-base-uncased; the paper's BERT-base row is 38.9).
#   mteb-nli             the SAME stage the MTEB numbers come from: attentive pooling
#                        head + supervised-SimCSE InfoNCE on MNLI+SNLI, batch 512,
#                        max_len 64, 3 epochs, STS-B dev checkpoint selection
#                        (OptiBERT App. D.2). Retrieval is then scored with that
#                        head, so every single-vector number in the suite shares
#                        one embedding recipe with the MTEB run.
#   retrieval-msmarco    this repo's own re-implementation of that DPR stage
#                        (finetune_contrastive.py: InfoNCE, batch 64, LLRD) --
#                        kept for reference, NOT the comparable one.
# RETRIEVAL_MODEL points at an already-trained contrastive checkpoint (a
# train_st_dpr.py / finetune_contrastive.py output) and skips the training stage.
RETRIEVAL_PROTOCOL="${RETRIEVAL_PROTOCOL:-st-msmarco}"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-}"
case "$RETRIEVAL_PROTOCOL" in
    st-msmarco)        _PROTOCOL_LR=8e-5 ;;   # Table 9: ModernBERT-base single-vector
    mteb-nli)          _PROTOCOL_LR=5e-5 ;;   # OptiBERT / SimCSE supervised lr
    retrieval-msmarco) _PROTOCOL_LR=8e-5 ;;
    *) echo "ERROR: RETRIEVAL_PROTOCOL must be st-msmarco, mteb-nli or retrieval-msmarco (got ${RETRIEVAL_PROTOCOL})" >&2; exit 1 ;;
esac
# The st-msmarco stage trains through sentence-transformers' HF Trainer, which
# needs accelerate; say so before spending time on the download. A stage that
# trains nothing (eval_nanobeir.sh zero-shot) sets NO_CONTRASTIVE_STAGE=1 first.
if [ "$RETRIEVAL_PROTOCOL" = "st-msmarco" ] && [ -z "${RETRIEVAL_MODEL:-}" ] \
        && [ "${NO_CONTRASTIVE_STAGE:-0}" != "1" ] \
        && ! "$PY" -c 'import accelerate' 2>/dev/null; then
    echo "ERROR: RETRIEVAL_PROTOCOL=st-msmarco needs accelerate>=1.1.0 (sentence-transformers' trainer)." >&2
    echo "       Install the eval extras: $PY -m pip install -e '.[eval]'   (or: pip install 'accelerate>=1.1.0')" >&2
    exit 1
fi
DPR_LR="${DPR_LR:-$_PROTOCOL_LR}"
DPR_LRS="${DPR_LRS:-}"                      # sweep list, e.g. "1e-5 2e-5 3e-5 5e-5 8e-5 1e-4"; empty = DPR_LR only
DPR_BATCH_SIZE="${DPR_BATCH_SIZE:-}"        # empty = the protocol's own (512 st-msmarco / 512 nli / 16 msmarco)
DPR_MAX_LEN="${DPR_MAX_LEN:-}"              # empty = the protocol's own (model max for st-msmarco / 64 nli / 256 msmarco)
DPR_EXTRA_ARGS="${DPR_EXTRA_ARGS:-}"        # extra args for the training script (e.g. "--mini_batch_size 8" on OOM)
# Cost of the contrastive stage.
#   DPR_BUDGET=full (default)  the protocol as published: 1.25M MS MARCO triplets =
#                    2441 steps at batch 512. The only budget behind a paper-comparable
#                    BEIR / MLDR / CoIR number.
#   DPR_BUDGET=lite  the same recipe (loss, batch 512 = 1023 negatives, lr, 5% warmup,
#                    pooling) on the first DPR_LITE_SAMPLES triplets (250k = 488 steps,
#                    1/5 of the compute). For the quick comparisons only
#                    (eval_nanobeir.sh NANOBEIR_MODE=dpr, eval_longembed.sh): every model
#                    of a table gets the same budget, none is comparable to a paper. A
#                    lite checkpoint lives in <OUT_DIR>/dpr_lite, records its protocol as
#                    "st-msmarco-250k" and its results carry a _lite suffix, so it can
#                    never be mistaken for -- or reused as -- the full one.
#   DPR_MINI_BATCH_SIZE  forward chunk of the cached loss (train_st_dpr.py
#                    --mini_batch_size; reference 16). Memory only: the loss and its
#                    gradients are those of the full batch whatever the chunk, so a
#                    larger one is a free speed-up -- 16 MS MARCO passages of ~80 tokens
#                    leave a GPU mostly idle. Empty = 16 (full) / 64 (lite); lower it on
#                    an out-of-memory error.
DPR_BUDGET="${DPR_BUDGET:-full}"
DPR_LITE_SAMPLES="${DPR_LITE_SAMPLES:-250000}"
DPR_MINI_BATCH_SIZE="${DPR_MINI_BATCH_SIZE:-}"
case "$DPR_BUDGET" in
    full) DPR_SUBDIR="dpr"; DPR_TAG="" ;;
    lite) DPR_SUBDIR="dpr_lite"; DPR_TAG="_lite"
          DPR_MINI_BATCH_SIZE="${DPR_MINI_BATCH_SIZE:-64}"
          # the stages behind the paper's table refuse it rather than file a reduced
          # checkpoint under beir_dpr.json / mldr_*_dpr.json / code_results.json
          if [ "${ALLOW_DPR_LITE:-0}" != "1" ]; then
              echo "ERROR: DPR_BUDGET=lite is only for eval_nanobeir.sh / eval_longembed.sh (this stage reports paper-comparable numbers)" >&2
              exit 1
          fi ;;
    *) echo "ERROR: DPR_BUDGET must be full or lite (got ${DPR_BUDGET})" >&2; exit 1 ;;
esac
MLDR_ID_BATCH_SIZE="${MLDR_ID_BATCH_SIZE:-}"   # empty = 32 (st) / 8 (finetune_contrastive); the paper states none
# MLDR in-domain lr: the paper gives none and the DPR lr (8e-5) was destructive
# (36.3 OOD -> 29.5 ID). Default: a single conservative point, 2e-5. Give several
# values ("1e-5 2e-5 5e-5") to sweep; the sweep is selected on held-out MLDR train
# queries (TripletEvaluator accuracy recorded by train_st_dpr.py), then one test run.
MLDR_ID_LRS="${MLDR_ID_LRS:-2e-5}"
BEIR_MAX_LEN="${BEIR_MAX_LEN:-512}"         # BEIR scoring length
MLDR_MAX_LEN="${MLDR_MAX_LEN:-8192}"        # MLDR scoring length (native context)
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-64}"
# Precision of the sentence-transformers scoring path: fp32 (default, what every
# existing result was computed with) | tf32 | bf16. fp32 is several times slower on
# an Ampere+/Hopper GPU; pick ONE value for all models of a comparison.
RETRIEVAL_DTYPE="${RETRIEVAL_DTYPE:-fp32}"
# Encode batch for the 8192-token stages (MLDR, CoIR); empty = RETRIEVAL_BATCH_SIZE.
# LFM2.5-Encoder's default attention path builds a dense (B,1,T,T) mask, ~134 MB
# per sample at 8192 in bf16: use 8-16 there, or ATTN_IMPLEMENTATION=flash_attention_2.
LONG_RETRIEVAL_BATCH_SIZE="${LONG_RETRIEVAL_BATCH_SIZE:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"   # sentence-transformers stages only
SELECTION_TASKS="${SELECTION_TASKS:-NFCorpus,SciFact,TRECCOVID,FiQA2018}"  # App. E.2 sweep subset
MLDR_TASK_ARGS=(--tasks MultiLongDocRetrieval --languages eng --eval_splits test)

# -- logging -----------------------------------------------------------------
# Logs go to stderr so helpers can be used inside $(...) without polluting the
# value they print.
log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
fmt_elapsed() { local s=$(( $(date +%s) - $1 )); echo "$(( s / 60 ))m $(( s % 60 ))s"; }
done_or_skip() {   # done_or_skip <results file>  -> 0 = already there, skip the stage
    if [ -f "$1" ] && [ "$FORCE" != "1" ]; then
        log "exists, skipping (FORCE=1 to rerun): $1"
        return 0
    fi
    return 1
}
# drop_stale_result <results file> <model dir>: a results file OLDER than the
# checkpoint it claims to score was computed from a previous training run in the
# same directory (retrained after a divergence, a fix, a new lr...). "exists,
# skipping" would then report the old model's numbers for the new one, so it is
# moved aside (<file>.stale) and the stage runs again.
drop_stale_result() {
    local out="$1" marker="$2/contrastive_run.json"
    if [ -f "$out" ] && [ -f "$marker" ] && [ "$out" -ot "$marker" ]; then
        log "stale (older than ${marker}), moved to ${out}.stale: ${out}"
        mv -f "$out" "${out}.stale"
    fi
    return 0
}
# retire_dead_checkpoint <checkpoint dir>: a run that diverged used to be saved like
# any other -- NaN weights plus the contrastive_run.json that marks the stage done --
# and every later pipeline run then skipped training and scored the dead model
# (nDCG@10 = 0.0, silently). Such a directory is moved aside so the stage retrains.
retire_dead_checkpoint() {
    local dir="$1" rc=0
    [ -f "${dir}/contrastive_run.json" ] || return 0
    "$PY" scripts/check_neobert_checkpoint.py --finite_only "$dir" > /dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 3 ]; then
        local aside="${dir}.diverged.$(date +%Y%m%d-%H%M%S)"
        log "checkpoint holds NaN/inf weights (its training diverged); moved to ${aside} -- retraining"
        mv "$dir" "$aside"
    fi
    return 0
}

# -- hardware ----------------------------------------------------------------
count_gpus() {
    "$PY" -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1
}
NPROC="${NPROC:-$(count_gpus)}"

# -- model resolution --------------------------------------------------------
# resolve_model <repo id or dir> -> prints the local backbone directory.
# The in-repo loaders (evaluate_glue.py, finetune_contrastive.py, the mteb
# encoder) read config.json + model.safetensors from a directory; a Hub id is
# therefore materialised once with snapshot_download. The PyLate/ColBERT stage
# can take the repo id directly (it loads through AutoModel + trust_remote_code)
# but is pointed at the same snapshot so every stage scores identical weights.
resolve_model() {
    local src="$1"
    if [ -d "$src" ]; then
        if [ ! -f "${src}/config.json" ]; then
            log "ERROR: ${src} has no config.json (not an exported backbone directory)"
            return 1
        fi
        printf '%s\n' "$src"
        return 0
    fi
    local dest="${HUB_CACHE_DIR}/${src//\//__}"
    if [ ! -f "${dest}/config.json" ]; then
        log "Downloading ${src} -> ${dest}"
        # stdout is this function's return value (callers do $(resolve_model ...)),
        # so everything the download prints must go to stderr.
        "$PY" - "$src" "$dest" 1>&2 <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=dest, repo_type="model")
print(f"  snapshot -> {dest}", file=sys.stderr)
EOF
    fi
    if [ ! -f "${dest}/model.safetensors" ] && [ ! -f "${dest}/pytorch_model.bin" ]; then
        log "ERROR: ${dest} carries no encoder weights (model.safetensors / pytorch_model.bin)"
        return 1
    fi
    printf '%s\n' "$dest"
}

# model_type <model dir> -> prints config.json's model_type ("nexterabert" for
# an export of this repo; "modernbert", "neobert", "bert", ... for a Hugging
# Face baseline that scripts/eval_mteb.sh pushes through the same stage).
model_type() {
    "$PY" -c 'import json, sys
try:
    print(json.load(open(sys.argv[1] + "/config.json", encoding="utf-8")).get("model_type", ""))
except Exception:
    print("")' "$1"
}

# tokenizer_for <model dir> -> prints the tokenizer to pass for that model: the
# caller's TOKENIZER when one was chosen, the model's own for a Hugging Face
# baseline (its snapshot / every directory fine-tuned from it carries it), the
# ModernBERT default for a NexteraBERT export.
tokenizer_for() {
    if [ -z "$TOKENIZER_EXPLICIT" ] && [ -f "$1/config.json" ] \
            && [ "$(model_type "$1")" != "nexterabert" ]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$TOKENIZER"
    fi
}

# -- single-vector stage helpers (shared by eval_dpr.sh and eval_code.sh) -----
# train_dpr <backbone dir> <lr> <output dir> [extra finetune_contrastive args...]
# The contrastive stage under ${RETRIEVAL_PROTOCOL} (see above). Batch size and
# max_len are passed only when set, so the protocol's own values apply otherwise.
train_dpr() {
    local backbone="$1" lr="$2" out="$3"; shift 3
    retire_dead_checkpoint "$out"
    if [ -f "${out}/contrastive_run.json" ] && [ "$FORCE" != "1" ]; then
        log "contrastive checkpoint exists, skipping training: ${out}"
        return 0
    fi
    local extra=()
    [ -n "$DPR_BATCH_SIZE" ] && extra+=(--batch_size "$DPR_BATCH_SIZE")
    [ "$DPR_BUDGET" = "lite" ] && extra+=(--max_samples "$DPR_LITE_SAMPLES")
    log "contrastive training (${RETRIEVAL_PROTOCOL}, budget ${DPR_BUDGET}): lr=${lr} bs=${DPR_BATCH_SIZE:-protocol} -> ${out}"
    if [ "$RETRIEVAL_PROTOCOL" = "st-msmarco" ]; then
        [ -n "$DPR_MINI_BATCH_SIZE" ] && extra+=(--mini_batch_size "$DPR_MINI_BATCH_SIZE")
        [ -n "$DPR_MAX_LEN" ] && extra+=(--max_seq_length "$DPR_MAX_LEN")
        [ -n "$ATTN_IMPLEMENTATION" ] && extra+=(--attn_implementation "$ATTN_IMPLEMENTATION")
        # shellcheck disable=SC2086
        "$PY" scripts/train_st_dpr.py --dataset msmarco \
            --model "$backbone" --output_dir "$out" --lr "$lr" "${extra[@]}" \
            ${DPR_EXTRA_ARGS} "$@"
    else
        [ -n "$DPR_MAX_LEN" ] && extra+=(--max_len "$DPR_MAX_LEN")
        # shellcheck disable=SC2086
        "$PY" scripts/finetune_contrastive.py \
            --protocol "$RETRIEVAL_PROTOCOL" \
            --model "$backbone" --tokenizer "$(tokenizer_for "$backbone")" \
            --output_dir "$out" \
            --lr "$lr" "${extra[@]}" \
            ${DPR_EXTRA_ARGS} "$@"
    fi
}

# train_mldr_id <contrastive checkpoint> <lr> <output dir>: the paper's 3.1.3
# "Single Vector - In Domain" pass, continuing from the DPR checkpoint on the
# MLDR-en training split at MLDR_MAX_LEN. Same framework as the DPR stage.
train_mldr_id() {
    local src="$1" lr="$2" out="$3"
    retire_dead_checkpoint "$out"
    if [ -f "${out}/contrastive_run.json" ] && [ "$FORCE" != "1" ]; then
        log "MLDR in-domain checkpoint exists, skipping training: ${out}"
        return 0
    fi
    if [ -f "${src}/modules.json" ]; then     # a sentence-transformers checkpoint
        local bs="${MLDR_ID_BATCH_SIZE:-32}"
        log "MLDR in-domain fine-tuning (st-mldr, lr=${lr}, bs=${bs}, max_seq_length ${MLDR_MAX_LEN}) -> ${out}"
        # shellcheck disable=SC2086
        local attn=()
        [ -n "$ATTN_IMPLEMENTATION" ] && attn=(--attn_implementation "$ATTN_IMPLEMENTATION")
        "$PY" scripts/train_st_dpr.py --dataset mldr \
            --model "$src" --output_dir "$out" --lr "$lr" --batch_size "$bs" \
            --max_seq_length "$MLDR_MAX_LEN" "${attn[@]}" ${MLDR_ID_EXTRA_ARGS:-}
    else
        local bs="${MLDR_ID_BATCH_SIZE:-8}"
        log "MLDR in-domain fine-tuning (retrieval-mldr, lr=${lr}, bs=${bs}, max_len ${MLDR_MAX_LEN}) -> ${out}"
        # shellcheck disable=SC2086
        "$PY" scripts/finetune_contrastive.py \
            --protocol retrieval-mldr \
            --model "$src" --tokenizer "$(tokenizer_for "$src")" \
            --output_dir "$out" \
            --lr "$lr" --batch_size "$bs" --max_len "$MLDR_MAX_LEN" \
            ${DPR_EXTRA_ARGS} ${MLDR_ID_EXTRA_ARGS:-}
    fi
}

# resolve_retrieval_model -> prints the contrastive checkpoint to score, or nothing.
# RETRIEVAL_MODEL (an existing directory with contrastive_run.json) wins; else the
# checkpoint eval_dpr.sh selected and recorded in ${OUT_DIR}/dpr/selected.json.
resolve_retrieval_model() {
    if [ -n "$RETRIEVAL_MODEL" ]; then
        if [ ! -f "${RETRIEVAL_MODEL}/contrastive_run.json" ]; then
            log "ERROR: RETRIEVAL_MODEL=${RETRIEVAL_MODEL} has no contrastive_run.json (not a finetune_contrastive.py output)"
            return 1
        fi
        printf '%s\n' "$RETRIEVAL_MODEL"
        return 0
    fi
    local selected="${OUT_DIR}/${DPR_SUBDIR}/selected.json"   # dpr, or dpr_lite under DPR_BUDGET=lite
    if [ -f "$selected" ]; then
        "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$selected"
    fi
}

# record_selected <selected.json> <lr> <model dir> <lrs>
record_selected() {
    "$PY" -c 'import json, sys
out, proto, lr, model, lrs = sys.argv[1:6]
json.dump({"protocol": proto, "lr": float(lr), "model": model,
           "sweep": [float(x) for x in lrs.split()]}, open(out, "w"), indent=2)' \
        "$1" "$RETRIEVAL_PROTOCOL" "$2" "$3" "$4"
}

# run_json_field <json file> <key> -> prints one top-level field of a JSON file
run_json_field() {
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$1" "$2"
}

# eval_retrieval <model dir> <results json> <max_len> [evaluate_retrieval args...]
eval_retrieval() {
    local model="$1" out="$2" max_len="$3"; shift 3
    drop_stale_result "$out" "$model"
    done_or_skip "$out" && return 0
    mkdir -p "$(dirname "$out")"
    local bs="$RETRIEVAL_BATCH_SIZE" attn=()
    if [ "$max_len" -gt 2048 ] && [ -n "$LONG_RETRIEVAL_BATCH_SIZE" ]; then
        bs="$LONG_RETRIEVAL_BATCH_SIZE"
    fi
    [ -n "$ATTN_IMPLEMENTATION" ] && attn=(--attn_implementation "$ATTN_IMPLEMENTATION")
    "$PY" scripts/evaluate_retrieval.py \
        --model "$model" --tokenizer "$(tokenizer_for "$model")" \
        --max_len "$max_len" --batch_size "$bs" --st_dtype "$RETRIEVAL_DTYPE" "${attn[@]}" \
        --output "$out" --output_folder "$MTEB_CACHE" "$@"
}

# pick_lr <lr=selection.json>... -> prints the lr with the best mean nDCG@10
pick_lr() { "$PY" scripts/summarize_eval_suite.py pick-lr "$@"; }

# with_gpu <index> <command...>: pin a single-process job to one GPU
with_gpu() { local gpu="$1"; shift; CUDA_VISIBLE_DEVICES="$gpu" "$@"; }

# The 15 BEIR datasets of mteb's "BEIR" benchmark (MSMARCO scored on dev), largest
# corpus first so a GPU wave starts with the jobs that dominate wall-clock time.
BEIR_TASKS_ALL="MSMARCO,FEVER,ClimateFEVER,HotpotQA,DBPedia,NQ,QuoraRetrieval,CQADupstackRetrieval,Touche2020,TRECCOVID,FiQA2018,SCIDOCS,ArguAna,SciFact,NFCorpus"

# run_wave <fn> <item>... : call "<fn> <item> <gpu>" for every item, NPROC jobs at
# a time (one per GPU), failing if any job fails. Items are processed in order, so
# put the expensive ones first.
run_wave() {
    local fn="$1"; shift
    local i=0 pids=() item pid
    for item in "$@"; do
        "$fn" "$item" "$(( i % NPROC ))" &
        pids+=("$!")
        i=$(( i + 1 ))
        if [ $(( i % NPROC )) -eq 0 ]; then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

# parallel_tasks <eval fn> <model> <out.json> <tasks csv> [extra args...]
# Scores retrieval tasks one job per GPU (NPROC at a time) and merges the per-task
# results into <out.json>; with one GPU it is a single call with --tasks. The eval
# function is called as "<fn> <model> <part.json> --tasks <task> [extra...]"; a part
# that already exists is reused, so an interrupted run resumes. MSMARCO gets
# --eval_splits dev, which is how the BEIR benchmark scores it.
parallel_tasks() {
    # NB: bash scopes locals dynamically -- the job below runs inside run_wave,
    # whose own local "fn" would shadow a variable of that name here.
    local eval_fn="$1" model="$2" out="$3" tasks="$4"; shift 4
    drop_stale_result "$out" "$model"
    done_or_skip "$out" && return 0
    if [ "$NPROC" -le 1 ]; then
        "$eval_fn" "$model" "$out" --tasks "$tasks" "$@"
        return
    fi
    local stem="${out%.json}" task part
    local -a task_list parts
    IFS=',' read -r -a task_list <<< "$tasks"
    _parallel_task_job() {   # <task> <gpu>
        local t="$1" gpu="$2"
        local p="${stem}.${t}.json"    # separate statement: expansions in one `local`
        local extra=()                  # happen before any of its assignments
        drop_stale_result "$p" "$model"
        if [ -f "$p" ] && [ "$FORCE" != "1" ]; then
            log "  ${t}: part exists, reused"
            return 0
        fi
        [ "$t" = "MSMARCO" ] && extra=(--eval_splits dev)
        log "  ${t} -> GPU ${gpu}"
        with_gpu "$gpu" "$eval_fn" "$model" "$p" --tasks "$t" "${extra[@]}" "${PARALLEL_EXTRA[@]}" \
            > "${stem}.${t}.log" 2>&1
    }
    PARALLEL_EXTRA=("$@")
    log "scoring ${#task_list[@]} tasks, ${NPROC} at a time (logs: ${stem}.<task>.log)"
    run_wave _parallel_task_job "${task_list[@]}"
    for task in "${task_list[@]}"; do parts+=("${stem}.${task}.json"); done
    "$PY" scripts/summarize_eval_suite.py merge "$out" "${parts[@]}"
}
