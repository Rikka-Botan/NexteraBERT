#!/usr/bin/env bash
# Launch DDP pretraining on all visible GPUs of a single node.
#
#   bash scripts/train_ddp.sh [extra pretrain.py args...]
#
# Examples:
#   bash scripts/train_ddp.sh                                   # phase 1
#   CONFIG=configs/pretrain_bert_phase2.yaml bash scripts/train_ddp.sh
#   NPROC=8 bash scripts/train_ddp.sh --grad_accum 1   # released phase-1 batch on 8 GPUs
set -euo pipefail

NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)')}"
CONFIG="${CONFIG:-configs/pretrain_bert_phase1.yaml}"

echo "Launching DDP on ${NPROC} process(es)"
torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    scripts/pretrain.py \
    --config "${CONFIG}" \
    "$@"
