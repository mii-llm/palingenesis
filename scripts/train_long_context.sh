#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Long-Context Training — Context Parallel (64K+ sequences)
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   ./scripts/train_long_context.sh                        # 8 GPUs, 64K context
#   NGPUS=4 ./scripts/train_long_context.sh configs/long_context.yaml
#
# Context Parallel (Ring Attention) shards the sequence across GPUs:
#   - 8 GPUs × 8K per GPU = 64K effective context
#   - 16 GPUs × 8K per GPU = 128K effective context
#
# Requirements: config must have parallel.context_parallel: true
set -euo pipefail

NGPUS="${NGPUS:-8}"
CONFIG="${1:-configs/long_context.yaml}"

echo "╔══════════════════════════════════════════╗"
echo "║  palingenesis — long context (CP)         ║"
echo "╚══════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo "GPUs: $NGPUS (sequence sharded across all)"
echo ""

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node="$NGPUS" -m palingenesis.train --config "$CONFIG"
