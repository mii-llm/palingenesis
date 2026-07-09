#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Multi-GPU Training (Single Node)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/train_multi_gpu.sh                                    # 8 GPUs, default
#   ./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
#   NGPUS=4 ./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
#
# FSDP2 shards model across GPUs. Supports models up to 35B on 8× A100-80GB.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

NGPUS="${NGPUS:-8}"
CONFIG="${1:-configs/qwen35_4b/a100_80gb_multigpu.yaml}"

echo "┌──────────────────────────────────────────────────┐"
echo "│  palingenesis — multi-GPU ($NGPUS GPUs)              │"
echo "├──────────────────────────────────────────────────┤"
echo "│  Config: $CONFIG"
echo "│  GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/│    /' || echo "│    (detection failed)"
echo "└──────────────────────────────────────────────────┘"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --redirects=3 \
    --tee=3 \
    -m palingenesis.train --config "$CONFIG"
