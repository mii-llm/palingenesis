#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Single GPU Training
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/train_single_gpu.sh                                  # quickstart
#   ./scripts/train_single_gpu.sh configs/qwen35_4b/a100_80gb.yaml # specific config
#
# Works on any GPU with 24+ GB VRAM (RTX 3090, RTX 4090, A100, H100, etc.)
# All memory optimizations applied automatically.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

CONFIG="${1:-configs/quickstart.yaml}"

echo "┌──────────────────────────────────────────────────┐"
echo "│  palingenesis — single GPU training              │"
echo "├──────────────────────────────────────────────────┤"
echo "│  Config: $CONFIG"
echo "│  GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "└──────────────────────────────────────────────────┘"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone --nproc_per_node=1 -m palingenesis.train --config "$CONFIG"
