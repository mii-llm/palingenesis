#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Universal Training Launcher
# ═══════════════════════════════════════════════════════════════════════════════
#
# Auto-detects environment and launches appropriately:
#   - SLURM detected → uses srun + torchrun (multi-node)
#   - Multiple GPUs  → standalone torchrun (single-node multi-GPU)
#   - Single GPU     → standalone torchrun (single process)
#
# Usage:
#   ./run.sh                                       # auto-detect, quickstart config
#   ./run.sh configs/qwen35_4b/a100_80gb.yaml      # specific config
#   NGPUS=4 ./run.sh configs/my_config.yaml        # override GPU count
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

CONFIG="${1:-configs/quickstart.yaml}"
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

# Common environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Detect environment ────────────────────────────────────────────────────────

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # We're inside a SLURM allocation
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT="${MASTER_PORT:-29500}"
    NNODES="$SLURM_NNODES"

    echo "┌──────────────────────────────────────────────────┐"
    echo "│  palingenesis — SLURM ($NNODES nodes × $NGPUS GPUs) │"
    echo "│  Config: $CONFIG"
    echo "│  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "└──────────────────────────────────────────────────┘"

    export NCCL_IB_TIMEOUT=120
    srun --kill-on-bad-exit=1 \
        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NGPUS" \
            --rdzv_id="$SLURM_JOB_ID" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
            -m palingenesis.train --config "$CONFIG"

elif [[ "$NGPUS" -gt 1 ]]; then
    # Multiple GPUs, single node
    echo "┌──────────────────────────────────────────────────┐"
    echo "│  palingenesis — multi-GPU ($NGPUS GPUs)              │"
    echo "│  Config: $CONFIG"
    echo "└──────────────────────────────────────────────────┘"

    torchrun --standalone --nproc_per_node="$NGPUS" \
        -m palingenesis.train --config "$CONFIG"

else
    # Single GPU
    echo "┌──────────────────────────────────────────────────┐"
    echo "│  palingenesis — single GPU                       │"
    echo "│  Config: $CONFIG"
    echo "└──────────────────────────────────────────────────┘"

    torchrun --standalone --nproc_per_node=1 \
        -m palingenesis.train --config "$CONFIG"
fi
