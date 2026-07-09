#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Multi-Node SLURM Training
# ═══════════════════════════════════════════════════════════════════════════════
#
# Submit:
#   sbatch scripts/train_slurm.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
#   sbatch --nodes=4 scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
#
# Override defaults via environment:
#   NGPUS=4 sbatch scripts/train_slurm.sh my_config.yaml
#
# Requirements:
#   - SLURM cluster with GPU nodes
#   - Shared filesystem for checkpoints (NFS, Lustre, etc.)
#   - NCCL-compatible network (InfiniBand recommended)
#
#SBATCH --job-name=palingenesis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j-%N.out
#SBATCH --error=logs/slurm-%j-%N.err
#SBATCH --exclusive
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG="${1:-configs/qwen35_4b/a100_80gb_multigpu.yaml}"
NGPUS="${NGPUS:-8}"

# ── Derived variables ─────────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="$SLURM_NNODES"
TOTAL_GPUS=$((NNODES * NGPUS))

# ── NCCL / CUDA environment ──────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# For InfiniBand clusters: ensure NCCL uses IB
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs

echo "┌──────────────────────────────────────────────────┐"
echo "│  palingenesis — multi-node SLURM training        │"
echo "├──────────────────────────────────────────────────┤"
echo "│  Job:     $SLURM_JOB_ID                          "
echo "│  Config:  $CONFIG                                 "
echo "│  Master:  $MASTER_ADDR:$MASTER_PORT               "
echo "│  Nodes:   $NNODES × $NGPUS GPUs = $TOTAL_GPUS total"
echo "└──────────────────────────────────────────────────┘"

# ── Launch ────────────────────────────────────────────────────────────────────
# srun launches ONE task per node (ntasks-per-node=1).
# torchrun on each node spawns NGPUS worker processes.
# rdzv_backend=c10d handles cross-node rendezvous via the master address.
#
# This is the correct SLURM + torchrun pattern:
#   srun (1 per node) → torchrun (spawns NGPUS) → python -m palingenesis.train

srun --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NGPUS" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --redirects=3 \
        --tee=3 \
        -m palingenesis.train --config "$CONFIG"
