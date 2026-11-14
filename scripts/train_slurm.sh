#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Multi-Node SLURM Training
# ═══════════════════════════════════════════════════════════════════════════════
#
# Submit (from the repository root; the log is slurm-<jobid>.out in that directory):
#   sbatch scripts/train_slurm.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
#   sbatch --nodes=4 scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
#
# Extra arguments are config overrides, e.g. to continue after a requeue or a time limit:
#   sbatch scripts/train_slurm.sh my_config.yaml --train.resume_from auto
#
# GPUs per node come from the allocation (--gpus-per-node); NGPUS overrides them:
#   sbatch --gpus-per-node=4 scripts/train_slurm.sh my_config.yaml
#
# Requirements:
#   - SLURM cluster with GPU nodes
#   - A filesystem shared by every node for train.output_dir, data.pretokenize_path,
#     the data and the Hugging Face cache (NFS, Lustre, etc.)
#   - NCCL-compatible network (InfiniBand recommended)
#
# One task per node: that task is torchrun, which starts one process per GPU. Set
# --cpus-per-task to the cores of a node (they are shared by the node's processes and
# their data workers).
#
#SBATCH --job-name=palingenesis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclusive
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG="${1:-configs/qwen35_4b/a100_80gb_multigpu.yaml}"
shift || true
# One process per allocated GPU of a node (all nodes alike)
NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-8}}"

# ── Derived variables ─────────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# A port per job: two jobs whose first node is the same must not share a rendezvous
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
NNODES="$SLURM_NNODES"
TOTAL_GPUS=$((NNODES * NGPUS))

# ── NCCL / CUDA environment ──────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Abort the process on an NCCL error instead of waiting in the collective
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# InfiniBand ACK timeout, an exponent: 4.096 us x 2^22 = 17 s per retry (valid 1-22)
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Set NCCL_SOCKET_IFNAME (e.g. ib0) in the submitting shell if NCCL picks the wrong interface.

# srun does not inherit --cpus-per-task from sbatch (Slurm >= 22.05): without this the
# torchrun task, its workers and their data loaders are confined to one core.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
fi

echo "┌──────────────────────────────────────────────────┐"
echo "│  palingenesis — multi-node SLURM training        │"
echo "├──────────────────────────────────────────────────┤"
echo "│  Job:     $SLURM_JOB_ID                          "
echo "│  Config:  $CONFIG                                 "
echo "│  Master:  $MASTER_ADDR:$MASTER_PORT               "
echo "│  Nodes:   $NNODES × $NGPUS GPUs = $TOTAL_GPUS total"
echo "└──────────────────────────────────────────────────┘"

# ── Launch ────────────────────────────────────────────────────────────────────
# srun launches ONE task per node (stated here too: the job may have been
# submitted with other task counts). torchrun on each node spawns NGPUS worker
# processes, each seeing all of the node's GPUs and taking cuda:<LOCAL_RANK>.
# rdzv_backend=c10d handles cross-node rendezvous via the master address.
#
#   srun (1 per node) → torchrun (spawns NGPUS) → python -m palingenesis.train
#
# Do not start the workers with srun directly (srun python -m palingenesis.train):
# without torchrun's RANK / WORLD_SIZE every task trains alone.

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NGPUS" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --redirects=3 \
        --tee=3 \
        -m palingenesis.train --config "$CONFIG" "$@"
