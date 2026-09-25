#!/bin/bash
# Multi-node RL: one torchrun agent per node, one rank per GPU, FSDP2 policy + one colocated
# vLLM engine per rank. Submit with:  sbatch examples/rl/slurm.sbatch configs/rl_math.yaml
#
# Preemption / time limit: Slurm sends SIGUSR1 ten minutes before the end; the trainer saves a
# resumable checkpoint and exits, and `--train.resume_from auto` continues from it on requeue.

#SBATCH --job-name=pgs-rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1                 # one torchrun agent per node
#SBATCH --gpus-per-node=8                   # every GPU visible to every rank (NCCL P2P); ranks pick theirs
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch examples/rl/slurm.sbatch <config.yaml> [--section.option value ...]}
shift

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

srun --label --kill-on-bad-exit=1 \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$SLURM_GPUS_ON_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --signals-to-handle=SIGTERM,SIGINT,SIGUSR1 \
    --log-dir="logs/$SLURM_JOB_ID" --tee=3 --local-ranks-filter=0 \
    -m palingenesis.rl.trainer --config "$CONFIG" --train.resume_from auto "$@"
