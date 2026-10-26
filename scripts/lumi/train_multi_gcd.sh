#!/usr/bin/env bash
# Multi-GCD FSDP on ONE LUMI node (RCCL intra-node over XGMI/Infinity Fabric) — Track B.
# Validates FSDP2 sharding + RCCL collectives on AMD before adding the Slingshot
# inter-node layer (see train_multi_node.sbatch). torchrun spawns one worker per GCD and
# sets RANK/LOCAL_RANK/WORLD_SIZE, which palingenesis reads as usual (backend "nccl" ->
# RCCL on ROCm). All LUMI/Slurm/container/scratch specifics stay in this file.
#
# Usage (from a LUMI login node, repo synced under /scratch):
#   bash scripts/lumi/train_multi_gcd.sh            # 2 GCDs
#   PGS_NGCD=8 bash scripts/lumi/train_multi_gcd.sh # whole node
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"
TIME="${PGS_TIME:-00:20:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NGCD="${PGS_NGCD:-2}"
CONFIG="${PGS_CONFIG:-configs/rocm_smoke.yaml}"
MODEL="${PGS_MODEL:-Qwen/Qwen3-0.6B}"

export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"
BIND="-B ${SCRATCH}:${SCRATCH}"

echo "=== [1/2] ensure $MODEL is cached under scratch (login node) ==="
singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

echo "=== [2/2] FSDP across ${NGCD} GCDs on one node (torchrun, RCCL) ==="
srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --gpus-per-node="$NGCD" --cpus-per-task="$((NGCD*7))" --time="$TIME" \
  singularity exec $BIND \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    --env NCCL_ASYNC_ERROR_HANDLING=1 \
    "$SIF" \
    bash -c "cd '${REPO}' && torchrun --standalone --nproc_per_node=${NGCD} \
       -m palingenesis.train --config ${CONFIG} \
       --parallel.fsdp true --train.output_dir '${SCRATCH}/outputs/rocm_multigcd'"
