#!/bin/bash
# Multi-node FSDP over RCCL+Slingshot on LUMI — Track B (LUMI test rig).
# Adds the inter-node interconnect layer on top of the validated intra-node path
# (train_multi_gcd.sh): RCCL talks over Slingshot (CXI) once the LUMI
# singularity-AI-bindings module binds /opt/cray + libcxi into the container.
# torchrun runs one process per GCD; node_rank comes from SLURM_NODEID, MASTER_ADDR
# from the nodelist. All LUMI/Slurm/scratch specifics live here, not in core.
#
# Prereq: the model must already be in the scratch HF cache (run
# scripts/lumi/sft_single_gcd.sh once, or any login-node prestage) — compute nodes
# have no internet. Submit with:
#   sbatch scripts/lumi/train_multi_node.sbatch
#SBATCH --job-name=pgs-rocm-mnode
#SBATCH --account=project_465002563
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_465002563/outputs/pgs_mnode_%j.log
set -euo pipefail

# LUMI container bindings for Slingshot (/opt/cray, libcxi) + Lustre filesystems.
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings/24.03

PROJECT=465002563
SCRATCH="/scratch/project_${PROJECT}"
REPO="${SCRATCH}/palingenesis_amd"
SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
NGCD=2

export HF_HOME="${SCRATCH}/hf_cache"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29500

echo "nodes=$SLURM_NNODES master=$MASTER_ADDR gcd/node=$NGCD bind=$SINGULARITY_BIND"

# One srun task per node; torchrun fans out to the GCDs. --cpu-bind=none lets torchrun's
# children use all cores on the node. RCCL env targets the Slingshot NICs (hsn0/hsn1).
srun --cpu-bind=none \
  singularity exec \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    --env MASTER_ADDR="$MASTER_ADDR" --env MASTER_PORT="$MASTER_PORT" \
    --env NCCL_SOCKET_IFNAME=hsn0,hsn1 --env NCCL_NET_GDR_LEVEL=PHB \
    --env NCCL_DEBUG=WARN \
    "$SIF" \
    bash -c "cd '${REPO}' && torchrun \
       --nnodes=${SLURM_NNODES} --node_rank=\${SLURM_NODEID} \
       --nproc_per_node=${NGCD} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
       -m palingenesis.train --config configs/rocm_smoke.yaml \
       --parallel.fsdp true --train.output_dir '${SCRATCH}/outputs/rocm_mnode'"
