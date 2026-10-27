#!/usr/bin/env bash
# Single-GCD SFT smoke run on LUMI (milestone 1) — Track B (LUMI test rig).
# Proves the training loop runs on one MI250X GCD with the ROCm-safe config
# configs/rocm_smoke.yaml. All LUMI/Slurm/container/scratch specifics live here.
#
# Flow: the model download needs internet (login node), training needs a GPU
# (dev-g, no internet) — so we pre-stage the model into an HF cache UNDER SCRATCH on
# the login node, then run training offline on dev-g.
#
# Usage (from a LUMI login node, repo synced under /scratch):
#   bash scripts/lumi/sft_single_gcd.sh
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"
TIME="${PGS_TIME:-00:20:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${PGS_MODEL:-Qwen/Qwen3-0.6B}"
CONFIG="${PGS_CONFIG:-configs/rocm_smoke.yaml}"   # any single-GCD config (SFT, DPO, ...)
OUT="${PGS_OUT:-${SCRATCH}/outputs/$(basename "${CONFIG%.yaml}")}"

# Everything under scratch (LUMI policy): HF cache and training outputs.
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"
BIND="-B ${SCRATCH}:${SCRATCH}"

echo "=== [1/2] pre-stage $MODEL into $HF_HOME (login node, has internet) ==="
singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

echo "=== [2/2] train on one GCD (dev-g, offline) ==="
# cd into the repo so the config's repo-relative paths (data, ./outputs) resolve; both
# the repo and the output dir are under scratch.
srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --time="$TIME" \
  singularity exec $BIND \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    "$SIF" \
    bash -c "cd '${REPO}' && python -m palingenesis train --config '${CONFIG}' --train.output_dir '${OUT}'"
