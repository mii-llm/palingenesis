#!/usr/bin/env bash
# RL (GRPO) smoke on one LUMI GCD — Track B. Runs the full RL loop (colocate vLLM
# rollouts + pure-Python reward + GRPO update) from configs/rocm_rl_smoke.yaml.
# Pre-stages the policy to the scratch HF cache on the login node, then trains offline.
#
#   bash scripts/lumi/rl_single_gcd.sh
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"
TIME="${PGS_TIME:-00:20:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PGS_CONFIG:-configs/rocm_rl_smoke.yaml}"
POLICY="${PGS_POLICY:-Qwen/Qwen3-0.6B}"

export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"
BIND="-B ${SCRATCH}:${SCRATCH}"

echo "=== [1/2] pre-stage policy $POLICY (login node) ==="
singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$POLICY')"

echo "=== [2/2] RL on one GCD (offline) ==="
srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=7 --time="$TIME" \
  singularity exec $BIND \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    "$SIF" \
    bash -c "cd '${REPO}' && python -m palingenesis.cli rl --config '${CONFIG}'"
