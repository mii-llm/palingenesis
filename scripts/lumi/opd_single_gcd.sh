#!/usr/bin/env bash
# End-to-end OPD distillation on one LUMI GCD — Track B (LUMI test rig).
# Runs the full on-policy loop (colocate vLLM student sampling + frozen hf teacher
# scoring + reverse-KL update) from configs/rocm_opd_smoke.yaml. Pre-stages both models
# to the scratch HF cache on the login node (compute nodes are offline), then distills.
#
# Usage (from a LUMI login node, repo synced under /scratch):
#   bash scripts/lumi/opd_single_gcd.sh
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"
TIME="${PGS_TIME:-00:30:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PGS_CONFIG:-configs/rocm_opd_smoke.yaml}"
STUDENT="${PGS_STUDENT:-Qwen/Qwen3-0.6B}"
TEACHER="${PGS_TEACHER:-Qwen/Qwen3-1.7B}"

export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"
BIND="-B ${SCRATCH}:${SCRATCH}"

echo "=== [1/2] pre-stage student + teacher (login node) ==="
for M in "$STUDENT" "$TEACHER"; do
  singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
done

echo "=== [2/2] distill on one GCD (offline) ==="
srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=7 --time="$TIME" \
  singularity exec $BIND \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    "$SIF" \
    bash -c "cd '${REPO}' && python -m palingenesis.cli distill --config '${CONFIG}'"
