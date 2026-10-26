#!/usr/bin/env bash
# Validate the AMD/ROCm port on ONE LUMI GCD (single node, single GPU), on the fast
# dev-g partition. Track B: this is the LUMI *test rig*, not part of core palingenesis —
# all the LUMI/Slurm/container specifics are allowed to live here and nowhere else.
#
# Usage (from a LUMI login node, with the repo synced under /scratch):
#   bash scripts/lumi/test_single_gcd.sh
#
# Everything runs under /scratch only (LUMI policy): the repo, the bind mount, outputs.
# Override any of these via the environment:
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"          # dev-g = fast, 3 h cap; good for quick checks
TIME="${PGS_TIME:-00:15:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
# Repo root = two levels up from this script (expected to sit under $SCRATCH).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "project=$PROJECT account=$ACCOUNT partition=$PARTITION"
echo "sif=$SIF"
echo "repo=$REPO"

srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --time="$TIME" \
  singularity exec -B "${SCRATCH}:${SCRATCH}" "$SIF" \
    env PGS_REPO="$REPO" bash "$REPO/scripts/lumi/_in_container_tests.sh"
