#!/usr/bin/env bash
# Real-model SFT quality run on one LUMI GCD: Qwen3-4B on a public chat slice — Track B.
# Proves a target-scale model trains well on AMD (not just the toy smoke). LUMI compute
# nodes are offline, so the login node pre-stages (a) the model and (b) a data slice as a
# local parquet under scratch; training then runs offline and reads that file.
#
# Usage (from a LUMI login node, repo synced under /scratch):
#   bash scripts/lumi/sft_qwen3_4b.sh
set -euo pipefail

PROJECT="${PGS_PROJECT:-465002563}"
ACCOUNT="${PGS_ACCOUNT:-project_${PROJECT}}"
PARTITION="${PGS_PARTITION:-dev-g}"
TIME="${PGS_TIME:-01:00:00}"
SIF="${PGS_SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
SCRATCH="/scratch/project_${PROJECT}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${PGS_MODEL:-Qwen/Qwen3-4B}"
NROWS="${PGS_NROWS:-6000}"
DATA="${SCRATCH}/data/ultrachat_slice.parquet"

export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME" "${SCRATCH}/data"
BIND="-B ${SCRATCH}:${SCRATCH}"

echo "=== [1/3] pre-stage model $MODEL (login node) ==="
singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

echo "=== [2/3] pre-stage ${NROWS}-row data slice -> $DATA (login node) ==="
singularity exec $BIND --env HF_HOME="$HF_HOME" "$SIF" python - "$DATA" "$NROWS" <<'PY'
import os, sys, itertools, pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
out, n = sys.argv[1], int(sys.argv[2])
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
rows = [{"messages": r["messages"]} for r in itertools.islice(ds, n)]
pq.write_table(pa.Table.from_pylist(rows), out)
print("wrote", len(rows), "rows ->", out, flush=True)
# datasets streaming + pyarrow can abort during interpreter finalization (GIL release
# on a background thread). The file is written; skip finalizers so the step exits clean.
os._exit(0)
PY

echo "=== [3/3] train Qwen3-4B on one GCD (offline) ==="
srun --account="$ACCOUNT" --partition="$PARTITION" \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=7 --time="$TIME" \
  singularity exec $BIND \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="${REPO}/src" \
    "$SIF" \
    bash -c "cd '${REPO}' && python -m palingenesis train --config configs/rocm_qwen3_4b.yaml \
       --data.dataset '${DATA}' --data.streaming false \
       --train.output_dir '${SCRATCH}/outputs/rocm_qwen3_4b'"
