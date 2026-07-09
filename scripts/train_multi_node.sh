#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# palingenesis — Multi-Node Training (bare-metal / cloud VMs)
# ═══════════════════════════════════════════════════════════════════════════════
#
# For clusters WITHOUT SLURM (e.g., cloud VMs, bare-metal servers).
# Run this script on EACH node. All nodes must be able to reach MASTER_ADDR.
#
# Usage (run on each node):
#   # Node 0 (master):
#   MASTER_ADDR=10.0.0.1 NODE_RANK=0 NNODES=2 ./scripts/train_multi_node.sh config.yaml
#
#   # Node 1:
#   MASTER_ADDR=10.0.0.1 NODE_RANK=1 NNODES=2 ./scripts/train_multi_node.sh config.yaml
#
# Required environment variables:
#   MASTER_ADDR   — IP/hostname of node 0 (reachable from all nodes)
#   NODE_RANK     — This node's rank (0 = master, 1, 2, ...)
#   NNODES        — Total number of nodes
#
# Optional:
#   MASTER_PORT   — Port for rendezvous (default: 29500)
#   NGPUS         — GPUs per node (default: 8)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Required ──────────────────────────────────────────────────────────────────
: "${MASTER_ADDR:?Set MASTER_ADDR to the IP of node 0}"
: "${NODE_RANK:?Set NODE_RANK (0 for master, 1+ for workers)}"
: "${NNODES:?Set NNODES to total number of nodes}"

# ── Optional ──────────────────────────────────────────────────────────────────
MASTER_PORT="${MASTER_PORT:-29500}"
NGPUS="${NGPUS:-8}"
CONFIG="${1:-configs/qwen35_4b/a100_80gb_multigpu.yaml}"

TOTAL_GPUS=$((NNODES * NGPUS))

echo "┌──────────────────────────────────────────────────┐"
echo "│  palingenesis — multi-node training              │"
echo "├──────────────────────────────────────────────────┤"
echo "│  Config:    $CONFIG"
echo "│  This node: rank $NODE_RANK / $NNODES nodes"
echo "│  Master:    $MASTER_ADDR:$MASTER_PORT"
echo "│  GPUs:      $NGPUS per node × $NNODES = $TOTAL_GPUS total"
echo "└──────────────────────────────────────────────────┘"

# ── NCCL / CUDA ──────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ── Launch ────────────────────────────────────────────────────────────────────
# torchrun on this node spawns NGPUS workers.
# c10d rendezvous connects all nodes via MASTER_ADDR.

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NGPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --redirects=3 \
    --tee=3 \
    -m palingenesis.train --config "$CONFIG"
