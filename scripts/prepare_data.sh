#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Data Preparation — Score, filter, and select optimal training subset
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   ./scripts/prepare_data.sh Qwen/Qwen3.5-4B data/traces.jsonl
#   ./scripts/prepare_data.sh google/gemma-4-E4B-it data/traces.jsonl 3000 optimal
#
# Arguments:
#   $1 — Model name/path (used for scoring)
#   $2 — Data path (JSONL file or HF dataset)
#   $3 — Budget (number of samples to select, default: all)
#   $4 — Strategy (optimal, balanced, flow, curriculum, default: optimal)
set -euo pipefail

MODEL="${1:?Usage: $0 <model> <data> [budget] [strategy]}"
DATA="${2:?Usage: $0 <model> <data> [budget] [strategy]}"
BUDGET="${3:-}"
STRATEGY="${4:-optimal}"
OUTPUT="${OUTPUT:-./prepared}"

echo "╔══════════════════════════════════════════╗"
echo "║  palingenesis — data preparation          ║"
echo "╚══════════════════════════════════════════╝"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Budget: ${BUDGET:-all}"
echo "Strategy: $STRATEGY"
echo "Output: $OUTPUT"
echo ""

CMD="palingenesis prepare --model $MODEL --data $DATA --output $OUTPUT --strategy $STRATEGY"

if [ -n "$BUDGET" ]; then
    CMD="$CMD --budget $BUDGET"
fi

# Add HES scoring for reasoning data (recommended)
if [ "${HES:-1}" = "1" ]; then
    CMD="$CMD --hes"
    echo "HES scoring: enabled (set HES=0 to disable)"
fi

echo "Running: $CMD"
echo ""
$CMD
