#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Pre-Training Diagnostic — verify config before spending GPU hours
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   ./scripts/diagnose.sh configs/qwen35_4b.yaml
#   ./scripts/diagnose.sh projects/qwen35_4b_agentic/config.yaml
#
# Checks:
#   - Memory estimate (will it fit?)
#   - Masking validation (are labels correct?)
#   - Config sanity (any obvious issues?)
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "╔══════════════════════════════════════════╗"
echo "║  palingenesis — pre-training diagnostic   ║"
echo "╚══════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo ""

echo "━━━ Step 1: Memory Profile ━━━"
palingenesis profile --config "$CONFIG" --gpu_memory_gb "${GPU_GB:-80}"
echo ""

echo "━━━ Step 2: Masking Validation ━━━"
palingenesis validate --config "$CONFIG" --num_samples 50
echo ""

echo "━━━ Step 3: Full Diagnostic ━━━"
palingenesis diagnose --config "$CONFIG" --mode pre --json
echo ""

echo "━━━ Done ━━━"
echo "If all checks pass, you're ready to train:"
echo "  ./scripts/train_single_gpu.sh $CONFIG"
