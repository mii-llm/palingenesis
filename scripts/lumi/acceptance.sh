#!/usr/bin/env bash
# One-command ROCm acceptance smoke on one LUMI GCD — Track B.
# The AMD/ROCm port has no GPU in GitHub CI (the monkeypatched unit tests run there via
# .github/workflows/container.yaml); this is the on-hardware counterpart: it runs the
# ROCm-relevant unit tests + an accelerator probe, then a quick SFT smoke, on one GCD.
# Use it to re-confirm the port after a change. The heavier modes have their own runners
# (sft_qwen3_4b.sh, opd_single_gcd.sh, rl_single_gcd.sh, train_multi_gcd.sh, ...).
#
#   bash scripts/lumi/acceptance.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######## 1/2 unit tests + accelerator probe (one GCD) ########"
bash "$HERE/test_single_gcd.sh"

echo "######## 2/2 SFT smoke (bf16/SDPA/AdamW, one GCD) ########"
bash "$HERE/sft_single_gcd.sh"

echo "######## ROCm acceptance smoke complete ########"
