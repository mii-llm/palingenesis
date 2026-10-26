#!/usr/bin/env bash
# Runs INSIDE the LUMI ROCm container (see test_single_gcd.sh). Track B (LUMI test rig):
# LUMI-specific, never imported by core palingenesis. Runs the ROCm-relevant unit tests
# (which monkeypatch torch's probes, so they need no GPU) plus a probe on the real device.
set -u
REPO="${PGS_REPO:-/scratch/project_465002563/palingenesis_amd}"
cd "$REPO"
export PYTHONPATH=src

echo "=== pytest (ROCm-relevant unit tests) ==="
python -m pytest tests/test_accelerator.py tests/test_muon_rocm.py -q
rc=$?

echo "=== accelerator probe on the real device ==="
python -c "from palingenesis import accelerator as a; print(a.describe()); print('is_rocm', a.is_rocm(), 'is_cuda', a.is_cuda(), 'arch', a.gpu_arch(), 'supports_fp8', a.supports_fp8(), 'capability', a.device_capability())"

exit $rc
