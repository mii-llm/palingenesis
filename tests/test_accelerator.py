"""Accelerator detection, exercised without a real GPU by faking torch's backend probes.

The key case is ROCm: torch's ROCm build masquerades as CUDA, so is_available() is True
and get_device_capability() returns gfx-derived numbers (gfx90a -> (9, 0)) that would make
a raw ``>= (8, 9)`` FP8 gate wrongly pass. These tests pin that behaviour down.
"""

import types

import pytest

from palingenesis import accelerator


def _fake_torch(monkeypatch, *, hip, available, capability=(0, 0), arch_name=None, name="FakeGPU", count=1):
    """Patch the torch attributes accelerator.py reads to simulate a backend."""
    import torch

    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    props = types.SimpleNamespace(name=name, major=capability[0], minor=capability[1], total_memory=64 * 2**30)
    if arch_name is not None:
        props.gcnArchName = arch_name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    accelerator.describe.cache_clear()


def test_cuda_hopper(monkeypatch):
    _fake_torch(monkeypatch, hip=None, available=True, capability=(9, 0), name="NVIDIA H100")
    assert accelerator.is_cuda() and not accelerator.is_rocm()
    assert accelerator.backend() == "cuda"
    assert accelerator.supports_fp8() is True
    assert accelerator.gpu_arch() == "sm_90"


def test_cuda_ampere_no_fp8(monkeypatch):
    _fake_torch(monkeypatch, hip=None, available=True, capability=(8, 0), name="NVIDIA A100")
    assert accelerator.is_cuda()
    assert accelerator.supports_fp8() is False  # SM80 < SM89


def test_rocm_gfx90a_reports_no_fp8(monkeypatch):
    # The critical false-positive: ROCm reports capability (9, 0) >= (8, 9) but has no FP8.
    _fake_torch(
        monkeypatch,
        hip="6.2.0",
        available=True,
        capability=(9, 0),
        arch_name="gfx90a:sramecc+:xnack-",
        name="AMD Instinct MI250X",
    )
    assert accelerator.is_rocm() and not accelerator.is_cuda()
    assert accelerator.backend() == "rocm"
    assert accelerator.supports_fp8() is False  # must NOT pass despite (9, 0)
    assert accelerator.gpu_arch() == "gfx90a"  # feature flags stripped


def test_cpu(monkeypatch):
    _fake_torch(monkeypatch, hip=None, available=False)
    assert accelerator.backend() == "cpu"
    assert accelerator.is_cuda() is False
    assert accelerator.supports_fp8() is False
    assert accelerator.device_capability() is None
    assert accelerator.gpu_arch() is None


def test_describe_never_raises(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    def boom(*a, **k):
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    accelerator.describe.cache_clear()
    assert accelerator.describe().startswith("accelerator: detection failed")


def test_float8_refuses_on_rocm(monkeypatch):
    """apply_float8_training must bail cleanly when supports_fp8() is False (e.g. ROCm)."""
    torchao = pytest.importorskip("torchao")  # noqa: F841 — gate present so the import path is taken
    from palingenesis import float8

    monkeypatch.setattr(accelerator, "supports_fp8", lambda: False)
    monkeypatch.setattr(accelerator, "backend", lambda: "rocm")
    monkeypatch.setattr(accelerator, "gpu_arch", lambda: "gfx90a")
    assert float8.apply_float8_training(object()) is False
