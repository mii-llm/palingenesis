"""The quack (NVIDIA CuTeDSL) Muon backend must stay inactive on ROCm and CPU.

ROCm reports a (9, 0) device capability for gfx90a, so the sm-version check alone would
admit quack on AMD; _quack_ops() gates on accelerator.is_cuda() first. torch probes are
monkeypatched, so these run without a GPU.
"""

from palingenesis import accelerator, muon


def test_quack_inactive_on_rocm(monkeypatch):
    # Simulate ROCm: is_cuda() False even though capability looks like (9, 0).
    monkeypatch.setattr(accelerator, "is_cuda", lambda: False)
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert muon._quack_ops() is None


def test_quack_inactive_on_cpu(monkeypatch):
    monkeypatch.setattr(accelerator, "is_cuda", lambda: False)
    assert muon._quack_ops() is None


def test_quack_skipped_below_sm90(monkeypatch):
    # Real NVIDIA but pre-Hopper: the sm check rejects it (no quack import attempted).
    monkeypatch.setattr(accelerator, "is_cuda", lambda: True)
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert muon._quack_ops() is None
