"""model.flash_kernel (palingenesis.accelerator.activate_flash_kernel) and the flash_attention_3 / _4 attention
implementations: selection per GPU generation, fallbacks of "auto", errors of explicit choices, config validation.
CPU: the GPU, the kernel packages and torch's registry are simulated."""

import types

import pytest
import torch

from palingenesis import accelerator as A
from palingenesis.config import Config, ConfigError


@pytest.mark.parametrize(
    "choice, major, expected",
    [
        ("builtin", 9, []),
        ("auto", None, []),
        ("auto", 8, []),
        ("auto", 9, ["fa3", "fa4"]),
        ("auto", 10, ["fa4"]),
        ("auto", 12, []),
        ("fa3", 9, ["fa3"]),
        ("fa4", 10, ["fa4"]),
    ],
)
def test_candidates(choice, major, expected):
    assert A.flash_kernel_candidates(choice, major) == expected


def test_candidates_rejects_unknown():
    with pytest.raises(ValueError, match="flash_kernel"):
        A.flash_kernel_candidates("fa2", 9)


def test_builtin_and_other_attention_do_nothing():
    assert A.activate_flash_kernel("builtin", "sdpa", 8, 2, 64) == "builtin"
    assert A.activate_flash_kernel("fa3", "flash_attention_2", 8, 2, 64) == "builtin"


@pytest.fixture
def gpu(monkeypatch):
    """A simulated NVIDIA GPU of a chosen generation, kernel packages and torch's flash registry."""
    calls = []
    registry = types.SimpleNamespace(
        activate_flash_attention_impl=lambda impl: calls.append(("activate", impl)),
        restore_flash_attention_impl=lambda _raise_warn=True: calls.append(("restore",)),
    )
    state = {"major": 9, "installed": {"fa3", "fa4"}, "probe_error": None}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (state["major"], 0))
    monkeypatch.setattr(A, "is_rocm", lambda: False)
    monkeypatch.setattr(A, "_flash_module_available", lambda kernel: kernel in state["installed"])

    def probe(impl, heads, kv_heads, head_dim, device):
        registry.activate_flash_attention_impl(impl)
        if state["probe_error"] and impl in state["probe_error"]:
            raise RuntimeError(f"{impl} differs")

    monkeypatch.setattr(A, "_probe_flash_kernel", probe)
    monkeypatch.setattr(torch.nn, "attention", registry, raising=False)
    import sys

    monkeypatch.setitem(sys.modules, "torch.nn.attention", registry)
    return state, calls


def test_auto_picks_fa3_on_hopper_and_fa4_on_blackwell(gpu):
    state, calls = gpu
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "fa3"
    assert calls == [("activate", "FA3")]
    state["major"] = 10
    calls.clear()
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "fa4"
    state["major"] = 8
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "builtin"


def test_auto_falls_back_when_missing_or_failing(gpu):
    state, calls = gpu
    state["installed"] = {"fa4"}
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "fa4"  # Hopper without FA3: FA4
    state["probe_error"] = {"FA4"}
    calls.clear()
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "builtin"
    assert calls[-1] == ("restore",)  # the failed kernel is not left behind SDPA
    state["installed"] = set()
    assert A.activate_flash_kernel("auto", "sdpa", 16, 4, 128) == "builtin"


def test_explicit_choice_errors(gpu):
    state, _ = gpu
    state["installed"] = set()
    with pytest.raises(ConfigError, match="flash-attn-3"):
        A.activate_flash_kernel("fa3", "sdpa", 16, 4, 128)
    state["major"] = 8
    with pytest.raises(ConfigError, match="sm90"):
        A.activate_flash_kernel("fa3", "sdpa", 16, 4, 128)
    state["major"], state["installed"], state["probe_error"] = 10, {"fa4"}, {"FA4"}
    with pytest.raises(ConfigError, match="startup check"):
        A.activate_flash_kernel("fa4", "sdpa", 16, 4, 128)


def test_explicit_choice_needs_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert A.activate_flash_kernel("auto", "sdpa", 8, 8, 64) == "builtin"
    with pytest.raises(ConfigError, match="NVIDIA GPU"):
        A.activate_flash_kernel("fa4", "sdpa", 8, 8, 64)


def test_attention_heads():
    cfg = types.SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, head_dim=256, hidden_size=1024)
    assert A.attention_heads(cfg) == (16, 4, 256)
    cfg = types.SimpleNamespace(num_attention_heads=8, num_key_value_heads=None, head_dim=None, hidden_size=512)
    assert A.attention_heads(cfg) == (8, 8, 64)


def test_flash_attention_3_and_4_are_varlen_and_checked(monkeypatch):
    assert A.is_flash_varlen("flash_attention_3") and A.is_flash_varlen("flash_attention_4")
    from transformers.utils import import_utils

    monkeypatch.setattr(import_utils, "is_flash_attn_3_available", lambda: False)
    with pytest.raises(ConfigError, match="flash-attn-3"):
        A.check_attn_implementation("flash_attention_3")
    monkeypatch.setattr(import_utils, "is_flash_attn_4_available", lambda: True)
    A.check_attn_implementation("flash_attention_4")


def test_config_validation():
    cfg = Config()
    assert cfg.model.flash_kernel == "builtin" and cfg.validate() == []
    cfg.model.flash_kernel = "auto"
    assert cfg.validate() == []
    cfg.model.attn_implementation = "flash_attention_3"
    assert cfg.validate() == []  # auto leaves non-sdpa attention alone
    cfg.model.flash_kernel = "fa3"
    with pytest.raises(ConfigError, match="replaces the kernel behind sdpa"):
        cfg.validate()
    cfg = Config()
    cfg.model.flash_kernel = "fa2"
    with pytest.raises(ConfigError, match="flash_kernel"):
        cfg.validate()
    cfg = Config()
    cfg.model.attn_implementation = "flash_attention_5"
    with pytest.raises(ConfigError, match="attn_implementation"):
        cfg.validate()
