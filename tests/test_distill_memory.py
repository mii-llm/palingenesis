"""The memory estimate counts off-policy distillation: online teachers' weights (exact, sharded by
placement) and the per-micro-batch teacher targets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.agent_tooling.profile_memory import estimate_memory  # noqa: E402


def test_estimate_includes_teachers(tmp_path, monkeypatch):
    tok = H.tokenizer()
    student = H.save_model(tmp_path / "student", tok)
    teacher = H.save_model(tmp_path / "teacher", tok, hidden=64, seed=1)
    c = H.cpu_config(tmp_path, tmp_path / "student", "unused.jsonl")
    base = estimate_memory(c)
    assert base["teacher_memory_gb"] == 0.0
    c.distill.enabled = True
    c.distill.teachers = {
        "t": {"model": str(tmp_path / "teacher"), "dtype": "float32"},
        "c": {"cache": "x", "top_k": 8},
    }
    est = estimate_memory(c)
    n = sum(p.numel() for p in teacher.parameters())
    assert abs(est["teacher_memory_gb"] - n * 4 / 2**30) < 1e-9
    assert est["teacher_buffers_gb"] > 0 and est["total_estimated_gb"] > base["total_estimated_gb"]
    monkeypatch.setenv("WORLD_SIZE", "4")
    c.distill.teachers["t"]["placement"] = "shard"
    assert abs(estimate_memory(c)["teacher_memory_gb"] - n * 4 / 4 / 2**30) < 1e-9
    assert student is not None
