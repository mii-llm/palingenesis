"""pgs fit: which memory settings are measured, and which one is recommended (the measurement itself is the
real trainer on a GPU; here it is replaced by fixed numbers)."""

import json
import types

import pytest

from palingenesis.agent_tooling import fit_memory
from palingenesis.agent_tooling.fit_memory import Candidate, apply_overrides, candidates, recommend
from palingenesis.config import Config


def _config(seq=16384, **data):
    c = Config()
    c.data.max_seq_length = seq
    c.model.compile = False
    for k, v in data.items():
        setattr(c.data, k, v)
    return c


def test_candidates_go_from_fastest_to_most_frugal():
    names = [c.name for c in candidates(_config())]
    assert names == [
        "as configured",  # selective checkpointing, the default
        "none checkpointing",
        "full checkpointing",
        "SeCO, 8192-token chunks",
        "SeCO, 4096-token chunks",
        "SeCO, 2048-token chunks",
        "SeCO, 1024-token chunks",
        "SeCO, 1024-token chunks, K/V offloaded",
    ]
    assert all(not c.skipped for c in candidates(_config()))


def test_seco_chunks_are_shorter_than_the_sequence():
    names = [c.name for c in candidates(_config(seq=4096))]
    assert "SeCO, 4096-token chunks" not in names and "SeCO, 2048-token chunks" in names
    assert not any("SeCO" in c.name for c in candidates(_config(seq=1024)))


def test_settings_the_config_refuses_are_skipped_with_the_reason():
    by_name = {c.name: c for c in candidates(_config(turn_scaling="progressive"))}
    assert "turn_scaling" in by_name["SeCO, 4096-token chunks"].skipped
    assert not by_name["full checkpointing"].skipped


def test_a_seco_config_also_tries_the_checkpointing_modes_without_it():
    c = _config()
    c.memory.seco = True
    by_name = {x.name: x for x in candidates(c)}
    assert by_name["none checkpointing"].overrides == {"train.gradient_checkpointing": "none", "memory.seco": False}
    assert "SeCO, 4096-token chunks" not in by_name  # the config as it is


def test_overrides_reach_the_config_and_the_command_line():
    c = Candidate("x", {"memory.seco": True, "memory.seco_chunk_size": 2048})
    assert c.flags() == ["--memory.seco", "true", "--memory.seco_chunk_size", "2048"]
    base = _config()
    changed = apply_overrides(base, c.overrides)
    assert changed.memory.seco and changed.memory.seco_chunk_size == 2048 and not base.memory.seco


def test_the_fastest_setting_within_the_limit_is_recommended():
    fast = Candidate("fast", result={"measured_peak_gb": 70.0, "tokens_per_sec": 9000.0})
    mid = Candidate("mid", result={"measured_peak_gb": 35.0, "tokens_per_sec": 6000.0})
    slow = Candidate("slow", result={"measured_peak_gb": 20.0, "tokens_per_sec": 3000.0})
    oom = Candidate("oom", result={"oom": True})
    cands = [fast, mid, slow, oom]
    assert recommend(cands, 73.6) is fast
    assert recommend(cands, 36.8) is mid
    assert recommend(cands, 10.0) is None
    text = fit_memory.table(cands, 36.8, mid)
    assert "over the budget" in text and "fastest that fits" in text and "out of memory" in text


@pytest.mark.parametrize(
    "budget, expected, code", [(80, "none checkpointing", 0), (24, "full checkpointing", 0), (4, None, 1)]
)
def test_report_recommends_within_the_budget(tmp_path, monkeypatch, capsys, budget, expected, code):
    import torch

    peaks = {"none": (60.0, 9000.0), "selective": (40.0, 7000.0), "full": (20.0, 5000.0)}

    def fake(config_path, extra, candidate, steps):
        mode = candidate.overrides.get("train.gradient_checkpointing", "selective")
        if candidate.overrides.get("memory.seco"):
            return {"measured_peak_gb": 8.0, "tokens_per_sec": 1000.0}
        return dict(zip(("measured_peak_gb", "tokens_per_sec"), peaks[mode]))

    monkeypatch.setattr(fit_memory, "measure_in_subprocess", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=80 * 2**30))
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  max_seq_length: 16384\nmodel:\n  compile: false\n")
    assert fit_memory.main(["--config", str(path), "--gpu", str(budget), "--json"]) == code
    report = json.loads(capsys.readouterr().out)
    assert report["recommended"] == expected
    assert report["limit_gb"] == pytest.approx(budget * 0.92)


@pytest.mark.parametrize("integrated, budget", [(1, 50), (0, 120)])
def test_unified_memory_default_budget_is_the_available_host_memory(tmp_path, monkeypatch, capsys, integrated, budget):
    """A GPU that allocates from the host's RAM (DGX Spark) reports all of it as its own: the
    default budget is the host memory available when pgs fit starts, not the device total."""
    import torch

    from palingenesis import accelerator

    measured = {"measured_peak_gb": 60.0, "tokens_per_sec": 1000.0}
    monkeypatch.setattr(fit_memory, "measure_in_subprocess", lambda *a: dict(measured))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = types.SimpleNamespace(total_memory=120 * 2**30, is_integrated=integrated)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(accelerator, "host_available_bytes", lambda: 50 * 2**30)
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  max_seq_length: 16384\nmodel:\n  compile: false\n")
    code = fit_memory.main(["--config", str(path), "--json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["budget_gb"] == pytest.approx(budget)
    assert (code, report["recommended"] is None) == ((1, True) if integrated else (0, False))
    assert ("Unified memory" in captured.err) is bool(integrated)
    # an explicit budget is taken as given on both
    assert fit_memory.main(["--config", str(path), "--gpu", "100", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["budget_gb"] == 100
