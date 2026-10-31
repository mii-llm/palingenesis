"""scripts/acceptance_smoke.py: its configs load and validate, and its log parsing reads the trainers' lines."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acceptance_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("acceptance_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_writes_valid_configs(smoke, tmp_path, capsys):
    assert smoke.main(["--out", str(tmp_path), "--dry-run", "--attn", "kernels-community/aiter-flash-attn@v2"]) == 0
    written = json.loads(capsys.readouterr().out)
    assert set(written) == set(smoke.STAGES)
    for stage in smoke.STAGES:
        rows = (tmp_path / "data" / f"{stage}.jsonl").read_text().splitlines()
        assert len(rows) >= 256
    from palingenesis.config import Config

    sft = Config.from_yaml(written["sft"])
    assert sft.model.attn_implementation == "kernels-community/aiter-flash-attn@v2"
    assert sft.data.dataset == str(tmp_path / "data" / "sft.jsonl") and not sft.data.streaming
    from palingenesis.opd.config import OPDConfig

    opd = OPDConfig.from_yaml(written["opd"])
    assert list(opd.teachers) == ["teacher"] and list(opd.sources) == ["chat"]  # sections replaced, not merged


def test_data_is_deterministic(smoke, tmp_path):
    a = smoke.write_data(tmp_path / "a")
    b = smoke.write_data(tmp_path / "b")
    assert all(a[s].read_bytes() == b[s].read_bytes() for s in smoke.STAGES)


def test_parse_logs(smoke):
    sft = "\n".join(
        f"2026-10-31 INFO step={i} loss={3 - i / 10:.4f} lr=1.00e-05 tok/s={1000 * i} grad_norm=1.000 dt=0.50s"
        for i in range(1, 5)
    )
    assert smoke.parse_log("sft", sft) == {
        "steps_logged": 4,
        "loss_first": 2.9,
        "loss_last": 2.6,
        "tok_s_median": 3500.0,
    }
    rl = "\n".join(
        f"INFO palingenesis.rl.trainer: step {i} | loss={0.1 * i:.4g} reward={0.2 * i:.4g} time/step={10 + i}"
        for i in range(1, 5)
    )
    parsed = smoke.parse_log("rl", rl)
    assert parsed["steps_logged"] == 4 and parsed["s_per_step_median"] == 13.5
    assert parsed["reward_median"] == pytest.approx(0.7)
