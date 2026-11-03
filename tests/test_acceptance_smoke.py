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
    from palingenesis.rl.config import RLConfig

    # the student solves this arithmetic: retiring solved prompts would empty the data before --steps
    assert RLConfig.from_yaml(written["rl"]).data.retire_above == 0.0


def test_gpus_keeps_the_work_per_gpu(smoke, tmp_path, capsys):
    assert smoke.main(["--out", str(tmp_path), "--dry-run", "--gpus", "2", "--rollout", "vllm"]) == 0
    written = json.loads(capsys.readouterr().out)
    rows = {s: len((tmp_path / "data" / f"{s}.jsonl").read_text().splitlines()) for s in smoke.STAGES}
    assert rows == {"sft": 1024, "dpo": 512, "opd": 256, "rl": 512}  # SFT/DPO: data for 30 steps on 2 ranks
    from palingenesis.config import Config
    from palingenesis.opd.config import OPDConfig
    from palingenesis.rl.config import RLConfig

    assert Config.from_yaml(written["sft"]).train.per_device_batch_size == 8
    assert RLConfig.from_yaml(written["rl"]).rollout.batch_prompts == 16  # global, 8 per rank
    assert OPDConfig.from_yaml(written["opd"]).teachers["teacher"].device == "cuda:1"
    sft = smoke.command("sft", Path("c.yaml"), 2)
    assert "--nproc_per_node=2" in sft and sft[-3:] == ["palingenesis.train", "--config", "c.yaml"]
    assert smoke.command("rl", Path("c.yaml"), 2)[-3:] == ["palingenesis.rl.trainer", "--config", "c.yaml"]
    assert "distill" in smoke.command("opd", Path("c.yaml"), 2)  # one process
    assert "rl" in smoke.command("rl", Path("c.yaml"), 1)  # one GPU: the CLI, as before


def test_gpus_refuses_hf_rl(smoke, tmp_path):
    with pytest.raises(SystemExit):  # hf generates with the sharded model: refused by the RL trainer too
        smoke.main(["--out", str(tmp_path), "--dry-run", "--gpus", "2"])
    assert smoke.main(["--out", str(tmp_path), "--dry-run", "--gpus", "2", "--stages", "sft,dpo,opd"]) == 0


def test_one_gpu_data_is_unchanged(smoke, tmp_path):
    one = smoke.write_data(tmp_path / "one")
    two = smoke.write_data(tmp_path / "two", gpus=2)
    for stage in ("sft", "dpo"):  # the first rows are the same: one-GPU runs stay comparable with earlier ones
        a, b = one[stage].read_text().splitlines(), two[stage].read_text().splitlines()
        assert len(b) == 2 * len(a) and b[: len(a)] == a


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
    # a step whose groups all got one reward trains nothing and logs no loss, but it ran
    parsed = smoke.parse_log("rl", rl + "\nINFO palingenesis.rl.trainer: step 5 | grad_norm=0 time/step=1")
    assert parsed["steps_logged"] == 5 and parsed["loss_last"] == pytest.approx(0.4)
