"""RL and on-policy distillation logging: their step and eval lines are read by `pgs monitor` / `pgs loss`,
and their metrics reach every backend of the shared tracker (wandb, trackio, logging.trackers) under the
names they always had."""

import io
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from palingenesis.agent_tooling._logparse import parse_evals, parse_steps  # noqa: E402
from palingenesis.agent_tooling.check_loss import parse_losses_from_text  # noqa: E402
from palingenesis.agent_tooling.monitor_run import analyze_run, parse_training_log  # noqa: E402
from palingenesis.logging import Tracker  # noqa: E402

RL_STEP = {
    "loss": 0.0123,
    "reward": 0.375,
    "rewards/correct": 0.4,
    "lr": 1e-6,
    "grad_norm": 0.41,
    "time/step": 31.2,
    "rollout_tok_s": 5120.0,
    "env/metric/verifier_failed": 0.0,
}
OPD_STEP = {"loss": 0.52, "k1": 0.31, "k1/qwen3-8b.v2": 0.31, "lr": 2e-6, "grad_norm": 1.5, "time/step": 12.0}


class _Wandb:
    """wandb's module surface the tracker uses: init() -> run with define_metric / log / finish."""

    def __init__(self):
        self.inits, self.rows, self.finished = [], [], []

    def init(self, **kwargs):
        self.inits.append(kwargs)
        outer = self
        return SimpleNamespace(
            id=kwargs["id"],
            define_metric=lambda *a, **k: None,
            log=lambda row: outer.rows.append(row),
            finish=lambda exit_code=0: outer.finished.append(exit_code),
        )


def _fake_backends(monkeypatch):
    wandb = _Wandb()
    trackio = types.ModuleType("trackio")
    trackio.rows, trackio.inits = [], []
    trackio.init = lambda **kwargs: trackio.inits.append(kwargs) or object()
    trackio.log = lambda row, step=None: trackio.rows.append((row, step))
    trackio.finish = lambda: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    return wandb, trackio


BACKEND = """
LOGGED, FINISHED, BUILT = [], [], []


class Recorder:
    def __init__(self, *, project, name, config, run_id, output_dir, resuming):
        BUILT.append(dict(project=project, name=name, run_id=run_id, resuming=resuming))

    def log(self, metrics, step):
        LOGGED.append((dict(metrics), step))

    def finish(self, exit_code):
        FINISHED.append(exit_code)


class Broken:
    def __init__(self, **kwargs):
        pass

    def log(self, metrics, step):
        raise RuntimeError("backend down")

    def finish(self, exit_code):
        raise RuntimeError("backend down")
"""


def _logged_lines(trainer_cls, module_name: str, tracker: Tracker) -> str:
    """The lines a trainer's _log writes, formatted as setup_logging formats them."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("[rank 0] %(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    module_logger = logging.getLogger(module_name)
    module_logger.addHandler(handler)
    level = module_logger.level
    module_logger.setLevel(logging.INFO)
    fake = SimpleNamespace(parallel=SimpleNamespace(main=True), tracker=tracker)
    step = RL_STEP if "rl" in module_name else OPD_STEP
    try:
        trainer_cls._log(fake, "eval", {"reward": 0.25, "dev_kl": 0.4}, 0)
        for i in range(1, 4):
            trainer_cls._log(fake, "step", {**step, "loss": step["loss"] * (4 - i)}, i)
        trainer_cls._log(fake, "eval", {"reward": 0.5, "dev_kl": 0.3}, 3)
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(level)
    return stream.getvalue()


def _check_trainer(monkeypatch, tmp_path, trainer_cls, module_name, metrics):
    wandb, trackio = _fake_backends(monkeypatch)
    (tmp_path / "backend.py").write_text(BACKEND)
    log_config = SimpleNamespace(
        use_wandb=True,
        use_trackio=True,
        trackers=[f"{tmp_path / 'backend.py'}:Recorder", f"{tmp_path / 'backend.py'}:Broken"],
        project="proj",
        run_name="",
    )
    tracker = Tracker.for_run(log_config, {"a": {"b": 1}}, str(tmp_path), None)
    text = _logged_lines(trainer_cls, module_name, tracker)

    # the log: `pgs monitor` / `pgs loss` read the step lines, the eval lines are evaluations
    steps = parse_steps(text)
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert abs(steps[0]["loss"] - 3 * metrics["loss"]) < 1e-3 * metrics["loss"]
    assert set(metrics) <= set(steps[0])  # every metric, names with digits, dots and dashes included
    assert [e["step"] for e in parse_evals(text)] == [0, 3] and parse_evals(text)[1]["reward"] == 0.5
    assert [step for step, _ in parse_losses_from_text(text)] == [1, 2, 3]
    run = analyze_run(parse_training_log(text))
    assert run["status"] == "HEALTHY" and run["current_step"] == 3 and run["step_time_s"] == metrics["time/step"]

    # every backend got the rows under the names wandb always had: eval/ prefix, the step as a metric
    (init,) = wandb.inits
    assert init["project"] == "proj" and init["name"] is None and init["id"] == tracker.run_id
    assert init["resume"] == "allow" and init["dir"] == str(tmp_path) and init["config"] == {"a": {"b": 1}}
    assert wandb.rows[0] == {"eval/reward": 0.25, "eval/dev_kl": 0.4, "train/global_step": 0}
    assert wandb.rows[1] == {**metrics, "loss": 3 * metrics["loss"], "train/global_step": 1}
    assert trackio.inits[0]["name"] == f"run-{tracker.run_id}" and trackio.rows[1][1] == 1
    backend = sys.modules["pgs_plugin_backend"]
    assert backend.BUILT == [{"project": "proj", "name": None, "run_id": tracker.run_id, "resuming": False}]
    assert backend.LOGGED[0] == ({"eval/reward": 0.25, "eval/dev_kl": 0.4}, 0) and len(backend.LOGGED) == 5
    tracker.finish(1)
    tracker.finish(0)  # idempotent
    assert wandb.finished == [1] and backend.FINISHED == [1]

    # a resumed run continues the same run id
    resumed = Tracker.for_run(log_config, {}, str(tmp_path), tracker.run_id)
    assert resumed.run_id == tracker.run_id and wandb.inits[1]["id"] == tracker.run_id
    assert backend.BUILT[1]["resuming"] is True
    # rank > 0: nothing is opened
    assert Tracker.for_run(log_config, {}, str(tmp_path), None, is_main=False).run_id is None
    assert len(wandb.inits) == 2


def test_rl_log_lines_and_tracker_rows(monkeypatch, tmp_path):
    from palingenesis.rl.trainer import RLTrainer

    _check_trainer(monkeypatch, tmp_path, RLTrainer, "palingenesis.rl.trainer", RL_STEP)


def test_opd_log_lines_and_tracker_rows(monkeypatch, tmp_path):
    from palingenesis.opd.trainer import OPDTrainer

    _check_trainer(monkeypatch, tmp_path, OPDTrainer, "palingenesis.opd.trainer", OPD_STEP)


def test_parser_keeps_sft_lines_and_reads_mixed_logs():
    text = "\n".join(
        [
            "[rank 0] 10:00:00 | INFO | step=0 eval=0.9000 (before training)",
            "[rank 0] 10:00:01 | INFO | step=1 loss=1.2000 lr=1.00e-05 tok/s=900 grad_norm=1.000 dt=2.00s",
            "[rank 0] 10:00:02 | INFO | Best model updated: step=1, eval_loss=0.8 -> /ckpt/best",
            "[rank 0] 10:00:03 | INFO | step=2 loss=1.1000 lr=1.00e-05 tok/s=900 grad_norm=1.000 dt=2.00s eval=0.8500",
            "[rank 0] 10:00:04 | WARNING | stop requested: saving a resumable checkpoint at step 3 and exiting",
            "[rank 0] 10:00:05 | INFO | step 3 | loss=nan reward=0",  # a resumed log of another trainer
        ]
    )
    steps = parse_steps(text)
    assert [s["step"] for s in steps] == [1, 2, 3] and steps[0]["tok/s"] == 900 and steps[1]["eval"] == 0.85
    assert [(e["step"], e["loss"]) for e in parse_evals(text)] == [(0, 0.9), (2, 0.85)]
    assert analyze_run(parse_training_log(text))["status"] == "CRASHED"  # the NaN loss is seen
