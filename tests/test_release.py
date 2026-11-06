"""Library use outside a checkout: shipped example configs and `pgs init`, in-process entry points (a notebook),
and cleanup when training stops early."""

import importlib
import runpy
import sys
from types import SimpleNamespace

import pytest

from palingenesis.config import Config
from palingenesis.examples import EXAMPLES, example_text, write_example
from palingenesis.examples import main as init_main


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_shipped_examples_load_with_their_config_class(name, tmp_path):
    path = write_example(name, tmp_path / f"{name}.yaml")
    if name in ("sft", "dpo", "kd"):
        config = Config.from_yaml(str(path))
        assert config.dpo.enabled == (name == "dpo")
        assert config.distill.enabled == (name == "kd")
        config.validate()
    elif name == "rl":
        from palingenesis.rl.config import RLConfig

        assert RLConfig.from_yaml(path).model.policy
    else:
        from palingenesis.opd.config import OPDConfig

        assert OPDConfig.from_yaml(path).model.student


def test_init_writes_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert init_main(["sft"]) == 0
    assert (tmp_path / "sft.yaml").read_text() == example_text("sft")
    assert init_main(["sft"]) == 1  # exists
    assert init_main(["sft", "--force"]) == 0
    assert init_main(["--list"]) == 0 and "distill" in capsys.readouterr().out


def test_from_cli_with_an_empty_list_ignores_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher", "-f", "/kernel.json", "--train.learning_rate", "1"])
    config = Config.from_cli(args=[])
    assert config.train.learning_rate == Config().train.learning_rate


def test_python_m_palingenesis_does_not_train_on_import(monkeypatch):
    import palingenesis.train

    monkeypatch.setattr(palingenesis.train, "main", lambda: pytest.fail("training started on import"))
    sys.modules.pop("palingenesis.__main__", None)
    importlib.import_module("palingenesis.__main__")
    sys.modules.pop("palingenesis.__main__", None)
    monkeypatch.setattr(palingenesis.train, "main", lambda: sys.exit(0))
    with pytest.raises(SystemExit):  # run as a script it trains: here main() is stubbed to exit
        runpy.run_module("palingenesis", run_name="__main__")


def test_setup_distributed_without_cuda_leaves_the_error_to_train(monkeypatch):
    import torch

    from palingenesis import distributed

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("set_device without CUDA"))
    assert distributed.setup_distributed() == (0, 0, 1)


def test_train_without_cuda_raises_the_friendly_error(monkeypatch):
    import torch

    from palingenesis import train as train_mod

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No usable CUDA GPU"):
        train_mod.train(Config())


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_train_closes_the_tracker_and_the_process_group_when_interrupted(monkeypatch, error):
    from palingenesis import train as train_mod

    calls = []
    tracker = SimpleNamespace(finish=lambda exit_code=0: calls.append(("finish", exit_code)))

    def failing(config, run):
        run["tracker"] = tracker
        raise error

    monkeypatch.setattr(train_mod, "_train", failing)
    monkeypatch.setattr(train_mod, "cleanup_distributed", lambda: calls.append(("cleanup",)))
    with pytest.raises(type(error)):
        train_mod.train(Config())
    assert calls == [("finish", 1), ("cleanup",)]


def test_train_returns_the_result(monkeypatch):
    from palingenesis import train as train_mod

    result = train_mod.TrainResult(output_dir="out", global_step=3, metrics={"train/loss": 1.5})
    calls = []
    tracker = SimpleNamespace(finish=lambda exit_code=0: calls.append(exit_code))

    def ok(config, run):
        run["tracker"] = tracker
        return result

    monkeypatch.setattr(train_mod, "_train", ok)
    monkeypatch.setattr(train_mod, "cleanup_distributed", lambda: None)
    assert train_mod.train(Config()) is result and calls == [0]


def test_tracker_keeps_the_latest_scalars():
    from palingenesis.logging import Tracker

    tracker = Tracker(Config(), is_main=False)
    tracker.log({"train/loss": 2.0, "note": "x"}, step=1)
    tracker.log({"train/loss": 1.0, "eval/loss": 1.2}, step=2)
    assert tracker.latest["train/loss"] == 1.0 and tracker.latest["eval/loss"] == 1.2 and "note" not in tracker.latest
    tracker.finish()
    tracker.finish()  # idempotent
