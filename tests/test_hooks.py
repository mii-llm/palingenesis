"""Trainer plugins (palingenesis.hooks): loading from config entries, the dispatch order and loss
scaling, hooks a path cannot run, and every hook firing in a real (CPU) training run."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.hooks import (  # noqa: E402
    LossInputs,
    PluginContext,
    PluginError,
    PluginManager,
    TrainerPlugin,
    build_plugin,
    load_plugins,
)


class Counter(TrainerPlugin):
    def __init__(self, scale: float = 1.0):
        self.scale = scale


def test_load_from_module_file_and_mapping(tmp_path):
    p = build_plugin("palingenesis.qat:TernaryQAT")
    assert p.plugin_name == "ternary_qat"
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(
        "from palingenesis.hooks import TrainerPlugin\n"
        "class Mine(TrainerPlugin):\n"
        "    def __init__(self, lam=0.0):\n"
        "        self.lam = lam\n"
    )
    mine = build_plugin({"class": f"{plugin_file}:Mine", "args": {"lam": 0.25}})
    assert type(mine).__name__ == "Mine" and mine.lam == 0.25
    with pytest.raises(PluginError, match="entry point"):
        build_plugin("not-installed-anywhere")
    with pytest.raises(PluginError, match="expected"):
        build_plugin({"klass": "x:y"})
    with pytest.raises(PluginError, match="has no"):
        build_plugin("palingenesis.qat:Nope")
    assert len(load_plugins(["palingenesis.qat:TernaryQAT"], [Counter()])) == 2


def test_duplicate_names_are_refused():
    with pytest.raises(PluginError, match="share the name"):
        PluginManager([Counter(), Counter()], PluginContext(config=None))


def test_loss_terms_are_summed_and_logged():
    class A(TrainerPlugin):
        name = "a"

        def loss(self, inputs, ctx):
            return {"x": torch.tensor(2.0), "y": torch.tensor(1.0)}

    class B(TrainerPlugin):
        name = "b"

        def loss(self, inputs, ctx):
            return torch.tensor(0.5)

    class C(TrainerPlugin):
        name = "c"

    mgr = PluginManager([A(), B(), C()], PluginContext(config=None))
    assert [PluginManager._name(p) for p in mgr.implementing("loss")] == ["a", "b"]
    inputs = LossInputs(None, {}, None, torch.zeros(1, 1), torch.zeros(0, dtype=torch.long), 1.0, 1)
    total, logged = mgr.loss(inputs)
    assert float(total) == 3.5 and logged == {"plugins/a/x": 2.0, "plugins/a/y": 1.0, "plugins/b/loss": 0.5}


def test_loss_hooks_on_paths_that_cannot_take_them():
    from palingenesis.config import Config
    from palingenesis.train import _check_plugin_paths

    class L(TrainerPlugin):
        def loss(self, inputs, ctx):
            return None

    c = Config()
    c.memory.seco = True
    with pytest.raises(PluginError, match="memory.seco"):
        _check_plugin_paths(c, PluginManager([L()], PluginContext(config=c)))
    c = Config()
    c.memory.gradient_release = True

    class S(TrainerPlugin):
        def on_after_optimizer_step(self, model, optimizer, ctx):
            pass

    with pytest.raises(PluginError, match="gradient_release"):
        _check_plugin_paths(c, PluginManager([S()], PluginContext(config=c)))


def test_every_hook_runs_in_training(tmp_path):
    """An SFT run (no teacher) with a plugin that implements every hook: order, context and effects."""
    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    c = H.cpu_config(tmp_path, tmp_path / "student", H.write_rows(tmp_path / "rows.jsonl"), steps=3)
    c.train.save_steps = 2
    calls = []

    class Everything(TrainerPlugin):
        name = "everything"

        def setup(self, ctx):
            calls.append("setup")

        def transform_model(self, model, ctx):
            calls.append("transform_model")
            model.extra_scale = torch.nn.Parameter(torch.ones(()))  # an added module / parameter
            return None

        def param_groups(self, model, ctx):
            calls.append("param_groups")
            return [{"names": r"extra_scale", "lr": 0.5, "weight_decay": 0.0}]

        def on_train_start(self, model, ctx):
            calls.append("on_train_start")
            assert ctx.total_steps == 3

        def on_forward_start(self, model, batch, ctx):
            calls.append("on_forward_start")

        def loss(self, inputs, ctx):
            calls.append("loss")
            assert inputs.hidden is not None and inputs.rows.numel() > 0
            return (inputs.model.extra_scale - 2.0) ** 2

        def on_before_optimizer_step(self, model, optimizer, ctx):
            calls.append("before_step")
            assert model.extra_scale.grad is not None

        def on_after_optimizer_step(self, model, optimizer, ctx):
            calls.append("after_step")

        def metrics(self, model, ctx):
            ctx.log("custom/logged", 1.0)
            return {"scale": float(model.extra_scale)}

        def export_state_dict(self, state, model, ctx):
            calls.append("export")
            return {k: v for k, v in state.items() if k != "extra_scale"}

        def state_dict(self):
            return {"calls": len(calls)}

        def on_train_end(self, model, ctx):
            calls.append("on_train_end")

    result = H.run_cpu(c, plugins=[Everything()])
    assert calls[:3] == ["setup", "transform_model", "param_groups"]
    assert calls.count("before_step") == 3 and calls.count("after_step") == 3
    assert calls.count("on_forward_start") == calls.count("loss") == 6  # 3 steps x 2 micro-batches
    assert calls[-2:] == ["on_train_end", "export"]
    assert result.metrics["plugins/everything/scale"] > 1.0  # pulled towards 2 at lr 0.5
    assert result.metrics["custom/logged"] == 1.0
    assert (Path(c.train.output_dir) / "step-2" / "plugins.pt").exists()
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final")  # exported without the extra
