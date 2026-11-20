"""LoRA in the RL trainer (the `lora:` section, palingenesis.rl.adapters): what trains, what the rollout engine
receives (the folded weights, with the trainer's base untouched), the adapter-free KL reference, checkpoints
(folded model + adapter) and resume, LoRA+ groups, FSDP on one rank, and configuration errors."""

import json
import sys
from pathlib import Path

import pytest
import torch
import transformers

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy  # noqa: E402

from palingenesis.rl.adapters import ADAPTER_DIR, PolicyAdapters, is_adapter_checkpoint  # noqa: E402
from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402

ADAPTER_SUFFIXES = (".lora_A", ".lora_B", ".lora_magnitude")


def _config(tmp_path, **overrides):
    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(30)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    policy = tmp_path / "policy"
    config = RLConfig()
    for key, value in {
        "model.policy": str(policy) if policy.exists() else tiny_policy(tmp_path),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": str(tmp_path / "data.jsonl"),
        "data.eval_size": 3,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 3,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "rewards.m.fn": "math",
        "rewards.m.weight": 0.0,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-2,
        "train.warmup_steps": 1,
        "lora.rank": 4,
        "lora.alpha": 8,
        **overrides,
    }.items():
        config.set(key, value)
    return config


def _trainer(tmp_path, **overrides):
    from palingenesis.rl.trainer import RLTrainer

    # dense, so a random policy's groups disagree
    return RLTrainer(_config(tmp_path, **overrides), rewards=[lambda completion: len(completion) / 40])


def _local(p):  # FSDP2 parameters are DTensors; on one rank the local shard is the whole tensor
    p = p.detach()
    return p.to_local() if hasattr(p, "to_local") else p


def _perturb(trainer, scale=0.05):
    """Move every adapter away from its init (B = 0 would make every check trivial)."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for m in trainer.adapters.plugin.modules.values():
            m.lora_B.copy_(scale * torch.randn(m.lora_B.shape, generator=g))
            if m.lora_magnitude is not None:
                m.lora_magnitude.mul_(1.0 + scale)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


# ------------------------------------------------------------------ what trains


@pytest.mark.parametrize("fsdp", [False, True], ids=["single", "fsdp2"])
def test_only_the_adapters_train_and_a_run_resumes_from_them(tmp_path, fsdp):
    from palingenesis.rl.trainer import TRAINER_STATE_FILE

    options = {
        "train.save_steps": 2,
        "train.fsdp": fsdp,
        "model.gradient_checkpointing": "full" if fsdp else "none",
        "lora.dora": not fsdp,  # refused under FSDP
    }
    trainer = _trainer(tmp_path, **options)
    named = dict(trainer.model.named_parameters())
    adapters = {n for n in named if n.endswith(ADAPTER_SUFFIXES)}
    assert adapters and {n for n, p in named.items() if p.requires_grad} == adapters
    assert trainer._trainable_names() == [n for n in named if n in adapters]
    assert len(trainer._trainable()) == len(adapters)  # the optimizer holds the adapters alone
    before = {n: _local(p).clone() for n, p in named.items()}
    trainer.train()
    after = {n: _local(p) for n, p in trainer.model.named_parameters()}
    assert all(torch.equal(before[n], after[n]) for n in after if n not in adapters)  # the base is untouched
    assert any(not torch.equal(before[n], after[n]) for n in adapters)
    assert all(n.endswith(".lora_B") is False or bool(after[n].abs().sum() > 0) for n in adapters)

    step = tmp_path / "run" / "step_2"
    assert (step / TRAINER_STATE_FILE).exists() and is_adapter_checkpoint(step)
    assert is_adapter_checkpoint(tmp_path / "run" / "final")
    saved = json.loads((step / ADAPTER_DIR / "adapter_config.json").read_text())
    assert (
        saved["r"] == 4
        and saved["use_dora"] == (not fsdp)
        and saved["base_model_name_or_path"] == str(tmp_path / "policy")
    )

    resumed = _trainer(tmp_path, **options, **{"train.steps": 3, "train.resume_from": "auto"})
    assert resumed.start_step == 2 and resumed.weights.version == 2
    for name, p in resumed.model.named_parameters():  # the base from model.policy, the adapters from the checkpoint
        torch.testing.assert_close(_local(p), after[name], msg=name)
    if not fsdp:  # a plain optimizer state per adapter (FSDP restores it through DCP)
        assert len(resumed.opt.state) == len(adapters)
        for a, b in zip(trainer.opt.state.values(), resumed.opt.state.values()):
            torch.testing.assert_close(a["exp_avg"], b["exp_avg"])
    resumed.train()


def test_a_checkpoint_is_a_plain_model_that_computes_what_the_adapted_policy_computed(tmp_path):
    trainer = _trainer(tmp_path, **{"lora.dora": True})
    trainer.train()
    ids = torch.randint(0, 1000, (2, 12), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    expected = _logits(trainer.model, ids)
    final = tmp_path / "run" / "final"
    plain = transformers.AutoModelForCausalLM.from_pretrained(final, dtype=torch.float32)
    assert not any(n.endswith(ADAPTER_SUFFIXES) for n in plain.state_dict())
    assert json.loads((final / "config.json").read_text())["dtype"] == "float32"
    torch.testing.assert_close(_logits(plain.eval(), ids), expected, rtol=1e-4, atol=1e-5)
    # and it differs from the base it started from
    base = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy", dtype=torch.float32)
    assert not torch.allclose(_logits(base.eval(), ids), expected, atol=1e-5)


def test_peft_loads_a_checkpoint_adapter_on_the_base(tmp_path):
    peft = pytest.importorskip("peft")
    trainer = _trainer(tmp_path, **{"lora.dora": True})
    trainer.train()
    ids = torch.randint(0, 1000, (2, 12), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    base = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy", dtype=torch.float32)
    adapted = peft.PeftModel.from_pretrained(base, tmp_path / "run" / "final" / ADAPTER_DIR).eval()
    torch.testing.assert_close(_logits(adapted, ids), _logits(trainer.model, ids), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------- what the engine gets


@pytest.mark.parametrize("dora", [False, True])
def test_the_engine_gets_base_plus_scaled_ba_and_the_base_is_not_written(tmp_path, dora):
    from palingenesis.lora import fold_weight
    from palingenesis.opd.rollout import checkpoint_named_parameters

    trainer = _trainer(tmp_path, **{"lora.dora": dora})
    _perturb(trainer)
    base = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy", dtype=torch.float32)
    plain_names = {n for n, _ in checkpoint_named_parameters(base)}
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    class Engine:  # what a vLLM engine does with a push: load every tensor under its name
        version = -1

        def update_weights(self, named, version):
            self.loaded, self.version = dict(named), version

    engine = Engine()
    trainer.weights.version = 1
    trainer.weights.sync(engine)
    assert engine.version == 1 and set(engine.loaded) == plain_names  # a plain model's names: no adapter tensors
    modules = trainer.adapters.plugin.modules
    for name, m in modules.items():
        pushed = engine.loaded[f"{name}.weight"]
        delta = m.scaling * (m.lora_B @ m.lora_A)
        expected = fold_weight(m.weight, m.lora_A, m.lora_B, m.scaling, m.lora_magnitude)
        torch.testing.assert_close(pushed, expected)
        if not dora:
            torch.testing.assert_close(pushed, before[f"{name}.weight"] + delta)
        assert pushed is not m.weight and not torch.equal(pushed, m.weight)
    for name, p in trainer.model.named_parameters():  # nothing of the trainer's was written
        assert torch.equal(p, before[name]), name
    untouched = [n for n in plain_names if n.rsplit(".", 1)[0] not in modules]
    assert untouched and all(torch.equal(engine.loaded[n], before[n]) for n in untouched)

    # the pushed weights loaded into a plain model compute what the adapted policy computes
    base.load_state_dict({k: v for k, v in engine.loaded.items()}, strict=False)
    ids = torch.randint(0, 1000, (2, 10), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    torch.testing.assert_close(_logits(base.eval(), ids), _logits(trainer.model, ids), rtol=1e-4, atol=1e-5)


def test_a_bf16_base_is_folded_once_and_the_push_reports_what_survives(tmp_path):
    """One GPU holds the frozen base in bf16. The fold is computed in float32 from the parameters the trainer
    computes with and rounded once into the base's dtype; lora/push_survival is the share of the delta the rounded
    weights carry (precision.edit_survival's projection)."""
    from palingenesis.precision import SurvivalMeter
    from palingenesis.rl import adapters as adapters_module

    trainer = _trainer(tmp_path)
    _perturb(trainer, scale=1e-3)  # a delta near bf16 resolution: part of it is rounded away
    modules = trainer.adapters.plugin.modules
    trainer.adapters.lower_base_precision(trainer.model)  # as on a GPU
    assert all(m.weight.dtype == torch.bfloat16 and m.lora_A.dtype == torch.float32 for m in modules.values())
    base = {name: m.weight.clone() for name, m in modules.items()}
    trainer.adapters.pushes = adapters_module.SURVIVAL_EVERY + 1  # the next push measures
    pushed = dict(trainer.adapters.engine_parameters(trainer.model))
    meter = SurvivalMeter()
    for name, m in modules.items():
        weight = pushed[f"{name}.weight"]
        exact = m.weight.float() + m.scaling * (m.lora_B @ m.lora_A)
        assert weight.dtype == torch.bfloat16 and torch.equal(weight, exact.to(torch.bfloat16))
        assert torch.equal(m.weight, base[name])  # the base is read, not written
        meter.update(m.weight, exact, weight)
    expected = meter.result()["projection"]
    assert 0.5 < expected < 1.5 and expected != 1.0  # rounding moved the delta, without losing or doubling it
    assert trainer.adapters.metrics(trainer.model, 1)["lora/push_survival"] == pytest.approx(expected)
    # other folds (a reference reset) do not count as pushes
    count = trainer.adapters.pushes
    trainer.adapters.folded_state(trainer.model)
    assert trainer.adapters.pushes == count


def test_dora_magnitude_follows_the_base_it_is_computed_with(tmp_path):
    """A fresh run takes DoRA's starting magnitude from the bf16 base (the adapted model is the base at step 0);
    a resumed run keeps the magnitude it trained."""
    trainer = _trainer(tmp_path, **{"lora.dora": True, "train.save_steps": 2})
    modules = trainer.adapters.plugin.modules
    trainer.adapters.lower_base_precision(trainer.model)
    for m in modules.values():
        assert m.weight.dtype == torch.bfloat16
        torch.testing.assert_close(m.lora_magnitude, torch.linalg.vector_norm(m.weight.float(), dim=1), rtol=0, atol=0)
    ids = torch.randint(0, 1000, (1, 8), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    adapted = _logits(trainer.model, ids)
    with trainer.adapters.disabled():
        assert torch.equal(adapted, _logits(trainer.model, ids))  # exactly the base at step 0
    # ... and the first push gives the engine exactly the base weights
    pushed = dict(trainer.adapters.engine_parameters(trainer.model))
    assert all(torch.equal(pushed[f"{name}.weight"], m.weight) for name, m in modules.items())

    _trainer(tmp_path, **{"lora.dora": True, "train.save_steps": 2}).train()
    resumed = _trainer(tmp_path, **{"lora.dora": True, "train.steps": 3, "train.resume_from": "auto"})
    assert resumed.adapters.resumed
    trained = {name: m.lora_magnitude.clone() for name, m in resumed.adapters.plugin.modules.items()}
    resumed.adapters.lower_base_precision(resumed.model)
    for name, m in resumed.adapters.plugin.modules.items():
        assert torch.equal(m.lora_magnitude, trained[name])
    assert any(
        not torch.equal(m.lora_magnitude, torch.linalg.vector_norm(m.weight.float(), dim=1))
        for m in resumed.adapters.plugin.modules.values()
    )


def test_qwen3_5_hybrid_folds_gated_deltanet_and_attention_under_checkpoint_names(tmp_path):
    """The hybrid's default targets (attention, MLP and Gated DeltaNet projections), and a push whose names are the
    checkpoint's own (transformers renames Qwen3.5's parameters at load; engines load the checkpoint's names)."""
    from seco_archs import build

    from palingenesis.opd.rollout import checkpoint_named_parameters

    stock = build("qwen3_5", dtype=torch.float32).eval()
    adapters = PolicyAdapters(_config(tmp_path))
    model = adapters.apply(build("qwen3_5", dtype=torch.float32)).eval()
    leaves = {name.rpartition(".")[2] for name in adapters.plugin.modules}
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} < leaves
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} < leaves and not leaves & {"in_proj_a", "in_proj_b", "lm_head"}
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for m in adapters.plugin.modules.values():
            m.lora_B.copy_(0.02 * torch.randn(m.lora_B.shape, generator=g))
    pushed = dict(adapters.engine_parameters(model))
    plain = dict(checkpoint_named_parameters(stock))
    assert set(pushed) == set(plain) and all(pushed[name].shape == plain[name].shape for name in plain)
    folded = dict(adapters.fold_stream(model.named_parameters()))  # the same fold under the model's own names
    assert set(folded) == {name for name, _ in stock.named_parameters()}
    assert sum(not torch.equal(folded[n], p) for n, p in stock.named_parameters()) == len(adapters.plugin.modules)
    stock.load_state_dict(folded, strict=False)
    ids = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(
        _logits(model, ids), _logits(build("qwen3_5", dtype=torch.float32).eval(), ids), atol=1e-5
    )
    torch.testing.assert_close(_logits(stock, ids), _logits(model, ids), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ reference


def test_the_reference_with_adapters_off_is_the_base_policy(tmp_path):
    from palingenesis.rl.reference import AdapterFreeReference, ReferencePolicy
    from palingenesis.rl.trajectory import Trajectory

    trainer = _trainer(tmp_path, **{"loss.kl_ref": "start", "loss.kl_coef": 0.1})
    assert isinstance(trainer.reference, AdapterFreeReference) and trainer.reference.model is trainer.model
    _perturb(trainer)
    base = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy", dtype=torch.float32).eval()
    ids = torch.randint(0, 1000, (2, 12), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    adapted = _logits(trainer.model, ids)
    with trainer.adapters.disabled():
        torch.testing.assert_close(_logits(trainer.model, ids), _logits(base, ids), rtol=0, atol=0)
    torch.testing.assert_close(_logits(trainer.model, ids), adapted, rtol=0, atol=0)  # and back on afterwards
    assert not torch.allclose(adapted, _logits(base, ids), atol=1e-5)

    # its log-probs are those of a frozen copy of the starting policy (the full-weight path's reference)
    trajectory = Trajectory({}, 0, [5, 6, 7], [])
    trajectory.append_generated([11, 12, 13, 14], [-1.0] * 4, 0)
    copy = ReferencePolicy(str(tmp_path / "policy"), "cpu", trainer.pad_id, 4096)
    copy.model.float()
    trainer.model.train()
    got = trainer.reference.logprobs([trajectory])[id(trajectory)]
    assert trainer.model.training  # the mode is restored
    torch.testing.assert_close(got, copy.logprobs([trajectory])[id(trajectory)], rtol=1e-4, atol=1e-5)
    with pytest.raises(RuntimeError, match="no weights of its own"):
        trainer.reference.reset_to(trainer.model)


def test_kl_to_the_adapter_free_reference_is_logged_and_starts_at_zero(tmp_path):
    trainer = _trainer(tmp_path, **{"loss.kl_ref": "start", "loss.kl_coef": 0.1, "train.steps": 3})
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    trainer.train()
    steps = [m for kind, m in logged if kind == "step"]
    assert len(steps) == 3 and all(m["policy/kl_ref"] >= 0 for m in steps)
    assert steps[0]["policy/kl_ref"] < 1e-8 < steps[-1]["policy/kl_ref"]  # B = 0 at step 0: the policy is the base
    assert all(m["lora/trainable_params"] > 0 and 0 < m["lora/trainable_share"] < 1 for m in steps)
    assert steps[0]["lora/delta_rel"] > 0  # logged after the step's update


def test_a_reset_reference_keeps_a_copy_and_takes_the_folded_policy(tmp_path):
    from palingenesis.rl.reference import AdapterFreeReference

    trainer = _trainer(tmp_path, **{"loss.kl_ref": "start", "loss.kl_ref_reset": 1})
    assert not isinstance(trainer.reference, AdapterFreeReference) and trainer.reference.model is not trainer.model
    trainer.train()
    folded = trainer._folded_state()
    reference = trainer.reference.model.state_dict()
    assert set(folded) == set(reference)
    changed = 0
    for name, tensor in folded.items():
        torch.testing.assert_close(reference[name].float(), tensor.float(), atol=1e-2, rtol=0, msg=name)
        changed += name.rsplit(".", 1)[0] in trainer.adapters.plugin.modules
    assert changed == len(trainer.adapters.plugin.modules)


# ---------------------------------------------------------- options and errors


def test_lora_plus_trains_b_at_its_own_rate(tmp_path):
    trainer = _trainer(tmp_path, **{"lora.lr_ratio": 4.0, "train.steps": 1, "train.warmup_steps": 0})
    assert trainer.lr_scales == (1.0, 4.0) and len(trainer.opt.param_groups) == 2
    b = {id(m.lora_B) for m in trainer.adapters.plugin.modules.values()}
    assert {id(p) for p in trainer.opt.param_groups[1]["params"]} == b
    assert not {id(p) for p in trainer.opt.param_groups[0]["params"]} & b
    trainer.train()
    rates = [group["lr"] for group in trainer.opt.param_groups]
    assert rates[1] == pytest.approx(4.0 * rates[0]) and rates[0] == pytest.approx(1e-2)


def test_prompt_sharing_and_targets(tmp_path):
    trainer = _trainer(tmp_path, **{"train.prompt_sharing": "on", "lora.targets": ["q_proj", "v_proj"]})
    leaves = {name.rsplit(".", 1)[1] for name in trainer.adapters.plugin.modules}
    assert leaves == {"q_proj", "v_proj"}
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    trainer.train()
    moved = {n for n, p in trainer.model.named_parameters() if not torch.equal(p, before[n])}
    assert moved and all(n.endswith(ADAPTER_SUFFIXES) for n in moved)


def test_full_weight_runs_build_no_adapters(tmp_path):
    trainer = _trainer(tmp_path, **{"lora.rank": 0, "lora.alpha": 32.0, "train.learning_rate": 1e-3})
    assert trainer.adapters is None and trainer.lr_scales == () and trainer.weights.named is None
    assert all(p.requires_grad for p in trainer.model.parameters())
    assert not any(n.endswith(ADAPTER_SUFFIXES) for n, _ in trainer.model.named_parameters())
    trainer.train()
    assert not is_adapter_checkpoint(tmp_path / "run" / "final")


def test_resume_refuses_a_checkpoint_of_the_other_kind(tmp_path):
    options = {"train.save_steps": 2}
    _trainer(tmp_path, **options).train()
    with pytest.raises(RLConfigError, match="saved by a run with LoRA adapters"):
        _trainer(tmp_path, **options, **{"lora.rank": 0, "lora.alpha": 32.0, "train.resume_from": "auto"})
    for changed, name in [
        ({"lora.rank": 2}, "lora.rank"),
        ({"lora.alpha": 16.0}, "lora.alpha"),
        ({"lora.rslora": True}, "lora.rslora"),
        ({"lora.dora": True}, "lora.dora"),
    ]:
        with pytest.raises(RLConfigError, match=f"trained with other settings: {name} "):
            _trainer(tmp_path, **options, **changed, **{"train.resume_from": "auto"})
    other = tmp_path / "other_base"
    other.symlink_to(tmp_path / "policy")
    with pytest.raises(RLConfigError, match="trained with other settings: model.policy "):
        _trainer(tmp_path, **options, **{"model.policy": str(other), "train.resume_from": "auto"})
    from palingenesis.hooks import PluginError

    with pytest.raises(PluginError, match="does not match this run's"):  # the same settings on other modules
        _trainer(tmp_path, **options, **{"lora.targets": ["q_proj"], "train.resume_from": "auto"})

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "policy").symlink_to(tmp_path / "policy")
    full = {"lora.rank": 0, "lora.alpha": 32.0, "train.learning_rate": 1e-3, "train.save_steps": 2}
    _trainer(plain, **full).train()
    with pytest.raises(RLConfigError, match="saved without adapters"):
        _trainer(plain, **{"train.save_steps": 2, "train.resume_from": "auto"})


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"lora.rank": -1}, "lora.rank must be an integer >= 0"),
        ({"lora.rank": 0, "lora.alpha": 16.0}, "lora.alpha set without lora.rank"),
        ({"lora.rank": 0, "lora.alpha": 32.0, "lora.targets": ["q_proj"]}, "lora.targets set without lora.rank"),
        ({"lora.alpha": 0.0}, "lora.alpha must be > 0"),
        ({"lora.lr_ratio": 0.0}, "lora.lr_ratio must be > 0"),
        ({"lora.targets": ["(q_proj"]}, "lora.targets: .* is not a regular expression"),
        ({"lora.exclude": ["[a"]}, "lora.exclude: .* is not a regular expression"),
        ({"train.optimizer": "muon"}, "lora with train.optimizer muon is not supported"),
        ({"model.multimodal": True}, "lora with model.multimodal is not supported"),
        ({"cmr.enabled": True}, "lora with cmr is not supported"),
        ({"train.fsdp": True, "lora.lr_ratio": 2.0}, "lora.lr_ratio .* is not supported with FSDP"),
        ({"train.fsdp": True, "lora.dora": True}, "lora.dora is not supported with FSDP"),
    ],
)
def test_configuration_errors(tmp_path, overrides, match):
    with pytest.raises(RLConfigError, match=match):
        _config(tmp_path, **overrides).validate(python_rewards=True)


def test_unknown_options_and_patterns_matching_nothing_are_errors(tmp_path):
    from palingenesis.config import ConfigError

    with pytest.raises(ConfigError, match="lora"):
        _config(tmp_path, **{"lora.dropout": 0.1})
    with pytest.raises(ValueError, match="no nn.Linear matched"):
        _trainer(tmp_path, **{"lora.targets": ["no_such_projection"]})


def test_yaml_section_and_learning_rate_warning(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "rl.yaml").write_text(
        "model: {policy: p}\ndata: {dataset: d.jsonl}\nrewards: {m: math}\n"
        "lora: {rank: 8, alpha: 16, dora: true, targets: [q_proj, v_proj]}\ntrain: {learning_rate: 1.0e-4}\n"
    )
    loaded = RLConfig.from_yaml(tmp_path / "rl.yaml")
    assert (loaded.lora.rank, loaded.lora.alpha, loaded.lora.dora) == (8, 16, True)
    assert not any("learning_rate" in w for w in loaded.validate())  # an adapter rate, not a full-weight one
    loaded.set("lora.rank", 0)
    loaded.set("lora.alpha", 32.0)
    loaded.set("lora.dora", False)
    loaded.set("lora.targets", [])
    assert any("RL recipes use" in w for w in loaded.validate())
    assert config.to_dict()["lora"]["rank"] == 4


# ------------------------------------------------------------------ GPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the bf16 base under autocast needs CUDA")
def test_gpu_base_is_bf16_adapters_stay_fp32_and_the_push_is_bf16(tmp_path):
    trainer = _trainer(tmp_path, **{"train.steps": 2, "loss.kl_ref": "start", "lora.dora": True})
    assert trainer.master is None  # no bf16 copies of the adapters: they are the float32 parameters themselves
    for p in trainer.model.parameters():
        assert p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16)
    base = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if not p.requires_grad}
    step0 = dict(trainer._engine_parameters())
    for name, m in trainer.adapters.plugin.modules.items():
        assert torch.equal(step0[f"{name}.weight"], m.weight)  # the first push is the base
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    trainer.train()
    steps = [m for kind, m in logged if kind == "step"]
    assert steps[0]["policy/kl_ref"] < 1e-6  # the starting policy is the base
    for name, p in trainer.model.named_parameters():
        if name in base:
            assert torch.equal(p, base[name]), name
    pushed = dict(trainer._engine_parameters())
    assert all(t.dtype == torch.bfloat16 for t in pushed.values() if t.is_floating_point())
    name, m = next(iter(trainer.adapters.plugin.modules.items()))
    assert not torch.equal(pushed[f"{name}.weight"], m.weight)
    plain = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "run" / "final", dtype=torch.float32)
    assert next(plain.parameters()).dtype == torch.float32
    exported = dict(plain.named_parameters())
    assert torch.equal(
        exported[f"{name}.weight"].to(torch.bfloat16).to(pushed[f"{name}.weight"].device), pushed[f"{name}.weight"]
    )


@pytest.mark.parametrize("dora", [False, True])
def test_with_a_bf16_base_the_checkpoint_rounds_to_the_weights_the_engine_sampled_with(tmp_path, dora):
    """The layout of one GPU, on the CPU: the frozen base in bf16, the adapters float32 parameters (no lowered
    copies). The policy starts exactly at the base, and the saved float32 fold rounds to the pushed bf16 weights."""
    trainer = _trainer(tmp_path, **{"lora.dora": dora})
    trainer.adapters.lower_base_precision(trainer.model)
    assert trainer.master is None and all(p.dtype == torch.float32 for p in trainer._trainable())
    ids = torch.randint(0, 1000, (1, 8), generator=torch.Generator().manual_seed(0))
    trainer.model.eval()
    with trainer.adapters.disabled():
        base_logits = _logits(trainer.model, ids)
    assert torch.equal(_logits(trainer.model, ids), base_logits)
    _perturb(trainer, scale=0.01)
    assert not torch.equal(_logits(trainer.model, ids), base_logits)
    pushed = dict(trainer.adapters.engine_parameters(trainer.model))
    trainer.save("final")
    saved = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "run" / "final", dtype=torch.float32)
    exported = dict(saved.named_parameters())
    for name, m in trainer.adapters.plugin.modules.items():
        key = f"{name}.weight"
        assert exported[key].dtype == torch.float32 and pushed[key].dtype == torch.bfloat16
        assert torch.equal(exported[key].to(torch.bfloat16), pushed[key]), name
        assert not torch.equal(pushed[key], m.weight)


def test_weight_decay_spares_dora_magnitudes(tmp_path):
    trainer = _trainer(tmp_path, **{"lora.dora": True, "lora.lr_ratio": 2.0, "train.weight_decay": 0.1})
    modules = trainer.adapters.plugin.modules.values()
    rest, b, magnitudes = trainer.opt.param_groups
    assert trainer.lr_scales == (1.0, 2.0, 1.0)
    assert {id(p) for p in rest["params"]} == {id(m.lora_A) for m in modules} and rest["weight_decay"] == 0.1
    assert {id(p) for p in b["params"]} == {id(m.lora_B) for m in modules} and b["weight_decay"] == 0.1
    assert {id(p) for p in magnitudes["params"]} == {id(m.lora_magnitude) for m in modules}
    assert magnitudes["weight_decay"] == 0.0
    plain = _trainer(tmp_path, **{"train.weight_decay": 0.1})  # nothing to split: one group
    assert plain.lr_scales == (1.0,) and len(plain.opt.param_groups) == 1


def test_a_push_that_keeps_little_of_the_delta_warns_once(tmp_path, caplog):
    from palingenesis.rl import adapters as adapters_module

    trainer = _trainer(tmp_path)
    trainer.adapters.lower_base_precision(trainer.model)
    _perturb(trainer, scale=1e-6)  # far below bf16 resolution: the engine would still sample the base
    for pushes, warns in [
        (1, False),
        (adapters_module.SURVIVAL_EVERY + 1, True),
        (2 * adapters_module.SURVIVAL_EVERY + 1, False),
    ]:
        trainer.adapters.pushes = pushes  # the next push measures
        caplog.clear()
        with caplog.at_level("WARNING", logger="palingenesis.rl.adapters"):
            dict(trainer.adapters.engine_parameters(trainer.model))
        assert trainer.adapters.push_survival < adapters_module.SURVIVAL_WARN
        assert ("lora/push_survival" in caplog.text) == warns  # not in the first steps, and only once


def test_a_resumed_run_continues_as_the_uninterrupted_one(tmp_path):
    def final_adapters(root, **overrides):
        root.mkdir(exist_ok=True)
        if not (root / "policy").exists():
            (root / "policy").symlink_to(tmp_path / "policy")
        trainer = _trainer(root, **{"train.save_steps": 2, **overrides})
        trainer.train()
        return {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}

    tiny_policy(tmp_path)
    whole = final_adapters(tmp_path / "whole", **{"train.steps": 4})
    final_adapters(tmp_path / "parts", **{"train.steps": 2})
    resumed = final_adapters(tmp_path / "parts", **{"train.steps": 4, "train.resume_from": "auto"})
    assert whole.keys() == resumed.keys()
    for name in whole:
        # equal up to float noise (a full-weight run resumes with the same ~1e-7 differences)
        torch.testing.assert_close(resumed[name], whole[name], rtol=0, atol=5e-6, msg=name)
    assert max(float(v.abs().max()) for n, v in whole.items() if n.endswith(".lora_B")) > 1e-3  # and it trained


def test_policy_adapters_is_importable_without_a_trainer(tmp_path):
    config = _config(tmp_path)
    adapters = PolicyAdapters(config)
    model = adapters.apply(transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy"))
    assert adapters.plugin.modules and not adapters.resumed
    assert sum(p.requires_grad for p in model.parameters()) == 2 * len(adapters.plugin.modules)


def test_fsdp_push_folds_the_gathered_parameters_under_checkpoint_names(tmp_path):
    """Under FSDP the trainer gathers every parameter (bf16, one at a time) and pushes them to its engine: with
    adapters the stream is folded on the way, wrapper prefixes (activation checkpointing) or not."""
    from palingenesis.opd.rollout import checkpoint_named_parameters

    trainer = _trainer(tmp_path, **{"train.fsdp": True, "model.gradient_checkpointing": "full"})
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for m in trainer.adapters.plugin.modules.values():
            _local(m.lora_B).copy_(0.05 * torch.randn(m.lora_B.shape, generator=g))
    full = trainer.adapters.fold_stream(trainer.parallel.full_parameters(trainer.model), measure=True)
    pushed = dict(checkpoint_named_parameters(trainer.model, full))
    base = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "policy", dtype=torch.float32)
    plain = dict(checkpoint_named_parameters(base))
    assert set(pushed) == set(plain)
    modules = trainer.adapters.plugin.modules
    for name, tensor in pushed.items():
        module = name.rsplit(".", 1)[0]
        if module in modules and name.endswith(".weight"):
            m = modules[module]
            a, b = _local(m.lora_A).to(torch.bfloat16).float(), _local(m.lora_B).to(torch.bfloat16).float()
            expected = plain[name].to(torch.bfloat16).float() + m.scaling * (b @ a)
            assert torch.equal(tensor, expected.to(tensor.dtype)), name
            assert not torch.equal(tensor.float(), plain[name].to(torch.bfloat16).float()), name
        else:
            assert torch.equal(tensor.float(), plain[name].to(tensor.dtype).float()), name
    assert trainer.adapters.pushes == 1
