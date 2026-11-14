"""The LoRA plugin: the adapted layer (init equivalence, rsLoRA scaling, DoRA), what trains and what holds
optimizer state, folding (exact in fp32, bf16 rounding), checkpoint resume, a Qwen3.5 hybrid with Gated
DeltaNet layers, activation checkpointing, train.freeze_non_attention, LoRA+ groups, configuration errors,
and runs through the trainer (SFT and DPO) whose exports compute what the adapted model computed."""

import json
import logging
import re
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.hooks import PluginContext, PluginError, apply_param_groups  # noqa: E402
from palingenesis.lora import DEFAULT_TARGETS, LoRA, LoRALinear, fold_weight  # noqa: E402

LLAMA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _ctx(config=None, **kw):
    return PluginContext(config=config, total_steps=10, **kw)


def _adapted(plugin=None, model=None, **kw):
    plugin = plugin or LoRA(rank=4, alpha=8, **kw)
    model = model if model is not None else H.tiny_llama(64, hidden=32)
    return plugin, plugin.transform_model(model, _ctx())


def _perturb(plugin, scale=0.05, seed=1):
    """Move every adapter away from its init (B = 0 would make every check trivial)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in plugin.modules.values():
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * scale)
            if m.lora_magnitude is not None:
                m.lora_magnitude.mul_(1 + 0.1 * torch.rand(m.lora_magnitude.shape, generator=g))


def _ids(n=11, vocab=64, seed=0):
    return torch.randint(0, vocab, (2, n), generator=torch.Generator().manual_seed(seed))


# ─────────────────────────────────────────────────────────────── the layer


@pytest.mark.parametrize("dora", [False, True])
@pytest.mark.parametrize("rslora", [False, True])
def test_adapted_model_equals_the_base_at_init(dora, rslora):
    base = H.tiny_llama(64, hidden=32)
    ids = _ids()
    with torch.no_grad():
        expected = base(input_ids=ids).logits
    plugin, model = _adapted(model=base, dora=dora, rslora=rslora)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, expected, rtol=0, atol=1e-6)
    names = {n.rpartition(".")[2] for n in plugin.modules}
    assert names == LLAMA_TARGETS
    assert type(model.lm_head) is nn.Linear  # the head is never adapted


def test_layer_math_rslora_and_dora():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    lora = LoRALinear(lin, rank=4, alpha=8)
    assert lora.scaling == 2.0 and lora.weight is lin.weight and lora.bias is lin.bias
    assert LoRALinear(nn.Linear(16, 8), rank=4, alpha=8, rslora=True).scaling == 4.0  # alpha / sqrt(rank)
    with torch.no_grad():
        lora.lora_B.normal_()
    expected = x @ lin.weight.T + lin.bias + 2.0 * x @ lora.lora_A.T @ lora.lora_B.T
    torch.testing.assert_close(lora(x), expected)
    torch.testing.assert_close(x @ lora.merged_weight().T + lin.bias, expected)

    dora = LoRALinear(lin, rank=4, alpha=8, dora=True)
    torch.testing.assert_close(dora.lora_magnitude.detach(), lin.weight.detach().norm(dim=1))
    with torch.no_grad():
        dora.lora_B.normal_()
        dora.lora_magnitude.mul_(1.5)
    w = lin.weight + 2.0 * dora.lora_B @ dora.lora_A
    w_dora = dora.lora_magnitude[:, None] * w / w.norm(dim=1, keepdim=True)
    torch.testing.assert_close(dora(x), x @ w_dora.T + lin.bias, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dora.merged_weight(), w_dora.detach(), rtol=1e-5, atol=1e-6)
    # the magnitude learns; the row norm is a constant of the backward (DoRA, section 4.3)
    dora(x).sum().backward()
    assert dora.lora_magnitude.grad is not None and dora.lora_magnitude.grad.abs().sum() > 0
    assert dora.lora_A.grad is not None and lin.weight.grad is None


def test_bf16_base_with_fp32_adapters():
    lin = nn.Linear(16, 8).to(torch.bfloat16)
    lora = LoRALinear(lin, rank=4, alpha=8)
    assert lora.weight.dtype == torch.bfloat16 and lora.lora_A.dtype == lora.lora_B.dtype == torch.float32
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    y = lora(x)  # no autocast: the adapter path computes in fp32 and returns the input's dtype
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()
    assert lora.lora_A.grad.abs().sum() == 0  # B = 0: A gets a zero gradient on the first step, by design
    assert lora.lora_B.grad.dtype == torch.float32 and lora.lora_B.grad.abs().sum() > 0


def test_adapter_init_is_seeded_by_module_name():
    a = LoRA(rank=4).transform_model(H.tiny_llama(64, hidden=32, seed=0), _ctx())
    torch.manual_seed(123)  # the global RNG (another rank's seed) does not matter
    b = LoRA(rank=4).transform_model(H.tiny_llama(64, hidden=32, seed=0), _ctx())
    q_a = a.model.layers[0].self_attn.q_proj.lora_A
    torch.testing.assert_close(q_a, b.model.layers[0].self_attn.q_proj.lora_A)
    assert not torch.equal(q_a, a.model.layers[0].self_attn.k_proj.lora_A[:, : q_a.shape[1]])


# ─────────────────────────────────────────────────────────────── what trains


def test_only_adapters_get_gradients_and_optimizer_state():
    from palingenesis.optim import build_optimizer

    plugin, model = _adapted()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(re.search(r"\.lora_(A|B)$", n) for n in trainable)
    assert len(trainable) == 2 * len(plugin.modules)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, 1e-2, 0.0)
    for _ in range(2):
        model(input_ids=_ids(), labels=_ids()).loss.backward()
        opt.step()
        opt.zero_grad()
    with_state = {id(p) for p in opt.state}
    params = dict(model.named_parameters())
    assert with_state == {id(params[n]) for n in trainable}
    for n, p in params.items():
        changed = not torch.equal(p.detach(), before[n])
        assert changed == (n in trainable), n


def test_default_targets_and_patterns():
    assert set(DEFAULT_TARGETS) >= LLAMA_TARGETS
    plugin, _ = _adapted(LoRA(rank=2, targets=[r"self_attn\.(q|v)_proj"]))
    assert sorted(plugin.modules) == [f"model.layers.{i}.self_attn.{p}" for i in (0, 1) for p in ("q_proj", "v_proj")]
    plugin, _ = _adapted(LoRA(rank=2, targets=["q_proj"], exclude=[r"layers\.1\."]))
    assert list(plugin.modules) == ["model.layers.0.self_attn.q_proj"]
    # a pattern matches whole name components: "proj" alone is not "q_proj"
    with pytest.raises(ValueError, match="no nn.Linear matched.*q_proj"):
        _adapted(LoRA(rank=2, targets=["proj"]))


def test_tied_embedding_model_exports_to_a_stock_model(tmp_path):
    from transformers import LlamaForCausalLM

    plugin, model = _adapted(model=H.tiny_llama(64, hidden=32, tie=True))
    assert model.lm_head.weight is model.model.embed_tokens.weight and not model.lm_head.weight.requires_grad
    _perturb(plugin)
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    assert not [k for k in state if "lora_" in k]
    model.save_pretrained(tmp_path, state_dict=state)
    stock = LlamaForCausalLM.from_pretrained(tmp_path)
    ids = _ids()
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=ids).logits, model(input_ids=ids).logits, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────── folding


@pytest.mark.parametrize("dora", [False, True])
def test_folding_reproduces_the_adapted_logits(dora, tmp_path):
    from transformers import LlamaForCausalLM

    plugin, model = _adapted(dora=dora, rslora=True)
    _perturb(plugin)
    live = {k: v.clone() for k, v in model.state_dict().items()}
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    for k, v in model.state_dict().items():  # the live model is untouched
        assert torch.equal(v, live[k]), k
    assert set(state) == set(H.tiny_llama(64, hidden=32).state_dict())
    model.save_pretrained(tmp_path, state_dict=state)
    stock = LlamaForCausalLM.from_pretrained(tmp_path)
    ids = _ids()
    with torch.no_grad():
        adapted = model(input_ids=ids).logits
        torch.testing.assert_close(stock(input_ids=ids).logits, adapted, rtol=1e-5, atol=1e-5)
        assert (adapted - H.tiny_llama(64, hidden=32)(input_ids=ids).logits).abs().max() > 1e-2


@pytest.mark.parametrize("scale, kept", [(2e-3, (0.95, 1.05)), (1e-5, (0.0, 0.3))])
def test_bf16_fold_rounds_to_nearest_and_reports_what_survives(scale, kept, caplog):
    """A bf16 base: the fold is the fp32 adapted weight rounded to nearest. A delta well above half an ulp
    survives; one far below it is mostly lost, and the export says so."""
    from palingenesis.precision import edit_survival

    plugin, model = _adapted(model=H.tiny_llama(64, hidden=64))
    model.to(torch.bfloat16)
    _perturb(plugin, scale=scale)
    state = dict(model.state_dict())
    with caplog.at_level(logging.INFO):
        folded = plugin.export_state_dict(state, model, _ctx())
    keys = [f"{n}.weight" for n in plugin.modules]
    intended = {k: plugin.modules[k[: -len(".weight")]].merged_weight() for k in keys}
    for k in keys:
        assert torch.equal(folded[k], intended[k].to(torch.bfloat16)), k
    survival = edit_survival({k: state[k] for k in keys}, intended, {k: folded[k] for k in keys})["projection"]
    assert kept[0] < survival < kept[1]
    assert f"keep {survival:.3f} of the delta" in caplog.text
    assert ("cannot hold much of the adapters' delta" in caplog.text) == (survival < 0.9)


def test_fold_weight_matches_the_layer():
    torch.manual_seed(0)
    lin = nn.Linear(12, 6, bias=False)
    m = LoRALinear(lin, rank=3, alpha=6, dora=True)
    with torch.no_grad():
        m.lora_B.normal_()
    torch.testing.assert_close(
        fold_weight(lin.weight, m.lora_A, m.lora_B, m.scaling, m.lora_magnitude), m.merged_weight()
    )


# ─────────────────────────────────────────────────────────────── resume


def test_checkpoint_resume_restores_adapters(tmp_path):
    from palingenesis.checkpoint import load_checkpoint, save_checkpoint
    from palingenesis.optim import build_optimizer

    tok = H.tokenizer()

    def build(seed):
        plugin = LoRA(rank=4, dora=True)
        model = plugin.transform_model(H.tiny_llama(len(tok), hidden=32, seed=seed), _ctx())
        return model, build_optimizer(model, 1e-2, 0.0)

    model, opt = build(0)
    ids = torch.randint(0, 100, (1, 6))
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    save_checkpoint(model, tok, opt, None, 1, str(tmp_path), keep_latest_k=0)
    fresh, fresh_opt = build(0)
    assert fresh.model.layers[0].mlp.up_proj.lora_B.abs().sum() == 0
    load_checkpoint(fresh, fresh_opt, None, str(tmp_path / "step-1"))
    for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        torch.testing.assert_close(a, b, msg=n)
    assert len(fresh_opt.state) == len(opt.state) > 0


# ─────────────────────────────────────────────────────────────── hybrid, checkpointing, freezing


def _hybrid():
    from seco_archs import build

    return build("qwen3_5", dtype=torch.float32)


def test_qwen3_5_hybrid_adapts_gated_deltanet_and_attention():
    from palingenesis.train import _freeze_non_attention_layers

    base = _hybrid()
    ids = _ids(vocab=97)
    with torch.no_grad():
        expected = base(input_ids=ids).logits
    plugin, model = _adapted(LoRA(rank=4, alpha=8), base)
    leaves = {n.rpartition(".")[2] for n in plugin.modules}
    assert leaves == LLAMA_TARGETS | {"in_proj_qkv", "in_proj_z", "out_proj"}
    assert not any(n.endswith(("in_proj_a", "in_proj_b")) for n in plugin.modules)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, expected, rtol=0, atol=1e-6)
    _perturb(plugin, scale=0.02)
    model(input_ids=ids, labels=ids).loss.backward()
    assert plugin.modules["model.layers.0.linear_attn.in_proj_qkv"].lora_A.grad.abs().sum() > 0
    assert plugin.modules["model.layers.2.self_attn.q_proj"].lora_A.grad.abs().sum() > 0
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    stock = _hybrid()
    stock.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=ids).logits, model(input_ids=ids).logits, rtol=1e-5, atol=1e-5)

    # train.freeze_non_attention after the plugin: the linear-attention layers' adapters freeze too, the base
    # stays frozen (the freeze never unfreezes), the attention layer's adapters train
    _freeze_non_attention_layers(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        f"{n}.{p}" for n in plugin.modules if n.startswith("model.layers.2.") for p in ("lora_A", "lora_B")
    }
    assert plugin.param_groups(model, _ctx()) == []  # no base weight was thawed


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_activation_checkpointing_delivers_adapter_gradients(mode):
    """The embedding is frozen, so no layer input requires grad: non-reentrant checkpointing must still
    give the adapters inside each checkpointed layer the gradients of the plain forward."""
    from palingenesis.kernels import apply_activation_checkpointing

    def grads(ac):
        plugin, model = _adapted()
        _perturb(plugin)
        if ac:
            apply_activation_checkpointing(model, mode=mode)
        ids = _ids()
        model(input_ids=ids, labels=ids).loss.backward()
        return {n: p.grad for n, p in model.named_parameters() if p.requires_grad}

    plain, checkpointed = grads(False), grads(True)
    assert len(plain) == len(checkpointed)
    for (n, g), (_, h) in zip(sorted(plain.items()), sorted(checkpointed.items())):
        assert g is not None and h is not None, n
        torch.testing.assert_close(h, g, rtol=1e-5, atol=1e-6, msg=n)


def test_a_thawed_base_weight_is_refused():
    plugin, model = _adapted()
    model.model.layers[0].mlp.up_proj.weight.requires_grad_(True)
    with pytest.raises(PluginError, match="made trainable"):
        plugin.param_groups(model, _ctx())


# ─────────────────────────────────────────────────────────────── LoRA+, metrics, configuration


def test_lora_plus_puts_b_in_a_group_of_its_own():
    from palingenesis.config import Config
    from palingenesis.optim import build_optimizer

    config = Config()
    config.train.learning_rate = 1e-4
    plugin, model = _adapted(LoRA(rank=4, lr_ratio=16))
    [override] = plugin.param_groups(model, _ctx(config))
    assert override["lr"] == pytest.approx(1.6e-3)
    assert {id(p) for p in override["params"]} == {id(m.lora_B) for m in plugin.modules.values()}
    opt = build_optimizer(model, 1e-4, 0.0)
    apply_param_groups(opt, model, [override])
    by_lr = {}
    for g in opt.param_groups:
        by_lr.setdefault(g["lr"], set()).update(id(p) for p in g["params"])
    assert by_lr[override["lr"]] == {id(m.lora_B) for m in plugin.modules.values()}
    assert {id(m.lora_A) for m in plugin.modules.values()} <= by_lr[1e-4]
    assert LoRA(rank=4).param_groups(model, _ctx(config)) == []


def test_metrics_report_trainable_share_and_the_relative_delta():
    plugin, model = _adapted()
    plugin.on_train_start(model, _ctx())
    m = plugin.metrics(model, _ctx())
    assert m["delta_rel"] == 0.0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert m["trainable_params"] == trainable and 0 < m["trainable_share"] < 0.5
    _perturb(plugin)
    m = plugin.metrics(model, _ctx())
    num = sum(((mod.merged_weight() - mod.weight) ** 2).sum() for mod in plugin.modules.values())
    den = sum((mod.weight**2).sum() for mod in plugin.modules.values())
    assert m["delta_rel"] == pytest.approx(float((num / den) ** 0.5), rel=1e-4)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rank": 0}, "rank must be a positive integer"),
        ({"rank": 4.0}, "rank must be a positive integer"),
        ({"alpha": 0}, "alpha must be > 0"),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"lr_ratio": 0}, "lr_ratio must be > 0"),
        ({"targets": []}, "targets must be a non-empty list"),
        ({"targets": ["q_proj("]}, "invalid pattern"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=f"lora: {match}"):
        LoRA(**kwargs)


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("train", "optimizer", "muon"),
        ("train", "hyperball", True),
        ("train", "mona", True),
        ("train", "base_merge", True),
        ("memory", "float8_training", True),
    ],
)
def test_setup_refuses_full_matrix_options(section, field, value):
    from palingenesis.config import Config

    config = Config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(PluginError, match="lora does not support"):
        LoRA().setup(_ctx(config))
    LoRA().setup(_ctx(Config()))


def test_setup_refuses_lora_plus_with_llrd():
    from palingenesis.config import Config

    config = Config()
    config.train.llrd_decay = 0.9
    with pytest.raises(PluginError, match="llrd_decay"):
        LoRA(lr_ratio=4).setup(_ctx(config))
    LoRA().setup(_ctx(config))  # plain LoRA with LLRD: the adapters take their layer's rate


# ─────────────────────────────────────────────────────────────── through the trainer


class _Recording(LoRA):
    """LoRA that records the adapters at the start and the adapted model's logits at the end."""

    def on_train_start(self, model, ctx):
        super().on_train_start(model, ctx)
        self.start_b = {n: m.lora_B.detach().clone() for n, m in self.modules.items()}

    def on_train_end(self, model, ctx):
        with torch.no_grad():
            self.final_logits = model(input_ids=self.probe).logits.clone()


def _rows_loss(model, tok) -> float:
    """Mean next-token loss of the training rows' assistant answers (fixed data: comparable across models)."""
    total = 0.0
    with torch.no_grad():
        for row in H.ROWS:
            text = tok.apply_chat_template(row["messages"][:1], add_generation_prompt=True, tokenize=False)
            prompt = tok(text)["input_ids"]
            answer = tok(row["messages"][1]["content"])["input_ids"]
            ids = torch.tensor([prompt + answer])
            labels = ids.clone()
            labels[0, : len(prompt)] = -100
            total += float(model(input_ids=ids, labels=labels).loss)
    return total / len(H.ROWS)


@pytest.mark.parametrize("dora", [False, True])
def test_sft_run_learns_and_exports_the_adapted_model(dora, tmp_path, caplog):
    from transformers import AutoModelForCausalLM

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=12)
    c.train.learning_rate, c.train.min_learning_rate, c.train.epochs = 3e-2, 3e-3, 2
    c.train.gradient_checkpointing = "selective"
    plugin = _Recording(rank=8, alpha=16, dora=dora, metrics_every=1)
    plugin.probe = torch.randint(0, len(tok), (2, 9), generator=torch.Generator().manual_seed(0))
    with caplog.at_level(logging.INFO):
        result = H.run_cpu(c, plugins=[plugin])
    m = result.metrics
    assert m["plugins/lora/delta_rel"] > 0 and 0 < m["plugins/lora/trainable_share"] < 0.5
    assert ("plugins/lora/magnitude_change" in m) == dora
    assert any("lora memory: frozen base" in r.getMessage() for r in caplog.records)

    final = Path(c.train.output_dir) / "final"
    exported = AutoModelForCausalLM.from_pretrained(final)
    with torch.no_grad():
        torch.testing.assert_close(exported(input_ids=plugin.probe).logits, plugin.final_logits, rtol=1e-5, atol=1e-5)
    # the base weights never moved: only what the adapters fold in differs from the student
    student = AutoModelForCausalLM.from_pretrained(tmp_path / "student")
    assert torch.equal(exported.model.embed_tokens.weight, student.model.embed_tokens.weight)
    assert not torch.equal(exported.model.layers[0].mlp.down_proj.weight, student.model.layers[0].mlp.down_proj.weight)
    # and it learned the rows: a lower loss than the base on the same data
    before, after = _rows_loss(student, tok), _rows_loss(exported, tok)
    assert after < before - 0.1, (before, after)

    cfg = json.loads((final / "adapter" / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 8 and cfg["use_dora"] == dora
    assert cfg["target_modules"] == sorted(plugin.modules)


def test_resumed_run_continues_from_the_saved_adapters(tmp_path):
    from safetensors.torch import load_file

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=4)
    c.train.save_steps = 2
    first = _Recording(rank=4)
    first.probe = torch.zeros(1, 3, dtype=torch.long)
    H.run_cpu(c, plugins=[first])
    assert all(b.abs().sum() == 0 for b in first.start_b.values())  # a fresh run starts from B = 0

    saved = {}
    for f in (Path(c.train.output_dir) / "step-2" / "model").glob("*.safetensors"):
        saved.update(load_file(str(f)))
    c.train.resume_from = str(Path(c.train.output_dir) / "step-2")
    resumed = _Recording(rank=4)
    resumed.probe = first.probe
    assert H.run_cpu(c, plugins=[resumed]).global_step == 4
    for name, b in resumed.start_b.items():
        torch.testing.assert_close(b, saved[f"{name}.lora_B"].float(), msg=name)
    assert any(b.abs().sum() > 0 for b in resumed.start_b.values())


def test_freeze_non_attention_run_on_a_dense_model_keeps_the_base_frozen(tmp_path):
    """A dense model has nothing to freeze: the run trains the adapters, not the base weights."""
    from transformers import AutoModelForCausalLM

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=2)
    c.train.freeze_non_attention = True
    c.plugins.custom = [{"class": "palingenesis.lora:LoRA", "args": {"rank": 4, "targets": ["q_proj", "v_proj"]}}]
    H.run_cpu(c)
    exported = AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final")
    student = AutoModelForCausalLM.from_pretrained(tmp_path / "student")
    for (n, a), (_, b) in zip(exported.named_parameters(), student.named_parameters()):
        assert torch.equal(a, b) == (not n.endswith(("q_proj.weight", "v_proj.weight"))), n


def test_dpo_run_with_lora(tmp_path):
    """The DPO reference model is loaded separately and stays the base; the policy trains its adapters."""
    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    rows = [
        {
            "prompt": [{"role": "user", "content": f"What is {i} plus {i + 1}?"}],
            "chosen": [{"role": "assistant", "content": f"It is {2 * i + 1}."}],
            "rejected": [{"role": "assistant", "content": f"It is {2 * i + 7}, I think."}],
        }
        for i in range(16)
    ]
    data = H.write_rows(tmp_path / "pairs.jsonl", rows)
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=4)
    c.dpo.enabled = True
    c.train.learning_rate = 1e-2
    result = H.run_cpu(c, plugins=[LoRA(rank=4)])
    assert result.global_step == 4
    assert result.metrics["plugins/lora/delta_rel"] > 0
    assert "train/rewards/accuracies" in result.metrics


def test_peft_loads_the_exported_adapter(tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=3)
    c.train.learning_rate = 1e-2
    for dora in (False, True):
        c.train.output_dir = str(tmp_path / f"out-{dora}")
        H.run_cpu(c, plugins=[LoRA(rank=4, alpha=8, rslora=True, dora=dora)])
        final = Path(c.train.output_dir) / "final"
        base = AutoModelForCausalLM.from_pretrained(tmp_path / "student")
        with_adapter = peft.PeftModel.from_pretrained(base, final / "adapter").eval()
        merged = AutoModelForCausalLM.from_pretrained(final)
        ids = _ids(vocab=len(tok))
        with torch.no_grad():
            torch.testing.assert_close(
                with_adapter(input_ids=ids).logits, merged(input_ids=ids).logits, rtol=1e-4, atol=1e-4
            )
