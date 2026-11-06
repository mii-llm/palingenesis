"""The ternary QAT reference plugin: quantiser, straight-through estimator, strength ramp, layer
substitution (parameters, names and ties preserved, survives checkpoint save / resume), the
weight-to-grid regulariser's gradient, latent clamping, parameter groups, folded export loadable by
a stock model, and a full QAT-while-distilling run through the trainer."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.hooks import PluginContext, Ramp  # noqa: E402
from palingenesis.qat import (  # noqa: E402
    TernaryEmbedding,
    TernaryLinear,
    TernaryQAT,
    absmean_scale,
    quantize_activations,
    quantize_weight,
    ternary,
)


def _assert_ternary(w, group):
    """Every group of `group` weights holds only 0 and one magnitude +-s."""
    for row in w.detach().reshape(-1, group):
        mags = row.abs()[row != 0]
        assert mags.numel() == 0 or torch.allclose(mags, mags[0].expand_as(mags), rtol=1e-5)


def test_ternary_values_and_scales():
    torch.manual_seed(0)
    w = torch.randn(4, 16)
    for g in (0, -1, 8):
        q = quantize_weight(w, g)
        codes, scale = ternary(w, g)
        assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
        torch.testing.assert_close(q, (codes * scale).reshape(w.shape))
    # per-row scale = mean |w| of the row
    torch.testing.assert_close(absmean_scale(w, -1).flatten(), w.abs().mean(1))


def test_straight_through_estimator_and_strength():
    torch.manual_seed(0)
    lin = nn.Linear(16, 4, bias=False)
    t = TernaryLinear(lin, group_size=8)
    assert t.weight is lin.weight
    x = torch.randn(3, 16)
    t(x).sum().backward()
    # STE: the gradient reaches the latent weight as if the forward used it, at the ternary activations
    torch.testing.assert_close(lin.weight.grad, torch.ones(3, 4).T @ x)
    with torch.no_grad():
        torch.testing.assert_close(t(x), x @ quantize_weight(lin.weight, 8).T)
        t.strength.fill_(0.0)
        torch.testing.assert_close(t(x), lin(x))  # strength 0: full precision
        t.strength.fill_(0.5)
        mid = x @ (0.5 * lin.weight + 0.5 * quantize_weight(lin.weight, 8)).T
        torch.testing.assert_close(t(x), mid)


def test_activation_quantisation_is_int8_with_ste():
    x = torch.randn(2, 5, requires_grad=True)
    y = quantize_activations(x, 8)
    scale = x.detach().abs().amax(-1, keepdim=True) / 127
    assert torch.allclose((y / scale), (y / scale).round(), atol=1e-4)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ramp():
    r = Ramp(0.2, 0.6, 0.0, 1.0)
    assert r(0.0) == 0.0 and r(0.4) == pytest.approx(0.5) and r(1.0) == 1.0
    assert Ramp(0.0, 1.0, shape="cosine")(0.5) == pytest.approx(0.5)
    assert Ramp(0.5, 0.5, 1.0, 2.0, "step")(0.49) == 1.0 and Ramp(0.5, 0.5, 1.0, 2.0, "step")(0.5) == 2.0
    assert Ramp.parse(0.3)(0.9) == 0.3
    with pytest.raises(ValueError):
        Ramp(0.6, 0.2)


def _ctx(progress=0.0):
    ctx = PluginContext(config=None, total_steps=100)
    ctx.step = int(progress * 100)
    return ctx


def test_substitution_keeps_parameters_names_and_ties():
    tok_vocab = 64
    model = H.tiny_llama(tok_vocab, hidden=32, tie=True)
    names_before = {n: p for n, p in model.named_parameters()}
    plugin = TernaryQAT(group_size=16, quantize_embedding=True, quantize_head=True)
    model = plugin.transform_model(model, _ctx())
    assert {n for n, _ in model.named_parameters()} == set(names_before)
    for n, p in model.named_parameters():
        assert p is names_before[n]
    assert isinstance(model.lm_head, TernaryLinear) and isinstance(model.model.embed_tokens, TernaryEmbedding)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert isinstance(model.model.layers[0].self_attn.q_proj, TernaryLinear)
    # the default excludes the embedding and the head
    plain = TernaryQAT(group_size=16).transform_model(H.tiny_llama(tok_vocab, hidden=32), _ctx())
    assert type(plain.lm_head) is nn.Linear and type(plain.model.embed_tokens) is nn.Embedding
    with pytest.raises(ValueError, match="no module matched"):
        TernaryQAT(include=["nothing_matches"]).transform_model(H.tiny_llama(tok_vocab), _ctx())


def test_grid_regulariser_gradient_and_clamp():
    model = TernaryQAT(group_size=16).transform_model(H.tiny_llama(64, hidden=32), _ctx())
    plugin = TernaryQAT(group_size=16, grid_weight=0.5, clamp=1.2)
    plugin.transform_model(H.tiny_llama(64, hidden=32), _ctx())
    m = next(iter(plugin.modules.values()))
    m.weight.grad = torch.zeros_like(m.weight)
    w = m.weight.detach().clone()
    plugin.on_before_optimizer_step(None, None, _ctx(0.5))
    torch.testing.assert_close(m.weight.grad, 2 * 0.5 * (w - quantize_weight(w, 16)))
    assert plugin._grid_value > 0
    with torch.no_grad():
        m.weight.mul_(5.0)
        m.weight[0, 0] = 100.0  # far outside its group's range
    scale = absmean_scale(m.weight, 16)  # the bound uses the scale before the clamp
    plugin.on_after_optimizer_step(None, None, _ctx(0.5))
    g = m.weight.detach().reshape(m.weight.shape[0], -1, 16)
    assert (g.abs() <= 1.2 * scale + 1e-6).all()
    assert float(m.weight[0, 0]) == pytest.approx(1.2 * float(scale[0, 0, 0]))
    assert model is not None


def test_param_groups_and_metrics():
    plugin = TernaryQAT(group_size=16, latent_lr=0.01, latent_eps=1e-12)
    model = plugin.transform_model(H.tiny_llama(64, hidden=32), _ctx())
    [group] = plugin.param_groups(model, _ctx())
    assert group["lr"] == 0.01 and group["eps"] == 1e-12 and group["weight_decay"] == 0.0
    assert {id(p) for p in group["params"]} == {id(m.weight) for m in plugin.modules.values()}
    plugin.on_train_start(model, _ctx())
    first = plugin.metrics(model, _ctx(0.0))
    assert first["codes_changed"] == 0.0 and 0 < first["zero_share"] < 1
    with torch.no_grad():
        for m in plugin.modules.values():
            m.weight.neg_()
    assert plugin.metrics(model, _ctx(0.5))["codes_changed"] > 0.5


def test_apply_param_groups_moves_parameters():
    from palingenesis.hooks import PluginError, apply_param_groups

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    moved = apply_param_groups(opt, model, [{"names": r"^1\.", "lr": 5e-3, "eps": 1e-12}])
    assert moved == 2
    assert [len(g["params"]) for g in opt.param_groups] == [2, 2]
    new = opt.param_groups[1]
    assert new["lr"] == 5e-3 and new["eps"] == 1e-12 and new["weight_decay"] == 0.1
    with pytest.raises(PluginError, match="selects no"):
        apply_param_groups(opt, model, [{"names": "missing", "lr": 1.0}])


def test_folded_export_loads_into_a_stock_model(tmp_path):
    from transformers import LlamaForCausalLM

    plugin = TernaryQAT(group_size=16, quantize_embedding=True, quantize_head=True)
    model = plugin.transform_model(H.tiny_llama(64, hidden=32, tie=True), _ctx())
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    model.save_pretrained(tmp_path, state_dict=state)
    stock = LlamaForCausalLM.from_pretrained(tmp_path)
    w = stock.model.layers[0].mlp.up_proj.weight
    _assert_ternary(w, 16)
    torch.testing.assert_close(stock.lm_head.weight, stock.model.embed_tokens.weight)
    # the stock model computes what the quantised training model computes
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=ids).logits, model(input_ids=ids).logits, rtol=1e-4, atol=1e-4)


def test_substitution_survives_checkpoint_resume(tmp_path):
    from palingenesis.checkpoint import load_checkpoint, save_checkpoint

    tok = H.tokenizer()

    def build(seed):
        plugin = TernaryQAT(group_size=16)
        model = plugin.transform_model(H.tiny_llama(len(tok), hidden=32, seed=seed), _ctx())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    model, opt = build(0)
    model(input_ids=torch.randint(0, 100, (1, 6))).logits.sum().backward()
    opt.step()
    save_checkpoint(model, tok, opt, None, 1, str(tmp_path), keep_latest_k=0)
    fresh, fresh_opt = build(1)
    load_checkpoint(fresh, fresh_opt, None, str(tmp_path / "step-1"))
    for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        torch.testing.assert_close(a, b, msg=n)


def test_qat_while_distilling_through_the_trainer(tmp_path):
    from transformers import AutoModelForCausalLM

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=4)
    c.train.save_steps = 2
    c.distill.enabled = True
    c.distill.teachers = {"fp": {"model": str(tmp_path / "student"), "dtype": "float32"}}  # the full-precision model
    c.plugins.custom = [
        {
            "class": "palingenesis.qat:TernaryQAT",
            "args": {
                "group_size": 16,
                "strength": {"start": 0.0, "end": 0.5},
                "grid_weight": 1e-3,
                "latent_lr": 2e-3,
                "hidden_layers": [0, -1],
                "metrics_every": 1,
            },
        }
    ]
    result = H.run_cpu(c)
    m = result.metrics
    assert m["plugins/ternary_qat/strength"] == 1.0
    assert "plugins/ternary_qat/codes_changed" in m and "train/feature/ternary_qat/0" in m
    assert (Path(c.train.output_dir) / "step-2" / "plugins.pt").exists()
    final = AutoModelForCausalLM.from_pretrained(Path(c.train.output_dir) / "final")
    w = final.model.layers[1].self_attn.o_proj.weight
    _assert_ternary(w, 16)

    # resume from the step-2 checkpoint: plugin state restored, two more steps
    c.train.resume_from = str(Path(c.train.output_dir) / "step-2")
    assert H.run_cpu(c).global_step == 4
