"""Ternary QAT components (palingenesis.qat.ternary): scale schemes, quantisers, straight-through estimators,
scale modes of the plugin, and the ternary merges (palingenesis.qat.merge), whose outputs must be ternary."""

import itertools
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.hooks import PluginContext  # noqa: E402
from palingenesis.merge import MergeError  # noqa: E402
from palingenesis.merge import main as merge_main  # noqa: E402
from palingenesis.qat import TernaryLinear, TernaryQAT, quantize_weight  # noqa: E402
from palingenesis.qat.merge import METHODS, check_ternary, merge_tensor, merge_ternary  # noqa: E402
from palingenesis.qat.ternary import (  # noqa: E402
    QUANTIZERS,
    SCALES,
    STES,
    TernarySpec,
    decode_ternary,
    group_view,
    is_ternary,
)


def _ctx(progress=0.0):
    ctx = PluginContext(config=None, total_steps=100)
    ctx.step = int(progress * 100)
    return ctx


# ─────────────────────────────────────────────────────────────── quantiser


@pytest.mark.parametrize("scale", sorted(SCALES))
@pytest.mark.parametrize("group", [0, -1, 8])
def test_every_scale_scheme_gives_ternary_groups(scale, group):
    torch.manual_seed(0)
    w = torch.randn(6, 16)
    spec = TernarySpec(group, scale)
    q = spec.quantize(w)
    assert q.shape == w.shape and is_ternary(q, group)
    codes, alpha = decode_ternary(q, group)
    assert set(codes.unique().tolist()) <= {-1, 0, 1} and (alpha >= 0).all()
    # codes never flip a sign, and the quantiser is idempotent on its own output's codes
    assert (torch.sign(q) * torch.sign(w) >= 0).all()


def test_absmean_matches_the_bitnet_formula_and_twn_absmax_their_definitions():
    torch.manual_seed(1)
    w = torch.randn(4, 16)
    torch.testing.assert_close(TernarySpec(8, "absmean").quantize(w), quantize_weight(w, 8))
    g = group_view(w, -1)
    alpha, delta = SCALES["twn"](g)
    torch.testing.assert_close(delta, 0.7 * g.abs().mean(-1, keepdim=True))
    for row, a, d in zip(g[:, 0], alpha.flatten(), delta.flatten()):
        torch.testing.assert_close(a, row.abs()[row.abs() > d].mean())
    alpha, delta = SCALES["absmax"](g)
    torch.testing.assert_close(alpha, g.abs().amax(-1, keepdim=True) * 2 / 3)
    torch.testing.assert_close(delta, alpha / 2)


def test_mse_scale_is_the_exact_optimum():
    torch.manual_seed(2)
    w = torch.randn(5, 7)
    spec = TernarySpec(-1, "mse")
    err = (w - spec.quantize(w)).pow(2).sum(-1)
    for i, row in enumerate(w):
        best = float("inf")
        for codes in itertools.product((-1.0, 0.0, 1.0), repeat=7):
            c = torch.tensor(codes)
            if c.abs().sum() == 0:
                continue
            alpha = (row @ c) / (c @ c)
            best = min(best, float((row - alpha * c).pow(2).sum()))
        assert float(err[i]) == pytest.approx(best, rel=1e-5)
    # and it is never worse than the other schemes
    for other in ("absmean", "twn", "absmax"):
        assert (err <= (w - TernarySpec(-1, other).quantize(w)).pow(2).sum(-1) + 1e-6).all()


def test_stochastic_quantiser_is_unbiased_inside_the_range_and_deterministic_at_eval():
    torch.manual_seed(3)
    g = torch.tensor([[[0.3, -0.6, 0.0, 1.7]]]).repeat(1, 1, 1)
    alpha, delta = torch.ones(1, 1, 1), torch.full((1, 1, 1), 0.5)
    draws = torch.stack([QUANTIZERS["stochastic"](g, alpha, delta) for _ in range(4000)]).mean(0)
    torch.testing.assert_close(draws, torch.tensor([[[0.3, -0.6, 0.0, 1.0]]]), atol=0.03, rtol=0)
    lin = nn.Linear(16, 4, bias=False)
    t = TernaryLinear(lin, 8, spec=TernarySpec(8, quantizer="stochastic"))
    x = torch.randn(2, 16)
    t.eval()
    torch.testing.assert_close(t(x), t(x))
    torch.testing.assert_close(t(x), x @ quantize_weight(lin.weight, 8).T)
    t.train()
    assert is_ternary(t.effective_weight().detach(), 8)


# ─────────────────────────────────────────────────────────────── estimators


@pytest.mark.parametrize("ste", sorted(STES))
def test_forward_is_ternary_whatever_the_estimator(ste):
    torch.manual_seed(4)
    w = torch.randn(4, 16, requires_grad=True)
    spec = TernarySpec(8, "mse", ste=ste)
    eff = spec.effective(w)
    torch.testing.assert_close(eff.detach(), spec.quantize(w))
    eff.sum().backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


def test_estimator_gradients():
    w = torch.tensor([[0.1, 0.45, 0.55, 0.9, 1.4, -2.5, -0.5, 0.0]], requires_grad=True)
    fixed = (torch.ones(1, 1, 1), torch.full((1, 1, 1), 0.5))

    def grad(ste, **kw):
        w.grad = None
        TernarySpec(-1, ste=ste, ste_args=kw).effective(w, fixed=fixed).sum().backward()
        return w.grad.clone()

    torch.testing.assert_close(grad("identity"), torch.ones(1, 8))
    clipped = grad("clipped", clip=1.0)
    torch.testing.assert_close(clipped, torch.tensor([[1.0, 1, 1, 1, 0, 0, 1, 1]]))
    soft = grad("tanh", sharpness=6.0)
    # the slope concentrates at the thresholds (|w| = 0.5) and vanishes far from them
    assert soft[0, 1] > soft[0, 0] and soft[0, 2] > soft[0, 3] > soft[0, 4]
    assert soft[0, 5] < 1e-3 and soft[0, 6] > soft[0, 7]
    sharper = grad("tanh", sharpness=12.0)
    assert sharper[0, 0] < soft[0, 0] and sharper[0, 3] < soft[0, 3]


def test_strength_mixes_latent_and_ternary_weight():
    torch.manual_seed(5)
    w = torch.randn(4, 16)
    spec = TernarySpec(8, "twn", ste="clipped")
    torch.testing.assert_close(spec.effective(w, 0.0), w)
    torch.testing.assert_close(spec.effective(w, 0.25), 0.75 * w + 0.25 * spec.quantize(w))


def test_learned_scale_gets_the_lsq_gradient():
    torch.manual_seed(6)
    lin = nn.Linear(16, 4, bias=False)
    t = TernaryLinear(lin, 8, spec=TernarySpec(8, "absmean"), scale_mode="learned")
    assert t.log_scale.shape == (4, 2, 1) and "scale_alpha" in dict(t.named_buffers())
    w0, alpha = lin.weight.detach().clone(), t.scale_alpha.clone()
    t.effective_weight().sum().backward()
    g = group_view(w0, 8)
    codes = torch.sign(g) * (g.abs() > alpha / 2)
    expected = (alpha * codes - g).sum(-1, keepdim=True) / 8**0.5  # d/d log alpha = alpha (code - w / alpha), LSQ scale
    torch.testing.assert_close(t.log_scale.grad, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(lin.weight.grad, torch.ones_like(lin.weight))
    # a larger factor scales the ternary weight and moves the dead zone with it
    with torch.no_grad():
        t.log_scale.fill_(0.2)
    eff = t.effective_weight().detach()
    assert is_ternary(eff, 8)
    _, a = decode_ternary(eff, 8)
    live = a.flatten() > 0
    torch.testing.assert_close(a.flatten()[live], (alpha * torch.tensor(0.2).exp()).flatten()[live])
    # frozen scales do not follow the latent weights
    f = TernaryLinear(nn.Linear(16, 4, bias=False), 8, spec=TernarySpec(8, "mse"), scale_mode="frozen")
    before = f.scale_alpha.clone()
    with torch.no_grad():
        f.weight.mul_(3.0)
    torch.testing.assert_close(f.scale_alpha, before)
    assert is_ternary(f.effective_weight().detach(), 8)


# ─────────────────────────────────────────────────────────────── the plugin with each component


def test_plugin_rejects_unknown_components():
    for kw in ({"scale": "nope"}, {"quantizer": "nope"}, {"ste": "nope"}, {"scale_mode": "nope"}):
        with pytest.raises(ValueError, match="must be one of"):
            TernaryQAT(**kw)


@pytest.mark.parametrize(
    "args",
    [
        {"scale": "mse"},
        {"scale": "twn", "ste": "clipped"},
        {"scale": "absmax", "scale_mode": "frozen", "ste": "tanh", "ste_sharpness": {"start_value": 2.0, "end_value": 8.0}},
        {"scale": "mse", "scale_mode": "learned", "scale_lr": 1e-3, "quantizer": "stochastic"},
    ],
)
def test_plugin_variants_export_ternary_weights(args, tmp_path):
    from transformers import LlamaForCausalLM

    plugin = TernaryQAT(group_size=16, grid_weight=0.1, **args)
    model = plugin.transform_model(H.tiny_llama(64, hidden=32), _ctx())
    plugin.on_train_start(model, _ctx())
    ids = torch.randint(0, 64, (1, 9))
    model(input_ids=ids).logits.sum().backward()
    plugin.on_before_optimizer_step(model, None, _ctx(0.5))
    plugin.on_after_optimizer_step(model, None, _ctx(0.5))
    metrics = plugin.metrics(model, _ctx(0.0))
    assert 0 < metrics["zero_share"] < 1
    groups = plugin.param_groups(model, _ctx())
    assert len(groups) == (2 if args.get("scale_mode") == "learned" else 1)
    if "ste_sharpness" in args:
        plugin.on_forward_start(model, {}, _ctx(1.0))
        assert float(next(iter(plugin.modules.values())).sharpness) == 8.0
    model.eval()
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    assert not [k for k in state if k.endswith(("scale_alpha", "scale_delta", "log_scale"))]
    model.save_pretrained(tmp_path, state_dict=state)
    plugin.on_export_saved(tmp_path, _ctx())
    layout = json.loads((tmp_path / "ternary_config.json").read_text())
    assert layout["quantizer"]["scale"] == args["scale"] and len(layout["tensors"]) == len(plugin.modules)
    stock = LlamaForCausalLM.from_pretrained(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=ids).logits, model(input_ids=ids).logits, rtol=1e-4, atol=1e-4)
    report = check_ternary(tmp_path)
    assert report["ok"] and report["checked"] == len(plugin.modules) and 1.58 < report["bits_per_weight"] < 2.7


def test_variant_trains_through_the_trainer_and_resumes(tmp_path):
    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=4)
    c.train.save_steps = 2
    c.distill.enabled = True
    c.distill.divergence = "jsd"
    c.distill.teachers = {"fp": {"model": str(tmp_path / "student"), "dtype": "float32"}}
    c.plugins.custom = [
        {
            "class": "palingenesis.qat:TernaryQAT",
            "args": {
                "group_size": 16,
                "scale": "mse",
                "scale_mode": "learned",
                "scale_lr": 1e-3,
                "ste": "tanh",
                "ste_sharpness": {"start_value": 2.0, "end_value": 8.0},
                "latent_lr": 2e-3,
                "metrics_every": 1,
            },
        }
    ]
    result = H.run_cpu(c)
    assert "plugins/ternary_qat/codes_changed" in result.metrics
    final = Path(c.train.output_dir) / "final"
    assert check_ternary(final)["ok"]
    c.train.resume_from = str(Path(c.train.output_dir) / "step-2")
    assert H.run_cpu(c).global_step == 4


# ─────────────────────────────────────────────────────────────── merging


def _ternary(seed, rows=6, cols=16, group=8, scale=1.0):
    torch.manual_seed(seed)
    return TernarySpec(group, "mse").quantize(scale * torch.randn(rows, cols))


def test_a_plain_average_of_ternary_tensors_is_not_ternary():
    a, b = _ternary(0), _ternary(1)
    assert is_ternary(a, 8) and is_ternary(b, 8)
    assert not is_ternary(0.5 * (a + b), 8)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [2, 3, 5])
def test_every_merge_returns_ternary_weights(method, k):
    spec = TernarySpec(8, "mse")
    sources = [_ternary(s, scale=1.0 + 0.1 * s) for s in range(k)]
    stats = {}
    merged = merge_tensor(method, sources, [1.0] * k, spec, base=_ternary(99), stats=stats)
    assert merged.shape == sources[0].shape and is_ternary(merged, 8)
    codes, _ = decode_ternary(merged, 8)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    if method != "requantize":
        assert stats["total"] == sources[0].numel()
    # latent (non-ternary) sources are quantised first / re-ternarised: still ternary
    torch.manual_seed(7)
    latents = [torch.randn(6, 16) for _ in range(k)]
    assert is_ternary(merge_tensor(method, latents, [1.0] * k, spec, base=_ternary(99)), 8)


def test_vote_resolves_sign_conflicts_by_majority():
    spec = TernarySpec(-1)
    a = torch.tensor([[1.0, 1, -1, 0, 1, 0]])
    b = torch.tensor([[1.0, -1, -1, 0, 0, 1]])
    c = torch.tensor([[1.0, -1, 1, 0, 0, -1]])
    merged = merge_tensor("vote", [a, b, c], [1, 1, 1], spec, refit=False)
    torch.testing.assert_close(torch.sign(merged), torch.tensor([[1.0, -1, -1, 0, 0, 0]]))
    # two models: a +1 against a -1 cancels; tie="sign" keeps what only one model set
    two = merge_tensor("vote", [a, b], [1, 1], spec, refit=False)
    torch.testing.assert_close(torch.sign(two), torch.tensor([[1.0, 0, -1, 0, 0, 0]]))
    low = merge_tensor("vote", [a, b], [1, 1], spec, tie="sign", refit=False)
    torch.testing.assert_close(torch.sign(low), torch.tensor([[1.0, 0, -1, 0, 1, 1]]))
    # weights: the heavier model wins a conflict
    heavy = merge_tensor("vote", [a, b], [3, 1], spec, refit=False)
    torch.testing.assert_close(torch.sign(heavy), torch.tensor([[1.0, 1, -1, 0, 1, 0]]))


def test_ties_elects_signs_by_mass_and_drops_the_minority():
    spec = TernarySpec(-1, "mse")
    a = 2.0 * torch.tensor([[1.0, 1, 0, -1]])
    b = 1.0 * torch.tensor([[1.0, -1, 1, -1]])
    stats = {}
    merged = merge_tensor("ties", [a, b], [1, 1], spec, stats=stats)
    assert is_ternary(merged, -1) and stats["conflict"] == 1
    # element 1: +2 against -1, the larger mass (+) wins; the mean keeps only the agreeing entry
    torch.testing.assert_close(torch.sign(merged), torch.tensor([[1.0, 1, 1, -1]]))
    # with a base, the election runs on the changes and the result is base + lam * change, re-ternarised
    base = torch.tensor([[1.0, 1, 1, 1]])
    same = merge_tensor("ties", [base.clone(), base.clone()], [1, 1], spec, base=base)
    torch.testing.assert_close(same, base)


def test_consensus_keeps_a_change_made_by_one_expert():
    spec = TernarySpec(-1)
    base = torch.tensor([[1.0, 1, 0, -1, 0, 1]])
    a = torch.tensor([[1.0, 0, 0, -1, 1, -1]])  # changed elements 1, 4, 5
    b = torch.tensor([[1.0, 1, -1, -1, -1, 1]])  # changed elements 2, 4
    c = torch.tensor([[1.0, 1, 0, -1, 0, 1]])  # changed nothing
    stats = {}
    merged = merge_tensor("consensus", [a, b, c], [1, 1, 1], spec, base=base, refit=False, stats=stats)
    # 1 and 2 and 5: one expert's change kept whole; 4: +1 against -1 is a tie, the base code stays
    torch.testing.assert_close(torch.sign(merged), torch.tensor([[1.0, 0, -1, -1, 0, -1]]))
    assert stats["changed"] == 3
    # vote on the same inputs loses the single-expert changes
    voted = merge_tensor("vote", [a, b, c], [1, 1, 1], spec, refit=False)
    torch.testing.assert_close(torch.sign(voted), torch.tensor([[1.0, 1, 0, -1, 0, 1]]))
    strict = merge_tensor("consensus", [a, b, c], [1, 1, 1], spec, base=base, refit=False, min_agree=0.5)
    torch.testing.assert_close(torch.sign(strict), torch.sign(base))
    with pytest.raises(MergeError, match="base"):
        merge_tensor("consensus", [a, b], [1, 1], spec)


def test_refit_scale_is_the_least_squares_alpha():
    spec = TernarySpec(-1)
    a = 1.0 * torch.tensor([[1.0, 1, 0, -1]])
    b = 3.0 * torch.tensor([[1.0, 1, 1, -1]])
    merged = merge_tensor("vote", [a, b], [1, 1], spec, tie="sign")
    mean = 0.5 * (a + b)
    codes = torch.sign(merged)
    assert float(merged.abs().max()) == pytest.approx(float((mean * codes).sum() / codes.abs().sum()))
    plain = merge_tensor("vote", [a, b], [1, 1], spec, tie="sign", refit=False)
    assert float(plain.abs().max()) == pytest.approx(2.0)


def _export(tmp_path, name, seed, latent=False):
    """A folded (or latent) export of a ternary tiny model, with its ternary_config.json."""
    plugin = TernaryQAT(group_size=16, scale="mse", export="latent" if latent else "folded")
    model = plugin.transform_model(H.tiny_llama(64, hidden=32, seed=seed), _ctx())
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    model.save_pretrained(tmp_path / name, state_dict=state)
    plugin.on_export_saved(tmp_path / name, _ctx())
    return tmp_path / name


@pytest.mark.parametrize("method", METHODS)
def test_merged_checkpoints_are_ternary_and_load(method, tmp_path):
    from transformers import LlamaForCausalLM

    models = [_export(tmp_path, f"m{i}", seed=i) for i in range(3)]
    base = _export(tmp_path, "base", seed=9)
    report = merge_ternary(models, tmp_path / "out", method=method, base=base if method == "consensus" else None)
    config = json.loads((tmp_path / "out" / "merge_config.json").read_text())
    assert config["method"] == f"ternary_{method}" and config["ternary_verified"] is True
    check = check_ternary(report.out)
    assert check["ok"] and check["checked"] == config["ternary_tensors"] > 0
    stock = LlamaForCausalLM.from_pretrained(tmp_path / "out")
    w = stock.model.layers[0].mlp.up_proj.weight
    assert is_ternary(w, 16)
    # tensors outside the layout are averaged
    norms = [LlamaForCausalLM.from_pretrained(m).model.embed_tokens.weight for m in models]
    torch.testing.assert_close(stock.model.embed_tokens.weight, sum(norms) / 3, rtol=1e-5, atol=1e-6)


def test_latent_exports_merge_by_requantising_their_mean(tmp_path):
    latents = [_export(tmp_path, f"l{i}", seed=i, latent=True) for i in range(2)]
    assert not check_ternary(latents[0])["ok"]  # a latent export is not ternary
    merge_ternary(latents, tmp_path / "out", method="requantize")
    from safetensors.torch import load_file

    key = "model.layers.0.mlp.up_proj.weight"
    a, b = (load_file(str(m / "model.safetensors"))[key] for m in latents)
    merged = load_file(str(tmp_path / "out" / "model.safetensors"))[key]
    torch.testing.assert_close(merged, TernarySpec(16, "mse").quantize(0.5 * (a + b)))
    assert check_ternary(tmp_path / "out")["ok"]


def test_merge_cli_and_errors(tmp_path, capsys):
    models = [str(_export(tmp_path, f"m{i}", seed=i)) for i in range(2)]
    assert merge_main(["ternary", *models, "--method", "vote", "--tie", "sign", "-o", str(tmp_path / "v")]) == 0
    assert merge_main(["ternary", "--check", str(tmp_path / "v")]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["ok"] is True
    weighted = [f"{models[0]}:0.7", f"{models[1]}:0.3"]
    assert merge_main(["ternary", *weighted, "--method", "ties", "-o", str(tmp_path / "t")]) == 0
    assert merge_main(["ternary", *models, "--method", "consensus", "-o", str(tmp_path / "c")]) == 1  # no base
    assert merge_main(["ternary", models[0], "-o", str(tmp_path / "one")]) == 1
    # a linear average of the same models is not ternary, and the check says so
    from palingenesis.merge import soup

    soup(models, tmp_path / "soup")
    (tmp_path / "soup" / "ternary_config.json").write_text((Path(models[0]) / "ternary_config.json").read_text())
    assert merge_main(["ternary", "--check", str(tmp_path / "soup")]) == 1
    # without a ternary_config.json the group size must be given
    (Path(models[0]) / "ternary_config.json").unlink()
    with pytest.raises(MergeError, match="group_size"):
        merge_ternary(models, tmp_path / "x")
    merge_ternary(models, tmp_path / "x", group_size=16)
    assert check_ternary(tmp_path / "x")["ok"]
