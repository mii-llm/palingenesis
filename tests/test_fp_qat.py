"""FP4 / FP8 quantisation-aware training (palingenesis.qat.fp, .formats, .compressed).

Number formats against hand-computed values (E2M1 grid and ties, E4M3, E8M0 scales, block sizes, NaN / inf,
all-zero blocks, saturation), straight-through and LSQ gradients, the plugin on / off, fused layers' shared
scales, MSE clipping, frozen scales, the compressed-tensors export (packing, round trip, layout matching, config),
and the plugin inside pgs train (distillation, resume), pgs distill and pgs rl, plus `pgs quantize`."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

from palingenesis.hooks import PluginContext, PluginError, PolicyPlugins, TrainerPlugin  # noqa: E402
from palingenesis.qat import FloatQAT, FloatQuantLinear  # noqa: E402
from palingenesis.qat.compressed import pack, quantization_config, unpack, write_checkpoint  # noqa: E402
from palingenesis.qat.formats import (  # noqa: E402
    E8M0_NAN,
    FORMATS,
    e8m0_decode,
    e8m0_encode,
    e8m0_exponent,
    e8m0_to_float,
    lsq_fake_quantize,
    pack_e2m1,
    quantize,
    quantize_activation,
    round_e2m1,
    round_e4m3,
    unpack_e2m1,
)

NAN = float("nan")


def _ctx(**kw):
    return PluginContext(config=None, **kw)


def _ctx_with(config):
    return PluginContext(config=config)


def _t(values):
    return torch.tensor(values, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────── element formats


def test_e2m1_rounds_to_nearest_even_and_saturates():
    x = _t([0.0, 0.2, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 6.0, 6.9, 100.0, -0.3, -2.6, -7.0])
    want = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, -0.5, -3.0, -6.0]
    assert round_e2m1(x).tolist() == want
    assert math.isnan(round_e2m1(_t([NAN])).item())
    assert torch.signbit(round_e2m1(_t([-0.1])))  # -0 keeps its sign (packs as 0x8)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    assert torch.equal(round_e2m1(grid), grid) and torch.equal(round_e2m1(-grid), -grid)


def test_e4m3_rounding_subnormals_saturation_nan():
    x = _t([1.0625, 1.1875, 2.0**-9, 2.0**-10, 1.5 * 2.0**-10, 3 * 2.0**-10, 448.0, 464.0, 1e6, -1e6, 240.0])
    want = [1.0, 1.25, 2.0**-9, 0.0, 2.0**-9, 2.0**-8, 448.0, 448.0, 448.0, -448.0, 240.0]
    assert round_e4m3(x).tolist() == want
    assert math.isnan(round_e4m3(_t([NAN])).item())


def test_e8m0_exponents_per_rounding_mode():
    amax = _t([1.0, 1.74, 1.75, 6.0, 7.0, 0.0, 1e-40, 3e38])
    assert e8m0_exponent(amax, "floor").tolist() == [-2, -2, -2, 0, 0, -127, -127, 125]
    assert e8m0_exponent(amax, "even").tolist() == [-2, -2, -1, 0, 1, -127, -127, 126]
    assert e8m0_exponent(amax, "ceil").tolist() == [-2, -1, -1, 0, 1, -127, -126, 126]
    for mode in ("floor", "even", "ceil"):
        assert e8m0_exponent(_t([float("inf"), NAN]), mode).tolist() == [E8M0_NAN - 127] * 2
    with pytest.raises(ValueError, match="mx_rounding"):
        e8m0_exponent(amax, "nearest")


def test_e8m0_scales_are_exact_powers_of_two_and_encode_to_bytes():
    e = torch.tensor([-127, -126, -1, 0, 1, 127, 128])
    s = e8m0_to_float(e)
    assert s[:-1].tolist() == [2.0**-127, 2.0**-126, 0.5, 1.0, 2.0, 2.0**127] and math.isnan(s[-1])
    assert e8m0_encode(s).tolist() == [0, 1, 126, 127, 128, 254, 255]
    decoded = e8m0_decode(e8m0_encode(s))
    assert decoded[:-1].tolist() == s[:-1].tolist() and math.isnan(decoded[-1])


# ─────────────────────────────────────────────────────────────── block formats


def test_nvfp4_scales_by_hand():
    # block 0 holds the tensor's amax: its scale is e4m3(amax / 6 * 448 * 6 / amax) = 448, the effective scale amax/6
    b0 = _t([12.0, 6.0, 3.0, -2.0, 1.0, 0.5, 0.0, -12.0] + [0.0] * 8)
    # block 1: amax 1 -> raw scale 1/6 * 2688/12 = 37.33 -> e4m3 36 (between 32 and 40 the step is 4, 37.33 -> 36)
    b1 = _t([1.0, -0.5, 0.25] + [0.0] * 13)
    # block 2: amax 1e-4 -> raw 0.0037 < 2^-6: clamped to 2^-6 (vLLM's Marlin zeroes smaller ones)
    b2 = _t([1e-4] + [0.0] * 15)
    w = torch.stack([torch.cat([b0, b1, b2])])
    q = quantize(w, FORMATS["nvfp4"])
    assert q.global_scale.item() == pytest.approx(2688 / 12)
    assert q.scale.tolist() == [[448.0, 36.0, 2.0**-6]]
    eff0 = 448.0 / q.global_scale.item()
    assert q.values[0, :8].tolist() == [6.0, 3.0, 1.5, -1.0, 0.5, 0.0, 0.0, -6.0]
    torch.testing.assert_close(q.dequantize()[0, :8], q.values[0, :8] * eff0)
    # block 1: 1 / (36 / 224) = 6.22 -> 6, 0.5 -> 3.11 -> 3, 0.25 -> 1.56 -> 1.5
    assert q.values[0, 16:19].tolist() == [6.0, -3.0, 1.5]
    assert q.values[0, 32].item() == 1.5  # block 2: 1e-4 / (2^-6 / 224) = 1.43 -> 1.5


def test_nvfp4_tiny_block_and_fused_global_scale():
    w = torch.zeros(1, 32)
    w[0, 0], w[0, 16] = 2.0, 1e-3
    q = quantize(w, FORMATS["nvfp4"], global_amax=torch.tensor(4.0))  # a fused partner holds amax 4
    assert q.global_scale.item() == pytest.approx(2688 / 4)
    assert q.scale[0, 0].item() == round_e4m3(_t([2.0 / 6 * 672])).item()


def test_mxfp4_floor_saturates_where_even_and_ceil_do_not():
    w = torch.zeros(1, 32)
    w[0, :3] = _t([7.0, 1.0, -0.3])
    floor = quantize(w, FORMATS["mxfp4"], mx_rounding="floor")
    assert floor.scale.item() == 1.0 and floor.values[0, :3].tolist() == [6.0, 1.0, -0.5]
    assert floor.saturated.tolist() == [[False]]  # 7 is the midpoint to the next value: ordinary rounding
    w[0, 0] = 7.5
    assert quantize(w, FORMATS["mxfp4"], mx_rounding="floor").saturated.tolist() == [[True]]
    even = quantize(w, FORMATS["mxfp4"], mx_rounding="even")
    assert even.scale.item() == 2.0 and even.values[0, 0].item() == 4.0  # 7.5 / 2 = 3.75 -> 4
    ceil = quantize(w, FORMATS["mxfp4"], mx_rounding="ceil")
    assert ceil.scale.item() == 2.0 and not ceil.saturated.any()


def test_all_zero_blocks_quantise_to_zero():
    for spec in FORMATS.values():
        w = torch.zeros(128, 128)
        q = quantize(w, spec)
        assert torch.equal(q.dequantize(), w) and torch.isfinite(q.scale).all()


def test_fp8_rowwise_and_block_scales():
    w = torch.zeros(256, 256)
    w[0, 5], w[1, 3], w[200, 200] = 448.0 * 3, -1.0, 0.5
    row = quantize(w, FORMATS["fp8"])
    assert row.scale.shape == (256, 1) and row.scale[0].item() == 3.0 and row.scale[1].item() == pytest.approx(1 / 448)
    assert row.scale[2].item() == 1.0  # an all-zero row
    torch.testing.assert_close(row.dequantize(), w)
    block = quantize(w, FORMATS["fp8_block"])
    assert block.scale.shape == (2, 2) and block.scale[0, 0].item() == 3.0
    assert block.scale[1, 1].item() == pytest.approx(0.5 / 448)
    assert block.scale[0, 1].item() == 1.0
    assert block.dequantize()[1, 3].item() == pytest.approx(round_e4m3(_t([-1.0 / 3])).item() * 3)


def test_shapes_the_formats_cannot_take():
    with pytest.raises(ValueError, match="16-element block"):
        quantize(torch.randn(4, 24), FORMATS["nvfp4"])
    with pytest.raises(ValueError, match="128-row block"):
        quantize(torch.randn(64, 128), FORMATS["fp8_block"])
    assert FORMATS["fp8"].check_shape(3, 7) is None


def test_non_finite_weights_give_nan():
    w = torch.randn(2, 64)
    w[0, 3] = NAN
    mx = quantize(w, FORMATS["mxfp4"]).dequantize()
    assert mx[0, :32].isnan().all() and not mx[0, 32:].isnan().any() and not mx[1].isnan().any()
    assert quantize(w, FORMATS["nvfp4"]).dequantize().isnan().all()  # through the per-tensor scale
    w[0, 3] = float("inf")
    assert quantize(w, FORMATS["mxfp4"]).dequantize()[0, :32].isnan().all()
    assert quantize(w, FORMATS["fp8"]).dequantize()[0, 3].isnan()


def test_quantisation_error_ordering_and_saturation_counts():
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    err = {n: float((quantize(w, s).dequantize() - w).pow(2).sum() / w.pow(2).sum()) for n, s in FORMATS.items()}
    assert err["fp8"] < err["fp8_block"] < err["nvfp4"] < err["mxfp4"] < 0.05
    assert not quantize(w, FORMATS["fp8"]).saturated.any()


def test_activation_quantisation_per_token():
    x = torch.randn(2, 3, 64)
    x[0, 0] *= 100  # one loud token does not coarsen the others' grid
    y = quantize_activation(x, FORMATS["nvfp4"])
    rel = (y - x).pow(2).sum(-1) / x.pow(2).sum(-1)
    assert rel.max() < 0.05
    fp8 = quantize_activation(x, FORMATS["fp8"])
    assert ((fp8 - x).abs() <= x.abs().amax(-1, keepdim=True) / 448 * 16).all()


def test_e2m1_packing_layout():
    values = _t([[0.5, -6.0, -0.0, 3.0]])
    packed = pack_e2m1(values)
    assert packed.dtype == torch.uint8 and packed.tolist() == [[0xF1, 0x58]]  # even column in the low nibble
    assert unpack_e2m1(packed).tolist() == [[0.5, -6.0, -0.0, 3.0]]
    v = round_e2m1(torch.randn(8, 64) * 4)
    assert torch.equal(unpack_e2m1(pack_e2m1(v)), v)
    with pytest.raises(ValueError, match="E2M1 grid"):
        pack_e2m1(_t([[0.7, 1.0]]))


# ─────────────────────────────────────────────────────────────── gradients


def test_straight_through_estimator_and_strength():
    torch.manual_seed(0)
    lin = nn.Linear(32, 8, bias=False)
    m = FloatQuantLinear(lin, FORMATS["nvfp4"])
    assert m.weight is lin.weight
    x = torch.randn(3, 32)
    m(x).sum().backward()
    torch.testing.assert_close(lin.weight.grad, torch.ones(3, 8).T @ x)
    with torch.no_grad():
        w_q = quantize(lin.weight, FORMATS["nvfp4"]).dequantize()
        torch.testing.assert_close(m(x), x @ w_q.T)
        m.strength.fill_(0.0)
        torch.testing.assert_close(m(x), lin(x))  # strength 0: the plain layer
        m.strength.fill_(0.5)
        torch.testing.assert_close(m(x), x @ (0.5 * lin.weight + 0.5 * w_q).T)


def test_lsq_gradients():
    torch.manual_seed(0)
    spec = FORMATS["nvfp4"]
    w = torch.randn(4, 32, requires_grad=True)
    clip = torch.full((4, 1), 0.6, requires_grad=True)
    out = lsq_fake_quantize(w, spec, clip)
    torch.testing.assert_close(out.detach(), quantize(w.detach(), spec, clip=clip.detach()).dequantize())
    out.sum().backward()
    q = quantize(w.detach(), spec, clip=clip.detach())
    eff = (q.scale / q.global_scale).repeat_interleave(16, dim=1)
    clipped = (w.detach() / eff).abs() > 6
    assert clipped.any() and torch.equal(w.grad[clipped], torch.zeros_like(w.grad[clipped]))
    assert torch.equal(w.grad[~clipped], torch.ones_like(w.grad[~clipped]))
    assert torch.isfinite(clip.grad).all() and clip.grad.abs().sum() > 0


def test_activation_ste():
    lin = nn.Linear(32, 4)
    m = FloatQuantLinear(lin, FORMATS["fp8"], activation=FORMATS["fp8"])
    x = torch.randn(2, 32, requires_grad=True)
    m(x).sum().backward()
    torch.testing.assert_close(x.grad, m.effective_weight().detach().sum(0).expand(2, 32))


# ─────────────────────────────────────────────────────────────── the plugin


def test_validation_errors():
    for kwargs, match in (
        ({"format": "fp4"}, "format must be one of"),
        ({"activations": "int8"}, "activations must be one of"),
        ({"scales": "ema"}, "scales must be one of"),
        ({"clip_min": 0.0}, "clip_min"),
        ({"clip_steps": 0}, "clip_steps"),
        ({"clip_lr": -1}, "clip_lr"),
        ({"freeze_scales_at": 1.5}, "freeze_scales_at"),
        ({"include": "layers"}, "list of regular expressions"),
        ({"exclude": ["("]}, "invalid regular expression"),
        ({"mx_rounding": "up"}, "mx_rounding"),
        ({"export": "packed"}, "export must be one of"),
        ({"quantized_export": "yes"}, "quantized_export"),
        ({"format": "nvfp4", "activations": "fp8", "quantized_export": True}, "no vLLM checkpoint format"),
        ({"strength": "high"}, "strength"),
        ({"fused_groups": [["q_proj", 3]]}, "fused_groups"),
        ({"quantize_head": "yes"}, "quantize_head"),
        ({"metrics_every": 0}, "metrics_every"),
    ):
        with pytest.raises(ValueError, match=match):
            FloatQAT(**kwargs)
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="float8_training"):
        FloatQAT().setup(_ctx_with(SimpleNamespace(memory=SimpleNamespace(float8_training=True))))
    assert FloatQAT(format="nvfp4", activations="fp8").quantized_export is False  # auto: fake quantisation only
    assert FloatQAT(format="fp8_block", activations="fp8").activation_spec.name == "fp8_group"


def test_substitution_selection_and_identity():
    model = H.tiny_llama(64, hidden=64)
    reference = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="mxfp4", strength=0.0)
    out = plugin.transform_model(model, _ctx())
    assert out is model and len(plugin.modules) == 14  # 7 linear layers x 2 decoder layers, not the head
    assert type(model.lm_head) is nn.Linear and isinstance(model.model.layers[0].mlp.up_proj, FloatQuantLinear)
    assert [n for n, _ in model.named_parameters()] == [n for n, _ in reference.named_parameters()]
    plugin.on_train_start(model, _ctx())
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, reference(input_ids=ids).logits)  # strength 0
    with pytest.raises(ValueError, match="no layer to quantise"):
        FloatQAT(include=["nothing"]).transform_model(H.tiny_llama(64, hidden=64), _ctx())


def test_fused_partners_share_scale_and_skip_together():
    model = H.tiny_llama(64, hidden=128)
    plugin = FloatQAT(format="fp8_block")
    plugin.transform_model(model, _ctx())
    attn = "model.layers.0.self_attn"
    # k/v have 64 rows (not a 128-row block): q stays with them, o_proj is quantised
    assert f"{attn}.k_proj" in plugin.skipped and "fused partner" in plugin.skipped[f"{attn}.q_proj"]
    assert f"{attn}.o_proj" in plugin.modules

    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4")
    plugin.transform_model(model, _ctx())
    with torch.no_grad():
        model.model.layers[0].self_attn.k_proj.weight.mul_(10)  # k holds the group's largest weight
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    scales = {
        k.split(".")[-2]: v[1]["weight_global_scale"].item() for k, v in plugin._pending.items() if ".0.self_attn" in k
    }
    assert scales["q_proj"] == scales["k_proj"] == scales["v_proj"] != scales["o_proj"]
    assert state.keys() == model.state_dict().keys()


def test_mse_clipping_never_worse_than_max():
    from palingenesis.qat.fp import search_clip

    torch.manual_seed(0)
    w = torch.distributions.Laplace(0.0, 1.0).sample((64, 256))  # heavy tails: clipping pays
    for name in ("nvfp4", "mxfp4", "fp8"):
        spec = FORMATS[name]
        clip = search_clip(w, spec, 0.5, 16)
        assert clip.shape == (64, 1) and ((clip > 0.49) & (clip <= 1.0)).all()
        mse = (quantize(w, spec, clip=clip).dequantize() - w).pow(2).sum(1)
        top = (quantize(w, spec).dequantize() - w).pow(2).sum(1)
        assert (mse <= top).all()  # factor 1 is a candidate: never worse, row by row
        if name != "mxfp4":  # power-of-two scales move in steps of 2: a few % of clipping rarely changes them
            assert mse.sum() < 0.99 * top.sum()


def test_frozen_scales_stay_while_weights_move():
    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4", freeze_scales_at=0.5)
    plugin.transform_model(model, _ctx())
    ctx = _ctx(total_steps=10, step=5)
    plugin.on_train_start(model, ctx)
    plugin.on_forward_start(model, {}, ctx)
    model(input_ids=torch.randint(0, 64, (1, 5)))
    m = model.model.layers[0].mlp.up_proj
    frozen = m.frozen_scale.clone()
    assert plugin.metrics(model, ctx)["frozen"] == 1.0
    with torch.no_grad():
        m.weight.mul_(3)
    q = m.quantize_now(m.weight, m.global_amax())
    assert torch.equal(q.scale, frozen) and q.saturated.any()  # the grid stayed: tripled weights saturate
    state = plugin.state_dict()
    fresh_model = H.tiny_llama(64, hidden=64)
    fresh = FloatQAT(format="nvfp4", freeze_scales_at=0.5)
    fresh.transform_model(fresh_model, _ctx())
    fresh.load_state_dict(state)
    fresh.on_train_start(fresh_model, ctx)
    assert torch.equal(fresh_model.model.layers[0].mlp.up_proj.frozen_scale, frozen) and fresh._frozen


def test_rollout_weights_are_the_quantised_ones():
    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4", scales="learned")
    plugin.transform_model(model, _ctx())
    pushed = dict(plugin.rollout_weights(iter(list(model.named_parameters())), _ctx()))
    assert not any(k.endswith("qat_clip") for k in pushed)
    m = model.model.layers[1].self_attn.v_proj
    torch.testing.assert_close(pushed["model.layers.1.self_attn.v_proj.weight"], m.effective_weight().detach())
    assert torch.equal(pushed["model.norm.weight"], model.model.norm.weight)
    lonely = [(n, p) for n, p in model.named_parameters() if n.endswith("self_attn.q_proj.weight")]
    with pytest.raises(RuntimeError, match="fused partners"):
        list(plugin.rollout_weights(iter(lonely), _ctx()))


@pytest.mark.parametrize("scales", ["max", "learned"])
def test_rollout_weights_follow_the_strength_ramp(scales):
    """While the quantisation is ramped in, the engine samples from the weights the trainer's forward computes
    with (strength * W_q + (1 - strength) * W), not from the fully quantised ones."""
    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4", scales=scales)
    plugin.transform_model(model, _ctx())
    key = "model.layers.1.self_attn.v_proj.weight"
    m = model.model.layers[1].self_attn.v_proj
    for strength in (0.0, 0.3, 1.0):
        for module in plugin.modules.values():
            module.strength.fill_(strength)
        pushed = dict(plugin.rollout_weights(iter(list(model.named_parameters())), _ctx()))
        assert torch.equal(pushed[key], m.effective_weight().detach()), strength
    assert torch.equal(pushed[key], m.quantize_now(m.weight, m.global_amax()).dequantize())  # strength 1: W_q
    for module in plugin.modules.values():
        module.strength.fill_(0.0)
    pushed = dict(plugin.rollout_weights(iter(list(model.named_parameters())), _ctx()))
    assert torch.equal(pushed[key], m.weight)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "fp8", "fp8_block"])
def test_export_equals_the_fake_quantised_forward(tmp_path, fmt):
    from transformers import LlamaForCausalLM

    hidden = 256 if fmt == "fp8_block" else 64
    model = H.tiny_llama(64, hidden=hidden, tie=True)
    plugin = FloatQAT(format=fmt, scales="mse")
    plugin.transform_model(model, _ctx())
    plugin.on_train_start(model, _ctx())
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    model.save_pretrained(tmp_path / "final", state_dict=state)
    stock = LlamaForCausalLM.from_pretrained(tmp_path / "final")
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(stock(input_ids=ids).logits, model(input_ids=ids).logits, rtol=0, atol=0)
    # the packed tensors hold exactly the exported weights
    for key, (deq, packed) in plugin._pending.items():
        assert torch.equal(unpack(packed, plugin.scheme), deq), key
    plugin.on_export_saved(tmp_path / "final", _ctx())
    qdir = tmp_path / f"final-{fmt}"
    config = json.loads((qdir / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors" and "lm_head" in config["ignore"]
    assert "model.embed_tokens" in config["ignore"]
    from safetensors.torch import load_file

    tensors = load_file(qdir / "model.safetensors")
    suffixes = {k.rsplit(".", 1)[-1] for k in tensors if ".mlp.up_proj." in k}
    assert suffixes == (
        {"weight_packed", "weight_scale", "weight_global_scale"}
        if fmt == "nvfp4"
        else {"weight_packed", "weight_scale"}
        if fmt == "mxfp4"
        else {"weight", "weight_scale"}
    )
    report = json.loads((qdir / "quantization_report.json").read_text())
    assert len(report) == len(plugin.modules) and all(0 < r["relative_error"] < 0.05 for r in report.values())
    assert plugin._pending == {}


def test_quantized_export_matches_renamed_layouts_by_value(tmp_path):
    from safetensors.torch import save_file

    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4", activations="nvfp4")
    plugin.transform_model(model, _ctx())
    model.train()
    model(input_ids=torch.randint(0, 64, (2, 5)))  # the activation observers see something
    state = plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    # a multimodal-style checkpoint: model.language_model.* names, plus an untrained copy under mtp.* that
    # shares the module path suffix but not the values
    saved = {k.replace("model.", "model.language_model.", 1): v.contiguous() for k, v in state.items()}
    saved["mtp.layers.0.mlp.up_proj.weight"] = torch.randn_like(state["model.layers.0.mlp.up_proj.weight"])
    src = tmp_path / "src"
    src.mkdir()
    save_file(saved, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps({"model_type": "llama", "torch_dtype": "float32"}))
    write_checkpoint(src, tmp_path / "out", plugin._pending, plugin.scheme)
    from safetensors.torch import load_file

    out = load_file(tmp_path / "out" / "model.safetensors")
    assert "model.language_model.layers.0.mlp.up_proj.weight_packed" in out
    assert "model.language_model.layers.0.mlp.up_proj.input_global_scale" in out
    assert out["mtp.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16  # copied, float32 -> bf16
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["torch_dtype"] == "bfloat16" and "mtp.layers.0.mlp.up_proj" in config["quantization_config"]["ignore"]
    group = config["quantization_config"]["config_groups"]["group_0"]
    assert group["input_activations"]["dynamic"] == "local" and group["weights"]["group_size"] == 16
    # a weight the source does not hold (or holds other values for) is an error, not a silent skip
    key = next(iter(plugin._pending))
    deq, packed = plugin._pending[key]
    with pytest.raises(RuntimeError, match="not found"):
        write_checkpoint(src, tmp_path / "bad", {key: (deq + 1, packed)}, plugin.scheme)


def test_quantization_configs_use_only_compressed_tensors_fields():
    allowed = {
        "num_bits",
        "type",
        "symmetric",
        "group_size",
        "strategy",
        "block_structure",
        "dynamic",
        "actorder",
        "observer",
        "observer_kwargs",
        "scale_dtype",
        "zp_dtype",
    }
    from palingenesis.qat.compressed import SCHEMES

    for scheme in SCHEMES:
        group = quantization_config(scheme, [])["config_groups"]["group_0"]
        assert set(group["weights"]) <= allowed
        assert group["input_activations"] is None or set(group["input_activations"]) <= allowed


def test_pack_shapes():
    w = torch.randn(32, 64)
    q = quantize(w, FORMATS["nvfp4"])
    p = pack(q, torch.tensor(3.0))
    assert p["weight_packed"].shape == (32, 32) and p["weight_scale"].dtype == torch.float8_e4m3fn
    assert p["weight_scale"].shape == (32, 4) and p["weight_global_scale"].shape == (1,)
    assert p["input_global_scale"].item() == pytest.approx(2688 / 3)


# ─────────────────────────────────────────────────────────────── trainers


def test_qat_while_distilling_through_the_trainer(tmp_path):
    from transformers import AutoModelForCausalLM

    tok = H.tokenizer()
    H.save_model(tmp_path / "student", tok, seed=0, hidden=64)
    data = H.write_rows(tmp_path / "rows.jsonl")
    c = H.cpu_config(tmp_path, tmp_path / "student", data, steps=4)
    c.train.save_steps = 2
    c.distill.enabled = True
    c.distill.teachers = {"fp": {"model": str(tmp_path / "student"), "dtype": "float32"}}
    c.plugins.custom = [
        {
            "class": "palingenesis.qat:FloatQAT",
            "args": {
                "format": "nvfp4",
                "scales": "learned",
                "freeze_scales_at": 0.5,
                "strength": {"start": 0.0, "end": 0.25},
                "metrics_every": 1,
                "layer_metrics": True,
            },
        }
    ]
    result = H.run_cpu(c)
    m = result.metrics
    assert m["plugins/float_qat/strength"] == 1.0 and m["plugins/float_qat/frozen"] == 1.0
    assert 0 < m["plugins/float_qat/qerr_mean"] < 0.05 and "plugins/float_qat/clip_mean" in m
    assert "plugins/float_qat/qerr/model.layers.0.mlp.up_proj" in m
    out = Path(c.train.output_dir)
    final = AutoModelForCausalLM.from_pretrained(out / "final")
    w = final.model.layers[1].mlp.down_proj.weight.detach()
    q = quantize(w, FORMATS["nvfp4"], global_amax=w.abs().amax())
    assert len(q.values.unique()) <= 15  # E2M1 codes times block scales: the exported weights sit on the grid
    assert (out / "final-nvfp4" / "config.json").exists() and (out / "step-2" / "plugins.pt").exists()
    c.train.resume_from = str(out / "step-2")
    assert H.run_cpu(c).global_step == 4


def _opd_config(tmp_path):
    from test_opd_trainer import tiny_model

    from palingenesis.opd.config import OPDConfig

    student = tmp_path / "student"
    if not student.exists():
        tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "student", 0)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": f"What is {i} plus {i}?"}]}) + "\n" for i in range(12)
        )
    )
    config = OPDConfig()
    for key, value in {
        "model.student": str(student),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "teachers.fp.model": str(student),
        "sources.s.path": str(prompts),
        "sources.s.max_new_tokens": 6,
        "sources.s.dev_size": 2,
        "rollout.batch_prompts": 2,
        "rollout.group_size": 2,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.save_steps": 1,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.eval_every": 0,
        "plugins.custom": [{"class": "palingenesis.qat:FloatQAT", "args": {"format": "mxfp4", "scales": "learned"}}],
    }.items():
        config.set(key, value, "test")
    return config


def test_qat_in_on_policy_distillation(tmp_path):
    from palingenesis.opd.trainer import OPDTrainer

    trainer = OPDTrainer(_opd_config(tmp_path))
    assert isinstance(trainer.student.model.layers[0].mlp.up_proj, FloatQuantLinear)
    clip_group = trainer.opt.param_groups[-1]
    assert clip_group["lr"] == 1e-3 and clip_group["weight_decay"] == 0.0 and len(clip_group["params"]) == 14
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append(metrics)
    trainer.train()
    assert any("plugins/float_qat/qerr_mean" in m for m in logged)
    run = tmp_path / "run"
    assert (run / "step_1" / "plugins.pt").exists() and (run / "final-mxfp4" / "config.json").exists()
    from safetensors import safe_open

    with safe_open(str(next((run / "step_1").glob("*.safetensors"))), "pt") as f:
        assert not any("qat_clip" in k for k in f.keys())  # resume checkpoints: the architecture's tensors
    config = _opd_config(tmp_path)
    config.train.steps, config.train.resume_from = 3, "auto"
    resumed = OPDTrainer(config)
    assert resumed.start_step == 2
    resumed.train()


def test_policy_trainers_refuse_hooks_they_cannot_run():
    class Lossy(TrainerPlugin):
        def loss(self, inputs, ctx):
            return None

    class Stepper(TrainerPlugin):
        def on_after_optimizer_step(self, model, optimizer, ctx):
            pass

    with pytest.raises(PluginError, match="loss hook"):
        PolicyPlugins([Lossy()], None, "distill")
    with pytest.raises(PluginError, match="optimizer-step hooks"):
        PolicyPlugins([Stepper()], None, "rl", optimizer_hooks=False)
    plugins = PolicyPlugins([FloatQAT(scales="learned")], None, "rl", optimizer_hooks=False)
    model = plugins.transform_model(H.tiny_llama(64, hidden=64))
    with pytest.raises(PluginError, match="scales: mse"):
        plugins.param_groups(model, torch.optim.AdamW(model.parameters()))


def test_rl_config_refuses_plugins_with_lora():
    from palingenesis.rl.config import RLConfig, RLConfigError

    config = RLConfig()
    config.set("model.policy", "x")
    config.set("data.dataset", "x.jsonl")
    config.set("rewards.m.fn", "math")
    config.set("lora.rank", 8)
    config.set("plugins.custom", ["palingenesis.qat:FloatQAT"])
    with pytest.raises(RLConfigError, match="plugins.custom with lora.rank"):
        config.validate()


def test_policy_configs_refuse_a_plugin_entry_that_is_not_a_list():
    """`plugins: {custom: palingenesis.qat:FloatQAT}` (a string) would otherwise be read letter by letter."""
    from palingenesis.opd.config import OPDConfig, OPDConfigError
    from palingenesis.rl.config import RLConfig, RLConfigError

    for config, error in ((OPDConfig(), OPDConfigError), (RLConfig(), RLConfigError)):
        assert config.plugins.custom == []  # the default: no plugin
        config.plugins.custom = "palingenesis.qat:FloatQAT"
        with pytest.raises(error, match="plugins.custom must be a list"):
            config.validate()


@pytest.mark.parametrize("fsdp", [False, True], ids=["single", "fsdp2"])
def test_qat_in_rl(tmp_path, fsdp):
    transformers = pytest.importorskip("transformers")
    from test_rl import tiny_policy

    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.trainer import RLTrainer

    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(12)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def make(steps, resume=""):
        config = RLConfig()
        for key, value in {
            "model.policy": tiny_policy(tmp_path),
            "model.use_liger_kernel": False,
            "model.chat_template_kwargs": {"enable_thinking": False},
            "data.dataset": str(tmp_path / "data.jsonl"),
            "data.eval_size": 2,
            "rollout.backend": "hf",
            "rollout.batch_prompts": 2,
            "rollout.group_size": 3,
            "rollout.max_new_tokens": 6,
            "rewards.m.fn": "math",
            "rewards.m.weight": 0.0,
            "train.output_dir": str(tmp_path / "run"),
            "train.steps": steps,
            "train.learning_rate": 1e-3,
            "train.warmup_steps": 1,
            "train.save_steps": 1,
            "train.resume_from": resume,
            "train.fsdp": fsdp,
            "model.gradient_checkpointing": "full" if fsdp else "none",
            "plugins.custom": [
                {
                    "class": "palingenesis.qat:FloatQAT",
                    "args": {"format": "nvfp4", "scales": "mse", "freeze_scales_at": 0.5},
                }
            ],
        }.items():
            config.set(key, value)
        return RLTrainer(config, rewards=[lambda completion: len(completion) / 40])

    trainer = make(2)
    assert isinstance(trainer.model.model.layers[0].self_attn.q_proj, FloatQuantLinear)
    trainer.train()
    run = tmp_path / "run"
    assert (run / "final-nvfp4" / "config.json").exists()
    state = torch.load(run / "step_2" / "plugins.pt", weights_only=False)["float_qat"]
    assert state["frozen"] and "frozen_scale" in state["modules"]["model.layers.0.mlp.up_proj"]
    resumed = make(3, resume="auto")
    resumed.train()
    del transformers


def test_pgs_quantize_cli(tmp_path):
    from transformers import AutoModelForCausalLM

    from palingenesis.qat.compressed import main

    tok = H.tokenizer()
    H.save_model(tmp_path / "model", tok, seed=0, hidden=64)
    assert (
        main(
            [
                "--model",
                str(tmp_path / "model"),
                "--format",
                "nvfp4",
                "--scales",
                "mse",
                "--device",
                "cpu",
                "--out",
                str(tmp_path / "nvfp4"),
                "--dequantized",
                str(tmp_path / "deq"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "nvfp4" / "config.json").read_text())["quantization_config"]["format"] == (
        "nvfp4-pack-quantized"
    )
    from safetensors.torch import load_file

    deq = AutoModelForCausalLM.from_pretrained(tmp_path / "deq", dtype=torch.bfloat16)
    packed = load_file(tmp_path / "nvfp4" / "model.safetensors")
    layer = "model.layers.0.mlp.up_proj."
    tensors = {k[len(layer) :]: v for k, v in packed.items() if k.startswith(layer)}
    want = unpack(tensors, ("nvfp4", "none")).to(torch.bfloat16)
    assert torch.equal(deq.model.layers[0].mlp.up_proj.weight.detach(), want)  # both exports: the same weights
    assert (tmp_path / "nvfp4" / "tokenizer_config.json").exists()


# ─────────────────────────────────────────────────────────────── what vLLM (0.26) needs from the checkpoint


def test_block_fp8_checkpoints_always_carry_the_dynamic_input_scheme(caplog):
    """vLLM's weight-only FP8 scheme asserts `is_static_input_scheme is False` for block weights, and computes
    that flag as `input_quant and not input_quant.dynamic`: None without input activations. So a block
    checkpoint is written as compressed-tensors' FP8_BLOCK preset whatever the activations trained with."""
    from palingenesis.qat.compressed import SCHEMES, SERVING_NOTES

    for scheme, (_, weights, inputs) in SCHEMES.items():
        if weights["strategy"] == "block":
            assert inputs is not None and inputs["dynamic"] is True, scheme
            assert weights["block_structure"] == [128, 128]
    assert quantization_config(("fp8_block", "none"), []) == quantization_config(("fp8_block", "fp8"), [])
    inputs = quantization_config(("fp8_block", "none"), [])["config_groups"]["group_0"]["input_activations"]
    assert inputs == {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": True,
    }
    # the formats with a weight-only scheme in vLLM keep it
    for scheme in (("nvfp4", "none"), ("mxfp4", "none"), ("fp8", "none")):
        assert quantization_config(scheme, [])["config_groups"]["group_0"]["input_activations"] is None
    # the user is told what a weight-only block run means on GPUs with FP8 matmuls
    assert "8.9" in SERVING_NOTES[("fp8_block", "none")] and ("fp8_block", "fp8") not in SERVING_NOTES
    with caplog.at_level("WARNING", logger="palingenesis.qat.fp"):
        FloatQAT(format="fp8_block").transform_model(H.tiny_llama(64, hidden=256), _ctx())
    assert any("dynamic FP8 input scheme" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger="palingenesis.qat.fp"):
        FloatQAT(format="fp8_block", quantized_export=False).transform_model(H.tiny_llama(64, hidden=256), _ctx())
        FloatQAT(format="fp8").transform_model(H.tiny_llama(64, hidden=64), _ctx())
    assert not any("input scheme" in r.message for r in caplog.records)


def test_other_modalities_encoders_stay_in_full_precision_by_default():
    """A vision / audio encoder's own `layers.N` match the default include; an engine builds those towers from
    plain layers and could not load a packed weight there."""

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleDict({"q_proj": nn.Linear(32, 32)})])

    class Multimodal(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = Tower()
            self.visual = Tower()
            self.audio_tower = Tower()
            self.language_model = Tower()

    model = Multimodal()
    plugin = FloatQAT(format="nvfp4", fused_groups=[])
    plugin.transform_model(model, _ctx())
    assert list(plugin.modules) == ["language_model.layers.0.q_proj"]
    assert type(model.vision_tower.layers[0]["q_proj"]) is nn.Linear
    # an explicit exclude list replaces the default (documented): the towers are then the caller's choice
    plugin = FloatQAT(format="nvfp4", fused_groups=[], exclude=[])
    plugin.transform_model(Multimodal(), _ctx())
    assert len(plugin.modules) == 4


def test_learned_clipping_factors_train_at_their_bounds_and_stay_in_range():
    """torch.clamp passes no gradient at or beyond a bound, and the search leaves most rows exactly at 1: the
    factors use a straight-through clamp, and are projected back into [clip_min, 1] before each forward."""
    torch.manual_seed(0)
    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format="nvfp4", scales="learned", clip_min=0.5)
    plugin.transform_model(model, _ctx())
    plugin.on_train_start(model, _ctx())
    m = model.model.layers[0].mlp.up_proj
    at_one = m.qat_clip.detach().flatten() == 1.0
    assert at_one.sum() > 10  # the search's usual result
    ids = torch.randint(0, 64, (2, 7))
    model(input_ids=ids, labels=ids).loss.backward()
    assert (m.qat_clip.grad.flatten()[at_one] != 0).all()  # rows at the bound train
    model.zero_grad()
    with torch.no_grad():
        m.qat_clip[0] = 1.3  # an optimizer step took them out of the range
        m.qat_clip[1] = 0.2
    reference = m.effective_weight().detach().clone()
    plugin.on_forward_start(model, {}, _ctx(step=1, total_steps=10))
    assert m.qat_clip[0].item() == 1.0 and m.qat_clip[1].item() == 0.5
    assert torch.equal(m.effective_weight().detach(), reference)  # the forward already used the clamped values
    model(input_ids=ids, labels=ids).loss.backward()
    assert m.qat_clip.grad[0].item() != 0.0 and m.qat_clip.grad[1].item() != 0.0
    # frozen scales no longer use the factors: nothing to project
    plugin._frozen = True
    with torch.no_grad():
        m.qat_clip[0] = 1.3
    plugin.on_forward_start(model, {}, _ctx(step=2, total_steps=10))
    assert m.qat_clip[0].item() == pytest.approx(1.3)


# ─────────────────────────────────────────────────────────────── vLLM's weight-only (Marlin) loading, re-implemented
#
# From vLLM 0.26: compressed_tensors/schemes/compressed_tensors_w4a4_{nvfp4,mxfp4}.py, quantization/utils/
# marlin_utils_fp4.py (prepare_fp4_layer_for_marlin, nvfp4_marlin_process_scales), csrc/.../marlin/dequant.h,
# gptq_marlin_repack.cu and marlin_template.h. Only the kernel's memory layout (tile permutations) is left out.


def _marlin_e2m1(packed):
    """weight_packed (uint8 [n, k/2]) as the kernel reads it: `.view(int32)`, element j of a row in bits
    4 * (j % 8) of its word (gptq_marlin_repack: `(word >> (pos * 4)) & 15`); each code placed in a bf16 lane as
    `sign | (code & 7) << 6` (dequant<nv_bfloat162, kFE2M1f>), a value 2^126 times too small until the scales
    make up for it."""
    words = packed.contiguous().view(torch.int32)
    codes = torch.stack([(words >> (4 * j)) & 0xF for j in range(8)], dim=-1).reshape(packed.shape[0], -1)
    lanes = (((codes & 0x8) << 12) | ((codes & 0x7) << 6)).to(torch.uint16).view(torch.int16)
    return lanes.view(torch.bfloat16).to(torch.float64) * 2.0**126


def _marlin_nvfp4_weight(shards, params_dtype=torch.bfloat16):
    """The float32 weight vLLM computes with for a layer fused from `shards` (their packed tensors, in order)."""
    q = _marlin_e2m1(torch.cat([s["weight_packed"] for s in shards]))
    # CompressedTensorsW4A4Fp4.process_weights_after_loading: one global scale per fused layer, the reciprocal
    stored = torch.cat([s["weight_global_scale"] for s in shards])
    global_scale = 1.0 / stored.max().to(torch.float32)
    # prepare_fp4_layer_for_marlin + nvfp4_marlin_process_scales
    scales = torch.cat([s["weight_scale"] for s in shards]).to(params_dtype).to(torch.half)
    factor = 1.0
    if params_dtype != torch.half:  # _nvfp4_compute_scale_factor
        top = (scales.float() * 2**7)[scales.float() > 0].max()
        if top < 448 * 2**7:
            factor = float((448 * 2**7 / top).log2().floor().exp2())
    if factor > 1.0:
        scales = (scales.float() * factor).to(torch.half)
    scales = scales * 2**7
    scales[scales < 2] = 0  # block scales below 2^-6 (times the factor) are dropped
    s0e5m3 = ((scales.view(torch.int16) << 1).view(torch.uint8))[:, 1::2]  # the high byte: 5 exponent, 3 mantissa bits
    decoded = (s0e5m3.to(torch.int16) << 7).view(torch.half).to(torch.float64) / 2**7 / factor
    # the kernel: q * scale in bf16 (hmul2), accumulated in float32, the output times the float32 global scale
    product = (q.reshape(q.shape[0], -1, 16) * decoded.unsqueeze(-1)).reshape(q.shape)
    assert torch.equal(product.to(torch.bfloat16).to(torch.float64), product)  # exact in bf16: nothing lost there
    return product.to(torch.float32) * global_scale


def _marlin_mxfp4_weight(shards):
    q = _marlin_e2m1(torch.cat([s["weight_packed"] for s in shards]))
    stored = torch.cat([s["weight_scale"] for s in shards])
    # `.view(torch.float8_e8m0fnu).to(params_dtype)`: the byte b is 2^(b - 127)
    scales = torch.pow(torch.tensor(2.0, dtype=torch.float64), stored.to(torch.float64) - 127).to(torch.bfloat16)
    product = (q.reshape(q.shape[0], -1, 32) * scales.to(torch.float64).unsqueeze(-1)).reshape(q.shape)
    assert torch.equal(product.to(torch.bfloat16).to(torch.float64), product)
    return product.to(torch.float32)


def test_marlin_reference_decodes_the_e2m1_codes():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]])
    assert torch.equal(_marlin_e2m1(pack_e2m1(values)).float(), values)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("scales", ["max", "mse"])
def test_packed_export_is_bit_for_bit_what_vllm_marlin_computes_with(fmt, scales):
    """The dequantized twin (float32 as exported, and so its bf16 cast) against the packed tensors decoded by
    vLLM's convention: nibble order, block-scale encoding and its 2^-6 floor, the reciprocal of the stored global
    scale, one global scale per fused layer (the maximum of the shards' stored divisors)."""
    torch.manual_seed(0)
    model = H.tiny_llama(64, hidden=64)
    with torch.no_grad():  # shards of very different sizes, a near-zero block, an all-zero block
        attn = model.model.layers[0].self_attn
        attn.k_proj.weight.mul_(7.0)
        attn.v_proj.weight.mul_(0.01)
        attn.q_proj.weight[3, :16] *= 1e-4
        attn.q_proj.weight[5, 16:48] = 0.0
    plugin = FloatQAT(format=fmt, scales=scales)
    plugin.transform_model(model, _ctx())
    plugin.on_train_start(model, _ctx())
    plugin.export_state_dict(dict(model.state_dict()), model, _ctx())
    pending = plugin._pending
    fused = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("o_proj",), ("down_proj",)]
    decode = _marlin_nvfp4_weight if fmt == "nvfp4" else _marlin_mxfp4_weight
    checked = 0
    for layer in range(2):
        for group in fused:
            block = "self_attn" if group[0].endswith(("q_proj", "o_proj")) else "mlp"
            keys = [f"model.layers.{layer}.{block}.{name}.weight" for name in group]
            twin = torch.cat([pending[k][0] for k in keys])
            assert twin.dtype == torch.float32
            reference = decode([pending[k][1] for k in keys])
            assert torch.equal(reference, twin), keys
            assert torch.equal(reference.to(torch.bfloat16), twin.to(torch.bfloat16))
            if fmt == "nvfp4":  # what the decoding relies on
                stored = torch.cat([pending[k][1]["weight_global_scale"] for k in keys])
                assert torch.unique(stored).numel() == 1  # vLLM keeps one per fused layer
                assert float(torch.cat([pending[k][1]["weight_scale"].float() for k in keys]).min()) >= 2.0**-6
            checked += len(keys)
    assert checked == len(pending) == 14
    if fmt == "nvfp4":  # the near-zero block survives (Marlin would zero a scale below 2^-6), as in training
        q_key = "model.layers.0.self_attn.q_proj.weight"
        assert pending[q_key][1]["weight_scale"][3, 0].float().item() == 2.0**-6  # at the floor
        assert pending[q_key][0][3, :16].abs().sum() > 0


def test_nvfp4_shards_with_their_own_global_scale_are_not_what_vllm_computes():
    """Why fused layers share the scale: vLLM takes the largest stored divisor for the whole fused layer."""
    torch.manual_seed(0)
    a, b = torch.randn(8, 32), torch.randn(8, 32) * 5
    own = [quantize(w, FORMATS["nvfp4"]) for w in (a, b)]
    reference = _marlin_nvfp4_weight([pack(q) for q in own])
    assert not torch.equal(reference, torch.cat([q.dequantize() for q in own]))
    amax = torch.stack([a.abs().amax(), b.abs().amax()]).amax()
    shared = [quantize(w, FORMATS["nvfp4"], global_amax=amax) for w in (a, b)]
    assert torch.equal(_marlin_nvfp4_weight([pack(q) for q in shared]), torch.cat([q.dequantize() for q in shared]))


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8"])
def test_code_metrics_count_the_quantised_values_training_changed(fmt):
    torch.manual_seed(0)
    model = H.tiny_llama(64, hidden=64)
    plugin = FloatQAT(format=fmt, code_metrics=True, metrics_every=1)
    plugin.transform_model(model, _ctx())
    plugin.on_train_start(model, _ctx())
    ctx = _ctx(step=1, total_steps=10)
    assert plugin.metrics(model, ctx)["codes_changed"] == 0.0
    m = model.model.layers[0].mlp.down_proj
    nonzero = int((m.quantize_now(m.weight.detach(), m.global_amax()).values[0, :8] != 0).sum())
    assert nonzero >= 4
    with torch.no_grad():
        m.weight[0, :8] = -m.weight[0, :8]  # a sign flip changes every code but zero's
    total = sum(x.weight.numel() for x in plugin.modules.values())
    assert plugin.metrics(model, ctx)["codes_changed"] == pytest.approx(nonzero / total)
    # off by default: no copy of the codes, no metric
    plain = FloatQAT(format=fmt, metrics_every=1)
    plain.transform_model(H.tiny_llama(64, hidden=64), _ctx())
    plain.on_train_start(model, _ctx())
    assert plain._initial_codes == {} and "codes_changed" not in plain.metrics(model, ctx)
    with pytest.raises(ValueError, match="code_metrics"):
        FloatQAT(code_metrics="yes")
