"""palingenesis.precision: stochastic rounding, state-dict casts, edit survival, simulated serving formats; and their
wiring into checkpoint.save_hf_model and merge (CPU, small tensors)."""

import json
import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis import merge as M  # noqa: E402
from palingenesis.precision import (  # noqa: E402
    cast_state_dict,
    edit_survival,
    simulate_cast,
    stochastic_round,
    tensor_seed,
)

TARGETS = [torch.bfloat16, torch.float16]


def _values(seed=0):
    """fp32 values in the normal range of both formats, both signs, plus tiny magnitudes (bf16 normal, fp16 subnormal)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(512, generator=g) * torch.logspace(-3, 3, 512)
    tiny = torch.randn(64, generator=g) * 1e-6
    return torch.cat([x, -x[:64], tiny])


def _ulp_bounds(x, dtype):
    """The two representable neighbours (toward zero, away from zero) of each x in dtype, as float64."""
    r = x.to(dtype)
    below = r.double().abs() <= x.double().abs()
    bits = r.view(torch.int16)
    other = torch.where(below, bits + 1, bits - 1).view(dtype).double()
    return torch.where(below, r.double(), other), torch.where(below, other, r.double())


@pytest.mark.parametrize("dtype", TARGETS)
def test_sr_is_unbiased_within_one_ulp_and_exact_on_grid(dtype):
    x = _values()
    lo, hi = _ulp_bounds(x, dtype)
    draws = torch.stack([stochastic_round(x, dtype, seed=s).double() for s in range(3000)])
    assert draws.dtype == torch.float64 and stochastic_round(x, dtype, seed=0).dtype == dtype
    # every draw is one of the two neighbours: error < one ulp
    assert ((draws == lo) | (draws == hi)).all()
    assert ((draws - x.double()).abs() < (hi - lo).abs()).all()
    # unbiased: mean within 5 sigma of x (Bernoulli over the ulp gap)
    ulp = (hi - lo).abs()
    p = (x.double().abs() - lo.abs()) / ulp
    sigma = ulp * (p * (1 - p)).clamp(min=1e-12).sqrt() / 3000**0.5
    assert ((draws.mean(0) - x.double()).abs() <= 5 * sigma + 1e-30).all()
    # on the grid: unchanged; inf / nan kept; the target dtype is returned as is
    on = x.to(dtype).float()
    assert torch.equal(stochastic_round(on, dtype, seed=3), on.to(dtype))
    special = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0, -0.0])
    out = stochastic_round(special, dtype, seed=1).float()
    assert out[0] == float("inf") and out[1] == -float("inf") and out[2].isnan() and (out[3:] == 0).all()
    y = x.to(dtype)
    assert stochastic_round(y, dtype) is y


@pytest.mark.parametrize("dtype", TARGETS)
def test_sr_seeded_deterministic_and_saturates(dtype):
    x = _values(1)
    a, b, c = (stochastic_round(x, dtype, seed=s) for s in (7, 7, 8))
    assert torch.equal(a, b) and not torch.equal(a, c)
    g = torch.Generator().manual_seed(7)
    assert torch.equal(stochastic_round(x, dtype, generator=g), a)  # seed = a fresh CPU generator with that seed
    big = torch.finfo(dtype).max
    top = torch.tensor([3.39e38 if dtype == torch.bfloat16 else 65519.0, 1e30 if dtype == torch.float16 else 3.4e38])
    for s in range(50):
        out = stochastic_round(top, dtype, seed=s).float()
        assert torch.isfinite(out).all() and (out.abs() <= big).all()


def test_sr_from_float64_and_negatives_symmetric():
    x = _values(2).double()
    for dtype in TARGETS:
        m = torch.stack([stochastic_round(x, dtype, seed=s).double() for s in range(2000)]).mean(0)
        lo, hi = _ulp_bounds(x.float(), dtype)
        assert ((m - x).abs() <= 0.1 * (hi - lo).abs() + 1e-30).all()
    pos = stochastic_round(x.float().abs(), torch.bfloat16, seed=5)
    neg = stochastic_round(-x.float().abs(), torch.bfloat16, seed=5)
    assert torch.equal(pos, -neg)  # magnitude rounding: the same noise rounds +x and -x to mirror images


def test_bf16_sources_drop_tiny_edits_nearest_but_not_stochastic():
    """The motivating case: W already on the bf16 grid, delta = 1e-4 of its scale."""
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(256, 256, generator=g) * 0.02).to(torch.bfloat16)
    delta = torch.randn(256, 256, generator=g) * 0.02 * 1e-4
    intended = w.float() + delta
    near = edit_survival(w, intended, intended.to(torch.bfloat16))
    sr = edit_survival(w, intended, stochastic_round(intended, seed=0))
    assert near["projection"] < 0.2 and near["changed_frac"] < 0.05
    assert abs(sr["projection"] - 1) < 0.1 and sr["norm_ratio"] > 1  # whole-ulp jumps: larger norm, right direction
    # dicts: summed over keys; a lost edit stays lost
    d = edit_survival({"a": w, "b": w}, {"a": intended, "b": intended}, {"a": intended.to(torch.bfloat16), "b": w})
    assert d["projection"] < 0.2
    with pytest.raises(ValueError):
        edit_survival({"a": w}, {"b": intended}, {"a": w})


def test_cast_state_dict_order_independent_nonfloat_and_ties():
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(8, 4, generator=g)
    state = {"a.weight": torch.randn(16, generator=g), "emb": emb, "head": emb, "pos": torch.arange(5)}
    rev = dict(reversed(list(state.items())))
    one = cast_state_dict(state, torch.bfloat16, "stochastic", seed=3)
    two = cast_state_dict(rev, torch.bfloat16, "stochastic", seed=3)
    for k in state:
        assert torch.equal(one[k], two[k])
    assert one["pos"] is state["pos"]  # non-float untouched
    assert one["emb"] is one["head"]  # tied weights stay one tensor
    alone = cast_state_dict({"a.weight": state["a.weight"]}, torch.bfloat16, "stochastic", seed=3)
    assert torch.equal(alone["a.weight"], one["a.weight"])  # independent of the other tensors present
    assert torch.equal(one["a.weight"], stochastic_round(state["a.weight"], seed=tensor_seed(3, "a.weight")))
    assert not torch.equal(cast_state_dict(state, torch.bfloat16, "stochastic", seed=4)["emb"], one["emb"])
    near = cast_state_dict(state, torch.float16)
    assert torch.equal(near["emb"], emb.half()) and near["pos"].dtype == torch.int64
    assert cast_state_dict(state, torch.float32, only_float=False)["pos"].dtype == torch.float32
    with pytest.raises(ValueError):
        cast_state_dict(state, torch.bfloat16, "floor")


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_simulate_cast_fp8(fmt):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no float8 in this torch")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(1_000_000, generator=g) * 0.02  # fp32, off any fp8 grid
    q = simulate_cast(w, fmt)
    assert q.dtype == torch.float32
    rel = 2.0 ** (-3 if fmt == "fp8_e4m3" else -2)  # half an ulp, relative, for normal values
    scale = w.abs().max() / torch.finfo(getattr(torch, "float8_e4m3fn" if fmt == "fp8_e4m3" else "float8_e5m2")).max
    assert ((q - w).abs() <= rel * w.abs() + 2 * scale * 2.0**-9 + 1e-12).all()
    # a 1% edit survives nearest fp8 serving in expectation (the weights' own grid offsets dither it); the element that
    # sets amax is not edited, so the per-tensor scale stays put
    delta = torch.randn(w.shape, generator=g) * w.abs() * 1e-2
    delta[w.abs().argmax()] = 0
    s = edit_survival(q, q + delta, simulate_cast(w + delta, fmt))  # served change vs the edit
    assert abs(s["projection"] - 1) < 0.1
    assert torch.equal(simulate_cast(w, "bf16"), w.bfloat16().float())
    with pytest.raises(ValueError):
        simulate_cast(w, "int4")


# ------------------------------------------------------------------ save_hf_model


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )


def _saved(path):
    from safetensors.torch import load_file

    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def test_save_hf_model_bf16_stochastic(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from palingenesis.checkpoint import hf_state_dict, save_hf_model

    model = _tiny_llama()
    save_hf_model(model, None, tmp_path / "sr", dtype=torch.bfloat16, rounding="stochastic", seed=5)
    expect = cast_state_dict(hf_state_dict(model.state_dict()), torch.bfloat16, "stochastic", 5)
    saved = _saved(tmp_path / "sr")
    assert saved and all(t.dtype == torch.bfloat16 for t in saved.values())
    for k, t in saved.items():
        assert torch.equal(t, expect[k]), k
    assert json.loads((tmp_path / "sr" / "config.json").read_text())["dtype"] == "bfloat16"
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "sr", dtype="auto")
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded.lm_head.weight, loaded.model.embed_tokens.weight)  # still tied
    assert next(model.parameters()).dtype == torch.float32  # the live model is untouched
    # default: the model's own dtype, as before
    save_hf_model(model, None, tmp_path / "plain")
    plain = _saved(tmp_path / "plain")
    assert all(t.dtype == torch.float32 for t in plain.values())
    assert json.loads((tmp_path / "plain" / "config.json").read_text())["dtype"] == "float32"
    sd = hf_state_dict(model.state_dict())
    assert all(torch.equal(plain[k], sd[k]) for k in plain)


def test_save_hf_model_source_layout_casts_trained_weights_only(tmp_path):
    pytest.importorskip("transformers")
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from palingenesis.checkpoint import save_hf_model

    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        tie_word_embeddings=True,
    )
    src = tmp_path / "source"
    Qwen3_5ForConditionalGeneration(config).save_pretrained(src)  # float32 source
    model = AutoModelForCausalLM.from_pretrained(src)
    out = tmp_path / "export"
    save_hf_model(model, None, out, source_layout=True, dtype=torch.bfloat16, rounding="stochastic", seed=1)
    dtypes = {}
    for f in out.glob("*.safetensors"):
        with safe_open(str(f), framework="pt") as h:
            dtypes.update({k: h.get_tensor(k).dtype for k in h.keys()})
    vision = {k: v for k, v in dtypes.items() if "visual" in k}
    text = {k: v for k, v in dtypes.items() if "visual" not in k}
    assert vision and all(v == torch.float32 for v in vision.values())  # copied from the source as stored
    assert text and all(v == torch.bfloat16 for v in text.values() if v.is_floating_point)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "qwen3_5" and cfg["dtype"] == "bfloat16"
    assert cfg["text_config"].get("dtype", "bfloat16") == "bfloat16"


# ------------------------------------------------------------------ merge


def _bf16_ckpt(path, w):
    from safetensors.torch import save_file

    path.mkdir(parents=True)
    save_file(w, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "toy"}))
    return path


def test_merge_cli_stochastic_keeps_a_tiny_interpolation(tmp_path, caplog):
    g = torch.Generator().manual_seed(0)
    wa = {"w": (torch.randn(128, 128, generator=g) * 0.02).bfloat16(), "n": torch.arange(4)}
    wb = {"w": (wa["w"].float() + torch.randn(128, 128, generator=g) * 0.02 * 1e-2).bfloat16(), "n": torch.arange(4)}
    a, b = _bf16_ckpt(tmp_path / "a", wa), _bf16_ckpt(tmp_path / "b", wb)
    lam = 0.01  # the merge moves each weight by ~1e-4 of its scale: far below one bf16 ulp
    intended = wa["w"].float() + lam * (wb["w"].float() - wa["w"].float())
    with caplog.at_level(logging.WARNING, logger="palingenesis.merge"):
        assert M.main(["interpolate", str(a), str(b), "--lam", str(lam), "-o", str(tmp_path / "near")]) == 0
    assert "stochastic" in caplog.text  # nearest to bf16 lost the edit: warned
    near = json.loads((tmp_path / "near" / "merge_config.json").read_text())
    assert near["rounding"] == "nearest" and near["edit_survival"]["projection"] < 0.5
    caplog.clear()
    argv = ["interpolate", str(a), str(b), "--lam", str(lam), "--dtype", "bf16", "--rounding", "stochastic"]
    with caplog.at_level(logging.WARNING, logger="palingenesis.merge"):
        assert M.main([*argv, "--seed", "3", "-o", str(tmp_path / "sr")]) == 0
    assert "stochastic" not in caplog.text
    cfg = json.loads((tmp_path / "sr" / "merge_config.json").read_text())
    assert cfg["rounding"] == "stochastic" and cfg["seed"] == 3 and cfg["dtype"] == "bfloat16"
    out = _saved(tmp_path / "sr")
    assert out["w"].dtype == torch.bfloat16 and torch.equal(out["n"], wa["n"])
    assert torch.equal(out["w"], stochastic_round(intended, seed=tensor_seed(3, "w")))
    s = edit_survival(wa["w"], intended, out["w"])
    assert abs(s["projection"] - 1) < 0.15 and abs(cfg["edit_survival"]["projection"] - s["projection"]) < 1e-3
    assert M.main([*argv, "--seed", "3", "-o", str(tmp_path / "sr2")]) == 0
    assert torch.equal(_saved(tmp_path / "sr2")["w"], out["w"])  # deterministic given the seed
