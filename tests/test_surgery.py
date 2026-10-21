"""Margin surgery (palingenesis.surgery) on a tiny CPU model."""

import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from test_cmr import twins  # noqa: E402
from test_opd_trainer import tiny_model  # noqa: E402

from palingenesis import surgery  # noqa: E402
from palingenesis.margins import MarginReader, decision_ids, load_twins  # noqa: E402
from palingenesis.surgery import (  # noqa: E402
    EvalConfig,
    FamilyConfig,
    Group,
    SurgeryConfig,
    SurgeryConfigError,
)


def _load(path, dtype=torch.float32):
    return transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype)


def _state(path):
    m = _load(path, None)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """reference = the tiny model; shifted = reference after 3 SGD steps raising the pos <tool_call> margin (+9 nats)."""
    tmp = tmp_path_factory.mktemp("surgery")
    ref = tiny_model(tmp, "Qwen/Qwen3-0.6B", "ref", 0)
    tok = transformers.AutoTokenizer.from_pretrained(ref)
    train = twins(tmp / "train.jsonl", 24)
    dev = twins(tmp / "dev.jsonl", 8, offset=100)
    m = _load(ref)
    reader = MarginReader(m, tok, device="cpu")
    tid = decision_ids(tok, "<tool_call>")
    sides = [r["pos"] for r in load_twins(train)]
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(3):
        loss = -reader.margins(sides[:8], tid).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    shifted = tmp / "shifted"
    m.save_pretrained(shifted)
    tok.save_pretrained(shifted)
    return {"tmp": tmp, "ref": ref, "shifted": str(shifted), "tok": tok, "train": train, "dev": dev, "tid": tid}


def _groups(setup, n_groups=3, size=4):
    ids, _ = surgery.load_sides(setup["tok"], setup["train"], "pos")
    return surgery.make_groups("pos", ids, setup["tid"], n_groups, size)


def _config(setup, out, **kw):
    base = dict(
        model=setup["shifted"],
        reference=setup["ref"],
        output_dir=str(setup["tmp"] / out),
        overwrite=True,
        families={"pos": FamilyConfig(path=setup["train"], side="pos", groups=4, group_size=4)},
        eval={"dev": EvalConfig(path=setup["dev"], side="both")},
        trust=1.0,
        cap=2.0,
        min_gap=0.05,
        iterations=6,
    )
    base.update(kw)
    return SurgeryConfig(**base)


# ------------------------------------------------------------------ building blocks


def test_group_gradient_and_weighted_step_equal_explicit(setup):
    model = _load(setup["shifted"]).eval()
    tok = setup["tok"]
    params = [p for p in model.parameters()]
    groups = _groups(setup)
    reader = MarginReader(model, tok, device="cpu")
    # the hook-based fp32 accumulation equals plain autograd of the group-mean margin
    logits = reader.logits_at_last(groups[0].ids)
    from palingenesis.margins import log_odds

    ref_grad = torch.autograd.grad(log_odds(logits, groups[0].token_ids).mean(), params, allow_unused=True)
    g0 = surgery.group_gradient(model, tok, groups[0], params, micro_batch=3)
    for a, b in zip(g0, ref_grad):
        b = torch.zeros_like(a) if b is None else b
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
    # one weighted backward of Σ c_i m̄_i == Σ c_i g_i
    coefs = [0.7, -1.3, 0.4]
    gs = [surgery.group_gradient(model, tok, g, params, micro_batch=2) for g in groups]
    step = surgery.weighted_step(model, tok, groups, coefs, params, micro_batch=3)
    for i, s in enumerate(step):
        explicit = sum(c * g[i] for c, g in zip(coefs, gs))
        assert torch.allclose(s, explicit, atol=1e-6, rtol=1e-4)
    assert all(p.grad is None for p in params)  # gradients freed


def test_gram_modes_agree(setup):
    model = _load(setup["shifted"]).eval()
    tok = setup["tok"]
    params = [p for n, p in model.named_parameters() if "layers.1" in n or "norm" in n]
    groups = _groups(setup)
    As = {
        mode: surgery.gram_matrix(model, tok, groups, params, mode=mode, micro_batch=3)
        for mode in ("device", "cpu", "recompute")
    }
    # explicit: bf16-stored gradients, float64 dots
    gs = [
        torch.cat([g.reshape(-1) for g in surgery.group_gradient(model, tok, gr, params)]).to(torch.bfloat16).double()
        for gr in groups
    ]
    explicit = torch.stack([torch.stack([a @ b for b in gs]) for a in gs])
    for mode, A in As.items():
        assert torch.allclose(A, explicit, rtol=1e-5), mode
        assert torch.allclose(A, A.T)
    gap = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
    coefs = [surgery.solve_coefficients(A, gap, 1e-6)[0] for A in As.values()]
    assert all(torch.allclose(c, coefs[0], rtol=1e-4, atol=1e-8) for c in coefs)  # hence the same step


def test_solve_coefficients_truncates_small_directions():
    V, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)))
    A = V @ torch.diag(torch.tensor([1.0, 0.3, 1e-4], dtype=torch.float64)) @ V.T
    c, info = surgery.solve_coefficients(A, V[:, 2], cutoff=1e-2)
    assert info["kept"] == 2 and c.abs().max() < 1e-10  # a gap along a dropped direction gets no step
    assert info["spectrum"][0] == pytest.approx(1.0) and info["spectrum"][2] == pytest.approx(1e-4)
    c, _ = surgery.solve_coefficients(A, V[:, 1], cutoff=1e-2)
    assert torch.allclose(c, V[:, 1] / 0.3)
    c, info = surgery.solve_coefficients(A, V[:, 2], cutoff=1e-6)
    assert info["kept"] == 3 and torch.allclose(c, V[:, 2] / 1e-4)


def test_hinge_rows():
    def g(constraint, target):
        return Group("f", 0, [[1], [2]], [0], constraint, target, torch.tensor([target, target], dtype=torch.float64))

    m = torch.tensor([0.0, 0.0], dtype=torch.float64)
    groups = [
        g("equal", 1.0),
        g("hinge_below", 1.0),
        g("hinge_below", -1.0),
        g("hinge_above", 1.0),
        g("hinge_above", -1.0),
    ]
    st = surgery.constraint_rows(groups, [m] * 5)
    # below its ceiling (target 1 > margin 0): free; above its ceiling (target -1): active; hinge_above mirrors it
    assert st["equality"] == [0] and st["active"] == [2, 3] and st["rows"] == [0, 2, 3]
    assert st["gaps"] == [1.0, 1.0, -1.0, 1.0, -1.0]


def test_hinge_only_inactive_does_nothing(setup):
    cfg = _config(
        setup,
        "hinge",
        reference=None,
        trust_ref="absolute",
        families={
            "neg": FamilyConfig(
                path=setup["train"], side="neg", target=100.0, constraint="hinge_below", groups=2, group_size=4
            )
        },
    )
    rep = surgery.run_surgery(cfg)
    assert rep["stop_reason"] == "nothing_to_do" and rep["edit"]["norm"] == 0.0
    assert rep["iterations"][0]["active_hinges"] == []
    cfg.families["neg"].target = -100.0  # now above its ceiling: active, pushed down
    rep = surgery.run_surgery(cfg)
    assert rep["iterations"][0]["active_hinges"] == ["neg/0", "neg/1"] and rep["edit"]["norm"] > 0


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def restored(setup):
    cfg = _config(
        setup, "restore", params={"include": [".*"], "exclude": [r"layers\.0\."]}, doses=[1.0, 2.0], gram="recompute"
    )
    return cfg, surgery.run_surgery(cfg)


def test_restoration_closes_the_gap_and_generalises(setup, restored):
    cfg, rep = restored
    first, last = rep["iterations"][0], rep["iterations"][-1]
    assert first["gap_mean"] < -5  # the shift: margins ~9 nats above the reference
    assert rep["stop_reason"] == "converged" and abs(last["gap_mean"]) < last["threshold"]
    dev = rep["eval"]["dev"]
    ref, start, end = dev["reference"]["pos"]["mean"], dev["start"]["pos"]["mean"], dev["end"]["pos"]["mean"]
    assert abs(end - ref) < 0.2 * abs(start - ref)  # held-out twins move most of the way back
    assert rep["edit"]["rel_reference"] < 1.0 and rep["edited"]["params"] < rep["edited"]["total_params"]
    assert rep["memory"]["mode"] == "recompute"
    saved = json.loads((setup["tmp"] / "restore" / "surgery_report.json").read_text())
    assert saved["stop_reason"] == rep["stop_reason"]


def test_excluded_parameters_unchanged_bit_for_bit(setup, restored):
    cfg, rep = restored
    before = _state(setup["shifted"])
    after = _state(rep["doses"][0]["path"])
    frozen = [k for k in before if "layers.0." in k]
    assert frozen and all(torch.equal(before[k], after[k]) for k in frozen)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and not any("layers.0." in k for k in changed)


def test_select_parameters_checks_tied_aliases(setup):
    model = _load(setup["ref"])
    names = [n for n, _ in surgery.select_parameters(model, ["lm_head"], [])]
    assert names == ["model.embed_tokens.weight"]  # the tied head is the embedding tensor
    assert not any("embed" in n for n, _ in surgery.select_parameters(model, [".*"], ["lm_head"]))


def test_doses_scale_the_edit(setup, restored):
    _, rep = restored
    t0, d1, d2 = _state(setup["shifted"]), _state(rep["doses"][0]["path"]), _state(rep["doses"][1]["path"])
    for k in t0:
        assert torch.allclose(d2[k] - t0[k], 2 * (d1[k] - t0[k]), atol=1e-6), k
    assert rep["doses"][1]["edit_rel"] == pytest.approx(2 * rep["doses"][0]["edit_rel"])
    # dose 2 overshoots: the group margins go below the reference
    assert rep["doses"][1]["gap_mean"] > rep["doses"][0]["gap_mean"] + 1.0
    assert rep["doses"][0]["edit_survival"]["projection"] == pytest.approx(1.0)  # fp32 save: exact


@pytest.fixture(scope="module")
def bf16_source(setup):
    path = setup["tmp"] / "shifted_bf16"
    _load(setup["shifted"]).to(torch.bfloat16).save_pretrained(path)
    setup["tok"].save_pretrained(path)
    return str(path)


def _tiny_edit(setup, bf16_source, out, rounding):
    """A tiny (sub-ulp) edit of the MLP weights (many elements share it, so stochastic rounding's noise averages out) on
    a bf16-native source: |Δ| = 1e-4 absolute."""
    cfg = _config(
        setup,
        out,
        model=bf16_source,
        reference=None,
        trust_ref="absolute",
        trust=1e-4,
        cap=1e-4,
        params={"include": ["mlp"], "exclude": []},
        iterations=1,
        work_dtype="bfloat16",
        rounding=rounding,
        families={"pos": FamilyConfig(path=setup["train"], side="pos", target=-30.0, groups=2, group_size=4)},
        doses=[1.0, 2.0],
        serving_formats=["fp8_e4m3"],
        seed=7,
    )
    return surgery.run_surgery(cfg)


def test_bf16_nearest_loses_a_tiny_edit_stochastic_keeps_it(setup, bf16_source):
    near = _tiny_edit(setup, bf16_source, "bf16_near", "nearest")
    sto = _tiny_edit(setup, bf16_source, "bf16_sto", "stochastic")
    assert near["save_dtype"] == sto["save_dtype"] == "bfloat16" and sto["rounding"] == "stochastic"
    assert near["doses"][0]["edit_survival"]["projection"] < 0.2
    assert 0.5 < sto["doses"][0]["edit_survival"]["projection"] < 1.5
    assert sto["doses"][0]["edit_survival"]["changed_frac"] > near["doses"][0]["edit_survival"]["changed_frac"]
    assert "fp8_e4m3" in sto["doses"][0]["edit_survival_served"]
    # the default for a bf16 source is stochastic
    assert (
        surgery.run_surgery(_config(setup, "bf16_default", model=bf16_source, iterations=0, work_dtype="bfloat16"))[
            "rounding"
        ]
        == "stochastic"
    )


def test_saved_checkpoint_reproduces_measured_margins(setup, bf16_source):
    rep = _tiny_edit(setup, bf16_source, "bf16_exact", "stochastic")
    tok = setup["tok"]
    ids, _ = surgery.load_sides(tok, setup["dev"], "both")
    sets = {"dev": {"side": "both", "ids": ids, "token_ids": setup["tid"]}}
    for dose in rep["doses"]:
        model = _load(dose["path"], torch.bfloat16)
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        measured = surgery.eval_margins(MarginReader(model, tok, device="cpu"), sets, micro_batch=4)["dev"]
        assert measured == rep["eval"]["dev"][f"dose_{dose['alpha']:g}"]  # exactly the weights that were measured
    assert rep["doses"][1]["edit_survival"]["projection"] > 0.5  # dose 2 survives stochastic rounding too


# ------------------------------------------------------------------ config, CLI


def test_config_validation(setup):
    ok = _config(setup, "v")
    assert ok.validate() == []
    bad = [
        dict(reference=None),  # target reference without a reference (and trust_ref reference_distance)
        dict(families={"n": FamilyConfig(path="x", constraint="hinge_below")}, reference=None, trust_ref="absolute"),
        dict(chat_template_kwargs={"enable_thinking": True}),
        dict(doses=[0.0]),
        dict(doses=[]),
        dict(gram="sketch"),
        dict(master="gpu"),
        dict(work_dtype="int8"),
        dict(rounding="up"),
        dict(cutoff=0.0),
        dict(params={"include": ["("], "exclude": []}),
        dict(families={"x": FamilyConfig(path="x", side="both")}),
        dict(serving_formats=["int4"]),
    ]
    for kw in bad:
        assert _config(setup, "v", **kw).validate(), kw
    with pytest.raises(SurgeryConfigError, match="did you mean"):
        SurgeryConfig.from_dict({"model": "m", "trsut": 0.1})
    with pytest.raises(SurgeryConfigError):
        SurgeryConfig.from_dict({"families": {"a": {"path": "x", "sides": "pos"}}})
    with pytest.raises(ValueError, match="one"):  # multi-token decisions are refused once the tokenizer is loaded
        cfg = _config(setup, "v")
        cfg.families["pos"].decision = "call the tool now"
        surgery.run_surgery(cfg)
    with pytest.raises(SurgeryConfigError, match="need"):  # max_side_tokens drops every side
        surgery.run_surgery(_config(setup, "v", max_side_tokens=5))
    ids, dropped = surgery.load_sides(setup["tok"], setup["train"], "pos", max_side_tokens=5)
    assert ids == [] and dropped == 24


def test_yaml_and_overrides(setup, tmp_path):
    import yaml

    path = tmp_path / "s.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "surgery": {
                    "model": "m",
                    "reference": "r",
                    "output_dir": "o",
                    "families": {"cont": {"path": "t.jsonl", "decision": "<|im_end|>", "groups": 16}},
                    "params": {"exclude": ["embed"]},
                    "doses": [1, 2],
                }
            }
        )
    )
    cfg = SurgeryConfig.from_yaml(path, ["trust=0.2", "families.cont.groups=4", "params.include=['layers']", "doses=3"])
    assert cfg.trust == 0.2 and cfg.families["cont"].groups == 4 and cfg.families["cont"].decision == "<|im_end|>"
    assert cfg.params.include == ["layers"] and cfg.params.exclude == ["embed"] and cfg.doses == [3]
    assert cfg.validate() == []


def test_cli_run_margins_span(setup, tmp_path, capsys):
    import yaml

    cfg = {
        "model": setup["shifted"],
        "reference": setup["ref"],
        "output_dir": str(tmp_path / "cli_out"),
        "families": {"pos": {"path": setup["train"], "groups": 2, "group_size": 4}},
        "eval": {"dev": {"path": setup["dev"], "side": "pos"}},
        "trust": 1.0,
        "cap": 2.0,
    }
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(cfg))
    argv = ["run", "--config", str(path), "--set", "iterations=1", "--set", "gram=cpu", "--set", "master=cpu"]
    assert surgery.main(argv + ["--set", "gradient_checkpointing=true"]) == 0
    assert "surgery: 1 step(s)" in capsys.readouterr().out
    assert (tmp_path / "cli_out" / "dose_1" / "config.json").exists()
    assert surgery.main(argv) == 1  # non-empty output without --overwrite
    capsys.readouterr()
    args = ["margins", "--model", setup["ref"], "--model", setup["shifted"], "--twins", setup["dev"]]
    assert surgery.main(args + ["--decision", "<tool_call>", "--prefix-side", "both"]) == 0
    out = capsys.readouterr().out
    assert "common" in out and setup["shifted"][-40:] in out
    span = ["span", "--model", setup["shifted"], "--groups", setup["train"], "--targets", setup["dev"]]
    assert surgery.main(span + ["--decision", "<tool_call>", "--k", "1,2,4", "--group-size", "4", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    cov = res["coverage"]
    assert set(cov) == {"1", "2", "4"} and 0.5 < cov["1"] <= cov["2"] + 1e-9 <= cov["4"] + 2e-9 <= 1.0 + 1e-6
