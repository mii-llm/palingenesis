"""Common-mode rejection (palingenesis.cmr) on a tiny CPU model."""

import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from test_opd_trainer import tiny_model  # noqa: E402

from palingenesis.cmr import CMRConfig, CommonModeRejection  # noqa: E402

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
OTHER = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}},
    },
}


def twins(path, n=8, offset=0):
    rows = []
    for i in range(n):
        q = [{"role": "user", "content": f"What is the weather in city number {i + offset}?"}]
        rows.append({"pos": {"messages": q, "tools": [TOOL]}, "neg": {"messages": q, "tools": [OTHER]}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cmr")
    path = tiny_model(tmp, "Qwen/Qwen3-0.6B", "student", 0)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    return tmp, path, tok


def fresh(path):
    m = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m.train()
    return m


def make(model, tok, families, **kw):
    cfg = CMRConfig(enabled=True, families=families, pairs_per_refresh=8, **kw)
    return CommonModeRejection(cfg, model, tok, [p for p in model.parameters() if p.requires_grad], device="cpu")


def test_margin_matches_manual(setup):
    tmp, path, tok = setup
    model = fresh(path)
    cmr = make(model, tok, {"a": twins(tmp / "a.jsonl")})
    side = cmr.twins["a"][0]["pos"]
    m = cmr.margins([side])[0]
    ids = torch.tensor([cmr._prompt_ids(side)])
    with torch.no_grad():
        p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)[cmr.token_id]
    assert torch.isclose(m.detach(), torch.log(p) - torch.log1p(-p), atol=1e-4)


def test_batched_margins_equal_single_rows(setup):
    """Right padding: a batch of prompts of different lengths gives each row's own unpadded margin."""
    tmp, path, tok = setup
    model = fresh(path)
    cmr = make(model, tok, {"a": twins(tmp / "b.jsonl")})
    sides = [cmr.twins["a"][0]["pos"], {"messages": [{"role": "user", "content": "hi"}], "tools": [TOOL, OTHER]}]
    with torch.no_grad():
        batch = cmr.margins(sides)
        single = torch.stack([cmr.margins([s])[0] for s in sides])
    assert torch.allclose(batch, single, atol=1e-4)


def _step(model, cmr=None, seed=0):
    """One AdamW step on a loss that pushes the decision token up on every twin (a pure common-mode push plus noise)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    if cmr is not None:
        cmr.pre_step()
    probe = cmr or make(model, cmr_tok[0], {"a": cmr_tok[1]})
    rows = probe.twins["a"]
    loss = -(probe.margins([r["pos"] for r in rows]).mean() + 0.3 * probe.margins([r["neg"] for r in rows]).mean())
    opt.zero_grad()
    loss.backward()
    opt.step()
    if cmr is not None:
        cmr.post_step()


cmr_tok = [None, None]


def test_hold_removes_common_mode_keeps_differential(setup):
    tmp, path, tok = setup
    fam = twins(tmp / "h.jsonl")
    cmr_tok[:] = [tok, fam]
    base = fresh(path)
    ref = make(base, tok, {"a": fam}).evaluate(make(base, tok, {"a": fam}).twins["a"])
    free = fresh(path)
    _step(free)
    held = fresh(path)
    cmr = make(held, tok, {"a": fam})
    _step(held, cmr)
    ev_free = make(free, tok, {"a": fam}).evaluate(cmr.twins["a"])
    ev_held = cmr.evaluate(cmr.twins["a"])
    d_common_free, d_common_held = ev_free["common"] - ref["common"], ev_held["common"] - ref["common"]
    assert abs(d_common_free) > 1e-3  # the push moved the common mode
    assert abs(d_common_held) <= 0.02 * abs(d_common_free) + 1e-5  # CMR held it (first order)
    d_diff_free, d_diff_held = (
        ev_free["differential"] - ref["differential"],
        ev_held["differential"] - ref["differential"],
    )
    assert abs(d_diff_held) >= 0.5 * abs(d_diff_free)  # the differential mode still moves


def test_gram_schmidt_drops_collinear_family(setup):
    tmp, path, tok = setup
    model = fresh(path)
    f = twins(tmp / "g.jsonl")
    cmr = make(model, tok, {"a": f, "b": f})  # the same twins twice: b is inside a's span
    cmr.refresh()
    assert list(cmr.dirs) == ["a"]
    d = cmr.dirs["a"]
    assert abs(sum(float((x.double() ** 2).sum()) for x in d) - 1.0) < 1e-6


def test_target_mode_moves_common_margin(setup):
    tmp, path, tok = setup
    model = fresh(path)
    f = twins(tmp / "t.jsonl")
    cmr = make(model, tok, {"a": f}, mode="target", target={"a": 0.0}, gain=0.5)
    start = cmr.evaluate(cmr.twins["a"])["common"]
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)  # no task gradient: CMR only
    for _ in range(5):
        cmr.pre_step()
        opt.step()
        cmr.post_step()
    end = cmr.evaluate(cmr.twins["a"])["common"]
    assert abs(end - 0.0) < abs(start - 0.0)


def test_config_validation(setup):
    tmp, path, tok = setup
    assert CMRConfig(enabled=True).validate()
    assert CMRConfig(enabled=True, families={"a": "x"}, mode="push").validate()
    assert CMRConfig(enabled=True, families={"a": "x"}, chat_template_kwargs={"enable_thinking": True}).validate()
    with pytest.raises(ValueError, match="one"):
        make(fresh(path), tok, {"a": twins(tmp / "v.jsonl")}, decision_token="call the tool")
