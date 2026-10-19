"""palingenesis.merge on small synthetic checkpoints (no model downloads)."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from palingenesis import merge as M

NAMES = ["model.embed.weight", "model.layers.0.w", "model.layers.1.w", "lm_head.weight"]
SHAPES = {
    "model.embed.weight": (6, 4),
    "model.layers.0.w": (4, 4),
    "model.layers.1.w": (4, 4),
    "lm_head.weight": (6, 4),
}


def make(path, seed, *, shards=1, dtype=torch.bfloat16, names=NAMES, extra=None, files=True):
    """A checkpoint dir: tensors (seeded), sharded into `shards` files with an index when > 1, plus config files."""
    path.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {n: torch.randn(SHAPES[n], generator=g).to(dtype) for n in names}
    tensors.update(extra or {})
    keys = list(tensors)
    if shards == 1:
        save_file(tensors, str(path / "model.safetensors"))
    else:
        weight_map = {}
        for i in range(shards):
            part = {k: tensors[k] for k in keys[i::shards]}
            fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            save_file(part, str(path / fname))
            weight_map.update({k: fname for k in part})
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    if files:
        (path / "config.json").write_text(json.dumps({"model_type": "toy", "seed": seed}))
        (path / "chat_template.jinja").write_text("{{ messages }}")
        (path / "training_meta.json").write_text("{}")
    return path


def weights(path):
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def test_interpolate_math_dtype_layout_and_files(tmp_path):
    a, b = make(tmp_path / "a", 0, shards=2), make(tmp_path / "b", 1, shards=1)  # different sharding: matched by name
    r = M.interpolate(a, b, 0.3, tmp_path / "out")
    wa, wb, wo = weights(a), weights(b), weights(tmp_path / "out")
    for n in NAMES:
        expect = (0.7 * wa[n].float() + 0.3 * wb[n].float()).to(torch.bfloat16)
        assert wo[n].dtype == torch.bfloat16 and torch.equal(wo[n], expect)
    # the output mirrors the template (the start): its two shards and index, its config; training files are not copied
    assert sorted(p.name for p in (tmp_path / "out").glob("*.safetensors")) == sorted(
        p.name for p in a.glob("*.safetensors")
    )
    index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(NAMES) and index["metadata"]["total_size"] > 0
    assert json.loads((tmp_path / "out" / "config.json").read_text())["seed"] == 0
    assert (tmp_path / "out" / "chat_template.jinja").exists() and not (
        tmp_path / "out" / "training_meta.json"
    ).exists()
    cfg = json.loads((tmp_path / "out" / "merge_config.json").read_text())
    assert cfg["method"] == "interpolate" and cfg["lam"] == 0.3 and [s["coef"] for s in cfg["sources"]] == [0.7, 0.3]
    assert r.tensors == len(NAMES) and r.from_template == 0


def test_soup_linear_and_weight_check(tmp_path):
    ms = [make(tmp_path / f"m{i}", i) for i in range(3)]
    M.soup(ms, tmp_path / "soup")
    ws = [weights(m) for m in ms]
    wo = weights(tmp_path / "soup")
    for n in NAMES:
        assert torch.equal(wo[n], (sum(w[n].float() for w in ws) / 3).to(torch.bfloat16))
    with pytest.raises(M.MergeError, match="sum to"):
        M.linear(ms[:2], [0.5, 0.6], tmp_path / "bad")
    M.linear(ms[:2], [1.3, -0.3], tmp_path / "extrapolated")  # affine: sums to 1
    M.linear(ms[:2], [0.5, 0.6], tmp_path / "unnorm", unnormalized=True)


def test_task_arithmetic_adds_each_delta_to_the_base(tmp_path):
    base, a, b = make(tmp_path / "base", 0), make(tmp_path / "a", 1), make(tmp_path / "b", 2)
    M.task_arithmetic(base, [a, b], [1.0, 0.4], tmp_path / "out", dtype="fp32")
    wbase, wa, wb, wo = weights(base), weights(a), weights(b), weights(tmp_path / "out")
    for n in NAMES:
        expect = wa[n].float() + 0.4 * (wb[n].float() - wbase[n].float())  # = A + lam (B - base)
        assert wo[n].dtype == torch.float32 and torch.allclose(wo[n], expect, atol=1e-6)


def test_missing_tensors_error_or_template(tmp_path):
    full = make(tmp_path / "full", 0)
    partial = make(tmp_path / "partial", 1, names=NAMES[:2])
    with pytest.raises(M.MergeError, match="missing"):
        M.interpolate(full, partial, 0.5, tmp_path / "out")
    r = M.interpolate(full, partial, 0.5, tmp_path / "out", missing="template")
    assert r.from_template == 2
    wf, wo = weights(full), weights(tmp_path / "out")
    assert torch.equal(wo["lm_head.weight"], wf["lm_head.weight"])


def test_materialize_completes_a_partial_fp32_checkpoint(tmp_path):
    export = make(
        tmp_path / "export", 0, extra={"visual.patch.weight": torch.ones(3, 3, dtype=torch.bfloat16)}, shards=2
    )
    ckpt = make(tmp_path / "ckpt", 5, dtype=torch.float32, files=False)  # text-only training save, fp32, no files
    r = M.materialize(ckpt, export, tmp_path / "out")
    wc, we, wo = weights(ckpt), weights(export), weights(tmp_path / "out")
    assert set(wo) == set(we) and r.from_template == 1
    for n in NAMES:
        assert wo[n].dtype == torch.bfloat16 and torch.equal(wo[n], wc[n].to(torch.bfloat16))
    assert torch.equal(wo["visual.patch.weight"], we["visual.patch.weight"])
    assert (tmp_path / "out" / "config.json").exists()


def test_shape_mismatch_nonfinite_extra_and_existing_out(tmp_path):
    a = make(tmp_path / "a", 0)
    b = tmp_path / "b"
    b.mkdir()
    bad = {n: torch.zeros(SHAPES[n]).to(torch.bfloat16) for n in NAMES}
    bad["lm_head.weight"] = torch.zeros(7, 4, dtype=torch.bfloat16)
    save_file(bad, str(b / "model.safetensors"))
    with pytest.raises(M.MergeError, match="shape"):
        M.interpolate(a, b, 0.5, tmp_path / "o1")
    c = tmp_path / "c"
    c.mkdir()
    inf = {n: torch.full(SHAPES[n], float("inf"), dtype=torch.float32) for n in NAMES}
    save_file(inf, str(c / "model.safetensors"))
    with pytest.raises(M.MergeError, match="non-finite"):
        M.interpolate(a, c, 0.5, tmp_path / "o2")
    d = make(tmp_path / "d", 1, extra={"extra.w": torch.zeros(2, dtype=torch.bfloat16)})
    with pytest.raises(M.MergeError, match="not in the template"):
        M.interpolate(a, d, 0.5, tmp_path / "o3")
    M.interpolate(a, d, 0.5, tmp_path / "o3", extra="drop")
    with pytest.raises(M.MergeError, match="not empty"):
        M.interpolate(a, d, 0.5, tmp_path / "o3", extra="drop")
    M.interpolate(a, d, 0.5, tmp_path / "o3", extra="drop", overwrite=True)


def test_integer_buffers_are_copied_and_must_agree(tmp_path):
    ids = torch.arange(5)
    a = make(tmp_path / "a", 0, extra={"pos_ids": ids})
    b = make(tmp_path / "b", 1, extra={"pos_ids": ids})
    M.interpolate(a, b, 0.5, tmp_path / "out")
    assert torch.equal(weights(tmp_path / "out")["pos_ids"], ids)
    c = make(tmp_path / "c", 2, extra={"pos_ids": ids + 1})
    with pytest.raises(M.MergeError, match="non-float"):
        M.interpolate(a, c, 0.5, tmp_path / "out2")


def test_cli(tmp_path, capsys):
    a, b, base = make(tmp_path / "a", 0), make(tmp_path / "b", 1), make(tmp_path / "base", 2)
    assert M.main(["interpolate", str(a), str(b), "--lam", "0.3", "-o", str(tmp_path / "i")]) == 0
    assert M.main(["linear", f"{a}:0.25", f"{b}:0.75", "-o", str(tmp_path / "l")]) == 0
    assert M.main(["soup", str(a), str(b), "-o", str(tmp_path / "s")]) == 0
    assert M.main(["task-arithmetic", "--base", str(base), f"{a}:1", f"{b}:0.3", "-o", str(tmp_path / "t")]) == 0
    assert M.main(["materialize", str(a), "--template", str(b), "-o", str(tmp_path / "m")]) == 0
    assert M.main(["linear", f"{a}:0.5", f"{b}:0.6", "-o", str(tmp_path / "bad")]) == 1  # weights do not sum to 1
    assert "sum to" in capsys.readouterr().err
    wa, wb, wl = weights(a), weights(b), weights(tmp_path / "l")
    assert torch.equal(
        wl["model.embed.weight"],
        (0.25 * wa["model.embed.weight"].float() + 0.75 * wb["model.embed.weight"].float()).to(torch.bfloat16),
    )


def test_model_stock_ratio_math():
    base = torch.zeros(4, 4)
    d = torch.randn(4, 4)
    # identical task vectors: c = 1 -> t = 1 (the average); orthogonal: c = 0 -> t = 0 (the base)
    assert torch.isclose(M.stock_ratio(base, [base + d, base + d]), torch.tensor([1.0])).all()
    e = torch.zeros(4, 4)
    e[0, 0] = 1.0
    f = torch.zeros(4, 4)
    f[1, 1] = 1.0
    assert torch.isclose(M.stock_ratio(base, [e, f]), torch.tensor([0.0])).all()
    # opposite task vectors (c = -1) are clamped to the base, not extrapolated
    assert M.stock_ratio(base, [base + d, base - d]).item() == 0.0
    # k = 3, c = 0.5 -> t = 3 * .5 / (1 + 2 * .5) = .75 ; built from unit vectors with pairwise cosine .5
    v = [
        torch.tensor([1.0, 0, 0]),
        torch.tensor([0.5, 0.75**0.5, 0]),
        torch.tensor([0.5, 0.75**0.5 / 3, (2 / 3) ** 0.5]),
    ]
    t = M.stock_ratio(torch.zeros(3), v)
    assert torch.isclose(t, torch.tensor([0.75]), atol=1e-5).all()
    # filter_wise: one t per output row
    assert M.stock_ratio(base, [base + d, base + 2 * d], filter_wise=True).shape == (4,)
    with pytest.raises(M.MergeError):
        M.stock_ratio(base, [d])


def test_model_stock_merge_and_config(tmp_path):
    base = make(tmp_path / "base", 0, dtype=torch.float32)
    wb = weights(base)
    paths = []
    for i in range(3):  # fine-tunes = base + shared update + private noise
        g = torch.Generator().manual_seed(100 + i)
        shared = {n: torch.ones(SHAPES[n]) for n in NAMES}
        p = tmp_path / f"m{i}"
        p.mkdir()
        save_file(
            {n: wb[n] + shared[n] + 0.5 * torch.randn(SHAPES[n], generator=g) for n in NAMES},
            str(p / "model.safetensors"),
        )
        (p / "config.json").write_text("{}")
        paths.append(p)
    M.model_stock(base, paths, tmp_path / "out")
    wo, ws = weights(tmp_path / "out"), [weights(p) for p in paths]
    cfg = json.loads((tmp_path / "out" / "merge_config.json").read_text())
    assert cfg["method"] == "model_stock" and set(cfg["t"]) == set(NAMES)
    for n in NAMES:
        t = M.stock_ratio(wb[n], [w[n] for w in ws])
        expect = wb[n] + t * (torch.stack([w[n] for w in ws]).mean(0) - wb[n])
        assert torch.allclose(wo[n], expect, atol=1e-6)
        assert 0.0 < cfg["t"][n] < 1.0  # partial agreement: between the base and the average
    assert M.main(["model-stock", "--base", str(base), *map(str, paths), "-o", str(tmp_path / "cli")]) == 0


def test_ram_unique_kept_shared_averaged(tmp_path):
    base = make(tmp_path / "base", 0, dtype=torch.float32)
    wb = weights(base)
    deltas = []
    for i, p in enumerate((tmp_path / "a", tmp_path / "b")):
        p.mkdir()
        d = {n: torch.zeros(SHAPES[n]) for n in NAMES}
        for n in NAMES:
            d[n][0, 0] = 1.0 + i  # shared: both change element (0, 0)
            d[n][1 + i, 1] = 3.0  # unique: a changes (1, 1), b changes (2, 1)
        save_file({n: wb[n] + d[n] for n in NAMES}, str(p / "model.safetensors"))
        (p / "config.json").write_text("{}")
        deltas.append(d)
    M.ram(base, [tmp_path / "a", tmp_path / "b"], tmp_path / "out")
    wo = weights(tmp_path / "out")
    for n in NAMES:
        got = wo[n] - wb[n]
        assert torch.isclose(got[0, 0], torch.tensor(1.5))  # shared: mean of 1 and 2
        assert torch.isclose(got[1, 1], torch.tensor(3.0)) and torch.isclose(
            got[2, 1], torch.tensor(3.0)
        )  # unique: whole
        assert got.abs().sum() == got[0, 0].abs() + got[1, 1].abs() + got[2, 1].abs()  # inactive: base
    cfg = json.loads((tmp_path / "out" / "merge_config.json").read_text())
    assert cfg["method"] == "ram" and cfg["unique_share"] > 0 and cfg["shared_share"] > 0
    assert (
        M.main(["ram", "--base", str(base), str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(tmp_path / "c")]) == 0
    )
