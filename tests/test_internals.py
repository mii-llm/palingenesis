"""palingenesis.internals on tiny random CPU models: hooks, positions, stores, probes, CKA, lenses, causal tools."""

import json
import math
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from seco_archs import ARCHS, COMMON, torch_kernels  # noqa: E402

from palingenesis.internals import hooks, stats  # noqa: E402
from palingenesis.internals.capture import StoreWriter, capture_dataset, run_backbone  # noqa: E402
from palingenesis.internals.probes import Probe, control_labels, grouped_split, sweep, train_probe  # noqa: E402
from palingenesis.internals.similarity import compare_stores, linear_cka  # noqa: E402
from palingenesis.internals.sites import find_sites, parse_layers  # noqa: E402

# ------------------------------------------------------------------------------------------------ tiny models


def arch_model(arch: str, seed: int = 0, **over):
    model_type, extra = ARCHS[arch]
    with torch_kernels(model_type):
        config = transformers.AutoConfig.for_model(model_type, **{**COMMON, **extra, **over})
        torch.manual_seed(seed)
        model = transformers.AutoModelForCausalLM.from_config(config)
    return model.float().eval()


@pytest.fixture(scope="module")
def hybrid():
    return arch_model("qwen3_5")


def ids(batch=2, length=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, COMMON["vocab_size"], (batch, length), generator=g)


# ------------------------------------------------------------------------------------------------ sites


@pytest.mark.parametrize("arch", ["llama", "qwen3_5", "gemma3"])
def test_find_sites(arch):
    model = arch_model(arch)
    sites = find_sites(model)
    assert sites.n_layers == COMMON["num_hidden_layers"] and sites.norm is not None and sites.head is not None
    assert sites.path.endswith("layers")
    if arch == "qwen3_5":
        assert sites.mixer(1) == "linear_attention" and sites.mixer(3) == "full_attention"
    assert sites.mixer(0) == "embedding"


def test_parse_layers():
    assert parse_layers("all", 5) == [0, 1, 2, 3, 4]
    assert parse_layers("1-3", 5) == [1, 2, 3]
    assert parse_layers("0,-1", 5) == [0, 4]
    assert parse_layers("every:3", 5) == [0, 3, 4]
    assert parse_layers(2, 5) == [2]
    with pytest.raises(ValueError):
        parse_layers("7", 5)


# ------------------------------------------------------------------------------------------------ hooks


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_5", "gemma3"])
def test_recorder_matches_hidden_states_and_cleans_up(arch):
    """r < n equals output_hidden_states[r]; r = n is the pre-norm residual whose norm is HF's (normed) last entry."""
    model = arch_model(arch)
    sites = find_sites(model)
    x = ids()
    n = sites.n_layers
    with hooks.Recorder(model, list(range(n + 1))) as rec:
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True)
        acts = rec.take()
    for r in range(n):
        torch.testing.assert_close(acts[r], out.hidden_states[r])
    torch.testing.assert_close(sites.norm(acts[n]), out.hidden_states[n])
    assert hooks.active_hooks(model) == 0


def test_hooks_removed_when_forward_raises(hybrid):
    with pytest.raises(RuntimeError, match="boom"):
        with hooks.Recorder(hybrid, [0, 2]):
            with hooks.steer(hybrid, {1: torch.zeros(64)}):
                raise RuntimeError("boom")
    assert hooks.active_hooks(hybrid) == 0


def test_selection_last_and_mean(hybrid):
    x = ids(2, 9)
    lengths = [9, 6]
    mask = torch.zeros(2, 9, dtype=torch.long)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
    with hooks.Recorder(hybrid, [2]) as rec, torch.no_grad():
        hybrid(input_ids=x, attention_mask=mask)
        full = rec.take()[2]
        rec.select = hooks.Selection.last(lengths, 9)
        hybrid(input_ids=x, attention_mask=mask)
        last = rec.take()[2]
        rec.select = hooks.Selection(torch.tensor([1, 2, 9 + 4]), torch.tensor([0, 0, 1]), 2, "mean")
        hybrid(input_ids=x, attention_mask=mask)
        pooled = rec.take()[2]
    torch.testing.assert_close(last, torch.stack([full[0, 8], full[1, 5]]))
    torch.testing.assert_close(pooled, torch.stack([full[0, 1:3].mean(0), full[1, 4]]))


def test_right_padding_is_batch_invariant_on_the_hybrid(hybrid):
    """A row's residuals at its real positions do not depend on the right padding of a batch (Gated DeltaNet too)."""
    x = ids(1, 7, seed=3)
    longer = ids(1, 12, seed=4)
    batch = torch.zeros(2, 12, dtype=torch.long)
    batch[0, :7], batch[1] = x[0], longer[0]
    mask = torch.zeros(2, 12, dtype=torch.long)
    mask[0, :7], mask[1] = 1, 1
    with hooks.Recorder(hybrid, [1, 4]) as rec, torch.no_grad():
        run_backbone(hybrid, x, torch.ones_like(x))
        alone = rec.take()
        run_backbone(hybrid, batch, mask)
        batched = rec.take()
    for r in (1, 4):
        torch.testing.assert_close(batched[r][0, :7], alone[r][0], atol=1e-5, rtol=1e-5)


def test_stop_cuts_the_forward(hybrid):
    sites = find_sites(hybrid)
    calls = []
    handle = sites.layers[3].register_forward_hook(lambda *a: calls.append(1))
    try:
        with hooks.Recorder(hybrid, [1], stop=True) as rec:
            run_backbone(hybrid, ids(), torch.ones(2, 11, dtype=torch.long))
            assert rec.take()[1].shape == (2, 11, 64)
    finally:
        handle.remove()
    assert calls == []


def test_steer_zero_is_a_noop_and_generation_unchanged(hybrid):
    x = ids(1, 6)
    with torch.no_grad():
        base_logits = hybrid(input_ids=x).logits
        base_gen = hybrid.generate(input_ids=x, max_new_tokens=8, do_sample=False, pad_token_id=0)
        with hooks.steer(hybrid, {0: torch.zeros(64), 2: torch.zeros(64), 4: torch.zeros(64)}):
            logits = hybrid(input_ids=x).logits
            gen = hybrid.generate(input_ids=x, max_new_tokens=8, do_sample=False, pad_token_id=0)
        with hooks.steer(hybrid, {2: torch.randn(64) * 50}):
            moved = hybrid(input_ids=x).logits
    torch.testing.assert_close(logits, base_logits)
    assert torch.equal(gen, base_gen)
    assert not torch.allclose(moved, base_logits)
    assert hooks.active_hooks(hybrid) == 0


def test_steer_last_and_new_touches_only_the_last_prompt_position(hybrid):
    x = ids(1, 6)
    v = torch.randn(64)
    with hooks.Recorder(hybrid, [2]) as rec, torch.no_grad():
        hybrid(input_ids=x)
        base = rec.take()[2]
        with hooks.steer(hybrid, {2: v}, positions="last_and_new"):
            hybrid(input_ids=x)
            steered = rec.take()[2]
    torch.testing.assert_close(steered[:, :-1], base[:, :-1])
    torch.testing.assert_close(steered[:, -1], base[:, -1] + v)


def test_ablate_removes_the_direction(hybrid):
    d = torch.randn(64)
    u = d / d.norm()
    with hooks.ablate(hybrid, d), hooks.Recorder(hybrid, [2, 4]) as rec, torch.no_grad():
        hybrid(input_ids=ids())
        acts = rec.take()
    for r in (2, 4):
        assert (acts[r] @ u).abs().max() < 1e-4


def test_patching_identical_prompts_is_a_noop_and_last_residual_carries_the_logits(hybrid):
    sites = find_sites(hybrid)
    n = sites.n_layers
    a, b = ids(1, 8, seed=1), ids(1, 8, seed=2)
    with hooks.Recorder(hybrid, [2, n]) as rec, torch.no_grad():
        logits_a = hybrid(input_ids=a).logits
        acts_a = rec.take()
        logits_b = hybrid(input_ids=b).logits
    pos = torch.tensor([[7]])
    with torch.no_grad():
        with hooks.patch(hybrid, 2, pos, acts_a[2][:, 7:8]):
            same = hybrid(input_ids=a).logits
        with hooks.patch(hybrid, n, pos, acts_a[n][:, 7:8]):
            patched = hybrid(input_ids=b).logits
    torch.testing.assert_close(same, logits_a)
    torch.testing.assert_close(patched[:, 7], logits_a[:, 7], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(patched[:, :7], logits_b[:, :7])


# ------------------------------------------------------------------------------------------------ stats


def test_auroc_matches_pairwise_count_with_ties():
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, 5, (60,), generator=g).double()
    y = torch.rand(60, generator=g) < 0.3
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None]).double() + 0.5 * (pos[:, None] == neg[None]).double()).mean()
    assert math.isclose(stats.auroc(s, y), float(brute), rel_tol=1e-12)


def test_bootstrap_interval_and_operating_points():
    g = torch.Generator().manual_seed(1)
    y = (torch.rand(400, generator=g) < 0.3).double()
    s = y + torch.randn(400, generator=g)
    groups = [i // 4 for i in range(400)]
    res = stats.bootstrap_auroc(s, y, groups, resamples=300)
    assert res["lo"] < res["auroc"] < res["hi"] and res["groups"] == 100
    t = stats.threshold_at_fpr(s, y, 0.05)
    assert stats.tpr_at(s, y, t)["fpr"] <= 0.05 + 1e-9
    p = torch.sigmoid(torch.randn(5000, generator=g))
    assert stats.ece(p, torch.bernoulli(p, generator=g)) < 0.03


def test_platt_recovers_a_scale():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(4000, generator=g) * 3
    y = torch.bernoulli(torch.sigmoid(0.5 * z + 1.0), generator=g)
    a, b = stats.fit_platt(z, y)
    assert abs(a - 0.5) < 0.1 and abs(b - 1.0) < 0.2


# ------------------------------------------------------------------------------------------------ probes


def planted_store(path, n=1600, d=48, group_size=4, seed=0, noise=0.3):
    """Rows of `group_size` items per group; label = sign of x . v (+ noise); residual 0 is pure noise."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    v /= v.norm()
    x = torch.randn(n, d, generator=g) * 2 + 0.5
    y = ((x - 0.5) @ v + noise * torch.randn(n, generator=g) > 0.3).float()
    meta = {
        "model": {"weights_sha": "planted", "hidden_size": d, "num_layers": 1},
        "position": "last_prompt",
        "layers": [0, 1],
        "mixers": ["embedding", "attention"],
        "dtype": "float32",
    }
    w = StoreWriter(path, meta, shard_rows=500)
    rows = [
        {"row": i, "id": i, "label": float(y[i]), "group": f"g{i // group_size}", "n_tokens": 10, "token": -1}
        for i in range(n)
    ]
    for s in range(0, n, 250):
        w.add({0: torch.randn(len(rows[s : s + 250]), d, generator=g), 1: x[s : s + 250]}, rows[s : s + 250])
    return w.close(), v


def test_grouped_split_keeps_groups_whole():
    groups = [f"g{i // 3}" for i in range(300)]
    split = grouped_split(groups, 0.1, 0.2, seed=0)
    seen = [{groups[i] for i in split[k].tolist()} for k in ("train", "val", "test")]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert sum(len(v) for v in split.values()) == 300
    assert 0.12 < len(split["test"]) / 300 < 0.28
    with pytest.raises(ValueError, match="too few"):
        grouped_split([0] * 50 + [1] * 50)  # two groups cannot make three splits
    c = control_labels(torch.tensor([1.0, 0.0] * 150), groups)
    assert all(c[i] == c[3 * (i // 3)] for i in range(300))


def test_planted_direction_is_recovered_and_control_stays_at_chance(tmp_path):
    store, v = planted_store(tmp_path / "acts")
    assert len(store) == 1600 and len(store.meta["shards"]) == 4
    report = sweep(store, resamples=200, save_dir=tmp_path / "probes")
    good, noise = report["layers"][1], report["layers"][0]
    assert report["best_layer"] == 1
    assert good["test"]["auroc"] > 0.97 and good["test"]["lo"] > 0.94
    assert abs(good["control_test_auroc"] - 0.5) < 0.1
    assert abs(noise["test"]["auroc"] - 0.5) < 0.1
    assert good["selectivity"] > 0.35
    best = Probe.load(tmp_path / "probes" / "best.safetensors")
    assert torch.nn.functional.cosine_similarity(best.direction, v, dim=0) > 0.9
    assert torch.nn.functional.cosine_similarity(best.weight, v, dim=0) > 0.9
    assert best.meta["position"] == "last_prompt" and best.layer == 1
    assert set(best.thresholds) == {"fpr_0.01", "fpr_0.05", "fpr_0.1"}
    assert json.loads((tmp_path / "probes" / "sweep.json").read_text())["best_layer"] == 1


@pytest.mark.parametrize("kind", ["diffmean", "lda"])
def test_other_probe_kinds_recover_the_direction(tmp_path, kind):
    store, v = planted_store(tmp_path / "acts")
    split = grouped_split(store.groups(), 0.1, 0.2)
    probe = train_probe(store.layer(1), store.labels(), split, kind=kind, layer=1)
    te = split["test"]
    assert stats.auroc(probe.logit(store.layer(1)[te]), store.labels()[te]) > 0.95
    assert torch.nn.functional.cosine_similarity(probe.weight, v, dim=0) > 0.85


def test_probe_save_load_round_trip(tmp_path):
    store, _ = planted_store(tmp_path / "acts", n=400)
    split = grouped_split(store.groups(), 0.2, 0.2)
    probe = train_probe(store.layer(1), store.labels(), split, layer=1, position="last_prompt", meta={"label": "known"})
    path = probe.save(tmp_path / "p.safetensors")
    back = Probe.load(path)
    x = store.layer(1)
    torch.testing.assert_close(back.logit(x), probe.logit(x))
    torch.testing.assert_close(back.proba(x), probe.proba(x))
    assert back.meta == json.loads(json.dumps(probe.meta)) and back.thresholds == probe.thresholds
    assert back.calibration == tuple(probe.calibration) and back.layer == 1
    torch.testing.assert_close(back.mean_diff, probe.mean_diff)


# ------------------------------------------------------------------------------------------------ CKA


def test_linear_cka_identities_and_invariances():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 32, generator=g) @ torch.randn(32, 32, generator=g)
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=g))
    for debiased in (True, False):
        assert math.isclose(linear_cka(x, x, debiased), 1.0, rel_tol=1e-9)
        assert math.isclose(linear_cka(x, 3.0 * x @ q + 5.0, debiased), 1.0, rel_tol=1e-9)
    stretched = x @ torch.diag(torch.linspace(0.1, 10, 32))
    assert linear_cka(x, stretched) < 0.99  # not invariant to non-isotropic linear maps
    a, b = torch.randn(64, 256, generator=g), torch.randn(64, 256, generator=g)  # n << d, independent
    assert linear_cka(a, b, debiased=False) > 0.5  # the biased estimator's inflation
    assert abs(linear_cka(a, b, debiased=True)) < 0.1


def test_compare_stores_same_and_rotated(tmp_path):
    a, _ = planted_store(tmp_path / "a", n=800)
    b, _ = planted_store(tmp_path / "b", n=800)  # identical (same seed)
    res = compare_stores(a, b, resamples=100)
    e = res["layers"][1]
    assert e["cka"] > 0.999 and e["cosine"] > 0.999
    t = e["transfer"]
    assert t["a_on_b"]["auroc"] == t["a_on_a"]["auroc"] and t["direction_cosine"] > 0.999
    with pytest.raises(ValueError, match="same rows"):
        compare_stores(a, _shifted(a, tmp_path))


def _shifted(store, tmp_path):
    """A store with one row fewer: not comparable."""
    meta = dict(store.meta)
    w = StoreWriter(tmp_path / "short", {k: meta[k] for k in ("model", "position", "layers", "mixers", "dtype")})
    w.add({r: store.layer(r)[:-1] for r in store.layers}, store.rows[:-1])
    return w.close()


# ------------------------------------------------------------------------------------------------ lens


@pytest.mark.parametrize("arch,over", [("qwen3_5", {}), ("gemma3", {"final_logit_softcapping": 30.0})])
def test_check_readout_and_lens_identity_at_last_layer(arch, over):
    from palingenesis.internals.lens import TunedLens, check_readout, readout

    model = arch_model(arch, **over)
    sites = find_sites(model)
    x = ids()
    assert check_readout(model, x) < 1e-4
    with hooks.Recorder(model, [1, sites.n_layers]) as rec, torch.no_grad():
        logits = model(input_ids=x).logits
        acts = rec.take()
    torch.testing.assert_close(readout(sites, acts[sites.n_layers]), logits, atol=1e-4, rtol=1e-4)
    lens = TunedLens(sites.n_layers, 64)
    torch.testing.assert_close(readout(sites, lens(acts[1], 1)), readout(sites, acts[1]))  # identity at init


# ------------------------------------------------------------------------------------------------ chat models


@pytest.fixture(scope="module")
def chat(tmp_path_factory):
    from test_opd_trainer import tiny_model

    tmp = tmp_path_factory.mktemp("internals_chat")
    path = tiny_model(tmp, "Qwen/Qwen3-0.6B", "m", 0)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    tok = transformers.AutoTokenizer.from_pretrained(path)
    rows = []
    for i in range(48):
        q = f"Who wrote book number {i}?" if i % 2 else f"What is the capital of country {i}?"
        rows.append(
            {
                "id": f"q{i}",
                "group": f"g{i // 2}",
                "label": float(i % 2),
                "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": f"Answer {i}."}],
            }
        )
    return {"tmp": tmp, "path": path, "model": model, "tok": tok, "rows": rows}


def test_capture_last_prompt_matches_a_manual_forward(chat):
    from palingenesis.internals.positions import Encoder

    model, tok, rows = chat["model"], chat["tok"], chat["rows"][:6]
    store = capture_dataset(
        model,
        tok,
        rows,
        chat["tmp"] / "lp",
        layers="all",
        position="last_prompt",
        dtype="float32",
        batch_size=4,
        shard_rows=4,
    )
    assert len(store) == 6 and len(store.meta["shards"]) == 2 and store.layers == [0, 1, 2]
    enc = Encoder(tok, "last_prompt")
    for j, row in enumerate(store.rows):
        p = enc.prompt_ids(rows[row["row"]])
        assert row["n_tokens"] == len(p) and row["token"] == len(p) - 1
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([p]), output_hidden_states=True).hidden_states
        torch.testing.assert_close(store.layer(1)[j], hs[1][0, -1], atol=1e-5, rtol=1e-5)
    # the prompt ends with the generation prompt: the row's own assistant turn is not in it
    text = tok.decode(enc.prompt_ids(rows[0]))
    assert "Answer" not in text and text.endswith("<|im_start|>assistant\n")


def test_capture_assistant_mean_uses_the_training_mask(chat):
    from palingenesis.data import IGNORE_INDEX, ChatDataset

    model, tok, rows = chat["model"], chat["tok"], chat["rows"][:3]
    store = capture_dataset(
        model, tok, rows, chat["tmp"] / "am", layers=[1], position="assistant_mean", dtype="float32"
    )
    ds = ChatDataset(None, tok, max_seq_length=4096)
    for j, row in enumerate(store.rows):
        out = ds._process(rows[row["row"]])
        mask = out["labels"] != IGNORE_INDEX
        with torch.no_grad():
            hs = model(input_ids=out["input_ids"][None], output_hidden_states=True).hidden_states
        # hidden_states[1] (not the last entry, which transformers returns after the final norm)
        torch.testing.assert_close(store.layer(1)[j], hs[1][0][mask].mean(0), atol=1e-5, rtol=1e-5)
    every = capture_dataset(
        model, tok, rows[:1], chat["tmp"] / "aa", layers=[1], position="assistant_all", dtype="float32"
    )
    first = ds._process(rows[0])
    assert len(every) == int((first["labels"] != IGNORE_INDEX).sum())
    assert len({r["row"] for r in every.rows}) == 1 and len(set(every.groups())) == 1


def test_lens_report_and_tuned_lens(chat):
    from palingenesis.internals.lens import fit_tuned_lens, lens_report

    model, tok, rows = chat["model"], chat["tok"], chat["rows"][:8]
    rep = lens_report(model, tok, rows, position="assistant_all", decision=[tok.convert_tokens_to_ids("<tool_call>")])
    last = rep["layers"][2]
    assert last["kl"] < 1e-6 and last["top1"] == 1.0 and rep["positions"] > 8
    assert rep["layers"][0]["kl"] > last["kl"]
    lens = fit_tuned_lens(model, tok, rows, steps=30, lr=1e-2)
    tuned = lens_report(model, tok, rows, position="assistant_all", tuned=lens)
    assert tuned["layers"][1]["kl"] < rep["layers"][1]["kl"]
    path = lens.save(chat["tmp"] / "lens.safetensors")
    from palingenesis.internals.lens import TunedLens

    back = TunedLens.load(path)
    torch.testing.assert_close(back.translators[1].weight, lens.translators[1].weight)


def test_scorer_reward_and_labels_match_the_store(chat):
    from palingenesis.internals import score as S

    model, tok, rows = chat["model"], chat["tok"], chat["rows"]
    store = capture_dataset(model, tok, rows, chat["tmp"] / "probe_acts", layers=[1, 2], dtype="float32")
    report = sweep(store, resamples=0, save_dir=chat["tmp"] / "probes")
    path = chat["tmp"] / "probes" / "best.safetensors"
    probe = Probe.load(path)
    scorer = S.ProbeScorer(path, model, tok, batch_size=5)
    p = scorer.probabilities(rows)
    order = [r["row"] for r in store.rows]
    torch.testing.assert_close(p[order], probe.proba(store.layer(probe.layer)), atol=1e-5, rtol=1e-5)
    flags = scorer.gate(rows, at="fpr_0.1")
    assert len(flags) == len(rows) and all(isinstance(f, bool) for f in flags)
    labelled = S.score_rows(scorer, rows[:4], "known", gate_at="fpr_0.1")
    assert labelled[0]["known"] == pytest.approx(float(p[0]), abs=1e-5) and "known_flag" in labelled[0]
    S._SCORERS[(str(path), None, "cpu")] = scorer  # the reward's cached scorer (no model reload in the test)
    from palingenesis.rl.rewards import Reward

    reward = Reward("known", S.probe_reward, args={"probe": str(path)})
    assert reward.batched
    values = S.probe_reward(prompts=[r["messages"][:1] for r in rows[:3]], probe=str(path))
    assert values == pytest.approx(p[:3].tolist(), abs=1e-5)
    assert report["best_layer"] in (1, 2)


def test_scorer_refuses_another_shape(chat, tmp_path):
    from palingenesis.internals.score import ProbeMismatch, check_compatible

    probe = Probe(
        "logistic", 1, "last_prompt", torch.zeros(48), 0.0, torch.zeros(48), meta={"model": {"hidden_size": 48}}
    )
    with pytest.raises(ProbeMismatch):
        check_compatible(probe, chat["model"], chat["tok"])


def test_direction_effect_and_patching(chat):
    from palingenesis.internals.causal import decision_margins, direction_effect, patching_sweep
    from palingenesis.internals.positions import Encoder

    model, tok, rows = chat["model"], chat["tok"], chat["rows"][:6]
    store = capture_dataset(model, tok, chat["rows"], chat["tmp"] / "ca", layers=[1], dtype="float32")
    probe = train_probe(store.layer(1), store.labels(), grouped_split(store.groups(), 0.2, 0.2), layer=1)
    decision = [tok.convert_tokens_to_ids("<think>")]
    res = direction_effect(model, tok, rows, probe, decision, alphas=(-1.0, 0.0, 1.0), n_random=3)
    ids = [Encoder(tok).prompt_ids(r) for r in rows]
    base = float(decision_margins(model, tok, ids, decision).mean())
    assert res["margins"][1] == pytest.approx(base, abs=1e-5) and res["base_margin"] == pytest.approx(base, abs=1e-5)
    assert len(res["random_slopes"]) == 3 and "ablation" in res
    twins = [
        {"pos": {"messages": rows[i]["messages"][:1]}, "neg": {"messages": rows[i + 1]["messages"][:1]}} for i in (0, 2)
    ]
    pat = patching_sweep(model, tok, twins, decision, layers="all", offsets=(-1, -2))
    last = find_sites(model).n_layers
    assert pat["layers"][last][0] == pytest.approx(1.0, abs=1e-4)  # the last residual at the decision decides it
    same = [{"pos": t["pos"], "neg": t["pos"]} for t in twins]
    assert all(
        v == 0 for vals in patching_sweep(model, tok, same, decision, offsets=(-1,))["layers"].values() for v in vals
    )
    assert hooks.active_hooks(model) == 0


# ------------------------------------------------------------------------------------------------ config + CLI


def test_config_validation():
    from palingenesis.internals.config import InternalsConfig, InternalsConfigError

    cfg = InternalsConfig.from_dict({"internals": {"models": {"a": "x"}, "train": "t.jsonl", "output_dir": "o"}})
    assert cfg.validate() == []
    cfg.apply_overrides(["position=assistant_all", "compare=[[a, b]]", "tests.it=it.jsonl"])
    errors = " ".join(cfg.validate())
    assert "*_all" in errors and "not in models" in errors and cfg.tests == {"it": "it.jsonl"}
    with pytest.raises(InternalsConfigError, match="did you mean 'position'"):
        InternalsConfig.from_dict({"positon": "x"})


def test_cli_capture_probe_compare_and_run(chat, tmp_path, capsys):
    import yaml

    from palingenesis.internals.cli import main

    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in chat["rows"]))
    acts = tmp_path / "acts"
    assert main(["capture", "--model", chat["path"], "--data", str(data), "--out", str(acts), "--layers", "1-2"]) == 0
    assert (
        main(
            ["probe", "--acts", str(acts), "--out", str(tmp_path / "pr"), "--bootstrap", "50", "--test", f"same={acts}"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "best layer (validation)" in out
    assert main(["compare", "--a", str(acts), "--b", str(acts), "--bootstrap", "20"]) == 0
    assert main(["probe", "--acts", str(tmp_path / "missing")]) == 1
    cfg = {
        "models": {"a": chat["path"], "b": chat["path"]},
        "train": str(data),
        "tests": {"again": str(data)},
        "compare": [["a", "b"]],
        "layers": "1-2",
        "bootstrap": 20,
        "output_dir": str(tmp_path / "run"),
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["run", "--config", str(tmp_path / "run.yaml")]) == 0
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["compare"]["a__b"]["layers"]["1"]["cka"] > 0.999
    assert (tmp_path / "run" / "probes" / "a" / "best.safetensors").exists()
    mtime = (tmp_path / "run" / "acts" / "a" / "train" / "meta.json").stat().st_mtime
    assert main(["run", "--config", str(tmp_path / "run.yaml")]) == 0  # stores reused
    assert (tmp_path / "run" / "acts" / "a" / "train" / "meta.json").stat().st_mtime == mtime
