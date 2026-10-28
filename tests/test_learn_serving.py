"""pgs learn serving on CPU: versioned publish, the parity canary, rollback and unload against in-process fake
OpenAI-compatible servers (vLLM's and SGLang's LoRA routes), with tiny adapter weights in a real store."""

import json
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from learn_fakes import CharReference, FakeServer, gold_answers, make_version  # noqa: E402

from palingenesis.learn import (  # noqa: E402
    LearnConfig,
    Learner,
    StoreError,
    serving,
)
from palingenesis.learn.serving import (  # noqa: E402
    CanaryError,
    PublishError,
    RuntimeLoRAError,
    parse_name,
    served_name,
)

ORG, TEAM, USER = "org:acme", "team:acme/support", "user:acme/support/ana"


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    config.serving.drain_seconds = 60
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


@pytest.fixture
def servers():
    made = []

    def make(**kw):
        made.append(FakeServer(**kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def live_version(learner, scope, note, passed=True):
    learner.add_notes(scope, [note], approve=True) if scope != USER else learner.remember(scope, note)
    version = make_version(learner, scope, passed=passed)
    learner.set_live(scope, version, force=not passed)
    return version


def publisher(learner, urls, answers=None, clock=None, **kw):
    reference = CharReference(answers if answers is not None else gold_answers(learner, [ORG, TEAM, USER]))
    return learner.publisher(servers=urls, reference=reference, clock=clock or Clock(), **kw), reference


# ---------------------------------------------------------------------------- names


def test_served_names():
    assert served_name(USER, 3) == "user:acme/support/ana@v3"
    scope, n = parse_name("user:acme/support/ana@v3")
    assert str(scope) == USER and n == 3
    assert parse_name("org:acme")[1] is None
    for bad in ("user:acme/support/ana@3", "user:acme/support/ana@v0", "nonsense", "org:acme@vx"):
        with pytest.raises(StoreError):
            parse_name(bad)
    assert serving.same_model("Qwen/Qwen3.5-4B", "/models/Qwen3.5-4B")
    assert not serving.same_model("Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B")


# ---------------------------------------------------------------------------- refusals


def test_publish_refuses_without_servers_or_live_version(learner, servers):
    with pytest.raises(PublishError, match="no server"):
        learner.publisher(servers=[]).publish(ORG)
    s = servers()
    with pytest.raises(PublishError, match="no live adapter"):
        learner.publisher(servers=[s.url]).publish(ORG)
    assert not s.paths("/v1/load_lora_adapter")


def test_publish_refuses_a_forced_version_unless_forced(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.", passed=False)  # forced live
    s = servers(answers=gold_answers(learner, [ORG]))
    pub, _ = publisher(learner, [s.url])
    with pytest.raises(PublishError, match="forced past"):
        pub.publish(ORG)
    assert not s.loaded
    result = pub.publish(ORG, force=True)
    assert result.forced and any("forced" in r for r in result.reasons)
    entry = learner.store.published()["scopes"][ORG]["versions"]["1"]
    assert entry["forced"] and entry["forced_reasons"] and entry["state"] == "live"
    # status shows what is served
    status = learner.status()
    assert status["scopes"][0]["published"]["live"] == "org:acme@v1"


def test_publish_refuses_a_child_evaluated_on_an_old_parent(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    live_version(learner, USER, "my editor is vim")
    live_version(learner, ORG, "The price of Orion Max is 40 euros.")  # the parent moved on
    s = servers()
    pub, _ = publisher(learner, [s.url])
    with pytest.raises(PublishError, match="evaluate it again"):
        pub.publish(USER)


def test_publish_refuses_another_base(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    s = servers(base="other-model")
    with pytest.raises(PublishError, match="not the store's base"):
        publisher(learner, [s.url])[0].publish(ORG)


# ---------------------------------------------------------------------------- publish, drain, unload


def test_versioned_publish_on_two_replicas(learner, servers):
    v1 = live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    a, b = servers(answers=answers), servers(answers=answers)
    clock = Clock()
    pub, ref = publisher(learner, [a.url, b.url], answers, clock)
    result = pub.publish(ORG)
    assert result.name == "org:acme@v1" and result.action == "publish"
    assert all(c.passed for c in result.canary) and len(result.canary) == 2
    for s in (a, b):
        load = s.paths("/v1/load_lora_adapter")
        assert [c["lora_name"] for c in load] == ["org:acme@v1"]
        assert load[0]["lora_path"].endswith(result.composite)
    composite = learner.store.root / result.composite
    manifest = json.loads((composite / "manifest.json").read_text())
    assert manifest["chain"] == [[ORG, v1]] and manifest["rank"] == 8  # 4 padded to vLLM's next rank
    assert ref.loaded == [str(composite)]  # the reference holds the same composite
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] == 1 and rec["versions"]["1"]["servers"] == {a.url: "loaded", b.url: "loaded"}
    assert rec["versions"]["1"]["canary"][a.url]["passed"]
    # the same chain again: nothing to do
    assert pub.publish(ORG).action == "unchanged"
    # a new version: a new name, the old one drains, then unloads
    live_version(learner, ORG, "The price of Orion Max is 40 euros.")
    pub._reference.answers.update(gold_answers(learner, [ORG]))
    for s in (a, b):
        s.answers.update(gold_answers(learner, [ORG]))
    result = pub.publish(ORG)
    assert result.name == "org:acme@v2" and result.previous == "org:acme@v1"
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] == 2 and rec["versions"]["1"]["state"] == "draining"
    assert set(a.loaded) == {"org:acme@v1", "org:acme@v2"}
    assert pub.unload() == []  # still inside the drain time
    clock.t += 61
    lines = pub.unload()
    assert len(lines) == 2 and set(a.loaded) == {"org:acme@v2"} == set(b.loaded)
    assert learner.store.published()["scopes"][ORG]["versions"]["1"]["state"] == "retired"
    # the bare alias was unloaded after each flip (the resolver plugin re-resolves it)
    assert [c["lora_name"] for c in a.paths("/v1/unload_lora_adapter")][:1] == [ORG]


def test_partial_failure_unloads_everywhere(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    good, bad = servers(answers=answers), servers(answers=answers, runtime=False)
    pub, _ = publisher(learner, [good.url, bad.url], answers)
    with pytest.raises(RuntimeLoRAError, match="VLLM_ALLOW_RUNTIME_LORA_UPDATING"):
        pub.publish(ORG)
    assert good.loaded == {}  # loaded on the first replica, then unloaded
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] is None and rec["versions"]["1"]["state"] == "failed"
    assert rec["history"][-1]["action"] == "publish_failed"


@pytest.mark.parametrize("fault", ["ignore_adapter", "half_adapter", "wrong_answers"])
def test_canary_fault_injection_keeps_the_old_version(learner, servers, fault):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    healthy = servers(answers=answers)
    pub, _ = publisher(learner, [healthy.url], answers)
    pub.publish(ORG)
    live_version(learner, ORG, "The price of Orion Max is 40 euros.")
    answers.update(gold_answers(learner, [ORG]))
    faulty = servers(answers=answers, fault=fault)
    pub2, _ = publisher(learner, [healthy.url, faulty.url], answers)
    healthy.answers = answers
    with pytest.raises(CanaryError) as e:
        pub2.publish(ORG)
    message = str(e.value)
    expected = {
        "ignore_adapter": "do not depend on the adapter",
        "half_adapter": "correlation",
        "wrong_answers": "more probes wrong",
    }[fault]
    assert expected in message and faulty.url in message
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] == 1 and rec["versions"]["2"]["state"] == "failed"
    assert "org:acme@v2" not in healthy.loaded and "org:acme@v2" not in faulty.loaded
    assert "org:acme@v1" in healthy.loaded


def test_no_canary_is_recorded(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    s = servers()
    result = publisher(learner, [s.url])[0].publish(ORG, canary=False)
    assert result.forced and "without the parity canary" in result.reasons[-1]


def test_canary_unit_metrics():
    ref = [-1.0, -2.0, -0.5]
    assert serving.k3_mismatch(ref, ref, [0, 1, 2]) == pytest.approx(0.0)
    import math

    k3 = serving.k3_mismatch([-1.0], [-1.5], [0])
    rho = math.exp(0.5)
    assert k3 == pytest.approx(rho - 1 - 0.5)
    corr, _, _ = serving._pearson([1, 2, 3], [2, 4, 6.5])
    assert corr > 0.99
    assert serving._pearson([1, 1, 1], [1, 2, 3])[0] is None


# ---------------------------------------------------------------------------- rollback and forget


def test_rollback_reinjects_notes_and_reverts_live(learner, servers):
    v1 = live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    s = servers(answers=answers)
    clock = Clock()
    learner.config.serving.drain_seconds = 0
    pub, _ = publisher(learner, [s.url], answers, clock)
    pub.publish(ORG)
    v2 = live_version(learner, ORG, "The price of Orion Max is 40 euros.")
    answers.update(gold_answers(learner, [ORG]))
    pub.publish(ORG)
    assert "org:acme@v1" not in s.loaded  # drain 0: unloaded at once
    assert [n.text for n in learner.wake_notes(ORG)] == []
    result = pub.rollback(ORG)
    assert result.name == "org:acme@v1" and result.previous == "org:acme@v2"
    assert "org:acme@v1" in s.loaded and "org:acme@v2" not in s.loaded  # loaded again; v2 drained at once
    ss = learner.store.scope(ORG)
    assert ss.live == v1
    # v2's note is in the served prompt again and unconsolidated (the next session trains it)
    assert [n.text for n in learner.wake_notes(ORG)] == ["The price of Orion Max is 40 euros."]
    assert all(f.consolidated_in != v2 for f in ss.facts())
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] == 1 and rec["history"][-1]["action"] == "rollback"
    with pytest.raises(PublishError, match="no earlier"):
        pub.rollback(ORG)
    # rolling forward to v2 works too (its composite is still in the store)
    assert pub.rollback(ORG, to=2).name == "org:acme@v2"


def test_forget_unloads_pruned_versions_and_refuses_rollback_to_them(learner, servers):
    learner.add_notes(ORG, ["Project Nightjar's codename is Bluefin Harbor."], approve=True)
    note = learner.store.scope(ORG).active_notes()[0].id
    make = make_version(learner, ORG)
    learner.set_live(ORG, make)
    answers = gold_answers(learner, [ORG])
    s = servers(answers=answers)
    pub, _ = publisher(learner, [s.url], answers)
    pub.publish(ORG)
    assert "org:acme@v1" in s.loaded
    report = learner.forget(ORG, [note])
    assert report.ok and report.pruned_versions
    assert report.unloaded == [f"unloaded org:acme@v1 from {s.url}"]
    assert "org:acme@v1" not in s.loaded
    rec = learner.store.published()["scopes"][ORG]
    assert rec["live"] is None and rec["versions"]["1"]["state"] == "pruned"
    assert not (learner.store.root / rec["versions"]["1"]["composite"]).exists()
    with pytest.raises(PublishError):
        pub.rollback(ORG, to=1)


# ---------------------------------------------------------------------------- engines and transports


def test_sglang_publish_uses_its_routes_and_prefix(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    s = servers(engine="sglang", answers=answers)
    pub, _ = publisher(learner, [s.url], answers, engine="sglang")
    result = pub.publish(ORG)
    assert result.canary[0].passed
    load = s.paths("/load_lora_adapter")[0]
    assert load["lora_name"] == "org:acme@v1" and load["lora_path"].endswith("/sglang")
    assert all(c.get("lora_path") in (None, "org:acme@v1") for c in s.paths("/generate"))


def test_engine_adapter_renames_language_model_modules(tmp_path):
    from safetensors.torch import load_file

    from palingenesis.learn import adapter as adapters

    a, b = torch.randn(2, 4), torch.randn(4, 2)
    adapters.write_adapter({"model.language_model.layers.0.mlp.up_proj": (a, b)}, tmp_path / "c", "base")
    out = serving.engine_adapter(tmp_path / "c", "sglang")
    keys = set(load_file(str(out / adapters.ADAPTER_WEIGHTS)))
    assert keys == {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight",
    }
    assert serving.engine_adapter(tmp_path / "c", "vllm") == tmp_path / "c"


def test_resolver_transport_loads_on_the_warm_request(learner, servers):
    from palingenesis.learn.vllm_plugin import resolve

    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])

    def plugin(name):
        found = resolve([learner.store.root], "base-model", name)
        return found[1] if found else None

    s = servers(answers=answers, resolver=plugin)
    pub, _ = publisher(learner, [s.url], answers, transport="resolver")
    result = pub.publish(ORG)
    assert result.canary[0].passed and not s.paths("/v1/load_lora_adapter")
    assert s.loaded["org:acme@v1"] == str(learner.store.root / result.composite)
    # a failed version never resolves again
    rec = learner.store.published()
    rec["scopes"][ORG]["versions"]["1"]["state"] = "failed"
    learner.store.write_published(rec)
    assert resolve([learner.store.root], "base-model", "org:acme@v1") is None


# ---------------------------------------------------------------------------- CLI


def test_cli_publish_rollback_unload_and_key(learner, servers, capsys, tmp_path):
    import yaml

    from palingenesis.learn import cli
    from palingenesis.learn.proxy import hash_key, load_keys

    root = str(learner.store.root)
    s = servers()
    assert cli.main(["publish", root, "--scope", ORG, "--servers", s.url, "--no-canary"]) == 1  # nothing live
    assert "no live adapter" in capsys.readouterr().err
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    assert cli.main(["publish", root, "--scope", ORG, "--servers", s.url, "--no-canary", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == "org:acme@v1" and out["forced"]
    live_version(learner, ORG, "The price of Orion Max is 40 euros.")
    assert cli.main(["publish", root, "--scope", ORG, "--servers", s.url, "--no-canary", "--drain-seconds", "0"]) == 0
    assert "published org:acme@v2" in capsys.readouterr().out and "org:acme@v1" not in s.loaded
    assert cli.main(["rollback", root, "--scope", ORG, "--to", "v1", "--drain-seconds", "0"]) == 0
    assert "rolled back to org:acme@v1" in capsys.readouterr().out
    assert cli.main(["unload", root]) == 0
    capsys.readouterr()
    assert cli.main(["status", root, "--json"]) == 1  # served past the canary (--no-canary): an issue
    status = json.loads(capsys.readouterr().out)
    assert any("serving org:acme@v1" in i and "canary" in i for i in status["issues"])
    keys = tmp_path / "keys.yaml"
    assert cli.main(["key", root, "--scope", "user:acme/support/ana", "--keys", str(keys)]) == 0
    key = capsys.readouterr().out.strip()
    entries = load_keys(keys, "acme")
    assert hash_key(key) in entries and key not in keys.read_text()
    assert yaml.safe_load(keys.read_text())["keys"][0]["name"] == "ana"


def test_an_interrupted_publish_is_settled(learner, servers):
    live_version(learner, ORG, "The price of Orion Pro is 25 euros.")
    answers = gold_answers(learner, [ORG])
    s = servers(answers=answers)
    pub, _ = publisher(learner, [s.url], answers)
    data = learner.store.published()
    rec = serving.scope_record(data, ORG)
    rec["next"] = 2
    rec["versions"]["1"] = {
        "name": "org:acme@v1",
        "composite": "x",
        "composite_id": "x",
        "chain": [],
        "rank": 8,
        "engine": "vllm",
        "transport": "api",
        "state": "loading",
        "servers": {s.url: "loaded"},
    }
    s.loaded["org:acme@v1"] = "/x"
    learner.store.write_published(data)
    assert pub.unload() == ["org:acme@v1: an interrupted publish, marked failed"]
    assert "org:acme@v1" not in s.loaded
    assert pub.publish(ORG).name == "org:acme@v2"
