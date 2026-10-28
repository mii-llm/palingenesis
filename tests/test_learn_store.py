"""pgs learn store: scopes, tenant isolation, atomic files, the lock, fingerprints, config strictness."""

import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, LearnConfigError, Learner, ScopeId, Store, StoreError  # noqa: E402
from palingenesis.learn.config import served_rank  # noqa: E402
from palingenesis.learn.store import read_jsonl, write_jsonl  # noqa: E402


def stub_config():
    config = LearnConfig()
    config.generate.generator = "stub"
    return config


@pytest.fixture
def learner(tmp_path):
    return Learner.init(tmp_path / "store", "base-model", "acme", stub_config())


def test_scope_ids_and_chain():
    user = ScopeId.parse("user:acme/support/ana")
    assert [str(s) for s in user.chain()] == ["org:acme", "team:acme/support", "user:acme/support/ana"]
    assert str(ScopeId.parse("user:acme/ana").parent) == "org:acme"
    assert not user.shared and ScopeId.parse("team:acme/support").shared
    for bad in ("acme", "org:acme/x", "team:acme", "user:acme/../x", "group:acme", "org:", "org:../etc"):
        with pytest.raises(StoreError):
            ScopeId.parse(bad)


def test_tenant_isolation(learner, tmp_path):
    with pytest.raises(StoreError, match="belongs to org"):
        learner.scope("team:other/support")
    other = Learner.init(tmp_path / "other", "base-model", "other", stub_config())
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    other.add_notes("org:other", ["The price of Vega is 3 euros."], approve=True)
    acme_files = {p.read_text() for p in (tmp_path / "store").rglob("*.jsonl")}
    assert not any("Vega" in text for text in acme_files)
    assert learner.store.root != other.store.root


def test_init_layout_and_reopen(learner):
    assert (learner.store.root / "learn_state.yaml").exists()
    assert (learner.store.root / "scopes" / "org" / "acme" / "sessions").is_dir()
    again = Learner.open(learner.store.root)
    assert again.store.org == "acme" and again.config.generate.generator == "stub"
    with pytest.raises(StoreError, match="already"):
        Store.init(learner.store.root, "base-model", "acme")
    with pytest.raises(StoreError, match="not a learn store"):
        Store.open(learner.store.root / "scopes")


def test_atomic_rewrite_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"a": 1}])
    write_jsonl(path, [{"a": 2}, {"a": 3}])
    assert read_jsonl(path) == [{"a": 2}, {"a": 3}]
    assert [p.name for p in tmp_path.iterdir()] == ["x.jsonl"]


def test_lock_is_exclusive_across_store_objects(learner):
    second = Store.open(learner.store.root)
    with learner.store.lock():
        with learner.store.lock():  # re-entrant within one object
            pass
        with pytest.raises(StoreError, match="locked"):
            with second.lock():
                pass
    with second.lock():
        pass


def test_fingerprint_recorded_then_enforced(learner):
    fp = {"base_revision": "abc", "tokenizer_sha": "t", "chat_template_sha": "c", "quantization": "none"}
    learner.store.check_fingerprint(fp)
    learner.store.check_fingerprint(dict(fp))
    for key in fp:
        with pytest.raises(StoreError, match=key):
            learner.store.check_fingerprint({**fp, key: "changed"})


def test_strict_config(tmp_path):
    config = LearnConfig()
    config.set("recipe.passes", "3")
    assert config.recipe.passes == 3
    config.set("recipe.mix", "{new: 1.0, anchor: 0.5}")
    assert config.recipe.mix == {"new": 1.0, "anchor": 0.5}
    with pytest.raises(LearnConfigError, match="did you mean recipe.passes"):
        config.set("recipe.pases", 2)
    with pytest.raises(LearnConfigError, match="unknown config section"):
        config.set("gate.x", 1)
    config.recipe.mode = "rl"
    with pytest.raises(LearnConfigError, match="recipe.mode"):
        config.validate()
    path = tmp_path / "learn.yaml"
    path.write_text("recipe:\n  passes: 2\nscopes:\n  ranks: {org: 32, team: 16, user: 16}\n")
    assert LearnConfig.from_yaml(path).recipe.passes == 2


def test_served_rank_pads_to_vllm_ranks():
    assert served_rank([32, 16, 16]) == 64
    assert served_rank([32, 16]) == 64
    assert served_rank([16]) == 16
    assert served_rank([8, 4]) == 16
    with pytest.raises(LearnConfigError, match="exceeds"):
        served_rank([512, 16])
