"""pgs learn training core on CPU: a tiny random model on the real Qwen3 tokenizer, stub generators.

These test mechanics, not learning: a random model cannot show that facts are acquired. They check that
a session trains only the scope's adapter and lowers its loss, records what it was trained on, stacks
on its ancestors, and leaves the live state untouched when it fails.
"""

import json
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from palingenesis.learn import LearnConfig, Learner, StoreError  # noqa: E402
from palingenesis.learn.store import sha256_file  # noqa: E402
from palingenesis.learn.trainer import PreflightError  # noqa: E402

USER = "user:acme/support/ana"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # noqa: BLE001 - offline or not cached
        pytest.skip(f"tokenizer unavailable: {e}")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        initializer_range=0.2,
    )
    path = tmp_path_factory.mktemp("learn_base") / "model"
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path), tok


class FixedSampler:
    """Deterministic completions (a stub generator for sampling): the same tokens for every prompt."""

    def __init__(self, tok, text="The price is 25 euros.", fail_after=None):
        self.ids = tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        self.calls, self.fail_after = 0, fail_after

    def generate(self, prompts, max_new_tokens, temperature, role):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("killed")
        return [list(self.ids) for _ in prompts]


def make_learner(tmp_path, base, **recipe):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.model.chat_template_kwargs = {"enable_thinking": False}
    config.model.device = "cpu"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    config.recipe.preflight = False
    config.recipe.max_new_tokens = 8
    config.recipe.rows_per_step = 8
    config.recipe.passes = 2
    for key, value in recipe.items():
        setattr(config.recipe, key, value)
    return Learner.init(tmp_path / "store", base, "acme", config)


def test_run_lowers_the_loss_and_records_the_version(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=12, learning_rate=2e-2, onpolicy_fraction=0.0)
    learner.add_notes(
        "org:acme", ["The price of Orion Pro is 25 euros.", "The seat limit of Orion Pro is 5."], approve=True
    )
    result = learner.run("org:acme", sampler=FixedSampler(tok))
    assert result.steps == 12 and len(result.losses) == 12
    assert sum(result.losses[-3:]) < sum(result.losses[:3]), result.losses
    ss = learner.store.scope("org:acme")
    manifest = ss.manifest(result.version)
    assert manifest["note_ids"] == [n.id for n in ss.active_notes()] and manifest["parent_versions"] == {}
    assert manifest["rank"] == 4 and manifest["gated"] is False
    session = ss.dir / "sessions" / result.session
    assert (session / "questions.jsonl").exists() and json.loads((session / "train.json").read_text())["losses"]
    assert not list((ss.dir / "sessions").glob("*.tmp"))
    assert ss.live is None  # a version goes live only when asked
    learner.set_live("org:acme", result.version)
    assert ss.live == result.version and all(f.consolidated_in == result.version for f in ss.active_facts())
    assert len(ss.pool()) == len((session / "questions.jsonl").read_text().splitlines())
    # The saved adapter is a PEFT LoRA on the MLP projections only.
    config = json.loads((ss.version_dir(result.version) / "adapter_config.json").read_text())
    assert config["r"] == 4 and "mlp" in config["target_modules"]


def test_sft_mode_and_divergences_run(tmp_path, base):
    path, tok = base
    for i, (mode, divergence) in enumerate([("sft", "fkl"), ("distill", "jsd"), ("distill", "rkl")]):
        learner = make_learner(tmp_path / str(i), path, steps=2, mode=mode, divergence=divergence)
        learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
        result = learner.run("org:acme", sampler=FixedSampler(tok))
        assert len(result.losses) == 2 and all(loss == loss for loss in result.losses)


def test_hf_sampler_ancestors_and_stacking(tmp_path, base):
    path, _ = base
    learner = make_learner(tmp_path, path, steps=2, max_new_tokens=4)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    org = learner.run("org:acme")  # HF generate, on-policy and teacher samples
    learner.set_live("org:acme", org.version)
    learner.remember(USER, "my project is Orion")
    user = learner.run(USER)
    manifest = learner.store.scope(USER).manifest(user.version)
    assert manifest["parent_versions"] == {"org:acme": org.version} and manifest["rank"] == 2
    learner.set_live(USER, user.version)
    composite = learner.stack(USER)
    stacked = json.loads((composite / "manifest.json").read_text())
    assert stacked["chain"] == [["org:acme", org.version], [USER, user.version]]
    assert stacked["rank"] == 8 and stacked["ranks"] == [4, 2]  # 4 + 2 padded to vLLM's rank 8
    assert json.loads((composite / "adapter_config.json").read_text())["r"] == 8
    # Training the user scope left the org's adapter file as it was.
    org_ss = learner.store.scope("org:acme")
    assert (
        sha256_file(org_ss.version_dir(org.version) / "adapter_model.safetensors")
        == org_ss.manifest(org.version)["sha256"]
    )


def test_warm_start_and_replay(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=2)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    first = learner.run("org:acme", sampler=FixedSampler(tok))
    learner.set_live("org:acme", first.version)
    learner.add_notes("org:acme", ["The colour of Orion Team is blue."], approve=True)
    second = learner.run("org:acme", sampler=FixedSampler(tok))
    manifest = learner.store.scope("org:acme").manifest(second.version)
    assert manifest["warm_start"] == first.version and second.new_facts == 1


@pytest.mark.parametrize("arm", [{"anchor_teacher": "previous"}, {"replay_teacher": "self"}])
def test_previous_model_teacher_arms_train(tmp_path, base, arm):
    """The "previous" teacher swaps the live LoRA weights in and out: it must not break the student's backward."""
    path, tok = base
    learner = make_learner(tmp_path, path, steps=2, onpolicy_fraction=0.0, **arm)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    learner.set_live("org:acme", learner.run("org:acme", sampler=FixedSampler(tok)).version)
    learner.add_notes("org:acme", ["The colour of Orion Team is blue."], approve=True)
    result = learner.run("org:acme", sampler=FixedSampler(tok))
    assert len(result.losses) == 2 and all(loss == loss for loss in result.losses)


def test_killed_session_leftovers_are_discarded(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=1)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    ss = learner.store.scope("org:acme")
    # What a SIGKILL leaves: a temporary session and adapter weights without a manifest.
    (ss.dir / "sessions" / "s0001.tmp").mkdir(parents=True)
    (ss.dir / "sessions" / "s0001.tmp" / "questions.jsonl").write_text('{"q": "x"}\n')
    (ss.version_dir("v0001")).mkdir(parents=True)
    (ss.version_dir("v0001") / "adapter_model.safetensors").write_bytes(b"weights")
    result = learner.run("org:acme", sampler=FixedSampler(tok))
    assert not list((ss.dir / "sessions").glob("*.tmp"))
    # The number is free again, and the new version's weights are the trained ones.
    assert ss.versions() == [result.version] == ["v0001"]
    weights = ss.version_dir("v0001") / "adapter_model.safetensors"
    assert weights.read_bytes() != b"weights" and ss.manifest("v0001")["sha256"] == sha256_file(weights)


def test_failed_session_leaves_the_live_state_untouched(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=3)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    first = learner.run("org:acme", sampler=FixedSampler(tok))
    learner.set_live("org:acme", first.version)
    learner.add_notes("org:acme", ["The colour of Orion Team is blue."], approve=True)
    ss = learner.store.scope("org:acme")
    before = sorted(str(p.relative_to(ss.dir)) for p in ss.dir.rglob("*"))
    with pytest.raises(RuntimeError, match="killed"):
        learner.run("org:acme", sampler=FixedSampler(tok, fail_after=1))
    after = sorted(str(p.relative_to(ss.dir)) for p in ss.dir.rglob("*"))
    assert after == before and ss.live == first.version
    # The store is unlocked and the session can run again.
    assert learner.run("org:acme", sampler=FixedSampler(tok)).version == "v0002"


def test_preflight_aborts_when_the_teacher_misses_the_probes(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=1, preflight=True)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    with pytest.raises(PreflightError, match="probes"):
        learner.run("org:acme", sampler=FixedSampler(tok, text="I do not know."))
    assert not list((learner.store.scope("org:acme").dir / "sessions").iterdir())
    result = learner.run("org:acme", sampler=FixedSampler(tok, text="It costs 25 euros."))
    assert result.dropped_probes == 0


def test_refuses_a_different_base(tmp_path, base):
    path, tok = base
    learner = make_learner(tmp_path, path, steps=1)
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    learner.run("org:acme", sampler=FixedSampler(tok))
    learner.store.state["fingerprint"]["chat_template_sha"] = "another template"
    learner.store.save_state()
    with pytest.raises(StoreError, match="chat_template_sha"):
        learner.run("org:acme", sampler=FixedSampler(tok))


def test_nothing_to_train(tmp_path, base):
    path, _ = base
    learner = make_learner(tmp_path, path)
    with pytest.raises(StoreError, match="nothing to train"):
        learner.run("org:acme")
