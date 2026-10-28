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


class KeySampler:
    """Answers from a {subject: value} table per role (student = candidate, previous = live, parent = P with
    or without notes): the value of the subject the user turn names, else "I don't know." Records calls."""

    def __init__(self, tok, knows: dict[str, dict[str, str]]):
        self.tok, self.knows, self.calls = tok, knows, []
        self.end = tok.convert_tokens_to_ids("<|im_end|>")

    def generate(self, prompts, max_new_tokens, temperature, role):
        out = []
        for p in prompts:
            user = self.tok.decode(p).split("<|im_start|>user")[-1]
            self.calls.append((role, user))
            table = self.knows.get(role, {})
            answer = next((f"It is {v}." for s, v in table.items() if s.lower() in user.lower()), "I don't know.")
            out.append(self.tok.encode(answer, add_special_tokens=False) + [self.end])
        return out


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
    config.eval.max_new_tokens = 8
    config.eval.drift_max_new_tokens = 4
    config.eval.max_drift_prompts = 3
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
    learner.set_live("org:acme", result.version, force=True)
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
    learner.set_live("org:acme", org.version, force=True)
    learner.remember(USER, "my project is Orion")
    user = learner.run(USER)
    manifest = learner.store.scope(USER).manifest(user.version)
    assert manifest["parent_versions"] == {"org:acme": org.version} and manifest["rank"] == 2
    learner.set_live(USER, user.version, force=True)
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
    learner.set_live("org:acme", first.version, force=True)
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
    learner.set_live("org:acme", learner.run("org:acme", sampler=FixedSampler(tok)).version, force=True)
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
    learner.set_live("org:acme", first.version, force=True)
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


PRICE, COLOUR = "The price of Orion Pro is 25 euros.", "The colour of Orion Team is blue."


def test_run_evaluates_and_gates_publishing(tmp_path, base):
    """Run -> eval -> publish on a tiny model, with a sampler that stands in for what each chain knows."""
    path, tok = base
    learner = make_learner(tmp_path, path, steps=2)
    # A random tiny model drifts far from any target; these gates are not what this test is about.
    learner.config.gates.drift_cap = learner.config.gates.drift_increment_max = 100.0
    learner.config.gates.retention_max_net_loss = 0
    learner.add_notes("org:acme", [PRICE], approve=True)
    knows_price = {"Orion Pro": "25 euros"}
    first = learner.run("org:acme", sampler=KeySampler(tok, {"student": knows_price, "parent": knows_price}))
    report = first.evaluation.report
    assert first.evaluation.passed, first.evaluation.summary()
    assert report["metrics"]["acquisition"]["relative"]["rate"] == 1.0 and report["live_chain"] == [["org:acme", None]]
    assert report["metrics"]["acquisition"]["live"]["k"] == 0  # nothing live: P does not know it closed-book
    session = learner.store.scope("org:acme").dir / "sessions" / first.session
    assert {"report.json", "report.md", "diagnostics.jsonl", "answers.jsonl"} <= {p.name for p in session.iterdir()}
    learner.set_live("org:acme", first.version)
    ss = learner.store.scope("org:acme")
    assert ss.manifest(first.version)["gated"] and ss.manifest(first.version)["evaluation"]["passed"]

    # Session 2: the candidate learns the colour but loses the price, which the live version knows.
    learner.add_notes("org:acme", [COLOUR], approve=True)
    sampler = KeySampler(
        tok,
        {
            "student": {"Orion Team": "blue"},
            "previous": knows_price,
            "parent": {"Orion Team": "blue", "Orion Pro": "25 euros"},
        },
    )
    second = learner.run("org:acme", sampler=sampler)
    m = second.evaluation.report["metrics"]
    assert m["retention"]["paired"]["losses"] == m["retention"]["paired"]["n"] > 0
    assert not second.evaluation.passed and second.evaluation.report["gates"]
    failed = [g["name"] for g in second.evaluation.report["gates"] if not g["passed"]]
    assert failed == ["retention"]
    # The live chain's answers to probes it already answered as a candidate are reused, not regenerated.
    live_prompts = [user for role, user in sampler.calls if role == "previous"]
    assert live_prompts and not any("capital of France" in u for u in live_prompts)
    assert not any("Orion Pro" in u for u in live_prompts if "price" in u.lower())
    with pytest.raises(StoreError, match="gates failed: retention"):
        learner.set_live("org:acme", second.version)
    assert ss.live == first.version
    learner.set_live("org:acme", second.version, force=True)
    assert ss.manifest(second.version)["forced"] and "forced" in learner.status()["issues"][0]
    # Evaluating again (no sampler given: the HF model) rewrites the report against the new live chain.
    again = learner.evaluate("org:acme", second.version, sampler=KeySampler(tok, {}))
    assert again.report["live_chain"] == [["org:acme", second.version]]


def test_cli_run_eval_status(tmp_path, base, capsys):
    from palingenesis.learn.cli import main

    path, _ = base
    config = tmp_path / "learn.yaml"
    config.write_text(
        "model:\n  device: cpu\n  chat_template_kwargs: {enable_thinking: false}\n"
        "scopes:\n  ranks: {org: 4, team: 2, user: 2}\n"
        "recipe:\n  preflight: false\n  steps: 1\n  rows_per_step: 4\n  max_new_tokens: 4\n"
        "eval:\n  max_new_tokens: 4\n  drift_max_new_tokens: 2\n  max_drift_prompts: 2\n"
        "gates:\n  acquisition_min: 1.0\n"
    )
    store = str(tmp_path / "store")

    def cli(*args):
        code = main(list(args))
        out = capsys.readouterr()
        return code, out.out, out.err

    assert cli("init", store, "--base", path, "--org", "acme", "--generator", "stub", "--config", str(config))[0] == 0
    cli("add", store, "--scope", "org:acme", "--note", PRICE, "--approve")
    # A random model does not answer the probes: the acquisition gate fails and blocks --live.
    code, out, err = cli("run", store, "--scope", "org:acme", "--live")
    assert code == 1 and "gates FAILED" in out and "acquisition" in out and "not live" in err
    code, out, _ = cli("eval", store, "--scope", "org:acme", "--json")
    assert code == 1 and json.loads(out)["version"] == "v0001" and not json.loads(out)["passed"]
    code, out, _ = cli("eval", store, "--scope", "org:acme", "--version", "v0001", "--live", "--force")
    assert code == 0 and "FORCED" in out
    code, out, _ = cli("status", store)
    assert code == 1 and "live v0001 (forced)" in out and "forced past" in out
    # The facts are consolidated now: the next run would replay them with anchors only.
    code, out, _ = cli("plan", store, "--scope", "org:acme")
    assert code == 0 and "0 new facts" in out and "0 questions" in out and "replay rows" in out


# ---------------------------------------------------------------------------- children after a parent update

KESTREL, FALCON = "The editor of Kestrel is vim.", "The desk of Falcon is B12."


def scramble(learner, scope, version, seed):
    """Large random weights for `version` (a few training steps leave a LoRA near zero), so every chain is a
    different function of the input; the manifest's sha256 follows."""
    from safetensors.torch import load_file, save_file

    from palingenesis.learn import adapter as adapters

    ss = learner.store.scope(scope)
    path = ss.version_dir(version) / adapters.ADAPTER_WEIGHTS
    gen = torch.Generator().manual_seed(seed)
    tensors = {k: 0.3 * torch.randn(v.shape, generator=gen) for k, v in load_file(str(path)).items()}
    save_file(tensors, str(path))
    manifest = ss.manifest(version)
    manifest["sha256"] = sha256_file(path)
    ss.write_manifest(version, manifest)


def chain_sampler(path, tok, learner, labels: dict[str, list], knows: dict[str, dict[str, str]]):
    """An HFSampler stand-in that answers as the chain the model computes under the role really is: it runs the
    model on a fixed input inside the role's context and takes the nearest of `labels` ({label: chain}),
    each computed on a fresh base with the chain stacked as frozen hooks; then answers from knows[label]."""
    from palingenesis.learn import adapter as adapters

    probe = torch.tensor([tok.encode("Kestrel Falcon Orion", add_special_tokens=False)])
    references = {}
    for label, chain in labels.items():
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        tensors = [adapters.read_adapter(learner.store.scope(s).version_dir(v))[0] for s, v in chain]
        adapters.FrozenDelta(model, adapters.stack(tensors))
        with torch.no_grad():
            references[label] = model(input_ids=probe).logits[0, -1]
    answers = KeySampler(tok, knows)
    seen = []

    class ChainSampler:
        def __init__(self, model, stop_ids, pad_id, roles, micro_seqs=8):
            self.model, self.roles = model, roles

        def generate(self, prompts, max_new_tokens, temperature, role):
            with self.roles[role](), torch.no_grad():
                logits = self.model(input_ids=probe).logits[0, -1]
            label = min(references, key=lambda k: float((references[k] - logits).norm()))
            seen.append((role, label))
            return answers.generate(prompts, max_new_tokens, temperature, label)

    ChainSampler.seen = seen
    return ChainSampler


@pytest.mark.parametrize("destroys", [True, False])
def test_a_parent_update_that_breaks_a_child_keeps_the_old_composite(tmp_path, base, monkeypatch, destroys):
    """A child re-evaluated on a new parent is paired against the composite it is served with (the old parent
    under it), not against itself: a parent version that loses a fact the child retained fails the child's
    retention gate, the child keeps serving the old composite and is marked for rebuild. A parent update that
    keeps it lets the child follow."""
    from types import SimpleNamespace

    from palingenesis.learn import trainer
    from palingenesis.learn.scheduler import sleep

    path, tok = base
    org = "org:acme"
    learner = make_learner(tmp_path, path, steps=1)
    learner.config.gates.drift_cap = learner.config.gates.drift_increment_max = 100.0
    learner.config.gates.retention_max_net_loss = 0

    def trained(scope, seed):
        version = learner.run(scope, sampler=FixedSampler(tok), evaluate=False).version
        scramble(learner, scope, version, seed)
        return version

    learner.add_notes(org, [PRICE], approve=True)
    learner.set_live(org, org1 := trained(org, 1), force=True)
    learner.add_notes(USER, [KESTREL], approve=True)
    learner.set_live(USER, trained(USER, 2), force=True)
    learner.add_notes(USER, [FALCON], approve=True)
    learner.set_live(USER, user := trained(USER, 3), force=True)
    served = [[org, org1], [USER, user]]
    published = learner.store.published()
    entry = {"name": f"{USER}@v1", "chain": served, "state": "live", "servers": {}}
    published["scopes"][USER] = {"live": 1, "next": 2, "versions": {"1": entry}, "history": []}
    learner.store.write_published(published)

    # The parent update: org's next version, trained and set live by a sleep pass.
    learner.add_notes(org, [COLOUR], approve=True)
    org2 = []

    def run(scope):  # the random tiny model fails org's own gates: set live forced here, not by the pass
        org2.append(trained(scope, 4))
        Learner.set_live(learner, scope, org2[-1], force=True)
        return SimpleNamespace(version=org2[-1], evaluation=SimpleNamespace(passed=True, report={"gates": []}))

    monkeypatch.setattr(learner, "set_live", lambda scope, version, force=False: None)
    both = {"Kestrel": "vim", "Falcon": "B12"}

    class Publisher:
        published = []

        def publish(self, scope, force=False):
            self.published.append(str(scope))
            return SimpleNamespace(action="publish", name=f"{scope}@v2")

        def unload(self, scope=None, everything_drained=False):
            return []

    class LazySampler:
        """The chain references need org's new version, which the pass trains first."""

        def __new__(cls, *args, **kwargs):
            labels = {"served": served, "new": [[org, org2[-1]], [USER, user]], "parent": [[org, org2[-1]]]}
            knows = {"served": both, "new": {"Falcon": "B12"} if destroys else both, "parent": both}
            sampler = chain_sampler(path, tok, learner, labels, knows)
            LazySampler.seen = sampler.seen
            return sampler(*args, **kwargs)

    monkeypatch.setattr(trainer, "HFSampler", LazySampler)
    report = sleep(learner, publish=True, publisher=Publisher(), run=run)
    assert len(org2) == 1 and learner.store.scope(org).live == org2[0]
    follow = next(e for e in report.entries if e.scope == USER)
    ss = learner.store.scope(USER)
    evaluated = json.loads((ss.dir / "sessions" / ss.manifest(user)["session"] / "report.json").read_text())
    if destroys:
        failed = [g["name"] for g in evaluated["gates"] if not g["passed"]]
        assert failed == ["retention"] and evaluated["metrics"]["retention"]["paired"]["losses"] > 0
    roles = dict(LazySampler.seen)
    assert roles["student"] == "new" and roles.get("previous") == "served" and roles["parent"] == "parent"
    team = ["team:acme/support", None]
    assert evaluated["baseline_chain"] == [served[0], team, served[1]]
    assert evaluated["live_chain"] == [[org, org2[0]], team, [USER, user]]
    assert "served composite" in evaluated["baseline"]
    assert not any("itself" in w for w in evaluated["warnings"])
    if destroys:
        assert follow.action == "failed_gates" and "old composite" in follow.detail and not report.ok
        assert Publisher.published == [org]  # the child was not republished on the new parent
        assert learner.store.published()["scopes"][USER]["versions"]["1"]["chain"] == served
        assert USER in learner.store.state["rebuild"]
    else:
        assert evaluated["passed"] and follow.action == "followed" and Publisher.published == [org, USER]
        assert USER not in (learner.store.state.get("rebuild") or [])
