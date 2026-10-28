"""pgs learn forget: no trace of a forgotten note in any store file, versions that held it pruned."""

import json
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.adapter import ADAPTER_WEIGHTS, write_adapter  # noqa: E402
from palingenesis.learn.forget import scan  # noqa: E402
from palingenesis.learn.generate import StubGenerator  # noqa: E402
from palingenesis.learn.recipe import plan_session  # noqa: E402
from palingenesis.learn.store import sha256_file  # noqa: E402
from palingenesis.learn.trainer import write_session_files  # noqa: E402

USER = "user:acme/support/ana"
SECRET = "The codename of Project Nightjar is Bluefin Harbor."
KEEP = "The price of Orion Pro is 25 euros."


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def fake_session(learner, scope, parents=None):
    """A session's files and an adapter version, as a training run leaves them (no model needed)."""
    ss = learner.store.scope(scope)
    plan = plan_session(ss, StubGenerator(), learner.config)
    number = ss.next_number()
    session, version = f"s{number:04d}", f"v{number:04d}"
    write_session_files(ss.dir / "sessions" / session, plan, learner.config)
    rank = learner.config.scopes.ranks[ss.scope.level]
    out = ss.version_dir(version)
    write_adapter({"model.layers.0.mlp.up_proj": (torch.randn(rank, 8), torch.randn(16, rank))}, out, "base-model")
    manifest = {
        "version": version,
        "scope": scope,
        "session": session,
        "rank": rank,
        "parent_versions": parents or {},
        "note_ids": [n.id for n in ss.active_notes()],
        "fact_ids": [f.id for f in plan.new_facts],
        "sha256": sha256_file(out / ADAPTER_WEIGHTS),
        "pruned": None,
    }
    ss.write_manifest(version, manifest)
    learner.set_live(scope, version)
    return version


def store_files(learner):
    return [p for p in learner.store.root.rglob("*") if p.is_file()]


def test_forget_leaves_no_trace(learner):
    learner.add_notes("org:acme", ["The support hours of Orion are 9 to 17."], approve=True)
    org_version = fake_session(learner, "org:acme")
    keep = learner.remember(USER, KEEP).note_id
    gone = learner.remember(USER, SECRET).note_id
    fake_session(learner, USER, parents={"org:acme": org_version})
    composite = learner.stack(USER)
    # A pending inbox item and a promotion request about the same note.
    learner.add_notes(USER, ["Project Nightjar ships in May"])
    learner.promote(USER, gone, "team:acme/support")
    ss = learner.store.scope(USER)
    assert scan(learner.store.root, [SECRET])  # the text is everywhere before
    assert any("Nightjar" in json.dumps(r) for r in ss.pool())

    report = learner.forget(USER, [gone])
    assert report.ok, report.leftovers
    for path in store_files(learner):
        data = path.read_bytes()
        assert SECRET.encode() not in data and b"Bluefin Harbor" not in data, path
    # Derived questions, probes and replay prompts are gone with it; the other note stays.
    assert not any("Bluefin" in json.dumps(r) for r in ss.pool())
    assert [n.id for n in ss.notes()] == [keep]
    assert not learner.admin_queue()
    # The version that held the note lost its weights; LIVE no longer points to it; the composite is gone.
    assert report.pruned_versions == ["v0001"]
    assert not (ss.version_dir("v0001") / ADAPTER_WEIGHTS).exists()
    assert ss.manifest("v0001")["pruned"]["reason"] == "forget"
    assert ss.live is None
    assert not composite.exists()
    # The kept note was consolidated in the pruned version only: the rebuild trains it again as new.
    assert all(f.consolidated_in is None for f in ss.facts()) and ss.notes()[0].consolidated_in is None
    assert plan_session(ss, StubGenerator(), learner.config).new_facts
    assert report.rebuild == [USER]
    # Tombstones hold an id and a hash only.
    tombstone = ss.tombstones()[0]
    assert set(tombstone) == {"id", "sha256", "forgotten_at"} and tombstone["id"] == gone
    # The pending inbox item is a separate item: forget it by id too.
    pending = learner.inbox(USER)[0]
    assert learner.forget(USER, [pending.id]).ok
    assert not learner.inbox(USER)
    assert all(b"Nightjar ships" not in p.read_bytes() for p in store_files(learner))


def test_forget_at_a_parent_marks_children_for_rebuild(learner):
    secret = learner.add_notes("org:acme", [SECRET], approve=True)[0].note_id
    learner.add_notes("org:acme", [KEEP], approve=True)
    org_version = fake_session(learner, "org:acme")
    learner.remember(USER, "my editor is vim")
    fake_session(learner, USER, parents={"org:acme": org_version})
    report = learner.forget("org:acme", [secret])
    assert report.ok and report.rebuild == ["org:acme", USER]
    assert set(learner.store.state["rebuild"]) == {"org:acme", USER}
    # The child's own adapter never held the org's notes: it is kept and only marked.
    assert (learner.store.scope(USER).version_dir("v0001") / ADAPTER_WEIGHTS).exists()


def test_forget_follows_the_supersede_chain_and_clears_stale_values(learner, tmp_path):
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "a.md"
    doc.write_text("# Nightjar\nThe codename of Project Nightjar is Bluefin Harbor.\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    doc.write_text("# Nightjar\nThe codename of Project Nightjar is Redwood Quay.\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    ss = learner.store.scope("org:acme")
    current = ss.active_notes()[0]
    assert ss.active_facts()[0].stale_values == ["Bluefin Harbor"]
    report = learner.forget("org:acme", [current.id])
    assert report.notes == 2 and report.ok
    assert not ss.notes() and not ss.facts()
    assert all(b"Bluefin" not in p.read_bytes() and b"Redwood" not in p.read_bytes() for p in store_files(learner))


def test_forget_unknown_id(learner):
    with pytest.raises(KeyError):
        learner.forget(USER, ["n_missing"])


def test_scan_reports_leftovers(learner):
    learner.remember(USER, SECRET)
    stray = learner.store.root / "notes-backup.txt"
    stray.write_text(SECRET)
    note = learner.store.scope(USER).active_notes()[0].id
    report = learner.forget(USER, [note])
    assert not report.ok and report.leftovers == ["notes-backup.txt"]


def test_forget_a_pending_document(learner, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# Nightjar\n" + SECRET + "\n")
    report = learner.add_documents("org:acme", "*.md", root=tmp_path / "docs")
    assert scan(learner.store.root, [SECRET])
    assert learner.forget("org:acme", report.held).ok
    assert not learner.inbox("org:acme")


def test_versions_warm_started_from_a_pruned_one_are_pruned(learner, tmp_path):
    """A retired note is in no later manifest, but a version warm-started from one that learned it holds it."""
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "a.md"
    doc.write_text("# A\n" + SECRET + "\n\n# B\n" + KEEP + "\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    first = fake_session(learner, "org:acme")
    doc.write_text("# B\n" + KEEP + "\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    ss = learner.store.scope("org:acme")
    retired = next(n for n in ss.notes() if n.status == "retired")
    second = fake_session(learner, "org:acme")
    manifest = ss.manifest(second)
    manifest["warm_start"] = first
    ss.write_manifest(second, manifest)
    assert retired.id not in manifest["note_ids"]
    report = learner.forget("org:acme", [retired.id])
    assert report.pruned_versions == [first, second] and ss.live is None
    assert not (ss.version_dir(second) / ADAPTER_WEIGHTS).exists()


def test_forget_deletes_weights_a_killed_run_left(learner):
    note = learner.remember(USER, SECRET).note_id
    fake_session(learner, USER)
    ss = learner.store.scope(USER)
    write_adapter({"model.layers.0.mlp.up_proj": (torch.randn(2, 8), torch.randn(16, 2))}, ss.version_dir("v0002"), "b")
    learner.forget(USER, [note])
    assert not ss.version_dir("v0002").exists()


def test_forget_at_a_parent_scrubs_descendant_conflicts_and_stale_values(learner):
    org_note = learner.add_notes("org:acme", [KEEP], approve=True)[0].note_id
    held = learner.remember(USER, "The price of Orion Pro is 30 euros.")
    assert held.status == "held"
    other = "user:acme/support/luca"
    learner.remember(other, "The price of Orion Pro is 35 euros.")
    learner.approve(other, [learner.inbox(other)[0].id], "supersede")
    assert learner.store.scope(other).active_facts()[0].stale_values == ["25 euros"]
    learner.forget("org:acme", [org_note])
    assert all(b"25 euros" not in p.read_bytes() for p in store_files(learner))
    assert learner.inbox(USER)[0].payload["conflicts"] == []
    assert learner.approve(USER, [held.inbox_id], "both")[0].status == "stored"
