"""pgs learn ingestion: chunking, source diffs, fact supersedes, contradiction check, replay pool upkeep."""

import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.generate import StubGenerator  # noqa: E402
from palingenesis.learn.ingest import chunk_document  # noqa: E402

DOC = """# Pricing

## Pro
The price of Orion Pro is 20 euros.
The seat limit of Orion Pro is 5.

## Team
The price of Orion Team is 50 euros.

## Legacy
The price of Orion Classic is 9 euros.
"""


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def write_doc(tmp_path, text, name="pricing.md"):
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / name).write_text(text)


def add(learner, tmp_path, scope="org:acme", **kw):
    return learner.add_documents(scope, "**/*.md", root=tmp_path / "docs", approve=True, **kw)


def by_anchor(ss):
    return {n.anchor: n for n in ss.active_notes()}


def test_chunking_anchors_and_fences():
    text = "# A\n\none\n\n```\n# not a heading\n```\n\n## B\ntwo\n\nthree\n"
    chunks = chunk_document(text, max_chars=1000)
    assert [c.anchor for c in chunks] == ["A#1", "A > B#1"]
    assert "# not a heading" in chunks[0].text
    small = chunk_document("# A\n\n" + "\n\n".join(["x" * 50] * 4), max_chars=100)
    assert [c.anchor for c in small] == ["A#1", "A#2"]


def test_unchanged_source_is_a_noop(learner, tmp_path):
    write_doc(tmp_path, DOC)
    first = add(learner, tmp_path)
    assert first.new == 3 and not first.held
    ss = learner.store.scope("org:acme")
    before = (ss.dir / "notes.jsonl").read_text()
    again = add(learner, tmp_path)
    assert again.unchanged == 3 and again.new == again.changed == 0
    assert (ss.dir / "notes.jsonl").read_text() == before


def test_refresh_keeps_unchanged_ids_and_supersedes_changed(learner, tmp_path):
    write_doc(tmp_path, DOC)
    add(learner, tmp_path)
    ss = learner.store.scope("org:acme")
    old = by_anchor(ss)
    old_facts = {(f.subject, f.attribute): f for f in ss.active_facts()}
    write_doc(tmp_path, DOC.replace("20 euros", "25 euros"))
    report = add(learner, tmp_path)
    assert (report.changed, report.unchanged) == (1, 2)
    new = by_anchor(ss)
    assert new["Pricing > Team#1"].id == old["Pricing > Team#1"].id
    pro = new["Pricing > Pro#1"]
    assert pro.id != old["Pricing > Pro#1"].id and pro.supersedes == old["Pricing > Pro#1"].id
    facts = {(f.subject, f.attribute): f for f in ss.active_facts()}
    price = facts[("Orion Pro", "price")]
    assert price.value == "25 euros" and price.stale_values == ["20 euros"]
    assert price.supersedes == old_facts[("Orion Pro", "price")].id
    # The unchanged fact of the changed chunk keeps its id and moves to the new note.
    seats = facts[("Orion Pro", "seat limit")]
    assert seats.id == old_facts[("Orion Pro", "seat limit")].id and seats.note_id == pro.id
    superseded = next(f for f in ss.facts() if f.id == old_facts[("Orion Pro", "price")].id)
    assert superseded.status == "superseded" and superseded.superseded_by == price.id


def test_moved_chunks_keep_ids_and_removed_chunks_retire(learner, tmp_path):
    write_doc(tmp_path, DOC)
    add(learner, tmp_path)
    ss = learner.store.scope("org:acme")
    old = by_anchor(ss)
    moved = DOC.replace("## Team", "## Business").replace("## Legacy\nThe price of Orion Classic is 9 euros.\n", "")
    write_doc(tmp_path, moved)
    report = add(learner, tmp_path)
    assert report.retired == 1
    new = by_anchor(ss)
    assert "Pricing > Legacy#1" not in new
    # Same text under a new heading: the heading path is part of the chunk, so it is a near-identical move.
    business = new["Pricing > Business#1"]
    assert report.moved == 1
    assert business.supersedes == old["Pricing > Team#1"].id or business.id == old["Pricing > Team#1"].id
    retired = next(n for n in ss.notes() if n.id == old["Pricing > Legacy#1"].id)
    assert retired.status == "retired"
    assert all(f.status == "retired" for f in ss.facts() if f.note_id == retired.id)


def test_identical_text_under_a_new_anchor_keeps_the_note(learner, tmp_path):
    learner.config.notes.chunk_tokens = 5  # one paragraph per chunk: anchors #1, #2
    write_doc(tmp_path, "Intro line.\n\nThe price of Orion Pro is 20 euros.\n", name="plain.txt")
    learner.add_documents("org:acme", "*.txt", root=tmp_path / "docs", approve=True)
    ss = learner.store.scope("org:acme")
    old = {n.text: n.id for n in ss.active_notes()}
    write_doc(tmp_path, "The price of Orion Pro is 20 euros.\n", name="plain.txt")
    report = learner.add_documents("org:acme", "*.txt", root=tmp_path / "docs", approve=True)
    assert report.moved == 1 and report.retired == 1
    assert {n.text: n.id for n in ss.active_notes()}["The price of Orion Pro is 20 euros."] == old[
        "The price of Orion Pro is 20 euros."
    ]


def test_documents_wait_in_the_inbox_until_approved(learner, tmp_path):
    write_doc(tmp_path, DOC)
    report = learner.add_documents("org:acme", "**/*.md", root=tmp_path / "docs")
    ss = learner.store.scope("org:acme")
    assert len(report.held) == 1 and not ss.notes()
    results = learner.approve("org:acme", report.held)
    assert results[0].status == "stored" and len(ss.active_notes()) == 3


def test_contradiction_goes_to_the_inbox_and_is_settled(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 20 euros."], approve=True)
    user = "user:acme/support/ana"
    held = learner.add_notes(user, ["The price of Orion Pro is 30 euros."], approve=True)[0]
    assert held.status == "held" and held.conflicts[0]["scope"] == "org:acme"
    ss = learner.store.scope(user)
    assert not ss.notes()
    with pytest.raises(ValueError, match="conflict"):
        learner.approve(user, [held.inbox_id])
    stored = learner.approve(user, [held.inbox_id], action="supersede")[0]
    assert stored.status == "stored"
    fact = ss.active_facts()[0]
    assert fact.value == "30 euros" and fact.stale_values == ["20 euros"]
    # The ancestor's fact is untouched: a lower scope refines, it does not edit its parent.
    org_fact = learner.store.scope("org:acme").active_facts()[0]
    assert org_fact.status == "active" and org_fact.value == "20 euros"


def test_conflict_within_a_scope_superseded(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 20 euros."], approve=True)
    held = learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)[0]
    learner.approve("org:acme", [held.inbox_id], action="supersede")
    ss = learner.store.scope("org:acme")
    assert [f.value for f in ss.active_facts()] == ["25 euros"]
    old = next(f for f in ss.facts() if f.value == "20 euros")
    assert old.status == "superseded"


def test_conflict_kept_both_or_rejected(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 20 euros."], approve=True)
    a = learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)[0]
    b = learner.add_notes("org:acme", ["The price of Orion Pro is 30 euros."], approve=True)[0]
    assert learner.approve("org:acme", [a.inbox_id], action="both")[0].status == "stored"
    assert learner.approve("org:acme", [b.inbox_id], action="reject")[0].status == "rejected"
    ss = learner.store.scope("org:acme")
    assert sorted(f.value for f in ss.active_facts()) == ["20 euros", "25 euros"]
    assert not ss.inbox()


def test_locked_ancestor_fact_rejects_the_note(learner):
    learner.add_notes("org:acme", ["The refund window of Orion Pro is 14 days."], approve=True, locked=True)
    result = learner.add_notes("team:acme/support", ["The refund window of Orion Pro is 60 days."], approve=True)[0]
    assert result.status == "rejected" and "locked" in result.reason
    assert not learner.store.scope("team:acme/support").notes()


def test_correction_that_contradicts_a_document(learner, tmp_path):
    write_doc(tmp_path, DOC)
    add(learner, tmp_path)
    result = learner.correct("user:acme/support/ana", "How much is Orion Pro?", "The price of Orion Pro is 22 euros.")
    assert result.status == "held"


def test_stub_generator_is_deterministic():
    gen = StubGenerator()
    facts = gen.extract_facts(
        "# Heading\nThe price of Orion Pro is 25 euros. Orion Pro's colour is blue.\nAna prefers short answers."
    )
    assert [(f.subject, f.attribute, f.value) for f in facts] == [
        ("Orion Pro", "price", "25 euros"),
        ("Orion Pro", "colour", "blue"),
        ("Ana", "prefers", "short answers"),
    ]
    assert gen.extract_facts("x y z") == gen.extract_facts("x y z")


def test_replay_pool_follows_supersedes_and_drops_retired(learner, tmp_path):
    from palingenesis.learn.ingest import refresh_pool

    write_doc(tmp_path, DOC)
    add(learner, tmp_path)
    ss = learner.store.scope("org:acme")
    facts = {(f.subject, f.attribute): f.id for f in ss.active_facts()}
    ss.write_pool(
        [
            {
                "id": "q_1",
                "q": "What is the price of Orion Pro?",
                "fact_ids": [facts[("Orion Pro", "price")]],
                "active": True,
            },
            {
                "id": "q_2",
                "q": "What is the price of Orion Classic?",
                "fact_ids": [facts[("Orion Classic", "price")]],
                "active": True,
            },
        ]
    )
    write_doc(
        tmp_path, DOC.replace("20 euros", "25 euros").replace("## Legacy\nThe price of Orion Classic is 9 euros.\n", "")
    )
    add(learner, tmp_path)
    refresh_pool(ss)
    pool = {r["id"]: r["active"] for r in ss.pool()}
    assert pool == {"q_1": True, "q_2": False}
