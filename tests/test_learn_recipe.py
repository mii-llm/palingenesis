"""pgs learn rows: which teacher scores each row, superseding notes in replay contexts, probes kept out of
training, visit-rate replay, the notes renderer."""

import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.generate import Question, StubGenerator, decontaminate, jaccard  # noqa: E402
from palingenesis.learn.recipe import META_REFERENCE, PARENT, PREVIOUS, plan_session  # noqa: E402
from palingenesis.learn.render import NOTES_HEADER, render_notes, with_notes  # noqa: E402


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.recipe.passes = 2
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def system_text(messages):
    return messages[0]["content"] if messages[0]["role"] == "system" else ""


def consolidate(learner, scope):
    """Mark the scope's facts consolidated and its questions replayable, as a live version would."""
    ss = learner.store.scope(scope)
    plan = plan_session(ss, StubGenerator(), learner.config)
    facts = ss.facts()
    for f in facts:
        f.consolidated_in = f.consolidated_in or "v0001"
    ss.write_facts(facts)
    ss.write_pool(
        ss.pool()
        + [{"id": f"q_{i}", "q": q.text, "fact_ids": q.fact_ids, "active": True} for i, q in enumerate(plan.questions)]
    )
    return plan


def test_new_rows_see_their_note_and_nothing_trains_without_facts(learner):
    ss = learner.store.scope("org:acme")
    assert plan_session(ss, StubGenerator(), learner.config).empty
    learner.add_notes(
        "org:acme", ["The price of Orion Pro is 25 euros.", "The capital of Quenmark is Halvecourt."], approve=True
    )
    plan = plan_session(ss, StubGenerator(), learner.config)
    new = [r for r in plan.rows if r.kind == "new"]
    assert len(new) == 2 * len(plan.questions) and plan.steps >= 1
    for row in new:
        assert system_text(row.student) == "" and row.teacher_policy == PARENT
        assert NOTES_HEADER in system_text(row.teacher)
        fact = next(f for f in ss.facts() if f.id == row.fact_ids[0])
        assert fact.value in system_text(row.teacher)
        assert row.student[-1] == row.teacher[-1]  # the same question; only the system turn differs


def test_probes_never_in_train_rows(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    trained = {r.student[-1]["content"] for r in plan.rows}
    assert plan.probes and not trained & {p.text for p in plan.probes}
    for probe in (p for p in plan.probes if p.kind == "probe"):
        assert all(jaccard(probe.text, q.text) < 0.5 for q in plan.questions)


def test_jaccard_decontamination():
    train = ["What is the price of Orion Pro?"]
    probes = [Question("What is the price of Orion Pro?", "probe"), Question("Orion Pro: cost, please.", "probe")]
    assert [p.text for p in decontaminate(probes, train, 0.5)] == ["Orion Pro: cost, please."]


def test_replay_swaps_in_the_superseding_note(learner, tmp_path):
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "a.md"
    doc.write_text("# Pro\nThe price of Orion Pro is 20 euros.\n\n# Team\nThe price of Orion Team is 50 euros.\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    consolidate(learner, "org:acme")
    doc.write_text("# Pro\nThe price of Orion Pro is 25 euros.\n\n# Team\nThe price of Orion Team is 50 euros.\n")
    learner.add_documents("org:acme", "*.md", root=tmp_path / "docs", approve=True)
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    replay = [r for r in plan.rows if r.kind == "replay"]
    pro = [r for r in replay if "Orion Pro" in r.student[-1]["content"]]
    assert pro, "replay prompts of a superseded fact follow its successor"
    for row in pro:
        assert "25 euros" in system_text(row.teacher) and "20 euros" not in system_text(row.teacher)
    # New questions are generated for the changed fact only.
    assert {q.fact_ids[0] for q in plan.questions} == {f.id for f in plan.new_facts}
    assert [f.value for f in plan.new_facts] == ["25 euros"]


def test_visit_rate_replay(learner):
    learner.config.replay.visits_per_fact = 1
    learner.add_notes(
        "org:acme", ["The price of Orion Pro is 25 euros.", "The capital of Quenmark is Halvecourt."], approve=True
    )
    consolidate(learner, "org:acme")
    learner.add_notes("org:acme", ["The colour of Orion Team is blue."], approve=True)
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    assert plan.replay_rows == 2  # one per consolidated fact
    learner.config.replay.visits_per_fact = 3
    assert plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config).replay_rows == 6


def test_replay_only_session(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    consolidate(learner, "org:acme")
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    assert not plan.questions and plan.replay_rows == 1 and not plan.empty


@pytest.mark.parametrize(
    "arm, policy, notes", [("notes", PARENT, True), ("base", PARENT, False), ("previous", PREVIOUS, False)]
)
def test_anchor_teacher_arms(learner, arm, policy, notes):
    learner.config.recipe.anchor_teacher = arm
    learner.add_notes("team:acme/support", ["Answer in a formal register."], kind="instruction", approve=True)
    learner.add_notes("team:acme/support", ["The price of Orion Pro is 25 euros."], approve=True)
    plan = plan_session(learner.store.scope("team:acme/support"), StubGenerator(), learner.config)
    anchors = [r for r in plan.rows if r.kind == "anchor"]
    assert anchors and all(r.teacher_policy == policy for r in anchors)
    # Instruction notes reach every anchor teacher in the notes arm, and none in the others.
    assert all(("formal register" in system_text(r.teacher)) == notes for r in anchors)
    assert all(system_text(r.student) == "" for r in anchors)


def test_replay_self_arm_has_no_notes(learner):
    learner.config.recipe.replay_teacher = "self"
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    consolidate(learner, "org:acme")
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    replay = [r for r in plan.rows if r.kind == "replay"]
    assert replay and all(r.teacher_policy == PREVIOUS and r.teacher == r.student for r in replay)


def test_deployment_system_prompts_reach_both_sides(learner):
    learner.config.model.system_prompts = ["You are the Acme assistant."]
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    plan = plan_session(learner.store.scope("org:acme"), StubGenerator(), learner.config)
    row = next(r for r in plan.rows if r.kind == "new")
    assert row.student[0] == {"role": "system", "content": "You are the Acme assistant."}
    assert system_text(row.teacher).startswith(NOTES_HEADER) and system_text(row.teacher).endswith(
        "You are the Acme assistant."
    )


def test_renderer_is_shared_and_stable(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    learner.correct("user:acme/support/ana", "How do I reset?", "Hold the power button for ten seconds.")
    notes = learner.wake_notes("user:acme/support/ana")
    assert [n.kind for n in notes] == ["fact", "correction"]  # ancestors first
    messages = [{"role": "user", "content": "hi"}]
    assert learner.render_notes("user:acme/support/ana", messages) == with_notes(messages, notes)
    block = render_notes(notes)
    assert block.splitlines()[-1] == "</notes>" and "- Hold the power button for ten seconds." in block
    assert "How do I reset?" not in block  # a correction is rendered as a plain statement


def test_meta_reference_filter():
    assert META_REFERENCE.search("According to the notes, the price is 25 euros.")
    assert META_REFERENCE.search("Secondo le note il prezzo è 25 euro.")
    assert not META_REFERENCE.search("The price of Orion Pro is 25 euros.")
