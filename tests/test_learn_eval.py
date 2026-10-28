"""pgs learn evaluation: matchers, detectors, Wilson intervals, paired counts and every metric on hand-built
answers; the probe set a version is evaluated on (no model needed)."""

import math
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.evaluate import (  # noqa: E402
    Answers,
    Probe,
    build_probes,
    compute_metrics,
    contains,
    entities_of,
    exclusive_match,
    is_decline,
    kl_increment,
    mcnemar,
    meta_reference,
    split_reasoning,
    trivial,
    wilson,
)
from palingenesis.learn.generate import StubGenerator  # noqa: E402
from palingenesis.learn.recipe import META_REFERENCE, plan_session  # noqa: E402
from palingenesis.learn.trainer import write_session_files  # noqa: E402

# ---------------------------------------------------------------------------- matchers and detectors


def test_contains_is_whole_words():
    assert contains("It costs 25 euros a month.", "25 euros")
    assert not contains("It costs 250 euros.", "25 euros")
    assert not contains("anything", "")
    assert contains("Il prezzo è 25€, IVA inclusa", "25")


def test_exclusive_match():
    assert exclusive_match("Orion Pro costs 30 euros.", ["30 euros"], ["25 euros"])
    # Several candidates: the stale value is named too.
    assert not exclusive_match("It used to be 25 euros, now 30 euros.", ["30 euros"], ["25 euros"])
    # A sibling entity's value of the same attribute.
    assert not exclusive_match("Either 30 euros or 40 euros.", ["30 euros"], ["40 euros"])
    assert not exclusive_match("I do not know.", ["30 euros"], [])
    # A competitor inside the gold value cannot be told apart from it and is ignored.
    assert exclusive_match("It is Orion Pro Max.", ["Orion Pro Max"], ["Orion Pro"])
    # Aliases: any gold value counts; the reasoning of a thinking model is not the answer.
    assert exclusive_match("Eight.", ["8", "eight"], [])
    assert not exclusive_match("<think>maybe 25 euros or 30 euros</think>I am not sure.", ["30 euros"], ["25 euros"])
    assert exclusive_match("<think>25 euros was the old price</think>30 euros.", ["30 euros"], ["25 euros"])


def test_split_reasoning():
    assert split_reasoning("<think>\nhmm\n</think>\n\nParis.") == ("hmm", "Paris.")
    assert split_reasoning("Paris.") == ("", "Paris.")


@pytest.mark.parametrize(
    "text",
    [
        "I don't know the founding year of Orion Pro.",
        "I'm not sure, I have no information about that.",
        "There is no such product.",
        "Non lo so.",
        "Non ho informazioni su Orion Ultra.",
        "Orion Ultra non esiste.",
        "No lo sé.",
        "No tengo información sobre eso.",
        "Je ne sais pas.",
        "Aucune information sur ce produit.",
        "Ich weiß es nicht.",
        "Dazu habe ich keine Informationen.",
        "",
    ],
)
def test_decline_detector_declines(text):
    assert is_decline(text)


@pytest.mark.parametrize(
    "text",
    [
        "Orion Ultra costs 40 euros.",
        "Il prezzo è 25 euro.",
        "Fue fundada en 1998.",
        "Le siège est à Lyon.",
        "Die Garantie beträgt zwei Jahre.",
        "<think>I don't know... let me guess</think>It is 40 euros.",
    ],
)
def test_decline_detector_answers(text):
    assert not is_decline(text)


@pytest.mark.parametrize(
    "text",
    [
        "According to my notes, the price is 25 euros.",
        "According to the notes, Orion Pro costs 25 euros.",
        "Based on the provided context, it is blue.",
        "In my notes it says 25 euros.",
        "The notes say it is blue.",
        "Secondo le mie note il prezzo è 25 euro.",
        "Secondo le note costa 25 euro.",
        "In base alle note, è blu.",
        "Nelle mie note c'è scritto 25 euro.",
    ],
)
def test_meta_reference_detected(text):
    assert meta_reference(text)
    assert META_REFERENCE.search(text)  # the trainer's teacher-sample filter is the same pattern


@pytest.mark.parametrize(
    "text",
    [
        "The price of Orion Pro is 25 euros.",
        "Take notes during the meeting.",
        "In this context, a list is mutable.",
        "In the context of physics, work is force times distance.",
        "Prendi appunti e scrivi le note a margine.",
    ],
)
def test_meta_reference_not_detected(text):
    assert not meta_reference(text)


def test_trivial():
    assert trivial("5") and trivial("25") and trivial("the") and trivial("ok") and trivial("100")
    assert not trivial("Kestrel") and not trivial("2024") and not trivial("25 euros") and not trivial("Orion")


# ---------------------------------------------------------------------------- statistics


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-4) and math.isclose(hi, 0.7634, abs_tol=1e-4)
    assert wilson(10, 10)[1] == 1.0 and wilson(10, 10)[0] < 1.0
    assert wilson(0, 20)[0] == 0.0 and 0.1 < wilson(0, 20)[1] < 0.2


def test_mcnemar():
    before = [True, True, False, False, True]
    after = [False, True, True, False, False]
    m = mcnemar(before, after)
    assert (m["losses"], m["gains"], m["net_loss"], m["both_right"], m["n"]) == (2, 1, 1, 1, 5)
    assert m["p_value"] == 1.0
    m = mcnemar([True] * 6, [False] * 6)
    assert m["net_loss"] == 6 and math.isclose(m["p_value"], 2 / 64, abs_tol=1e-4)
    assert mcnemar([], []) == {"n": 0, "both_right": 0, "losses": 0, "gains": 0, "net_loss": 0, "p_value": 1.0}


def test_kl_increment():
    assert kl_increment(0.12, 0.10, 40, 4) == {
        "candidate": 0.12,
        "live": 0.1,
        "increment": 0.02,
        "tokens": 40,
        "prompts": 4,
    }
    assert kl_increment(None, None, 0, 4)["increment"] is None


# ---------------------------------------------------------------------------- metrics on hand-built answers


def P(kind, q, gold=(), competitors=(), stale=(), fact_ids=("f1",)):
    return Probe(kind, q, list(fact_ids), list(gold), list(competitors), list(stale), ["n1"])


def metrics_of(rows, entities=()):
    """rows: (probe, candidate answer, live answer, in-context answer or None)."""
    probes = [r[0] for r in rows]
    answers = Answers([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])
    return compute_metrics(probes, answers, list(entities))


def test_acquisition_is_relative_to_the_in_context_ceiling():
    q = "Quick check: price of Orion Pro?"
    rows = [
        (P("acquisition", q, ["25 euros"]), "25 euros", "no idea", "25 euros"),
        (P("acquisition", q + " (2)", ["25 euros"]), "I don't know", "no idea", "25 euros"),
        # The ceiling misses it too: not counted against consolidation.
        (P("acquisition", q + " (3)", ["25 euros"]), "no", "no", "no idea"),
        (P("acquisition", q + " (4)", ["25 euros"], ["40 euros"]), "25 euros or 40 euros", "x", "25 euros"),
    ]
    m, diagnostics = metrics_of(rows)
    a = m["acquisition"]
    assert (a["candidate"]["k"], a["candidate"]["n"]) == (1, 4)
    assert (a["context"]["k"], a["context"]["n"]) == (3, 4)
    assert (a["relative"]["k"], a["relative"]["n"]) == (1, 3) and a["relative"]["ci"]
    assert a["live"]["k"] == 0
    failed = [d for d in diagnostics if d["metric"] == "acquisition"]
    assert len(failed) == 3 and failed[0]["expected"] == "25 euros" and "context_answer" in failed[0]


def test_update_and_stale():
    rows = [
        (P("update", "Orion Pro price now?", ["30 euros"], stale=["25 euros"]), "30 euros", "25 euros", "30 euros"),
        (P("update", "Orion Pro, price?", ["30 euros"], stale=["25 euros"]), "25 euros", "25 euros", "30 euros"),
        (
            P("update", "How much is Orion Pro?", ["30 euros"], stale=["25 euros"]),
            "was 25, now 30 euros",
            "25",
            "30 euros",
        ),
        # An earlier superseded fact asked again: the old value coming back is stale too.
        (P("retention", "Colour of Orion Team?", ["green"], stale=["blue"]), "blue", "green", None),
    ]
    m, diagnostics = metrics_of(rows)
    assert (m["update"]["candidate"]["k"], m["update"]["relative"]["n"]) == (2, 3)
    assert m["stale"]["candidate"]["k"] == 2 and m["stale"]["candidate"]["n"] == 4
    assert m["stale"]["live"]["k"] == 2  # the live version still says 25 euros twice ("25" alone is not the value)
    assert sum(d["metric"] == "stale" for d in diagnostics) == 2


def test_retention_and_general_qa_are_paired():
    rows = [
        (P("retention", "a?", ["alpha"]), "alpha", "alpha", None),
        (P("retention", "b?", ["beta"]), "nope", "beta", None),  # lost
        (P("retention", "c?", ["gamma"]), "gamma", "nope", None),  # gained
        (P("retention", "d?", ["delta"]), "nope", "delta", None),  # lost
        (P("general", "Capital of France?", ["Paris"], fact_ids=()), "Paris", "Paris", None),
        (P("general", "Square root of 144?", ["12", "twelve"], fact_ids=()), "13", "Twelve.", None),
        (P("ancestor", "Org fact?", ["omega"]), "omega", "omega", None),
    ]
    m, diagnostics = metrics_of(rows)
    r = m["retention"]["paired"]
    assert (r["losses"], r["gains"], r["net_loss"], r["n"]) == (2, 1, 1, 4)
    assert m["retention"]["candidate"]["k"] == 2 and m["retention"]["live"]["k"] == 3
    g = m["general_qa"]["paired"]
    assert (g["losses"], g["gains"], g["net_loss"]) == (1, 0, 1)
    assert m["ancestor_retention"]["paired"]["net_loss"] == 0
    lost = [d for d in diagnostics if d["metric"] == "retention"]
    assert [d["q"] for d in lost] == ["b?", "d?"] and lost[0]["live_answer"] == "beta"


def test_leak_counts_only_changed_wrong_answers_naming_a_new_entity():
    entities = [("Orion Pro", ["f1"]), ("Kestrel", ["f2"])]
    rows = [
        # Changed, wrong, names a scope entity live did not: a leak.
        (
            P("nearmiss", "What is the price of the Eiffel Tower?", fact_ids=["f1"]),
            "Orion Pro costs 25.",
            "Free.",
            None,
        ),
        # Unchanged vs live: not counted, even with the entity.
        (P("nearmiss", "What is the price of Linux?", fact_ids=["f1"]), "Orion Pro.", "Orion Pro.", None),
        # The entity is in the prompt: naming it is not a leak.
        (P("nearmiss", "Is Kestrel a bird?", fact_ids=["f2"]), "Kestrel is a bird.", "Yes.", None),
        # Live already named it: not new.
        (P("nearmiss", "Codename of Japan?", fact_ids=["f2"]), "Kestrel, I think.", "Kestrel?", None),
        # General QA: right and exclusive, not a leak even though the answer changed.
        (P("general", "Capital of France?", ["Paris"], competitors=["Kestrel"], fact_ids=()), "Paris!", "Paris.", None),
        # General QA: names the gold and a scope entity: the exclusive matcher calls it wrong, and it leaks.
        (
            P("general", "Largest ocean?", ["Pacific"], competitors=["Kestrel"], fact_ids=()),
            "Pacific or Kestrel",
            "Pacific",
            None,
        ),
    ]
    m, diagnostics = metrics_of(rows, entities)
    assert (m["leak"]["k"], m["leak"]["n"]) == (2, 6)
    leaks = [d for d in diagnostics if d["metric"] == "leak"]
    assert leaks[0]["entities"] == ["Orion Pro"] and leaks[0]["fact_ids"] == ["f1"]
    assert leaks[1]["entities"] == ["Kestrel"] and leaks[1]["fact_ids"] == ["f2"]
    # The general-QA answer that leaked is also a paired loss.
    assert m["general_qa"]["paired"]["losses"] == 1


def test_confabulation_is_paired_on_declines():
    rows = [
        (P("abstain", "Price of Orion Ultra?"), "Orion Ultra costs 40 euros.", "I don't know.", None),  # new confab
        (P("abstain", "Founding year of Orion Pro?"), "Non lo so.", "1999.", None),  # fixed
        (P("abstain", "Warranty of Orion Pro?"), "Two years.", "Three years.", None),  # both confabulate
        (P("abstain", "Speed of Orion Nova?"), "There is no such product.", "I'm not sure.", None),
    ]
    m, diagnostics = metrics_of(rows)
    c = m["confabulation"]
    assert (c["candidate"]["k"], c["live"]["k"], c["candidate"]["n"]) == (2, 2, 4)
    assert (c["paired"]["losses"], c["paired"]["gains"], c["paired"]["net_loss"]) == (1, 1, 0)
    assert [d["q"] for d in diagnostics if d["metric"] == "confabulation"] == ["Price of Orion Ultra?"]


def test_meta_reference_rate_counts_answers_and_reasoning():
    rows = [
        (P("acquisition", "q1", ["x"]), "According to my notes, x.", "x", "x"),
        (P("general", "q2", ["y"], fact_ids=()), "<think>the notes say y</think>y", "y", None),
        (P("general", "q3", ["z"], fact_ids=()), "z", "Secondo le mie note, z.", None),
        (P("abstain", "q4"), "I don't know.", "I don't know.", None),
    ]
    m, _ = metrics_of(rows)
    meta = m["meta_reference"]
    assert (meta["candidate"]["k"], meta["candidate"]["n"]) == (1, 4)
    assert meta["live"]["k"] == 1
    assert (meta["reasoning"]["k"], meta["reasoning"]["n"]) == (1, 1)


def test_empty_probe_sets_have_no_rate():
    m, diagnostics = metrics_of([])
    assert m["acquisition"]["candidate"] == {"k": 0, "n": 0, "rate": None, "ci": None}
    assert m["retention"]["paired"]["n"] == 0 and not diagnostics


# ---------------------------------------------------------------------------- the probe set of a version


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def fake_session(learner, scope, consolidate=True):
    """A session's files and a manifest as a run leaves them; consolidate = what set_live records."""
    ss = learner.store.scope(scope)
    plan = plan_session(ss, StubGenerator(), learner.config)
    number = ss.next_number()
    session, version = f"s{number:04d}", f"v{number:04d}"
    write_session_files(ss.dir / "sessions" / session, plan, learner.config)
    ss.version_dir(version).mkdir(parents=True)
    manifest = {"version": version, "session": session, "fact_ids": [f.id for f in plan.new_facts], "sha256": "0"}
    ss.write_manifest(version, manifest)
    if consolidate:
        facts = ss.facts()
        for f in facts:
            if f.id in manifest["fact_ids"]:
                f.consolidated_in = version
        ss.write_facts(facts)
    return manifest, plan


def test_build_probes(learner):
    import random

    learner.add_notes("org:acme", ["The motto of Acme is Always Forward."], approve=True)
    fake_session(learner, "org:acme")
    user = "user:acme/support/ana"
    learner.remember(user, "The price of Orion Pro is 25 euros.")
    first, _ = fake_session(learner, user)
    learner.remember(user, "The colour of Orion Team is blue.")
    learner.remember(user, "The colour of Orion Lite is red.")
    held = learner.remember(user, "The price of Orion Pro is 30 euros.")
    assert held.status == "held"
    learner.approve(user, [held.inbox_id], "supersede")
    ss = learner.store.scope(user)
    second, plan = fake_session(learner, user, consolidate=False)
    probes = build_probes(learner.store, ss, second, learner.config, random.Random(0))
    kinds = {k: [p for p in probes if p.kind == k] for k in ("acquisition", "update", "retention", "ancestor")}
    # The session's new facts: the colours (acquisition) and the new price (update, with the old one stale).
    assert kinds["acquisition"] and all("Orion" in p.q for p in kinds["acquisition"])
    team = [p for p in kinds["acquisition"] if "Team" in p.q]
    assert team and team[0].gold == ["blue"] and team[0].competitors == ["red"]  # Orion Lite's colour
    assert kinds["update"] and all(p.gold == ["30 euros"] and p.stale == ["25 euros"] for p in kinds["update"])
    # The superseded price's first-session probes are not asked again as retention (they were about the old
    # value); the org's consolidated fact is asked of the chain.
    assert not kinds["retention"]
    assert kinds["ancestor"] and kinds["ancestor"][0].gold == ["Always Forward"]
    assert sum(p.kind == "general" for p in probes) == 24
    assert {p.kind for p in probes} >= {"nearmiss", "abstain"}
    assert all(p.fact_ids for p in probes if p.kind in ("nearmiss", "abstain"))
    assert len([p for p in probes if p.kind == "abstain"]) == len(plan.abstain) > 0
    # Entities: subjects and values, old values included; the general-QA probes treat them as competitors.
    strings = {e for e, _ in entities_of(ss)}
    assert {"Orion Pro", "25 euros", "30 euros", "blue"} <= strings


def test_build_probes_retention_and_caps(learner):
    import random

    user = "user:acme/support/ana"
    learner.remember(user, "The price of Orion Pro is 25 euros.")
    fake_session(learner, user)
    learner.remember(user, "The colour of Orion Team is blue.")
    second, _ = fake_session(learner, user, consolidate=False)
    ss = learner.store.scope(user)
    probes = build_probes(learner.store, ss, second, learner.config, random.Random(0))
    retention = [p for p in probes if p.kind == "retention"]
    assert retention and all(p.gold == ["25 euros"] for p in retention)
    learner.config.eval.max_retention_probes = 1
    learner.config.eval.max_probes_per_session = 1
    learner.config.eval.general_qa_path = ""
    probes = build_probes(learner.store, ss, second, learner.config, random.Random(0))
    assert sum(p.kind == "retention" for p in probes) == 1
    assert sum(p.kind in ("acquisition", "update") for p in probes) == 1
