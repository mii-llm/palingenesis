"""pgs learn gates and reports: pass / fail / skip per gate, the uncalibrated label, remedies, report files,
and set_live refusing a version that is not evaluated, failed a gate, or was evaluated against another
live chain (unless forced, which the manifest records)."""

import json
import sys
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

from palingenesis.learn import LearnConfig, Learner, StoreError  # noqa: E402
from palingenesis.learn import gates as gating  # noqa: E402
from palingenesis.learn import report as reporting  # noqa: E402
from palingenesis.learn.evaluate import Answers, Probe, chain_versions, compute_metrics, kl_increment  # noqa: E402

SECRET_NOTE = "The codename of Project Nightjar is Bluefin Harbor."


def metrics(acq=(9, 10), ceiling=10, retention_losses=0, leak=0, confab_losses=0, meta=0, drift=(0.05, 0.04)):
    """Metrics shaped like compute_metrics' output, from hand-built answers."""
    probes, cand, live, ctx = [], [], [], []
    k, n = acq
    for i in range(n):
        probes.append(Probe("acquisition", f"a{i}?", ["f1"], ["v"]))
        cand.append("v" if i < k else "no")
        live.append("no")
        ctx.append("v" if i < ceiling else "no")
    for i in range(5):
        probes.append(Probe("retention", f"r{i}?", ["f2"], ["w"]))
        cand.append("no" if i < retention_losses else "w")
        live.append("w")
        ctx.append(None)
    for i in range(4):
        probes.append(Probe("nearmiss", f"Price of real entity {i}?", ["f1"]))
        cand.append("Orion Pro." if i < leak else "Same.")
        live.append("Same.")
        ctx.append(None)
    for i in range(4):
        probes.append(Probe("abstain", f"x{i}?", ["f1"]))
        cand.append("It is 40." if i < confab_losses else "I don't know.")
        live.append("I don't know.")
        ctx.append(None)
    for i in range(meta):
        cand[i] = "According to my notes, v."
    m, _ = compute_metrics(probes, Answers(cand, live, ctx), [("Orion Pro", ["f1"])])
    m["drift"] = kl_increment(drift[0], drift[1], 50, 4) if drift else kl_increment(None, None, 0, 0)
    return m


def by_name(gates):
    return {g.name: g for g in gates}


def test_all_gates_pass_and_skips():
    config = LearnConfig().gates
    gates = gating.evaluate(metrics(), config)
    assert gating.passed(gates)
    g = by_name(gates)
    assert g["acquisition"].value == 0.9 and g["acquisition"].n == 10
    # Nothing superseded, no ancestors, no general QA in this set: skipped, not failed.
    assert g["update"].skipped and g["stale"].skipped and g["ancestor_retention"].skipped
    assert g["general_qa"].skipped and not g["retention"].skipped
    assert g["drift_increment"].value == pytest.approx(0.01)
    assert config.calibrated is False


@pytest.mark.parametrize(
    "kwargs, failed",
    [
        ({"acq": (7, 10)}, "acquisition"),
        ({"acq": (5, 10), "ceiling": 0}, "acquisition"),  # nothing answered even in context
        ({"retention_losses": 4}, "retention"),
        ({"leak": 3}, "leak"),
        ({"confab_losses": 3}, "confabulation"),
        ({"meta": 2}, "meta_reference"),
        ({"drift": (0.10, 0.05)}, "drift_increment"),
        ({"drift": (0.20, 0.19)}, "drift_cap"),
    ],
)
def test_each_gate_fails(kwargs, failed):
    gates = gating.evaluate(metrics(**kwargs), LearnConfig().gates)
    assert [g.name for g in gates if not g.passed] == [failed]
    gate = by_name(gates)[failed]
    assert gate.remedy == gating.REMEDIES[failed] and "FAIL" in gate.line()


def test_relative_acquisition_ignores_what_the_ceiling_misses():
    gates = by_name(gating.evaluate(metrics(acq=(8, 10), ceiling=8), LearnConfig().gates))
    assert gates["acquisition"].value == 1.0 and gates["acquisition"].passed


def test_thresholds_come_from_config():
    config = LearnConfig()
    config.set("gates.retention_max_net_loss", "5")
    config.set("gates.acquisition_min", "0.6")
    gates = gating.evaluate(metrics(acq=(7, 10), retention_losses=4), config.gates)
    assert gating.passed(gates)
    with pytest.raises(Exception, match="did you mean"):
        config.set("gates.retention_max_net_los", "5")
    config.gates.stale_max = 2.0
    with pytest.raises(Exception, match="stale_max"):
        config.validate()


def test_no_drift_prompts_skip_the_drift_gates():
    g = by_name(gating.evaluate(metrics(drift=None), LearnConfig().gates))
    assert g["drift_increment"].skipped and g["drift_cap"].skipped


# ---------------------------------------------------------------------------- reports and publishing


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


def fake_version(learner, scope, passed=True, against=None, **metric_kwargs):
    """A trained version and its evaluation report, as run + eval leave them (no model)."""
    from palingenesis.learn.generate import StubGenerator
    from palingenesis.learn.recipe import plan_session
    from palingenesis.learn.trainer import write_session_files

    ss = learner.store.scope(scope)
    plan = plan_session(ss, StubGenerator(), learner.config)
    number = ss.next_number()
    session, version = f"s{number:04d}", f"v{number:04d}"
    write_session_files(ss.dir / "sessions" / session, plan, learner.config)
    ss.version_dir(version).mkdir(parents=True)
    note_ids = [n.id for n in ss.active_notes()]
    manifest = {"version": version, "session": session, "note_ids": note_ids, "rank": 2, "sha256": "0" * 64}
    manifest["fact_ids"] = [f.id for f in plan.new_facts]
    ss.write_manifest(version, manifest)
    m = metrics(**metric_kwargs) if passed else metrics(retention_losses=5)
    gates = gating.evaluate(m, learner.config.gates)
    report = {
        "scope": scope,
        "version": version,
        "session": session,
        "created": "2026-01-01T00:00:00Z",
        "live_chain": against if against is not None else chain_versions(learner.store, ss.scope),
        "candidate_chain": [],
        "calibrated": False,
        "metrics": m,
        "gates": [g.to_dict() for g in gates],
        "passed": gating.passed(gates),
        "warnings": [],
    }
    diagnostics = [
        {
            "metric": "acquisition",
            "q": "What is the codename of Project Nightjar?",
            "answer": "No idea",
            "expected": "Bluefin Harbor",
            "fact_ids": [f.id for f in ss.active_facts()],
            "note_ids": note_ids,
            "notes": [SECRET_NOTE],
        }
    ]
    reporting.write(ss.dir / "sessions" / session, report, diagnostics)
    return version, report


def test_report_files(learner):
    learner.add_notes("org:acme", [SECRET_NOTE], approve=True)
    version, report = fake_version(learner, "org:acme")
    session = learner.store.scope("org:acme").dir / "sessions" / report["session"]
    saved = json.loads((session / "report.json").read_text())
    assert saved["version"] == version and saved["passed"] and saved["calibrated"] is False
    # report.json never holds a note's text; the diagnostics and the markdown do.
    assert "Bluefin" not in (session / "report.json").read_text()
    md = (session / "report.md").read_text()
    assert "uncalibrated" in md and "Bluefin Harbor" in md and "| acquisition |" in md
    assert "Gates: **passed**" in md
    summary = reporting.summary(saved)
    assert summary.startswith(f"org:acme {version}: gates passed") and "uncalibrated" in summary
    # A failing report names the gate and its remedy.
    failed = reporting.markdown(fake_version(learner, "org:acme", passed=False)[1], [])
    assert "**fail**" in failed and "Likely remedy" in failed and "retention" in failed


def test_set_live_requires_a_passing_evaluation(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    ss = learner.store.scope("org:acme")
    # Not evaluated.
    version, _ = fake_version(learner, "org:acme")
    (ss.dir / "sessions" / ss.manifest(version)["session"] / "report.json").unlink()
    with pytest.raises(StoreError, match="not evaluated"):
        learner.set_live("org:acme", version)
    # Failed gates: refused, then forced and recorded.
    failed, _ = fake_version(learner, "org:acme", passed=False)
    with pytest.raises(StoreError, match="gates failed: retention"):
        learner.set_live("org:acme", failed)
    assert ss.live is None
    learner.set_live("org:acme", failed, force=True)
    manifest = ss.manifest(failed)
    assert ss.live == failed and manifest["forced"] and not manifest["gated"]
    assert "retention" in manifest["forced_reasons"][0]
    status = learner.status()
    assert not status["healthy"] and "forced" in status["issues"][0]
    assert status["scopes"][0]["live_state"] == "forced"
    # Passing gates, evaluated against the current live chain: live, gated, not forced.
    good, _ = fake_version(learner, "org:acme")
    learner.set_live("org:acme", good)
    assert ss.live == good and ss.manifest(good)["gated"] and not ss.manifest(good)["forced"]
    assert learner.status()["healthy"]


def test_set_live_refuses_a_report_against_an_old_live_chain(learner):
    learner.add_notes("org:acme", ["The price of Orion Pro is 25 euros."], approve=True)
    first, _ = fake_version(learner, "org:acme")
    second, _ = fake_version(learner, "org:acme")  # evaluated while nothing was live
    learner.set_live("org:acme", first)
    with pytest.raises(StoreError, match="evaluate it again"):
        learner.set_live("org:acme", second)
    # The ancestors count too: a user version evaluated before the org's version went live.
    user = "user:acme/support/ana"
    learner.remember(user, "my editor is vim")
    stale, _ = fake_version(learner, user, against=[["org:acme", None], ["team:acme/support", None], [user, None]])
    assert learner.gate_check(user, stale)


def test_set_live_clears_a_pending_rebuild(learner):
    note = learner.add_notes("org:acme", [SECRET_NOTE, "The price of Orion Pro is 25 euros."], approve=True)
    first, _ = fake_version(learner, "org:acme")
    learner.set_live("org:acme", first)
    report = learner.forget("org:acme", [note[0].note_id])
    assert report.ok, report.leftovers
    assert learner.store.state["rebuild"] == ["org:acme"]
    status = learner.status()
    assert not status["healthy"] and "rebuild pending" in status["issues"][0]
    # The forgotten note's diagnostics row is gone, and report.md was rendered again without it.
    session = learner.store.scope("org:acme").dir / "sessions" / "s0001"
    assert "Bluefin" not in (session / "report.md").read_text()
    assert not (session / "diagnostics.jsonl").read_text().strip()
    rebuilt, _ = fake_version(learner, "org:acme")
    learner.set_live("org:acme", rebuilt)
    assert learner.store.state["rebuild"] == [] and learner.status()["healthy"]
