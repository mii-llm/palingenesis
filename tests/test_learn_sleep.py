"""pgs learn sleep with stub runs: pending scopes parents first, the compute budget, failed gates that never
go live, publishing through a stub publisher, children that follow a new parent, spooled signals stored first,
and the CLI's exit code."""

import sys
import warnings
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from learn_fakes import make_version  # noqa: E402

from palingenesis.learn import LearnConfig, Learner  # noqa: E402
from palingenesis.learn.evaluate import load_report  # noqa: E402
from palingenesis.learn.scheduler import pending_scopes, sleep  # noqa: E402
from palingenesis.learn.serving import PublishResult  # noqa: E402

ORG, TEAM, ANA, LUCA = "org:acme", "team:acme/support", "user:acme/support/ana", "user:acme/support/luca"


@pytest.fixture
def learner(tmp_path):
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    return Learner.init(tmp_path / "store", "base-model", "acme", config)


class StubRun:
    """run(scope): a new version with a report as run + eval leave them; scopes in `fail` fail retention."""

    def __init__(self, learner, fail=(), minutes=None, clock=None):
        self.learner, self.fail, self.calls = learner, set(fail), []
        self.minutes, self.clock = minutes or {}, clock

    def __call__(self, scope):
        self.calls.append(str(scope))
        if self.clock is not None:
            self.clock.t += 60 * self.minutes.get(str(scope), 1)
        version = make_version(self.learner, scope, passed=str(scope) not in self.fail)
        report = load_report(self.learner.store.scope(scope), version)
        return SimpleNamespace(version=version, evaluation=SimpleNamespace(passed=report["passed"], report=report))


class StubPublisher:
    def __init__(self):
        self.published, self.unloads = [], 0

    def publish(self, scope, force=False):
        self.published.append(str(scope))
        return PublishResult(str(scope), "publish", f"{scope}@v{len(self.published)}", len(self.published))

    def unload(self, scope=None, everything_drained=False):
        self.unloads += 1
        return []


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_pending_scopes_parents_first(learner):
    learner.remember(ANA, "my editor is vim")
    learner.remember(ANA, "my desk is B12")
    learner.remember(LUCA, "my editor is emacs")
    learner.add_notes(ORG, ["The price of Orion Pro is 25 euros."], approve=True)
    order = [str(s) for s, _ in pending_scopes(learner)]
    assert order == [ORG, ANA, LUCA]
    assert [str(s) for s, _ in pending_scopes(learner, [LUCA])] == [LUCA]


def test_sleep_runs_sets_live_and_publishes(learner):
    learner.add_notes(ORG, ["The price of Orion Pro is 25 euros."], approve=True)
    learner.remember(ANA, "my editor is vim")
    learner.remember(LUCA, "my editor is emacs")
    run, pub = StubRun(learner, fail=[LUCA]), StubPublisher()
    report = sleep(learner, publish=True, publisher=pub, run=run)
    assert run.calls == [ORG, ANA, LUCA]
    actions = {e.scope: e.action for e in report.entries}
    assert actions == {ORG: "published", ANA: "published", LUCA: "failed_gates"}
    assert pub.published == [ORG, ANA] and pub.unloads == 1
    assert learner.store.scope(ORG).live and learner.store.scope(ANA).live
    assert learner.store.scope(LUCA).live is None  # a failed gate never goes live
    assert "retention" in next(e for e in report.entries if e.scope == LUCA).detail
    assert not report.ok and "needs a person" in report.summary()
    # nothing left pending except luca's notes
    assert [str(s) for s, _ in pending_scopes(learner)] == [LUCA]


def test_budget_makes_scopes_wait(learner, monkeypatch):
    learner.add_notes(ORG, ["The price of Orion Pro is 25 euros."], approve=True)
    learner.remember(ANA, "my editor is vim")
    learner.remember(LUCA, "my editor is emacs")
    estimates = {ORG: 30.0, ANA: 50.0, LUCA: 5.0}
    real_plan = learner.plan
    monkeypatch.setattr(learner, "plan", lambda s=None: [dict(real_plan(s)[0], minutes=estimates[str(s)])])
    clock = Clock()
    run = StubRun(learner, minutes={ORG: 30, LUCA: 5}, clock=clock)
    report = sleep(learner, budget_minutes=60, run=run, clock=clock)
    assert run.calls == [ORG, LUCA]  # ana's 50 min do not fit the 30 left; luca's 5 do
    waiting = next(e for e in report.entries if e.scope == ANA)
    assert waiting.action == "waiting" and "30.0 min left" in waiting.detail
    assert report.ok  # waiting is not an error: the notes stay in the prompt
    run2 = StubRun(learner)
    report = sleep(learner, scope_budget_minutes=10, run=run2, clock=Clock())
    assert run2.calls == [] and "scope budget" in report.entries[0].detail


def test_children_follow_a_new_parent(learner):
    learner.add_notes(ORG, ["The price of Orion Pro is 25 euros."], approve=True)
    learner.set_live(ORG, make_version(learner, ORG))
    learner.remember(ANA, "my editor is vim")
    learner.set_live(ANA, make_version(learner, ANA))
    published = learner.store.published()
    published["scopes"][ANA] = {"live": 1, "next": 2, "versions": {}, "history": []}
    learner.store.write_published(published)
    learner.add_notes(ORG, ["The price of Orion Max is 40 euros."], approve=True)
    evaluated = []

    def evaluate(scope, version):
        evaluated.append((str(scope), version))
        # what Learner.evaluate leaves: a report of the child's live version on the new parent
        ss = learner.store.scope(scope)
        report = load_report(ss, version)
        report["live_chain"] = [[ORG, learner.store.scope(ORG).live], ["team:acme/support", None], [ANA, version]]
        return SimpleNamespace(passed=True, report=report)

    pub = StubPublisher()
    report = sleep(learner, publish=True, publisher=pub, run=StubRun(learner), evaluate=evaluate)
    assert evaluated == [(ANA, learner.store.scope(ANA).live)]
    assert pub.published == [ORG, ANA]
    follow = next(e for e in report.entries if e.scope == ANA)
    assert follow.action == "followed" and "new parent" in follow.detail


def test_spooled_signals_are_stored_first(learner):
    learner.spool(ANA, "remembered", "my editor is vim", "proxy:ana")
    run = StubRun(learner)
    report = sleep(learner, run=run)
    assert report.stored_signals == 1 and run.calls == [ANA]
    assert [n.text for n in learner.store.scope(ANA).active_notes()] == ["my editor is vim"]


def test_cli_sleep_exit_codes(learner, monkeypatch, capsys):
    from palingenesis.learn import cli

    root = str(learner.store.root)
    assert cli.main(["sleep", root]) == 0
    assert "nothing pending" in capsys.readouterr().out
    learner.remember(ANA, "my editor is vim")
    monkeypatch.setattr(Learner, "run", lambda self, scope, **kw: StubRun(self, fail=[ANA])(scope))
    assert cli.main(["sleep", root]) == 1
    assert "failed_gates" in capsys.readouterr().out
