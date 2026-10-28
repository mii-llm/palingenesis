"""pgs learn sleep: one pass over the scopes with pending notes, under a compute budget. A command for cron,
not a daemon:

    pgs learn sleep STORE [--scopes S ...] [--budget-minutes N] [--scope-budget-minutes N] [--publish]

    1. signals spooled while a job held the store are stored; with --publish, drained and forgotten versions
       are unloaded from the servers
    2. the scopes with pending work (unconsolidated active facts, or a pending rebuild) are ordered parents
       first, then by pending facts
    3. per scope: plan (minutes estimated from the times measured on this store) -> run (train, then evaluate
       against the live version) -> set live when every gate passes -> publish (with the parity canary) when
       --publish. A scope that does not fit the remaining budget waits: its notes stay in the served prompt
    4. sleep.children_follow gated (with --publish): after a scope goes live, each published descendant that
       did not train in this pass is evaluated on the new parent and republished when its gates pass

A failed gate never publishes. The exit code is 1 when something needs a person (a failed gate or canary,
an error), so cron can mail it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from palingenesis.learn.store import ScopeId, StoreError

logger = logging.getLogger(__name__)


@dataclass
class SleepEntry:
    scope: str
    action: str  # published | live | unchanged | failed_gates | waiting | error | followed
    version: str | None = None
    minutes: float = 0.0
    detail: str = ""

    @property
    def needs_person(self) -> bool:
        return self.action in ("failed_gates", "error")

    def line(self) -> str:
        version = f" {self.version}" if self.version else ""
        minutes = f" ({self.minutes:.1f} min)" if self.minutes else ""
        return f"{self.scope}: {self.action}{version}{minutes}" + (f": {self.detail}" if self.detail else "")


@dataclass
class SleepReport:
    entries: list = field(default_factory=list)
    minutes: float = 0.0
    unloaded: list = field(default_factory=list)
    stored_signals: int = 0

    @property
    def ok(self) -> bool:
        return not any(e.needs_person for e in self.entries)

    def summary(self) -> str:
        lines = [e.line() for e in self.entries] or ["nothing pending"]
        lines += self.unloaded
        lines.append(
            f"{self.minutes:.1f} min; {self.stored_signals} spooled signal(s) stored; "
            + ("nothing needs a person" if self.ok else "something needs a person")
        )
        return "\n".join(lines)


def pending_scopes(learner, scopes=None) -> list[tuple[ScopeId, int]]:
    """(scope, pending facts) of the scopes with work to do, parents first, then the most pending first."""
    rebuild = set(learner.store.state.get("rebuild") or [])
    wanted = None if not scopes else {str(ScopeId.parse(s)) for s in scopes}
    out = []
    for s in learner.store.scopes():
        if wanted is not None and str(s) not in wanted:
            continue
        facts = learner.store.scope(s).active_facts()
        pending = sum(1 for f in facts if f.consolidated_in is None)
        if pending or (str(s) in rebuild and facts):
            out.append((s, pending))
    return sorted(out, key=lambda x: (len(x[0].chain()), -x[1], str(x[0])))


def sleep(
    learner,
    scopes=None,
    budget_minutes: float | None = None,
    scope_budget_minutes: float | None = None,
    publish: bool = False,
    publisher=None,
    force: bool = False,
    run: Callable | None = None,
    evaluate: Callable | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SleepReport:
    """One sleep pass (see the module docstring). `run(scope)` and `evaluate(scope, version)` default to
    Learner.run / Learner.evaluate; `publisher` to Learner.publisher() when publishing. `force` goes to the
    publisher (past a forced or ungated ancestor, recorded); a version whose own gates fail never goes live here."""
    from palingenesis.learn.trainer import PreflightError

    config = learner.config.sleep
    budget = config.budget_minutes if budget_minutes is None else budget_minutes
    per_scope = config.scope_budget_minutes if scope_budget_minutes is None else scope_budget_minutes
    run = run or (lambda s: learner.run(s))
    evaluate = evaluate or (lambda s, v: learner.evaluate(s, v))
    report = SleepReport()
    started = clock()
    errors = (StoreError, PreflightError, ValueError, RuntimeError, KeyError)

    report.stored_signals = len(learner.drain_spool())
    if publish:
        publisher = publisher or learner.publisher()
        report.unloaded = publisher.unload()

    def remaining() -> float:
        return float("inf") if not budget else budget - (clock() - started) / 60

    def do_publish(scope: ScopeId, entry: SleepEntry) -> None:
        try:
            result = publisher.publish(scope, force=force)
        except errors as e:
            entry.action, entry.detail = "error", f"live, not published: {e}"
            return
        entry.action = "unchanged" if result.action == "unchanged" else "published"
        entry.detail = result.name or ""

    went_live = []
    for scope, pending in pending_scopes(learner, scopes):
        estimate = learner.plan(scope)[0].get("minutes")
        left = remaining()
        if left <= 0:
            report.entries.append(SleepEntry(str(scope), "waiting", detail="the budget is spent"))
            continue
        if estimate is not None and per_scope and estimate > per_scope:
            detail = f"estimated {estimate} min > the scope budget {per_scope} min (split its notes or raise it)"
            report.entries.append(SleepEntry(str(scope), "waiting", detail=detail))
            continue
        if estimate is not None and estimate > left:
            detail = f"estimated {estimate} min > {left:.1f} min left"
            report.entries.append(SleepEntry(str(scope), "waiting", detail=detail))
            continue
        t0 = clock()
        entry = SleepEntry(str(scope), "error")
        report.entries.append(entry)
        try:
            result = run(scope)
            entry.version = result.version
            evaluation = result.evaluation
            if evaluation is None or not evaluation.passed:
                failed = [g["name"] for g in evaluation.report["gates"] if not g["passed"]] if evaluation else []
                entry.action = "failed_gates"
                entry.detail = f"failed {', '.join(failed)}; notes stay in the prompt" if failed else "not evaluated"
            else:
                learner.set_live(scope, result.version)
                entry.action = "live"
                went_live.append(scope)
                if publish:
                    do_publish(scope, entry)
        except errors as e:
            entry.action, entry.detail = "error", str(e)
        entry.minutes = (clock() - t0) / 60

    if publish and config.children_follow == "gated" and went_live:
        trained = {e.scope for e in report.entries}
        for s in sorted(learner.store.published()["scopes"], key=lambda x: len(ScopeId.parse(x).chain())):
            child = ScopeId.parse(s)
            if s in trained or not any(p in child.chain()[:-1] for p in went_live):
                continue
            entry = SleepEntry(s, "followed")
            report.entries.append(entry)
            if remaining() <= 0:
                entry.action, entry.detail = "waiting", "the budget is spent (re-evaluate on the new parent later)"
                continue
            t0 = clock()
            try:
                live = learner.store.scope(child).live
                if live is not None:
                    evaluation = evaluate(child, live)
                    if not evaluation.passed:
                        failed = [g["name"] for g in evaluation.report["gates"] if not g["passed"]]
                        entry.action = "failed_gates"
                        entry.detail = f"on the new parent: failed {', '.join(failed)}; sleep the scope again"
                        continue
                    entry.version = live
                do_publish(child, entry)
                if entry.action in ("published", "unchanged"):
                    entry.action, entry.detail = "followed", f"republished as {entry.detail} on the new parent"
            except errors as e:
                entry.action, entry.detail = "error", str(e)
            finally:
                entry.minutes = (clock() - t0) / 60
    report.minutes = (clock() - started) / 60
    return report
