"""The audit of an environment's reward: an oracle must be paid, degenerate policies must not.

A verifier is a program a policy optimizes against, and a small model finds its holes within tens of RL steps: a
reward that pays an empty reply, the prompt repeated, every tool called once or a claim of success teaches exactly
that. The audit plays scripted policies through the environment (catalog.runner.play, the rollout loop without a
model) on the same rows:

  oracle        the spec's own solver (it reads the row's gold): every row must reach the success threshold. A row
                it does not solve is a broken task, a verifier that rejects the intended answer, or a gold that is wrong.
  degenerate    the spec's own attacks (what is cheap in this environment: a hard-coded sample output, SELECT *,
                a schema-valid empty object, one word guessed six times) plus generic ones run on every environment:
                  empty         an empty reply
                  refusal       "I cannot help with that."
                  echo_prompt   the user's message repeated (verifiers that search the reply for the gold)
                  tool_spam     every tool called each turn with placeholder arguments, then "Done."
                No more than spec.chance of the rows may pass (0 unless guessing or abstention rows pay honestly).
  reuse         the oracle again after the attacks, on the pool's reused instances: an environment that carries
                state from one episode into the next fails here.

A degenerate policy above chance is a reward-hacking hole, reported with the first row it passes. An environment that
raises on a degenerate input (instead of answering "Error: ..." or scoring 0) is reported too. The report also
carries the scripted throughput on CPU (episodes/s and steps/s of the oracle).

    from palingenesis.rl.catalog import audit, get
    report = audit(get("text-to-sql"), n=64)
    assert report["ok"], report["failures"]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from palingenesis.rl.catalog.runner import Episode, last_user, load_rows, make_pool, play_many, turn_index
from palingenesis.rl.catalog.spec import EnvSpec, Policy

REFUSAL = "I cannot help with that."


def empty(row: dict, messages: list[dict]) -> str:
    return ""


def refusal(row: dict, messages: list[dict]) -> str:
    return REFUSAL


def echo_prompt(row: dict, messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "") or last_user(messages)


def placeholder_arguments(schema: dict) -> dict:
    """Arguments that satisfy a tool's required parameters by type, and say nothing."""
    parameters = schema.get("function", schema).get("parameters") or {}
    properties = parameters.get("properties") or {}
    values = {"string": "a", "integer": 1, "number": 1, "boolean": True, "array": [], "object": {}}
    out = {}
    for name in parameters.get("required") or list(properties):
        prop = properties.get(name) or {}
        kind = prop.get("type")
        kind = kind[0] if isinstance(kind, list) else kind
        out[name] = prop["enum"][0] if prop.get("enum") else values.get(kind, "a")
    return out


def tool_spam(schemas_of) -> Policy:
    """Every tool of the episode, called each turn with placeholder arguments; "Done." on the last turns."""

    def policy(row: dict, messages: list[dict]) -> Any:
        schemas = schemas_of(row)
        if not schemas or turn_index(messages) >= 3:
            return "Done."
        return [(s["function"]["name"], placeholder_arguments(s)) for s in schemas]

    return policy


def _summary(episodes: list[Episode], threshold: float, seconds: float) -> dict[str, Any]:
    scored = [e for e in episodes if e.reward is not None]
    passed = [e for e in scored if e.reward >= threshold]
    errors = [e for e in episodes if e.finish == "error"]
    steps = sum(e.steps for e in episodes)
    return {
        "episodes": len(episodes),
        "scored": len(scored),
        "passed": len(passed),
        "pass_rate": len(passed) / len(episodes) if episodes else 0.0,
        "mean_reward": sum(e.reward for e in scored) / len(scored) if scored else None,
        "errors": len(errors),
        "first_error": errors[0].error if errors else "",
        "tool_calls": sum(e.tool_calls for e in episodes),
        "tool_errors": sum(e.tool_errors for e in episodes),
        "steps": steps,
        "seconds": seconds,
        "episodes_per_s": len(episodes) / seconds if seconds > 0 else 0.0,
        "steps_per_s": steps / seconds if seconds > 0 else 0.0,
    }


def _row_id(episode: Episode) -> str:
    return str(episode.row.get("id", "?"))


async def audit_async(
    spec: EnvSpec,
    n: int = 32,
    seed: int = 0,
    dataset: str = "",
    rows: list[dict] | None = None,
    concurrency: int = 32,
    generic: bool = True,
    env_args: dict | None = None,
    scripted_only: bool = False,
) -> dict[str, Any]:
    from palingenesis.rl.env import row_tools

    report: dict[str, Any] = {
        "env": spec.name,
        "version": spec.version,
        "dataset": dataset or spec.default_dataset,
        "threshold": spec.success_threshold,
        "chance": spec.chance,
        "oracle": None,
        "degenerate": {},
        "reuse": None,
        "failures": [],
        "holes": [],
        "warnings": [],
    }
    failures = report["failures"]
    missing = spec.missing_requirements()
    if missing:
        failures.append(f"missing Python modules: {', '.join(missing)}")
        report["ok"] = False
        return report
    if spec.audit is not None and scripted_only is False:
        # the environment audits itself (Harbor: each task's own oracle / nop check)
        custom = spec.audit(spec, n=n, seed=seed, dataset=dataset, rows=rows)
        if asyncio.iscoroutine(custom):
            custom = await custom
        report.update(custom)
        report["ok"] = not report["failures"]
        return report
    rows = rows if rows is not None else load_rows(spec, dataset, n, seed)
    report["rows"] = len(rows)
    if not rows:
        failures.append("no rows to audit")
        report["ok"] = False
        return report
    pool = make_pool(spec, env_args=env_args)
    threshold = spec.success_threshold

    async def run(policy: Policy, subset: list[dict] | None = None) -> tuple[list[Episode], dict]:
        start = time.perf_counter()
        episodes = await play_many(pool, subset or rows, policy, spec.max_turns, concurrency)
        return episodes, _summary(episodes, threshold, time.perf_counter() - start)

    try:
        unsolved: list[Episode] = []
        if spec.oracle is None:
            report["warnings"].append(
                "no oracle policy: the audit shows what must not be paid, not that the tasks can be solved"
            )
        else:
            episodes, report["oracle"] = await run(spec.oracle)
            unsolved = [e for e in episodes if e.reward is None or e.reward < threshold]
        if unsolved:
            first = unsolved[0]
            why = first.error or f"reward {first.reward} (finish {first.finish}, {first.turns} turns)"
            failures.append(
                f"oracle: {len(unsolved)} of {len(episodes)} rows are not solved; row {_row_id(first)}: {why}"
            )
        policies: dict[str, Policy] = dict(spec.degenerate)
        if generic:
            schemas_of = (lambda row: pool.schemas) if not pool.dynamic else (lambda row: row_tools(row, "tools") or [])
            for name, policy in (
                ("empty", empty),
                ("refusal", refusal),
                ("echo_prompt", echo_prompt),
                ("tool_spam", tool_spam(schemas_of)),
            ):
                policies.setdefault(name, policy)
        # with a chance level, one lucky row is not a hole however few rows are audited
        allowed = max(spec.chance * len(rows), 1.0 if spec.chance > 0 else 0.0) + 1e-9
        for name, policy in policies.items():
            episodes, summary = await run(policy)
            report["degenerate"][name] = summary
            passed = [e for e in episodes if e.reward is not None and e.reward >= threshold]
            if len(passed) > allowed:
                hole = {
                    "policy": name,
                    "passed": len(passed),
                    "rows": len(episodes),
                    "row": _row_id(passed[0]),
                    "reply": next(
                        (str(m.get("content"))[:200] for m in reversed(passed[0].messages) if m["role"] == "assistant"),
                        "",
                    ),
                }
                report["holes"].append(hole)
                failures.append(
                    f"degenerate {name}: passes {len(passed)} of {len(episodes)} rows (allowed: "
                    f"{spec.chance:.0%}), a reward-hacking hole; first on row {hole['row']}"
                )
            elif summary["mean_reward"] is not None and summary["mean_reward"] > max(spec.chance, 0.25) * threshold:
                report["warnings"].append(
                    f"degenerate {name}: mean reward {summary['mean_reward']:.2f} without solving the task "
                    "(partial credit a policy can collect)"
                )
            if summary["errors"]:
                failures.append(
                    f"degenerate {name}: the environment raised on {summary['errors']} of {len(episodes)} episodes "
                    f"({summary['first_error']}); a bad input must be an observation or a zero, not an exception"
                )
        again = rows[: min(len(rows), 8)]
        stale: list[Episode] = []
        if spec.oracle is not None:
            episodes, report["reuse"] = await run(spec.oracle, again)
            stale = [e for e in episodes if e.reward is None or e.reward < threshold]
        if stale and not unsolved:
            failures.append(
                f"reuse: the oracle fails {len(stale)} of {len(again)} rows after the degenerate episodes; the "
                f"environment carries state between episodes (row {_row_id(stale[0])})"
            )
    finally:
        await pool.shutdown()
    report["ok"] = not failures
    return report


def audit_without_oracle(spec: EnvSpec, n: int = 32, seed: int = 0, dataset: str = "", rows=None) -> Any:
    """An EnvSpec.audit for environments nobody wrote a solver for (imported ones): the degenerate policies only."""
    return _degenerate_report(spec, n, seed, dataset, rows)


async def _degenerate_report(spec: EnvSpec, n: int, seed: int, dataset: str, rows) -> dict[str, Any]:
    report = await audit_async(spec, n=n, seed=seed, dataset=dataset, rows=rows, scripted_only=True)
    return {
        k: report[k] for k in ("rows", "oracle", "degenerate", "reuse", "failures", "holes", "warnings") if k in report
    }


def audit(spec: EnvSpec, **kwargs) -> dict[str, Any]:
    """The audit report of `spec` (see the module docstring); report["ok"] says whether it passed."""
    return asyncio.run(audit_async(spec, **kwargs))


def render(report: dict[str, Any]) -> str:
    """The report as the lines `pgs env test` prints."""
    lines = [f"{report['env']} {report['version']}  dataset {report['dataset']}  rows {report.get('rows', '-')}"]

    def line(kind: str, name: str, s: dict) -> str:
        mean = "-" if s.get("mean_reward") is None else f"{s['mean_reward']:.3f}"
        return (
            f"  {kind:<11}{name:<24} pass {s['passed']:>3}/{s['episodes']:<3} mean reward {mean:>6}  "
            f"{s['episodes_per_s']:>8.1f} episodes/s {s['steps_per_s']:>9.1f} steps/s"
        )

    if report.get("oracle"):
        lines.append(line("oracle", "", report["oracle"]))
    for name, summary in (report.get("degenerate") or {}).items():
        lines.append(line("degenerate", name, summary))
    if report.get("reuse"):
        lines.append(line("reuse", "oracle again", report["reuse"]))
    for task in report.get("tasks") or []:
        lines.append(f"  task       {task}")
    lines += [f"  WARN  {w}" for w in report.get("warnings") or []]
    lines += [f"  FAIL  {f}" for f in report.get("failures") or []]
    lines.append(f"  audit: {'passed' if report.get('ok') else 'FAILED'}")
    return "\n".join(lines)
