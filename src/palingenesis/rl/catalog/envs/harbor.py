"""harbor: the Harbor / shell-task family (palingenesis.rl.envs.harbor) as a catalogue entry.

A Harbor task is a directory: an instruction, a Docker environment, tests that write the reward, optionally an
oracle solution. The catalogue adds nothing to how they run (HarborEnv, with harness profiles, is the environment);
it makes them discoverable and puts them under the same commands as every other environment:

    pgs env test harbor --dataset data/harbor_tasks      each task's oracle / nop check (Docker)
    pgs env run  harbor --dataset data/harbor_tasks --model M --k 4
    pgs env rows harbor --dataset data/harbor_tasks -o rows.jsonl

The audit is each task's own: solution/solve.sh (or the oracle reply) must score 1, doing nothing must not, and a
reply-graded task must score 0 on an empty reply (rl.envs.harbor.check_task). There is no built-in dataset: tasks
are yours, or a published task set you downloaded (check its licence).
"""

from __future__ import annotations

import time
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec


def _summary(passed: int, total: int, seconds: float) -> dict[str, Any]:
    return {
        "episodes": total,
        "scored": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "mean_reward": passed / total if total else None,
        "errors": 0,
        "first_error": "",
        "tool_calls": 0,
        "tool_errors": 0,
        "steps": total,
        "seconds": seconds,
        "episodes_per_s": total / seconds if seconds > 0 else 0.0,
        "steps_per_s": total / seconds if seconds > 0 else 0.0,
    }


def audit_tasks(spec: EnvSpec, n: int = 0, seed: int = 0, dataset: str = "", rows: list[dict] | None = None,
                workers: int = 8) -> dict[str, Any]:  # fmt: skip
    """The oracle / nop check of every task behind the rows (a container per check: needs Docker)."""
    from concurrent.futures import ThreadPoolExecutor

    from palingenesis.rl.catalog.runner import load_rows
    from palingenesis.rl.envs import harbor

    if rows is None:
        if not dataset:
            return {"failures": ["harbor has no built-in tasks: pass --dataset <directory of Harbor tasks>"]}
        rows = load_rows(spec, dataset, n, seed)
    directories = list(dict.fromkeys(row["task_dir"] for row in rows))
    start = time.perf_counter()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(harbor.check_task, directories))
    seconds = time.perf_counter() - start
    failures, tasks = [], []
    for result in results:
        if result["valid"]:
            continue
        reason = (
            result.get("oracle_error")
            or result.get("nop_error")
            or ("no oracle solution to check the tests against" if result.get("oracle") is None else "")
            or f"oracle={result.get('oracle')} nop={result.get('nop')}"
            + (f" nop_reply={result.get('nop_reply')}" if "nop_reply" in result else "")
        )
        tasks.append(f"{result['task']}: {reason}")
        if result.get("nop") is not None and result["nop"] >= 1.0:
            failures.append(f"degenerate nop: task {result['task']} pays doing nothing, a reward-hacking hole")
        elif result.get("nop_reply"):
            failures.append(f"degenerate nop_reply: task {result['task']} pays an empty reply, a reward-hacking hole")
        else:
            failures.append(f"oracle: task {result['task']} is not valid ({reason})")
    total = len(results)
    return {
        "rows": total,
        "oracle": _summary(sum(r.get("oracle") == 1.0 for r in results), total, seconds),
        "degenerate": {
            "nop": _summary(sum((r.get("nop") or 0.0) >= 1.0 for r in results), total, seconds),
            "nop_reply": _summary(sum((r.get("nop_reply") or 0.0) > 0.0 for r in results), total, seconds),
        },
        "tasks": tasks,
        "holes": [f for f in failures if "hole" in f],
        "failures": failures,
    }


SPEC = EnvSpec(
    name="harbor",
    version="1.0.0",
    family="shell",
    summary="Harbor task directories: shell and file-editing tasks in Docker, rewarded by the task's own tests.",
    env="palingenesis.rl.envs.harbor:HarborEnv",
    license="Apache-2.0",
    verifier=(
        "The task's tests/test.sh, run after the episode in a clean container that holds only a copy of the "
        "working directory, writes the reward. Audited per task: the oracle solution must score 1, doing nothing "
        "must not."
    ),
    datasets={
        "tasks": DatasetSpec(
            description="Your task directories: pass --dataset <dir> (rows are built with harbor_rows).",
        )
    },
    sandbox="docker",
    max_turns=40,
    max_new_tokens=2048,
    audit=audit_tasks,
    tags=("multi-turn", "agentic", "docker", "harness-profiles"),
    notes=(
        "env_args take HarborEnv's options (tools, profiles: {opencode: 1, pi: 1}, ...); see the Harbor and "
        "Harness Profiles guides."
    ),
)
