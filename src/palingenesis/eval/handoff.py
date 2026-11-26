"""The files a trainer and its asynchronous evaluator share, under <output_dir>/eval_tasks/ (standard library only).

    spec.json           the trainer: what to evaluate and how (tasks, model, tokenizer, engine, cadence, its GPUs)
    service.json        the evaluator: where it listens (push), its GPUs, its process; service.<id>.json for more
    step_<n>/           the evaluator: the result files of step n (the `pgs compare` format, as inline evaluation)
    step_<n>/result.json  the evaluator, last: {"status": done | failed | skipped, "metrics": {...}, ...}
    step_<n>/logged     the trainer: the metrics went to its tracker
    best_tasks.json     the evaluator (eval_tasks.select): the best step so far by that metric
    run_end.json        the trainer: the run's last step and final checkpoint; the evaluator ends after it

and, in a checkpoint the evaluator has not loaded yet, `.eval_pending`: the checkpoint purge leaves it alone (for
LEASE_SECONDS, so an evaluator that went away does not keep checkpoints forever).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DIRECTORY = "eval_tasks"
SPEC = "spec.json"
SERVICE = "service.json"
RESULT = "result.json"
LOGGED = "logged"
BEST = "best_tasks.json"
RUN_END = "run_end.json"
LEASE = ".eval_pending"
LEASE_SECONDS = 2 * 3600.0
SPEC_VERSION = 1


def root(output_dir: str | Path) -> Path:
    return Path(output_dir) / DIRECTORY


def step_dir(output_dir: str | Path, step: int) -> Path:
    return root(output_dir) / f"step_{step}"


def write_json(path: str | Path, data: Any) -> None:
    """Written whole, then renamed: a reader never sees half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    """The file's content, or None when it is missing or not whole."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def results(output_dir: str | Path) -> dict[int, dict]:
    """{step: result.json} of every step the evaluator finished (done, failed or skipped)."""
    out = {}
    base = root(output_dir)
    if not base.is_dir():
        return out
    for entry in base.iterdir():
        if entry.name.startswith("step_") and entry.name[5:].isdigit():
            result = read_json(entry / RESULT)
            if isinstance(result, dict):
                out[int(entry.name[5:])] = result
    return out


def write_result(output_dir: str | Path, step: int, status: str, **fields) -> dict:
    result = {"step": step, "status": status, "finished": time.time(), **fields}
    write_json(step_dir(output_dir, step) / RESULT, result)
    return result


def services(output_dir: str | Path) -> list[dict]:
    """The service.json files of the run's evaluators."""
    base = root(output_dir)
    if not base.is_dir():
        return []
    found = [read_json(p) for p in sorted(base.glob("service*.json"))]
    return [s for s in found if isinstance(s, dict)]


# ------------------------------------------------------------------ leases


def write_lease(checkpoint: str | Path, step: int) -> None:
    """Mark a checkpoint as waiting for the evaluator."""
    try:
        write_json(Path(checkpoint) / LEASE, {"step": step, "since": time.time()})
    except OSError:
        pass  # a checkpoint that is gone already has nothing to hold


def release(checkpoint: str | Path) -> None:
    try:
        (Path(checkpoint) / LEASE).unlink()
    except OSError:
        pass


def held(checkpoint: str | Path, output_dir: str | Path | None = None, now: float | None = None) -> bool:
    """Whether the checkpoint purge must keep this checkpoint: a fresh lease (the evaluator has not loaded it yet),
    or the step best_tasks.json selected."""
    checkpoint = Path(checkpoint)
    lease = read_json(checkpoint / LEASE)
    if isinstance(lease, dict) and (now or time.time()) - float(lease.get("since", 0)) < LEASE_SECONDS:
        return True
    if output_dir is not None:
        best = read_json(root(output_dir) / BEST)
        if isinstance(best, dict) and best.get("path"):
            try:
                return Path(best["path"]).resolve() == checkpoint.resolve()
            except OSError:
                return False
    return False
