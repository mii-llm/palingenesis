"""GPU check of asynchronous in-training evaluation (eval_tasks.placement async): helpers of async_eval.sh.

    python async_eval_check.py topology                 # mig | gpus | cpu, and the devices to use (shell lines)
    python async_eval_check.py config ARM OUT.yaml ...  # the pgs train config of one arm
    python async_eval_check.py verdict WORK             # the pre-registered criteria, one JSON line each

Pre-registered criteria (async_eval.sh runs the arms):
  step_time     the wall-clock time per step of `push` over steps 10..STEPS is within 2 % of `none` (the same run
                without eval_tasks); the snapshot at due steps is included (eval/async/snapshot_seconds, reported)
  snapshot      eval/async/snapshot_seconds <= 1 s for a 0.8B model (INFO for other models)
  losses        train/loss of `push`, `checkpoint` and `oom` equal `none` step by step: relative difference
                <= max(1e-6, the floor measured between two `none` runs)
  refusal       eval_tasks.devices naming the trainer's GPU stops the run before any model loads, with the message
  oom           an evaluator whose every evaluation runs out of memory (PGS_TEST_EVAL_OOM=always; on MIG / two GPUs
                also a real one, `real_oom`) never stops training: exit 0, every step done, losses as `none`
  disk          `checkpoint` with `pgs eval watch` as a separate process: the run directory stays under 20 GB
  evaluated     the evaluator produced results (push and checkpoint: the final step done)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


class JsonlTracker:
    """logging.trackers entry: every logged metric with its step and the wall-clock time, to metrics.jsonl."""

    def __init__(self, *, project, name, config, run_id, output_dir, resuming):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics, step):
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, "time": time.time(), "metrics": metrics}, default=float) + "\n")

    def finish(self, exit_code):
        pass


# ------------------------------------------------------------------ topology


def topology() -> None:
    """Shell lines: MODE (mig | gpus | cpu), TRAIN_DEV, EVAL_DEV (CUDA_VISIBLE_DEVICES entries; cpu: EVAL_DEV="")."""
    sys.path.insert(0, os.environ.get("PGS_SRC", ""))
    from palingenesis.eval.placement import nvidia_devices

    devices = nvidia_devices()
    wanted = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if wanted:
        entries = [e.strip() for e in wanted.split(",") if e.strip()]
        devices = [d for i, d in enumerate(devices) if str(i) in entries or d["uuid"] in entries] or devices
    migs = [m for d in devices for m in d["mig"]]
    if len(migs) >= 2:
        print(f"MODE=mig TRAIN_DEV={migs[0]} EVAL_DEV={migs[1]}")
    elif len(devices) >= 2:
        print(f"MODE=gpus TRAIN_DEV={devices[0]['index']} EVAL_DEV={devices[1]['index']}")
    else:
        index = devices[0]["index"] if devices else 0
        print(f"MODE=cpu TRAIN_DEV={index} EVAL_DEV=")


# -------------------------------------------------------------------- config


def config(arm: str, out: str, base: str, work: str, mode: str) -> None:
    import yaml

    with open(base) as f:
        raw = yaml.safe_load(f)
    env = os.environ
    steps = int(env.get("STEPS", "300"))
    every = int(env.get("EVERY", "50"))
    run = Path(work) / arm
    raw.setdefault("model", {})
    if env.get("MODEL"):
        raw["model"]["name_or_path"] = env["MODEL"]
    raw.setdefault("data", {}).update(
        {"dataset": env["DATA"], "streaming": False, "eval_dataset": "", "eval_every": every}
    )
    raw.setdefault("train", {}).update(
        {"output_dir": str(run), "max_steps": steps, "save_steps": 0, "save_final": False, "seed": 0,
         "logging_steps": 1, "resume_from": ""}
    )  # fmt: skip
    raw.setdefault("logging", {}).update(
        {"use_wandb": False, "use_trackio": False, "trackers": [f"{Path(__file__).resolve()}:JsonlTracker"]}
    )
    if arm in ("none", "none2"):
        raw.pop("eval_tasks", None)
    else:
        task = {
            "name": "check",
            "dataset": env["EVAL_DATA"],
            "prompt": "{{" + env.get("EVAL_PROMPT_FIELD", "question") + "}}",
            "scorer": "numeric",
            "sampling": {"max_tokens": int(env.get("EVAL_MAX_TOKENS", "256")), "temperature": 0.0},
        }
        section = {"tasks": [task], "every": every, "limit": int(env.get("EVAL_LIMIT", "50")), "final_timeout_s": 1800}
        engine = {"backend": "hf"} if mode == "cpu" else {"backend": "vllm", "gpu_memory_utilization": 0.85}
        if arm == "inline":
            section.update({"placement": "inline"})
        elif arm == "refusal":
            section.update({"placement": "async", "devices": [0], "engine": engine})
        elif arm == "checkpoint":
            section.update({"placement": "async", "service": "auto", "transport": "checkpoint", "engine": engine})
            raw["train"].update({"save_steps": every, "keep_checkpoints": 1})
        else:  # push, oom, real_oom
            placement = {"service": "auto"} if mode == "cpu" else {"devices": [1]}
            section.update({"placement": "async", "transport": "push", "engine": engine, **placement})
            if arm == "real_oom":
                section["engine"] = {**engine, "gpu_memory_utilization": 0.95}
        raw["eval_tasks"] = section
    with open(out, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


# ------------------------------------------------------------------- verdict


def _metrics(work: Path, arm: str) -> list[dict]:
    path = work / arm / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _steps(rows: list[dict]) -> dict[int, dict]:
    """step -> (time, loss) of the training lines (those with train/loss)."""
    out = {}
    for row in rows:
        m = row["metrics"]
        if "train/loss" in m and row["step"] is not None:
            out[int(row["step"])] = {"time": row["time"], "loss": float(m["train/loss"]), **m}
    return out


def _per_step(steps: dict[int, dict], first: int = 10) -> float | None:
    keys = sorted(k for k in steps if k >= first)
    if len(keys) < 2:
        return None
    return (steps[keys[-1]]["time"] - steps[keys[0]]["time"]) / (keys[-1] - keys[0])


def _loss_diff(a: dict, b: dict) -> float | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    return max(abs(a[k]["loss"] - b[k]["loss"]) / max(abs(b[k]["loss"]), 1e-12) for k in common)


def verdict(work: str) -> int:
    work = Path(work)
    steps_wanted = int(os.environ.get("STEPS", "300"))
    lines = []

    def emit(check: str, status: str, **fields) -> None:
        lines.append({"check": check, "status": status, **fields})

    def exit_code(arm: str):
        path = work / f"{arm}.exit"
        return int(path.read_text().strip()) if path.exists() else None

    none = _steps(_metrics(work, "none"))
    floor = _loss_diff(_steps(_metrics(work, "none2")), none) if (work / "none2").exists() else None
    tolerance = max(1e-6, floor or 0.0)
    base_time = _per_step(none)
    for arm in ("push", "checkpoint", "oom", "real_oom", "inline"):
        if not (work / arm).exists():
            continue
        steps = _steps(_metrics(work, arm))
        done = max(steps) if steps else 0
        complete = exit_code(arm) == 0 and done >= steps_wanted
        if arm != "inline":
            diff = _loss_diff(steps, none)
            ok = complete and diff is not None and diff <= tolerance
            emit(f"losses/{arm}", "PASS" if ok else "FAIL", max_rel_diff=diff, tolerance=tolerance,
                 floor_between_none_runs=floor, exit=exit_code(arm), last_step=done)  # fmt: skip
        per_step = _per_step(steps)
        if base_time and per_step:
            ratio = per_step / base_time
            if arm == "push":
                emit("step_time/push", "PASS" if ratio <= 1.02 else "FAIL", ratio=round(ratio, 4),
                     seconds_per_step=per_step, none_seconds_per_step=base_time)  # fmt: skip
            else:
                emit(f"step_time/{arm}", "INFO", ratio=round(ratio, 4), seconds_per_step=per_step)
        if arm in ("push", "checkpoint"):
            results = {}
            for p in (work / arm / "eval_tasks").glob("step_*/result.json"):
                r = json.loads(p.read_text())
                results[r["step"]] = r["status"]
            evaluated = [s for s, v in results.items() if v == "done"]
            final_done = results.get(done) == "done" or results.get(steps_wanted) == "done"
            emit(f"evaluated/{arm}", "PASS" if evaluated and final_done else "FAIL", results=results)
        if arm == "push":
            snaps = [
                s.get("eval/async/snapshot_seconds", 0.0)
                for s in steps.values()
                if s.get("eval/async/snapshot_seconds")
            ]
            worst = max(snaps) if snaps else None
            model = os.environ.get("MODEL", "")
            status = (
                "INFO" if "0.8B" not in model and model else ("PASS" if worst is not None and worst <= 1.0 else "FAIL")
            )
            emit("snapshot", status, max_snapshot_seconds=worst, model=model or "the config's")
        if arm in ("oom", "real_oom"):
            emit(f"oom/{arm}", "PASS" if complete else "FAIL", exit=exit_code(arm), last_step=done)
    refusal = work / "refusal.log"
    if refusal.exists():
        text = refusal.read_text()
        seconds = float((work / "refusal.seconds").read_text()) if (work / "refusal.seconds").exists() else None
        refused = exit_code("refusal") not in (0, None) and "needs a GPU the trainer does not use" in text
        no_model = "Loading weights" not in text and "Loading checkpoint shards" not in text
        emit("refusal", "PASS" if refused and no_model else "FAIL", exit=exit_code("refusal"), seconds=seconds,
             loaded_a_model=not no_model)  # fmt: skip
    disk = work / "checkpoint.disk_max_kb"
    if disk.exists():
        gb = int(disk.read_text().strip() or 0) / 2**20
        emit("disk", "PASS" if gb <= 20 else "FAIL", max_gb=round(gb, 2), limit_gb=20)
    for line in lines:
        print(json.dumps(line, default=str))
    return 1 if any(line["status"] == "FAIL" for line in lines) else 0


def main() -> int:
    command = sys.argv[1]
    if command == "topology":
        topology()
        return 0
    if command == "config":
        arm, out = sys.argv[2], sys.argv[3]
        config(arm, out, os.environ["CONFIG"], os.environ["WORK"], os.environ.get("MODE", "gpus"))
        return 0
    if command == "verdict":
        return verdict(sys.argv[2])
    if command == "hog":  # hold GPU memory (the real out-of-memory arm)
        import torch

        held = torch.empty(int(float(sys.argv[2]) * 2**30), dtype=torch.uint8, device="cuda")
        print(f"holding {held.numel() / 2**30:.1f} GiB", flush=True)
        time.sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 7200)
        return 0
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    sys.exit(main())
