"""Asynchronous RL on one GPU: staleness x objective on a small math run (driver of async_rl.sh).

Every arm is one `pgs rl` run of the same config (configs/rl_math.yaml by default: Qwen3.5-0.8B, GSM8K-style rows)
that differs in rollout.max_staleness and loss.type only, on the colocated in-process vLLM engine; the "server" arms
run rollout.backend vllm_server (an engine in another process on the same GPU: the stand-in for engines on other
nodes, whose weights travel through a process boundary). The "inflight" arms exercise rollout.inflight tag / resume
with rollout.stale mask and rollout.sync publish, which have only run on CPU.

Per arm (a JSON line): seconds of training, the evaluation reward before and after, its gain per minute of
wall-clock, the mean training reward of the last steps, the sampler / trainer mismatch (policy/mismatch_k3, mean and
of current-version tokens), the share of groups dropped as stale and the batches the orchestrator dropped.

Pre-registered verdict (one line, "check": "verdict"), written before any run:
  PASS  decoupled at staleness 8 gains >= 1.2 x the on-policy run's evaluation reward per minute of wall-clock
        (masked_is at staleness 0) and ends within 1 point (0.01) of its final evaluation reward;
  KILL  masked_is at staleness 8 meets both conditions too (the decoupled objective buys nothing here);
  FAIL  otherwise.
One seed and --eval-samples rows per evaluation: the line reports the binomial standard error of an evaluation, and
a difference inside two of them is noise whatever the verdict says. loss.proximal step with one optimizer step per
batch is cispo's gradient exactly (see the RL guide); --proximal reference runs the variant with a trust region.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

PAIR = re.compile(r"(\S+?)=(-?[\d.]+(?:e[-+]?\d+)?|nan|inf)\b")


def emit(**fields) -> None:
    print(json.dumps(fields), flush=True)


def parse(log: Path) -> tuple[list[dict], list[dict], int]:
    """(training steps, evaluations, batches the orchestrator dropped) from a run's log."""
    steps, evals, dropped = [], [], 0
    for line in log.read_text(errors="replace").splitlines():
        found = re.search(r"\b(step|eval) (\d+) \| (.*)$", line)
        if found:
            row = {k: float(v) for k, v in PAIR.findall(found.group(3))}
            row["_step"] = int(found.group(2))
            (steps if found.group(1) == "step" else evals).append(row)
        elif "dropped a batch" in line:
            dropped += 1
    return steps, evals, dropped


def mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if key in r and math.isfinite(r[key])]
    return sum(values) / len(values) if values else None


def run_arm(args, name: str, overrides: dict) -> dict:
    work = Path(args.work) / name
    work.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("data", {}).update({"dataset": args.data, "eval_dataset": args.eval_data})
    config["data"].pop("retire_above", None)  # the prompt mix must not differ between arms
    config.setdefault("model", {})["policy"] = args.model
    config.setdefault("train", {}).update(
        {
            "output_dir": str(work / "run"),
            "steps": args.steps,
            "eval_every": args.steps,  # an evaluation before the first step and one after the last
            "eval_samples": args.eval_samples,
            "save_steps": 0,
            "save_final": False,
            "seed": args.seed,
        }
    )
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate
    config.setdefault("rollout", {}).update({"batch_prompts": args.batch_prompts, "group_size": args.group_size})
    config.setdefault("logging", {})["log_every"] = 1
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        config.setdefault(section, {})[field] = value
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    log = work / "train.log"
    start = time.time()
    with log.open("w") as out:
        code = subprocess.call(
            [sys.executable, "-m", "palingenesis.rl.trainer", "--config", str(path)],
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    steps, evals, dropped = parse(log)
    result = {
        "check": f"arm/{name}",
        "exit_code": code,
        "seconds_total": round(time.time() - start, 1),
        "log": str(log),
    }
    if code != 0 or len(steps) < args.steps or len(evals) < 2:
        tail = log.read_text(errors="replace")[-600:]
        emit(**result, status="FAIL", steps=len(steps), evals=len(evals), log_tail=tail)
        return {"name": name, "ok": False}
    elapsed = steps[-1]["elapsed"]
    first, last = evals[0]["reward"], evals[-1]["reward"]
    groups = args.steps * args.batch_prompts
    row = {
        "name": name,
        "ok": True,
        "train_seconds": elapsed,
        "eval_first": first,
        "eval_final": last,
        "gain_per_minute": (last - first) / (elapsed / 60.0),
        "train_reward_last": mean(steps[-5:], "outcome/reward_all"),
        "mismatch_k3": mean(steps, "policy/mismatch_k3"),
        "mismatch_k3_current": mean(steps, "policy/mismatch_k3/current"),
        "mismatch_k3_stale": mean(steps, "policy/mismatch_k3/stale"),
        "stale_tokens": mean(steps, "policy/stale_tokens"),
        "clipped": mean(steps, "policy/clipped"),
        "staleness_mean": mean(steps, "staleness"),
        "stale_group_drop_rate": steps[-1].get("stale_groups_dropped", 0.0) / max(groups, 1),
        "batches_dropped": dropped,
        "wait_batch_s": mean(steps, "time/wait_batch"),
        "rollout_tok_s": mean(steps, "rollout_tok_s"),
        "trained_steps": sum(1 for s in steps if s.get("trained_sequences", 0) > 0),
        "resumed": steps[-1].get("engine/resumed"),
        "stale_masked": mean(steps, "policy/stale_masked"),
    }
    emit(**result, status="INFO", **{k: v for k, v in row.items() if k not in ("name", "ok")})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-prompts", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--staleness", default="0,1,4,8")
    parser.add_argument("--losses", default="masked_is,decoupled")
    parser.add_argument("--proximal", choices=("step", "reference"), default="step")
    parser.add_argument("--parts", default="colocated,server,inflight")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()
    staleness = [int(s) for s in args.staleness.split(",")]
    losses = args.losses.split(",")
    parts = args.parts.split(",")

    def loss_options(kind: str) -> dict:
        options = {"loss.type": kind}
        if kind == "decoupled" and args.proximal == "reference":
            options.update({"loss.proximal": "reference", "loss.kl_ref": "start", "loss.kl_ref_reset": 4})
        return options

    def stale_options(k: int) -> dict:
        return {"rollout.max_staleness": k, **({"rollout.sleep": False} if k else {})}

    rows: dict[str, dict] = {}
    if "colocated" in parts:
        for k in staleness:
            for kind in losses:
                name = f"{kind}_s{k}"
                rows[name] = run_arm(args, name, {**loss_options(kind), **stale_options(k)})
    if "server" in parts:  # an engine in another process on the same GPU: weights cross a process boundary
        for k in (0, max(staleness)):
            for kind in losses:
                name = f"server_{kind}_s{k}"
                options = {**loss_options(kind), "rollout.max_staleness": k, "rollout.backend": "vllm_server"}
                rows[name] = run_arm(args, name, options)
    if "inflight" in parts:  # the per-token paths of the vLLM engine: first run on a GPU here
        top = max(staleness)
        for mode in ("tag", "resume"):
            name = f"inflight_{mode}_s{top}"
            options = {
                **loss_options("decoupled"),
                **stale_options(top),
                "rollout.inflight": mode,
                "rollout.stale": "mask",
                "rollout.sync": "publish",
            }
            row = rows[name] = run_arm(args, name, options)
            if row["ok"]:
                # the engine samples the weights the trainer scores: tokens of the current version match numerically
                current = row["mismatch_k3_current"]
                good = (
                    row["trained_steps"] > 0 and row["stale_tokens"] is not None and (current is None or current < 1e-2)
                )
                good = good and (mode != "resume" or (row["resumed"] or 0) > 0)
                emit(
                    check=f"inflight/{mode}",
                    status="PASS" if good else "FAIL",
                    mismatch_k3_current=current,
                    stale_tokens=row["stale_tokens"],
                    stale_masked=row["stale_masked"],
                    resumed=row["resumed"],
                )

    top = max(staleness)
    base, stale, control = rows.get("masked_is_s0"), rows.get(f"decoupled_s{top}"), rows.get(f"masked_is_s{top}")
    if not (base and stale and control and base["ok"] and stale["ok"] and control["ok"]):
        emit(check="verdict", status="SKIP", reason="needs masked_is at staleness 0 and both objectives at the top one")
        return 1 if any(not r["ok"] for r in rows.values()) else 0

    def meets(row: dict) -> bool:
        faster = base["gain_per_minute"] > 0 and row["gain_per_minute"] >= 1.2 * base["gain_per_minute"]
        return faster and row["eval_final"] >= base["eval_final"] - 0.01

    p = min(max(base["eval_final"], 1e-6), 1 - 1e-6)
    verdict = "KILL" if meets(control) else "PASS" if meets(stale) else "FAIL"
    emit(
        check="verdict",
        status=verdict,
        criterion=f"decoupled_s{top}: gain per minute >= 1.2 x masked_is_s0 and final eval within 0.01; KILL if "
        f"masked_is_s{top} meets it",
        on_policy={k: base[k] for k in ("gain_per_minute", "eval_final", "train_seconds")},
        decoupled={k: stale[k] for k in ("gain_per_minute", "eval_final", "train_seconds", "mismatch_k3", "clipped")},
        masked_is={k: control[k] for k in ("gain_per_minute", "eval_final", "train_seconds", "mismatch_k3", "clipped")},
        speedup_decoupled=stale["gain_per_minute"] / base["gain_per_minute"] if base["gain_per_minute"] > 0 else None,
        speedup_masked_is=control["gain_per_minute"] / base["gain_per_minute"] if base["gain_per_minute"] > 0 else None,
        eval_standard_error=math.sqrt(p * (1 - p) / args.eval_samples),
        proximal=args.proximal,
        seeds=1,
    )
    failed = any(not r["ok"] for r in rows.values())
    return 1 if failed or verdict == "FAIL" else 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
