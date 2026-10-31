#!/usr/bin/env python
"""Short SFT / DPO / OPD / RL runs that print throughput and losses, to compare GPUs.

Run it unchanged on a reference NVIDIA GPU and on the GPU under test (e.g. an AMD
MI250 or MI300), then compare the two summaries: every stage should finish, losses
should start and end close to the reference run's, and tok/s says how fast it is.

    python scripts/acceptance_smoke.py --out runs/acceptance
    python scripts/acceptance_smoke.py --out runs/acceptance-aiter --stages sft,dpo \\
        --attn kernels-community/aiter-flash-attn@v2
    python scripts/acceptance_smoke.py --out runs/acceptance-vllm --stages opd,rl --rollout vllm
    python scripts/acceptance_smoke.py --out /tmp/x --dry-run   # write data + configs, validate, run nothing

Each stage starts from the repository's own config (configs/quickstart.yaml,
dpo_qwen35_08b.yaml, distill_chat.yaml, rl_math.yaml) with the overrides below,
written to <out>/configs/<stage>.yaml. The data is synthetic arithmetic, generated
with a fixed seed into <out>/data, so two machines train on the same rows. Models are
downloaded from the Hugging Face Hub. Logs go to <out>/logs/<stage>.log and the summary
to <out>/summary.json. Exit code 1 when a stage fails.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("sft", "dpo", "opd", "rl")
BASE_CONFIGS = {
    "sft": "configs/quickstart.yaml",
    "dpo": "configs/dpo_qwen35_08b.yaml",
    "opd": "configs/distill_chat.yaml",
    "rl": "configs/rl_math.yaml",
}
NO_THINKING = {"enable_thinking": False}


# ---------------------------------------------------------------------- data


def _problems(n: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 9)
        if rng.random() < 0.5:
            out.append((f"What is {a} + {b} * {c}?", a + b * c))
        else:
            out.append((f"What is {a} * {c} - {b}?", a * c - b))
    return out


def write_data(data_dir: Path, seed: int = 0) -> dict[str, Path]:
    """Synthetic arithmetic rows for every stage; the same bytes on every machine."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "sft": [
            {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"Working it out step by step, the answer is {a}."},
                ]
            }
            for q, a in _problems(512, seed)
        ],
        "dpo": [
            {
                "prompt": [{"role": "user", "content": q}],
                "chosen": [{"role": "assistant", "content": f"The answer is {a}."}],
                "rejected": [{"role": "assistant", "content": f"The answer is {a + (i % 7) + 1}."}],
                "chat_template_kwargs": NO_THINKING,
            }
            for i, (q, a) in enumerate(_problems(256, seed + 1))
        ],
        "opd": [
            {"messages": [{"role": "user", "content": f"{q} Explain briefly."}], "chat_template_kwargs": NO_THINKING}
            for q, _ in _problems(256, seed + 2)
        ],
        "rl": [
            {"prompt": f"{q} Put the final answer in \\boxed{{}}.", "answer": str(a)}
            for q, a in _problems(512, seed + 3)
        ],
    }
    paths = {}
    for stage, stage_rows in rows.items():
        paths[stage] = data_dir / f"{stage}.jsonl"
        paths[stage].write_text("".join(json.dumps(r) + "\n" for r in stage_rows))
    return paths


# ------------------------------------------------------------------- configs


def _merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and not key.startswith("="):
            _merge(base[key], value)
        else:
            base[key.lstrip("=")] = value
    return base


def stage_overrides(stage: str, args, data: dict[str, Path], out: Path) -> dict:
    """Overrides on the stage's base config. A key starting with "=" replaces the whole section."""
    run_dir = str(out / "runs" / stage)
    sft_like_train = {
        "output_dir": run_dir,
        "max_steps": args.steps,
        "save_steps": 1_000_000,
        "save_final": False,
        "logging_steps": 1,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 1,
    }
    logging = {"use_wandb": False, "use_trackio": False}
    if stage == "sft":
        return {
            "model": {"name_or_path": args.student, "attn_implementation": args.attn},
            "data": {
                "dataset": str(data["sft"]),
                "dataset_split": "train",
                "streaming": False,
                "max_seq_length": 512,
                "num_workers": 2,
            },
            "train": {**sft_like_train, "per_device_batch_size": 8, "learning_rate": 1e-5},
            "logging": logging,
        }
    if stage == "dpo":
        return {
            "model": {"name_or_path": args.student, "attn_implementation": args.attn, "torch_dtype": "float32"},
            "data": {
                "dataset": str(data["dpo"]),
                "dataset_split": "train",
                "max_seq_length": 512,
                "eval_dataset": "",
                "num_workers": 2,
            },
            "dpo": {"reference_model": ""},
            "train": {**sft_like_train, "per_device_batch_size": 4, "epochs": 1},
            "logging": logging,
        }
    if stage == "opd":
        return {
            "model": {"student": args.student, "stop_tokens": [], "chat_template_kwargs": NO_THINKING},
            "=teachers": {"teacher": {"model": args.teacher}},
            "=sources": {
                "chat": {"format": "messages", "path": str(data["opd"]), "dev_size": 32, "max_new_tokens": 128}
            },
            "rollout": {"backend": args.rollout, "batch_prompts": 8, "group_size": 2},
            "train": {
                "output_dir": run_dir,
                "steps": args.steps,
                "warmup_steps": 2,
                "eval_every": 0,
                "eval_samples": 32,
                "save_steps": 0,
                "score_micro_seqs": 4,
            },
            "logging": {"log_every": 1, "use_wandb": False},
        }
    if stage == "rl":
        return {
            "model": {"policy": args.student, "chat_template_kwargs": NO_THINKING},
            "data": {"dataset": str(data["rl"]), "eval_dataset": ""},
            "rollout": {
                "backend": args.rollout,
                "batch_prompts": 8,
                "group_size": 4,
                "max_new_tokens": 256,
                "max_model_len": 1024,
            },
            "train": {
                "output_dir": run_dir,
                "steps": args.steps,
                "warmup_steps": 2,
                "eval_every": 0,
                "save_steps": 0,
                "save_final": False,
            },
            "logging": {"log_every": 1, "use_wandb": False},
        }
    raise ValueError(stage)


def write_config(stage: str, args, data: dict[str, Path], out: Path) -> Path:
    base = yaml.safe_load((ROOT / BASE_CONFIGS[stage]).read_text())
    config = _merge(base, stage_overrides(stage, args, data, out))
    path = out / "configs" / f"{stage}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def validate_config(stage: str, path: Path) -> None:
    """Load and validate the config as the trainer will (no GPU, no model download)."""
    if stage in ("sft", "dpo"):
        from palingenesis.config import Config

        Config.from_yaml(path).validate()
    elif stage == "opd":
        from palingenesis.opd.config import OPDConfig

        OPDConfig.from_yaml(path).validate()
    else:
        from palingenesis.rl.config import RLConfig

        RLConfig.from_yaml(path).validate()


# ---------------------------------------------------------------------- runs


def command(stage: str, config: Path) -> list[str]:
    if stage in ("sft", "dpo"):
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"] + [
            "-m",
            "palingenesis.train",
            "--config",
            str(config),
        ]
    cli = "from palingenesis.cli import main; main()"
    return [sys.executable, "-c", cli, "distill" if stage == "opd" else "rl", "--config", str(config)]


_SFT_LINE = re.compile(r"step=(\d+) loss=([-\d.eE+naif]+).*?tok/s=(\d+)")
_KV_LINE = re.compile(r"\bstep (\d+) \| (.*)$")


def parse_log(stage: str, text: str) -> dict:
    """Losses and throughput from a stage's log: SFT/DPO step lines, OPD/RL `step N | k=v` lines."""
    losses, speed, extra = [], [], {}
    for line in text.splitlines():
        if stage in ("sft", "dpo"):
            if m := _SFT_LINE.search(line):
                losses.append(float(m.group(2)))
                speed.append(float(m.group(3)))
        elif m := _KV_LINE.search(line):
            kv = dict(item.split("=", 1) for item in m.group(2).split() if "=" in item)
            if "loss" in kv:
                losses.append(float(kv["loss"]))
            if "time/step" in kv:
                speed.append(float(kv["time/step"]))
            for key in ("reward", "rollout_tok_s", "k1", "policy/mismatch_k3"):
                if key in kv:
                    extra.setdefault(key, []).append(float(kv[key]))
    later = speed[len(speed) // 2 :]  # the first steps include compilation and warm-up
    summary = {
        "steps_logged": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    summary["tok_s_median" if stage in ("sft", "dpo") else "s_per_step_median"] = (
        statistics.median(later) if later else None
    )
    for key, values in extra.items():
        summary[f"{key}_median"] = statistics.median(values[len(values) // 2 :])
    return summary


def environment() -> dict:
    import torch

    from palingenesis.accelerator import describe

    return {"torch": torch.__version__, "gpu": describe()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--student", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher", default="Qwen/Qwen3-1.7B", help="OPD teacher (same tokenizer as the student)")
    p.add_argument("--attn", default="sdpa", help="model.attn_implementation for SFT and DPO")
    p.add_argument("--rollout", default="hf", choices=("hf", "vllm", "vllm_server"), help="OPD/RL rollout backend")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dry-run", action="store_true", help="write data and configs and validate them; run nothing")
    args = p.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if unknown := set(stages) - set(STAGES):
        p.error(f"unknown stages {sorted(unknown)}; choose from {', '.join(STAGES)}")

    out = args.out.resolve()
    data = write_data(out / "data")
    configs = {stage: write_config(stage, args, data, out) for stage in stages}
    for stage, path in configs.items():
        validate_config(stage, path)
    if args.dry_run:
        print(json.dumps({stage: str(path) for stage, path in configs.items()}, indent=2))
        return 0

    results = {"environment": environment(), "args": {k: str(v) for k, v in vars(args).items()}, "stages": {}}
    (out / "logs").mkdir(parents=True, exist_ok=True)
    failed = False
    for stage in stages:
        log = out / "logs" / f"{stage}.log"
        print(f"== {stage}: {' '.join(command(stage, configs[stage]))}\n   log: {log}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            code = subprocess.run(
                command(stage, configs[stage]), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT
            ).returncode
        result = {"exit_code": code, "wall_s": round(time.time() - t0, 1), **parse_log(stage, log.read_text())}
        failed |= code != 0 or not result["steps_logged"]
        results["stages"][stage] = result
        print(f"   {json.dumps(result)}", flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
