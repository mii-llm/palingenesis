#!/usr/bin/env python3
"""The fastest memory settings of a training config that fit a GPU memory budget, measured.

Usage:
    pgs fit --config configs/long_context.yaml                 # budget: this GPU
    pgs fit --config configs/long_context.yaml --gpu 40        # budget: 40 GiB
    pgs fit --config my.yaml --first-fit                       # stop at the first setting that fits

Long sequences are bounded by activation memory, and the settings that lower it cost
speed: activation checkpointing (none -> selective -> full) recomputes part or all of
every layer in the backward, and SeCO (memory.seco) runs the sequence chunk by chunk.
Which one is the fastest that still fits depends on the model, the sequence length and
the GPU, so this tool measures: for each setting, the real trainer runs a few optimizer
steps on worst-case rows (every row at data.max_seq_length, as `pgs profile --measure`
does) in its own process, and reports the peak allocated memory and the throughput. The
recommendation is the fastest setting whose peak stays under the budget.

Exit code 0 when a setting fits, 1 when none does.
"""

import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field

from palingenesis.config import Config, ConfigError

RESULT_PREFIX = "FIT_RESULT "
SECO_CHUNKS = (8192, 4096, 2048, 1024)


@dataclass
class Candidate:
    """One memory setting: its overrides of the config as `section.option` -> value."""

    name: str
    overrides: dict[str, object] = field(default_factory=dict)
    skipped: str = ""  # why it is not measured (a combination the config refuses)
    result: dict | None = None

    def flags(self) -> list[str]:
        """The overrides as command-line options of `pgs train`."""
        out: list[str] = []
        for key, value in self.overrides.items():
            out += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
        return out

    @property
    def peak_gb(self) -> float | None:
        return self.result.get("measured_peak_gb") if self.result else None

    @property
    def tokens_per_sec(self) -> float | None:
        return self.result.get("tokens_per_sec") if self.result else None

    def fits(self, limit_gb: float) -> bool:
        return self.peak_gb is not None and self.peak_gb <= limit_gb


def apply_overrides(config: Config, overrides: dict[str, object]) -> Config:
    """A copy of `config` with the overrides set."""
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        section, option = key.split(".", 1)
        setattr(getattr(out, section), option, value)
    return out


def candidates(config: Config) -> list[Candidate]:
    """The settings to measure, from the one expected to be fastest to the most frugal: the config as it is,
    the three activation-checkpointing modes, then SeCO at decreasing chunk sizes and with its K/V cache
    offloaded. Settings the config cannot take (SeCO with DPO, ...) are listed as skipped."""
    out = [Candidate("as configured")]
    seco_on = config.memory.seco
    for mode in ("none", "selective", "full"):
        if seco_on or mode != config.train.gradient_checkpointing:
            overrides: dict[str, object] = {"train.gradient_checkpointing": mode}
            if seco_on:
                overrides["memory.seco"] = False
            out.append(Candidate(f"{mode} checkpointing", overrides))
    chunks = [c for c in SECO_CHUNKS if c < config.data.max_seq_length]
    for chunk in chunks:
        if seco_on and chunk == config.memory.seco_chunk_size and not config.memory.seco_kv_offload:
            continue  # the config as it is
        out.append(Candidate(f"SeCO, {chunk}-token chunks", {"memory.seco": True, "memory.seco_chunk_size": chunk}))
    if chunks:
        out.append(
            Candidate(
                f"SeCO, {chunks[-1]}-token chunks, K/V offloaded",
                {"memory.seco": True, "memory.seco_chunk_size": chunks[-1], "memory.seco_kv_offload": True},
            )
        )
    for c in out:
        try:
            apply_overrides(config, c.overrides).validate()
        except ConfigError as e:
            lines = [line.strip(" ✗") for line in str(e).splitlines()[1:]]
            c.skipped = lines[0] if lines else str(e)
    return out


def recommend(measured: list[Candidate], limit_gb: float) -> Candidate | None:
    """The fastest candidate whose peak is within the limit (None when none fits)."""
    fitting = [c for c in measured if c.fits(limit_gb) and c.tokens_per_sec]
    return max(fitting, key=lambda c: c.tokens_per_sec) if fitting else None


def measure_in_subprocess(config_path: str, extra_args: list[str], candidate: Candidate, steps: int) -> dict:
    """Measure one candidate in its own process: an out-of-memory error ends that process only, and every
    candidate starts from an empty allocator."""
    command = [
        sys.executable,
        "-m",
        "palingenesis.agent_tooling.fit_memory",
        "--config",
        config_path,
        "--measure-one",
        "--steps",
        str(steps),
        *extra_args,
        *candidate.flags(),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
    return {"error": f"exit code {proc.returncode}: {tail[0][:200]}"}


def _measure_one(config: Config, steps: int) -> dict:
    """In the child process: the peak and throughput of this config, or why it did not run."""
    import torch

    from palingenesis.agent_tooling.profile_memory import measure_memory

    try:
        return measure_memory(config, steps=steps)
    except torch.OutOfMemoryError:
        return {"oom": True, "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}
    except Exception as e:  # noqa: BLE001 - reported per candidate; the others still run
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def table(cands: list[Candidate], limit_gb: float, best: Candidate | None) -> str:
    rows = [f"  {'setting':<42} {'peak GiB':>9} {'tok/s':>9}  verdict"]
    for c in cands:
        if c.skipped:
            verdict, peak, speed = f"not applicable: {c.skipped}", "-", "-"
        elif c.result is None:
            verdict, peak, speed = "not measured", "-", "-"
        elif c.result.get("oom"):
            verdict, peak, speed = "out of memory on this GPU", "OOM", "-"
        elif c.result.get("error"):
            verdict, peak, speed = f"failed: {c.result['error']}", "-", "-"
        else:
            peak, speed = f"{c.peak_gb:.1f}", f"{c.tokens_per_sec:.0f}" if c.tokens_per_sec else "-"
            verdict = "fits" if c.fits(limit_gb) else "over the budget"
            if c is best:
                verdict += "  <- fastest that fits"
        rows.append(f"  {c.name:<42} {peak:>9} {speed:>9}  {verdict}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gpu_memory_gb",
        "--gpu",
        type=float,
        default=0.0,
        dest="gpu_memory_gb",
        help="budget in GiB (default: this GPU; on a GPU that shares the host's RAM, the host memory available now)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.92,
        help="share of the budget the measured peak may use (default 0.92: allocator fragmentation and "
        "everything else on the GPU take the rest)",
    )
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps per setting (the last one is timed)")
    parser.add_argument("--first-fit", action="store_true", help="stop at the first setting that fits")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--measure-one", action="store_true", help=argparse.SUPPRESS)
    args, rest = parser.parse_known_args(argv)
    config = Config.from_cli(["--config", args.config, *rest])

    if args.measure_one:
        print(RESULT_PREFIX + json.dumps(_measure_one(config, args.steps)))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("pgs fit measures on a GPU: none is available")
    from palingenesis.accelerator import is_unified_memory, memory_budget_bytes

    budget_gb = args.gpu_memory_gb or memory_budget_bytes() / 2**30
    limit_gb = budget_gb * args.headroom
    progress = sys.stderr if args.json else sys.stdout
    if is_unified_memory() and not args.gpu_memory_gb:
        print(
            "Unified memory: the GPU allocates from the host's RAM, so the budget is the host memory available "
            "now (--gpu overrides it). The training process's own host memory is not in the measured peaks.",
            file=progress,
        )
    print(
        f"Budget {budget_gb:.0f} GiB (peak allowed: {limit_gb:.1f} GiB) | {config.model.name_or_path} | "
        f"{config.train.per_device_batch_size} x {config.data.max_seq_length} tokens per micro-batch",
        file=progress,
    )
    cands = candidates(config)
    for c in cands:
        if c.skipped:
            continue
        print(f"  measuring: {c.name} ...", file=progress, flush=True)
        c.result = measure_in_subprocess(args.config, rest, c, args.steps)
        if args.first_fit and c.fits(limit_gb):
            break
    best = recommend(cands, limit_gb)

    if args.json:
        report = {
            "budget_gb": budget_gb,
            "limit_gb": limit_gb,
            "candidates": [
                {"name": c.name, "overrides": c.overrides, "skipped": c.skipped, "result": c.result} for c in cands
            ],
            "recommended": best.name if best else None,
            "recommended_overrides": best.overrides if best else None,
        }
        print(json.dumps(report, indent=2))
        return 0 if best else 1

    print()
    print(table(cands, limit_gb, best))
    print()
    if best is None:
        print(
            "  No setting fits. Lower data.max_seq_length or train.per_device_batch_size (raise "
            "train.gradient_accumulation_steps to keep the effective batch), freeze part of the model "
            "(train.freeze_non_attention on hybrids), or split sequences over GPUs (parallel.context_parallel)."
        )
        return 1
    if best.overrides:
        print(f"  Recommended: {best.name}. Add to the command line:\n    {' '.join(best.flags())}")
        print("  or to the config:")
        for key, value in best.overrides.items():
            section, option = key.split(".", 1)
            print(f"    {section}.{option}: {str(value).lower() if isinstance(value, bool) else value}")
    else:
        print("  Recommended: the config as it is.")
    print(
        "  Throughput is measured on worst-case rows (every row at data.max_seq_length); shorter rows train "
        "faster in every setting."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
