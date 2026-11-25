"""Difficulty screening of an RL prompt pool: k rollouts per row with the policy, scored by the RL config's own rewards
and environment, and the rows the policy solves sometimes (0 < pass rate < 1) kept.

A group whose rollouts all succeed or all fail carries no gradient in group-relative RL (the advantages are 0), so
rows the policy always or never solves cost rollouts and teach nothing; dynamic sampling (rollout.max_refill) drops
such groups during training, after paying for them. Screening once before the run spends k rollouts per row instead
and gives the share of the pool that can teach the policy, per family. Used by Tülu 3's RLVR data (prompts the SFT
model always solves removed), Polaris and Skywork-OR1 (pass-rate bands), DAPO / ScaleRL in-run (zero-variance
filtering, retirement).

    pgs screen --config configs/rl_math.yaml --k 8 --output pool_screened.jsonl
    pgs screen --config rl.yaml --k 16 --max_rate 0.9 --model runs/sft/final --rollout.backend hf

Rollouts go through the RL pipeline itself (palingenesis.rl.pipeline): the same prompt rendering, tool calls,
environments, harness profiles, budgets and rewards as training, at rollout.temperature unless --temperature is given
(and rollout.top_p / top_k).
The policy is model.policy unless --model is given. Every row of data.dataset is screened (--split eval: the held-out
rows, data.eval_dataset or the eval_size split; all: both), --rows N takes the first N.

Outputs:
    <output>              every screened row, with a "screen" field: samples, scored, successes, pass_rate,
                          mean_reward, wilson_lo / wilson_hi (95%), finishes {stop: n, length: n, ...},
                          completion_tokens (mean), kept. Appended chunk by chunk: re-running the same command
                          skips the rows already in it (resume), matched by content.
    <output>.kept.jsonl   the kept rows as they were (no screen field), ready for data.dataset.
A summary follows (stdout, or --json): rows by band (unsolved / partial / solved / failed), the kept share, the
pass-rate histogram and, with data.metrics_key, the same per family. Exit code 0, or 1 when nothing is kept.

A row is kept when --min_rate < pass rate < --max_rate (defaults 0 and 1: solved sometimes, never always). A
success is a reward >= data.success_threshold. Rows whose rollouts all failed to score (environment or verifier
errors) are reported as failed and not kept. With k samples a row at true pass rate p is all-fail with probability
(1 - p)^k: at k = 8 a p = 0.1 row looks unsolved 43% of the time, so screen hard pools with larger k or keep a share
of unsolved rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palingenesis.evalstats import wilson_interval

logger = logging.getLogger(__name__)

BANDS = ("unsolved", "partial", "solved", "failed")
HISTOGRAM_EDGES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def row_key(row: dict) -> str:
    from palingenesis.rl.data import _key

    return _key({k: v for k, v in row.items() if k != "screen"})


def group_stats(group: list, threshold: float) -> dict[str, Any]:
    """One row's screen record from its k trajectories (t.scored, t.reward, t.finish, t.sampled_tokens)."""
    scored = [t for t in group if t.scored]
    successes = sum(t.reward >= threshold for t in scored)
    record: dict[str, Any] = {
        "samples": len(group),
        "scored": len(scored),
        "successes": successes,
        "pass_rate": successes / len(scored) if scored else None,
        "mean_reward": sum(t.reward for t in scored) / len(scored) if scored else None,
        "finishes": dict(Counter(t.finish or "unknown" for t in group)),
        "completion_tokens": sum(t.sampled_tokens for t in group) / max(1, len(group)),
    }
    lo, hi = wilson_interval(successes, len(scored))
    record["wilson_lo"], record["wilson_hi"] = (lo, hi) if scored else (None, None)
    return record


def band(record: dict) -> str:
    rate = record["pass_rate"]
    if rate is None:
        return "failed"
    return "unsolved" if rate == 0 else "solved" if rate == 1 else "partial"


def keep(record: dict, min_rate: float, max_rate: float) -> bool:
    rate = record["pass_rate"]
    return rate is not None and min_rate < rate < max_rate


def summarize(records: list[tuple[dict, dict]], families: list[str] | None = None) -> dict[str, Any]:
    """Counts by band, kept share and pass-rate histogram over (row, screen record) pairs; per family with
    `families` (data.metrics_key)."""

    def block(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
        bands = Counter(band(r) for _, r in pairs)
        rates = [r["pass_rate"] for _, r in pairs if r["pass_rate"] is not None]
        histogram = [0] * (len(HISTOGRAM_EDGES) - 1)
        for rate in rates:
            i = next(i for i in range(len(histogram)) if rate <= HISTOGRAM_EDGES[i + 1] or i == len(histogram) - 1)
            histogram[i] += 1
        kept = sum(bool(r.get("kept")) for _, r in pairs)
        return {
            "rows": len(pairs),
            **{b: bands.get(b, 0) for b in BANDS},
            "kept": kept,
            "kept_share": kept / len(pairs) if pairs else 0.0,
            "mean_pass_rate": sum(rates) / len(rates) if rates else None,
            "histogram": {
                f"({HISTOGRAM_EDGES[i]:g}, {HISTOGRAM_EDGES[i + 1]:g}]" if i else f"[0, {HISTOGRAM_EDGES[1]:g}]": n
                for i, n in enumerate(histogram)
            },
        }

    out = block(records)
    if families:
        from palingenesis.rl.pipeline import row_family

        by: dict[str, list] = defaultdict(list)
        for row, record in records:
            by[row_family(row, families)].append((row, record))
        out["families"] = {name: block(pairs) for name, pairs in sorted(by.items())}
    return out


def screen(
    pipeline,
    rows: list[dict],
    k: int,
    temperature,
    threshold: float,
    min_rate: float = 0.0,
    max_rate: float = 1.0,
    chunk: int = 256,
    sink=None,
) -> list[tuple[dict, dict]]:
    """Screen `rows` with an RLPipeline: k rollouts each, `chunk` rows at a time; `sink(row, record)` is called for
    every finished row (in order). Returns (row, record) pairs; record["kept"] says whether the row is kept."""
    out = []
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        groups = pipeline._await(pipeline._eval(part, temperature, k))
        for row, group in zip(part, groups):
            record = group_stats(group, threshold)
            record["kept"] = keep(record, min_rate, max_rate)
            out.append((row, record))
            if sink is not None:
                sink(row, record)
        logger.info("screened %d / %d rows", min(start + chunk, len(rows)), len(rows))
    return out


# ------------------------------------------------------------------------- CLI


def _env_pool(config, sandbox):
    """The RL config's environment pool (None for single-turn), as the trainer builds it."""
    from palingenesis.rl.env import EnvPool, ToolEnv
    from palingenesis.rl.rewards import load_object
    from palingenesis.rl.trainer import _parameters

    e = config.env
    if e.type == "single_turn":
        return None
    if e.type == "tools":
        functions = [load_object(spec) for spec in e.tools]
        return EnvPool(lambda: ToolEnv(functions), max_concurrent=e.max_concurrent, allowed=e.allowed_tools)
    factory = load_object(e.type)
    args = dict(e.args)
    if "sandbox" in _parameters(factory) and "sandbox" not in args:
        args["sandbox"] = sandbox
    return EnvPool(factory, args, max_concurrent=e.max_concurrent, allowed=e.allowed_tools)


def _needs_sandbox(config, rewards) -> bool:
    from palingenesis.rl.rewards import code_reward, load_object
    from palingenesis.rl.trainer import _parameters

    if any(r.fn is code_reward or "sandbox" in _parameters(r.fn) for r in rewards):
        return True
    e = config.env
    if e.type in ("single_turn", "tools") or "sandbox" in e.args:
        return False
    return "sandbox" in _parameters(load_object(e.type))


def _engine(config, model: str, stop_ids, pad_id: int):
    from palingenesis.rl.config import max_num_seqs

    r = config.rollout
    if r.backend == "hf":
        import torch

        from palingenesis.opd.rollout import HFRollout
        from palingenesis.opd.teachers import load_causal_lm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = load_causal_lm(model, torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        return HFRollout(policy, stop_ids, pad_id, r.micro_seqs)
    if r.backend == "vllm":
        from palingenesis.opd.rollout import VLLMColocateRollout, precision_options

        extra = {k: v for k, v in (("kv_cache_dtype", r.kv_cache_dtype), ("quantization", r.quantization)) if v}
        extra.update(r.vllm_args)
        extra.update(precision_options(r.fp32_head, r.ssm_state_dtype, r.vllm_args.get("hf_overrides")))
        return VLLMColocateRollout(
            model,
            stop_ids,
            r.gpu_memory_utilization,
            r.max_model_len,
            r.enforce_eager,
            config.train.seed,
            sleep_mode=False,
            prefix_caching=r.prefix_caching,
            max_num_seqs=max_num_seqs(config),
            engine_kwargs=extra,
        )
    if r.backend == "sglang":
        from palingenesis.opd.sglang_rollout import SGLangRollout

        return SGLangRollout(
            model,
            stop_ids,
            gpu_memory_utilization=r.gpu_memory_utilization,
            max_model_len=r.max_model_len,
            enforce_eager=r.enforce_eager,
            seed=config.train.seed,
            prefix_caching=r.prefix_caching,
            max_num_seqs=r.max_num_seqs,
            fp32_head=r.fp32_head,
            ssm_state_dtype=r.ssm_state_dtype,
            kv_cache_dtype=r.kv_cache_dtype,
            quantization=r.quantization,
            engine_kwargs=r.sglang_args,
        )
    raise SystemExit(
        f"pgs screen: rollout.backend {r.backend!r} is not supported here; use vllm or sglang (in-process) or hf "
        "(--rollout.backend vllm)."
    )


def _rows(config, split: str) -> list[dict]:
    from palingenesis.rl.data import load_rows, split_rows
    from palingenesis.rl.formats import convert

    d = config.data
    rows = convert(load_rows(d.dataset, d.split), d.format)
    if d.eval_dataset:
        train, held = rows, convert(load_rows(d.eval_dataset, d.eval_split), d.format)
    else:
        train, held = split_rows(rows, d.eval_size)
    return {"train": train, "eval": held, "all": train + held}[split]


def main(argv: list[str] | None = None) -> int:
    from palingenesis.logging import setup_logging

    parser = argparse.ArgumentParser(
        prog="pgs screen",
        description="RL pool difficulty screening: k rollouts per row, keep rows with min_rate < pass rate < max_rate. "
        "Other --section.option arguments override the RL config.",
    )
    parser.add_argument("--config", required=True, help="the RL config (model, data, rewards, env, rollout)")
    parser.add_argument("--k", type=int, default=8, help="rollouts per row")
    parser.add_argument("--output", required=True, help="annotated rows (JSONL); kept rows go to <output>.kept.jsonl")
    parser.add_argument("--model", default="", help="the checkpoint to screen with (default: model.policy)")
    parser.add_argument("--temperature", type=float, default=None, help="default: rollout.temperature")
    parser.add_argument("--min_rate", type=float, default=0.0, help="keep rows with pass rate > this")
    parser.add_argument("--max_rate", type=float, default=1.0, help="keep rows with pass rate < this")
    parser.add_argument("--split", choices=("train", "eval", "all"), default="train")
    parser.add_argument("--rows", type=int, default=0, help="screen the first N rows only (0 = all)")
    parser.add_argument("--chunk", type=int, default=256, help="rows per batch of rollouts (and per write)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args, overrides = parser.parse_known_args(argv)
    setup_logging(rank=0)
    if args.k < 1 or args.chunk < 1 or args.rows < 0:
        parser.error("--k and --chunk must be >= 1, --rows >= 0")
    if not 0 <= args.min_rate < args.max_rate:
        parser.error("--min_rate must be >= 0 and below --max_rate (above 1 keeps the always-solved rows too)")

    from transformers import AutoTokenizer

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.opd.trainer import student_stop_ids
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler, check_row_limits
    from palingenesis.rl.pipeline import RLPipeline
    from palingenesis.rl.rewards import resolve_rewards
    from palingenesis.rl.sandbox import make_sandbox
    from palingenesis.rl.trainer import _think_tags

    config = RLConfig.from_cli(["--config", args.config, *overrides])
    for warning in config.validate():
        logger.warning(warning)
    if config.scenario.type:
        parser.error("multi-role scenarios are not screened by this tool (pgs scenario audit checks their rows)")
    if config.model.multimodal:
        parser.error("multimodal policies are not supported by pgs screen yet")
    m = config.model
    model = args.model or m.policy
    rows = _rows(config, args.split)
    if args.rows:
        rows = rows[: args.rows]
    check_row_limits(rows, config.data.dataset, config.completion_budget)

    output = Path(args.output)
    done: dict[str, tuple[dict, dict]] = {}
    if output.exists():
        with output.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row_key(row)] = ({k: v for k, v in row.items() if k != "screen"}, row["screen"])
    todo = [row for row in rows if row_key(row) not in done]
    logger.info("%d rows to screen (%d already in %s), k = %d", len(todo), len(rows) - len(todo), output, args.k)

    tok = AutoTokenizer.from_pretrained(model)
    kwargs = m.chat_template_kwargs
    stop_ids = student_stop_ids(tok, model, m.stop_tokens, kwargs)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]
    chat = ChatFormat(tok, end_of_turn_id(tok, kwargs), kwargs, _think_tags(tok, kwargs))
    rewards = resolve_rewards(config.rewards)
    sandbox = make_sandbox(config.sandbox) if _needs_sandbox(config, rewards) else None
    env_pool = _env_pool(config, sandbox)
    if not rewards and not (env_pool and env_pool.has_reward):
        parser.error("no reward: configure rewards: in the RL config, or use an environment with get_reward")

    import torch

    engine = _engine(config, model, stop_ids, pad_id) if todo else None
    temperature = config.rollout.temperature if args.temperature is None else args.temperature
    threshold = config.data.success_threshold
    pipeline = None
    try:
        if todo:
            pipeline = RLPipeline(
                tok,
                chat,
                engine,
                PublishedWeights(torch.nn.Linear(1, 1)),
                config,
                PromptSampler(todo),
                rewards,
                env_pool,
                sandbox,
                stop_ids,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as f:

                def sink(row, record):
                    f.write(json.dumps({**row, "screen": record}, ensure_ascii=False, default=str) + "\n")
                    f.flush()

                sampling = temperature
                if config.rollout.top_p < 1 or config.rollout.top_k > 0:  # truncated as in training
                    from palingenesis.opd.rollout import Sampling

                    sampling = Sampling(temperature, config.rollout.top_p, config.rollout.top_k)
                fresh = screen(pipeline, todo, args.k, sampling, threshold, args.min_rate, args.max_rate,
                               args.chunk, sink)  # fmt: skip
            for row, record in fresh:
                done[row_key(row)] = (row, record)
    finally:
        if pipeline is not None:
            pipeline.close()
    records = []
    for row in rows:
        kept_row, record = done[row_key(row)]
        record = {**record, "kept": keep(record, args.min_rate, args.max_rate)}  # bands may differ from a resumed run
        records.append((kept_row, record))
    kept_path = output.with_name(output.name.removesuffix(".jsonl") + ".kept.jsonl")
    with kept_path.open("w") as f:
        for row, record in records:
            if record["kept"]:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = summarize(records, config.data.metrics_key)
    summary.update({"k": args.k, "model": model, "temperature": temperature, "kept_file": str(kept_path)})
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{summary['rows']} rows, k = {args.k}: unsolved {summary['unsolved']}, partial {summary['partial']}, "
            f"solved {summary['solved']}, failed {summary['failed']}; kept {summary['kept']} "
            f"({summary['kept_share']:.1%}) -> {kept_path}"
        )
        for edges, n in summary["histogram"].items():
            print(f"  pass rate {edges:<14} {n}")
        for name, s in summary.get("families", {}).items():
            print(f"  {name}: {s['rows']} rows, kept {s['kept']} ({s['kept_share']:.1%}), mean pass rate "
                  f"{s['mean_pass_rate'] if s['mean_pass_rate'] is None else round(s['mean_pass_rate'], 3)}")  # fmt: skip
    return 0 if summary["kept"] else 1


if __name__ == "__main__":
    sys.exit(main())
