"""The GPU checks of the SGLang rollout backend; run by scripts/gpu_checks/sglang.sh, one step per process.

Each subcommand writes a JSON file (--out) that `report` turns into the JSON lines the shell script prints. The
generation steps run in the SGLang environment or (--engine vllm) in a vLLM one; the scoring, weight-update and
sleep steps need SGLang. Nothing here is imported by palingenesis.

    python sglang_checks.py generate --engine sglang --model Qwen/Qwen3-0.6B --out gen.json
    python sglang_checks.py score --model Qwen/Qwen3-0.6B --inputs gen.json --out score.json
    python sglang_checks.py update --model Qwen/Qwen3-0.6B --mode colocated --out update.json
    python sglang_checks.py sleep --model Qwen/Qwen3-0.6B --out sleep.json
    python sglang_checks.py throughput --engine sglang --model Qwen/Qwen3-0.6B --out tp.json
    python sglang_checks.py smoke-config --kind rl --model ... --workdir ... --out rl.yaml
    python sglang_checks.py report --check parity --inputs a.json b.json ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

QUESTIONS = [
    "What is 17 * 23? Answer with the number only.",
    "Name the capital of Australia and one fact about it.",
    "Write a haiku about rain on a tin roof.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
    "Translate 'good morning, how are you?' into Italian.",
    "What is the derivative of x^3 + 2x? Show one step.",
    "Give a Python one-liner that reverses a string s.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "If a train travels 120 km in 1.5 hours, what is its average speed?",
    "What does HTTP stand for?",
    "Describe a cat to someone who has never seen one, in 30 words.",
    "Is 2027 a prime number? Explain briefly.",
    "Write the word 'banana' backwards.",
    "What is the boiling point of water at sea level in Fahrenheit?",
    "Suggest a name for a bakery run by robots.",
]


def _emit(path: str, record: dict) -> None:
    with open(path, "w") as f:
        json.dump(record, f)


def _prompts(model: str, n: int):
    from transformers import AutoTokenizer

    from palingenesis.opd.trainer import student_stop_ids

    tok = AutoTokenizer.from_pretrained(model)
    kwargs = {"enable_thinking": False}
    prompts = []
    for k in range(n):
        question = QUESTIONS[k % len(QUESTIONS)] + ("" if k < len(QUESTIONS) else f" (variant {k})")
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, **kwargs
        )
        prompts.append(tok(text, add_special_tokens=False)["input_ids"])
    return tok, prompts, tuple(student_stop_ids(tok, model, [], kwargs))


def _engine(kind: str, model: str, stop_ids, memory: float, sleep: bool = False, max_len: int = 2048):
    if kind == "vllm":
        from palingenesis.opd.rollout import VLLMColocateRollout, precision_options

        return VLLMColocateRollout(
            model,
            stop_ids,
            memory,
            max_len,
            enforce_eager=False,
            seed=0,
            sleep_mode=sleep,
            prefix_caching=True,
            max_num_seqs=256,
            engine_kwargs=precision_options(False, "float32"),
        )
    from palingenesis.opd.sglang_rollout import SGLangRollout

    return SGLangRollout(
        model,
        stop_ids,
        gpu_memory_utilization=memory,
        max_model_len=max_len,
        seed=0,
        sleep_mode=sleep,
        prefix_caching=True,
        max_num_seqs=0,  # SGLang sizes it (hybrid models: from their recurrent-state slots)
    )


def _close(engine) -> None:
    close = getattr(engine, "close", None)
    if close is not None:
        close()


def _trainer_model(model: str):
    """The policy as the RL trainer holds it: fp32 master weights on the GPU, computing in bf16 (autocast)."""
    import torch

    from palingenesis.kernels import match_vllm_gated_norm, model_type_of
    from palingenesis.opd.teachers import load_causal_lm

    match_vllm_gated_norm(model_type_of(model))
    return load_causal_lm(model, torch.float32).to("cuda").eval()


def _trainer_logprobs(policy, prompt: list[int], tokens: list[int], temperature: float = 1.0) -> list[float]:
    import torch

    ids = torch.tensor([prompt + tokens], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = policy(ids).logits[0, len(prompt) - 1 : -1].float() / temperature
    return torch.log_softmax(logits, -1).gather(1, ids[0, len(prompt) :, None]).squeeze(1).tolist()


TRUNCATED = (0.7, 0.8, 20)  # temperature, top_p, top_k of the truncated generation


def _truncated_logprobs(policy, prompt: list[int], tokens: list[int], engine: str) -> list[float]:
    """The trainer's log-probs of `tokens` under the truncated sampling distribution, renormalised over the support
    as the engine builds it. vLLM (and palingenesis' truncate_scores): top-k first, the nucleus on the renormalised
    rest. SGLang: both on the same sorted probabilities (the nucleus is not renormalised after top-k). -inf: a
    token the trainer puts outside the support."""
    import torch

    from palingenesis.opd.rollout import truncate_scores

    temperature, top_p, top_k = TRUNCATED
    ids = torch.tensor([prompt + tokens], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = policy(ids).logits[0, len(prompt) - 1 : -1].float() / temperature
    if engine == "vllm":
        scores = truncate_scores(logits, top_k, top_p)
    else:
        probs, order = torch.softmax(logits, -1).sort(dim=-1, descending=True)
        keep = torch.arange(probs.shape[-1], device=probs.device)[None] < top_k
        keep &= (probs.cumsum(-1) - probs) <= top_p
        scores = logits.masked_fill(~torch.zeros_like(keep).scatter(-1, order, keep), float("-inf"))
    return torch.log_softmax(scores, -1).gather(1, ids[0, len(prompt) :, None]).squeeze(1).tolist()


def _mismatch(engine_lp: list[float], trainer_lp: list[float]) -> tuple[float, float, int]:
    """(mean |delta log-prob|, mean k3 = exp(d) - 1 - d with d = trainer - engine, tokens)."""
    deltas = [t - e for e, t in zip(engine_lp, trainer_lp)]
    if not deltas:
        return math.nan, math.nan, 0
    return (
        sum(abs(d) for d in deltas) / len(deltas),
        sum(math.exp(d) - 1 - d for d in deltas) / len(deltas),
        len(deltas),
    )


# ------------------------------------------------------------------ steps


def generate(args) -> None:
    """Greedy and sampled completions of fixed prompts with one engine, and its tokens / log-probs."""
    from palingenesis.opd.rollout import Sampling

    _, prompts, stop_ids = _prompts(args.model, args.prompts)
    engine = _engine(args.engine, args.model, stop_ids, args.memory)
    try:
        greedy = engine.generate(prompts, [args.max_new_tokens] * len(prompts), 0.0)
        repeated = [p for p in prompts for _ in range(args.samples)]
        sampled = engine.generate(repeated, [args.max_new_tokens] * len(repeated), Sampling(1.0))
        truncated = engine.generate(prompts, [args.max_new_tokens] * len(prompts), Sampling(*TRUNCATED))
    finally:
        _close(engine)
    _emit(
        args.out,
        {
            "engine": args.engine,
            "model": args.model,
            "greedy": [r.completion_ids for r in greedy],
            "sampled": [
                {"prompt": p, "tokens": r.completion_ids, "logprobs": r.logprobs, "finish": r.finish_reason}
                for p, r in zip(repeated, sampled)
            ],
            "truncated": [
                {"prompt": p, "tokens": r.completion_ids, "logprobs": r.logprobs} for p, r in zip(prompts, truncated)
            ],
        },
    )


def score(args) -> None:
    """The trainer's log-probs of each engine's sampled tokens: mean |delta| and k3 per engine."""
    policy = _trainer_model(args.model)
    result = {}
    for path in args.inputs:
        data = json.load(open(path))
        engine_lp, trainer_lp = [], []
        for item in data["sampled"]:
            engine_lp += item["logprobs"]
            trainer_lp += _trainer_logprobs(policy, item["prompt"], item["tokens"])
        mean_abs, k3, tokens = _mismatch(engine_lp, trainer_lp)
        # truncated sampling: the engine's log-probs must be the renormalised ones of its own support
        pairs = [
            (e, t)
            for item in data["truncated"]
            for e, t in zip(
                item["logprobs"], _truncated_logprobs(policy, item["prompt"], item["tokens"], data["engine"])
            )
        ]
        inside = [(e, t) for e, t in pairs if math.isfinite(t) and math.isfinite(e)]
        lengths = [len(item["tokens"]) for item in data["sampled"]]
        result[data["engine"]] = {
            "mean_abs_dlogprob": mean_abs,
            "k3": k3,
            "tokens": tokens,
            "logprobs_per_token": all(len(i["logprobs"]) == len(i["tokens"]) for i in data["sampled"]),
            "mean_len": sum(lengths) / max(len(lengths), 1),
            "greedy": data["greedy"],
            "truncated_lp_mean": statistics.fmean(
                [x for item in data["truncated"] for x in item["logprobs"]] or [math.nan]
            ),
            "truncated_logprobs_per_token": all(len(i["logprobs"]) == len(i["tokens"]) for i in data["truncated"]),
            "truncated_mean_abs_dlogprob": _mismatch([e for e, _ in inside], [t for _, t in inside])[0],
            "truncated_outside_support_share": 1 - len(inside) / max(len(pairs), 1),
        }
    _emit(args.out, {"model": args.model, "engines": result})


def _perturbed(policy, scale: float = 0.05, seed: int = 0):
    """Add seeded noise (scale x each tensor's std) to every floating weight, in place."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for p in policy.parameters():
            if p.is_floating_point() and p.numel() > 1:
                noise = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
                p.add_(noise * scale * p.std())


def _same(a: list[list[int]], b: list[list[int]]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(len(a), 1)


def update(args) -> None:
    """A weight update changes the outputs, the engine samples the new weights' log-probs, and pushing the original
    weights back restores the original greedy outputs (radix cache on: stale prefixes would show)."""
    import torch

    from palingenesis.opd.rollout import Sampling, checkpoint_named_parameters

    _, prompts, stop_ids = _prompts(args.model, args.prompts)
    budgets = [args.max_new_tokens] * len(prompts)
    if args.mode == "colocated":
        engine = _engine("sglang", args.model, stop_ids, args.memory)
    else:
        from palingenesis.opd.sglang_rollout import SGLangServer, SGLangServerRollout, engine_options

        options = engine_options(args.model, gpu_memory_utilization=args.memory, max_model_len=2048, seed=0)
        options["base_gpu_id"] = torch.cuda.current_device()
        server = SGLangServer(
            args.model, options=options, log_path=os.path.join(args.workdir, f"server_{args.mode}.log")
        )
        engine = SGLangServerRollout(
            server,
            stop_ids,
            model=args.model,
            max_model_len=2048,
            weight_sync=args.mode,
            sync_dir=os.path.join(args.workdir, "weights_sync"),
        )
    policy = _trainer_model(args.model)
    original = {n: p.detach().clone() for n, p in policy.named_parameters()}
    record: dict = {"model": args.model, "mode": args.mode}
    try:
        before = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        _perturbed(policy)
        start = time.perf_counter()
        engine.update_weights(checkpoint_named_parameters(policy), 1)
        record["update_seconds"] = time.perf_counter() - start
        changed = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        sampled = engine.generate(prompts, budgets, Sampling(1.0))
        engine_lp, new_lp = [], []
        for prompt, r in zip(prompts, sampled):
            engine_lp += r.logprobs
            new_lp += _trainer_logprobs(policy, prompt, r.completion_ids)
        with torch.no_grad():
            for n, p in policy.named_parameters():
                p.copy_(original[n])
        old_lp = []
        for prompt, r in zip(prompts, sampled):
            old_lp += _trainer_logprobs(policy, prompt, r.completion_ids)
        engine.update_weights(checkpoint_named_parameters(policy), 2)
        restored = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        record.update(
            {
                "changed_share": 1 - _same(before, changed),
                "restored_share": _same(before, restored),
                "mean_abs_dlogprob_new_weights": _mismatch(engine_lp, new_lp)[0],
                "mean_abs_dlogprob_old_weights": _mismatch(engine_lp, old_lp)[0],
                "k3_new_weights": _mismatch(engine_lp, new_lp)[1],
                "tokens": len(engine_lp),
                "versions": sorted({r.policy_version for r in sampled}),
            }
        )
    finally:
        _close(engine)
        if args.mode != "colocated":
            server.close()
    _emit(args.out, record)


def _used_gib() -> float:
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def sleep_check(args) -> None:
    """Colocated sleep: the engine releases its weights, KV cache and CUDA graphs, and generates the same greedy
    completions after an update and a wake."""
    import torch

    from palingenesis.opd.rollout import checkpoint_named_parameters

    _, prompts, stop_ids = _prompts(args.model, args.prompts)
    budgets = [args.max_new_tokens] * len(prompts)
    torch.cuda.init()
    idle = _used_gib()
    engine = _engine("sglang", args.model, stop_ids, args.memory, sleep=True)
    record: dict = {"model": args.model, "gpu_memory_utilization": args.memory}
    try:
        awake = _used_gib()
        before = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        engine.sleep()
        asleep = _used_gib()
        policy = _trainer_model(args.model)
        with_policy = _used_gib()
        start = time.perf_counter()
        engine.update_weights(checkpoint_named_parameters(policy), 1)
        engine.wake()
        record["update_wake_seconds"] = time.perf_counter() - start
        woke = _used_gib()
        after = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        # a second cycle, as every training step does: what the first resume left must release and come back again
        engine.sleep()
        asleep_again = _used_gib()
        engine.update_weights(checkpoint_named_parameters(policy), 2)
        engine.wake()
        again = [r.completion_ids for r in engine.generate(prompts, budgets, 0.0)]
        record.update(
            {
                "idle_gib": idle,
                "awake_gib": awake,
                "asleep_gib": asleep,
                "after_wake_gib": woke,
                "trainer_gib": with_policy - asleep,
                "released_share": (awake - asleep) / max(awake - idle, 1e-9),
                "same_greedy_share": _same(before, after),
                "asleep_again_gib": asleep_again,
                "released_again_share": (woke - asleep_again) / max(woke - with_policy + asleep - idle, 1e-9),
                "same_greedy_again_share": _same(before, again),
            }
        )
    finally:
        _close(engine)
    _emit(args.out, record)


def throughput(args) -> None:
    """Decode throughput at an equal batch: tokens/s over a few runs, median."""
    from palingenesis.opd.rollout import Sampling

    _, prompts, stop_ids = _prompts(args.model, args.batch)
    engine = _engine(args.engine, args.model, stop_ids, args.memory)
    runs = []
    try:
        engine.generate(prompts[:8], [32] * 8, Sampling(1.0))  # warm-up
        for _ in range(args.runs):
            start = time.perf_counter()
            out = engine.generate(prompts, [args.max_new_tokens] * len(prompts), Sampling(1.0))
            seconds = time.perf_counter() - start
            runs.append(sum(len(r.completion_ids) for r in out) / seconds)
    finally:
        _close(engine)
    _emit(
        args.out, {"engine": args.engine, "model": args.model, "tokens_per_s": runs, "median": statistics.median(runs)}
    )


def smoke_config(args) -> None:
    """A 10-step pgs rl or pgs distill config with the sglang backend, its data and a JSONL metrics tracker."""
    import random

    os.makedirs(args.workdir, exist_ok=True)
    rng = random.Random(0)
    rows = []
    for _ in range(64):
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        rows.append({"prompt": f"What is {a} + {b}? Put the final answer in \\boxed{{}}.", "answer": str(a + b)})
    data = os.path.join(args.workdir, "arith.jsonl")
    with open(data, "w") as f:
        for row in rows:
            f.write(
                json.dumps(row if args.kind == "rl" else {"messages": [{"role": "user", "content": row["prompt"]}]})
            )
            f.write("\n")
    tracker = os.path.join(args.workdir, "metrics_tracker.py")
    with open(tracker, "w") as f:
        f.write(
            "import json, os\n"
            "class JsonlTracker:\n"
            "    def __init__(self, *, project, name, config, run_id, output_dir, resuming):\n"
            "        self.path = os.path.join(output_dir, 'metrics.jsonl')\n"
            "    def log(self, metrics, step):\n"
            "        with open(self.path, 'a') as f:\n"
            "            f.write(json.dumps({'step': step, **{k: v for k, v in metrics.items()"
            " if isinstance(v, (int, float))}}) + '\\n')\n"
            "    def finish(self, exit_code):\n"
            "        pass\n"
        )
    output = os.path.join(args.workdir, f"{args.kind}_run")
    if args.kind == "rl":
        config = {
            "model": {"policy": args.model, "chat_template_kwargs": {"enable_thinking": False}},
            "data": {"dataset": data},
            "rollout": {
                "backend": args.backend,
                "batch_prompts": 4,
                "group_size": 4,
                "max_new_tokens": 192,
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.3,
            },
            "rewards": {"correct": {"fn": "math", "args": {"field": "answer"}}},
            "train": {"output_dir": output, "steps": 10, "learning_rate": 1.0e-6, "eval_every": 0, "save_steps": 0},
            "logging": {"trackers": [f"{tracker}:JsonlTracker"]},
        }
    else:
        config = {
            "model": {"student": args.model, "chat_template_kwargs": {"enable_thinking": False}},
            "teachers": {"teacher": {"model": args.teacher}},
            "sources": {"arith": {"format": "messages", "path": data, "max_new_tokens": 192}},
            "rollout": {
                "backend": args.backend,
                "batch_prompts": 8,
                "gpu_memory_utilization": 0.3,
                "max_model_len": 1024,
            },
            "train": {"output_dir": output, "steps": 10, "learning_rate": 1.0e-6, "eval_every": 0, "save_steps": 0},
            "logging": {"trackers": [f"{tracker}:JsonlTracker"]},
        }
    import yaml

    with open(args.out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# ------------------------------------------------------------------ report


def _line(check: str, ok: bool, **numbers) -> None:
    print(json.dumps({"check": check, "status": "PASS" if ok else "FAIL", **numbers}), flush=True)


def report(args) -> None:
    data = [json.load(open(p)) for p in args.inputs if os.path.exists(p)]
    if args.check == "parity":
        (scores,) = data
        engines = scores["engines"]
        sg = engines.get("sglang")
        vl = engines.get("vllm")
        numbers = {"model": scores["model"], "sglang": {k: v for k, v in sg.items() if k != "greedy"}}
        ok = sg["logprobs_per_token"] and sg["mean_abs_dlogprob"] < args.max_delta
        # truncated (top-k / top-p): renormalised log-probs for every token, close to the trainer's over the same
        # support (looser: a token at the nucleus' edge moves the normaliser of its position)
        ok = ok and sg["truncated_logprobs_per_token"] and sg["truncated_outside_support_share"] <= 0.05
        ok = ok and sg["truncated_mean_abs_dlogprob"] < 3 * args.max_delta
        if vl is not None:
            prefix = []
            for a, b in zip(sg["greedy"], vl["greedy"]):
                same = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                prefix.append(same / max(len(a), len(b), 1))
            numbers["vllm"] = {k: v for k, v in vl.items() if k != "greedy"}
            numbers["greedy_identical_share"] = _same(sg["greedy"], vl["greedy"])
            numbers["greedy_common_prefix"] = sum(prefix) / max(len(prefix), 1)
            ok = ok and numbers["greedy_common_prefix"] >= 0.6
            ok = ok and sg["mean_abs_dlogprob"] <= 1.5 * vl["mean_abs_dlogprob"] + 0.01
        else:
            numbers["vllm"] = "skipped (no VLLM_PYTHON)"
        _line("1_generation_logprob_parity", ok, **numbers)
    elif args.check == "update":
        for record in data:
            ok = (
                record["changed_share"] >= 0.5
                and record["restored_share"] >= 0.75
                and record["mean_abs_dlogprob_new_weights"] < args.max_delta
                and record["mean_abs_dlogprob_new_weights"] < 0.5 * record["mean_abs_dlogprob_old_weights"]
            )
            _line(f"2_weight_update_{record['mode']}", ok, **record)
    elif args.check == "sleep":
        (record,) = data
        ok = record["released_share"] >= 0.8 and record["same_greedy_share"] >= 0.75
        ok = ok and record["released_again_share"] >= 0.8 and record["same_greedy_again_share"] >= 0.75
        _line("3_sleep_wake", ok, **record)
    elif args.check in ("smoke_rl", "smoke_distill"):
        rows = []
        for path in args.inputs:
            if os.path.exists(path):
                rows = [json.loads(line) for line in open(path)]
        steps = sorted({r.get("step") for r in rows if r.get("step") is not None})
        last = rows[-1] if rows else {}
        keys = [k for k in last if any(s in k for s in ("loss", "kl", "mismatch", "reward", "tok_s", "time/sync"))]
        finite = all(math.isfinite(v) for r in rows for k, v in r.items() if isinstance(v, float))
        ok = args.exit_code == 0 and len(steps) >= 10 and finite
        first = next((r for r in rows if "policy/mismatch_k3_step0" in r), {})
        _line(
            f"4_{args.check}",
            ok,
            exit_code=args.exit_code,
            steps_logged=len(steps),
            finite=finite,
            last={k: last[k] for k in keys[:12]},
            **({"mismatch_k3_step0": first["policy/mismatch_k3_step0"]} if first else {}),
        )
    elif args.check == "throughput":
        by = {d["engine"]: d for d in data}
        sg, vl = by.get("sglang"), by.get("vllm")
        numbers = {"model": sg["model"] if sg else "", "sglang_tok_s": sg and sg["median"]}
        if vl is None:
            _line("5_throughput", sg is not None, **numbers, vllm_tok_s="skipped (no VLLM_PYTHON)")
            return
        ratio = sg["median"] / vl["median"]
        _line("5_throughput", ratio >= args.min_ratio, **numbers, vllm_tok_s=vl["median"], ratio=ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "score", "update", "sleep", "throughput", "smoke-config", "report"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="Qwen/Qwen3-0.6B")
        p.add_argument("--out", default="")
        p.add_argument("--inputs", nargs="*", default=[])
        p.add_argument("--engine", default="sglang", choices=("sglang", "vllm"))
        p.add_argument("--memory", type=float, default=0.4)
        p.add_argument("--prompts", type=int, default=16)
        p.add_argument("--samples", type=int, default=4)
        p.add_argument("--max-new-tokens", type=int, default=128)
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--mode", default="colocated", choices=("colocated", "tensor", "disk"))
        p.add_argument("--workdir", default=".")
        p.add_argument("--kind", default="rl", choices=("rl", "distill"))
        p.add_argument("--backend", default="sglang")
        p.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
        p.add_argument("--check", default="")
        p.add_argument("--exit-code", type=int, default=0)
        p.add_argument("--max-delta", type=float, default=0.08)
        p.add_argument("--min-ratio", type=float, default=0.7)
    args = parser.parse_args()
    if args.command == "throughput" and args.max_new_tokens == 128:
        args.max_new_tokens = 256
    {
        "generate": generate,
        "score": score,
        "update": update,
        "sleep": sleep_check,
        "throughput": throughput,
        "smoke-config": smoke_config,
        "report": report,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
