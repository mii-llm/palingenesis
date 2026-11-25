"""GPU checks of train.packing (palingenesis.tree_packing) in the RL trainer: padded vs varlen vs tree.

Called by scripts/gpu_checks/rl_packing.sh; every result is one JSON line on stdout
({"check": ..., "status": "PASS" | "FAIL" | "INFO", ...numbers}).

  env                       versions and which kernels the packed paths will use
  equality   --model M      per-token log-probs and RL-step gradients, padded / varlen / tree, against each sequence
                            forwarded alone (bf16 autocast, as the trainer runs)
  throughput --model M      RL-step forward/backward time, tokens/s and peak memory per packing mode, on a recorded
             --scenario S   batch of synthetic multi-turn groups (median of --repeats timed steps after a warm-up)
  smoke-config --mode P     writes an RL config + dataset for a 20-step `pgs rl` smoke run of one packing mode
  smoke-report --log F      reads a smoke run's log: steps reached, finite losses, step times

Scenarios (groups of 8 rollouts, a system prompt shared by every group):
  rl         a task prompt per group, rollouts diverging at their first sampled token, 1-4 assistant turns with
             tool results between them
  branching  per group, two shared two-turn histories, each continued by four rollouts (tree search, onset
             branches, re-sampled final turns)
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

DEVICE = "cuda:0"  # --device cpu: a dry run of the script's logic (tiny models; no flash kernels)


def sync(empty: bool = False) -> None:
    import torch

    if torch.cuda.is_available() and DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
        if empty:
            torch.cuda.empty_cache()


def emit(check: str, status: str, **values) -> None:
    print(json.dumps({"check": check, "status": status, **values}), flush=True)


# ── environment ────────────────────────────────────────────────────────────────


def env_report() -> None:
    import torch
    import transformers

    from palingenesis.tree_packing import verify_flash_tree

    def has(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:
            return False

    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "flash_linear_attention": has("fla"),
        "causal_conv1d": has("causal_conv1d"),
        "flash_attn_package": has("flash_attn"),
        "liger_kernel": has("liger_kernel"),
        # without the flash-attn package, varlen attention runs torch's own FlashAttention-2
        # (aten._flash_attention_forward / _backward), the kernel seco._varlen_attention uses for SFT packing
        "varlen_attention": "aten._flash_attention_forward (torch's FlashAttention-2)",
    }
    emit("env", "INFO", **info)
    try:
        verify_flash_tree("cuda:0", torch.bfloat16)
        from palingenesis.tree_packing import _VERIFIED

        errors = next(v for k, v in _VERIFIED.items() if k[1] == torch.bfloat16)
        emit("tree_attention_selfcheck", "PASS", max_errors=errors)
    except Exception as error:  # noqa: BLE001
        emit("tree_attention_selfcheck", "FAIL", error=str(error)[:500])


# ── synthetic batches ──────────────────────────────────────────────────────────


def make_groups(scenario: str, groups: int, vocab: int, seed: int):
    from palingenesis.rl.trajectory import Trajectory

    rng = random.Random(seed)

    def tokens(n):
        return [rng.randrange(1000, vocab - 1000) for _ in range(n)]

    system = tokens(400)
    out = []
    for g in range(groups):
        prompt = system + tokens(rng.randint(200, 1500))
        group = []
        if scenario == "rl":
            for _ in range(8):
                t = Trajectory({}, g, prompt, [])
                turns = rng.randint(1, 4)
                for k in range(turns):
                    n = rng.randint(100, 600)
                    t.append_generated(tokens(n), [-1.0] * n, 0)
                    if k < turns - 1:
                        t.append_context(tokens(rng.randint(50, 400)))
                group.append(t)
        elif scenario == "branching":
            for _ in range(2):
                history = []
                for _ in range(2):
                    history.append(("gen", tokens(rng.randint(100, 500))))
                    history.append(("ctx", tokens(rng.randint(50, 300))))
                for _ in range(4):
                    t = Trajectory({}, g, prompt, [])
                    for kind, ids in history:
                        if kind == "gen":
                            t.append_generated(list(ids), [-1.0] * len(ids), 0)
                        else:
                            t.append_context(list(ids))
                    n = rng.randint(100, 600)
                    t.append_generated(tokens(n), [-1.0] * n, 0)
                    group.append(t)
        else:
            raise SystemExit(f"unknown scenario {scenario}")
        for t in group:
            t.reward = rng.random()
        out.append(group)
    return out


def raw_tokens(groups) -> int:
    return sum(len(t.prompt_ids) + len(t.tokens) - 1 for g in groups for t in g)


# ── a trainer without engines ──────────────────────────────────────────────────


def load_model(name: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from palingenesis.kernels import apply_liger_kernel, model_type_of

    try:
        apply_liger_kernel(model_type_of(AutoConfig.from_pretrained(name)))
    except Exception as error:  # noqa: BLE001 - Liger is optional
        emit("liger", "INFO", model=name, applied=False, error=str(error)[:200])
    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEVICE)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    return model


def make_trainer(model, packing: str, micro_tokens: int, prompt_sharing: str = "off"):
    from palingenesis.logits import output_head
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.losses import LowPrecisionWeight
    from palingenesis.rl.parallel import Parallel
    from palingenesis.rl.trainer import RLTrainer
    from palingenesis.seco import use_chunk_attention

    trainer = object.__new__(RLTrainer)
    trainer.config = RLConfig()
    trainer.config.set("train.packing", packing)
    trainer.config.set("train.prompt_sharing", prompt_sharing)
    trainer.config.set("train.micro_tokens", micro_tokens)
    trainer.config.set("loss.seq_mask", 0.0)
    trainer.parallel, trainer.device, trainer.pad_id = Parallel(fsdp=False), DEVICE, 0
    trainer.model = model
    use_chunk_attention(model)
    trainer.head, trainer.head_weight = output_head(model), LowPrecisionWeight()
    trainer.__dict__.pop("_packing", None)
    return trainer


# ── equality ───────────────────────────────────────────────────────────────────


def logprobs_of(model, trainer, groups, packing: str):
    """Per trajectory: the policy's log-prob of each of its tokens, fp32 head, bf16 autocast."""
    import torch

    from palingenesis.logits import final_hidden_states
    from palingenesis.opd.teachers import right_pad
    from palingenesis.rl.losses import target_logprobs
    from palingenesis.tree_packing import pack

    trajectories = [t for g in groups for t in g]
    out = {}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.startswith("cuda")):
        if packing == "alone":
            batches = [[t] for t in trajectories]
        else:
            batches = trainer._micro_batches(trajectories)
        for micro in batches:
            sequences = [t.prompt_ids + t.tokens[:-1] for t in micro]
            if packing in ("alone", "padded"):
                ids, _ = right_pad(sequences, 0, DEVICE)
                states = final_hidden_states(model, ids, None)
                rows = [states[r, len(t.prompt_ids) - 1 : len(sequences[r])] for r, t in enumerate(micro)]
            else:
                packed = pack(sequences, share=packing == "tree")
                kwargs = packed.forward_kwargs(DEVICE)
                states = trainer._packed_forward(trainer._packing_support(), packed, kwargs.pop("input_ids"), kwargs)
                rows = []
                for r, t in enumerate(micro):
                    pos = torch.from_numpy(packed.positions(r)[len(t.prompt_ids) - 1 :]).to(DEVICE)
                    rows.append(states[0, pos])
            for t, h in zip(micro, rows):
                lp, _ = target_logprobs(h.float(), trainer.head, torch.tensor(t.tokens, device=DEVICE), False)
                out[id(t)] = lp.float().cpu()
    model.train()
    return out


def step_grads(model, trainer, groups):
    model.zero_grad(set_to_none=True)
    metrics = trainer._train_step(groups)
    sync()
    grads = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return metrics, grads


def compare_grads(ref: dict, got: dict) -> dict:
    """Relative L2 error and cosine of two gradients (per-parameter tensors on the host, compared in fp32)."""
    num = den = dot = norm_got = 0.0
    worst = (0.0, "")
    for n, g in ref.items():
        g, h = g.float(), got[n].float()
        diff = float((h - g).pow(2).sum())
        size = float(g.pow(2).sum())
        num, den, dot, norm_got = num + diff, den + size, dot + float((h * g).sum()), norm_got + float(h.pow(2).sum())
        if size > 0:
            worst = max(worst, (math.sqrt(diff / size), n))
    norm_got = math.sqrt(norm_got)
    return {
        "grad_rel_l2": math.sqrt(num / max(den, 1e-30)),
        "grad_cosine": dot / max(math.sqrt(den) * norm_got, 1e-30),
        "worst_param_rel_l2": worst[0],
        "worst_param": worst[1],
    }


def equality(model_name: str, groups_count: int) -> None:
    import torch

    model = load_model(model_name)
    vocab = model.config.get_text_config().vocab_size
    groups = make_groups("branching", groups_count, vocab, seed=0) + make_groups("rl", groups_count, vocab, seed=1)
    for i, g in enumerate(groups):
        for t in g:
            t.group = i
    base = make_trainer(model, "padded", 16384)
    reference = logprobs_of(model, base, groups, "alone")
    for t in (t for g in groups for t in g):  # behaviour log-probs: the policy's own (ratios near 1, real gradients)
        t.logprobs = [lp if m else 0.0 for lp, m in zip(reference[id(t)].tolist(), t.mask)]
    results = {}
    for packing, micro in (("padded", 16384), ("padded_alt", 6144), ("varlen", 16384), ("tree", 16384)):
        mode = "padded" if packing.startswith("padded") else packing
        trainer = make_trainer(model, mode, micro)
        lps = logprobs_of(model, trainer, groups, mode)
        diffs = torch.cat([(lps[k] - reference[k]).abs() for k in reference])
        metrics, grads = step_grads(model, trainer, groups)
        if packing == "padded":
            base_grads = grads
            results[packing] = (diffs, metrics, None)
        else:  # compared now: one host copy of the gradients at a time besides the padded one
            results[packing] = (diffs, metrics, compare_grads(base_grads, grads))
        del grads
    base_metrics = results["padded"][1]
    floor = results["padded_alt"][2]
    floor_lp = float(results["padded"][0].max())
    for packing in ("padded_alt", "varlen", "tree"):
        diffs, metrics, cmp = results[packing]
        values = {
            "model": model_name,
            "mode": packing,
            "logprob_max_abs_diff_vs_alone": float(diffs.max()),
            "logprob_mean_abs_diff_vs_alone": float(diffs.mean()),
            "padded_logprob_max_abs_diff_vs_alone": floor_lp,
            "loss_diff": abs(metrics["loss"] - base_metrics["loss"]),
            "noise_floor_grad_rel_l2": floor["grad_rel_l2"],
            **cmp,
        }
        if packing == "padded_alt":
            emit("equality_noise_floor", "INFO", **values)
            continue
        # bf16: packed must stay within the noise between two padded layouts (other batch shapes and kernels)
        ok = (
            values["logprob_max_abs_diff_vs_alone"] <= max(3 * floor_lp, 0.05)
            and cmp["grad_rel_l2"] <= max(3 * floor["grad_rel_l2"], 0.02)
            and cmp["grad_cosine"] >= 0.999
        )
        emit("equality", "PASS" if ok else "FAIL", **values)
    del model
    sync(empty=True)


# ── throughput ─────────────────────────────────────────────────────────────────


def throughput(model_name: str, scenario: str, groups_count: int, repeats: int, micro_tokens: int) -> None:
    import torch

    from palingenesis.tree_packing import pack

    model = load_model(model_name)
    vocab = model.config.get_text_config().vocab_size
    groups = make_groups(scenario, groups_count, vocab, seed=2)
    trajectories = [t for g in groups for t in g]
    raw = raw_tokens(groups)
    sampled = sum(t.sampled_tokens for t in trajectories)
    # padded_shared: padded rows with train.prompt_sharing on (each group's prompt once, palingenesis.seco_tree), what
    # train.prompt_sharing auto picks for long prompts on one GPU; its micro-batch count covers the padded rest only
    for packing in ("padded", "padded_shared", "varlen", "tree"):
        shared = packing == "padded_shared"
        trainer = make_trainer(model, "padded" if shared else packing, micro_tokens, "on" if shared else "off")
        micro = trainer._micro_batches(trajectories)
        if shared:
            forwarded = sum(len(g[0].prompt_ids) - 1 + sum(len(t.tokens) for t in g) for g in groups)
        elif packing == "padded":
            forwarded = sum(max(t.context_tokens - 1 for t in m) * len(m) for m in micro)
        else:
            forwarded = sum(
                pack([t.prompt_ids + t.tokens[:-1] for t in m], share=packing == "tree").num_tokens
                for m in micro
            )
        try:
            step_grads(model, trainer, groups)  # warm-up (autotuning, allocator)
            times = []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                model.zero_grad(set_to_none=True)
                sync()
                t0 = time.perf_counter()
                trainer._train_step(groups)
                sync()
                times.append(time.perf_counter() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            median = statistics.median(times)
            emit(
                "throughput",
                "INFO",
                model=model_name,
                scenario=scenario,
                mode=packing,
                micro_tokens=micro_tokens,
                micro_batches=len(micro),
                raw_tokens=raw,
                forwarded_tokens=forwarded,
                sampled_tokens=sampled,
                step_s_median=median,
                step_s_all=times,
                raw_tok_s=raw / median,
                sampled_tok_s=sampled / median,
                peak_gib=peak,
            )
        except torch.OutOfMemoryError as error:
            emit("throughput", "FAIL", model=model_name, scenario=scenario, mode=packing, error=str(error)[:300])
        model.zero_grad(set_to_none=True)
        sync(empty=True)


# ── smoke runs ─────────────────────────────────────────────────────────────────


def smoke_config(work: str, mode: str, model: str, steps: int) -> None:
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, "smoke_rows.jsonl")
    if not os.path.exists(data):
        rng = random.Random(0)
        with open(data, "w") as f:
            for _ in range(512):
                a, b = rng.randint(10, 99), rng.randint(10, 99)
                prompt = f"What is {a} + {b}? Think briefly, then give the final answer as \\boxed{{n}}."
                f.write(json.dumps({"prompt": prompt, "answer": str(a + b)}) + "\n")
    path = os.path.join(work, f"smoke_{mode}.yaml")
    with open(path, "w") as f:
        f.write(
            f"""model:
  policy: {model}
  chat_template_kwargs: {{enable_thinking: false}}
data:
  dataset: {data}
rollout:
  backend: vllm
  batch_prompts: 8
  group_size: 8
  max_new_tokens: 256
  max_model_len: 1024
  gpu_memory_utilization: 0.3
rewards:
  correct: {{fn: math, args: {{field: answer}}}}
train:
  output_dir: {os.path.join(work, "smoke_" + mode)}
  steps: {steps}
  learning_rate: 1.0e-6
  packing: {mode}
  save_final: false
  seed: 0
logging:
  log_every: 1
"""
        )
    print(path)


def smoke_report(log: str, mode: str, steps: int) -> None:
    import re

    lines = [line for line in open(log, errors="replace") if re.search(r"\bstep (\d+) \|", line)]
    reached, losses, step_s, train_s = 0, [], [], []
    for line in lines:
        reached = max(reached, int(re.search(r"\bstep (\d+) \|", line).group(1)))
        for key, bucket in (("loss", losses), ("time/step", step_s), ("time/train", train_s)):
            m = re.search(rf"(?<![\w/]){re.escape(key)}=([-+0-9.eE]+|nan|inf)", line)
            if m:
                bucket.append(float(m.group(1)))
    ok = reached >= steps and losses and all(math.isfinite(v) for v in losses)
    tail = "".join(open(log, errors="replace").readlines()[-5:]) if not ok else ""
    emit(
        "smoke",
        "PASS" if ok else "FAIL",
        mode=mode,
        steps_reached=reached,
        losses_finite=bool(losses) and all(math.isfinite(v) for v in losses),
        time_step_median=statistics.median(step_s[1:]) if len(step_s) > 1 else None,
        time_train_median=statistics.median(train_s[1:]) if len(train_s) > 1 else None,
        log_tail=tail[-800:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["env", "equality", "throughput", "smoke-config", "smoke-report"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--scenario", default="rl")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--micro_tokens", type=int, default=16384)
    parser.add_argument("--work", default=".")
    parser.add_argument("--mode", default="padded")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log", default="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    if args.part == "env":
        env_report()
    elif args.part == "equality":
        equality(args.model, args.groups)
    elif args.part == "throughput":
        throughput(args.model, args.scenario, args.groups, args.repeats, args.micro_tokens)
    elif args.part == "smoke-config":
        smoke_config(args.work, args.mode, args.model, args.steps)
    else:
        smoke_report(args.log, args.mode, args.steps)


if __name__ == "__main__":
    sys.exit(main())
