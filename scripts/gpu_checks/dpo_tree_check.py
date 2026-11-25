"""GPU checks of dpo.packing (palingenesis.dpo.PackedPairs over palingenesis.tree_packing): padded [2P, S] rows
vs one packed row per micro-batch (varlen: no padding; tree: every pair's shared prefix forwarded once).

Called by scripts/gpu_checks/dpo_tree.sh; every result is one JSON line on stdout
({"check": ..., "status": "PASS" | "FAIL" | "INFO" | "SKIP", ...numbers}).

  env                       versions, the GPU's free memory, the kernels the packed paths will use
  equality   --model M      per-token / per-sequence log-probs, loss and gradients of one DPO micro-step, varlen and
                            tree against the padded rows (fp32 weights, bf16 autocast, selective checkpointing: as
                            the trainer runs); FAIL when the gradient cosine is below 0.999
  throughput --model M      micro-step time (reference forward + policy forward + loss + backward), pair tokens/s and
             --share S      peak memory per layout on synthesised pairs whose shared prompt is S of each sequence
             [--compiled]   (median of --repeats timed steps after a warm-up); --compiled: the compiled padded
                            baseline alone, in a process of its own
  verdict    --results F    the pre-registered verdict per model and prompt share, from the throughput lines
  smoke-config --mode P     writes a DPO config + pairs for a short `python -m palingenesis.train` run
  smoke-report --log F      reads a smoke run's log: steps reached, finite losses, step times

The micro-step is the trainer's (train.py's DPO branch): the same functions, denominators left at their defaults.
Layouts measured:
  padded            [2P, S] rows padded to a multiple of 64 with their attention mask, as the trainer forwards them
  padded_nomask     the same rows without the mask (right padding is inert under causal attention): the fastest
                    padded forward, measured so that the packed layouts are not compared with a slow baseline
  varlen / tree     one packed row (dpo.packing)
  padded_compiled   padded rows with model.compile (per-layer torch.compile, no Liger: the trainer's defaults),
                    which dpo.packing refuses: measured in its own process, on the same pairs
Pre-registered: tree >= 1.5x the fastest padded layout's step speed at prompt shares >= 70% (PASS); below 1.2x, or
a gradient cosine below 0.999, the feature is dropped (FAIL); in between, INFO with target_met false.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

DEVICE = "cuda:0"  # --device cpu: a dry run of the script's logic on a tiny random model (no flash kernels)
TARGET_SPEEDUP, KILL_SPEEDUP, TARGET_SHARE, MIN_COSINE = 1.5, 1.2, 0.7, 0.999


def cuda() -> bool:
    return DEVICE.startswith("cuda")


def sync(empty: bool = False) -> None:
    import torch

    if cuda() and torch.cuda.is_available():
        torch.cuda.synchronize()
        if empty:
            torch.cuda.empty_cache()


def emit(check: str, status: str, **values) -> None:
    print(json.dumps({"check": check, "status": status, **values}), flush=True)


def free_memory() -> tuple[float | None, float | None]:
    import torch

    if not (cuda() and torch.cuda.is_available()):
        return None, None
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


# ── environment ────────────────────────────────────────────────────────────────


def env_report() -> None:
    import torch
    import transformers

    def has(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:
            return False

    free, total = free_memory()
    emit(
        "env",
        "INFO",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_free_gib=free,
        gpu_total_gib=total,
        flash_linear_attention=has("fla"),
        causal_conv1d=has("causal_conv1d"),
        liger_kernel=has("liger_kernel"),
    )
    if not cuda():
        return
    try:
        from palingenesis.tree_packing import _VERIFIED, verify_flash_tree

        verify_flash_tree(DEVICE, torch.bfloat16)
        errors = next(v for k, v in _VERIFIED.items() if k[1] == torch.bfloat16)
        emit("tree_attention_selfcheck", "PASS", max_errors=errors)
    except Exception as error:  # noqa: BLE001
        emit("tree_attention_selfcheck", "FAIL", error=str(error)[:500])


# ── models, as the trainer prepares them for DPO ───────────────────────────────


def tiny_model():
    """--device cpu: a small random Qwen3 (the dry run checks the script, not the numbers)."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=4096,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32768,
    )
    return Qwen3ForCausalLM(config)


def load_models(name: str, checkpointing: str, compiled: bool = False):
    """(policy, reference, packing supports, vocabulary size): fp32 weights (model.torch_dtype float32, what DPO
    needs), Liger kernels, dropout off, activation checkpointing on the policy, attention on the fused sdpa route,
    both models checked and routed for packed rows. compiled: the trainer's model.compile setup instead (no
    Liger, the policy's layers compiled, nothing routed for packed rows: supports None)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from palingenesis.dpo import disable_dropout
    from palingenesis.kernels import apply_activation_checkpointing, apply_liger_kernel, model_type_of
    from palingenesis.seco import use_chunk_attention
    from palingenesis.tree_packing import prepare_model

    def load():
        if not cuda():
            return tiny_model()
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        for cfg in (model.config, getattr(model.config, "text_config", None)):
            if cfg is not None and hasattr(cfg, "use_cache"):
                cfg.use_cache = False
        return model

    if cuda() and not compiled:
        try:
            apply_liger_kernel(model_type_of(AutoConfig.from_pretrained(name)))
        except Exception as error:  # noqa: BLE001 - Liger is optional
            emit("liger", "INFO", model=name, applied=False, error=str(error)[:200])
    policy = load()
    reference = load().requires_grad_(False).eval()
    disable_dropout(policy)
    if checkpointing != "none":
        apply_activation_checkpointing(policy, mode=checkpointing)
    use_chunk_attention(policy)
    use_chunk_attention(reference)
    supports = None
    if not compiled:
        supports = prepare_model(policy, DEVICE), prepare_model(reference, DEVICE)
    policy.to(DEVICE).train()
    reference.to(DEVICE)
    if compiled:
        from palingenesis.train import _compile_layers

        _compile_layers(policy)
    return policy, reference, supports, policy.config.get_text_config().vocab_size


# ── synthesised pairs ──────────────────────────────────────────────────────────


def make_pairs(count: int, share: float, vocab: int, seed: int, edge_cases: bool = False) -> list[dict]:
    """Preference pairs as the data pipeline emits them (token ids, labels on the answers). A pair's prompt is
    `share` of its sequences on average: answers of 100-700 tokens (chosen 150-600, rejected 100-700: a rejected
    answer may ramble), the prompt sized to them, +-40%. Random token ids: speed and equality do not read text.
    edge_cases: two more pairs with an earlier assistant turn scored inside the prompt (data.last_turn_only off)
    and answers that start alike (minimal-edit pairs)."""
    import torch

    from palingenesis.data import IGNORE_INDEX

    rng = random.Random(seed)
    low, high = min(1000, vocab // 8), vocab - min(1000, vocab // 8)

    def tokens(n):
        return [rng.randrange(low, high) for _ in range(n)]

    def side(prompt, answer, scored=()):
        ids = torch.tensor(prompt + answer)
        labels = torch.full_like(ids, IGNORE_INDEX)
        labels[len(prompt) :] = ids[len(prompt) :]
        for a, b in scored:
            labels[a:b] = ids[a:b]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

    pairs = []
    for _ in range(count):
        chosen, rejected = tokens(rng.randint(150, 600)), tokens(rng.randint(100, 700))
        prompt = tokens(max(8, round(375 * share / (1 - share) * rng.uniform(0.6, 1.4))))
        pairs.append({"chosen": side(prompt, chosen), "rejected": side(prompt, rejected)})
    if edge_cases:
        for _ in range(2):
            first, turn, user = (
                tokens(rng.randint(200, 500)),
                tokens(rng.randint(60, 200)),
                tokens(rng.randint(50, 300)),
            )
            prompt, scored = first + turn + user, [(len(first), len(first) + len(turn))]
            start = tokens(rng.randint(20, 80))
            pairs.append(
                {
                    "chosen": side(prompt, start + tokens(rng.randint(40, 200)), scored),
                    "rejected": side(prompt, start + tokens(rng.randint(40, 200)), scored),
                }
            )
    return pairs


def collate(pairs: list[dict], multiple: int) -> dict:
    from palingenesis.dpo import collate_preferences

    batch = collate_preferences(pairs, 0, pad_to_multiple=multiple)
    return {k: v.to(DEVICE) for k, v in batch.items()}


# ── the DPO micro-step ─────────────────────────────────────────────────────────

STEP_OPTIONS = {"loss_type": "sigmoid", "beta": 0.1, "sft_weight": 0.2}  # configs/dpo_qwen35_08b.yaml


def dpo_step(policy, reference, supports, pairs: list[dict], layout: str, vocab: int, detail: bool = False) -> dict:
    """One micro-step as train.py's DPO branch runs it (reference log-probs, policy forward, loss, backward).
    Returns the loss and the layout's token counts; detail: the per-token policy log-probs [2P, S] too."""
    import torch

    from palingenesis.dpo import (
        PackedPairs,
        packed_reference_logps,
        preference_step,
        reference_logps,
        token_logps,
    )
    from palingenesis.loss import shift_labels
    from palingenesis.train import _dynamic_num_chunks, _get_hidden_states, _get_lm_head
    from palingenesis.tree_packing import forward_packed

    packed_layout = layout in ("varlen", "tree")
    batch = collate(pairs, 1 if packed_layout else 64)  # the trainer's collation for each layout
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    num_pairs = ids.shape[0] // 2
    head = _get_lm_head(policy)
    packed = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda()):
        if packed_layout:
            packed = PackedPairs.build(ids, mask, labels, share=layout == "tree")
            chunks = _dynamic_num_chunks(packed.num_terms, vocab)
            ref = packed_reference_logps(reference, supports[1], packed, _get_lm_head(reference), chunks)
            hidden = forward_packed(policy, supports[0], packed.row, DEVICE)
            forwarded = packed.row.num_tokens
        else:
            chunks = _dynamic_num_chunks(ids.numel(), vocab)
            forward_mask = None if layout == "padded_nomask" else mask
            ref = reference_logps(reference, _get_hidden_states, _get_lm_head, ids, forward_mask, labels, chunks)
            hidden = _get_hidden_states(policy, ids, forward_mask)
            forwarded = ids.numel()
        logps = None
        if detail:
            if packed is not None:
                logps = packed.token_logps(hidden.detach(), head, chunks)
            else:
                logps = token_logps(hidden.detach(), shift_labels(labels), head, chunks)
            logps = logps[:, : int(mask.sum(1).max())].float()  # compared across layouts: no padding multiple
        step = preference_step(
            hidden,
            labels,
            head,
            ref,
            num_pairs,
            num_chunks=chunks,
            sft_pair_weight=batch["sft_pair_weight"],
            packed=packed,
            **STEP_OPTIONS,
        )
    step.loss.backward()
    return {
        "loss": float(step.loss.detach()),
        "metrics": step.metrics,
        "forwarded_tokens": int(forwarded),
        "levels": len(packed.row.levels) if packed is not None else None,
        "logps": logps,
    }


def take_grads(model) -> list:
    """The parameters' gradients as fp32 clones (on the GPU), in parameter order; clears them."""
    grads = [p.grad.detach().float().clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    return grads


def compare_grads(ref: list, got: list) -> dict:
    """Relative L2 error and cosine of two gradients."""
    num = den = dot = norm = 0.0
    for g, h in zip(ref, got, strict=True):
        num += float((h - g).pow(2).sum())
        den += float(g.pow(2).sum())
        dot += float((h * g).sum())
        norm += float(h.pow(2).sum())
    return {
        "grad_rel_l2": math.sqrt(num / max(den, 1e-30)),
        "grad_cosine": dot / max(math.sqrt(den) * math.sqrt(norm), 1e-30),
    }


# ── equality ───────────────────────────────────────────────────────────────────


def equality(model_name: str, pairs_count: int, checkpointing: str) -> None:
    import torch

    free, _ = free_memory()  # read before anything is loaded
    policy, reference, supports, vocab = load_models(model_name, checkpointing)
    torch.manual_seed(0)
    with torch.no_grad():  # the policy off the reference: non-trivial ratios (1% relative noise on the matrices)
        for p in policy.parameters():
            if p.ndim >= 2:
                p.mul_(1 + 0.01 * torch.randn_like(p))
    pairs = make_pairs(pairs_count, 0.8, vocab, seed=0, edge_cases=True)
    results = {}
    # padded_alt: the same pairs in another padded layout (reversed order, no padding multiple, no mask): the
    # numerical noise between two padded forwards in bf16, which the packed layouts must stay within
    for layout in ("padded", "padded_alt", "varlen", "tree"):
        policy.zero_grad(set_to_none=True)
        if layout == "padded_alt":
            out = dpo_step(policy, reference, supports, pairs[::-1], "padded_nomask", vocab, detail=True)
            half = len(pairs)
            order = list(range(half - 1, -1, -1)) + list(range(2 * half - 1, half - 1, -1))
            out["logps"] = out["logps"][order]
        else:
            out = dpo_step(policy, reference, supports, pairs, layout, vocab, detail=True)
        sync()
        results[layout] = (out, take_grads(policy))
    base, base_grads = results["padded"]
    width = min(out["logps"].shape[1] for out, _ in results.values())
    floor = None
    for layout in ("padded_alt", "varlen", "tree"):
        out, grads = results[layout]
        diff = (out["logps"][:, :width] - base["logps"][:, :width]).abs()
        sequence = (out["logps"].sum(1) - base["logps"].sum(1)).abs()
        values = {
            "model": model_name,
            "layout": layout,
            "pairs": len(pairs),
            "checkpointing": checkpointing,
            "gpu_free_gib": free,
            "forwarded_tokens": out["forwarded_tokens"],
            "padded_tokens": base["forwarded_tokens"],
            "tree_levels": out["levels"],
            "token_logprob_max_abs_diff": float(diff.max()),
            "token_logprob_mean_abs_diff": float(diff.mean()),
            "sequence_logprob_max_abs_diff": float(sequence.max()),
            "loss": out["loss"],
            "loss_diff": abs(out["loss"] - base["loss"]),
            "margin_diff": abs(out["metrics"]["rewards/margins"] - base["metrics"]["rewards/margins"]),
            **compare_grads(base_grads, grads),
        }
        if layout == "padded_alt":
            floor = values
            emit("equality_noise_floor", "INFO", **values)
            continue
        ok = (
            values["grad_cosine"] >= MIN_COSINE
            and values["token_logprob_max_abs_diff"] <= max(3 * floor["token_logprob_max_abs_diff"], 0.05)
            and values["grad_rel_l2"] <= max(3 * floor["grad_rel_l2"], 0.02)
            and (layout != "tree" or not cuda() or out["levels"] == 2)  # the row really was a tree
        )
        emit(
            "equality",
            "PASS" if ok else "FAIL",
            **values,
            min_cosine=MIN_COSINE,
            noise_floor_grad_rel_l2=floor["grad_rel_l2"],
            noise_floor_token_logprob=floor["token_logprob_max_abs_diff"],
        )


# ── throughput ─────────────────────────────────────────────────────────────────


def throughput(
    model_name: str, share: float, padded_tokens: int, repeats: int, checkpointing: str, compiled: bool
) -> None:
    import torch

    free, total = free_memory()  # read before anything is loaded: someone else may hold part of the GPU
    if free is not None and free < 40 and padded_tokens > 8192:
        emit("throughput_budget", "INFO", model=model_name, gpu_free_gib=free, padded_tokens=padded_tokens // 2)
        padded_tokens //= 2
    policy, reference, supports, vocab = load_models(model_name, checkpointing, compiled)
    names = ("padded", "padded_nomask", "varlen", "tree")
    layouts = {"padded_compiled": "padded"} if compiled else {name: name for name in names}
    common = {
        "model": model_name,
        "prompt_share": share,
        "checkpointing": checkpointing,
        "device": DEVICE,
        "gpu_free_gib": free,
    }

    def measure(layout: str, pairs: list[dict], warmups: int = 1) -> dict:
        for _ in range(warmups):  # autotuning, allocator, compilation
            policy.zero_grad(set_to_none=True)
            dpo_step(policy, reference, supports, pairs, layout, vocab)
        sync()
        if cuda():
            torch.cuda.reset_peak_memory_stats()
        times, out = [], None
        for _ in range(repeats):
            policy.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            out = dpo_step(policy, reference, supports, pairs, layout, vocab)
            sync()
            times.append(time.perf_counter() - start)
        policy.zero_grad(set_to_none=True)
        median = statistics.median(times)
        tokens = sum(p[s]["input_ids"].numel() for p in pairs for s in ("chosen", "rejected"))
        return {
            "step_s_median": median,
            "step_s_all": times,
            "pair_tokens": tokens,
            "pair_tok_s": tokens / median,
            "forwarded_tokens": out["forwarded_tokens"],
            "tree_levels": out["levels"],
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda() else 0.0,
            "loss": out["loss"],
        }

    # The pairs of one micro-batch: as many as fit `padded_tokens` padded tokens ([2P, S]); every layout runs the
    # same pairs. Out of memory (someone else may hold part of the GPU): half the pairs, down to a quarter, then
    # SKIP with the reason: the check measures speed.
    pool = make_pairs(256, share, vocab, seed=int(share * 100))
    results, reason, budget = None, "", padded_tokens
    while results is None and budget >= max(padded_tokens // 4, 1):
        pairs, longest = [], 0
        for pair in pool:
            width = max(longest, pair["chosen"]["input_ids"].numel(), pair["rejected"]["input_ids"].numel())
            if pairs and 2 * (len(pairs) + 1) * width > budget:
                break
            pairs.append(pair)
            longest = width
        try:
            results = {name: measure(layout, pairs, 3 if compiled else 1) for name, layout in layouts.items()}
        except torch.OutOfMemoryError as error:
            reason = str(error)[:300]
            policy.zero_grad(set_to_none=True)
            sync(empty=True)
            budget //= 2
    if results is None:
        emit(
            "throughput",
            "SKIP",
            **common,
            gpu_total_gib=total,
            reason=f"out of memory down to {budget * 2} padded tokens per micro-batch ({reason})",
        )
        return
    note = {"reduced_from": padded_tokens, "out_of_memory_at": reason} if budget != padded_tokens else {}
    for layout, values in results.items():
        emit("throughput", "INFO", **common, layout=layout, pairs=len(pairs), padded_budget=budget, **values, **note)


def verdict(results_path: str) -> None:
    """Per model and prompt share: tree against the fastest padded layout that ran (compiled included), in pair
    tokens/s (the same pairs in every layout unless one ran out of memory and was measured on fewer)."""
    measured: dict = {}
    for line in open(results_path, errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("check") == "throughput" and "pair_tok_s" in row:
            measured.setdefault((row["model"], row["prompt_share"]), {})[row["layout"]] = row
    if not measured:
        emit("speedup", "SKIP", reason="no throughput measurements")
    for (model, share), layouts in sorted(measured.items()):
        padded = {k: v["pair_tok_s"] for k, v in layouts.items() if k.startswith("padded")}
        tree = layouts.get("tree")
        if tree is None or not padded:
            emit("speedup", "SKIP", model=model, prompt_share=share, reason="tree or padded was not measured")
            continue
        baseline = max(padded, key=padded.get)
        speedup = tree["pair_tok_s"] / padded[baseline]
        values = {
            "model": model,
            "prompt_share": share,
            "baseline": baseline,
            "speedup": speedup,
            "speedup_vs_padded_as_trained": tree["pair_tok_s"] / padded["padded"] if "padded" in padded else None,
            "speedup_varlen": layouts["varlen"]["pair_tok_s"] / padded[baseline] if "varlen" in layouts else None,
            "same_pairs": tree["pairs"] == layouts[baseline]["pairs"],
            "forwarded_share": tree["forwarded_tokens"] / tree["pair_tokens"],
            "tree_levels": tree["tree_levels"],
            "step_s_tree": tree["step_s_median"],
            "step_s_baseline": layouts[baseline]["step_s_median"],
            "peak_gib_tree": tree["peak_gib"],
            "peak_gib_baseline": layouts[baseline]["peak_gib"],
            "target_speedup": TARGET_SPEEDUP,
            "kill_below": KILL_SPEEDUP,
        }
        if not str(tree.get("device", "")).startswith("cuda"):
            emit("speedup", "INFO", **values, note="a dry run off the GPU: no verdict")
        elif share < TARGET_SHARE:
            emit("speedup", "INFO", **values, note="no target below a 70% prompt share")
        elif speedup >= TARGET_SPEEDUP:
            emit("speedup", "PASS", **values, target_met=True)
        elif speedup < KILL_SPEEDUP:
            emit("speedup", "FAIL", **values, target_met=False, verdict="below the kill threshold")
        else:
            emit("speedup", "INFO", **values, target_met=False, verdict="above the kill threshold, below the target")


# ── smoke runs ─────────────────────────────────────────────────────────────────


def smoke_config(work: str, mode: str, model: str, steps: int) -> None:
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, "smoke_pairs.jsonl")
    if not os.path.exists(data):
        rng = random.Random(0)
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma".split()
        with open(data, "w") as f:
            for _ in range(64 * steps):
                a, b = rng.randint(10, 99), rng.randint(10, 99)
                context = " ".join(rng.choice(words) for _ in range(rng.randint(200, 900)))
                row = {
                    "prompt": [{"role": "user", "content": f"Notes: {context}\n\nWhat is {a} + {b}? Answer briefly."}],
                    "chosen": [{"role": "assistant", "content": f"{a} + {b} = {a + b}."}],
                    "rejected": [
                        {"role": "assistant", "content": f"{a} + {b} = {a + b + rng.randint(1, 9)}, I think."}
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                f.write(json.dumps(row) + "\n")
    path = os.path.join(work, f"smoke_{mode}.yaml")
    budget = "\n  micro_tokens: 12288" if mode != "padded" else ""
    with open(path, "w") as f:
        f.write(
            f"""model:
  name_or_path: {model}
  torch_dtype: float32
  attn_implementation: sdpa
  compile: false
data:
  dataset: {data}
  streaming: false
  max_seq_length: 4096
  num_workers: 0
  last_turn_only: true
dpo:
  enabled: true
  sft_weight: 0.2
  packing: {mode}{budget}
train:
  output_dir: {os.path.join(work, "smoke_" + mode)}
  max_steps: {steps}
  per_device_batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-7
  logging_steps: 1
  save_steps: 0
  save_final: false
  seed: 0
logging:
  use_wandb: false
  use_trackio: false
"""
        )
    print(path)


def smoke_report(log: str, mode: str, steps: int) -> None:
    import re

    reached, losses, times = 0, [], []
    routed = mode == "padded"  # a packed run says so at start-up
    for line in open(log, errors="replace"):
        routed = routed or f"DPO packing: {mode}" in line
        m = re.search(r"\bstep=(\d+) loss=([-+0-9.eE]+|nan|inf)", line)
        if not m:
            continue
        reached = max(reached, int(m.group(1)))
        losses.append(float(m.group(2)))
        dt = re.search(r"\bdt=([0-9.]+)s", line)
        if dt:
            times.append(float(dt.group(1)))
    ok = reached >= steps and losses and all(math.isfinite(v) for v in losses) and routed
    tail = "".join(open(log, errors="replace").readlines()[-5:]) if not ok else ""
    emit(
        "smoke",
        "PASS" if ok else "FAIL",
        mode=mode,
        packing_logged=routed,
        steps_reached=reached,
        first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None,
        step_s_median=statistics.median(times[1:]) if len(times) > 1 else None,
        log_tail=tail[-800:],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["env", "equality", "throughput", "verdict", "smoke-config", "smoke-report"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--share", type=float, default=0.8)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--padded_tokens", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpointing", default="selective", choices=["none", "selective", "full"])
    parser.add_argument("--work", default=".")
    parser.add_argument("--mode", default="tree")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log", default="")
    parser.add_argument("--results", default="")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    if not cuda() and args.checkpointing == "selective":
        # off the GPU scaled_dot_product_attention runs as separate operations, which torch's selective
        # checkpointing does not replay on small CPU builds: the dry run checks the script's logic without it
        args.checkpointing = "none"
    if args.part == "env":
        env_report()
    elif args.part == "equality":
        equality(args.model, args.pairs, args.checkpointing)
    elif args.part == "throughput":
        throughput(args.model, args.share, args.padded_tokens, args.repeats, args.checkpointing, args.compiled)
    elif args.part == "verdict":
        verdict(args.results)
    elif args.part == "smoke-config":
        smoke_config(args.work, args.mode, args.model, args.steps)
    else:
        smoke_report(args.log, args.mode, args.steps)


if __name__ == "__main__":
    sys.exit(main())
