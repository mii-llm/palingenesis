"""Benchmark train.optimizer choices on one GPU: optimizer step time, memory and how much of a fine-tuning update
survives, measured, not estimated.

    # the optimizer step alone, on a model's real parameter shapes (random gradients)
    python scripts/bench_optimizers.py step --model Qwen/Qwen3.5-0.8B [--iters 20] [--json out.jsonl]

    # end to end: forward, backward and step on fixed random batches (memorised at a fine-tuning LR)
    python scripts/bench_optimizers.py train --model Qwen/Qwen3.5-0.8B --seq 4096 --steps 60 [--lr 1e-5]

Regimes (each on a freshly loaded model, in this process):

    fp32_master   model.torch_dtype float32 + bf16 autocast, torch AdamW (fused): the exact baseline, 16 B/param
    bf16_adamw    bf16 weights, torch AdamW (fused, moments bf16, round-to-nearest): what the default config runs
    adamw_bf16    bf16 weights, palingenesis.bf16_adamw.BF16AdamW (stochastic rounding, compiled update)
    adamw8bit     bf16 weights, bitsandbytes AdamW8bit (if installed)

Every timing is the median of 3 timed windows after warm-up (compilation included in warm-up). `train` also reports
the loss after --steps (lower = more of each update kept; fp32_master is the reference) and the share of weights that
changed at all since the start. Peak memory is torch's allocated peak.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

REGIMES = ("fp32_master", "bf16_adamw", "adamw_bf16", "adamw8bit")


def _optimizer(regime: str, params, lr: float):
    if regime in ("fp32_master", "bf16_adamw"):
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    if regime == "adamw_bf16":
        from palingenesis.bf16_adamw import BF16AdamW

        return BF16AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def _available(regime: str) -> bool:
    if regime != "adamw8bit":
        return True
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _load(model_id: str, dtype: torch.dtype, meta_only: bool = False):
    from transformers import AutoConfig, AutoModelForCausalLM

    if meta_only:
        cfg = AutoConfig.from_pretrained(model_id)
        with torch.device("cuda"):
            return AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).cuda()


def _windows(fn, iters: int, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


def bench_step(model_id: str, regime: str, iters: int) -> dict:
    dtype = torch.float32 if regime == "fp32_master" else torch.bfloat16
    torch.cuda.empty_cache()
    model = _load(model_id, dtype, meta_only=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    before = torch.cuda.memory_allocated()
    opt = _optimizer(regime, params, 1e-5)
    opt.step()  # materialises the state (and compiles)
    state = torch.cuda.memory_allocated() - before
    for _ in range(2):
        opt.step()
    ms = _windows(opt.step, iters) * 1e3
    n = sum(p.numel() for p in params)

    return {
        "mode": "step",
        "regime": regime,
        "params_m": round(n / 1e6, 1),
        "step_ms": round(ms, 2),
        "state_bytes_per_param": round(state / n, 2),
    }


def bench_train(model_id: str, regime: str, seq: int, batches: int, steps: int, lr: float) -> dict:
    dtype = torch.float32 if regime == "fp32_master" else torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = _load(model_id, dtype)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = _optimizer(regime, [p for p in model.parameters() if p.requires_grad], lr)
    g = torch.Generator(device="cpu").manual_seed(0)
    vocab = model.config.get_text_config().vocab_size
    data = [torch.randint(0, vocab, (1, seq), generator=g).cuda() for _ in range(batches)]
    state = {"i": 0, "loss": None}

    def step():
        ids = data[state["i"] % len(data)]
        state["i"] += 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["loss"] = loss.detach()

    for _ in range(3):
        step()
    per = max(1, (steps - 3) // 3)
    sec = _windows(step, per)
    final = float(state["loss"])
    changed = sum(int((p.detach() != start[n]).sum()) for n, p in model.named_parameters())
    total = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**30

    return {
        "mode": "train",
        "regime": regime,
        "seq": seq,
        "tokens_per_s": round(seq / sec),
        "step_s": round(sec, 3),
        "peak_gib": round(peak, 2),
        "loss": round(final, 4),
        "changed_frac": round(changed / total, 4),
        "steps": state["i"],
        "lr": lr,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("step", "train"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--iters", type=int, default=20, help="step mode: optimizer steps per timed window")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batches", type=int, default=4, help="train mode: distinct batches, cycled")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    print(f"{torch.cuda.get_device_name()} (sm{''.join(map(str, torch.cuda.get_device_capability()))}), {a.model}")
    rows = []
    for regime in a.regimes.split(","):
        if regime not in REGIMES:
            raise SystemExit(f"unknown regime {regime!r}; one of {', '.join(REGIMES)}")
        if not _available(regime):
            print(f"{regime}: skipped (bitsandbytes not installed)")
            continue
        if a.mode == "step":
            row = bench_step(a.model, regime, a.iters)
        else:
            row = bench_train(a.model, regime, a.seq, a.batches, a.steps, a.lr)
        print(json.dumps(row))
        rows.append(row)
    if a.json:
        with open(a.json, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "model": a.model, **row}) + "\n")


if __name__ == "__main__":
    main()
