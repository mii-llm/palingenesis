"""Benchmark the flash kernel behind sdpa (model.flash_kernel: builtin / fa3 / fa4) on a model's head shape, and
end to end on the model itself: measured, not estimated.

    python scripts/bench_attention.py --model Qwen/Qwen3-8B [--seqs 4096,16384,65536] [--json out.jsonl]
    python scripts/bench_attention.py --model Qwen/Qwen3-8B --e2e --seq 16384      # whole model, fwd + bwd

Attention mode: forward + backward of (a) dense causal SDPA on one sequence and (b) the variable-length kernel on a
packed row of documents of mixed lengths (what data.packing and SeCO run), per kernel available on this GPU (FA3:
Hopper with flash-attn-3 installed; FA4: Hopper / Blackwell with flash-attn-4). Reports ms per call (median of 3
timed windows), TFLOP/s (causal: half the square) and the speedup over the built-in kernel.

e2e mode: tokens/s and peak memory of the model's forward + backward on one row of --seq tokens (bf16, gradient
checkpointing off), per kernel, through the same activate_flash_kernel the trainer calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from palingenesis.accelerator import FLASH_KERNELS, _flash_module_available, activate_flash_kernel, attention_heads


def _median_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(times)


def _docs(seq: int) -> torch.Tensor:
    """Cumulative lengths of a packed row: documents of 1/2, 1/4, 1/8, ... of the row."""
    lengths, left = [], seq
    while left > 256:
        lengths.append(left // 2)
        left -= left // 2
    lengths.append(left)
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")


def bench_attention(heads: int, kv_heads: int, head_dim: int, seq: int, iters: int) -> dict:
    q = torch.randn(1, heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = _docs(seq)
    longest = int((cu[1:] - cu[:-1]).max())
    flat = [t[0].transpose(0, 1).contiguous().detach().requires_grad_(True) for t in (q, k, v)]

    def dense():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=heads != kv_heads
            )
        out.sum().backward()

    def varlen():
        out = torch.ops.aten._flash_attention_forward(*flat, cu, cu, longest, longest, 0.0, True, False)[0]
        out.sum().backward()

    dense_ms, varlen_ms = _median_ms(dense, iters), _median_ms(varlen, iters)
    flops = 4 * heads * head_dim * seq * seq / 2 * 3.5  # fwd (2 GEMMs) + bwd (2.5x fwd), causal half
    doc_sq = float(((cu[1:] - cu[:-1]).double() ** 2).sum())
    varlen_flops = 4 * heads * head_dim * doc_sq / 2 * 3.5
    return {
        "seq": seq,
        "dense_ms": round(dense_ms, 3),
        "dense_tflops": round(flops / dense_ms / 1e9, 1),
        "varlen_ms": round(varlen_ms, 3),
        "varlen_tflops": round(varlen_flops / varlen_ms / 1e9, 1),
    }


def bench_e2e(model_id: str, seq: int, iters: int) -> dict:
    from transformers import AutoModelForCausalLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda().train()
    model.config.use_cache = False
    ids = torch.randint(0, model.config.get_text_config().vocab_size, (1, seq), device="cuda")

    def step():
        model(input_ids=ids, labels=ids).loss.backward()
        model.zero_grad(set_to_none=True)

    ms = _median_ms(step, iters)
    return {
        "seq": seq,
        "tokens_per_s": round(seq / ms * 1e3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seqs", default="4096,16384,65536")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--seq", type=int, default=16384, help="e2e mode: tokens per row")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    from transformers import AutoConfig

    heads, kv_heads, head_dim = attention_heads(AutoConfig.from_pretrained(a.model))
    major = torch.cuda.get_device_capability()[0]
    print(
        f"{torch.cuda.get_device_name()} (sm{major}x), {a.model}: {heads} heads, {kv_heads} K/V heads, dim {head_dim}"
    )
    kernels = ["builtin"] + [k for k in FLASH_KERNELS[2:] if _flash_module_available(k)]
    rows, base = [], {}
    for kernel in kernels:
        try:
            active = activate_flash_kernel(kernel, "sdpa", heads, kv_heads, head_dim)
        except Exception as e:  # not for this GPU, or failed its startup check
            print(f"{kernel}: skipped ({e})")
            continue
        results = (
            [bench_e2e(a.model, a.seq, a.iters)]
            if a.e2e
            else [bench_attention(heads, kv_heads, head_dim, s, a.iters) for s in map(int, a.seqs.split(","))]
        )
        for r in results:
            key = r["seq"]
            if kernel == "builtin":
                base[key] = r
            ref = base.get(key)
            if ref and kernel != "builtin":
                for metric in ("dense_ms", "varlen_ms"):
                    if metric in r:
                        r[metric.replace("_ms", "_speedup")] = round(ref[metric] / r[metric], 2)
                if "tokens_per_s" in r:
                    r["speedup"] = round(r["tokens_per_s"] / ref["tokens_per_s"], 2)
            row = {"kernel": active, "mode": "e2e" if a.e2e else "attention", **r}
            print(json.dumps(row))
            rows.append(row)
        torch.nn.attention.restore_flash_attention_impl(_raise_warn=False)
    if a.json:
        with open(a.json, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "model": a.model, **row}) + "\n")


if __name__ == "__main__":
    main()
