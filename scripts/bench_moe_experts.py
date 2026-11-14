"""Benchmark MoE expert compute on this machine: one experts module, forward + backward, per implementation.

    python scripts/bench_moe_experts.py                       # Qwen3-30B-A3B's layer shape, a few token counts
    python scripts/bench_moe_experts.py --experts 256 --top-k 8 --hidden 4096 --intermediate 1024 --json out.jsonl
    python scripts/bench_moe_experts.py --model                # also a whole random model, with and without balancing

Implementations: transformers' `eager` (a loop over experts), `grouped_mm` (its default) and, with --batched,
`batched_mm` (gathers a weight matrix per token: memory heavy); this library's `palingenesis` (dispatch + torch's
grouped GEMM) and `palingenesis_padded` (one batched GEMM). Two weight settings, as the trainers use them: bf16
weights, and fp32 master weights under bf16 autocast. Routing is uniform or skewed (a Zipf load over the experts).
Every row reports the time of one forward + backward, the tokens per second, the peak memory and the largest
difference of the output to `eager`. Use it to choose `moe.experts_implementation` for a GPU.
"""

import argparse
import json
import time

import torch


def experts_module(args, dtype):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    config = Qwen3MoeConfig(
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
    )
    config._experts_implementation = "eager"
    module = Qwen3MoeExperts(config)
    torch.nn.init.normal_(module.gate_up_proj, std=0.02)
    torch.nn.init.normal_(module.down_proj, std=0.02)
    return module.to("cuda", dtype)


def routing(tokens: int, experts: int, top_k: int, skew: float, generator) -> torch.Tensor:
    """[tokens, top_k] distinct experts per token; skew 0 = uniform, else expert i has weight (i + 1)^-skew."""
    weights = torch.arange(1, experts + 1, device="cuda", dtype=torch.float32) ** (-skew)
    return torch.multinomial(weights.expand(tokens, -1), top_k, replacement=False, generator=generator)


def run(module, name, hidden, index, gates, autocast, iters):
    from palingenesis.moe_experts import register_experts_implementations

    register_experts_implementations()
    module.config._experts_implementation = name
    params = list(module.parameters())

    def step():
        for p in params:
            p.grad = None
        x = hidden.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = module(x, index, gates)
        out.float().square().mean().backward()
        return out

    for _ in range(3):
        out = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, torch.cuda.max_memory_allocated() / 2**30, out.detach().float()


def bench_layer(args, rows):
    generator = torch.Generator("cuda").manual_seed(0)
    names = ["eager", "grouped_mm", "palingenesis", "palingenesis_padded"] + (["batched_mm"] if args.batched else [])
    for weights in ("bf16", "fp32+autocast"):
        dtype = torch.bfloat16 if weights == "bf16" else torch.float32
        module = experts_module(args, dtype)
        for tokens in args.tokens:
            for skew in args.skew:
                hidden = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
                if weights != "bf16":
                    hidden = hidden.float()
                index = routing(tokens, args.experts, args.top_k, skew, generator)
                gates = torch.softmax(torch.randn(tokens, args.top_k, device="cuda", generator=generator), -1)
                gates = gates.to(hidden.dtype)
                reference = None
                for name in names:
                    try:
                        ms, gib, out = run(module, name, hidden, index, gates, weights != "bf16", args.iters)
                    except (RuntimeError, ImportError, ValueError) as e:  # out of memory, a kernel this GPU lacks
                        print(f"{weights:14s} tokens={tokens:6d} skew={skew:.1f} {name:20s} failed: {str(e)[:80]}")
                        torch.cuda.empty_cache()
                        continue
                    reference = out if reference is None else reference
                    diff = float((out - reference).abs().max() / reference.abs().max().clamp(min=1e-9))
                    row = {
                        "bench": "layer",
                        "weights": weights,
                        "tokens": tokens,
                        "skew": skew,
                        "implementation": name,
                        "ms": round(ms, 3),
                        "tokens_per_s": round(tokens / ms * 1e3),
                        "peak_gib": round(gib, 2),
                        "max_rel_diff_to_eager": diff,
                    }
                    rows.append(row)
                    print(
                        f"{weights:14s} tokens={tokens:6d} skew={skew:.1f} {name:20s} {ms:8.2f} ms "
                        f"{row['tokens_per_s']:>9d} tok/s {gib:6.2f} GiB  diff {diff:.1e}"
                    )


def bench_model(args, rows):
    """A random Qwen3-MoE of the layer shape (few layers): forward + backward tokens/s per implementation, with
    and without the router balancer (its hooks, statistics and injected gradient)."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from palingenesis.logits import final_hidden_states
    from palingenesis.moe import RouterBalancer, find_moe_layers
    from palingenesis.moe_experts import set_experts_implementation

    config = Qwen3MoeConfig(
        vocab_size=32000,
        hidden_size=args.hidden,
        intermediate_size=4 * args.hidden,
        moe_intermediate_size=args.intermediate,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        num_hidden_layers=args.model_layers,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=args.hidden // 16,
        norm_topk_prob=True,
    )
    model = Qwen3MoeForCausalLM(config).to("cuda", torch.bfloat16)
    for layer in find_moe_layers(model):
        torch.nn.init.normal_(layer.router.weight, std=0.02)
    ids = torch.randint(0, 32000, (args.model_rows, args.model_tokens // args.model_rows), device="cuda")
    for name in ("eager", "grouped_mm", "grouped", "padded"):
        set_experts_implementation(model, name)
        for balanced in (False, True):
            balancer = RouterBalancer(find_moe_layers(model), balance_coef=1e-3, z_coef=1e-3) if balanced else None

            def step():
                model.zero_grad(set_to_none=True)
                if balancer is not None:
                    balancer.begin(None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = final_hidden_states(model, ids, None)
                loss = hidden.float().square().mean()
                if balancer is not None:
                    balancer.sync()
                loss.backward()
                if balancer is not None:
                    balancer.step()

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1e3
            if balancer is not None:
                balancer.detach()
            row = {
                "bench": "model",
                "implementation": name,
                "balancer": balanced,
                "tokens": ids.numel(),
                "ms": round(ms, 2),
                "tokens_per_s": round(ids.numel() / ms * 1e3),
            }
            rows.append(row)
            print(f"model  {name:12s} balancer={balanced!s:5s} {ms:8.2f} ms {row['tokens_per_s']:>9d} tok/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--skew", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batched", action="store_true", help="also transformers' batched_mm (memory heavy)")
    parser.add_argument("--model", action="store_true", help="also a whole random model, with and without balancing")
    parser.add_argument("--model-layers", type=int, default=6)
    parser.add_argument("--model-tokens", type=int, default=8192)
    parser.add_argument("--model-rows", type=int, default=4)
    parser.add_argument("--json", default="", help="append the rows to this JSON-lines file")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_moe_experts.py measures GPU kernels: no CUDA device is visible")
    print(
        f"{torch.cuda.get_device_name()} (capability {torch.cuda.get_device_capability()}), torch {torch.__version__}"
    )
    rows: list[dict] = []
    bench_layer(args, rows)
    if args.model:
        bench_model(args, rows)
    if args.json:
        with open(args.json, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), **row}) + "\n")


if __name__ == "__main__":
    main()
