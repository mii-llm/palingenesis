"""Throughput, MFU and peak memory of FSDP, tensor and pipeline parallelism on one multi-GPU node.

NEEDS A MULTI-GPU NODE (NCCL, one process per GPU). Weights are random (built on the meta device, sharded, then
materialised), so any size runs without downloading it: only the model's config is read.

    torchrun --standalone --nproc_per_node 8 scripts/bench_parallel.py \\
        --model Qwen/Qwen3-8B --layouts fsdp,tp2,tp4,pp2,pp4,tp2pp2 --seq 4096 --rows 8

    # 32B / 70B-class (one node of 8 x 80 GB)
    torchrun --standalone --nproc_per_node 8 scripts/bench_parallel.py --model Qwen/Qwen3-32B --layouts fsdp,tp4,tp8,pp4,tp4pp2
    torchrun --standalone --nproc_per_node 8 scripts/bench_parallel.py --model Qwen/Qwen2.5-72B --layouts fsdp,tp8,tp4pp2,pp8

A layout is `fsdp`, `tp<N>`, `pp<N>` or `tp<N>pp<M>`; the remaining ranks are data parallel (FSDP2). Every layout
runs the trainer's own pieces (palingenesis.tensor_parallel, palingenesis.pipeline_parallel, apply_fsdp, the chunked
cross-entropy, AdamW) for `--warmup` + `--repeats` optimizer steps of `--rows` x `--seq` random tokens per data
rank, full activation checkpointing (`--checkpointing`), bf16. Rank 0 prints one JSON line per layout: the median
step time, tokens/s (global), MFU (model FLOPs 6N + 12 L H S per token over --peak_tflops per GPU: 312 for A100
bf16, 989 for H100 SXM) and the largest peak memory over the ranks. A layout that fails (OOM, divisibility) prints
its error and the next one runs.
"""

import argparse
import gc
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_layout(text: str) -> tuple[int, int]:
    """(tp, pp) of a layout name: fsdp, tp<N>, pp<N> or tp<N>pp<M>."""
    import re

    if text == "fsdp":
        return 1, 1
    match = re.fullmatch(r"(?:tp(\d+))?(?:pp(\d+))?", text)
    if not match or not any(match.groups()):
        raise ValueError(f"layout {text!r}: expected fsdp, tp<N>, pp<N> or tp<N>pp<M>")
    return int(match.group(1) or 1), int(match.group(2) or 1)


def materialise(model, device) -> None:
    """Random weights for a model built on the meta device and sharded: every local shard drawn on the GPU,
    norms at one, rotary tables recomputed."""
    from torch.distributed.tensor import DTensor

    model.to_empty(device=device)
    with torch.no_grad():
        for name, p in model.named_parameters():
            local = p.to_local() if isinstance(p, DTensor) else p
            if "norm" in name or name.endswith("A_log") or name.endswith("dt_bias"):
                local.fill_(1.0 if "norm" in name else 0.0)
            else:
                local.normal_(0.0, 0.02)
        for module in model.modules():
            if hasattr(module, "inv_freq") and hasattr(module, "config"):
                fresh = type(module)(config=module.config)
                for key, buf in fresh.named_buffers(recurse=False):
                    getattr(module, key).copy_(buf.to(device))


def model_flops_per_token(config, params: int) -> float:
    """6 x the parameters that multiply each token (the layers and the output head; not the embedding lookup)."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    lookup = 0 if getattr(text, "tie_word_embeddings", False) else text.vocab_size * text.hidden_size
    return 6 * (params - lookup)


def bench(layout: str, args, device) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    from palingenesis.config import ParallelConfig
    from palingenesis.distributed import apply_fsdp, build_mesh, clip_grad_norm_, data_coordinates
    from palingenesis.kernels import apply_activation_checkpointing
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    tp, pp = parse_layout(layout)
    world = dist.get_world_size()
    parallel = ParallelConfig(tp_degree=tp, pp_degree=pp, pp_schedule=args.schedule, pp_microbatches=args.microbatches)
    mesh = build_mesh(world, parallel)
    config = AutoConfig.from_pretrained(args.model)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    if pp > 1:
        text.tie_word_embeddings = False
        config.tie_word_embeddings = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    params = sum(p.numel() for p in model.parameters())
    flops_token = (
        model_flops_per_token(model.config, params) + 12 * text.num_hidden_layers * text.hidden_size * args.seq
    )

    pipe = tp_obj = None
    seq = -(-args.seq // tp) * tp
    if pp > 1:
        from palingenesis.pipeline_parallel import PipelineParallel

        pipe = PipelineParallel(model, mesh, parallel, seq_len=seq, flatten_packed=False)
        model = pipe.model
    if tp > 1:
        from palingenesis.tensor_parallel import apply_tensor_parallel

        tp_obj = apply_tensor_parallel(model, mesh["tp"], pipeline=pipe is not None)
        if pipe is not None:
            pipe.tp = tp_obj
    apply_activation_checkpointing(model, mode=args.checkpointing)
    if pipe is not None:
        pipe.loss_kind, pipe.num_chunks = "chunked", (lambda n: max(1, n * text.vocab_size * 4 // 2**31))
        pipe.build(device, lambda m: apply_fsdp(m, mesh, parallel, bf16=True, reshard_layers=False))
    else:
        model = apply_fsdp(model, mesh, parallel, bf16=True)
    # through the FSDP roots (each stage module under pipelining): FSDP2 re-registers its shards on to_empty
    for root in pipe.stage_modules if pipe is not None else [model]:
        materialise(root, device)
    if pipe is not None:
        pipe.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    data_rank, data_world = data_coordinates(mesh, dist.get_rank(), world)
    g = torch.Generator().manual_seed(data_rank)
    ids = torch.randint(0, text.vocab_size, (args.rows, seq), generator=g).to(device)
    labels = ids.roll(-1, dims=1)
    labels[:, -1] = -100
    tokens = int((labels != -100).sum()) * data_world

    def step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if pipe is not None:
                pipe.step(ids, labels, None, None, tokens)
            else:
                hidden = final_hidden_states(model, ids)
                lab = tp_obj.shard(labels, -100) if tp_obj is not None else labels
                chunks = max(1, hidden.shape[0] * hidden.shape[1] * text.vocab_size * 4 // 2**31)
                loss = chunked_cross_entropy_loss(hidden, lab, output_head(model), chunks, global_valid_tokens=tokens)
                loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.repeats):
        _sync(device)
        dist.barrier()
        start = time.perf_counter()
        step()
        _sync(device)
        times.append(time.perf_counter() - start)
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    step_s = statistics.median(times)
    tok_s = args.rows * seq * data_world / step_s
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "layout": layout,
        "mesh": f"pp={pp} dp={data_world} tp={tp}",
        "model": args.model,
        "params_B": round(params / 1e9, 2),
        "seq": seq,
        "rows_per_data_rank": args.rows,
        "step_s_median": round(step_s, 4),
        "tokens_per_s": round(tok_s, 1),
        "mfu": round(tok_s * flops_token / (world * args.peak_tflops * 1e12), 4),
        "peak_memory_gib": round(float(peak), 2),
        "checkpointing": args.checkpointing,
        "schedule": args.schedule if pp > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layouts", default="fsdp,tp2,tp4,pp2,pp4,tp2pp2")
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=8, help="rows per data-parallel rank per step")
    parser.add_argument("--schedule", default="1f1b")
    parser.add_argument("--microbatches", type=int, default=0, help="pipeline micro-batches (0: one per row)")
    parser.add_argument("--checkpointing", default="full", choices=["full", "selective", "none"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--peak_tflops", type=float, default=312.0, help="bf16 peak per GPU (A100 312, H100 SXM 989)")
    parser.add_argument("--cpu", action="store_true", help="smoke-test the code path on CPU ranks (gloo); no numbers")
    args = parser.parse_args()

    local = int(os.environ.get("LOCAL_RANK", "0"))
    if args.cpu:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise SystemExit("scripts/bench_parallel.py needs CUDA GPUs (launch with torchrun on a multi-GPU node)")
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    for layout in args.layouts.split(","):
        try:
            result = bench(layout.strip(), args, device)
        except Exception as e:  # noqa: BLE001 - report and go on with the next layout
            result = {"layout": layout, "error": f"{type(e).__name__}: {e}"[:500]}
        gc.collect()  # the layout's model and optimizer are gone with bench()'s frame
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dist.barrier()
        if dist.get_rank() == 0:
            print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
