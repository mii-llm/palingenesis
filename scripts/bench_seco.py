"""Benchmark SeCO (memory.seco) on this machine: one forward + backward over a set of documents, per mode.

    python scripts/bench_seco.py --model Qwen/Qwen3.5-0.8B --lengths 131072
    python scripts/bench_seco.py --model Qwen/Qwen3.5-0.8B --documents 48 --min-length 512 --max-length 65536 \
        --block 131072 --modes rows,packed,full --trained-share 0.1 --json out.jsonl
    python scripts/bench_seco.py --model Qwen/Qwen3-0.6B --lengths 65536 --chunks 1024,2048,4096,8192 --profile
    python scripts/bench_seco.py --tiny --device cpu          # smoke test (its timings mean nothing)

Modes (the same documents, the same gradient, in each):
  rows    every document as a sequence of its own (SeCO without packing, batch size 1)
  packed  the documents packed into blocks of --block tokens with position_ids (SeCO with data.packing):
          long documents chunk by chunk, short ones batched; reports the padding it ran
  full    full backpropagation of every document with the chunked loss (no SeCO), where it fits

Every row reports the time of one pass over all documents (median of --repeats, after a warm-up pass: Triton
kernels tune themselves on new shapes), the tokens per second, the peak allocated and reserved GPU memory,
the number of chunks and the padding tokens. Weights are fp32 under bf16 autocast, as the trainer runs SeCO.
Tokens are random (the cost does not depend on their content); --trained-share sets the share of positions
with a label (agent traces: about 0.1), which is what the output head's work scales with.

What to measure with it (none of this has been measured on a GPU yet):
  1. Packing. --modes rows,packed on a length mix like the training data's (--documents 64 --min-length 256
     --max-length 131072, --block 131072 and 262144, --trained-share 0.1), chunks 4096 and 8192, with a
     hybrid (Qwen3.5) and a dense model. Expected from the cost model: packed is faster where many
     documents are shorter than a chunk (one batched forward instead of one small forward each), equal
     elsewhere; the padding share is the price (2-30% of the short documents' tokens in simulations).
  2. The last chunk. --lengths at 1, 2, 4 and 8 chunks (e.g. 4096, 8192, 16384, 32768 at --chunks 4096):
     stage 1 no longer runs the last chunk, so the time should scale like (5k - 1) / 5k of the previous
     schedule with full checkpointing, (4k - 1) / 4k without (k chunks): 20-25% at one chunk, 3% at eight.
  3. Small chunks. --lengths 131072 --chunks 1024,2048,4096,8192,16384 --checkpointing full and none, with
     --profile: the share of time outside the attention / linear-attention / GEMM kernels. The prefix is
     attended block by block (one flash call per 8192 prefix tokens plus an fp32 merge of the outputs;
     by a count of operations the merges are 10-25% of the forward attention time, a few percent of a
     step). --kv-block 65536 or 1048576 runs the prefix in fewer calls or one: if that is not faster,
     a single call over a contiguous [prefix; chunk] is not worth building.
  4. SeCO against full backpropagation with checkpointing: --modes rows,full at 16k-262k, both models.
  5. --kv-offload at 262k-1M (dense model): the cost of the transfers, and whether the K/V gradient that
     stays on the GPU is what limits the length.
"""

import argparse
import json
import math
import random
import statistics
import time

import torch

DEVICE = "cuda"


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load(args):
    from transformers import AutoModelForCausalLM

    from palingenesis.kernels import apply_activation_checkpointing
    from palingenesis.seco import use_chunk_attention, verify_chunked_forward

    if args.tiny:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1 << 20,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model = model.to(DEVICE).train()
    use_chunk_attention(model)  # the trainer's attention route for sdpa
    difference = verify_chunked_forward(model)  # the trainer's start-up check
    print(f"chunked forward matches the full forward (relative difference {difference:.1e})")
    apply_activation_checkpointing(model, mode=args.checkpointing)
    return model


def documents(args, vocab: int):
    """[(input_ids [n], labels [n])]: random tokens, labels on runs covering about --trained-share of them."""
    rng = random.Random(args.seed)
    if args.lengths:
        lengths = [int(n) for n in args.lengths.split(",")]
    else:  # log-uniform: many short documents and a long tail, like agent traces
        lo, hi = math.log(args.min_length), math.log(args.max_length)
        lengths = [int(math.exp(rng.uniform(lo, hi))) for _ in range(args.documents)]
    generator = torch.Generator().manual_seed(args.seed)
    out = []
    for n in lengths:
        ids = torch.randint(1, vocab, (n,), generator=generator)
        labels = ids.clone()
        if args.trained_share < 1.0:  # turns of 64 tokens, trained with this probability
            keep = (torch.rand(-(-n // 64), generator=generator) < args.trained_share).repeat_interleave(64)[:n]
            keep[-1] = True  # the reply a trace ends with
            labels[~keep] = -100
        out.append((ids, labels))
    return out


def blocks(docs, size: int):
    """The documents packed first-fit into blocks of at most `size` tokens (longest first, like
    PackedDataset): [(input_ids [1, S], labels, position_ids)]."""
    bins: list[list] = []
    for doc in sorted(docs, key=lambda d: d[0].numel(), reverse=True):
        for b in bins:
            if sum(d[0].numel() for d in b) + doc[0].numel() <= size:
                b.append(doc)
                break
        else:
            bins.append([doc])
    return [
        (
            torch.cat([d[0] for d in b])[None],
            torch.cat([d[1] for d in b])[None],
            torch.cat([torch.arange(d[0].numel()) for d in b])[None],
        )
        for b in bins
    ]


def run_pass(model, mode: str, docs, packed, args, vocab: int) -> dict:
    """One forward + backward over every document; gradients accumulate in .grad."""
    from palingenesis.loss import chunked_cross_entropy_loss, shift_labels
    from palingenesis.seco import chunkwise_forward_backward
    from palingenesis.train import _dynamic_num_chunks

    model.zero_grad(set_to_none=True)
    chunks = padding = 0
    options = dict(
        chunk_size=args.chunk,
        kv_offload=args.kv_offload,
        num_loss_chunks=lambda n: _dynamic_num_chunks(n, vocab),
    )
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        if mode == "rows":
            for ids, labels in docs:
                result = chunkwise_forward_backward(model, ids[None].to(DEVICE), labels[None].to(DEVICE), **options)
                chunks += result.num_chunks
        elif mode == "packed":
            for ids, labels, positions in packed:
                result = chunkwise_forward_backward(
                    model, ids.to(DEVICE), labels.to(DEVICE), position_ids=positions.to(DEVICE), **options
                )
                chunks, padding = chunks + result.num_chunks, padding + result.padding
        else:
            head = model.get_output_embeddings()
            for ids, labels in docs:
                hidden = model.base_model(input_ids=ids[None].to(DEVICE)).last_hidden_state
                shifted = shift_labels(labels[None].to(DEVICE))
                loss = chunked_cross_entropy_loss(
                    hidden, shifted, head, num_chunks=_dynamic_num_chunks(ids.numel(), vocab), global_valid_tokens=1.0
                )
                loss.backward()
                del hidden, loss
    _sync()
    return {"chunks": chunks, "padding": padding}


def measure(model, mode: str, docs, packed, args, vocab: int) -> dict:
    tokens = sum(d[0].numel() for d in docs)
    row = {"mode": mode, "chunk": args.chunk, "documents": len(docs), "tokens": tokens}
    try:
        run_pass(model, mode, docs, packed, args, vocab)  # warm-up
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            row.update(run_pass(model, mode, docs, packed, args, vocab))
            times.append(time.perf_counter() - start)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {**row, "error": "out of memory"}
    seconds = statistics.median(times)
    row.update(seconds=round(seconds, 3), tokens_per_second=round(tokens / seconds))
    if DEVICE == "cuda":
        row.update(
            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        )
    return row


def profile_pass(model, mode: str, docs, packed, args, vocab: int) -> None:
    """Where one pass spends its time: the operators by self time (kernels against everything else)."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    with profile(activities=activities) as prof:
        run_pass(model, mode, docs, packed, args, vocab)
    key = "self_cuda_time_total" if DEVICE == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=25))


def main() -> None:
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="a causal LM on the Hub or on disk")
    parser.add_argument("--tiny", action="store_true", help="a tiny random Llama instead of --model (smoke test)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default="rows,packed")
    parser.add_argument("--lengths", help="document lengths, comma separated (instead of a random mix)")
    parser.add_argument("--documents", type=int, default=32, help="documents of the random mix")
    parser.add_argument("--min-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--block", type=int, default=131072, help="tokens per packed block (data.max_seq_length)")
    parser.add_argument("--chunks", default="4096", help="memory.seco_chunk_size values, comma separated")
    parser.add_argument("--checkpointing", default="full", choices=["none", "selective", "full"])
    parser.add_argument("--kv-offload", action="store_true", help="memory.seco_kv_offload")
    parser.add_argument("--kv-block", type=int, help="prefix tokens per attention block (default: seco.KV_BLOCK)")
    parser.add_argument("--trained-share", type=float, default=1.0, help="share of positions with a label")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="also print a torch profiler table per row")
    parser.add_argument("--json", help="append one JSON line per row to this file")
    args = parser.parse_args()
    if not args.model and not args.tiny:
        parser.error("--model or --tiny")
    DEVICE = args.device
    if args.tiny and not args.lengths:
        args.documents, args.min_length, args.max_length, args.block, args.chunks = 12, 8, 400, 512, "64"

    if args.kv_block:
        import palingenesis.seco

        palingenesis.seco.KV_BLOCK = args.kv_block
    model = load(args)
    vocab = model.get_output_embeddings().weight.shape[0]
    docs = documents(args, vocab)
    packed = blocks(docs, max(args.block, max(d[0].numel() for d in docs)))
    lengths = sorted(d[0].numel() for d in docs)
    print(
        f"{len(docs)} documents, {sum(lengths)} tokens (shortest {lengths[0]}, median {lengths[len(lengths) // 2]}, "
        f"longest {lengths[-1]}), {len(packed)} packed blocks"
    )
    for chunk in (int(c) for c in args.chunks.split(",")):
        args.chunk = chunk
        for mode in args.modes.split(","):
            row = measure(model, mode, docs, packed, args, vocab)
            print(json.dumps(row))
            if args.json:
                settings = {
                    "model": args.model or "tiny",
                    "checkpointing": args.checkpointing,
                    "kv_offload": args.kv_offload,
                    "kv_block": args.kv_block,
                    "trained_share": args.trained_share,
                    "block": args.block,
                }
                with open(args.json, "a") as f:
                    f.write(json.dumps({**row, **settings}) + "\n")
            if args.profile and "error" not in row:
                profile_pass(model, mode, docs, packed, args, vocab)


if __name__ == "__main__":
    main()
