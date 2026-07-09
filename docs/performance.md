# Performance Engineering

Low-level optimizations that make the training loop fast. These are automatic -- no config needed. They compose with all other features (FSDP, compile, plugins).

## CUDA Prefetcher

**What**: Overlaps Host-to-Device (H2D) memory transfer with GPU compute using a separate CUDA stream.

**Why**: Without prefetching, the training step is:
```
[CPU: get batch] -> [PCIe: copy to GPU] -> [GPU: forward] -> [GPU: backward] -> ...
                    ~~~~ IDLE GPU ~~~~
```

With prefetching:
```
Step N:   [GPU: forward+backward on batch N]
Stream 2: [PCIe: copy batch N+1 to GPU]      <- overlapped!
Step N+1: [GPU: forward+backward on batch N+1]  <- batch already there
```

**Impact**: 1-3ms saved per step. On short steps (small models, short sequences) this is 2-5% throughput. On long steps it's negligible but free.

**How it works**:
```python
class CUDAPrefetcher:
    def __init__(self, dataloader, device):
        self._stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        # Transfer next batch on copy stream while current computes on default stream
        with torch.cuda.stream(self._stream):
            next_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # Before using next_batch, synchronize
        torch.cuda.current_stream(device).wait_stream(self._stream)
```

**Enabled by default**: Yes, always active.

---

## Garbage Collection Control

**What**: Disables Python's automatic garbage collector and runs it manually every 100 steps.

**Why**: Python's GC uses reference counting + generational collection. The generational collector runs at unpredictable intervals and can pause for 50-100ms. In distributed training, if one rank pauses for GC while others don't, ALL ranks wait at the next collective (allreduce/allgather), multiplying the stall by world_size.

**Impact**: Eliminates random 50-100ms stalls. On 8-GPU training, a single rank GC pause costs 8 * 50ms = 400ms of aggregate GPU-time.

**How it works**:
```python
gc.disable()  # No more random pauses

# Every 100 steps, all ranks GC simultaneously
if step % 100 == 0:
    gc.collect()  # Coordinated pause is better than random pauses
```

**Enabled by default**: Yes, always active. GC runs every 100 optimizer steps.

---

## Auto-Tuned Chunked Loss

**What**: Automatically calculates the optimal number of chunks for the chunked cross-entropy loss based on sequence length and vocab size.

**Why**: Fixed `num_chunks=8` might be too few for very long sequences (OOM) or too many for short sequences (unnecessary loop overhead). The optimal number depends on `batch_size * seq_len * vocab_size * 4 bytes`.

**Formula**:
```
full_logit_gb = B * S * V * 4 / 1e9
num_chunks = ceil(full_logit_gb / target_chunk_memory_gb)
# Round to power of 2 for even splitting
# Clamp to [1, 64]
```

**Examples**:
| Seq Length | Vocab Size | Batch | Full Logits | Auto Chunks |
|-----------|-----------|-------|-------------|-------------|
| 4096 | 32000 | 1 | 0.5 GB | 1 (no chunking needed) |
| 8192 | 128256 | 1 | 4.0 GB | 4 |
| 32768 | 128256 | 1 | 16.1 GB | 16 |
| 65536 | 128256 | 1 | 32.2 GB | 32 |

**Behavior**: Takes the MAX of the config value and the auto-tuned value. If you set `loss_num_chunks: 8` but auto-tune says 16, it uses 16. If you set 32, it uses 32 (your override wins).

**Enabled by default**: Yes, when `memory.chunked_loss: true`.

---

## Step Timer (Profiling Utility)

**What**: Instruments each phase of the training step to identify bottlenecks.

**Why**: When training is slow, you need to know WHERE time is spent:
- Data loading? -> Increase `num_workers`
- H2D transfer? -> Prefetcher is helping, sequences may be very long
- Forward? -> Expected to be the largest component
- Backward? -> ~2x forward, expected
- Optimizer? -> Compiled optimizer should be fast
- Grad clipping? -> Should be negligible

**Usage**: Available in `perf.py` as `StepTimer`. Can be enabled for profiling runs.

**Output**:
```
[step 50] Timing (avg 50 steps): data=1.2ms h2d=0.3ms fwd=45.3ms bwd=52.1ms clip=0.3ms optim=2.1ms total=101.3ms
```

**Overhead**: ~0.1ms per step when enabled (requires `torch.cuda.synchronize()` for accurate GPU timing). Disabled by default.

---

## MaxTokensBatcher (Variable-Length Optimization)

**What**: Sorts samples by length before padding to minimize wasted compute on padding tokens.

**Why**: Standard fixed-batch collation with dynamic padding:
```
Batch: [seq_8000, seq_200, seq_500] -> pad all to 8000
GPU computes: 3 * 8000 = 24000 tokens
Useful tokens: 8000 + 200 + 500 = 8700  (only 36% utilization!)
```

With length-sorted batching:
```
Batch 1: [seq_8000]  -> no padding needed (100% utilization)
Batch 2: [seq_500, seq_200] -> pad to 500 (70% utilization)
```

**Impact**: 20-40% throughput gain on highly variable-length agentic data. Less impact when sequences are similar length or packing is enabled.

**Available in**: `perf.py` as `MaxTokensBatcher`. Not yet default (requires DataLoader integration with `batch_size=None` and custom batch sampler).

---

## Performance Composition

All optimizations compose and stack:

```
Step without any optimizations:      150ms
+ GC control (no random pauses):     150ms (same avg, but no spikes)
+ Prefetcher (overlap H2D):          147ms (-3ms)
+ Auto-chunk tuning (right # chunks): 147ms (prevents OOM, no speed diff if already tuned)
+ Compiled optimizer:                 145ms (-2ms on optim step)
+ Per-layer compile:                  125ms (-20ms from kernel fusion)
+ Liger Kernel:                       110ms (-15ms from fused ops)
+ Selective AC:                       Same speed, but fits 2x longer sequences
+ FSDP2:                              Same speed per step, but trains on N GPUs

Total training loop overhead (Python, logging, GC): < 2ms/step
GPU utilization target: > 95% MFU
```

---

## Diagnosing Slow Training

Use the step timer to find your bottleneck:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `data` > 10ms | DataLoader starved | Increase `num_workers`, enable `pin_memory` |
| `h2d` > 5ms | Very long sequences | Already overlapped by prefetcher |
| `fwd` >> `bwd` | Unusual, check model | Ensure `use_cache=False` |
| `bwd` > 2.5x `fwd` | Activation checkpointing cost | Switch to `selective` from `full` |
| `optim` > 5ms | Uncompiled optimizer | Ensure `fused=True` or compiled step |
| Random spikes of 50ms+ | GC pauses | GC control (already enabled) |
| All steps slow | Low GPU utilization | Check with `nvidia-smi`, increase batch |
