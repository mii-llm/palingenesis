Tier A: High confidence, implement after first successful training run
SeCO on Qwen3.5 — install causal-conv1d + fla, verify chunked prefill. Unblocks 32K+ agentic training on hybrid models.

torch.compile attention-only — skip DeltaNet layers, compile only the 25% attention blocks. Requires testing which layers break fullgraph=True.

Liger Kernel Qwen3.5 compat — test (1+weight) RMSNorm variant. Either it works (free 15% speedup) or we need a custom kernel.

Tier B: Investigate after benchmarking reveals specific gaps
AlphaDecay (2506.14562) — per-module weight decay from heavy-tail analysis. Implement if some layers overfit faster than others in practice.

Progressive Residual Warmup (2603.05369) — early layers first, later layers warm up gradually. Implement if early training is unstable.

Dataset Decomposition seq-len curriculum (2405.13226) — the SeqLenCurriculum class exists but isn't wired into the training loop. Wire it if throughput matters more than complexity.

FG2-GDN (2604.19021) — enhanced Gated Delta Networks for Qwen3.5. Only relevant if we unfreeze DeltaNet layers.

Tier C: Future hardware / scale
Quartet FP4 (2505.14669) — native FP4 training on Blackwell B200. Wait for hardware availability.

FCP scalable context parallel (2605.08524) — variable sequence length handling in CP. Implement when training >128K sequences on 8+ GPUs.

DeltaNet parallelization (2406.06484) — sequence-parallel DeltaNet. Only needed if training unfrozen Qwen3.5 on very long sequences.

Tier D: Research directions (not implementation-ready yet)
ScheduleFree+ with Muon — needs experimental validation that the two compose. Currently untested combination.

RASFT (2606.07006) — rollout-adaptive SFT requires on-policy generation. Fundamentally different paradigm. Park for v1.0.

AdEMAMix (2409.03137) — dual-EMA for very long training runs (>100K steps). Our SFT runs are too short to benefit.