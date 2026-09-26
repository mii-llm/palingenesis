"""
palingenesis — Boutique multi-GPU/multi-node SFT trainer for agentic LLM loops.

Ultra-optimized for long sequences. Trains like a bliss.
"""

import os

# Keep Triton autotuning results on disk (read when a kernel's autotuner is created, so before
# flash-linear-attention or Liger load): otherwise every run re-benchmarks every kernel config
# on its first steps, 370 autotunings and a 180-200 s first RL step on Qwen3.5 (37 s cached).
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

__version__ = "0.3.0"
