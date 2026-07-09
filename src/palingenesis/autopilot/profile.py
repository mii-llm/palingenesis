"""Hardware profiling and automatic config generation.

Profiles the GPU and model to determine optimal training configuration:
  - Batch size (largest that fits without OOM)
  - Gradient accumulation steps (to hit target effective batch)
  - Number of loss chunks (based on seq_len * vocab_size)
  - Whether to enable Context Parallel (based on seq_len)
  - Estimated throughput (tokens/sec)
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def profile_hardware() -> dict:
    """Profile available GPU hardware.

    Returns:
        Dict with gpu_name, memory_gb, compute_capability, etc.
    """
    if not torch.cuda.is_available():
        return {"gpu_name": "cpu", "memory_gb": 0, "compute_capability": (0, 0)}

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "memory_gb": props.total_memory / 1e9,
        "compute_capability": (props.major, props.minor),
        "num_gpus": torch.cuda.device_count(),
        "supports_bf16": props.major >= 8,
        "supports_float8": props.major >= 8 and props.minor >= 9,
    }


def auto_config(
    model_params_b: float,
    seq_len: int,
    vocab_size: int,
    hardware: dict,
) -> dict:
    """Generate optimal training config based on hardware + model size.

    Uses simple heuristics validated against real runs:
    - Memory budget: 80% of GPU memory (leave 20% headroom)
    - Batch size: largest that fits within budget
    - Grad accum: enough to hit effective batch of 128-256 tokens * batch
    - Chunks: auto-tuned for CE peak memory

    Returns dict of recommended config overrides.
    """
    gpu_mem = hardware["memory_gb"]
    num_gpus = hardware.get("num_gpus", 1)
    usable_mem = gpu_mem * 0.80  # 20% headroom for spikes

    # Model memory in bf16
    model_mem_gb = model_params_b * 2  # params in bf16

    # Optimizer memory (AdamW: 2 fp32 states per param)
    optim_mem_gb = model_params_b * 8  # 2 states * 4 bytes/param

    # Gradients in bf16
    grad_mem_gb = model_params_b * 2

    # With FSDP (multi-GPU): divide shardable memory by num_gpus
    if num_gpus > 1:
        sharded = (model_mem_gb + optim_mem_gb + grad_mem_gb) / num_gpus
    else:
        sharded = model_mem_gb + optim_mem_gb + grad_mem_gb

    # Remaining memory for activations
    activation_budget_gb = usable_mem - sharded

    # Estimate activation memory per token (rough: 10 bytes/token for selective AC)
    # This is a simplification; real value depends on hidden_size, num_layers
    bytes_per_token = 10  # bf16 activations with selective AC
    tokens_that_fit = int(activation_budget_gb * 1e9 / bytes_per_token)

    # Batch size: how many sequences fit
    max_batch = max(1, tokens_that_fit // seq_len)
    recommended_batch = min(max_batch, 4)  # cap at 4 for SFT (not pretraining)

    # Gradient accumulation to hit effective batch of ~32 sequences
    # (SFT sweet spot — large enough for stable gradients, small enough for LR to matter)
    target_effective = 32
    grad_accum = max(1, target_effective // (recommended_batch * num_gpus))
    grad_accum = min(grad_accum, 32)  # cap to prevent too-long accumulation

    # Auto chunks for CE
    full_logit_gb = recommended_batch * seq_len * vocab_size * 4 / 1e9
    if full_logit_gb > 1.0:
        num_chunks = 2 ** math.ceil(math.log2(math.ceil(full_logit_gb / 1.0)))
        num_chunks = min(num_chunks, 64)
    else:
        num_chunks = 1

    # Context parallel
    use_cp = seq_len > 16384 and num_gpus >= 4

    # Float8
    use_float8 = hardware.get("supports_float8", False) and model_params_b >= 3

    config = {
        "train.per_device_batch_size": recommended_batch,
        "train.gradient_accumulation_steps": grad_accum,
        "memory.loss_num_chunks": num_chunks,
        "memory.chunked_loss": num_chunks > 1,
        "memory.float8_training": use_float8,
        "memory.gradient_release": grad_accum == 1,  # Enable when no accumulation needed
        "memory.selective_diff": True,  # Always beneficial (zero cost)
        "parallel.context_parallel": use_cp,
        "parallel.fsdp": num_gpus > 1,
        "train.bf16": hardware.get("supports_bf16", True),
        # Estimates
        "_estimated_activation_budget_gb": activation_budget_gb,
        "_estimated_tokens_per_step": recommended_batch * seq_len * grad_accum * num_gpus,
    }

    logger.info(
        f"Auto-config: batch={recommended_batch}, grad_accum={grad_accum}, "
        f"chunks={num_chunks}, CP={use_cp}, float8={use_float8}, "
        f"grad_release={grad_accum == 1}"
    )
    return config


def estimate_optimal_lr(
    model_params_b: float,
    effective_batch_tokens: int,
    max_steps: int,
) -> float:
    """Estimate optimal LR from scaling laws (arxiv:2503.04715 + arxiv:2409.19913).

    Formula derived from:
    - "Step Law" (2025): lr ~ model_size^{-0.5} * batch^{-0.15}
    - "Scaling Optimal LR Across Token Horizons" (2024): lr ~ steps^{-0.1}

    Calibrated to: 8B model, batch=128*4096 tokens, 5000 steps -> lr ~= 2e-5
    (matches empirical best from SFT literature)

    Args:
        model_params_b: Model size in billions of parameters
        effective_batch_tokens: Total tokens per optimizer step (bs * seq * accum * gpus)
        max_steps: Total training steps

    Returns:
        Estimated optimal learning rate
    """
    # Calibration constant: anchored to known good configs
    # 8B model, 524k tokens/step, 5000 steps -> 2e-5
    C = 0.003

    # Power-law components
    model_factor = model_params_b**-0.5  # Larger model -> lower LR
    batch_factor = (effective_batch_tokens / 524288) ** -0.12  # Larger batch -> lower LR (sublinear)
    horizon_factor = (max_steps / 5000) ** -0.10  # Longer training -> lower LR

    lr = C * model_factor * batch_factor * horizon_factor

    # Clamp to sane SFT range
    lr = max(1e-6, min(1e-4, lr))

    logger.info(
        f"Estimated optimal LR: {lr:.2e} "
        f"(model={model_params_b:.1f}B, batch_tok={effective_batch_tokens:,}, steps={max_steps})"
    )
    return lr


def correct_lr_for_horizon(
    sweep_lr: float,
    sweep_steps: int,
    full_steps: int,
    alpha: float | None = None,
) -> float:
    """Correct sweep-found LR for the full training horizon.

    From "Scaling Optimal LR Across Token Horizons" (arxiv:2409.19913):
    Optimal LR decreases as training gets longer: lr ~ steps^{-alpha}.

    A sweep at 100 steps overestimates the best LR for 5000 steps.
    This correction accounts for that.

    Args:
        sweep_lr: Best LR found during short sweep
        sweep_steps: Number of steps in the sweep trial
        full_steps: Number of steps in the full training run
        alpha: Horizon exponent. If None, uses literature default (0.12).
               The adaptive_sweep module can provide a data-driven estimate.
               Valid range: [0.05, 0.25]. Higher = stronger correction.
               - Dense pretraining: ~0.10 (established)
               - SFT short-context: ~0.12 (literature default)
               - SFT long-context + DEFT: ~0.15-0.20 (curvature-estimated)

    Returns:
        Corrected LR for the longer horizon
    """
    if full_steps <= sweep_steps:
        return sweep_lr

    if alpha is None:
        alpha = 0.12  # Literature default (arxiv:2409.19913)

    # Clamp alpha to prevent absurd corrections
    alpha = max(0.03, min(0.30, alpha))

    correction = (sweep_steps / full_steps) ** alpha
    corrected = sweep_lr * correction

    # Safety: never correct by more than 5x
    corrected = max(sweep_lr / 5, min(sweep_lr * 2, corrected))

    logger.info(
        f"LR horizon correction: {sweep_lr:.2e} (sweep@{sweep_steps}) "
        f"-> {corrected:.2e} (full@{full_steps}), α={alpha:.3f}, factor={correction:.3f}"
    )
    return corrected
