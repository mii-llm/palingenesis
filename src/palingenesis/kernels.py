"""Liger Kernel + selective activation checkpointing for memory efficiency.

Two orthogonal optimizations that compose well:
1. Liger Kernel: Fused Triton ops (CE, RMSNorm, SwiGLU, RoPE) — 20% throughput, 60% memory
2. Selective AC: Save expensive ops (attention, matmuls), recompute cheap ones (norms, activations)
"""

import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

logger = logging.getLogger(__name__)


# ─── Liger Kernel ─────────────────────────────────────────────────────────────


def apply_liger_kernel(model_type: str | None = None) -> bool:
    """Patch HF model implementations with Liger fused kernels. Call BEFORE model load."""
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        logger.warning("liger-kernel not installed, skipping kernel patches.")
        return False

    patch_map: dict[str, list] = {
        "llama": [lk.apply_liger_kernel_to_llama],
        "mistral": [lk.apply_liger_kernel_to_mistral],
        "gemma": [lk.apply_liger_kernel_to_gemma],
        "gemma2": [lk.apply_liger_kernel_to_gemma2],
        "qwen2": [lk.apply_liger_kernel_to_qwen2],
        "qwen": [lk.apply_liger_kernel_to_qwen2],
    }

    if model_type and model_type in patch_map:
        for fn in patch_map[model_type]:
            fn()
        logger.info(f"Liger Kernel applied for {model_type}")
    else:
        # Apply all available patches
        for fns in patch_map.values():
            for fn in fns:
                try:
                    fn()
                except Exception:
                    pass
        logger.info("Liger Kernel patches applied (all architectures)")
    return True


# ─── Activation Checkpointing ────────────────────────────────────────────────


# Ops whose outputs are expensive to recompute — save them
_SAVE_OPS = {
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten.linear.default,
}


def apply_activation_checkpointing(model: nn.Module, mode: str = "selective"):
    """Apply activation checkpointing to transformer layers.

    Modes:
        - "full": Recompute entire layer during backward (maximum memory savings)
        - "selective": Save expensive ops (SDPA, every other matmul), recompute the rest
                       (inspired by torchtitan's SelectiveAC — best memory/compute tradeoff)
        - "none": No checkpointing
    """
    layers = _find_layers(model)
    if not layers:
        logger.warning("No transformer layers found for activation checkpointing")
        return

    if mode == "none":
        return

    for layer_id, layer_module in layers:
        if mode == "full":
            wrapped = ptd_checkpoint_wrapper(
                layer_module,
                preserve_rng_state=True,
            )
        elif mode == "selective":
            wrapped = ptd_checkpoint_wrapper(
                layer_module,
                context_fn=lambda: create_selective_checkpoint_contexts(_selective_policy()),
                preserve_rng_state=True,
            )
        else:
            continue

        # Replace in parent
        _replace_layer(model, layer_id, wrapped)

    logger.info(f"Applied {mode} activation checkpointing to {len(layers)} layers")


def _selective_policy():
    """Selective AC policy: save SDPA + every other matmul, recompute the rest.

    This balances memory and compute — norms, activations, and half the matmuls
    are recomputed (cheap), while attention and the other half of matmuls are
    saved (expensive to recompute).
    """
    meta = {"mm_count": 0}

    def policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        if func in _SAVE_OPS:
            if func == torch.ops.aten.linear.default:
                meta["mm_count"] += 1
                # Save every other matmul
                if meta["mm_count"] % 2 == 0:
                    return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _find_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find transformer layers for checkpointing."""
    for attr_path in ("model.layers", "transformer.h", "transformer.layers"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__iter__"):
                return [(str(i), m) for i, m in enumerate(obj)]
        except (AttributeError, TypeError):
            continue
    return []


def _replace_layer(model: nn.Module, layer_id: str, new_module: nn.Module):
    """Replace a layer in the model's layer list."""
    for attr_path in ("model.layers", "transformer.h", "transformer.layers"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__setitem__"):
                obj[int(layer_id)] = new_module
                return
        except (AttributeError, TypeError, ValueError):
            continue
