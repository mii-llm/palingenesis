"""Liger Kernel + selective activation checkpointing for memory efficiency.

Two orthogonal optimizations that compose well:
1. Liger Kernel: Fused Triton ops (CE, RMSNorm, SwiGLU, RoPE) — 20% throughput, 60% memory
2. Selective AC: Save expensive ops (attention, matmuls), recompute cheap ones (norms, activations)
"""

import logging

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

logger = logging.getLogger(__name__)


# ─── Liger Kernel ─────────────────────────────────────────────────────────────


def apply_liger_kernel(model_type: str | None) -> bool:
    """Patch the HF implementation of `model_type` (the model config's `model_type`,
    e.g. "qwen3", "qwen3_5", "llama") with Liger's fused kernels: RMSNorm, SwiGLU, RoPE
    and the model-specific extras Liger provides. Call BEFORE the model is created.

    Liger's loss patches stay off: the trainer computes the loss itself from hidden
    states (chunked CE, DEFT, DPO, SeCO), never through the model's forward.
    Returns whether a patch was applied.
    """
    try:
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN, _apply_liger_kernel
    except ImportError:
        logger.warning("liger-kernel not installed: training without its fused kernels.")
        return False

    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
        logger.info(f"Liger Kernel has no patch for model_type={model_type!r}: training without it.")
        return False
    _apply_liger_kernel(model_type, cross_entropy=False, fused_linear_cross_entropy=False)
    logger.info(f"Liger Kernel applied for model_type={model_type}")
    return True


def model_type_of(name_or_path: str, trust_remote_code: bool = False) -> str | None:
    """`model_type` of a model's config (its text config for multimodal checkpoints
    loaded as causal LMs, when Liger knows only that one)."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        logger.warning(f"Could not read the config of {name_or_path!r} ({exc}); Liger Kernel not applied.")
        return None
    return getattr(config, "model_type", None)


# ─── Activation Checkpointing ────────────────────────────────────────────────


# Ops whose outputs are expensive to recompute: saved by the selective policy. These
# are the ops the checkpoint's dispatch mode sees: nn.Linear reaches it as aten.mm
# (never aten.linear), and SDPA as its backend-specific op (flash on A100/H100).
def _save_ops() -> set:
    names = (
        ("aten", "mm"),
        ("aten", "_scaled_dot_product_flash_attention"),
        ("aten", "_scaled_dot_product_efficient_attention"),
        ("aten", "_scaled_dot_product_cudnn_attention"),
        ("aten", "_scaled_dot_product_fused_attention_overrideable"),
        ("aten", "_flash_attention_forward"),
        ("aten", "_efficient_attention_forward"),
        ("_c10d_functional", "reduce_scatter_tensor"),
    )
    ops = set()
    for namespace, op in names:
        packet = getattr(getattr(torch.ops, namespace), op, None)
        if packet is not None and hasattr(packet, "default"):
            ops.add(packet.default)
    return ops


_SAVE_OPS = _save_ops()


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
    """Selective AC policy (torchtitan's op-level SAC): save attention outputs and every
    other matmul, recompute the rest (norms, activations, the other matmuls).

    The matmul parity is counted separately for the forward and for the recompute:
    both must pick the same matmuls, and one shared counter would shift the parity
    whenever a layer has an odd number of matmuls.
    """
    counts = {"forward": 0, "recompute": 0}
    mm = torch.ops.aten.mm.default

    def policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        if func == mm:
            key = "recompute" if ctx.is_recompute else "forward"
            counts[key] += 1
            if counts[key] % 2 == 0:
                return CheckpointPolicy.PREFER_RECOMPUTE
        if func in _SAVE_OPS:
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
