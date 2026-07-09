"""Context Parallel: shard sequences across GPUs for ultra-long context training.

Uses PyTorch's experimental context_parallel API which replaces SDPA with
Ring Attention — each GPU holds a shard of the sequence and KV are rotated
via all-gather or all-to-all collectives.

This enables training on sequences far longer than a single GPU can hold,
by splitting the O(seq²) attention memory across devices.
"""

import logging

import torch
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def shard_for_context_parallel(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    cp_mesh: DeviceMesh,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shard batch tensors along the sequence dimension for Context Parallel.

    Each rank gets seq_len // cp_world_size tokens. Tensors are split evenly.

    Args:
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        labels: [batch, seq_len]
        cp_mesh: DeviceMesh for the "cp" dimension

    Returns:
        Tuple of sharded (input_ids, attention_mask, labels)
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_world_size = cp_mesh.size()
    seq_len = input_ids.size(1)

    assert seq_len % cp_world_size == 0, (
        f"Sequence length {seq_len} must be divisible by CP world size {cp_world_size}. "
        f"Pad your sequences to a multiple of {cp_world_size}."
    )

    chunk_size = seq_len // cp_world_size
    start = cp_rank * chunk_size
    end = start + chunk_size

    return (
        input_ids[:, start:end].contiguous(),
        attention_mask[:, start:end].contiguous(),
        labels[:, start:end].contiguous(),
    )


def enable_context_parallel(cp_mesh: DeviceMesh, rotate_method: str = "allgather"):
    """Enable the context parallel SDPA dispatcher.

    This makes all calls to F.scaled_dot_product_attention within a
    `context_parallel()` context use Ring Attention automatically.

    Args:
        cp_mesh: DeviceMesh for the CP dimension
        rotate_method: "allgather" or "alltoall"
    """
    try:
        from torch.distributed.tensor.experimental._attention import set_rotate_method

        set_rotate_method(rotate_method)
        logger.info(f"Context Parallel enabled (rotate_method={rotate_method})")
    except ImportError:
        logger.warning(
            "Context Parallel requires PyTorch 2.7+. " "torch.distributed.tensor.experimental._attention not available."
        )


def get_cp_context(cp_mesh: DeviceMesh, buffers: tuple, buffer_seq_dims: tuple):
    """Get the context_parallel context manager for use during forward pass.

    Wraps `torch.distributed.tensor.experimental.context_parallel()`.

    Args:
        cp_mesh: DeviceMesh for CP dimension
        buffers: Tuple of tensors to shard (typically freq_cis / rotary embeddings)
        buffer_seq_dims: Corresponding sequence dimensions for each buffer

    Returns:
        Context manager that enables Ring Attention for SDPA calls.
    """
    from torch.distributed.tensor.experimental import context_parallel

    return context_parallel(cp_mesh, buffers=buffers, buffer_seq_dims=buffer_seq_dims)
