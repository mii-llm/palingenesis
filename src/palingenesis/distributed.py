"""Distributed training: FSDP2, Context Parallel, process group management."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from palingenesis.config import ParallelConfig

# NOTE: FSDP2 symbols (fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy,
# FSDPModule) are imported lazily inside the functions that use them. Those
# names only exist under torch.distributed.fsdp in torch>=2.6, and are only
# needed for multi-GPU (world_size>1) runs -- build_mesh() returns None for a
# single GPU, so apply_fsdp() is never called there. Keeping them out of module
# scope lets this module import on older torch (e.g. 2.5.x) for single-GPU
# training, where FSDP2 is unnecessary.


def setup_distributed() -> tuple[int, int, int]:
    """Initialize process group. Returns (rank, local_rank, world_size).

    Safe to call multiple times -- skips init if already initialized.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


_skip_cleanup = False


def set_skip_cleanup(skip: bool):
    """Set whether cleanup_distributed() should be a no-op.

    Used by autopilot to keep process group alive across multiple train() calls.
    """
    global _skip_cleanup
    _skip_cleanup = skip


def cleanup_distributed():
    if _skip_cleanup:
        return
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def build_mesh(world_size: int, config: ParallelConfig) -> DeviceMesh | None:
    """Build device mesh for FSDP + optional Context Parallel.

    With CP enabled: mesh = (cp_size, dp_size) where dp handles FSDP sharding
    and cp handles sequence sharding.

    Without CP: mesh = (world_size,) for pure FSDP.
    """
    if world_size <= 1:
        return None

    if config.context_parallel and world_size >= 2:
        # Use all GPUs: split between CP and DP
        # Heuristic: CP degree = min(world_size, 8) for reasonable communication
        # User can tune via the mesh. For now, use all GPUs as CP if <= 8,
        # otherwise split into CP groups of 8.
        cp_degree = min(world_size, 8)
        dp_degree = world_size // cp_degree
        if dp_degree * cp_degree != world_size:
            # Fallback: no CP split if world_size not evenly divisible
            cp_degree = world_size
            dp_degree = 1

        mesh = init_device_mesh(
            "cuda",
            (dp_degree, cp_degree),
            mesh_dim_names=("dp", "cp"),
        )
        return mesh
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        return mesh


def apply_fsdp(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    config: ParallelConfig,
    bf16: bool = True,
) -> torch.nn.Module:
    """Apply FSDP2 (fully_shard) bottom-up to transformer layers then root.

    Inspired by torchtitan: shard each layer individually for communication
    overlap, then shard the root model.

    Optimizations (aligned with torchtitan latest):
      - Last transformer layer: reshard_after_forward=False (FSDP would prefetch
        immediately for backward anyway, avoiding a wasted reshard+allgather)
      - Root (embeddings, final norm, head; tied or not): gathered once per step and
        kept until backward, so the chunked losses can apply the head outside the
        model's forward (tests/test_fsdp_trainer_path.py checks the gradients)
    """
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    # Get the DP submesh (or full mesh if no CP)
    dp_mesh = mesh["dp"] if "cp" in (mesh.mesh_dim_names or ()) else mesh

    # bf16=False: parameters stay in the model's dtype (no float16: without a
    # gradient scaler, fp16 training over/underflows)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if bf16 else None,
        reduce_dtype=torch.float32,
    )

    offload_policy = CPUOffloadPolicy() if config.cpu_offload else None

    fsdp_kwargs = dict(
        mesh=dp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
    )

    # Find and shard transformer layers bottom-up
    layers = _find_transformer_layers(model)
    for i, layer in enumerate(layers):
        # Last layer optimization: don't reshard after forward since FSDP
        # would immediately prefetch it for backward (saves one reshard op)
        is_last = i == len(layers) - 1
        layer_reshard = False if is_last else config.reshard_after_forward
        fully_shard(layer, **{**fsdp_kwargs, "reshard_after_forward": layer_reshard})

    # The root owns the rest: embeddings, final norm and output head (tied or not).
    # The losses reach the head outside the model's forward (chunked CE, DPO), after a
    # root forward (logits.final_hidden_states) has gathered these parameters; the root
    # keeps them gathered until its backward.
    fully_shard(model, **{**fsdp_kwargs, "reshard_after_forward": False})

    # The loss is already normalised by the global valid-token count: gradients are
    # summed across ranks, not averaged (divide factor 1, plain SUM reduction).
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
            if hasattr(module, "set_force_sum_reduction_for_comms"):
                module.set_force_sum_reduction_for_comms(True)

    return model


def _find_transformer_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find repeated transformer layers in HF models."""
    for attr_path in ("model.layers", "transformer.h", "transformer.layers"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__iter__") and len(list(obj)) > 0:
                return list(obj)
        except (AttributeError, TypeError):
            continue

    # Fallback: largest ModuleList
    largest: list[torch.nn.Module] = []
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > len(largest):
            largest = list(module)
    return largest
