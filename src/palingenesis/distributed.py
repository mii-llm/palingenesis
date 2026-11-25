"""Distributed training: FSDP2, Context Parallel, process group management."""

import gc
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from palingenesis.config import ParallelConfig

logger = logging.getLogger(__name__)

# NOTE: FSDP2 symbols (fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy,
# FSDPModule) are imported lazily inside the functions that use them. Those
# names only exist under torch.distributed.fsdp in torch>=2.6, and are only
# needed for multi-GPU (world_size>1) runs and single-GPU CPU offload -- build_mesh() returns None for a
# single GPU otherwise, so apply_fsdp() is never called there. Keeping them out of module
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
        _warn_unlaunched_tasks()

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Without a usable GPU the caller raises its own, explanatory error (set_device would fail first with a
    # less helpful one).
    if torch.cuda.is_available():
        torch.cuda.set_device(local_device_index(local_rank, torch.cuda.device_count()))
    return rank, local_rank, world_size


def init_single_process_group() -> None:
    """A process group of one rank, for FSDP2 on a single GPU (parallel.cpu_offload: FSDP2's CPU offload needs a
    mesh even with nothing to shard). Gloo for CPU tensors and NCCL for CUDA ones: with offload the gradients and the
    optimizer step live on the CPU."""
    if dist.is_initialized():
        return
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def local_device_index(local_rank: int, visible: int) -> int:
    """The index of this process's GPU among the `visible` ones: its local rank.

    The launcher must leave every GPU of the node visible to each of its processes (torchrun's default), or
    give each process exactly one and LOCAL_RANK=0 (torchrun --virtual-local-rank). A scheduler that binds
    one GPU per process while LOCAL_RANK still counts them (srun --gpus-per-task=1 around the workers) makes
    cuda:{local_rank} point past the visible devices."""
    if local_rank < visible:
        return local_rank
    raise RuntimeError(
        f"LOCAL_RANK={local_rank} but this process sees {visible} GPU(s) "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). Start one process per visible "
        "GPU (torchrun --nproc_per_node), keep every GPU of the node visible to each process (no per-task GPU "
        "binding around the workers), or launch with torchrun --virtual-local-rank."
    )


def _warn_unlaunched_tasks() -> None:
    """Without torchrun's RANK / WORLD_SIZE this process trains alone. Several tasks of one Slurm step started
    that way (srun python -m palingenesis.train) would each train a full copy into the same output directory."""
    tasks = os.environ.get("SLURM_STEP_NUM_TASKS", "")
    if tasks.isdigit() and int(tasks) > 1:
        logger.warning(
            "This process is 1 of %s tasks of a Slurm step, started without torchrun: every task trains on its "
            "own (no process group). Launch one torchrun per node instead: srun --ntasks-per-node=1 torchrun "
            "--nnodes=$SLURM_NNODES --nproc_per_node=<GPUs per node> ... (see scripts/train_slurm.sh).",
            tasks,
        )


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


def barrier() -> None:
    """dist.barrier() over the default group; nothing without a process group or with a group of one.

    One process has nobody to wait for, and NCCL allocates on the GPU for every barrier: right after
    a checkpoint save, with a rollout engine awake on the same GPU, that allocation can fail and kill
    a run whose files are already written. Under NCCL the barrier runs on the current device, named
    explicitly (otherwise NCCL guesses it from the global rank and warns)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def main_value(value):
    """Rank 0's `value` on every rank (one collective); `value` itself without a process group or with one rank.

    For what every rank must agree on but each would otherwise work out by itself, from a filesystem it may see
    differently or later than rank 0 (which checkpoint to resume, whether a cache has to be built), or from
    its own floating-point result (a new best evaluation loss): ranks that decide differently run different
    collectives next, and the run hangs. `value` must be picklable and small."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    box = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(box, src=0)  # NCCL: through the current device (set_device in setup_distributed)
    return box[0]


def rank_values(value) -> list:
    """Every rank's `value`, in rank order (one collective); [value] without a process group or with one rank.

    For a local check the ranks must act on together (can this rank read the file rank 0 wrote?): a rank
    raising alone leaves the others waiting in their next collective. `value` must be picklable and small."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    values: list = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def release_cuda_memory() -> None:
    """Return the caching allocator's free blocks to the device (after a save's transient copies and
    gathers, before a collective or another process needs the memory). Local to the rank: no
    collective, so every rank still reaches the next barrier."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_mesh(world_size: int, config: ParallelConfig) -> DeviceMesh | None:
    """The device mesh for FSDP2 and, optionally, context, tensor and pipeline parallelism.

    Dimensions, outermost first (palingenesis.parallel_dims): "pp" (pipeline stages), "dp" (different data),
    "cp" (ranks splitting the same sequences, palingenesis.context_parallel) and "tp" (ranks splitting every
    layer, palingenesis.tensor_parallel; innermost, so a group is one node's consecutive GPUs). "dp" is always
    there (FSDP2 shards over it, also when it has one rank); the others only when enabled.

    Without context, tensor or pipeline parallelism: a 1-D mesh ("dp",) over every rank. One process: no mesh,
    except with parallel.cpu_offload (FSDP2's CPU offload on a single GPU), which gets a mesh of one rank.

    FSDP2 shards the layers over "dp" x "cp" flattened (`fsdp_mesh`): every rank of it computes gradients from
    different tokens, and all of them are summed. `parallel.cp_degree` sets cp (0: min(world_size, 8) when that
    divides world_size, else world_size; groups of up to 8 keep the K/V all-gathers on one node's NVLink).
    """
    from palingenesis.config import ConfigError
    from palingenesis.parallel_dims import ParallelDims

    tp, pp = config.tp_degree, config.pp_degree
    if world_size <= 1 and not config.cpu_offload:
        if tp > 1 or pp > 1:
            raise ConfigError(
                f"parallel.tp_degree={tp} / pp_degree={pp} need that many processes (torchrun --nproc_per_node); "
                "this run has one."
            )
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if world_size <= 1:  # one GPU with CPU offload: FSDP2 over a mesh of one rank
        return init_device_mesh(device, (1,), mesh_dim_names=("dp",))

    cp_degree = 1
    if config.context_parallel:
        rest = world_size // max(tp * pp, 1) if world_size % max(tp * pp, 1) == 0 else world_size
        cp_degree = config.cp_degree or min(rest, 8)
        if not config.cp_degree and rest % cp_degree:
            cp_degree = rest
        if cp_degree < 1 or world_size % cp_degree:
            raise ConfigError(
                f"parallel.cp_degree={config.cp_degree} must divide the number of processes ({world_size})."
            )
    dims = ParallelDims(world_size, tp=tp, pp=pp, cp=cp_degree)
    if errors := dims.errors():
        raise ConfigError(errors[0])
    if tp == 1 and pp == 1:
        if config.context_parallel:
            return init_device_mesh(device, (dims.dp, cp_degree), mesh_dim_names=("dp", "cp"))
        return init_device_mesh(device, (world_size,), mesh_dim_names=("dp",))
    shape, names = [], []
    for name, size in (("pp", pp), ("dp", dims.dp), ("cp", cp_degree), ("tp", tp)):
        if size > 1 or name == "dp" or (name == "cp" and config.context_parallel):
            shape.append(size)
            names.append(name)
    logger.info(f"Device mesh: {dims.describe()} ({world_size} processes)")
    return init_device_mesh(device, tuple(shape), mesh_dim_names=tuple(names))


def _flat(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """`mesh` restricted to the dimensions `names`, flattened to one dimension (the same one every call)."""
    if len(names) == 1:
        return mesh[names[0]] if len(mesh.mesh_dim_names or ()) > 1 else mesh
    flat = "_".join(names)
    try:
        return mesh[flat]  # flattened before (the student and a sharded teacher share it)
    except (KeyError, RuntimeError, ValueError):
        return mesh[names]._flatten(flat)


def fsdp_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """The mesh FSDP2 shards the layers over: "dp" and "cp", flattened to one dimension.

    Under context parallelism the ranks of a "cp" group hold different tokens of the same
    sequences, so their gradients must be summed like those of data-parallel ranks; sharding over
    "dp" alone would leave each cp rank with the gradient of its own slice only. Tensor-parallel
    ranks hold different slices of the same parameters (the "tp" dimension of their DTensors) and
    pipeline ranks different layers: neither is part of it."""
    names = mesh.mesh_dim_names or ()
    return _flat(mesh, tuple(n for n in ("dp", "cp") if n in names))


def sequence_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """The mesh of the parameters every rank of a pipeline stage uses whole on its own tokens: "dp", "cp" and
    "tp" flattened. Under tensor parallelism the embeddings, final norm and output head are not split
    (palingenesis.tensor_parallel): each tensor-parallel rank applies them to its slice of the sequence, so
    their gradients are summed over the tensor-parallel group as well, and FSDP2 shards them over all of it."""
    names = mesh.mesh_dim_names or ()
    return _flat(mesh, tuple(n for n in ("dp", "cp", "tp") if n in names))


def data_coordinates(mesh: DeviceMesh | None, rank: int, world_size: int) -> tuple[int, int]:
    """(index, count) of this rank's data shard: the ranks of a context-, tensor- or pipeline-parallel group
    read the same rows; only the "dp" dimension holds different data."""
    if mesh is None or len(mesh.mesh_dim_names or ()) <= 1:
        return rank, world_size
    return mesh["dp"].get_local_rank(), mesh["dp"].size()


def apply_fsdp(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    config: ParallelConfig,
    bf16: bool = True,
    ignored_params: set | None = None,
    reshard_layers: bool | None = None,
) -> torch.nn.Module:
    """Apply FSDP2 (fully_shard) bottom-up to transformer layers then root.

    `reshard_layers`: overrides parallel.reshard_after_forward for the layers (a pipeline stage keeps its
    layers gathered between micro-batches, as torchtitan: one all-gather per step instead of one per
    micro-batch and direction). Layers without parameters (another stage's) are skipped.

    `ignored_params`: parameters FSDP leaves alone (expert parallelism keeps each rank's own
    experts: palingenesis.moe_experts).

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

    # Every rank (data- and context-parallel): their gradients come from different tokens
    dp_mesh = fsdp_mesh(mesh)
    # Tensor parallelism: the root's parameters (embeddings, final norm, head) are whole on every tensor-parallel
    # rank, which applies them to its own slice of the sequence (palingenesis.tensor_parallel): FSDP shards them
    # over the tensor-parallel group too, and sums their gradients over it.
    root_mesh = sequence_mesh(mesh) if "tp" in (mesh.mesh_dim_names or ()) else dp_mesh

    # bf16=False: parameters stay in the model's dtype (no float16: without a
    # gradient scaler, fp16 training over/underflows)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if bf16 else None,
        reduce_dtype=torch.float32,
    )

    # Pinned (page-locked) host memory only speeds up the copies to a GPU, and pinning needs a CUDA device: without
    # one (a CPU run on a CUDA build of PyTorch) `pin_memory()` raises "No CUDA GPUs are available".
    offload_policy = CPUOffloadPolicy(pin_memory=torch.cuda.is_available()) if config.cpu_offload else None

    fsdp_kwargs = dict(
        mesh=dp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
    )
    if ignored_params:
        fsdp_kwargs["ignored_params"] = set(ignored_params)

    # A multimodal model's vision blocks are shards of their own (frozen or trained): otherwise the
    # whole tower would sit in the root group, gathered for the entire step.
    from palingenesis.multimodal import vision_layers

    for block in vision_layers(model):
        fully_shard(block, **fsdp_kwargs)

    # Find and shard transformer layers bottom-up
    layers = [layer for layer in _find_transformer_layers(model) if any(True for _ in layer.parameters())]
    reshard = config.reshard_after_forward if reshard_layers is None else reshard_layers
    for i, layer in enumerate(layers):
        # Last layer optimization: don't reshard after forward since FSDP
        # would immediately prefetch it for backward (saves one reshard op)
        is_last = i == len(layers) - 1
        layer_reshard = False if is_last else reshard
        fully_shard(layer, **{**fsdp_kwargs, "reshard_after_forward": layer_reshard})

    # The root owns the rest: embeddings, final norm and output head (tied or not).
    # The losses reach the head outside the model's forward (chunked CE, DPO), after a
    # root forward (logits.final_hidden_states) has gathered these parameters; the root
    # keeps them gathered until its backward.
    fully_shard(model, **{**fsdp_kwargs, "mesh": root_mesh, "reshard_after_forward": False})

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
    # model.language_model.layers: a multimodal model's language model (its vision tower's blocks
    # are palingenesis.multimodal.vision_layers)
    for attr_path in ("model.language_model.layers", "model.layers", "transformer.h", "transformer.layers"):
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


def clip_grad_norm_(parameters, max_norm: float = 0.0, plain_copies: int = 1) -> torch.Tensor:
    """The global L2 norm of the gradients (and, when max_norm > 0, the gradients scaled down to it) for
    parameters split in several ways at once: DTensors over different meshes (tensor-parallel layers next to
    the whole embeddings and head of palingenesis.tensor_parallel), and different parameters on different
    pipeline stages.

    torch.nn.utils.clip_grad_norm_ needs one mesh for every DTensor and knows nothing of pipeline stages
    (torchtitan reduces the norm over the pipeline group itself for that reason, distributed/utils.py). Here
    every rank adds the squares of the gradient values it holds, each divided by the number of ranks holding the
    same values (the replicated dimensions of its placement; `plain_copies` for plain tensors), and one
    all-reduce over every process adds them up: one collective, and no gradient is ever gathered.
    Returns the norm before clipping."""
    from torch.distributed.tensor import DTensor

    groups: dict[tuple, list[torch.Tensor]] = {}
    for p in parameters:
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            if any(pl.is_partial() for pl in g.placements):  # never after FSDP's reduce-scatter
                g = g.redistribute(placements=[pl if not pl.is_partial() else _replicate() for pl in g.placements])
            copies = 1
            for dim, pl in enumerate(g.placements):
                if pl.is_replicate():
                    copies *= g.device_mesh.size(dim)
            local = g.to_local()
        else:
            copies, local = plain_copies, g
        groups.setdefault((local.device, local.dtype, copies), []).append(local)
    if not groups:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        total = torch.zeros((), device=device, dtype=torch.float32)
    else:
        total = sum(
            torch.stack(torch._foreach_norm(grads)).float().pow(2).sum() / copies
            for (_, _, copies), grads in groups.items()
        )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    norm = total.sqrt()
    if max_norm > 0:
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for (device, _, _), grads in groups.items():
            torch._foreach_mul_(grads, coef.to(device))
    return norm


def _replicate():
    from torch.distributed.tensor import Replicate

    return Replicate()
