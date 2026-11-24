"""Tensor parallelism for transformers' decoder models: every layer split over a group of ranks.

Layout (Megatron-LM's column / row parallel linears with sequence parallelism, Korthikanti et al. 2022,
built on torch.distributed.tensor.parallel as torchtitan does):

  * between the blocks, the residual stream is split along the sequence: a DTensor Shard(1) of the global
    [B, S, H] shape over the "tp" mesh. The norms run on each rank's tokens (SequenceParallel);
  * entering attention, linear attention or an MLP, one all-gather rebuilds the whole sequence; the
    projections into heads / MLP columns are column-parallel (each rank holds its heads), the projection
    back is row-parallel and ends in one reduce-scatter onto the sequence split. Per layer and direction:
    two all-gathers and two reduce-scatters of B x S x H, nothing else;
  * Gated DeltaNet layers (Qwen3.5) split by heads like attention: the fused q / k / v projection and the
    short convolution are stored rank by rank ([q_0 k_0 v_0 | q_1 k_1 v_1 | ...], `Sections`), so each rank's
    slice is the layout transformers' own forward expects for its heads, with no copy at run time. The
    decay (A_log, dt_bias) splits by value head; the gated norm's weight is shared;
  * the embeddings, the final norm and the output head are not split. Each rank applies them to its own
    slice of the sequence: the losses (chunked cross-entropy, Cut Cross-Entropy, Liger, the gated
    objectives) run unchanged on [B, S / tp, H] hidden states and the matching labels. Per rank that is
    1 / tp of the logits FLOPs and memory, the same as a vocabulary-parallel cross-entropy, without its
    custom softmax (which the fused kernels cannot use). Their gradients come from different tokens on
    every rank: FSDP2 shards them over data x tensor ranks and sums them there
    (palingenesis.distributed.sequence_mesh).

The trainer pads each batch to a multiple of tp tokens and gives every rank its labels' slice
(`TensorParallel.shard`); the model gets the whole (padded) input ids, so positions, masks and packed
document boundaries are computed for the whole sequence as without tensor parallelism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

logger = logging.getLogger(__name__)

# model_type of the text models whose layers this module splits (structure is checked layer by layer anyway).
SUPPORTED_MODEL_TYPES = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_text",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
)


class TensorParallelError(RuntimeError):
    """The model (or a module a plugin put into it) does not fit the tensor-parallel plan."""


@dataclass
class Sections:
    """A parameter whose dimension `dim` concatenates `sizes` blocks (q | k | v), each split by heads: it is
    stored with the blocks interleaved rank by rank, and `canonical` restores the checkpoint layout."""

    dim: int
    sizes: tuple[int, ...]

    def order(self, tp: int) -> torch.Tensor:
        """Positions of the rank-by-rank layout in the canonical one."""
        starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        parts = []
        for rank in range(tp):
            for start, size in zip(starts, self.sizes):
                part = size // tp
                parts.append(torch.arange(start + rank * part, start + (rank + 1) * part))
        return torch.cat(parts)

    def to_ranks(self, t: torch.Tensor, tp: int) -> torch.Tensor:
        return t.index_select(self.dim, self.order(tp).to(t.device))

    def canonical(self, t: torch.Tensor, tp: int) -> torch.Tensor:
        inverse = torch.argsort(self.order(tp)).to(t.device)
        return t.index_select(self.dim, inverse)


@dataclass
class TensorParallel:
    """One tensor-parallel group of a run and what its plan changed in the model."""

    mesh: object  # DeviceMesh of the "tp" dimension
    size: int
    rank: int
    # parameter name (as in the model's state dict) -> its rank-by-rank layout
    sections: dict[str, Sections] = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, mesh) -> TensorParallel:
        return cls(mesh=mesh, size=mesh.size(), rank=mesh.get_local_rank())

    @property
    def group(self):
        return self.mesh.get_group()

    # -- inputs

    def padded_length(self, seq_len: int) -> int:
        return -(-seq_len // self.size) * self.size

    def pad(self, t: torch.Tensor, pad_value=0) -> torch.Tensor:
        """A per-token batch field [B, S, ...] right-padded with `pad_value` to a multiple of tp tokens."""
        total = self.padded_length(t.shape[1])
        if total == t.shape[1]:
            return t
        pad = t.new_full((t.shape[0], total - t.shape[1], *t.shape[2:]), pad_value)
        return torch.cat([t, pad], dim=1)

    def pad_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Packed rows' position ids padded like `pad`: the padding is a document of its own (positions from 0),
        so packed attention and linear attention keep it apart from the last real document."""
        total = self.padded_length(position_ids.shape[1])
        extra = total - position_ids.shape[1]
        if not extra:
            return position_ids
        pad = torch.arange(extra, device=position_ids.device, dtype=position_ids.dtype)
        return torch.cat([position_ids, pad.expand(position_ids.shape[0], -1)], dim=1)

    def shard(self, t: torch.Tensor, pad_value=0) -> torch.Tensor:
        """This rank's tokens of a per-token field [B, S, ...] (padded first): the contiguous slice the model's
        final hidden states hold on this rank."""
        t = self.pad(t, pad_value)
        local = t.shape[1] // self.size
        return t.narrow(1, self.rank * local, local).contiguous()

    def all_reduce_sum(self, values: list[float], device) -> list[float]:
        import torch.distributed as dist

        t = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.group)
        return t.tolist()

    # -- checkpoints

    def canonical_state_dict(self, state: dict) -> dict:
        """A gathered (full) state dict in the checkpoint's own layout: the rank-by-rank parameters restored."""
        if not self.sections:
            return state
        out = dict(state)
        for name, tensor in state.items():
            key = _strip(name)
            if key in self.sections:
                out[name] = self.sections[key].canonical(tensor, self.size)
        return out

    def layout(self) -> dict:
        """What a resumed run must match to read this run's sharded checkpoints."""
        return {"tp_degree": self.size, "rank_by_rank": sorted(self.sections)}


def _strip(name: str) -> str:
    for prefix in ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module."):
        name = name.replace(prefix, "")
    return name


# ── checks ──────────────────────────────────────────────────────────────────────


def _text_config(model: nn.Module):
    config = model.config
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def support_errors(model_config, tp: int) -> list[str]:
    """Why `tp` cannot split a model with this (text) config; [] if it can. Also used before any weight is
    loaded (pgs fit, the memory estimator)."""
    from palingenesis.parallel_dims import head_errors

    model_type = getattr(model_config, "model_type", "")
    errors = []
    if model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(
            f"parallel.tp_degree: model_type {model_type!r} has no tensor-parallel plan (supported: "
            f"{', '.join(SUPPORTED_MODEL_TYPES)}). Use FSDP (parallel.tp_degree: 1) or pipeline parallelism."
        )
    heads = {
        "num_attention_heads": getattr(model_config, "num_attention_heads", 0),
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", 0)
        or getattr(model_config, "num_attention_heads", 0),
    }
    if "linear_attention" in (getattr(model_config, "layer_types", None) or []):
        heads["linear_num_key_heads"] = getattr(model_config, "linear_num_key_heads", 0)
        heads["linear_num_value_heads"] = getattr(model_config, "linear_num_value_heads", 0)
    for name in ("intermediate_size", "moe_intermediate_size"):
        if getattr(model_config, name, None):
            heads[name] = getattr(model_config, name)
    errors += head_errors(tp, heads)
    if getattr(model_config, "num_experts", 0) and getattr(model_config, "shared_expert_intermediate_size", 0):
        errors.append(
            "parallel.tp_degree: MoE layers with a shared expert (Qwen3.5-MoE, Qwen3-Next) are not split yet; "
            "use FSDP with parallel.expert_parallel."
        )
    return errors


# ── parallel styles ─────────────────────────────────────────────────────────────


def _gather_sequence(module: nn.Module) -> None:
    """The block's input (the first positional tensor, or `hidden_states=`) from the sequence split to the whole
    sequence on every rank: one all-gather, whose backward is one reduce-scatter of the summed partial
    gradients of the column-parallel projections (DTensor autograd)."""
    from torch.distributed.tensor import DTensor, Replicate

    def gather(x):
        if not isinstance(x, DTensor):
            raise TensorParallelError(
                f"{type(module).__name__} received a plain tensor; under tensor parallelism the residual stream is "
                "a sequence-split DTensor (a module outside the plan replaced it)."
            )
        return x.redistribute(placements=[Replicate()], async_op=True)

    def hook(mod, args, kwargs):
        if args and torch.is_tensor(args[0]):
            return (gather(args[0]), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": gather(kwargs["hidden_states"])}
        raise TensorParallelError(f"{type(mod).__name__}: no hidden states among its inputs")

    module.register_forward_pre_hook(hook, with_kwargs=True)


class _LocalView(torch.autograd.Function):
    """A DTensor parameter's local shard as a plain tensor, for code that reads weights directly (kernels,
    depthwise convolutions). Backward: the local gradient, summed over the group first when every rank
    computed a different part of it (a replicated weight applied to each rank's heads)."""

    @staticmethod
    def forward(ctx, param, reduce_group):
        ctx.mesh, ctx.placements, ctx.group = param.device_mesh, param.placements, reduce_group
        ctx.shape, ctx.stride = param.shape, param.stride()
        return param.to_local().detach()

    @staticmethod
    def backward(ctx, grad):
        from torch.distributed.tensor import DTensor

        if ctx.group is not None:
            import torch.distributed._functional_collectives as funcol

            grad = funcol.all_reduce(grad.contiguous(), "sum", ctx.group)
        grad = DTensor.from_local(grad, ctx.mesh, ctx.placements, run_check=False, shape=ctx.shape, stride=ctx.stride)
        return grad, None


def _local_params(module: nn.Module, mesh, placements: dict[str, tuple]) -> None:
    """Distribute `module`'s parameters (name -> (placement, sum gradients over the group?)) and give its
    forward their local shards as plain tensors."""
    from torch.distributed.tensor import distribute_tensor

    for name, (placement, _) in placements.items():
        param = module._parameters[name]
        module.register_parameter(
            name,
            nn.Parameter(distribute_tensor(param.data, mesh, [placement]), requires_grad=param.requires_grad),
        )
    reduce = {name: mesh.get_group() if summed else None for name, (_, summed) in placements.items()}

    def pre(mod, args):
        saved = {}
        for name in placements:
            param = mod._parameters[name]
            saved[name] = param
            mod._parameters[name] = _LocalView.apply(param, reduce[name])
        mod._pgs_tp_saved = saved

    def post(mod, args, output):
        for name, param in mod.__dict__.pop("_pgs_tp_saved").items():
            mod._parameters[name] = param
        return output

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def _sequence_local_embedding(embedding: nn.Module, tp: TensorParallel) -> None:
    """The embedding looks up only this rank's slice of the input ids and returns it as the sequence-split
    DTensor of the whole [B, S, H] (the shape the model's forward derives positions and masks from)."""
    from torch.distributed.tensor import DTensor, Shard

    state = {}

    def pre(mod, args):
        ids = args[0]
        seq = ids.shape[1]
        if seq % tp.size:
            raise TensorParallelError(
                f"tensor parallelism: {seq} tokens per row is not a multiple of tp_degree={tp.size} (the trainer "
                "pads every batch; a caller of the model must too)."
            )
        state["seq"] = seq
        local = seq // tp.size
        return (ids.narrow(1, tp.rank * local, local), *args[1:])

    def post(mod, args, output):
        b, _, h = output.shape
        seq = state.pop("seq")
        return DTensor.from_local(
            output, tp.mesh, [Shard(1)], run_check=False, shape=torch.Size((b, seq, h)), stride=(seq * h, h, 1)
        )

    embedding.register_forward_pre_hook(pre)
    embedding.register_forward_hook(post)


def _sequence_local_input(module: nn.Module) -> None:
    """The final norm receives this rank's slice of the sequence-split residual stream as a plain tensor: it,
    the output head and the loss then work on [B, S / tp, H]."""
    from torch.distributed.tensor import DTensor, Shard

    def pre(mod, args):
        x = args[0]
        if isinstance(x, DTensor):
            if tuple(x.placements) != (Shard(1),):
                x = x.redistribute(placements=[Shard(1)])
            x = x.to_local()
        return (x, *args[1:])

    module.register_forward_pre_hook(pre)


def _is_norm(module: nn.Module) -> bool:
    params = dict(module.named_parameters(recurse=False))
    return "norm" in type(module).__name__.lower() and set(params) <= {"weight", "bias"} and not list(module.children())


# ── plans ───────────────────────────────────────────────────────────────────────


def _attention_plan(attn: nn.Module, prefix: str) -> dict:
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel

    missing = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj") if not isinstance(getattr(attn, n, None), nn.Linear)]
    if missing:
        raise TensorParallelError(f"{prefix}: attention without plain nn.Linear {', '.join(missing)}")
    plan = {f"{prefix}.{n}": ColwiseParallel() for n in ("q_proj", "k_proj", "v_proj")}
    plan[f"{prefix}.o_proj"] = RowwiseParallel(output_layouts=Shard(1), use_local_output=False)
    for norm in ("q_norm", "k_norm"):  # per head, over head_dim: on each rank's heads, weight shared
        if getattr(attn, norm, None) is not None:
            plan[f"{prefix}.{norm}"] = SequenceParallel(sequence_dim=2, use_local_output=True)
    return plan


def _mlp_plan(mlp: nn.Module, prefix: str) -> dict:
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    names = [n for n in ("gate_proj", "up_proj", "down_proj") if isinstance(getattr(mlp, n, None), nn.Linear)]
    if names != ["gate_proj", "up_proj", "down_proj"]:
        raise TensorParallelError(f"{prefix}: MLP without plain nn.Linear gate_proj / up_proj / down_proj")
    return {
        f"{prefix}.gate_proj": ColwiseParallel(),
        f"{prefix}.up_proj": ColwiseParallel(),
        f"{prefix}.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
    }


def _gated_delta_net(gdn: nn.Module, name: str, tp: TensorParallel) -> dict:
    """Qwen3.5's Gated DeltaNet split by heads. Returns the plan of its projections; distributes the rest."""
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"):
        if getattr(gdn, attr, None) is None:
            raise TensorParallelError(
                f"{name}: a linear-attention layer without {attr} (only Qwen3.5's Gated DeltaNet layout is split; "
                "Qwen3-Next's fused in_proj_qkvz is not)"
            )
    n = tp.size
    sections = Sections(0, (gdn.key_dim, gdn.key_dim, gdn.value_dim))
    with torch.no_grad():
        gdn.in_proj_qkv.weight.data = sections.to_ranks(gdn.in_proj_qkv.weight.data, n)
        gdn.conv1d.weight.data = sections.to_ranks(gdn.conv1d.weight.data, n)
        if gdn.conv1d.bias is not None:
            gdn.conv1d.bias.data = Sections(0, sections.sizes).to_ranks(gdn.conv1d.bias.data, n)
    tp.sections[f"{name}.in_proj_qkv.weight"] = sections
    tp.sections[f"{name}.conv1d.weight"] = sections
    conv = {"weight": (Shard(0), False)}
    if gdn.conv1d.bias is not None:
        tp.sections[f"{name}.conv1d.bias"] = sections
        conv["bias"] = (Shard(0), False)
    _local_params(gdn.conv1d, tp.mesh, conv)
    _local_params(gdn, tp.mesh, {"A_log": (Shard(0), False), "dt_bias": (Shard(0), False)})
    _local_params(gdn.norm, tp.mesh, {k: (Replicate(), True) for k, _ in gdn.norm.named_parameters(recurse=False)})
    # transformers' forward reads its sizes from these attributes: this rank's heads
    gdn.num_k_heads //= n
    gdn.num_v_heads //= n
    gdn.key_dim //= n
    gdn.value_dim //= n
    gdn.conv_dim //= n
    gdn.conv1d.in_channels = gdn.conv1d.out_channels = gdn.conv1d.groups = gdn.conv_dim
    local = gdn.key_dim * 2 + gdn.value_dim
    if gdn.conv1d.weight.to_local().shape[0] != local:
        raise TensorParallelError(f"{name}: the convolution's channels do not split by heads")
    return {
        "in_proj_qkv": ColwiseParallel(),
        "in_proj_z": ColwiseParallel(),
        "in_proj_b": ColwiseParallel(),
        "in_proj_a": ColwiseParallel(),
        "out_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
    }


class _ReduceBackward(torch.autograd.Function):
    """Identity forward; the gradient summed over the group (an input every rank uses with its own experts'
    slice: each rank's gradient is a part of the whole)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        import torch.distributed._functional_collectives as funcol

        return funcol.all_reduce(grad.contiguous(), "sum", ctx.group), None


def _moe_block(block: nn.Module, name: str, tp: TensorParallel) -> None:
    """A sparse MoE block (Qwen3-MoE): every rank routes the whole sequence identically and runs its slice of
    every expert's intermediate columns (expert tensor parallelism); the experts' partial outputs are summed by
    the reduce-scatter onto the sequence split. The router is replicated: its gradient is the same on every
    rank (the routing weights' gradient is summed over the group first), so it is not reduced."""
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

    experts, router = getattr(block, "experts", None), getattr(block, "gate", None)
    if experts is None or router is None or not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        raise TensorParallelError(f"{name}: an MoE block without fused experts (gate_up_proj, down_proj) and a router")
    if getattr(block, "shared_expert", None) is not None:
        raise TensorParallelError(f"{name}: MoE blocks with a shared expert are not split yet")
    n = tp.size
    inter = experts.gate_up_proj.shape[1] // 2
    sections = Sections(1, (inter, inter))
    with torch.no_grad():
        experts.gate_up_proj.data = sections.to_ranks(experts.gate_up_proj.data, n)
    tp.sections[f"{name}.experts.gate_up_proj"] = sections
    _local_params(experts, tp.mesh, {"gate_up_proj": (Shard(1), False), "down_proj": (Shard(2), False)})
    if hasattr(experts, "intermediate_dim"):
        experts.intermediate_dim //= n
    _local_params(router, tp.mesh, {k: (Replicate(), False) for k, _ in router.named_parameters(recurse=False)})

    def gather(mod, args):
        x = args[0]
        if not isinstance(x, DTensor):
            raise TensorParallelError(f"{name}: expected the sequence-split residual stream")
        return (x.redistribute(placements=[Replicate()]).to_local(grad_placements=[Replicate()]), *args[1:])

    def experts_in(mod, args):
        hidden, index, weights = args[:3]
        return (_ReduceBackward.apply(hidden, tp.group), index, _ReduceBackward.apply(weights, tp.group), *args[3:])

    def experts_out(mod, args, output):
        return DTensor.from_local(output, tp.mesh, [Partial()], run_check=False)

    def scatter(mod, args, output):
        if isinstance(output, tuple):  # (hidden, router logits)
            return (output[0].redistribute(placements=[Shard(1)]), *output[1:])
        return output.redistribute(placements=[Shard(1)])

    block.register_forward_pre_hook(gather)
    experts.register_forward_pre_hook(experts_in)
    experts.register_forward_hook(experts_out)
    block.register_forward_hook(scatter)


def _parallelize_layer(layer: nn.Module, name: str, tp: TensorParallel) -> None:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import SequenceParallel, parallelize_module

    plan: dict = {}
    for child_name, child in layer.named_children():
        path = child_name
        if child_name == "self_attn":
            _gather_sequence(child)
            plan.update(_attention_plan(child, path))
        elif child_name == "linear_attn":
            _gather_sequence(child)
            plan.update({f"{path}.{k}": v for k, v in _gated_delta_net(child, f"{name}.{path}", tp).items()})
        elif child_name == "mlp" and hasattr(child, "experts"):
            _moe_block(child, f"{name}.{path}", tp)
        elif child_name == "mlp":
            _gather_sequence(child)
            plan.update(_mlp_plan(child, path))
        elif _is_norm(child):
            plan[path] = SequenceParallel()
        elif any(True for _ in child.parameters()):
            raise TensorParallelError(
                f"{name}.{child_name} ({type(child).__name__}) is not covered by the tensor-parallel plan (a plugin "
                "that wraps or replaces layers? It must run with parallel.tp_degree: 1)."
            )
    parallelize_module(layer, tp.mesh, plan)
    plain = [n for n, p in layer.named_parameters() if not isinstance(p, DTensor)]
    if plain:
        raise TensorParallelError(f"{name}: parameters left out of the tensor-parallel plan: {plain[:4]}")


def apply_tensor_parallel(model: nn.Module, tp_mesh) -> TensorParallel:
    """Split `model`'s decoder layers over `tp_mesh` (before activation checkpointing, compile and FSDP2)."""
    from palingenesis.distributed import _find_transformer_layers
    from palingenesis.logits import backbone_of

    tp = TensorParallel.from_mesh(tp_mesh)
    errors = support_errors(_text_config(model), tp.size)
    if errors:
        raise TensorParallelError("\n".join(errors))
    backbone = backbone_of(model)
    embed, norm = getattr(backbone, "embed_tokens", None), getattr(backbone, "norm", None)
    if embed is None or norm is None:
        raise TensorParallelError("tensor parallelism needs the backbone's embed_tokens and final norm")
    names = {id(m): n for n, m in model.named_modules()}
    _sequence_local_embedding(embed, tp)
    _sequence_local_input(norm)
    layers = _find_transformer_layers(model)
    for layer in layers:
        _parallelize_layer(layer, names[id(layer)], tp)
    logger.info(
        f"Tensor parallel: {len(layers)} layers split over {tp.size} ranks (heads and MLP columns; sequence-parallel "
        "norms); embeddings, final norm and output head on each rank's slice of the sequence"
    )
    return tp
