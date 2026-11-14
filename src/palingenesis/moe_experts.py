"""Expert compute for MoE layers: token dispatch, grouped GEMM, expert parallelism.

A MoE layer applies to every token the MLPs of its `top_k` chosen experts. transformers stores
all experts of a layer in stacked parameters ([experts, out, in]) and offers several forwards
over them; its default sorts the (token, expert) pairs by expert and runs one grouped GEMM per
projection. This module has the same computation on the same parameters (so checkpoints keep
transformers' layout), in two forms:

  `experts_forward`   one device: dispatch (a stable sort of narrow integer keys), the two
                      projections as grouped GEMMs, combine. Registered with transformers as
                      the experts implementations "palingenesis" (torch's grouped GEMM) and
                      "palingenesis_padded" (every expert's tokens padded to the busiest
                      expert's count, one batched GEMM: no per-expert kernel launches, more
                      memory under imbalance); `moe.experts_implementation` selects one.

  `ExpertParallel`    several ranks: every rank keeps E / ranks experts of each layer instead of
                      gathering all of them for every forward (FSDP), and the tokens travel:
                      each (token, expert) pair goes to the rank that owns the expert
                      (all-to-all), is computed there with the tokens of every other rank, and
                      comes back. The expert parameters are sharded on their expert dimension
                      over the FSDP mesh (DTensors, as FSDP keeps them between forwards), so the
                      optimizer, gradient clipping and checkpoints treat them like any other
                      sharded parameter; everything else stays under FSDP. What moves per layer
                      is tokens x top_k x hidden activations instead of the layer's expert
                      weights (the bulk of a MoE model's parameters).

Gradients: an expert's weights receive the gradient of every token routed to it, from every
rank, on the rank that owns it: the sum FSDP's reduce-scatter would have produced.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.distributed as dist
from torch import Tensor, nn

logger = logging.getLogger(__name__)

GROUPED = "palingenesis"
PADDED = "palingenesis_padded"


# ─────────────────────────────────────────────────────────────── dispatch and kernels


def dispatch(top_k_index: Tensor, num_experts: int) -> tuple[Tensor, Tensor, Tensor]:
    """Order the (token, expert) pairs by expert.

    top_k_index: [N, top_k] expert of each pair. Returns (order [S], expert of each pair in that
    order [S], pairs per expert [E]); S = N * top_k, pair s belongs to token s // top_k. The sort
    is stable (a token's pairs keep their order) and runs on the narrowest integer type that
    holds an expert index: a radix sort of bytes for up to 256 experts."""
    flat = top_k_index.reshape(-1)
    key = flat.to(torch.uint8 if num_experts <= 256 else torch.int16)
    sorted_key, order = torch.sort(key, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    return order, sorted_key.long(), counts


def _loop_linear(x: Tensor, weight: Tensor, counts: Tensor) -> Tensor:
    """x[group e] @ weight[e], one GEMM per non-empty group (weight: [E, in, out])."""
    parts, start = [], 0
    for e, n in enumerate(counts.tolist()):
        if n:
            parts.append(x[start : start + n] @ weight[e])
            start += n
    return torch.cat(parts) if parts else x.new_zeros(0, weight.shape[-1])


def _native_grouped_mm(x: Tensor, weight: Tensor) -> bool:
    """Whether torch's grouped GEMM takes these operands (fp32 / bf16 / fp16; where its strides are
    aligned: a multiple of 16 bytes per row)."""
    if not hasattr(torch.nn.functional, "grouped_mm") or weight.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        return False
    return all(t.shape[-1] * t.element_size() % 16 == 0 for t in (x, weight))


def _compute_weight(weight: Tensor, x: Tensor) -> Tensor:
    """fp32 weights in the autocast dtype while autocast is on, as autocast does for nn.Linear:
    torch's grouped GEMM is not autocast-aware and would compute in fp32 (and cast every
    activation up to it)."""
    device = x.device.type
    if weight.dtype == torch.float32 and torch.is_autocast_enabled(device):
        return weight.to(torch.get_autocast_dtype(device))
    return weight


def grouped_linear(
    x: Tensor,
    weight: Tensor,
    counts: Tensor,
    *,
    transposed: bool = False,
    bias: Tensor | None = None,
    experts: Tensor | None = None,
    padded: bool = False,
) -> Tensor:
    """y[s] = W[expert of s] x[s] (+ b[expert of s]) for rows sorted by expert.

    x: [S, in]; weight: [E, out, in], or [E, in, out] when `transposed`; counts: [E] rows per
    expert; bias: [E, out] with `experts` [S] the expert of each row. `padded`: one batched GEMM
    over [E, busiest expert's rows, in] instead of a grouped GEMM."""
    w = _compute_weight(weight, x)
    w = w if transposed else w.transpose(-2, -1)  # [E, in, out]
    x = x.to(w.dtype)
    if padded:
        y = _padded_linear(x, w, counts)
    elif _native_grouped_mm(x, w):
        y = torch.nn.functional.grouped_mm(x, w, offs=torch.cumsum(counts, 0, dtype=torch.int32))
    else:
        y = _loop_linear(x, w, counts)
    if bias is not None:
        y = y + bias[experts].to(y.dtype)
    return y


def _padded_linear(x: Tensor, weight: Tensor, counts: Tensor) -> Tensor:
    """One batched GEMM: every expert's rows in a [E, M, in] block (M: the busiest expert's rows)."""
    num_experts = counts.shape[0]
    width = int(counts.max()) if x.shape[0] else 0
    if width == 0:
        return x.new_zeros(0, weight.shape[-1])
    starts = torch.cumsum(counts, 0) - counts
    expert = torch.repeat_interleave(torch.arange(num_experts, device=x.device), counts)
    slot = expert * width + (torch.arange(x.shape[0], device=x.device) - starts[expert])
    block = x.new_zeros(num_experts * width, x.shape[-1]).index_copy(0, slot, x)
    out = torch.bmm(block.view(num_experts, width, -1), weight)
    return out.reshape(num_experts * width, -1)[slot]


def expert_mlp(
    module: nn.Module,
    x: Tensor,
    experts: Tensor,
    counts: Tensor,
    weights: dict[str, Tensor] | None = None,
    padded: bool = False,
) -> Tensor:
    """The experts' MLP on rows sorted by expert: x [S, hidden] -> [S, hidden]. `module` is a
    transformers experts module (its layout flags and gating are used); `weights` replaces its
    parameters by name (expert parallelism: the rank's own experts)."""
    get = (lambda name: weights[name]) if weights is not None else (lambda name: getattr(module, name))
    has_bias = getattr(module, "has_bias", False)
    transposed = getattr(module, "is_transposed", False)
    up = "gate_up_proj" if getattr(module, "has_gate", True) else "up_proj"
    h = grouped_linear(
        x,
        get(up),
        counts,
        transposed=transposed,
        bias=get(f"{up}_bias") if has_bias else None,
        experts=experts,
        padded=padded,
    )
    h = module._apply_gate(h) if getattr(module, "has_gate", True) else module.act_fn(h)
    return grouped_linear(
        h,
        get("down_proj"),
        counts,
        transposed=transposed,
        bias=get("down_proj_bias") if has_bias else None,
        experts=experts,
        padded=padded,
    )


def _combine(y: Tensor, order: Tensor, gates: Tensor, tokens: int, dtype: torch.dtype) -> Tensor:
    """Gate the sorted pair outputs and sum each token's top_k of them (in the gates' precision:
    fp32 when the router ran in fp32; a token's pairs are summed in a fixed order)."""
    y = y * gates.reshape(-1)[order].unsqueeze(-1)
    unsorted = torch.empty_like(y).index_copy(0, order, y)
    return unsorted.view(tokens, -1, y.shape[-1]).sum(dim=1).to(dtype)


@torch.compiler.disable  # data-dependent group sizes: under model.compile the experts stay eager (a graph break)
def experts_forward(
    self: nn.Module, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor, padded: bool = False
) -> Tensor:
    """sum over a token's chosen experts of gate * expert MLP(token): hidden_states [N, hidden],
    top_k_index and top_k_weights [N, top_k] -> [N, hidden]. The forward of a transformers experts
    module (same signature, same parameters)."""
    top_k = top_k_index.shape[-1]
    order, experts, counts = dispatch(top_k_index, self.num_experts)
    x = hidden_states[order // top_k]
    y = expert_mlp(self, x, experts, counts, padded=padded)
    return _combine(y, order, top_k_weights, hidden_states.shape[0], hidden_states.dtype)


def _padded_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return experts_forward(self, hidden_states, top_k_index, top_k_weights, padded=True)


def register_experts_implementations() -> None:
    """Make "palingenesis" and "palingenesis_padded" valid `experts_implementation` values of
    transformers models (from_pretrained(..., experts_implementation=...), set_experts_implementation)."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    for name, fn in ((GROUPED, experts_forward), (PADDED, _padded_experts_forward)):
        if name not in ALL_EXPERTS_FUNCTIONS:
            ALL_EXPERTS_FUNCTIONS.register(name, fn)


def set_experts_implementation(model: nn.Module, name: str) -> None:
    """Apply `moe.experts_implementation` to a loaded model: "auto" keeps transformers' choice,
    "grouped" / "padded" are this module's forwards, anything else a transformers name (eager,
    grouped_mm, batched_mm, ...). Nothing for a model without transformers experts modules."""
    if name in ("", "auto") or not hasattr(model, "set_experts_implementation"):
        return
    if name in ("grouped", "padded"):
        register_experts_implementations()
        name = GROUPED if name == "grouped" else PADDED
    try:
        model.set_experts_implementation(name)
    except (ValueError, KeyError) as e:
        from palingenesis.config import ConfigError

        raise ConfigError(f"moe.experts_implementation={name!r}: {e}") from None
    logger.info(f"MoE experts implementation: {name}")


# ─────────────────────────────────────────────────────────────── expert parallelism


def _exchange(x: Tensor, matrix: Tensor, rank: int, group) -> Tensor:
    """Send rows of x to the other ranks and receive theirs: matrix[src, dst] rows go from src to
    dst; x holds this rank's rows by destination, the result the received rows by source."""
    send, recv = matrix[rank].tolist(), matrix[:, rank].tolist()
    out = x.new_empty(sum(recv), *x.shape[1:])
    if dist.get_backend(group) == "nccl":
        dist.all_to_all_single(out, x.contiguous(), output_split_sizes=recv, input_split_sizes=send, group=group)
        return out
    # Backends without all-to-all (gloo: CPU tests): every rank's rows to every rank, padded.
    world = matrix.shape[0]
    longest = int(matrix.sum(1).max())
    padded = x.new_zeros(longest, *x.shape[1:])
    padded[: x.shape[0]] = x
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    start = 0
    for src in range(world):
        offset = int(matrix[src, :rank].sum())
        out[start : start + recv[src]] = gathered[src][offset : offset + recv[src]]
        start += recv[src]
    return out


class _AllToAll(torch.autograd.Function):
    """Rows to their destination ranks; the backward sends the gradients the way they came."""

    @staticmethod
    def forward(ctx, x, matrix, rank, group):
        ctx.matrix, ctx.rank, ctx.group = matrix, rank, group
        return _exchange(x, matrix, rank, group)

    @staticmethod
    def backward(ctx, grad):
        return _exchange(grad, ctx.matrix.t().contiguous(), ctx.rank, ctx.group), None, None, None


EXPERT_PARAMETERS = ("gate_up_proj", "gate_up_proj_bias", "up_proj", "up_proj_bias", "down_proj", "down_proj_bias")


class ExpertParallel:
    """Expert parallelism over a 1-D device mesh (see the module docstring).

    `ExpertParallel.apply(model, mesh)` before FSDP is applied: every MoE layer's expert
    parameters become DTensors sharded on the expert dimension (this rank keeps experts
    [rank * E / ranks, (rank + 1) * E / ranks)), and the experts' forward becomes dispatch /
    local compute / return. The returned parameters must be passed to FSDP as ignored."""

    def __init__(self, mesh, compute_dtype: torch.dtype | None = None, padded: bool = False):
        self.mesh = mesh
        self.group = mesh.get_group()
        self.world = mesh.size()
        self.rank = mesh.get_local_rank()
        self.compute_dtype = compute_dtype
        self.padded = padded
        self.parameters: list[nn.Parameter] = []

    @classmethod
    def apply(cls, model: nn.Module, mesh, compute_dtype: torch.dtype | None = None, padded: bool = False):
        """Returns the ExpertParallel (its `.parameters`: what FSDP must ignore), or None for a
        model without MoE layers."""
        from torch.distributed.tensor import DTensor, Shard

        from palingenesis.config import ConfigError
        from palingenesis.moe import find_moe_layers

        layers = find_moe_layers(model)
        if not layers:
            return None
        if mesh.ndim != 1:
            raise ConfigError("parallel.expert_parallel needs a 1-D mesh (no context parallelism).")
        ep = cls(mesh, compute_dtype, padded)
        device = torch.device("cuda", torch.cuda.current_device()) if mesh.device_type == "cuda" else None
        for layer in layers:
            if layer.num_experts % ep.world:
                raise ConfigError(
                    f"parallel.expert_parallel: {layer.num_experts} experts do not divide over {ep.world} ranks "
                    "(every rank keeps the same number of experts of each layer)."
                )
            local = layer.num_experts // ep.world
            lo = ep.rank * local
            for name in EXPERT_PARAMETERS:
                full = layer.experts._parameters.get(name)
                if full is None:
                    continue
                own = full.detach()[lo : lo + local].clone()
                if device is not None:
                    own = own.to(device)
                sharded = nn.Parameter(
                    DTensor.from_local(own, mesh, [Shard(0)], run_check=False), requires_grad=full.requires_grad
                )
                layer.experts._parameters[name] = sharded
                ep.parameters.append(sharded)
            layer.experts.forward = types.MethodType(ep._forward, layer.experts)
        logger.info(
            f"Expert parallel: {len(layers)} MoE layers, {layers[0].num_experts // ep.world} of "
            f"{layers[0].num_experts} experts per rank on {ep.world} ranks; tokens are dispatched all-to-all"
        )
        return ep

    def _local_weights(self, module: nn.Module, like: Tensor) -> dict[str, Tensor]:
        """This rank's experts, in the dtype the forward computes in (FSDP's mixed precision casts
        its gathered parameters the same way)."""
        dtype = self.compute_dtype
        if dtype is None and like.dtype in (torch.bfloat16, torch.float16):
            dtype = like.dtype
        out = {}
        for name in EXPERT_PARAMETERS:
            p = module._parameters.get(name)
            if p is not None:
                w = p.to_local()
                out[name] = w.to(dtype) if dtype is not None and w.dtype == torch.float32 else w
        return out

    @torch.compiler.disable  # collectives and data-dependent sizes stay out of compiled graphs
    def _forward(self, module: nn.Module, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        num_experts, top_k = module.num_experts, top_k_index.shape[-1]
        local = num_experts // self.world
        lo = self.rank * local
        # 1. this rank's pairs by expert, hence by owning rank
        order, _, counts = dispatch(top_k_index, num_experts)
        x = hidden_states[order // top_k]
        # 2. every rank's pairs per expert: who sends how many rows to whom, and for which experts
        all_counts = [torch.empty_like(counts) for _ in range(self.world)]
        dist.all_gather(all_counts, counts, group=self.group)
        all_counts = torch.stack(all_counts)  # [ranks, E]
        matrix = all_counts.view(self.world, self.world, local).sum(-1).cpu()  # [src, dst]
        # 3. dispatch; the rows arrive by source rank, each source's sorted by expert
        received = _AllToAll.apply(x, matrix, self.rank, self.group)
        mine = all_counts[:, lo : lo + local]  # [src, local experts]
        expert_of_row = torch.repeat_interleave(
            torch.arange(local, device=x.device).repeat(self.world), mine.reshape(-1)
        )
        by_expert = torch.sort(expert_of_row.to(torch.int16), stable=True)[1]
        experts = expert_of_row[by_expert]
        # 4. this rank's experts on every rank's tokens
        weights = self._local_weights(module, received)
        y = expert_mlp(module, received[by_expert], experts, mine.sum(0), weights, self.padded)
        y = torch.empty_like(y).index_copy(0, by_expert, y)  # back to arrival order
        # 5. return to the senders (the same rows, the other way), gate and sum per token
        returned = _AllToAll.apply(y, matrix.t().contiguous(), self.rank, self.group)
        return _combine(returned, order, top_k_weights, hidden_states.shape[0], hidden_states.dtype)
