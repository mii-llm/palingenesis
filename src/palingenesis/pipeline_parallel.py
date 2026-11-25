"""Pipeline parallelism for transformers' decoder models: the layers split into stages over the "pp" ranks.

Built on torch.distributed.pipelining (PipelineStage + its schedules: GPipe, 1F1B, interleaved 1F1B, looped BFS,
interleaved zero bubble, ZBV), with torchtitan's stage split (layers balanced with the embeddings and the head
counted as layers; looped or V-shaped stage placement) and Megatron-Core's convention of fixed micro-batch
shapes.

How a stage runs transformers' own model:

  * each pipeline rank keeps only its stages' layers; the others become parameter-less stand-ins
    (`_AbsentLayer`) that pass the hidden states through, so the backbone's forward runs unchanged: rotary
    embeddings, causal and packed masks, per-layer mask and position choices by layer index (Qwen3.5's linear /
    full attention, Gemma3's sliding / global) are computed as without pipelining. Parameter names are the
    model's own (model.layers.<i>...): checkpoints (DCP) and the HF export are the same files at any pp_degree;
    * the first stage embeds the token ids; the others take the previous stage's hidden states as
      `inputs_embeds`. Under tensor parallelism the stages exchange each rank's sequence slice only;
    * the last stage applies the final norm and computes the loss itself (chunked cross-entropy, Cut
      Cross-Entropy or plain cross-entropy, per-token weights, the global valid-token denominator of the
      optimizer step), so the schedule only backpropagates a scalar;
  * every micro-batch has the same shape ([rows / pp_microbatches, data.max_seq_length], rounded up to a
    multiple of tp_degree): PipelineStage sizes its buffers once. Packed rows keep their documents apart through
    their position ids, which every stage receives (variable-length kernels: flattened per micro-batch).

Gradients: the schedule runs the backward of every micro-batch and FSDP2 reduces once at the end of each step
(PipelineStage.perform_reduce_grad); the loss is already normalised by the global token count, so nothing is
rescaled (scale_grads=False).
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field

import torch
from torch import nn

logger = logging.getLogger(__name__)

# model_type of the models whose backbone takes `inputs_embeds` unchanged (gemma / gemma2 scale them in the
# backbone's forward, which every stage would repeat).
SUPPORTED_MODEL_TYPES = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_text",
    "gemma3",
    "gemma3_text",
)


class PipelineParallelError(RuntimeError):
    pass


class _AbsentLayer(nn.Module):
    """A decoder layer that lives on another pipeline stage: the hidden states pass through. Keeps the layer's
    plain attributes (attention_type, layer_type, ...) for backbones that read them."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        for key, value in vars(layer).items():
            if not key.startswith("_") and isinstance(value, (bool, int, float, str, type(None))):
                setattr(self, key, value)

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class _NoNorm(nn.Module):
    """The final norm of a stage that is not the last: the hidden states go to the next stage as they are."""

    def forward(self, hidden_states):
        return hidden_states


@dataclass
class StagePlan:
    index: int
    layers: list[int]
    first: bool
    last: bool


@dataclass
class _Step:
    """What the last stage needs to compute the loss of the current step."""

    loss_denom: float = 1.0
    evaluating: bool = False
    eval_sums: list = field(default_factory=list)


def _module_copy(module: nn.Module, children: dict) -> nn.Module:
    """A new module object of `module`'s class sharing its parameters, buffers and attributes, with its own
    child-module dict (`children` replaced): a stage's view of a model without copying any weight."""
    new = module.__class__.__new__(module.__class__)
    state = {}
    for key, value in vars(module).items():
        state[key] = value.copy() if isinstance(value, (dict, set)) else value
    new.__dict__.update(state)
    for name, child in children.items():
        new._modules[name] = child
    return new


def _layer_list(backbone: nn.Module) -> nn.ModuleList:
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise PipelineParallelError(
            "pipeline parallelism needs the backbone's layers as an nn.ModuleList (model.layers)"
        )
    return layers


class PipelineParallel:
    """This rank's pipeline stages and the schedule that runs them."""

    def __init__(self, model: nn.Module, mesh, parallel, seq_len: int, flatten_packed: bool):
        from palingenesis.logits import backbone_of
        from palingenesis.parallel_dims import V_SHAPED, split_layers, stage_indices, stages_per_rank

        config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        if config.model_type not in SUPPORTED_MODEL_TYPES:
            raise PipelineParallelError(
                f"parallel.pp_degree: model_type {config.model_type!r} is not supported (supported: "
                f"{', '.join(SUPPORTED_MODEL_TYPES)})."
            )
        if getattr(config, "tie_word_embeddings", False) or _tied(model):
            raise PipelineParallelError(
                "parallel.pp_degree: this model ties its output head to its embeddings, which live on different "
                "pipeline stages (Megatron keeps both and all-reduces their gradients; not implemented). Use FSDP / "
                "tensor parallelism for it: tied models are small."
            )
        self.mesh = pp_mesh = mesh["pp"]
        self.size, self.rank = pp_mesh.size(), pp_mesh.get_local_rank()
        # the rank that writes this stage's part of an export: coordinate 0 on every other dimension
        self.leader = all(mesh[d].get_local_rank() == 0 for d in mesh.mesh_dim_names if d != "pp")
        self.schedule_name = parallel.pp_schedule
        self.per_rank = stages_per_rank(self.schedule_name, parallel.pp_stages_per_rank)
        self.num_stages = self.size * self.per_rank
        self.v_shape = self.schedule_name in V_SHAPED
        self.seq_len = seq_len
        self.flatten_packed = flatten_packed
        self.microbatches = parallel.pp_microbatches
        self.step_state = _Step()
        self.tp = None
        self.loss_kind = "chunked"
        self.stable = False
        self.num_chunks = None
        self.weighted = False  # this step's batch has per-token loss weights

        backbone = backbone_of(model)
        layers = _layer_list(backbone)
        split = split_layers(
            len(layers), self.num_stages, parallel.pp_first_stage_less_layers, parallel.pp_last_stage_less_layers
        )
        local = stage_indices(self.rank, self.size, self.per_rank, self.v_shape)
        self.plans = [StagePlan(s, split[s], s == 0, s == self.num_stages - 1) for s in local]
        owned = {i for plan in self.plans for i in plan.layers}
        first = any(p.first for p in self.plans)
        last = any(p.last for p in self.plans)

        # This rank's part of the model (every local stage's layers; the others are stand-ins): what the trainer
        # treats as "the model" (optimizer, checkpoints, export, train / eval mode). The stage modules made from it
        # in `build` share its layers.
        new_layers = nn.ModuleList([layer if i in owned else _AbsentLayer(layer) for i, layer in enumerate(layers)])
        new_backbone = _module_copy(
            backbone,
            {
                "layers": new_layers,
                "embed_tokens": backbone.embed_tokens if first else None,
                "norm": backbone.norm if last else _NoNorm(),
            },
        )
        head_name = "lm_head"
        self.model = _module_copy(
            model, {_backbone_name(model): new_backbone, head_name: getattr(model, head_name) if last else None}
        )
        self.model._pgs_pipeline = self
        self.stages: list = []
        self.stage_modules: list[nn.Module] = []
        self.schedule = None
        logger.info(
            f"Pipeline parallel: {self.num_stages} stages ({self.per_rank} per rank, {self.schedule_name}); this rank "
            + "; ".join(f"stage {p.index}: layers {p.layers[0]}-{p.layers[-1]}" for p in self.plans)
        )

    # -- construction

    def _stage_module(self, plan: StagePlan) -> nn.Module:
        from palingenesis.logits import backbone_of

        backbone = backbone_of(self.model)
        layers = _layer_list(backbone)
        stage_layers = nn.ModuleList(
            [layer if i in plan.layers else _AbsentLayer(layer) for i, layer in enumerate(layers)]
        )
        stage_backbone = _module_copy(
            backbone,
            {
                "layers": stage_layers,
                "embed_tokens": backbone.embed_tokens if plan.first else None,
                "norm": backbone.norm if plan.last else _NoNorm(),
            },
        )
        stage = _module_copy(
            self.model,
            {_backbone_name(self.model): stage_backbone, "lm_head": self.model.lm_head if plan.last else None},
        )
        stage.__dict__.pop("_pgs_pipeline", None)
        stage._pgs_stage = plan
        stage.forward = types.MethodType(_stage_forward(self), stage)
        return stage

    def build(self, device, shard) -> None:
        """Make the stage modules (`shard(module)` applies FSDP2 to each), their PipelineStages and the schedule."""
        from torch.distributed.pipelining import PipelineStage
        from torch.distributed.pipelining.schedules import PipelineScheduleSingle, get_schedule_class

        from palingenesis.parallel_dims import PP_SCHEDULES

        group = self.mesh.get_group()
        for plan in self.plans:
            module = shard(self._stage_module(plan))
            self.stage_modules.append(module)
            self.stages.append(PipelineStage(module, plan.index, self.num_stages, device, group=group))
        cls = get_schedule_class(PP_SCHEDULES[self.schedule_name][0])
        single = issubclass(cls, PipelineScheduleSingle)
        self._schedule_cls = cls
        self._single = single

    def _make_schedule(self, rows: int):
        n = self.microbatches or rows
        if rows % n:
            raise PipelineParallelError(
                f"{rows} rows per micro-batch do not split into parallel.pp_microbatches={n} equal pipeline "
                "micro-batches."
            )
        if n < self.num_stages:
            logger.warning(
                f"{n} pipeline micro-batches for {self.num_stages} stages: most of the pipeline idles (raise "
                "train.per_device_batch_size or parallel.pp_microbatches)."
            )
        stages = self.stages[0] if self._single else self.stages
        self.schedule = self._schedule_cls(stages, n_microbatches=n, loss_fn=_loss_passthrough, scale_grads=False)
        self._rows = rows

    # -- running

    def train(self, mode: bool = True) -> None:
        self.model.train(mode)
        for module in self.stage_modules:
            module.train(mode)

    def _inputs(self, input_ids, labels, position_ids, loss_weights):
        b = input_ids.shape[0]
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(b, -1)
        kwargs = {"position_ids": position_ids.contiguous(), "labels": labels.contiguous()}
        self.weighted = loss_weights is not None
        kwargs["loss_weights"] = (
            loss_weights.contiguous() if loss_weights is not None else torch.ones_like(labels, dtype=torch.float32)
        )
        return input_ids, kwargs

    def step(self, input_ids, labels, position_ids, loss_weights, loss_denom: float) -> float:
        """Forward and backward of one micro-batch of the trainer (all its pipeline micro-batches); returns this
        rank's share of the loss (0 on ranks without the last stage)."""
        if self.schedule is None or self._rows != input_ids.shape[0]:
            self._make_schedule(input_ids.shape[0])
        self.step_state.loss_denom = loss_denom
        self.step_state.evaluating = False
        args, kwargs = self._inputs(input_ids, labels, position_ids, loss_weights)
        losses: list = []
        target = torch.zeros(input_ids.shape[0], device=input_ids.device)
        self.schedule.step(args, target=target, losses=losses, return_outputs=False, **kwargs)
        return float(sum(loss.detach().float() for loss in losses)) if losses else 0.0

    @torch.no_grad()
    def evaluate_batch(self, input_ids, labels, position_ids=None) -> tuple[float, int]:
        """Summed cross-entropy and scored tokens of one evaluation batch (forward only; on ranks without the last
        stage (0, 0))."""
        self.step_state.evaluating = True
        self.step_state.eval_sums = []
        args, kwargs = self._inputs(input_ids, labels, position_ids, None)
        try:
            self._forward_only(args, kwargs)
        finally:
            self.step_state.evaluating = False
        total = sum(s for s, _ in self.step_state.eval_sums)
        count = sum(n for _, n in self.step_state.eval_sums)
        return float(total), int(count)

    def _forward_only(self, input_ids, kwargs) -> None:
        """Every stage's forward, micro-batch by micro-batch and stage by stage, each rank running its own stages
        and passing the hidden states on (point to point, a small header with their shape first). Evaluation
        only: no schedule (torch's schedules cannot run forward-only on several stages per rank), no shape
        inference, any row count."""
        import torch.distributed as dist

        rows = input_ids.shape[0]
        n = self.microbatches if self.microbatches and rows % self.microbatches == 0 else rows
        size = rows // n
        group = self.mesh.get_group()
        owner = {s: r for r in range(self.size) for s in _stage_indices(r, self)}
        modules = {plan.index: module for plan, module in zip(self.plans, self.stage_modules)}
        last = self.num_stages - 1
        for m in range(n):
            part = slice(m * size, (m + 1) * size)
            x = input_ids[part]
            mb_kwargs = {k: v[part] for k, v in kwargs.items()}
            for s in range(self.num_stages):
                if owner[s] != self.rank:
                    continue
                if s > 0 and owner[s - 1] != self.rank:
                    x = _receive(dist.get_global_rank(group, owner[s - 1]), group, input_ids.device)
                out = modules[s](x, **mb_kwargs)
                if s < last and owner[s + 1] != self.rank:
                    _send(out, dist.get_global_rank(group, owner[s + 1]), group)
                x = out

    def pad_rows(self, input_ids, labels, position_ids, loss_weights, observed, attention_mask):
        """A trainer micro-batch padded to the pipeline's fixed row length (labels ignored, no loss weight). Packed
        rows' padding is a document of its own; unpacked rows must be right-padded (causal attention keeps the
        padding inert, so no attention mask is needed)."""
        seq = input_ids.shape[1]
        if seq > self.seq_len:
            raise PipelineParallelError(f"a batch of {seq} tokens per row exceeds the pipeline's {self.seq_len}")
        if (
            position_ids is None
            and attention_mask is not None
            and bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any())
        ):
            raise PipelineParallelError("pipeline parallelism needs right-padded rows; this batch has left padding")
        extra = self.seq_len - seq
        if not extra:
            return input_ids, labels, position_ids, loss_weights, observed

        def pad(t, value):
            if t is None:
                return None
            return torch.cat([t, t.new_full((t.shape[0], extra, *t.shape[2:]), value)], dim=1)

        if position_ids is not None:
            tail = torch.arange(extra, device=position_ids.device, dtype=position_ids.dtype)
            position_ids = torch.cat([position_ids, tail.expand(position_ids.shape[0], -1)], dim=1)
        return pad(input_ids, 0), pad(labels, -100), position_ids, pad(loss_weights, 0), pad(observed, 0)

    @torch.no_grad()
    def evaluate(self, batches: list[dict], rows: int, device) -> tuple[float, int]:
        """Summed cross-entropy and scored tokens over collated evaluation batches, every rank's result the same:
        the rows regrouped into micro-batches of the training shape (`rows` x the pipeline's length), forward
        only, the last stage's sums shared over the pipeline (and tensor-parallel) group."""
        import torch.distributed as dist

        from palingenesis.loss import shift_labels

        ids, labels = [], []
        for batch in batches:
            shifted = shift_labels(batch["labels"])
            for r in range(batch["input_ids"].shape[0]):
                ids.append(batch["input_ids"][r])
                labels.append(shifted[r])
        total, count = 0.0, 0
        for start in range(0, len(ids), rows):
            group_ids = torch.zeros(rows, self.seq_len, dtype=torch.long)
            group_labels = torch.full((rows, self.seq_len), -100, dtype=torch.long)
            for i, (row_ids, row_labels) in enumerate(zip(ids[start : start + rows], labels[start : start + rows])):
                n = min(row_ids.shape[0], self.seq_len)
                group_ids[i, :n], group_labels[i, :n] = row_ids[:n], row_labels[:n]
            s, n = self.evaluate_batch(group_ids.to(device), group_labels.to(device))
            total, count = total + s, count + n
        sums = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
        dist.all_reduce(sums, group=self.mesh.get_group())  # only the last stage has them
        if self.tp is not None:  # each tensor-parallel rank scored its slice
            dist.all_reduce(sums, group=self.tp.group)
        return float(sums[0]), int(sums[1])

    # -- checkpoints

    def gather_full_state(self, folder) -> dict | None:
        """The whole model's state dict (every stage's parameters, gathered from their shards) on global rank 0,
        None elsewhere. Each stage's first rank writes its part to `folder` (next to the export), rank 0 reads them
        all (a shared file system, as the sharded checkpoints already need) and removes them."""
        from pathlib import Path

        import torch.distributed as dist
        from safetensors.torch import load_file, save_file
        from torch.distributed.tensor import DTensor

        from palingenesis.distributed import barrier

        state = {}
        for name, tensor in self.model.state_dict().items():
            full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            state[name] = full.detach().to("cpu", copy=True)
        folder = Path(folder)
        if self.leader:  # one writer per pipeline rank
            folder.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in state.items()}, str(folder / f"stage-{self.rank}.safetensors"))
        del state
        barrier()
        merged = None
        if dist.get_rank() == 0:
            merged = {}
            for p in range(self.size):
                merged.update(load_file(f"{folder}/stage-{p}.safetensors"))
            for p in folder.glob("stage-*.safetensors"):
                p.unlink()
            folder.rmdir()
        barrier()
        return merged


_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


def _stage_indices(rank: int, pipe: PipelineParallel) -> list[int]:
    from palingenesis.parallel_dims import stage_indices

    return stage_indices(rank, pipe.size, pipe.per_rank, pipe.v_shape)


def _send(t: torch.Tensor, dst: int, group) -> None:
    import torch.distributed as dist

    t = t.contiguous()
    header = torch.zeros(8, dtype=torch.long, device=t.device)
    header[0], header[1] = t.dim(), _DTYPES.index(t.dtype)
    header[2 : 2 + t.dim()] = torch.tensor(t.shape)
    dist.send(header, dst, group=group)
    dist.send(t, dst, group=group)


def _receive(src: int, group, device) -> torch.Tensor:
    import torch.distributed as dist

    header = torch.zeros(8, dtype=torch.long, device=device)
    dist.recv(header, src, group=group)
    ndim, dtype = int(header[0]), _DTYPES[int(header[1])]
    t = torch.empty(tuple(header[2 : 2 + ndim].tolist()), dtype=dtype, device=device)
    dist.recv(t, src, group=group)
    return t


def _backbone_name(model: nn.Module) -> str:
    for name in ("model", "transformer"):
        if isinstance(getattr(model, name, None), nn.Module):
            return name
    raise PipelineParallelError("cannot find the model's backbone (model.model)")


def _tied(model: nn.Module) -> bool:
    try:
        head = model.get_output_embeddings()
        embed = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return False
    return head is not None and embed is not None and head.weight is embed.weight


def _loss_passthrough(output, target):
    """The last stage returns its loss already: the schedule backpropagates it as is."""
    return output


def _stage_forward(pipe: PipelineParallel):
    def forward(self, x, position_ids, labels, loss_weights):
        from torch.distributed.tensor import DTensor, Shard

        from palingenesis.logits import _last_hidden, backbone_of, output_head
        from palingenesis.packing import packed_forward_kwargs

        plan: StagePlan = self._pgs_stage
        tp = pipe.tp
        packed = bool((position_ids[:, 1:] < position_ids[:, :-1]).any()) or pipe.flatten_packed
        flatten = pipe.flatten_packed
        kwargs = packed_forward_kwargs(position_ids, flatten) if packed else {"position_ids": position_ids}
        kwargs.setdefault("attention_mask", None)
        backbone = backbone_of(self)
        if plan.first:
            ids = x.reshape(1, -1) if flatten else x
            out = backbone(input_ids=ids, use_cache=False, **kwargs)
        else:
            hidden = x
            if tp is not None:  # this rank's sequence slice -> the sequence-split residual stream
                b, local, h = hidden.shape
                seq = local * tp.size
                hidden = DTensor.from_local(
                    hidden, tp.mesh, [Shard(1)], run_check=False, shape=torch.Size((b, seq, h)), stride=(seq * h, h, 1)
                )
            out = backbone(inputs_embeds=hidden, use_cache=False, **kwargs)
        hidden = _last_hidden(out)
        if not plan.last:
            return hidden.to_local() if isinstance(hidden, DTensor) else hidden
        # the last stage: final norm applied by the backbone; the loss over this rank's tokens
        if flatten:
            labels, loss_weights = labels.reshape(1, -1), loss_weights.reshape(1, -1)
        if tp is not None:
            labels, loss_weights = tp.shard(labels, -100), tp.shard(loss_weights)
        if not torch.is_grad_enabled() and not pipe.step_state.evaluating:
            # PipelineStage's shape inference: a no-grad forward on zero inputs before the first training
            # micro-batch; the output only has to have the loss's shape
            return hidden.new_zeros((), dtype=torch.float32)
        return _stage_loss(pipe, hidden, labels, loss_weights, output_head(self))

    return forward


def _stage_loss(pipe: PipelineParallel, hidden, labels, weights, head):
    from palingenesis.loss import chunked_cross_entropy_loss, cross_entropy_loss, cut_cross_entropy_loss

    state = pipe.step_state
    weights = weights if pipe.weighted else None
    if state.evaluating:
        valid = labels != -100
        rows = hidden[valid]
        total = torch.zeros((), dtype=torch.float64, device=hidden.device)
        if rows.shape[0]:
            logits = head(rows).float()
            total += torch.nn.functional.cross_entropy(logits, labels[valid], reduction="sum").double()
        state.eval_sums.append((float(total), int(valid.sum())))
        return total.float()
    if pipe.loss_kind == "cce":
        return cut_cross_entropy_loss(hidden, labels, head, global_valid_tokens=state.loss_denom, weights=weights)
    if pipe.loss_kind == "chunked":
        chunks = pipe.num_chunks(labels.numel()) if pipe.num_chunks is not None else 8
        return chunked_cross_entropy_loss(
            hidden,
            labels,
            head,
            num_chunks=chunks,
            global_valid_tokens=state.loss_denom,
            weights=weights,
            stable=pipe.stable,
        )
    logits = head(hidden)
    return cross_entropy_loss(logits, labels, state.loss_denom, weights=weights, stable=pipe.stable)
