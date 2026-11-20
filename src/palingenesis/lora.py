"""Low-rank adaptation (LoRA) as a trainer plugin: the base weights stay frozen, a low-rank delta trains.

    plugins:
      custom:
        - class: palingenesis.lora:LoRA       # or the entry point name: lora
          args:
            rank: 16
            alpha: 32                         # scaling alpha / rank (rslora: alpha / sqrt(rank))
            rslora: false
            dropout: 0.0                      # on the adapter's input only
            dora: false                       # weight-decomposed LoRA (magnitude per output row)
            lr_ratio: 1.0                     # LoRA+: B learns at lr_ratio x train.learning_rate
            targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
                      in_proj_qkv, in_proj_z, in_proj_qkvz, out_proj]
            exclude: []
            save_adapter: true                # also <export>/adapter/ in the PEFT layout

Each targeted nn.Linear (W [out, in], frozen, in its storage dtype) becomes

    y = x W^T + b + scaling * dropout(x) A^T B^T          A [rank, in] Kaiming-uniform, B [out, rank] zero
    DoRA: y = m / ||W + scaling B A||_row * (x W^T + scaling * dropout(x) A^T B^T) + b,   m = ||W||_row at start

(LoRA arXiv:2106.09685, rsLoRA arXiv:2312.03732, DoRA arXiv:2402.09353 with its detached norm, LoRA+
arXiv:2402.12354). B = 0 makes the adapted model equal to the base at the start. A, B and m are float32
Parameters on the wrapper next to the base `weight` / `bias` (the same Parameters under the same names), so
resume checkpoints carry them like any weight; A is seeded from (train.seed, module name), the same on every
rank. Every base parameter is frozen, so the optimizer holds state for the adapters only.

Export (final/, best/model/): every adapter is folded into its base weight in float32, W' = W + scaling B A
(DoRA: rows rescaled to m), and rounded to nearest in the weight's dtype: a plain model of the original
architecture. Exact for a float32 base. For a bf16 base, the elements of the delta below half an ulp of
their weight are lost (training added the delta in full); the share of the delta the saved weights keep is
logged (precision.edit_survival's projection). Stochastic rounding would keep the delta in expectation, but
its noise exceeds a sub-ulp delta in any single forward, so it reproduces the adapted model worse. To serve
exactly what was trained, load the adapter (`save_adapter`: <export>/adapter/, PEFT layout) on the base.

The RL trainer (`pgs rl`) uses the same plugin through its `lora:` section (palingenesis.rl.adapters): the
rollout engine receives the folded weights (`fold_weight`) after every step, a KL reference at the starting
policy is the model with its adapters switched off (`disabled`), and a checkpoint is resumed from its adapter
(`load_adapter`).

Memory: LoRA removes the base weights' gradients and optimizer state. It does not shrink activations
(each adapted layer still saves its input for the backward): long sequences still need activation
checkpointing, memory.seco or `pgs fit`.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from palingenesis.hooks import PluginContext, PluginError, TrainerPlugin
from palingenesis.precision import SurvivalMeter, tensor_seed

logger = logging.getLogger(__name__)

# Attention and MLP projections of Llama / Qwen / Mistral-style models, and the Gated DeltaNet projections of
# Qwen3.5 (in_proj_qkv, in_proj_z) and Qwen3-Next (in_proj_qkvz). The per-head gate projections (in_proj_a,
# in_proj_b, in_proj_ba) have a few rows each: a rank-16 adapter on them adds nothing.
DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_qkvz",
    "out_proj",
)
ADAPTER_KEYS = ("lora_A", "lora_B", "lora_magnitude")
# DoRA divides by each row's norm: an all-zero row of the base (unused outputs exist in released weights)
# has magnitude 0 and norm 0, and 0 / 0 made the whole forward NaN.
ROW_NORM_FLOOR = 1e-12


# ─────────────────────────────────────────────────────────────── the adapted layer


def fold_weight(
    weight: Tensor, lora_A: Tensor, lora_B: Tensor, scaling: float, magnitude: Tensor | None = None
) -> Tensor:
    """The adapted weight in float32: W + scaling B A, with DoRA's rows rescaled to the magnitude m."""
    w = weight.float() + scaling * (lora_B.float() @ lora_A.float())
    if magnitude is not None:
        w = w * (magnitude.float() / torch.linalg.vector_norm(w, dim=1).clamp_min(ROW_NORM_FLOOR)).unsqueeze(1)
    return w


# The named adapter a forward uses: "" is the layer's own lora_A / lora_B. Two levels. use_adapter sets a context
# variable: threads and asyncio tasks each see their own value, so a rollout thread generating with one role's
# adapter and a training forward of another's cannot change each other's. set_adapter sets the process's adapter,
# which every thread without a use_adapter block computes with: the trainer selects a micro-batch's adapter with
# it, because autograd recomputes checkpointed segments during backward on its own threads (one per CUDA device),
# which do not see the context variables of the thread that called backward.
_ACTIVE: contextvars.ContextVar[str | None] = contextvars.ContextVar("palingenesis_lora_adapter", default=None)
_PROCESS = [""]
BASE = "base"  # use_adapter(BASE): no adapter at all, the frozen base (a reserved adapter name)


def active_adapter() -> str:
    """The adapter a forward on this thread computes with."""
    name = _ACTIVE.get()
    return _PROCESS[0] if name is None else name


def set_adapter(name: str) -> None:
    """Select the adapter of every forward that is not inside a use_adapter block, on any thread (a training
    forward, and its recomputation during backward under gradient checkpointing)."""
    _PROCESS[0] = name or ""


@contextlib.contextmanager
def use_adapter(name: str):
    """Forwards inside the block (in this thread or task) compute with the adapter called `name` in every
    adapted layer that has one (LoRALinear.add_adapter); "" is the default adapter."""
    token = _ACTIVE.set(name or "")
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class Factors(nn.Module):
    """One named adapter of a layer: its own A [rank, in] and B [out, rank]."""

    def __init__(self, a: Tensor, b: Tensor):
        super().__init__()
        self.lora_A, self.lora_B = nn.Parameter(a), nn.Parameter(b)


class LoRALinear(nn.Module):
    """nn.Linear with a frozen weight and a trainable low-rank delta (optionally DoRA)."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        rslora: bool = False,
        dropout: float = 0.0,
        dora: bool = False,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight  # the same Parameter: names and checkpoints carry over
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / (math.sqrt(rank) if rslora else rank)
        device = self.weight.device
        a = torch.empty(rank, self.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)  # nn.Linear's (and PEFT's) init
        self.lora_A = nn.Parameter(a.to(device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=device))
        if dora:
            norm = torch.linalg.vector_norm(self.weight.detach().float(), dim=1)
            self.lora_magnitude = nn.Parameter(norm)
        else:
            self.register_parameter("lora_magnitude", None)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True  # False: the frozen base alone (LoRA.disabled: a reference policy without a copy)
        self.roles: nn.ModuleDict | None = None  # further adapters by name (add_adapter), chosen by use_adapter

    def add_adapter(self, name: str, generator: torch.Generator | None = None) -> Factors:
        """A second (third, ...) adapter on this layer, the same rank and scaling, B zero: several policies on
        one frozen base. Not with DoRA (a magnitude per adapter is not implemented)."""
        if self.lora_magnitude is not None:
            raise ValueError("named adapters are not supported with DoRA")
        if self.roles is None:
            self.roles = nn.ModuleDict()
        a = torch.empty(self.rank, self.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        device = self.weight.device
        factors = Factors(a.to(device), torch.zeros(self.out_features, self.rank, dtype=torch.float32, device=device))
        self.roles[name] = factors
        return factors

    def factors(self, name: str | None = None) -> tuple[Tensor, Tensor]:
        """(A, B) of the adapter `name` (default: the one use_adapter selected; "" or a name this layer does not
        have: the layer's own)."""
        name = active_adapter() if name is None else name
        if name and self.roles is not None and name in self.roles:
            return self.roles[name].lora_A, self.roles[name].lora_B
        return self.lora_A, self.lora_B

    def delta(self, x: Tensor) -> Tensor:
        """scaling * dropout(x) A^T B^T, computed against the float32 adapters (under autocast: in its dtype)."""
        h = self.lora_dropout(x)
        if not torch.is_autocast_enabled(x.device.type):
            # under autocast the matmuls cast both sides themselves: a float32 copy of every adapted
            # layer's input would only cost memory and time
            h = h.to(self.lora_A.dtype)
        a, b = self.factors()
        return F.linear(F.linear(h, a), b) * self.scaling

    def dora_scale(self) -> Tensor:
        """m / ||W + scaling B A||_row: the norm in float32 outside autocast, detached (DoRA, section 4.3)."""
        with torch.no_grad(), torch.autocast(self.weight.device.type, enabled=False):
            w = self.weight.float() + self.scaling * (self.lora_B.float() @ self.lora_A.float())
            norm = torch.linalg.vector_norm(w, dim=1).clamp_min(ROW_NORM_FLOOR)
        return self.lora_magnitude / norm

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled or active_adapter() == BASE:
            return F.linear(x, self.weight, self.bias)
        if self.lora_magnitude is None:
            out = F.linear(x, self.weight, self.bias)
            return out + self.delta(x).to(out.dtype)
        out = F.linear(x, self.weight)
        y = ((out + self.delta(x).to(out.dtype)) * self.dora_scale()).to(out.dtype)
        return y if self.bias is None else y + self.bias

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        """The adapted weight, in float32."""
        return fold_weight(self.weight, self.lora_A, self.lora_B, self.scaling, self.lora_magnitude)

    def extra_repr(self) -> str:
        dora = ", dora" if self.lora_magnitude is not None else ""
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, scaling={self.scaling:g}{dora}"


# ─────────────────────────────────────────────────────────────── helpers


def _local(t: Tensor) -> Tensor:
    """The local shard of an FSDP2 (DTensor) parameter, else the tensor."""
    return t.to_local() if hasattr(t, "to_local") else t


def _full(t: Tensor) -> Tensor:
    """The whole tensor of an FSDP2 parameter (a collective: every rank must call it), else the tensor."""
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e8 else f"{n_bytes / 1e6:.1f} MB"


# ─────────────────────────────────────────────────────────────── the plugin


class LoRA(TrainerPlugin):
    """Wrap the targeted linears with low-rank adapters, train only those, fold them in on export."""

    name = "lora"

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        rslora: bool = False,
        dropout: float = 0.0,
        dora: bool = False,
        lr_ratio: float = 1.0,
        targets: list[str] | None = None,
        exclude: list[str] | None = None,
        save_adapter: bool = True,
        metrics_every: int = 50,
    ):
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"lora: rank must be a positive integer (got {rank!r})")
        if not alpha > 0:
            raise ValueError(f"lora: alpha must be > 0 (got {alpha!r})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"lora: dropout must be in [0, 1) (got {dropout!r})")
        if not lr_ratio > 0:
            raise ValueError(f"lora: lr_ratio must be > 0 (got {lr_ratio!r}; 1 = plain LoRA)")
        targets = list(DEFAULT_TARGETS if targets is None else targets)
        if not targets or not all(isinstance(t, str) and t for t in targets):
            raise ValueError(f"lora: targets must be a non-empty list of module name patterns (got {targets!r})")
        try:
            # a pattern matches a whole trailing part of the module name: "q_proj", "self_attn\.q_proj",
            # "layers\.(1|2)\.mlp\..*"
            self.targets = [re.compile(rf"(^|\.)(?:{t})$") for t in targets]
            self.exclude = [re.compile(p) for p in (exclude or [])]
        except re.error as e:
            raise ValueError(f"lora: invalid pattern in targets / exclude: {e}") from None
        self.rank, self.alpha, self.rslora, self.dropout, self.dora = rank, float(alpha), rslora, dropout, dora
        self.lr_ratio = float(lr_ratio)
        self.save_adapter = save_adapter
        self.metrics_every = max(1, metrics_every)
        self.modules: dict[str, LoRALinear] = {}
        self._base_ids: set[int] = set()
        self._base_sq = 0.0  # sum of ||W||_F^2 over the adapted weights (for the relative delta)
        self._magnitude0: dict[str, Tensor] = {}
        self._last: dict[str, float] = {}

    @property
    def scaling(self) -> float:
        return self.alpha / (math.sqrt(self.rank) if self.rslora else self.rank)

    # -- config

    def setup(self, ctx: PluginContext) -> None:
        train = getattr(ctx.config, "train", None)
        if train is None:
            return
        refused = {
            "train.optimizer: muon": train.optimizer == "muon",
            "train.hyperball": train.hyperball,
            "train.mona": train.mona,
            "train.base_merge": train.base_merge,
            "memory.float8_training": ctx.config.memory.float8_training,
        }
        on = [k for k, v in refused.items() if v]
        if on:
            raise PluginError(
                f"lora does not support {', '.join(on)}: these act on the full weight matrices (orthogonalised "
                "or norm-constrained updates, merges toward the base, float8 nn.Linear swaps) and are not defined "
                "for a frozen base with low-rank factors. Turn them off with the LoRA plugin."
            )
        if self.lr_ratio != 1.0 and train.llrd_decay < 1.0:
            raise PluginError(
                "lora: lr_ratio (LoRA+) sets one learning rate for every B matrix, which would override "
                "train.llrd_decay's per-layer rates: use one of them"
            )

    # -- model

    def _is_target(self, name: str) -> bool:
        return any(p.search(name) for p in self.targets) and not any(p.search(name) for p in self.exclude)

    def transform_model(self, model: nn.Module, ctx: PluginContext):
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        seed = int(getattr(getattr(ctx.config, "train", None), "seed", 0) or 0)
        chosen, linears = [], []
        for name, module in model.named_modules():
            if type(module) is not nn.Linear:
                continue
            linears.append(name)
            # The output head and a tied embedding are never adapted: the chunked and cut cross-entropy
            # losses read lm_head.weight directly, so an adapter there would be skipped by the loss.
            if module is head or module is embed:
                continue
            if self._is_target(name):
                chosen.append((name, module))
        if not chosen:
            leaves = sorted({n.rpartition(".")[2] for n in linears})
            raise ValueError(
                f"lora: no nn.Linear matched targets / exclude; the model's linear layers are named {leaves}"
            )
        model.requires_grad_(False)
        self._base_ids = {id(p) for p in model.parameters()}
        small = []
        for name, linear in chosen:
            if self.rank >= min(linear.in_features, linear.out_features):
                small.append(name)
            g = torch.Generator().manual_seed(tensor_seed(seed, name))
            new = LoRALinear(linear, self.rank, self.alpha, self.rslora, self.dropout, self.dora, generator=g)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, new)
            self.modules[name] = new
        if small:
            logger.warning(
                f"lora: rank {self.rank} is not below the smaller dimension of {len(small)} module(s) "
                f"({small[:3]}{' ...' if len(small) > 3 else ''}): their adapter can express any update, at "
                "more parameters than the weight itself"
            )
        leaves = sorted({n.rpartition(".")[2] for n in self.modules})
        logger.info(
            f"lora: {len(self.modules)} linear modules adapted ({', '.join(leaves)}), rank {self.rank}, "
            f"alpha {self.alpha:g}, scaling {self.scaling:g}{', rsLoRA' if self.rslora else ''}"
            f"{', DoRA' if self.dora else ''}"
        )
        return model

    def param_groups(self, model: nn.Module, ctx: PluginContext) -> list[dict]:
        """Refuse a base weight made trainable after transform_model; LoRA+: B at lr_ratio x the rate."""
        thawed = [n for n, p in model.named_parameters() if id(p) in self._base_ids and p.requires_grad]
        if thawed:
            raise PluginError(
                f"lora: {len(thawed)} base parameter(s) were made trainable after the adapters were added "
                f"({thawed[:3]}): something in the run unfreezes weights, which LoRA keeps frozen"
            )
        if self.lr_ratio == 1.0:
            return []
        b = [m.lora_B for m in self.modules.values() if m.lora_B.requires_grad]
        if not b:
            return []
        return [{"params": b, "lr": ctx.config.train.learning_rate * self.lr_ratio}]

    # -- training

    def _counts(self, model: nn.Module) -> tuple[int, int, int]:
        """(trainable, adapter, total) parameter counts (global numel under FSDP)."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adapter = sum(
            p.numel() for m in self.modules.values() for p in (m.lora_A, m.lora_B, m.lora_magnitude) if p is not None
        )
        total = sum(p.numel() for p in model.parameters())
        return trainable, adapter, total

    @torch.no_grad()
    def on_train_start(self, model: nn.Module, ctx: PluginContext) -> None:
        sq = torch.stack([_local(m.weight).float().pow(2).sum() for m in self.modules.values()]).sum()
        if ctx.is_fsdp and dist.is_available() and dist.is_initialized():
            if ctx.device is not None:  # CPU-offloaded shards: the collective runs on the rank's device
                sq = sq.to(ctx.device)
            dist.all_reduce(sq)
        self._base_sq = float(sq)
        if self.dora:
            # the starting magnitude is the frozen base's row norm: the same after a resume
            self._magnitude0 = {
                n: torch.linalg.vector_norm(_local(m.weight).float(), dim=1) for n, m in self.modules.items()
            }
        trainable, adapter, total = self._counts(model)
        frozen_adapters = adapter - sum(
            p.numel()
            for m in self.modules.values()
            for p in (m.lora_A, m.lora_B, m.lora_magnitude)
            if p is not None and p.requires_grad
        )
        self._last = {"trainable_params": float(trainable), "trainable_share": trainable / max(total, 1)}
        if ctx.rank != 0:
            return
        base = [p for p in model.parameters() if id(p) in self._base_ids]
        base_bytes = sum(p.numel() * p.element_size() for p in base)
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in base})
        # AdamW (or any two-moment optimizer) on the trainable adapters: weight + gradient + two moments, fp32
        adapter_bytes = trainable * 4
        full_ft = sum(p.numel() * p.element_size() * 3 for p in base)  # gradient + two moments in the weights' dtype
        logger.info(
            f"lora: {trainable:,} trainable parameters of {total:,} ({100 * trainable / max(total, 1):.2f}%)"
            + (
                f"; {frozen_adapters:,} adapter parameters frozen by train.freeze_non_attention"
                if frozen_adapters
                else ""
            )
        )
        logger.info(
            f"lora memory: frozen base {_gb(base_bytes)} ({'/'.join(dtypes)}, no gradient, no optimizer state); "
            f"adapters {_gb(adapter_bytes)} + gradients {_gb(adapter_bytes)} + AdamW moments {_gb(2 * adapter_bytes)} "
            f"(float32). Full fine-tuning would add ~{_gb(full_ft)} of gradients and AdamW moments. Activations "
            "are the same as full fine-tuning: long sequences still need activation checkpointing"
            + (", memory.seco or `pgs fit`." if hasattr(ctx.config, "memory") else " (model.gradient_checkpointing).")
        )

    @torch.no_grad()
    def metrics(self, model: nn.Module, ctx: PluginContext) -> dict[str, float]:
        if ctx.step % self.metrics_every and ctx.step != ctx.total_steps and "delta_rel" in self._last:
            return dict(self._last)
        # ||B A||_F^2 = tr((A A^T)(B^T B)): rank x rank products, never the full delta
        delta_sq = torch.zeros((), dtype=torch.float64)
        for m in self.modules.values():
            a, b = _full(m.lora_A).double(), _full(m.lora_B).double()
            delta_sq += (m.scaling**2) * torch.trace((a @ a.T) @ (b.T @ b)).cpu()
        out = dict(self._last)
        out["delta_rel"] = math.sqrt(float(delta_sq) / self._base_sq) if self._base_sq > 0 else 0.0
        if self.dora and self._magnitude0:
            change = total = 0.0
            for n, m in self.modules.items():
                m0 = self._magnitude0[n].to(_local(m.lora_magnitude).device)
                change += float((_local(m.lora_magnitude).float() - m0).abs().sum())
                total += float(m0.abs().sum())
            sums = torch.tensor([change, total], dtype=torch.float64)
            if ctx.is_fsdp and dist.is_available() and dist.is_initialized():
                sums = sums.to(ctx.device)
                dist.all_reduce(sums)
                sums = sums.cpu()
            out["magnitude_change"] = float(sums[0] / sums[1].clamp(min=1e-12))
        self._last = out
        return dict(out)

    # -- the base alone, and adapters from a file

    @contextlib.contextmanager
    def disabled(self):
        """The model computes its frozen base inside the block: every adapter is skipped (not removed)."""
        before = [m.enabled for m in self.modules.values()]
        for m in self.modules.values():
            m.enabled = False
        try:
            yield
        finally:
            for m, enabled in zip(self.modules.values(), before):
                m.enabled = enabled

    def add_adapter(self, name: str, seed: int = 0) -> None:
        """A further adapter called `name` on every adapted layer (after transform_model), A seeded from
        (seed, module name, adapter name): the same on every rank and in a resumed run."""
        if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"lora: adapter names are letters, digits and underscores (got {name!r})")
        for module_name, m in self.modules.items():
            g = torch.Generator().manual_seed(tensor_seed(seed, f"{module_name}#{name}"))
            m.add_adapter(name, g)
        self.names = [*getattr(self, "names", []), name]

    def adapter_tensors(self, adapter: str = "") -> dict[str, dict[str, Tensor]]:
        """{module name: {"lora_A": ..., "lora_B": ...[, "lora_magnitude": ...]}} of one adapter ("": the default)."""
        out = {}
        for name, m in self.modules.items():
            a, b = m.factors(adapter)
            out[name] = {"lora_A": a, "lora_B": b}
            if not adapter and m.lora_magnitude is not None:
                out[name]["lora_magnitude"] = m.lora_magnitude
        return out

    def save_adapter(self, path: str | Path, ctx: PluginContext, adapter: str = "") -> None:
        """Write one adapter to `path` in the PEFT layout (float32), as export does for the default adapter."""
        self._write_peft_adapter(Path(path), self.adapter_tensors(adapter), ctx)

    @torch.no_grad()
    def load_adapter(self, path: str | Path, adapter: str = "") -> None:
        """Copy the adapters saved at `path` (a directory in the PEFT layout, as export writes it) into the model's,
        after transform_model and before the model is sharded. The file must hold exactly this run's adapters:
        the same modules, shapes and DoRA setting (a file from other lora settings is refused, not partly loaded).
        `adapter`: the named adapter to load into (add_adapter) instead of the default one."""
        from safetensors.torch import load_file

        file = Path(path) / "adapter_model.safetensors"
        if not file.exists():
            raise PluginError(f"lora: no adapter at {file}")
        saved = load_file(str(file))
        wanted: dict[str, nn.Parameter] = {}
        for name, parts in self.adapter_tensors(adapter).items():
            prefix = f"base_model.model.{name}"
            wanted[f"{prefix}.lora_A.weight"] = parts["lora_A"]
            wanted[f"{prefix}.lora_B.weight"] = parts["lora_B"]
            if "lora_magnitude" in parts:
                wanted[f"{prefix}.lora_magnitude_vector"] = parts["lora_magnitude"]
        missing, extra = sorted(wanted.keys() - saved.keys()), sorted(saved.keys() - wanted.keys())
        if missing or extra:
            raise PluginError(
                f"lora: the adapter at {path} does not match this run's (missing {missing[:3]}, unexpected "
                f"{extra[:3]}): it was saved with other targets, exclude or dora settings"
            )
        for key, param in wanted.items():
            if saved[key].shape != param.shape:
                raise PluginError(
                    f"lora: {key} at {path} has shape {tuple(saved[key].shape)}, this run's is {tuple(param.shape)} "
                    "(another rank?)"
                )
            param.copy_(saved[key].to(device=param.device, dtype=param.dtype))
        logger.info(f"lora: {len(self.modules)} adapters loaded from {path}")

    # -- export

    def fold_state(
        self, state: dict[str, Tensor], meter: SurvivalMeter | None = None
    ) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        """(`state` with every adapter folded into its base weight and the adapter tensors removed, the adapters
        per module). `state` is keyed like the model's state dict (wrapper prefixes removed) and is not modified:
        a folded weight is a new tensor in its base weight's dtype (computed in float32, rounded to nearest), and
        the other tensors pass through. `meter` accumulates how much of the delta that rounding keeps."""
        out = {k: v for k, v in state.items() if not any(k.endswith(f".{a}") for a in ADAPTER_KEYS)}
        adapters: dict[str, dict[str, Tensor]] = {}
        for name, m in self.modules.items():
            key = f"{name}.weight"
            if key not in state:
                continue
            parts = {a: state.get(f"{name}.{a}") for a in ADAPTER_KEYS}
            if parts["lora_A"] is None or parts["lora_B"] is None or (self.dora and parts["lora_magnitude"] is None):
                raise PluginError(f"lora: the state has {key} but not its adapter")
            w = state[key]
            folded = fold_weight(w, parts["lora_A"], parts["lora_B"], m.scaling, parts["lora_magnitude"])
            cast = folded.to(w.dtype)
            if meter is not None:
                meter.update(w, folded, cast)
            out[key] = cast
            adapters[name] = {a: t for a, t in parts.items() if t is not None}
        return out, adapters

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict:
        meter = SurvivalMeter()
        out, adapters = self.fold_state(state, meter)
        if adapters:
            s = meter.result()
            logger.info(
                f"lora: {len(adapters)} adapters folded into the base weights: the saved weights keep "
                f"{s['projection']:.3f} of the delta (projection; 1 = all of it), "
                f"{100 * s['changed_frac']:.1f}% of the elements changed"
            )
            if s["projection"] < 0.9:
                logger.warning(
                    "lora: the base weights' dtype cannot hold much of the adapters' delta (most of it is below "
                    "half an ulp of its weight): the exported model differs from the trained one. Serve the base "
                    "with the adapter (save_adapter), or train with model.torch_dtype: float32"
                )
        if self.save_adapter and adapters and ctx.export_dir:
            self._write_peft_adapter(Path(ctx.export_dir) / "adapter", adapters, ctx)
        return out

    def _write_peft_adapter(self, path: Path, adapters: dict[str, dict[str, Tensor]], ctx: PluginContext) -> None:
        """adapter_model.safetensors + adapter_config.json, loadable with peft.PeftModel.from_pretrained on the
        causal-LM model the run trained (module names as the trainer loads it)."""
        from safetensors.torch import save_file

        path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, parts in adapters.items():
            prefix = f"base_model.model.{name}"
            tensors[f"{prefix}.lora_A.weight"] = parts["lora_A"]
            tensors[f"{prefix}.lora_B.weight"] = parts["lora_B"]
            if "lora_magnitude" in parts:
                tensors[f"{prefix}.lora_magnitude_vector"] = parts["lora_magnitude"]
        tensors = {k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}
        save_file(tensors, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
        model_cfg = getattr(ctx.config, "model", None)
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            # the SFT config names the base model.name_or_path, the RL config model.policy
            "base_model_name_or_path": getattr(model_cfg, "name_or_path", None) or getattr(model_cfg, "policy", None),
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "use_rslora": self.rslora,
            "use_dora": self.dora,
            "target_modules": sorted(adapters),
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": True,
            "inference_mode": True,
            "modules_to_save": None,
        }
        (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        logger.info(f"lora: adapter saved (PEFT layout) -> {path}")
