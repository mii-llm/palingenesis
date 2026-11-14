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

Memory: LoRA removes the base weights' gradients and optimizer state. It does not shrink activations
(each adapted layer still saves its input for the backward): long sequences still need activation
checkpointing, memory.seco or `pgs fit`.
"""

from __future__ import annotations

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


# ─────────────────────────────────────────────────────────────── the adapted layer


def fold_weight(
    weight: Tensor, lora_A: Tensor, lora_B: Tensor, scaling: float, magnitude: Tensor | None = None
) -> Tensor:
    """The adapted weight in float32: W + scaling B A, with DoRA's rows rescaled to the magnitude m."""
    w = weight.float() + scaling * (lora_B.float() @ lora_A.float())
    if magnitude is not None:
        w = w * (magnitude.float() / torch.linalg.vector_norm(w, dim=1)).unsqueeze(1)
    return w


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

    def delta(self, x: Tensor) -> Tensor:
        """scaling * dropout(x) A^T B^T, computed against the float32 adapters (under autocast: in its dtype)."""
        h = self.lora_dropout(x).to(self.lora_A.dtype)
        return F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scaling

    def dora_scale(self) -> Tensor:
        """m / ||W + scaling B A||_row: the norm in float32 outside autocast, detached (DoRA, section 4.3)."""
        with torch.no_grad(), torch.autocast(self.weight.device.type, enabled=False):
            w = self.weight.float() + self.scaling * (self.lora_B.float() @ self.lora_A.float())
            norm = torch.linalg.vector_norm(w, dim=1)
        return self.lora_magnitude / norm

    def forward(self, x: Tensor) -> Tensor:
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
            "are the same as full fine-tuning: long sequences still need activation checkpointing, memory.seco "
            "or `pgs fit`."
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

    # -- export

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict:
        out = {k: v for k, v in state.items() if not any(k.endswith(f".{a}") for a in ADAPTER_KEYS)}
        adapters: dict[str, dict[str, Tensor]] = {}
        meter = SurvivalMeter()
        for name, m in self.modules.items():
            key = f"{name}.weight"
            if key not in state:
                continue
            parts = {a: state.get(f"{name}.{a}") for a in ADAPTER_KEYS}
            if parts["lora_A"] is None or parts["lora_B"] is None or (self.dora and parts["lora_magnitude"] is None):
                raise PluginError(f"lora: the exported state has {key} but not its adapter")
            w = state[key]
            folded = fold_weight(w, parts["lora_A"], parts["lora_B"], m.scaling, parts["lora_magnitude"])
            cast = folded.to(w.dtype)
            meter.update(w, folded, cast)
            out[key] = cast
            adapters[name] = {a: t for a, t in parts.items() if t is not None}
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
            "base_model_name_or_path": getattr(model_cfg, "name_or_path", None),
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
