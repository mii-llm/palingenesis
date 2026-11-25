"""FP4 / FP8 quantisation-aware training as a trainer plugin: NVFP4, MXFP4, FP8 per row and FP8 128 x 128 blocks.

    plugins:
      custom:
        - class: palingenesis.qat:FloatQAT
          args:
            format: nvfp4                 # nvfp4 | mxfp4 | fp8 | fp8_block
            activations: none             # none (weight-only) | fp8 | nvfp4 | mxfp4 (dynamic, per token)
            scales: mse                   # max | mse (MSE-optimal clipping, searched at the start) | learned
            strength: {start: 0.0, end: 0.1}   # ramp the quantisation in over the first 10% of training
            freeze_scales_at: 0.9         # from 90% of training, the scales are fixed: the weights settle on them

Every substituted nn.Linear keeps its full-precision **latent** weight (the same Parameter under the same name,
so checkpoints, resume, FSDP and tied weights work unchanged). The forward computes with

    W_q = dequantize(quantize(W))                                    (palingenesis.qat.formats)
    W_eff = W + strength * (W_q - W).detach()                        straight-through estimator

so the gradient reaches W as if W_eff were W (scales: learned uses LSQ gradients instead, below). Activations
(`activations:`) are quantised the same way per token before the matmul, with dynamic scales.

Scales
  max      each block's scale from its largest magnitude (round-to-nearest PTQ, as llm-compressor / torchao).
  mse      a clipping factor per output row (per 128 rows for fp8_block), searched before the first step over
           `clip_steps` values from 1 down to `clip_min`, minimising the row's squared quantisation error; fixed
           afterwards (the scales still follow the weights).
  learned  the same factors, initialised by the search, trained with LSQ gradients (Esser et al. 2020,
           arXiv:1902.08153): weights clipped at the format's max get no gradient, the factor gets the scale's.
           They are float32 parameters (`<module>.qat_clip`) in an optimizer group of their own (`clip_lr`).

`freeze_scales_at` fixes every block scale (and NVFP4's per-tensor scale) at that fraction of training, from the
weights at that step, with no moving average: the last steps adapt the weights to the grid they are exported on.
NVFP4's per-tensor scale is shared by the layers vLLM fuses into one (`fused_groups`: q/k/v, gate/up, the Gated
DeltaNet input projections), as vLLM keeps one per fused layer. With NVFP4 activations the per-tensor activation
scale is dynamic per token until the freeze, then static: the largest activation seen (all ranks), which is
what the exported checkpoint's `input_global_scale` holds.

Export (final/, best/model/)
  the model itself  `export: dequantized` (default): W_q in the original layout and dtype, loadable anywhere, equal
                    to the training forward's weights; `latent`: the latent weights as trained.
  <dir>-<format>/   (`quantized_export`) the packed checkpoint in the compressed-tensors format vLLM loads
                    (palingenesis.qat.compressed), next to the model: final-nvfp4/, best/model-nvfp4/.

Metrics (plugins/float_qat/*): strength, frozen; every `metrics_every` steps qerr_mean / qerr_max (relative squared
quantisation error ||W_q - W||^2 / ||W||^2 over the layers), saturated_blocks (share of blocks with a clipped
element), clip_mean; per layer with `layer_metrics` (qerr/<layer>, saturated/<layer>).

Trainers: pgs train (SFT, off-policy distillation), pgs distill (OPD) and pgs rl, where the rollout engine samples
from the quantised weights (rollout_weights). With the teacher at full precision, distillation is QAT's recovery
recipe (see the QAT guide).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from palingenesis.hooks import PluginContext, Ramp, TrainerPlugin
from palingenesis.qat.formats import (
    ACTIVATION_FORMATS,
    FORMATS,
    MX_ROUNDING,
    FormatSpec,
    Quantized,
    lsq_fake_quantize,
    quantize,
    quantize_activation,
    straight_through,
)

logger = logging.getLogger(__name__)

SCALES = ("max", "mse", "learned")
ACTIVATIONS = ("none", "fp8", "nvfp4", "mxfp4")
EXPORTS = ("dequantized", "latent")
DEFAULT_INCLUDE = (r"(^|\.)layers\.\d+\.",)  # the decoder layers' linear layers (not a vision tower, not the head)
# Kept in full precision by default: Gated DeltaNet's decay / write-gate projections (in_proj_a / in_proj_b: one output
# per head, a few thousand weights, numerically sensitive, and too narrow for the 4-bit kernels) and MoE routers
# and the encoders of other modalities, whose own `layers.N` the default include would match (an inference engine
# builds them from plain layers: it cannot load a packed weight there)
DEFAULT_EXCLUDE = (
    r"\.in_proj_[ab]$",
    r"\.mlp\.gate$",
    r"\.shared_expert_gate$",
    r"(^|\.)(visual|vision_tower|vision_model|audio_tower|audio_model|multi_modal_projector)\.",
)
# Layers an inference engine fuses into one matmul: NVFP4 shards of one fused layer share the per-tensor scale
# (vLLM keeps one per fused layer). Matched among the children of one parent module.
FUSED_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),  # Gated DeltaNet (Qwen3.5): vLLM's in_proj_qkvz
    ("in_proj_b", "in_proj_a"),  # vLLM's in_proj_ba
)
CLIP_KEY = "qat_clip"


def activation_spec(activations: str, weight_format: str) -> FormatSpec | None:
    """The activation format: per token; fp8 with fp8_block weights per token and 128 inputs (vLLM's FP8 block
    scheme)."""
    if activations == "none":
        return None
    if activations == "fp8" and weight_format == "fp8_block":
        return ACTIVATION_FORMATS["fp8_group"]
    return ACTIVATION_FORMATS[activations]


def _full(t: Tensor) -> Tensor:
    """The whole tensor of an FSDP2 (DTensor) parameter (a collective: every rank must call it), else t."""
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _distributed(ctx: PluginContext) -> bool:
    return ctx.world_size > 1 and dist.is_available() and dist.is_initialized()


# ─────────────────────────────────────────────────────────────── modules


class _FloatQuantMixin:
    """Weight fake quantisation shared by the linear and embedding modules."""

    spec: FormatSpec
    weight: Tensor

    def _init_quant(self, spec: FormatSpec, mx_rounding: str, scales: str, clip_min: float):
        self.spec, self.mx_rounding, self.scales, self.clip_min = spec, mx_rounding, scales, clip_min
        self.register_buffer("strength", torch.ones(()), persistent=False)
        rows = self.weight.shape[0] // spec.block_rows
        device = self.weight.device
        if scales == "learned":
            self.qat_clip = nn.Parameter(torch.ones(rows, 1, dtype=torch.float32, device=device))
        elif scales == "mse":
            self.register_buffer(CLIP_KEY, torch.ones(rows, 1, dtype=torch.float32, device=device), persistent=False)
        else:
            self.qat_clip = None
        # frozen stored scales (freeze_scales_at): set by the first forward after the plugin asks for them
        self.register_buffer("frozen_scale", None, persistent=False)
        self.register_buffer("frozen_global", None, persistent=False)
        self.freeze_pending = False
        self.siblings: tuple = ()  # the modules of this one's fused group (itself included): shared global scale

    def clip_factor(self, clip: Tensor | None = None) -> Tensor | None:
        clip = self.qat_clip if clip is None else clip
        if clip is None:
            return None
        # the clamped value with the factor's own gradient: torch.clamp passes none at or beyond a bound, and
        # the search leaves most rows exactly at 1 (FloatQAT._project_clips keeps the parameter in range)
        return clip + (clip.clamp(self.clip_min, 1.0) - clip).detach()

    def global_amax(self, weights: Iterable[Tensor] | None = None) -> Tensor | None:
        """NVFP4: the largest |w| over the fused group (the forward's own, unsharded weights by default)."""
        if not self.spec.has_global_scale:
            return None
        if weights is None:
            weights = [m.weight for m in (self.siblings or (self,))]
        return torch.stack([w.detach().abs().amax().float() for w in weights]).amax()

    def quantize_now(
        self, weight: Tensor, global_amax: Tensor | None, clip: Tensor | None = None, statistics: bool = True
    ) -> Quantized:
        """`weight` (a whole matrix, on any device) quantised with this module's settings, frozen scales
        included. `clip`: the clipping factors to use instead of the module's own (a gathered copy).
        `statistics`: False skips the saturation statistic (the forward)."""
        dev = weight.device
        if self.frozen_scale is not None:
            frozen_global = None if self.frozen_global is None else self.frozen_global.to(dev)
            return quantize(
                weight,
                self.spec,
                scale=self.frozen_scale.to(dev),
                global_scale=frozen_global,
                mx_rounding=self.mx_rounding,
                track_saturation=statistics,
            )
        clip = self.qat_clip if clip is None else clip
        return quantize(
            weight,
            self.spec,
            clip=None if clip is None else self.clip_factor(clip.detach().to(dev)),
            global_amax=None if global_amax is None else global_amax.to(dev),
            mx_rounding=self.mx_rounding,
            track_saturation=statistics,
        )

    def _freeze(self, weight: Tensor) -> None:
        with torch.no_grad():
            q = self.quantize_now(weight, self.global_amax(), statistics=False)
        self.frozen_scale = q.scale
        self.frozen_global = q.global_scale
        self.freeze_pending = False

    def effective_weight(self) -> Tensor:
        w = self.weight
        if self.freeze_pending:
            self._freeze(w)
        if self.frozen_scale is None and self.scales == "learned":
            w_q = lsq_fake_quantize(w, self.spec, self.clip_factor(), self.global_amax(), self.mx_rounding)
            s = self.strength
            return s * w_q.to(w.dtype) + (1 - s) * w  # exactly w_q at strength 1 (see straight_through)
        with torch.no_grad():
            w_q = self.quantize_now(w, self.global_amax(), statistics=False).dequantize()
        return straight_through(w, w_q, self.strength)


class FloatQuantLinear(_FloatQuantMixin, nn.Module):
    """nn.Linear with fake-quantised weights (and optionally activations) and a full-precision latent weight."""

    def __init__(
        self,
        linear: nn.Linear,
        spec: FormatSpec,
        activation: FormatSpec | None = None,
        mx_rounding: str = "even",
        scales: str = "max",
        clip_min: float = 0.5,
    ):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight  # the same Parameter: names, ties and optimizer state carry over
        self.bias = linear.bias
        self.activation = activation
        self._init_quant(spec, mx_rounding, scales, clip_min)
        # NVFP4 activations: the largest |x| seen (calibration of the static per-tensor scale), and that scale's
        # amax once frozen
        self.register_buffer("act_amax", torch.zeros((), device=self.weight.device), persistent=False)
        self.register_buffer("act_global_amax", None, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation is not None:
            x = self._quantize_input(x)
        return F.linear(x, self.effective_weight(), self.bias)

    def _quantize_input(self, x: Tensor) -> Tensor:
        spec = self.activation
        if spec.has_global_scale and self.training:
            with torch.no_grad():
                torch.maximum(self.act_amax, x.detach().abs().amax().float(), out=self.act_amax)
        x_q = quantize_activation(x, spec, self.act_global_amax, self.mx_rounding)
        return straight_through(x, x_q, self.strength)

    def extra_repr(self) -> str:
        act = self.activation.name if self.activation is not None else "none"
        return f"in={self.in_features}, out={self.out_features}, weights={self.spec.name}, activations={act}"


class FloatQuantEmbedding(_FloatQuantMixin, nn.Module):
    """nn.Embedding whose rows are fake-quantised (blocks along the embedding dimension)."""

    def __init__(
        self,
        embedding: nn.Embedding,
        spec: FormatSpec,
        mx_rounding: str = "even",
        scales: str = "max",
        clip_min: float = 0.5,
    ):
        super().__init__()
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.weight = embedding.weight
        self._init_quant(spec, mx_rounding, scales, clip_min)

    def forward(self, ids: Tensor) -> Tensor:
        return F.embedding(ids, self.effective_weight(), self.padding_idx)


QUANT_MODULES = (FloatQuantLinear, FloatQuantEmbedding)


def search_clip(
    weight: Tensor,
    spec: FormatSpec,
    clip_min: float,
    steps: int,
    global_amax: Tensor | None = None,
    mx_rounding: str = "even",
) -> Tensor:
    """MSE-optimal clipping: per row block, the factor among `steps` values from 1 down to clip_min that minimises
    the squared quantisation error. float32 [rows / block_rows, 1]."""
    with torch.no_grad():
        w = weight.to(torch.float32)
        groups = w.shape[0] // spec.block_rows
        best = torch.ones(groups, 1, dtype=torch.float32, device=w.device)
        best_err = torch.full((groups,), float("inf"), device=w.device)
        candidates = torch.linspace(1.0, clip_min, steps).tolist() if steps > 1 else [1.0]
        for alpha in candidates:
            d = quantize(w, spec, clip=alpha, global_amax=global_amax, mx_rounding=mx_rounding).dequantize()
            err = (d - w).pow(2).reshape(groups, -1).sum(1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best[better] = alpha
        return best


# ─────────────────────────────────────────────────────────────── the plugin


class FloatQAT(TrainerPlugin):
    """FP4 / FP8 quantisation-aware training: fake-quantised linear layers, scales, ramp, freeze, export."""

    name = "float_qat"

    def __init__(
        self,
        format: str = "nvfp4",
        activations: str = "none",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        quantize_embedding: bool = False,
        quantize_head: bool = False,
        scales: str = "max",
        clip_min: float = 0.5,
        clip_steps: int = 16,
        clip_lr: float = 1.0e-3,
        mx_rounding: str = "even",
        strength=1.0,
        freeze_scales_at: float | None = None,
        fused_groups: list | None = None,
        export: str = "dequantized",
        quantized_export="auto",
        layer_metrics: bool = False,
        metrics_every: int = 50,
    ):
        from palingenesis.qat.compressed import scheme_for

        def choice(arg: str, value, allowed) -> str:
            if value not in allowed:
                raise ValueError(f"float_qat: {arg} must be one of {list(allowed)}, got {value!r}")
            return value

        def patterns(arg: str, value, default) -> list[re.Pattern]:
            if value is None:
                value = list(default)
            if isinstance(value, str) or not all(isinstance(p, str) for p in value):
                raise ValueError(f"float_qat: {arg} must be a list of regular expressions, got {value!r}")
            try:
                return [re.compile(p) for p in value]
            except re.error as e:
                raise ValueError(f"float_qat: {arg}: invalid regular expression ({e})") from None

        def boolean(arg: str, value) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"float_qat: {arg} must be true or false, got {value!r}")
            return value

        self.format = choice("format", format, FORMATS)
        self.activations = choice("activations", activations, ACTIVATIONS)
        self.spec = FORMATS[self.format]
        self.activation_spec = activation_spec(self.activations, self.format)
        self.include = patterns("include", include, DEFAULT_INCLUDE)
        self.exclude = patterns("exclude", exclude, DEFAULT_EXCLUDE)
        self.quantize_embedding = boolean("quantize_embedding", quantize_embedding)
        self.quantize_head = boolean("quantize_head", quantize_head)
        self.scales = choice("scales", scales, SCALES)
        if not isinstance(clip_min, (int, float)) or not 0.0 < clip_min <= 1.0:
            raise ValueError(f"float_qat: clip_min must be in (0, 1], got {clip_min!r}")
        if not isinstance(clip_steps, int) or isinstance(clip_steps, bool) or clip_steps < 1:
            raise ValueError(f"float_qat: clip_steps must be a positive integer, got {clip_steps!r}")
        if not isinstance(clip_lr, (int, float)) or clip_lr < 0:
            raise ValueError(f"float_qat: clip_lr must be >= 0 (0: train.learning_rate), got {clip_lr!r}")
        self.clip_min, self.clip_steps, self.clip_lr = float(clip_min), clip_steps, float(clip_lr)
        self.mx_rounding = choice("mx_rounding", mx_rounding, MX_ROUNDING)
        try:
            self.strength = Ramp.parse(strength)
        except (TypeError, ValueError) as e:
            raise ValueError(f"float_qat: strength: {e}") from None
        if freeze_scales_at is not None and (
            not isinstance(freeze_scales_at, (int, float)) or not 0.0 <= freeze_scales_at <= 1.0
        ):
            raise ValueError(f"float_qat: freeze_scales_at must be null or in [0, 1], got {freeze_scales_at!r}")
        self.freeze_scales_at = None if freeze_scales_at is None else float(freeze_scales_at)
        groups = FUSED_GROUPS if fused_groups is None else fused_groups
        if not all(isinstance(g, (list, tuple)) and all(isinstance(n, str) for n in g) for g in groups):
            raise ValueError(f"float_qat: fused_groups must be a list of lists of layer names, got {fused_groups!r}")
        self.fused_groups = [tuple(g) for g in groups]
        self.export = choice("export", export, EXPORTS)
        if quantized_export not in ("auto", True, False):
            raise ValueError(f"float_qat: quantized_export must be auto, true or false, got {quantized_export!r}")
        self.scheme = scheme_for(self.format, self.activations)
        if quantized_export is True and self.scheme is None:
            from palingenesis.qat.compressed import SCHEMES

            supported = ", ".join(f"{f} + {a}" for f, a in SCHEMES)
            raise ValueError(
                f"float_qat: no vLLM checkpoint format for {self.format} weights with {self.activations} activations "
                f"(supported: {supported}); set quantized_export: false to train it with fake quantisation only"
            )
        self.quantized_export = self.scheme is not None and quantized_export in ("auto", True)
        self.layer_metrics = boolean("layer_metrics", layer_metrics)
        if not isinstance(metrics_every, int) or metrics_every < 1:
            raise ValueError(f"float_qat: metrics_every must be a positive integer, got {metrics_every!r}")
        self.metrics_every = metrics_every

        self.modules: dict[str, nn.Module] = {}
        self.skipped: dict[str, str] = {}  # layer -> why it stays in full precision
        self._frozen = False
        self._restored: dict = {}
        self._tied_keys: tuple[str, str] | None = None
        self._pending: dict = {}  # export_state_dict -> on_export_saved: the packed tensors of the last export
        self._report: dict = {}
        self._head_names: set[str] = set()

    def setup(self, ctx: PluginContext) -> None:
        memory = getattr(ctx.config, "memory", None)
        if getattr(memory, "float8_training", False):
            raise ValueError(
                "float_qat with memory.float8_training: float8 training swaps the linear layers this plugin "
                "quantises (and computes them in its own FP8); turn one of them off"
            )

    # -- model

    def transform_model(self, model: nn.Module, ctx: PluginContext):
        from palingenesis.qat.compressed import SERVING_NOTES

        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        chosen: dict[str, nn.Module] = {}
        head_name = embed_name = None
        for name, module in model.named_modules():
            if module is head:
                head_name = name
            if module is embed:
                embed_name = name
        self._head_names = {head_name} if head_name is not None else set()
        for name, module in model.named_modules():
            if module is head:
                if self.quantize_head and isinstance(module, nn.Linear):
                    chosen[name] = module
                continue
            if module is embed:
                if self.quantize_embedding and isinstance(module, nn.Embedding):
                    chosen[name] = module
                continue
            if type(module) is nn.Linear and any(p.search(name) for p in self.include):
                if not any(p.search(name) for p in self.exclude):
                    chosen[name] = module
        for name, module in list(chosen.items()):
            rows, cols = module.weight.shape
            problem = self.spec.check_shape(rows, cols)
            if problem is None and self.activation_spec is not None and module is not head and module is not embed:
                problem = self.activation_spec.check_shape(1, cols)
            if problem:
                self.skipped[name] = problem
                del chosen[name]
        # a fused layer is quantised whole or not at all (an engine cannot mix schemes inside one matmul)
        for name in list(chosen):
            parent, _, leaf = name.rpartition(".")
            for group in self.fused_groups:
                if leaf in group:
                    members = [f"{parent}.{m}" if parent else m for m in group]
                    gone = [m for m in members if m in self.skipped]
                    if gone and name in chosen:
                        self.skipped[name] = f"its fused partner {gone[0]} is not quantised"
                        del chosen[name]
        for name, why in self.skipped.items():
            logger.warning(f"float_qat: {name} stays in full precision ({why})")
        if not chosen:
            raise ValueError("float_qat: no layer to quantise (check include / exclude and the layer shapes)")

        for name, module in chosen.items():
            if isinstance(module, nn.Embedding):
                new = FloatQuantEmbedding(module, self.spec, self.mx_rounding, self.scales, self.clip_min)
            else:
                act = None if module is head else self.activation_spec
                new = FloatQuantLinear(module, self.spec, act, self.mx_rounding, self.scales, self.clip_min)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, new)
            self.modules[name] = new
        for name, m in self.modules.items():
            parent, _, leaf = name.rpartition(".")
            for group in self.fused_groups:
                if leaf in group:
                    members = [f"{parent}.{g}" if parent else g for g in group]
                    m.siblings = tuple(self.modules[k] for k in members if k in self.modules)
        tied = head is not None and embed is not None and head.weight is embed.weight
        if tied:
            if head_name is not None and embed_name is not None:
                self._tied_keys = (f"{embed_name}.weight", f"{head_name}.weight")
            if self.quantize_head != self.quantize_embedding:
                logger.warning(
                    "float_qat: the head is tied to the embedding but only one of them is quantised; the shared "
                    "latent weight then serves a quantised and a full-precision module"
                )
        if self.quantized_export and self.scheme in SERVING_NOTES:
            logger.warning(f"float_qat: {SERVING_NOTES[self.scheme]}")
        n = sum(m.weight.numel() for m in self._unique_modules())
        act = f", {self.activations} activations" if self.activation_spec else ", weight-only"
        logger.info(
            f"float_qat: {len(self.modules)} layers in {self.format}{act}, scales {self.scales} "
            f"({n / 1e6:.1f}M latent weights; {len(self.skipped)} kept in full precision)"
        )
        return model

    def _unique_modules(self) -> list[nn.Module]:
        seen, out = set(), []
        for m in self.modules.values():
            if id(m.weight) not in seen:
                seen.add(id(m.weight))
                out.append(m)
        return out

    def param_groups(self, model: nn.Module, ctx: PluginContext) -> list[dict]:
        """scales: learned: the clipping factors in a group of their own (no weight decay, lr clip_lr)."""
        clips = [m.qat_clip for m in self.modules.values() if isinstance(m.qat_clip, nn.Parameter)]
        if not clips:
            return []
        override = {"params": clips, "weight_decay": 0.0}
        if self.clip_lr:
            override["lr"] = self.clip_lr
        return [override]

    # -- training

    def on_train_start(self, model: nn.Module, ctx: PluginContext) -> None:
        restored = self._restored.get("modules", {})
        if restored:
            self._apply_restored(restored)
        self._frozen = bool(self._restored.get("frozen", False))
        if self.scales in ("mse", "learned") and not restored:
            self.calibrate()
        self._set_strength(ctx)

    @torch.no_grad()
    def calibrate(self) -> None:
        """MSE-optimal clipping factors for every layer, from its current weights (every rank: it gathers
        sharded weights)."""
        for group in self._groups():
            weights = [_full(m.weight.detach()) for m in group]
            amax = group[0].global_amax(weights)
            for m, w in zip(group, weights):
                if m.qat_clip is None:
                    continue
                clip = search_clip(w, m.spec, m.clip_min, self.clip_steps, amax, m.mx_rounding)
                _assign(m.qat_clip, clip)
        clips = [m.qat_clip for m in self.modules.values() if m.qat_clip is not None]
        if clips:
            mean = float(torch.cat([_full(c.detach()).flatten() for c in clips]).mean())
            logger.info(f"float_qat: MSE-optimal clipping searched (mean factor {mean:.3f})")

    def _groups(self) -> list[list[nn.Module]]:
        """The quantised modules grouped by shared global scale (each once, in a fixed order on every rank)."""
        seen, out = set(), []
        for m in self.modules.values():
            if id(m) in seen:
                continue
            group = list(m.siblings) if m.siblings else [m]
            seen.update(id(g) for g in group)
            out.append(group)
        return out

    def _set_strength(self, ctx: PluginContext) -> None:
        value = float(self.strength(ctx.progress))
        for m in self.modules.values():
            m.strength.fill_(value)

    @torch.no_grad()
    def _project_clips(self) -> None:
        """scales: learned: the factors back into [clip_min, 1] (projected gradient). The forward uses their
        clamped value; without the projection a factor pushed past a bound would drift away from it, and need as
        many steps to come back into effect."""
        for m in self.modules.values():
            if isinstance(m.qat_clip, nn.Parameter):
                clip = m.qat_clip.to_local() if hasattr(m.qat_clip, "to_local") else m.qat_clip
                clip.clamp_(m.clip_min, 1.0)

    def on_forward_start(self, model: nn.Module, batch: dict, ctx: PluginContext) -> None:
        self._set_strength(ctx)
        if self.scales == "learned" and not self._frozen:
            self._project_clips()
        if self.freeze_scales_at is not None and not self._frozen and ctx.progress >= self.freeze_scales_at:
            self._reduce_activation_amax(ctx)
            for m in self.modules.values():
                m.freeze_pending = True  # the next forward computes them from the whole (unsharded) weights
                if isinstance(m, FloatQuantLinear) and m.activation is not None and m.activation.has_global_scale:
                    m.act_global_amax = m.act_amax.clone()
            self._frozen = True
            logger.info(f"float_qat: scales frozen at step {ctx.step} (progress {ctx.progress:.2f})")

    def _activation_modules(self) -> list[FloatQuantLinear]:
        return [
            m
            for m in self.modules.values()
            if isinstance(m, FloatQuantLinear) and m.activation is not None and m.activation.has_global_scale
        ]

    @torch.no_grad()
    def _reduce_activation_amax(self, ctx: PluginContext) -> None:
        """The activation observers' maximum over the ranks (each saw its own data)."""
        mods = self._activation_modules()
        if not mods or not _distributed(ctx):
            return
        values = torch.stack([m.act_amax for m in mods])
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        for m, v in zip(mods, values):
            m.act_amax.copy_(v)

    @torch.no_grad()
    def metrics(self, model: nn.Module, ctx: PluginContext) -> dict[str, float]:
        out = {"strength": float(self.strength(ctx.progress)), "frozen": float(self._frozen)}
        if ctx.step % self.metrics_every and ctx.step != ctx.total_steps:
            return out
        self._reduce_activation_amax(ctx)
        errors, saturated, blocks = [], 0.0, 0
        names = {id(m): n for n, m in self.modules.items()}
        for group in self._groups():
            weights = [_full(m.weight.detach()) for m in group]
            amax = group[0].global_amax(weights)
            for m, w in zip(group, weights):
                clip = _full(m.qat_clip.detach()) if m.qat_clip is not None else None
                q = m.quantize_now(w, amax, clip)
                w32 = w.float()
                err = float((q.dequantize() - w32).pow(2).sum() / w32.pow(2).sum().clamp(min=1e-30))
                sat = q.saturated.float()
                errors.append(err)
                saturated += float(sat.sum())
                blocks += sat.numel()
                if self.layer_metrics:
                    short = names[id(m)]
                    out[f"qerr/{short}"] = err
                    out[f"saturated/{short}"] = float(sat.mean())
        out["qerr_mean"] = sum(errors) / len(errors)
        out["qerr_max"] = max(errors)
        out["saturated_blocks"] = saturated / max(blocks, 1)
        clips = [_full(m.qat_clip.detach()).flatten() for m in self.modules.values() if m.qat_clip is not None]
        if clips:
            clips = torch.cat(clips)
            out["clip_mean"] = float(clips.clamp(self.clip_min, 1.0).mean())
        acts = self._activation_modules()
        if acts:
            out["act_amax_max"] = float(torch.stack([m.act_amax for m in acts]).max())
        return out

    def on_train_end(self, model: nn.Module, ctx: PluginContext) -> None:
        self._reduce_activation_amax(ctx)

    # -- rollout engines (pgs rl, pgs distill with vLLM)

    def rollout_weights(self, named: Iterable[tuple[str, Tensor]], ctx: PluginContext) -> Iterator[tuple[str, Tensor]]:
        """The engine samples from the quantised policy: each quantised weight replaced by W_q (dequantized,
        in the tensor's dtype); the clipping factors dropped. A fused group's weights are held back until the
        whole group arrived (they share NVFP4's per-tensor scale)."""
        keys = {f"{name}.weight": m for name, m in self.modules.items()}
        held: dict[str, tuple[str, Tensor]] = {}
        for name, tensor in named:
            clean = _clean_name(name)
            if clean.endswith("." + CLIP_KEY):
                continue
            module = keys.get(clean)
            if module is None:
                yield name, tensor
                continue
            group = module.siblings or (module,)
            held[clean] = (name, tensor)
            group_keys = [k for k, m in keys.items() if any(m is g for g in group)]
            if not all(k in held for k in group_keys):
                continue
            weights = [held[k][1] for k in group_keys]
            amax = module.global_amax(weights)
            for k, w in zip(group_keys, weights):
                original, _ = held.pop(k)
                m = keys[k]
                clip = _full(m.qat_clip.detach()) if m.qat_clip is not None else None
                with torch.no_grad():
                    w_q = m.quantize_now(w, amax, clip, statistics=False).dequantize().to(w.dtype)
                yield original, w_q
        if held:
            raise RuntimeError(f"float_qat: rollout weights {sorted(held)[:3]} arrived without their fused partners")

    # -- export and resume

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict:
        from palingenesis.qat.compressed import pack

        out = {k: v for k, v in state.items() if not k.endswith("." + CLIP_KEY)}
        self._pending = {}
        report = {}
        for group in self._groups():
            keys = [f"{self._name_of(m)}.weight" for m in group]
            missing = [k for k in keys if k not in state]
            if missing:
                raise RuntimeError(f"float_qat: the exported state has no {missing[0]} (a layer quantised in training)")
            weights = [state[k] for k in keys]
            if not all(bool(torch.isfinite(w).all()) for w in weights):
                raise RuntimeError(f"float_qat: non-finite values in {keys[0]}: nothing sensible to export")
            amax = group[0].global_amax(weights)
            for m, key, w in zip(group, keys, weights):
                clip_key = key[: -len("weight")] + CLIP_KEY
                clip = state[clip_key] if clip_key in state else m.qat_clip
                q = m.quantize_now(w, amax, None if clip is None else clip.detach().cpu())
                deq = q.dequantize().to(w.dtype)
                w32 = w.float()
                report[key[: -len(".weight")]] = {
                    "relative_error": float((q.dequantize() - w32).pow(2).sum() / w32.pow(2).sum().clamp(min=1e-30)),
                    "saturated_blocks": float(q.saturated.float().mean()),
                }
                if self.export == "dequantized":
                    out[key] = deq
                packable = isinstance(m, FloatQuantLinear) and self._name_of(m) not in self._head_names
                if self.quantized_export and packable:
                    act_amax = None
                    if m.activation is not None and m.activation.has_global_scale:
                        act_amax = (m.act_global_amax if m.act_global_amax is not None else m.act_amax).detach().cpu()
                        if float(act_amax) <= 0:
                            raise RuntimeError(
                                f"float_qat: {key}: no activation was observed for its static input scale "
                                "(the model never ran a training forward)"
                            )
                    self._pending[key] = (out[key] if self.export == "dequantized" else deq, pack(q, act_amax))
        if self._tied_keys and self.export == "dequantized":
            # tied embedding / head: both keys get the quantised matrix (the trained model used one tensor)
            source = next((k for k in self._tied_keys if k[: -len(".weight")] in self.modules), None)
            if source is not None and source in out:
                for key in self._tied_keys:
                    if key in out:
                        out[key] = out[source]
        self._report = report
        return out

    def _name_of(self, module: nn.Module) -> str:
        for name, m in self.modules.items():
            if m is module:
                return name
        raise KeyError(module)

    def on_export_saved(self, path: Path, ctx: PluginContext) -> None:
        if not self._pending:
            return
        from palingenesis.qat.compressed import write_checkpoint

        path = Path(path)
        target = path.parent / f"{path.name}-{self.format}"
        try:
            write_checkpoint(path, target, self._pending, self.scheme, getattr(self, "_report", {}))
        finally:
            self._pending = {}
        logger.info(f"float_qat: {self.format} checkpoint for vLLM (compressed-tensors) -> {target}")

    def state_dict(self) -> dict:
        modules = {}
        for name, m in self.modules.items():
            entry = {}
            if m.qat_clip is not None:
                entry["clip"] = _full(m.qat_clip.detach()).cpu()
            if m.frozen_scale is not None:
                entry["frozen_scale"] = m.frozen_scale.cpu()
                if m.frozen_global is not None:
                    entry["frozen_global"] = m.frozen_global.cpu()
            if isinstance(m, FloatQuantLinear) and m.activation is not None and m.activation.has_global_scale:
                entry["act_amax"] = m.act_amax.cpu()
                if m.act_global_amax is not None:
                    entry["act_global_amax"] = m.act_global_amax.cpu()
            modules[name] = entry
        return {"frozen": self._frozen, "modules": modules}

    def load_state_dict(self, state: dict) -> None:
        self._restored = dict(state)

    def _apply_restored(self, restored: dict) -> None:
        unknown = sorted(set(restored) - set(self.modules))
        if unknown:
            raise RuntimeError(
                f"float_qat: the checkpoint's plugin state has layers this run does not quantise: {unknown[:3]}"
            )
        for name, entry in restored.items():
            m = self.modules[name]
            device = m.weight.device
            if "clip" in entry and m.qat_clip is not None:
                _assign(m.qat_clip, entry["clip"].to(device))
            if "frozen_scale" in entry:
                m.frozen_scale = entry["frozen_scale"].to(device)
                m.frozen_global = entry["frozen_global"].to(device) if "frozen_global" in entry else None
            if "act_amax" in entry:
                m.act_amax.copy_(entry["act_amax"].to(device))
            if "act_global_amax" in entry:
                m.act_global_amax = entry["act_global_amax"].to(device)


def _assign(target: Tensor, full: Tensor) -> None:
    """Copy a whole tensor into a parameter or buffer, sharded (FSDP2 DTensor) or not."""
    with torch.no_grad():
        if hasattr(target, "to_local"):
            from torch.distributed.tensor import distribute_tensor

            local = distribute_tensor(full.to(target.dtype), target.device_mesh, target.placements).to_local()
            target.to_local().copy_(local)
        else:
            target.copy_(full.to(device=target.device, dtype=target.dtype))


def _clean_name(name: str) -> str:
    for prefix in ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module."):
        name = name.replace(prefix, "")
    return name
