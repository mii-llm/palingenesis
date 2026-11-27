"""Quantisation-aware training plugins.

    palingenesis.qat:FloatQAT     FP4 / FP8 (NVFP4, MXFP4, FP8 per row / 128 x 128 blocks), exported for vLLM
                                  (palingenesis.qat.fp, .formats, .compressed)
    palingenesis.qat:TernaryQAT   ternary weights (this module), a reference for writing your own

Ternary quantisation-aware training, as a trainer plugin:

    plugins:
      custom:
        - class: palingenesis.qat:TernaryQAT
          args:
            group_size: 128                 # one scale per 128 input weights (0: per tensor, -1: per row)
            scale: absmean                  # absmean | absmax | twn | mse    (palingenesis.qat.ternary)
            scale_mode: dynamic             # dynamic (from the latents, every forward) | frozen | learned (LSQ)
            quantizer: round                # round | stochastic (training forwards sample the codes)
            ste: identity                   # identity | clipped | tanh (ste_args: {clip: 1.0} / {sharpness: 4.0})
            exclude: ["lm_head", "embed"]   # nn.Linear names kept in full precision (regular expressions)
            quantize_embedding: false       # also the input embedding (and a tied head through it)
            quantize_head: false            # also the output head
            strength: {start: 0.0, end: 0.2}            # quantisation ramp: 0 = full precision, 1 = ternary
            grid_weight: {start: 0.0, end: 1.0, start_value: 0.0, end_value: 1.0e-4}
            latent_lr: 1.0e-4               # optimizer group of the latent weights (0: train.learning_rate)
            latent_lr_relative: false       # true: latent_lr is in units of each tensor's mean |W| (code units)
            latent_eps: 1.0e-12             # Adam eps of that group (0: the optimizer's)
            clamp: 1.5                      # after each step, |w| <= clamp * scale (0: off)
            hidden_layers: [-1]             # match these decoder layers' outputs to the teacher's (MSE)
            hidden_weight: 1.0
            export: folded                  # final model: ternary values in the original layout (or latent)
            save_latent: false              # with export: folded, also write <export>-latent/ (the latent weights)

    distill:
      enabled: true
      teachers: {fp: {model: <the full-precision model>}}

Every substituted module keeps its full-precision **latent** weight (the same Parameter, under the same
name, so checkpoints, resume and tied embeddings work unchanged). The forward uses

    W_q = s * clip(round(W / s), -1, 1),     s = mean |W| per group            (absmean, BitNet b1.58,
    W_eff = W + strength * (W_q - W).detach()                                   arXiv:2402.17764)

a straight-through estimator: the gradient reaches W as if W_eff were W. `strength` ramps the
quantisation in (gradual quantisation); `grid_weight` adds the gradient of
grid_weight * sum (W - W_q)^2 to the latent weights before every optimizer step, pulling them towards
their grid points so that the final rounding changes little. Optional int8 per-token activations
(absmax, `activation_bits: 8`). Export "folded" writes W_q in place of W (any HF loader reads it; the
values are exactly ternary times a scale), "latent" writes the latent weights and scales as trained.
In pgs distill with a rollout engine (vLLM, SGLang), the engine is given W_eff at every weight push
(rollout_weights), so it samples from the policy the trainer scores.

The quantiser is assembled from three registries of palingenesis.qat.ternary (`scale`, `quantizer`, `ste`): a new
scheme is one function added to a registry. `scale_mode: frozen` computes the scales once from the loaded weights;
`learned` also trains a log-factor per scale in its own optimizer group (`scale_lr`). Frozen and learned scales live
in buffers / parameters next to each weight (`scale_alpha`, `scale_delta`, `log_scale`): single process only (not
FSDP), dropped from the folded export. Every export gets a `ternary_config.json` (which tensors are ternary, their
group size and quantiser), which `pgs merge ternary` and `palingenesis.qat.merge.check_ternary` read.

Defaults are BitNet b1.58's (absmean, round, identity). They are a starting point, not a recipe tuned for a model.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from palingenesis.hooks import FeatureSpec, PluginContext, Ramp, TrainerPlugin
from palingenesis.qat.ternary import TernarySpec, group_view, lsq_grad_scale

TERNARY_CONFIG = "ternary_config.json"
SCALE_MODES = ("dynamic", "frozen", "learned")
_AUX_KEYS = ("scale_alpha", "scale_delta", "log_scale")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────── quantiser


def _groups(w: Tensor, group_size: int) -> Tensor:
    """w [rows, cols] as [rows, n_groups, g] (group_size > 0), [rows, 1, cols] (-1: per row),
    or [1, 1, rows*cols] (0: per tensor)."""
    rows, cols = w.shape
    if group_size == 0:
        return w.reshape(1, 1, -1)
    if group_size < 0:
        return w.reshape(rows, 1, cols)
    return w.reshape(rows, cols // group_size, group_size)


def absmean_scale(w: Tensor, group_size: int, eps: float = 1e-8) -> Tensor:
    """Per-group mean |w| (detached), shaped to broadcast against _groups(w)."""
    return _groups(w.detach(), group_size).abs().mean(-1, keepdim=True).clamp(min=eps)


def ternary(w: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    """(codes in {-1, 0, 1}, scale): w ~= codes * scale, both in _groups layout."""
    scale = absmean_scale(w, group_size)
    return (_groups(w, group_size) / scale).round().clamp(-1, 1), scale


def quantize_weight(w: Tensor, group_size: int) -> Tensor:
    """The ternary weight s * clip(round(w / s), -1, 1), in w's shape (no gradient path)."""
    codes, scale = ternary(w.detach(), group_size)
    return (codes * scale).reshape(w.shape).to(w.dtype)


def quantize_activations(x: Tensor, bits: int = 8) -> Tensor:
    """Per-token symmetric absmax quantisation with a straight-through estimator."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-5) / qmax
    return x + ((x / scale).round().clamp(-qmax - 1, qmax) * scale - x).detach()


class _TernaryMixin:
    def _init_quant(self, group_size: int, activation_bits: int, spec: TernarySpec | None = None, scale_mode="dynamic"):
        self.group_size = group_size
        self.activation_bits = activation_bits
        self.spec = spec if spec is not None else TernarySpec(group_size)
        self.scale_mode = scale_mode
        # the BitNet default keeps its original code path (and its FSDP support)
        self._default = scale_mode == "dynamic" and self.spec.describe() == TernarySpec(group_size).describe()
        # buffers (not Python floats): changing them never makes torch.compile recompile
        self.register_buffer("strength", torch.ones(()), persistent=False)
        sharpness = float(self.spec.ste_args.get("sharpness", 4.0))
        self.register_buffer("sharpness", torch.full((), sharpness), persistent=False)
        if scale_mode != "dynamic":
            alpha, delta = self.spec.scales(group_view(self.weight.detach(), group_size))
            self.register_buffer("scale_alpha", alpha.clone())
            self.register_buffer("scale_delta", delta.clone())
        if scale_mode == "learned":
            self.log_scale = nn.Parameter(torch.zeros_like(self.scale_alpha))
            n = self.weight.numel() // self.scale_alpha.numel()
            self.log_scale.register_hook(lambda grad, k=lsq_grad_scale(n): grad * k)

    def fixed_scales(self) -> tuple[Tensor, Tensor] | None:
        """(alpha, delta) of frozen / learned scales, the learned factor applied; None when dynamic."""
        if self.scale_mode == "dynamic":
            return None
        alpha, delta = self.scale_alpha, self.scale_delta
        if self.scale_mode == "learned":
            factor = self.log_scale.detach().exp().to(alpha.dtype)
            alpha, delta = alpha * factor, delta * factor
        return alpha, delta

    @torch.no_grad()
    def quantize(self, w: Tensor) -> Tensor:
        """The ternary weight alpha * code of `w` (this module's weight, on any device), deterministic."""
        if self._default:
            return quantize_weight(w, self.group_size)
        fixed = self.fixed_scales()
        if fixed is not None:
            fixed = tuple(t.to(device=w.device, dtype=w.dtype) for t in fixed)
        return self.spec.quantize(w, fixed)

    def effective_weight(self) -> Tensor:
        w = self.weight
        if self._default:
            w_q = quantize_weight(w, self.group_size)
            return w + self.strength.to(w.dtype) * (w_q - w).detach()
        fixed = None
        if self.scale_mode != "dynamic":
            fixed = (self.scale_alpha.to(w.dtype), self.scale_delta.to(w.dtype))
        return self.spec.effective(
            w,
            self.strength,
            fixed=fixed,
            log_scale=self.log_scale if self.scale_mode == "learned" else None,
            sample=self.training and self.spec.quantizer == "stochastic",
            sharpness=self.sharpness.to(w.dtype) if self.spec.ste == "tanh" else None,
        )


class TernaryLinear(_TernaryMixin, nn.Module):
    """nn.Linear with ternary weights in the forward and a full-precision latent weight."""

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        activation_bits: int = 0,
        spec: TernarySpec | None = None,
        scale_mode: str = "dynamic",
    ):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight  # the same Parameter: names, ties and optimizer state carry over
        self.bias = linear.bias
        self._init_quant(group_size, activation_bits, spec, scale_mode)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_bits:
            x = quantize_activations(x, self.activation_bits)
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group_size={self.group_size}, a{self.activation_bits or 16}"


class TernaryEmbedding(_TernaryMixin, nn.Module):
    """nn.Embedding whose rows are ternary (scales per group of the embedding dimension)."""

    def __init__(
        self,
        embedding: nn.Embedding,
        group_size: int = 128,
        spec: TernarySpec | None = None,
        scale_mode: str = "dynamic",
    ):
        super().__init__()
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.weight = embedding.weight
        self._init_quant(group_size, 0, spec, scale_mode)

    def forward(self, ids: Tensor) -> Tensor:
        return F.embedding(ids, self.effective_weight(), self.padding_idx)


QUANT_MODULES = (TernaryLinear, TernaryEmbedding)


# ─────────────────────────────────────────────────────────────── sharded parameters


def _local(t: Tensor) -> Tensor:
    """The local shard of an FSDP2 (DTensor) parameter or gradient, else the tensor. FSDP2 shards
    dimension 0, so every local row is a whole row: per-row and per-group scales are local."""
    return t.to_local() if hasattr(t, "to_local") else t


def _scale_local(p: Tensor, group_size: int) -> Tensor:
    """absmean scale for the local rows of p; per-tensor scale reduced over the shards."""
    w = _local(p.detach())
    if group_size != 0 or not hasattr(p, "to_local"):
        return absmean_scale(w, group_size)
    total = torch.stack([w.abs().sum().float(), torch.tensor(float(w.numel()), device=w.device)])
    dist.all_reduce(total, group=p.device_mesh.get_group())
    return (total[0] / total[1].clamp(min=1)).clamp(min=1e-8).reshape(1, 1, 1).to(w.dtype)


def _local_codes(p: Tensor, group_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """(local weight in _groups layout, scale, codes)."""
    w = _local(p.detach())
    collective = group_size == 0 and hasattr(p, "to_local")
    scale = _scale_local(p, group_size) if collective else None  # every rank joins the all-reduce
    if w.numel() == 0:  # an uneven FSDP split can leave a rank without rows
        empty = w.reshape(0, 1, 1)
        return empty, empty, empty
    if scale is None:
        scale = _scale_local(p, group_size)
    g = _groups(w, group_size)
    return g, scale, (g / scale).round().clamp(-1, 1)


def saved_names(keys, saved) -> dict[str, str]:
    """{training-time key: name in the written model files}. An export may be written under other names than the
    trained model's (a text tower trained as `model.layers...` and stored as `model.language_model.layers...`):
    a key maps to itself when present, else to the one saved name sharing the longest run of trailing name
    components (at least two). Keys without a counterpart (the second name of a tied tensor) are left out."""
    saved = list(saved)
    present = set(saved)
    tails = [(name, name.split(".")[::-1]) for name in saved]
    out = {}
    for key in keys:
        if key in present:
            out[key] = key
            continue
        parts = key.split(".")[::-1]
        best, best_n, unique = None, 1, False
        for name, rev in tails:
            n = 0
            for a, b in zip(parts, rev):
                if a != b:
                    break
                n += 1
            if n > best_n:
                best, best_n, unique = name, n, True
            elif n == best_n and best is not None:
                unique = False
        if best is not None and unique:
            out[key] = best
    return out


# ─────────────────────────────────────────────────────────────── the plugin


class TernaryQAT(TrainerPlugin):
    """Substitute ternary modules, ramp them in, pull latents to the grid, match hidden states."""

    name = "ternary_qat"

    def __init__(
        self,
        group_size: int = 128,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        quantize_embedding: bool = False,
        quantize_head: bool = False,
        activation_bits: int = 0,
        scale: str = "absmean",
        scale_mode: str = "dynamic",
        scale_args: dict | None = None,
        scale_lr: float = 0.0,
        quantizer: str = "round",
        ste: str = "identity",
        ste_args: dict | None = None,
        ste_sharpness=None,
        strength=1.0,
        grid_weight=0.0,
        latent_lr: float = 0.0,
        latent_lr_relative: bool = False,
        latent_eps: float = 0.0,
        latent_weight_decay: float = 0.0,
        clamp: float = 1.5,
        hidden_layers: list | None = None,
        hidden_weight=1.0,
        hidden_loss: str = "mse",
        export: str = "folded",
        save_latent: bool = False,
        metrics_every: int = 50,
    ):
        if activation_bits not in (0, 4, 8):
            raise ValueError(f"activation_bits must be 0, 4 or 8 (got {activation_bits})")
        if export not in ("folded", "latent"):
            raise ValueError(f"export must be folded or latent (got {export!r})")
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"scale_mode must be one of {SCALE_MODES} (got {scale_mode!r})")
        TernarySpec(group_size, scale, quantizer, ste, scale_args, ste_args)  # validates the names
        self.scale, self.scale_mode, self.scale_args, self.scale_lr = scale, scale_mode, scale_args, scale_lr
        self.quantizer, self.ste, self.ste_args = quantizer, ste, ste_args
        # tanh estimator: its sharpness over training (a number or a ramp; default: ste_args' value, constant)
        self.ste_sharpness = Ramp.parse(ste_sharpness) if ste_sharpness is not None else None
        self.group_size = group_size
        self.include = [re.compile(p) for p in (include or [r".*"])]
        self.exclude = [re.compile(p) for p in (exclude if exclude is not None else [r"lm_head", r"embed"])]
        self.quantize_embedding = quantize_embedding
        self.quantize_head = quantize_head
        self.activation_bits = activation_bits
        self.strength = Ramp.parse(strength)
        self.grid_weight = Ramp.parse(grid_weight)
        self.latent_lr, self.latent_eps, self.latent_weight_decay = latent_lr, latent_eps, latent_weight_decay
        if latent_lr_relative and not latent_lr:
            raise ValueError("latent_lr_relative needs latent_lr (the step in units of mean |W|)")
        self.latent_lr_relative = latent_lr_relative
        self.clamp = clamp
        self.hidden_layers = list(hidden_layers or [])
        self.hidden_weight = Ramp.parse(hidden_weight)
        self.hidden_loss = hidden_loss
        self.export = export
        self.save_latent = save_latent
        self._latent_state: dict[str, Tensor] = {}
        self.metrics_every = max(1, metrics_every)
        self.modules: dict[str, nn.Module] = {}
        self._initial_codes: dict[str, Tensor] = {}
        self._grid_value = 0.0
        self._tied_keys: tuple[str, str] | None = None

    # -- model

    def _spec(self, group_size: int) -> TernarySpec:
        return TernarySpec(group_size, self.scale, self.quantizer, self.ste, self.scale_args, self.ste_args)

    def _group_size_for(self, name: str, cols: int) -> int:
        g = self.group_size
        if g > 0 and cols % g:
            logger.warning(f"ternary_qat: {name} has {cols} input columns, not a multiple of {g}: one scale per row")
            return -1
        return g

    def transform_model(self, model: nn.Module, ctx: PluginContext):
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        replacements = []
        for name, module in model.named_modules():
            if module is head:
                if self.quantize_head and isinstance(module, nn.Linear):
                    g = self._group_size_for(name, module.in_features)
                    replacements.append((name, TernaryLinear(module, g, 0, self._spec(g), self.scale_mode)))
                continue
            if module is embed:
                if self.quantize_embedding and isinstance(module, nn.Embedding):
                    g = self._group_size_for(name, module.embedding_dim)
                    replacements.append((name, TernaryEmbedding(module, g, self._spec(g), self.scale_mode)))
                continue
            if type(module) is nn.Linear and any(p.search(name) for p in self.include):
                if any(p.search(name) for p in self.exclude):
                    continue
                g = self._group_size_for(name, module.in_features)
                replacements.append(
                    (name, TernaryLinear(module, g, self.activation_bits, self._spec(g), self.scale_mode))
                )
        if not replacements:
            raise ValueError("ternary_qat: no module matched (check include / exclude)")
        for name, new in replacements:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, new)
            self.modules[name] = new
        tied = head is not None and embed is not None and head.weight is embed.weight
        if tied:
            module_names = {id(m): n for n, m in model.named_modules()}
            head_name = next(
                (n for n, m in self.modules.items() if isinstance(m, TernaryLinear) and m.weight is head.weight), None
            )
            embed_name = next((n for n, m in self.modules.items() if isinstance(m, TernaryEmbedding)), None)
            head_name = head_name or module_names.get(id(head))
            embed_name = embed_name or module_names.get(id(embed))
            if head_name and embed_name:
                self._tied_keys = (f"{embed_name}.weight", f"{head_name}.weight")
        if tied and (self.quantize_head != self.quantize_embedding):
            logger.warning(
                "ternary_qat: the head is tied to the embedding but only one of them is quantised; the shared "
                "latent weight then serves a ternary and a full-precision module"
            )
        n = sum(m.weight.numel() for m in self.modules.values())
        logger.info(f"ternary_qat: {len(self.modules)} modules ternary ({n / 1e6:.1f}M latent weights)")
        return model

    def feature_requests(self, ctx: PluginContext) -> list[FeatureSpec]:
        specs = []
        for item in self.hidden_layers:
            s, t = (item, item) if isinstance(item, int) else tuple(item)
            specs.append(FeatureSpec(student=s, teacher=t, weight=self.hidden_weight, loss=self.hidden_loss))
        return specs

    def param_groups(self, model: nn.Module, ctx: PluginContext) -> list[dict]:
        """The latent weights in a group of their own (weight decay latent_weight_decay, 0 by default:
        decay would pull latents to 0, i.e. towards the zero code)."""
        latents = [m.weight for m in self._unique_modules() if m.weight.requires_grad]
        if not latents:
            return []
        common = {"weight_decay": self.latent_weight_decay}
        if self.latent_eps:
            common["eps"] = self.latent_eps
        if self.latent_lr_relative:
            # Adam moves a weight by about lr per step whatever its gradient: a step that is a fixed fraction of the
            # tensor's own scale crosses rounding thresholds at the same rate in every tensor (and in every model)
            groups = [{"params": [p], "lr": self.latent_lr * float(_scale_local(p, 0)), **common} for p in latents]
        else:
            groups = [{"params": latents, **({"lr": self.latent_lr} if self.latent_lr else {}), **common}]
        if self.scale_mode == "learned":
            scales = {"params": [m.log_scale for m in self.modules.values()], "weight_decay": 0.0}
            if self.scale_lr:
                scales["lr"] = self.scale_lr
            groups.append(scales)
        return groups

    def _unique_modules(self) -> list[nn.Module]:
        seen, out = set(), []
        for m in self.modules.values():
            if id(m.weight) not in seen:
                seen.add(id(m.weight))
                out.append(m)
        return out

    # -- training

    @staticmethod
    def _codes(m) -> tuple[Tensor, Tensor, Tensor]:
        """(weight in group layout, alpha, codes) of a module's local rows, deterministic."""
        if m._default:
            return _local_codes(m.weight, m.group_size)
        g = group_view(_local(m.weight.detach()), m.group_size)
        alpha, delta = m.fixed_scales() or m.spec.scales(g)
        return g, alpha, m.spec.codes(g, alpha, delta)

    def on_train_start(self, model: nn.Module, ctx: PluginContext) -> None:
        unsupported = (
            not m._default and (m.scale_mode != "dynamic" or m.group_size == 0) for m in self.modules.values()
        )
        if ctx.is_fsdp and any(unsupported):
            raise ValueError(
                "ternary_qat: under FSDP, frozen / learned scales and per-tensor scales of a scheme other than "
                "absmean are not supported (use group_size > 0 or -1 with scale_mode: dynamic)"
            )
        self._set_strength(ctx)
        for name, m in self.modules.items():
            _, _, codes = self._codes(m)
            # A resumed run has loaded the codes it started from (load_state_dict runs before
            # this hook): keep them, so codes_changed continues instead of restarting at 0.
            # Several ranks each hold their own shard and only the main rank's codes are
            # saved, so they start again from the resumed weights.
            restored = self._initial_codes.get(name)
            if ctx.world_size > 1 or restored is None or restored.shape != codes.shape:
                self._initial_codes[name] = codes.to(torch.int8).cpu()

    def _set_strength(self, ctx: PluginContext) -> None:
        value = float(self.strength(ctx.progress))
        sharp = float(self.ste_sharpness(ctx.progress)) if self.ste_sharpness is not None else None
        for m in self.modules.values():
            m.strength.fill_(value)
            if sharp is not None:
                m.sharpness.fill_(sharp)

    def on_forward_start(self, model: nn.Module, batch: dict, ctx: PluginContext) -> None:
        self._set_strength(ctx)

    @torch.no_grad()
    def on_before_optimizer_step(self, model: nn.Module, optimizer, ctx: PluginContext) -> None:
        lam = float(self.grid_weight(ctx.progress))
        if lam <= 0:
            self._grid_value = 0.0
            return
        value = 0.0
        for m in self._unique_modules():
            p = m.weight
            if p.grad is None:
                continue
            g, scale, codes = self._codes(m)
            residual = g - codes * scale  # W - W_q
            _local(p.grad).add_((2.0 * lam * residual).reshape(_local(p.grad).shape).to(p.grad.dtype))
            value += lam * float(residual.float().pow(2).sum())
        self._grid_value = value

    @torch.no_grad()
    def on_after_optimizer_step(self, model: nn.Module, optimizer, ctx: PluginContext) -> None:
        if self.clamp <= 0:
            return
        for m in self._unique_modules():
            g, scale, _ = self._codes(m)
            if g.numel():
                bound = self.clamp * scale
                g.copy_(torch.maximum(torch.minimum(g, bound), -bound))
                _local(m.weight.data).copy_(g.reshape(_local(m.weight.data).shape))

    @torch.no_grad()
    def metrics(self, model: nn.Module, ctx: PluginContext) -> dict[str, float]:
        out = {"strength": float(self.strength(ctx.progress)), "grid_reg": self._grid_value}
        if ctx.step % self.metrics_every and ctx.step != ctx.total_steps:
            return out
        changed = zeros = total = 0
        for name, m in self.modules.items():
            _, _, codes = self._codes(m)
            start = self._initial_codes.get(name)
            if start is not None and start.shape == codes.shape:
                changed += int((codes.to(torch.int8).cpu() != start).sum())
            zeros += int((codes == 0).sum())
            total += codes.numel()
        counts = torch.tensor([changed, zeros, total], dtype=torch.float64)
        if dist.is_available() and dist.is_initialized() and ctx.is_fsdp:
            counts = counts.to(ctx.device)
            dist.all_reduce(counts)
            counts = counts.cpu()
        changed, zeros, total = counts.tolist()
        out["codes_changed"] = changed / max(total, 1)
        out["zero_share"] = zeros / max(total, 1)
        return out

    # -- rollout engines (pgs distill, pgs rl)

    def rollout_weights(self, named, ctx: PluginContext):
        """The engine samples from the policy the trainer computes with: each ternary module's weight replaced by
        W + strength * (W_q - W), the forward's effective weight (W_q itself at strength 1), from the whole tensor
        (per-tensor scales included). What an engine cannot be given stays as trained: activation quantisation
        (activation_bits), and a tied embedding / head of which only one is quantised (one tensor serves both)."""
        from palingenesis.qat.fp import _clean_name

        keys = {f"{name}.weight": m for name, m in self.modules.items()}
        for name, tensor in named:
            module = keys.get(_clean_name(name))
            if module is None:
                yield name, tensor
                continue
            with torch.no_grad():
                w = tensor.detach()
                strength = module.strength.to(device=w.device, dtype=w.dtype)
                yield name, w + strength * (module.quantize(w) - w)

    # -- export and resume

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict:
        if self.export == "latent":
            return state
        aux = {f"{name}.{key}" for name in self.modules for key in _AUX_KEYS}
        out = {k: v for k, v in state.items() if k not in aux}
        names = {f"{name}.weight": m for name, m in self.modules.items()}
        if self.save_latent:
            self._latent_state = {k: v for k, v in state.items() if k in names or k in aux}
        for key, m in names.items():
            if key in state:
                out[key] = m.quantize(state[key].float()).to(state[key].dtype)
        # tied embedding / head: both keys get the quantised matrix (the trained model used one tensor)
        if self._tied_keys:
            a, b = self._tied_keys
            source = a if a in names else b if b in names else None
            if source is not None and source in out:
                for key in (a, b):
                    if key in out:
                        out[key] = out[source]
        return out

    def describe(self) -> dict:
        """What `ternary_config.json` records: the ternary tensors (state-dict key -> group size) and the quantiser."""
        tensors = {f"{name}.weight": m.group_size for name, m in self.modules.items()}
        if self._tied_keys:
            sizes = [tensors[k] for k in self._tied_keys if k in tensors]
            if sizes:
                tensors.update({k: sizes[0] for k in self._tied_keys})
        return {
            "tensors": tensors,
            "export": self.export,
            "scale_mode": self.scale_mode,
            "quantizer": self._spec(self.group_size).describe(),
        }

    def on_export_saved(self, path, ctx: PluginContext) -> None:
        from safetensors import safe_open

        path = Path(path)
        shards = sorted(path.glob("*.safetensors"))
        on_disk = [k for shard in shards for k in safe_open(str(shard), "pt").keys()]
        layout = self.describe()
        if on_disk:  # the layout names tensors as the files do
            names = saved_names(layout["tensors"], on_disk)
            layout["tensors"] = {names[k]: g for k, g in layout["tensors"].items() if k in names}
        (path / TERNARY_CONFIG).write_text(json.dumps(layout, indent=2))
        if not self._latent_state:
            return
        # the same model directory with the latent weights (and frozen / learned scales) in place of the ternary ones:
        # what a later QAT stage or a latent-space merge starts from
        import shutil

        from safetensors.torch import load_file, save_file

        target = path.parent / f"{path.name}-latent"
        target.mkdir(parents=True, exist_ok=True)
        try:
            for f in path.iterdir():
                if f.is_file() and f not in shards:
                    shutil.copy(f, target / f.name)
            renamed = saved_names(self._latent_state, on_disk)
            extra = {renamed.get(k, k): v for k, v in self._latent_state.items()}
            for i, shard in enumerate(shards):
                tensors = load_file(str(shard))
                for key in tensors:
                    if key in extra:
                        tensors[key] = extra.pop(key).to(tensors[key].dtype).contiguous()
                if i == len(shards) - 1:  # scale buffers / parameters have no slot in the folded files (a weight
                    # left over is the second name of a tied tensor, which the model file stores once)
                    tensors.update({k: v.contiguous() for k, v in extra.items() if k.endswith(_AUX_KEYS)})
                save_file(tensors, str(target / shard.name), metadata={"format": "pt"})
            (target / TERNARY_CONFIG).write_text(json.dumps({**layout, "export": "latent"}, indent=2))
        finally:
            self._latent_state = {}
        logger.info(f"ternary_qat: latent weights -> {target}")

    def state_dict(self) -> dict:
        return {"initial_codes": self._initial_codes}

    def load_state_dict(self, state: dict) -> None:
        self._initial_codes = dict(state.get("initial_codes", {}))


# FP4 / FP8 QAT (palingenesis.qat.fp), importable from here: palingenesis.qat:FloatQAT
_FLOAT_NAMES = ("FloatQAT", "FloatQuantLinear", "FloatQuantEmbedding")


def __getattr__(name: str):
    if name in _FLOAT_NAMES:
        from palingenesis.qat import fp

        return getattr(fp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
