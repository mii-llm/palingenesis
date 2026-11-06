"""Ternary quantisation-aware training, as a trainer plugin: a reference for writing your own.

    plugins:
      custom:
        - class: palingenesis.qat:TernaryQAT
          args:
            group_size: 128                 # absmean scale per 128 input weights (0: per tensor, -1: per row)
            exclude: ["lm_head", "embed"]   # nn.Linear names kept in full precision (regular expressions)
            quantize_embedding: false       # also the input embedding (and a tied head through it)
            quantize_head: false            # also the output head
            strength: {start: 0.0, end: 0.2}            # quantisation ramp: 0 = full precision, 1 = ternary
            grid_weight: {start: 0.0, end: 1.0, start_value: 0.0, end_value: 1.0e-4}
            latent_lr: 1.0e-4               # optimizer group of the latent weights (0: train.learning_rate)
            latent_eps: 1.0e-12             # Adam eps of that group (0: the optimizer's)
            clamp: 1.5                      # after each step, |w| <= clamp * scale (0: off)
            hidden_layers: [-1]             # match these decoder layers' outputs to the teacher's (MSE)
            hidden_weight: 1.0
            export: folded                  # final model: ternary values in the original layout (or latent)

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

This plugin is a starting point, not a recipe: its defaults are not tuned for any model.
"""

from __future__ import annotations

import logging
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from palingenesis.hooks import FeatureSpec, PluginContext, Ramp, TrainerPlugin

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
    def _init_quant(self, group_size: int, activation_bits: int):
        self.group_size = group_size
        self.activation_bits = activation_bits
        # a buffer (not a Python float): changing it never makes torch.compile recompile
        self.register_buffer("strength", torch.ones(()), persistent=False)

    def effective_weight(self) -> Tensor:
        w = self.weight
        w_q = quantize_weight(w, self.group_size)
        return w + self.strength.to(w.dtype) * (w_q - w).detach()


class TernaryLinear(_TernaryMixin, nn.Module):
    """nn.Linear with ternary weights in the forward and a full-precision latent weight."""

    def __init__(self, linear: nn.Linear, group_size: int = 128, activation_bits: int = 0):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight  # the same Parameter: names, ties and optimizer state carry over
        self.bias = linear.bias
        self._init_quant(group_size, activation_bits)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_bits:
            x = quantize_activations(x, self.activation_bits)
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group_size={self.group_size}, a{self.activation_bits or 16}"


class TernaryEmbedding(_TernaryMixin, nn.Module):
    """nn.Embedding whose rows are ternary (scales per group of the embedding dimension)."""

    def __init__(self, embedding: nn.Embedding, group_size: int = 128):
        super().__init__()
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.weight = embedding.weight
        self._init_quant(group_size, 0)

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
        strength=1.0,
        grid_weight=0.0,
        latent_lr: float = 0.0,
        latent_eps: float = 0.0,
        latent_weight_decay: float = 0.0,
        clamp: float = 1.5,
        hidden_layers: list | None = None,
        hidden_weight=1.0,
        hidden_loss: str = "mse",
        export: str = "folded",
        metrics_every: int = 50,
    ):
        if activation_bits not in (0, 4, 8):
            raise ValueError(f"activation_bits must be 0, 4 or 8 (got {activation_bits})")
        if export not in ("folded", "latent"):
            raise ValueError(f"export must be folded or latent (got {export!r})")
        self.group_size = group_size
        self.include = [re.compile(p) for p in (include or [r".*"])]
        self.exclude = [re.compile(p) for p in (exclude if exclude is not None else [r"lm_head", r"embed"])]
        self.quantize_embedding = quantize_embedding
        self.quantize_head = quantize_head
        self.activation_bits = activation_bits
        self.strength = Ramp.parse(strength)
        self.grid_weight = Ramp.parse(grid_weight)
        self.latent_lr, self.latent_eps, self.latent_weight_decay = latent_lr, latent_eps, latent_weight_decay
        self.clamp = clamp
        self.hidden_layers = list(hidden_layers or [])
        self.hidden_weight = Ramp.parse(hidden_weight)
        self.hidden_loss = hidden_loss
        self.export = export
        self.metrics_every = max(1, metrics_every)
        self.modules: dict[str, nn.Module] = {}
        self._initial_codes: dict[str, Tensor] = {}
        self._grid_value = 0.0
        self._tied_keys: tuple[str, str] | None = None

    # -- model

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
                    replacements.append((name, TernaryLinear(module, g, 0)))
                continue
            if module is embed:
                if self.quantize_embedding and isinstance(module, nn.Embedding):
                    g = self._group_size_for(name, module.embedding_dim)
                    replacements.append((name, TernaryEmbedding(module, g)))
                continue
            if type(module) is nn.Linear and any(p.search(name) for p in self.include):
                if any(p.search(name) for p in self.exclude):
                    continue
                g = self._group_size_for(name, module.in_features)
                replacements.append((name, TernaryLinear(module, g, self.activation_bits)))
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
        override = {"params": latents, "weight_decay": self.latent_weight_decay}
        if self.latent_lr:
            override["lr"] = self.latent_lr
        if self.latent_eps:
            override["eps"] = self.latent_eps
        return [override]

    def _unique_modules(self) -> list[nn.Module]:
        seen, out = set(), []
        for m in self.modules.values():
            if id(m.weight) not in seen:
                seen.add(id(m.weight))
                out.append(m)
        return out

    # -- training

    def on_train_start(self, model: nn.Module, ctx: PluginContext) -> None:
        self._set_strength(ctx)
        for name, m in self.modules.items():
            _, _, codes = _local_codes(m.weight, m.group_size)
            self._initial_codes[name] = codes.to(torch.int8).cpu()

    def _set_strength(self, ctx: PluginContext) -> None:
        value = float(self.strength(ctx.progress))
        for m in self.modules.values():
            m.strength.fill_(value)

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
            g, scale, codes = _local_codes(p, m.group_size)
            residual = g - codes * scale  # W - W_q
            _local(p.grad).add_((2.0 * lam * residual).reshape(_local(p.grad).shape).to(p.grad.dtype))
            value += lam * float(residual.float().pow(2).sum())
        self._grid_value = value

    @torch.no_grad()
    def on_after_optimizer_step(self, model: nn.Module, optimizer, ctx: PluginContext) -> None:
        if self.clamp <= 0:
            return
        for m in self._unique_modules():
            g, scale, _ = _local_codes(m.weight, m.group_size)
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
            _, _, codes = _local_codes(m.weight, m.group_size)
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

    # -- export and resume

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict:
        if self.export == "latent":
            return state
        out = dict(state)
        names = {f"{name}.weight": m for name, m in self.modules.items()}
        for key, m in names.items():
            if key in state:
                out[key] = quantize_weight(state[key].float(), m.group_size).to(state[key].dtype)
        # tied embedding / head: both keys get the quantised matrix (the trained model used one tensor)
        if self._tied_keys:
            a, b = self._tied_keys
            source = a if a in names else b if b in names else None
            if source is not None and source in out:
                for key in (a, b):
                    if key in out:
                        out[key] = out[source]
        return out

    def state_dict(self) -> dict:
        return {"initial_codes": self._initial_codes}

    def load_state_dict(self, state: dict) -> None:
        self._initial_codes = dict(state.get("initial_codes", {}))
