"""Mixture-of-Experts training: router load balancing, routing statistics, routing replay.

A MoE layer sends every token to `top_k` of its experts, chosen by a small router. Trained
with the task loss alone the router can collapse onto a few experts (the others stop
receiving tokens, hence gradients), on long fine-tunes and in RL above all. The trainers
compute their losses from the hidden states themselves, so the auxiliary loss a transformers
model would add in its own forward (`output_router_logits`, `router_aux_loss_coef`) never
reaches them. `RouterBalancer` puts it back, for every trainer and memory path:

  load-balancing loss   coef * mean over MoE layers of  E * sum_i f_i * P_i
                        f_i: tokens routed to expert i / tokens (not differentiable, sums to top_k),
                        P_i: mean router probability of expert i (differentiable); E experts.
                        Switch Transformer (arXiv:2101.03961, eq. 4-6), with f measured over the
                        GLOBAL batch: every data-parallel rank and every micro-batch of the
                        optimizer step so far (Qwen, arXiv:2501.11873, algorithm 1: only the count
                        vector is communicated, each rank keeps its local P). Value at uniform
                        routing: top_k, its minimum.
  router z-loss         z_coef * mean over layers and tokens of (logsumexp(router logits))^2
                        (ST-MoE, arXiv:2202.08906, eq. 5): keeps the logits small.
  selection-bias update b_i += rate * sign(mean load - load_i) after every optimizer step, on
                        routers that carry a selection bias in their checkpoint (DeepSeek-V3's
                        `e_score_correction_bias`: added to the scores for the top-k choice only).
                        Auxiliary-loss-free balancing (arXiv:2408.15664; DeepSeek-V3,
                        arXiv:2412.19437, with rate 0.001).

How the loss reaches the routers. The trainers' forwards differ (one pass, activation
checkpointing, SeCO chunk by chunk with its own backward, a loss computed from hidden states),
so there is no single scalar to add a term to. Instead the gradient of the two losses with
respect to each layer's router logits is injected where the logits are produced: a forward hook
keeps the logits, a pre-hook on the experts routes the gate values through an identity whose
backward returns that gradient for the logits. f is a constant of the loss, and both losses are
sums over tokens, so this is exactly the gradient of the losses above. The values are computed
after the forward (`sync`: one all-reduce of the counts), before any backward runs.

Per micro-batch:

    balancer.begin(token_mask, grad_accum)   # real (non-padding) tokens of the forward
    ... forward ...
    balancer.sync()                          # global counts (collective); SeCO calls it itself
    ... backward ...
    balancer.step()                          # after the optimizer step: bias update, new buffer

Routing replay (RL). The rollout engine and the trainer do not pick the same experts for the
same tokens (other kernels and precision, other weights when rollouts are stale), which adds
noise to the importance ratio of every sampled token. With the experts the engine picked
(`begin(..., replay=indices)`; vLLM returns them with `enable_return_routed_experts`), the
trainer routes each token to those experts and computes their gate values from its own router
logits, so the router still trains (Rollout Routing Replay, arXiv:2510.11370). The share of
expert choices that differ is logged either way (moe/route_mismatch).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# The largest balance coefficient taken from a model config without being asked: Switch Transformer's.
# Some checkpoints ship placeholder values far above anything a model was trained with.
MAX_AUTO_BALANCE_COEF = 0.01
# DeepSeek-V3's bias update rate (arXiv:2412.19437, section 4.2)
DEFAULT_BIAS_UPDATE_RATE = 1e-3

# Columns of the token window the forward being run covers, and the weight of its tokens in the
# losses (SeCO: a chunk; SpaCO scales the chunks it visits). None: the whole micro-batch.
_WINDOW: contextvars.ContextVar[tuple[int, int, float] | None] = contextvars.ContextVar("pgs_moe_window", default=None)
# The balancer of the micro-batch being run, for code that only sees the model (SeCO).
_ACTIVE: contextvars.ContextVar["RouterBalancer | None"] = contextvars.ContextVar("pgs_moe_active", default=None)


@contextlib.contextmanager
def token_window(lo: int, hi: int, weight: float = 1.0):
    """The forward inside covers columns [lo, hi) of the micro-batch (a SeCO chunk)."""
    token = _WINDOW.set((lo, hi, weight))
    try:
        yield
    finally:
        _WINDOW.reset(token)


def routing_active() -> bool:
    """A `RouterBalancer` is measuring the micro-batch being run."""
    return _ACTIVE.get() is not None


def sync_routing() -> None:
    """`RouterBalancer.sync()` of the micro-batch being run, if any: for forward-backward routines
    that run their own backward (SeCO), called once every token has been routed."""
    balancer = _ACTIVE.get()
    if balancer is not None:
        balancer.sync()


# ─────────────────────────────────────────────────────────────── MoE layers of a model


@dataclass
class MoELayer:
    """One MoE block of a transformers model: its router and its experts module."""

    index: int
    name: str
    router: nn.Module
    experts: nn.Module
    num_experts: int
    # "softmax": probabilities are softmax(logits). "sigmoid": sigmoid(logits) normalised over the
    # experts (DeepSeek-V3 and the models that follow it).
    scoring: str
    # how the gate values follow from the scores of the chosen experts (found on the first forward):
    # normalised over the chosen experts or not, times a constant
    gate_normalised: bool | None = None
    gate_scale: float = 1.0

    @property
    def selection_bias(self) -> Tensor | None:
        """The router's selection bias buffer, when its checkpoint has one (read from the module:
        moving a model to a device replaces its buffers)."""
        bias = getattr(self.router, "e_score_correction_bias", None)
        return bias if isinstance(bias, Tensor) else None


def _is_experts(module: nn.Module) -> bool:
    """transformers' experts modules: every expert's weights stacked in 3-D parameters."""
    if not isinstance(getattr(module, "num_experts", None), int):
        return False
    down = getattr(module, "down_proj", None)
    return isinstance(down, Tensor) and down.ndim == 3


def find_moe_layers(model: nn.Module) -> list[MoELayer]:
    """The MoE blocks of a transformers model, in forward order: a module with an experts child
    (stacked expert weights) and a router child (`gate` or `router`, a [experts, hidden] weight).
    Wrappers (activation checkpointing, FSDP) do not matter. Models whose experts are separate
    modules (custom code) are not recognised."""
    modules = dict(model.named_modules())
    layers: list[MoELayer] = []
    for name, module in modules.items():
        if not _is_experts(module):
            continue
        parent = modules.get(name.rpartition(".")[0])
        if parent is None:
            continue
        router = next(
            (
                r
                for attr in ("gate", "router")
                if isinstance(r := getattr(parent, attr, None), nn.Module)
                and isinstance(getattr(r, "weight", None), Tensor)
                and r.weight.ndim == 2
                and r.weight.shape[0] == module.num_experts
            ),
            None,
        )
        if router is None:
            continue
        has_bias = isinstance(getattr(router, "e_score_correction_bias", None), Tensor)
        layers.append(
            MoELayer(
                index=len(layers),
                name=name.rpartition(".")[0],
                router=router,
                experts=module,
                num_experts=module.num_experts,
                scoring="sigmoid" if has_bias else "softmax",
            )
        )
    return layers


def is_moe(model: nn.Module) -> bool:
    return bool(find_moe_layers(model))


def config_is_moe(name_or_path: str, trust_remote_code: bool = False) -> bool:
    """Whether a checkpoint's config describes routed experts (read without loading the weights)."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    except (OSError, ValueError):
        return False
    configs = [config, getattr(config, "text_config", None)]
    return any(
        isinstance(n := getattr(c, attr, None), int) and n > 0
        for c in configs
        if c is not None
        for attr in ("num_experts", "num_local_experts", "n_routed_experts")
    )


def _full_precision(logits: Tensor) -> Tensor:
    """Router logits in at least fp32 (routers are computed in fp32: ST-MoE, section 3.2)."""
    return logits if logits.dtype in (torch.float32, torch.float64) else logits.float()


def router_scores(logits: Tensor, scoring: str) -> Tensor:
    """The router's scores [N, E] in fp32: softmax(logits), or sigmoid(logits) for sigmoid routers."""
    logits = _full_precision(logits)
    return torch.sigmoid(logits) if scoring == "sigmoid" else torch.softmax(logits, dim=-1)


def router_probabilities(logits: Tensor, scoring: str) -> Tensor:
    """The distribution over experts the balancing loss uses: softmax(logits), or the sigmoid scores
    normalised over the experts (DeepSeek-V3's s', arXiv:2412.19437 eq. 19)."""
    scores = router_scores(logits, scoring)
    return scores / scores.sum(-1, keepdim=True) if scoring == "sigmoid" else scores


def balance_loss(logits: Tensor, load: Tensor, scoring: str = "softmax") -> Tensor:
    """Per-token load-balancing loss E * sum_i load_i * p_i(token) for router logits [N, E] and the
    experts' load [E] (tokens routed to each expert / tokens). Its mean over the tokens that `load`
    was measured on is the Switch Transformer loss."""
    p = router_probabilities(logits, scoring)
    return p.shape[-1] * (p * load.to(p.dtype)).sum(-1)


def z_loss(logits: Tensor) -> Tensor:
    """Per-token router z-loss: logsumexp(logits)^2 (ST-MoE)."""
    return torch.logsumexp(_full_precision(logits), dim=-1) ** 2


# ─────────────────────────────────────────────────────────────── the balancer


class _InjectRouterGradient(torch.autograd.Function):
    """Identity on the gate values whose backward also returns d(balancing losses)/d(router logits)."""

    @staticmethod
    def forward(ctx, weights, logits, balancer, layer, mask, weight):
        ctx.save_for_backward(logits)
        ctx.balancer, ctx.layer, ctx.mask, ctx.weight = balancer, layer, mask, weight
        return weights.clone()

    @staticmethod
    def backward(ctx, grad_weights):
        (logits,) = ctx.saved_tensors
        grad_logits = ctx.balancer._router_gradient(ctx.layer, logits, ctx.mask, ctx.weight)
        return grad_weights, grad_logits, None, None, None, None


# columns after the per-expert counts and probability sums of a layer's row of statistics
_TOKENS, _ENTROPY, _Z, _REPLAYED, _MISMATCHED, _EXTRA = 0, 1, 2, 3, 4, 5


class RouterBalancer:
    """Balancing losses, bias updates, routing statistics and routing replay for a model's MoE
    layers (see the module docstring). One per trained model; `attach` returns None for a model
    without MoE layers or when everything is switched off."""

    def __init__(
        self,
        layers: list[MoELayer],
        *,
        balance_coef: float = 0.0,
        z_coef: float = 0.0,
        bias_update_rate: float = 0.0,
        log_layers: bool = False,
    ):
        self.layers = layers
        self.balance_coef = float(balance_coef)
        self.z_coef = float(z_coef)
        self.bias_update_rate = float(bias_update_rate)
        self.log_layers = log_layers
        self.max_experts = max(layer.num_experts for layer in layers)
        self._width = 2 * self.max_experts + _EXTRA
        self._handles: list = []
        self._logits: dict[int, Tensor] = {}
        self._active = False
        self._synced = False
        self._thread = 0
        self._mask: Tensor | None = None
        self._replay: Tensor | None = None
        self._grad_accum = 1
        self._local: Tensor | None = None  # this micro-batch, this rank: [layers, width]
        self._batch: Tensor | None = None  # this micro-batch, every rank
        self._buffer: Tensor | None = None  # the optimizer step so far, every rank
        self._last: Tensor | None = None  # the last complete optimizer step (for the metrics)
        self._load: Tensor | None = None  # [layers, max_experts]: f of the buffer
        self._loss_sums: Tensor | None = None  # balance loss, z-loss, micro-batches since the last metrics()
        self._warned: set[str] = set()
        self._tokens: int | None = None
        self._bias_masters: dict[int, Tensor] = {}
        # expert parallelism: ranks the experts are split over (contiguous blocks), for moe/rank_load_max
        self.expert_ranks = 1
        # tensor parallelism: ranks that route the same tokens (each counts them; the all-reduce divides them out)
        self.replicas = 1
        self._token: contextvars.Token | None = None
        for layer in layers:
            self._handles.append(layer.router.register_forward_hook(self._router_hook(layer)))
            self._handles.append(layer.experts.register_forward_pre_hook(self._experts_hook(layer)))

    # -- construction

    @classmethod
    def attach(cls, model: nn.Module, config=None, replay: bool = False) -> "RouterBalancer | None":
        """The balancer for `model` under a `MoEConfig` (palingenesis.config; None: its defaults).
        `replay`: routing replay will be used, so a balancer is needed even with balancing off."""
        from palingenesis.config import MoEConfig

        config = config or MoEConfig()
        layers = find_moe_layers(model)
        if not layers:
            return None
        if not config.balance:
            logger.info(f"MoE: {len(layers)} layers; router balancing and statistics are off (moe.balance: false)")
            return cls(layers) if replay else None
        model_coef = float(getattr(getattr(model, "config", None), "router_aux_loss_coef", 0.0) or 0.0)
        if config.balance_loss_coef >= 0:
            coef = config.balance_loss_coef
        else:
            coef = min(model_coef, MAX_AUTO_BALANCE_COEF)
            if model_coef > MAX_AUTO_BALANCE_COEF:
                logger.warning(
                    f"MoE: the model config's router_aux_loss_coef={model_coef} is above any published "
                    f"value; using {MAX_AUTO_BALANCE_COEF} (set moe.balance_loss_coef to choose)"
                )
        has_bias = any(layer.selection_bias is not None for layer in layers)
        if config.bias_update_rate >= 0:
            rate = config.bias_update_rate
            if rate > 0 and not has_bias:
                from palingenesis.config import ConfigError

                raise ConfigError(
                    "moe.bias_update_rate needs routers with a selection bias in their checkpoint (DeepSeek-V3's "
                    "e_score_correction_bias); this model has none, and a bias kept outside the checkpoint would "
                    "not reach inference. Use the balancing loss (moe.balance_loss_coef)."
                )
        else:
            rate = DEFAULT_BIAS_UPDATE_RATE if has_bias else 0.0
        balancer = cls(
            layers,
            balance_coef=coef,
            z_coef=config.z_loss_coef,
            bias_update_rate=rate,
            log_layers=config.log_layers,
        )
        experts = sorted({layer.num_experts for layer in layers})
        logger.info(
            f"MoE: {len(layers)} layers, {'/'.join(map(str, experts))} experts ({layers[0].scoring} routers); "
            f"balancing loss coef {coef:g} over the global batch, z-loss coef {config.z_loss_coef:g}, "
            f"selection-bias update rate {rate:g}"
        )
        return balancer

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def trains_routers(self) -> bool:
        return self.balance_coef > 0 or self.z_coef > 0

    # -- the micro-batch

    def begin(
        self,
        token_mask: Tensor | None = None,
        grad_accum: int = 1,
        replay: Tensor | None = None,
        tokens: int | None = None,
    ) -> None:
        """Before a micro-batch's forward.

        token_mask: [B, S], non-zero at real tokens, in the layout the model sees (None: all real).
                    Padding is left out of the loads, the losses and the statistics.
        grad_accum: micro-batches per optimizer step (the losses are averaged over them).
        replay:     [B, S, layers, top_k] expert indices to route with (a negative first index:
                    this token keeps the model's own choice), or None.
        tokens:     the real tokens the losses are averaged over, when the forwards of this
                    micro-batch are not all run before its first backward (prompt-sharing trees:
                    branches run and backpropagate one group at a time). The loads are then
                    those of the tokens routed before the first backward. One process only.
        """
        if self._active:
            self.end()
        device = self.layers[0].router.weight.device
        self._mask = token_mask.to(device).bool() if token_mask is not None else None
        self._replay = replay.to(device) if replay is not None else None
        self._grad_accum = max(int(grad_accum), 1)
        self._tokens = tokens
        self._local = torch.zeros(len(self.layers), self._width, dtype=torch.float64, device=device)
        self._batch = None
        self._logits.clear()
        self._active, self._synced = True, False
        self._thread = threading.get_ident()
        self._token = _ACTIVE.set(self)

    def sync(self) -> None:
        """Every token of the micro-batch has been routed: add up the ranks' counts (one all-reduce)
        and extend the optimizer step's buffer. Before the backward; a second call does nothing."""
        if not self._active or self._synced:
            return
        batch = self._local
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(batch, op=dist.ReduceOp.SUM)
            if self.replicas > 1:
                batch /= self.replicas
        self._batch = batch
        self._buffer = batch.clone() if self._buffer is None else self._buffer + batch
        e = self.max_experts
        tokens = self._buffer[:, 2 * e + _TOKENS].clamp(min=1.0)
        self._load = self._buffer[:, :e] / tokens[:, None]
        self._synced = True
        # the value the injected gradient differentiates: this micro-batch's P against the buffer's f
        batch_tokens = batch[:, 2 * e + _TOKENS].clamp(min=1.0)
        p = batch[:, e : 2 * e] / batch_tokens[:, None]
        sizes = torch.tensor([layer.num_experts for layer in self.layers], dtype=torch.float64, device=batch.device)
        sums = torch.stack(
            [
                ((self._load * p).sum(-1) * sizes).mean(),
                (batch[:, 2 * e + _Z] / batch_tokens).mean(),
                torch.ones((), dtype=torch.float64, device=batch.device),
            ]
        )
        self._loss_sums = sums if self._loss_sums is None else self._loss_sums + sums

    def end(self) -> None:
        """After the micro-batch's backward (or when it had none)."""
        self.sync()
        self._active = False
        self._mask = self._replay = None
        self._logits.clear()
        if self._token is not None:
            try:
                _ACTIVE.reset(self._token)
            except ValueError:  # ended from another context than it began in
                _ACTIVE.set(None)
            self._token = None

    @contextlib.contextmanager
    def batch(self, token_mask: Tensor | None = None, grad_accum: int = 1, replay: Tensor | None = None):
        """`begin` ... `end` around a micro-batch's forward and backward (call `sync` between them)."""
        self.begin(token_mask, grad_accum, replay)
        try:
            yield self
        finally:
            self.end()

    def step(self) -> None:
        """After the optimizer step: move the selection biases against the step's loads, start a new buffer."""
        if self._active:
            self.end()
        if self._buffer is None:
            return
        if self.bias_update_rate > 0:
            for layer in self.layers:
                bias = layer.selection_bias
                if bias is None:
                    continue
                counts = self._buffer[layer.index, : layer.num_experts]
                # an fp32 master: a bf16 buffer would round the small steps away
                master = self._bias_masters.setdefault(layer.index, bias.detach().float().clone())
                master += self.bias_update_rate * torch.sign(counts.mean() - counts).to(master)
                with torch.no_grad():
                    bias.copy_(master)
        self._last, self._buffer, self._load = self._buffer, None, None

    # -- hooks

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    def _router_hook(self, layer: MoELayer):
        def hook(module, args, output):
            if self._active and threading.get_ident() == self._thread:
                self._logits[layer.index] = output[0] if isinstance(output, (tuple, list)) else output

        # under model.compile the hooks stay Python (a graph break): they keep state between calls
        return torch.compiler.disable(hook)

    def _window(self, tokens: int, layer: MoELayer):
        """(mask [N] or None, replay [N, top_k] or None, loss weight) of the forward being run."""
        window = _WINDOW.get()
        lo, hi, weight = window if window is not None else (0, None, 1.0)
        mask = replay = None
        if self._mask is not None:
            mask = self._mask[:, lo:hi].reshape(-1)
            if mask.numel() != tokens:
                self._warn(
                    "mask",
                    f"MoE: the token mask covers {mask.numel()} tokens but {layer.name} routed {tokens}; "
                    "counting every token (padding included) for such forwards",
                )
                mask = None
        if self._replay is not None:
            replay = self._replay[:, lo:hi, layer.index].reshape(-1, self._replay.shape[-1])
            if replay.shape[0] != tokens:
                self._warn(
                    "replay", f"MoE: recorded routing for {replay.shape[0]} tokens, {tokens} routed: not replayed"
                )
                replay = None
        return mask, replay, weight

    def _experts_hook(self, layer: MoELayer):
        def hook(module, args):
            # only the thread that trains: a rollout thread may run the same model between micro-batches
            if not self._active or threading.get_ident() != self._thread or len(args) < 3:
                return None
            logits = self._logits.pop(layer.index, None)
            if logits is None:
                return None
            hidden, index, weights = args[:3]
            flat_logits = logits.reshape(-1, logits.shape[-1])
            mask, replay, weight = self._window(flat_logits.shape[0], layer)
            if layer.gate_normalised is None:
                _calibrate_gates(layer, flat_logits.detach(), index, weights.detach())
            replayed = mismatched = None
            if replay is not None and layer.gate_normalised is not None:
                index, weights, replayed, mismatched = _replay_routing(layer, flat_logits, index, weights, replay, mask)
            if not self._synced:
                self._measure(layer, flat_logits.detach(), index, mask, replayed, mismatched)
            if self.trains_routers and torch.is_grad_enabled() and logits.requires_grad:
                weights = _InjectRouterGradient.apply(weights, logits, self, layer, mask, weight)
            return (hidden, index, weights, *args[3:])

        return torch.compiler.disable(hook)

    @torch.no_grad()
    def _measure(self, layer: MoELayer, logits: Tensor, index: Tensor, mask, replayed, mismatched) -> None:
        e, row = layer.num_experts, self._local[layer.index]
        p = router_probabilities(logits, layer.scoring)
        entropy = -(p * torch.log(p.clamp(min=1e-30))).sum(-1)
        lse_sq = z_loss(logits)
        chosen = index.reshape(logits.shape[0], -1)
        if mask is not None:
            keep = mask.to(p.dtype)
            p, entropy, lse_sq = p * keep[:, None], entropy * keep, lse_sq * keep
            chosen = chosen[mask]
            tokens = keep.sum()
        else:
            tokens = torch.tensor(float(logits.shape[0]), device=logits.device)
        row[:e] += torch.bincount(chosen.reshape(-1), minlength=e)[:e]
        m = self.max_experts
        row[m : m + e] += p.sum(0)
        row[2 * m + _TOKENS] += tokens
        row[2 * m + _ENTROPY] += entropy.sum()
        row[2 * m + _Z] += lse_sq.sum()
        if replayed is not None:
            row[2 * m + _REPLAYED] += replayed
            row[2 * m + _MISMATCHED] += mismatched

    def _router_gradient(self, layer: MoELayer, logits: Tensor, mask: Tensor | None, weight: float) -> Tensor:
        """d/d(logits) of this layer's share of the losses, for this forward's tokens."""
        if not self._synced:
            if dist.is_initialized() and dist.get_world_size() > 1:
                raise RuntimeError(
                    "RouterBalancer.sync() must run after the forward and before the backward (it adds up the "
                    "ranks' expert counts, a collective every rank has to reach)."
                )
            self.sync()
        e = self.max_experts
        tokens = float(self._tokens or self._batch[layer.index, 2 * e + _TOKENS].clamp(min=1.0))
        scale = weight / (tokens * self._grad_accum * len(self.layers))
        with torch.enable_grad():
            z = _full_precision(logits.detach().reshape(-1, logits.shape[-1])).requires_grad_(True)
            per_token = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
            if self.balance_coef > 0:
                load = self._load[layer.index, : layer.num_experts]
                per_token = per_token + self.balance_coef * balance_loss(z, load, layer.scoring)
            if self.z_coef > 0:
                per_token = per_token + self.z_coef * z_loss(z)
            if mask is not None:
                per_token = per_token * mask.to(per_token.dtype)
            (grad,) = torch.autograd.grad(per_token.sum() * scale, z)
        return grad.reshape(logits.shape).to(logits.dtype)

    # -- statistics

    def metrics(self) -> dict[str, float]:
        """Scalars to log, over the last optimizer step's tokens on every rank.

        moe/load_max: the busiest expert's load relative to uniform (1 = balanced), worst layer;
        moe/load_cv: coefficient of variation of the expert loads, mean over layers;
        moe/dead_experts: share of experts that received no token, mean over layers;
        moe/router_entropy: mean entropy of the router distribution / log(experts), mean over layers;
        moe/balance_loss, moe/z_loss: the two losses (before their coefficients);
        moe/route_mismatch: with recorded routing, the share of recorded expert choices the trainer's
        own router would not have made; moe/rank_load_max: under expert parallelism, the busiest
        rank's share of the routed tokens relative to an even split, worst layer (what a placement of
        the experts by load could remove). With `log_layers`, the first four also per layer."""
        stats = self._buffer if self._buffer is not None else self._last
        if stats is None:
            return {}
        stats = stats.cpu()
        m = self.max_experts
        per_layer = []
        for layer in self.layers:
            row = stats[layer.index]
            counts = row[: layer.num_experts]
            tokens = float(row[2 * m + _TOKENS])
            if tokens <= 0:
                continue
            mean = float(counts.mean())
            per_layer.append(
                {
                    "load_max": float(counts.max()) / max(mean, 1e-12),
                    "load_cv": float(counts.std(unbiased=False)) / max(mean, 1e-12),
                    "dead_experts": float((counts == 0).double().mean()),
                    "router_entropy": float(row[2 * m + _ENTROPY]) / tokens / math.log(layer.num_experts),
                    "index": layer.index,
                }
            )
        if not per_layer:
            return {}
        out = {
            "moe/load_max": max(s["load_max"] for s in per_layer),
            "moe/load_cv": sum(s["load_cv"] for s in per_layer) / len(per_layer),
            "moe/dead_experts": sum(s["dead_experts"] for s in per_layer) / len(per_layer),
            "moe/router_entropy": sum(s["router_entropy"] for s in per_layer) / len(per_layer),
        }
        if self._loss_sums is not None:
            balance, z, n = self._loss_sums.tolist()
            out["moe/balance_loss"] = balance / n
            out["moe/z_loss"] = z / n
            self._loss_sums = None
        if self.expert_ranks > 1:
            worst = 1.0
            for layer in self.layers:
                per_rank = stats[layer.index, : layer.num_experts].view(self.expert_ranks, -1).sum(-1)
                if float(per_rank.sum()) > 0:
                    worst = max(worst, float(per_rank.max() / per_rank.mean()))
            out["moe/rank_load_max"] = worst
        replayed = float(stats[:, 2 * m + _REPLAYED].sum())
        if replayed > 0:
            out["moe/route_mismatch"] = float(stats[:, 2 * m + _MISMATCHED].sum()) / replayed
        if self.log_layers:
            for s in per_layer:
                for key in ("load_max", "load_cv", "dead_experts", "router_entropy"):
                    out[f"moe/layer_{s['index']}/{key}"] = s[key]
        return out

    def expert_loads(self) -> Tensor | None:
        """[layers, max experts]: each expert's share of the last optimizer step's routed tokens."""
        stats = self._buffer if self._buffer is not None else self._last
        if stats is None:
            return None
        counts = stats[:, : self.max_experts]
        return (counts / counts.sum(-1, keepdim=True).clamp(min=1.0)).float().cpu()

    @staticmethod
    def summary(metrics: dict) -> str:
        """The log line's MoE part ('' without MoE metrics)."""
        if "moe/load_max" not in metrics:
            return ""
        text = (
            f" moe[load_max={metrics['moe/load_max']:.2f} cv={metrics['moe/load_cv']:.2f} "
            f"dead={metrics['moe/dead_experts']:.3f} ent={metrics['moe/router_entropy']:.2f}"
        )
        if "moe/route_mismatch" in metrics:
            text += f" mismatch={metrics['moe/route_mismatch']:.3f}"
        return text + "]"


# ─────────────────────────────────────────────────────────────── routing replay


@torch.no_grad()
def _calibrate_gates(layer: MoELayer, logits: Tensor, index: Tensor, weights: Tensor) -> None:
    """How this model turns the chosen experts' scores into gate values: as they are or normalised
    over the chosen experts, times a constant (DeepSeek-V3's routed_scaling_factor). Read off one
    forward, so no per-architecture table has to follow transformers."""
    if logits.shape[0] == 0:
        return
    index = index.reshape(logits.shape[0], -1)
    weights = weights.reshape(index.shape).float()
    chosen = router_scores(logits, layer.scoring).gather(1, index).float()
    for normalised in (True, False):
        base = chosen / chosen.sum(-1, keepdim=True).clamp(min=1e-20) if normalised else chosen
        scale = float((weights.sum() / base.sum().clamp(min=1e-20)))
        if torch.allclose(base * scale, weights, rtol=2e-2, atol=1e-4):
            layer.gate_normalised, layer.gate_scale = normalised, scale
            return
    layer.gate_normalised = None


def gate_values(layer: MoELayer, logits: Tensor, index: Tensor) -> Tensor:
    """The gate values this layer gives the experts `index` [N, top_k] of each token, from router logits
    [N, E]: differentiable in the logits (the router trains under replay)."""
    chosen = router_scores(logits, layer.scoring).gather(1, index)
    if layer.gate_normalised:
        chosen = chosen / chosen.sum(-1, keepdim=True).clamp(min=1e-20)
    return chosen * layer.gate_scale


def _replay_routing(layer: MoELayer, logits: Tensor, index: Tensor, weights: Tensor, replay: Tensor, mask):
    """Route with the recorded experts where there are any: (indices, gate values, recorded expert
    choices, those among them the model's own router did not make)."""
    shape = index.shape
    own = index.reshape(logits.shape[0], -1)
    k = own.shape[1]
    recorded = replay[:, :k].to(own.dtype)
    use = (recorded[:, 0] >= 0) & (recorded.max(-1).values < layer.num_experts)
    if mask is not None:
        use = use & mask
    safe = torch.where(use[:, None], recorded, own)
    with torch.no_grad():
        same = (safe[:, :, None] == own[:, None, :]).any(-1)
        replayed = use.sum() * k
        mismatched = ((~same) & use[:, None]).sum()
    gates = gate_values(layer, logits, safe).to(weights.dtype)
    new_weights = torch.where(use[:, None], gates, weights.reshape(own.shape))
    return safe.reshape(shape), new_weights.reshape(weights.shape), replayed, mismatched


class RoutingRecorder:
    """Records the experts a model routes each token to: `with RoutingRecorder(model) as rec:` then
    `rec.indices()` is [tokens, layers, top_k] (tokens in the order the forwards saw them; with
    `rows`, the forwards concatenated along the sequence: [B, S, layers, top_k]). One byte per
    choice for up to 256 experts, else int16."""

    def __init__(self, model: nn.Module):
        self.layers = find_moe_layers(model)
        self.dtype = torch.uint8 if all(layer.num_experts <= 256 for layer in self.layers) else torch.int16
        self._chunks: list[list[Tensor]] = [[] for _ in self.layers]
        self._handles: list = []

    def __enter__(self):
        for layer in self.layers:

            def hook(module, args, layer=layer):
                if len(args) >= 3:
                    self._chunks[layer.index].append(args[1].detach().to(self.dtype).cpu())

            self._handles.append(layer.experts.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def indices(self, rows: int | None = None) -> Tensor:
        """[rows, S, layers, top_k] when `rows` is given (every forward had that many rows), else
        [tokens, layers, top_k]."""
        if not self.layers or not self._chunks[0]:
            return torch.empty(0, len(self.layers), 0, dtype=self.dtype)
        per_layer = []
        for chunks in self._chunks:
            if rows is None:
                per_layer.append(torch.cat([c.reshape(-1, c.shape[-1]) for c in chunks]))
            else:
                per_layer.append(torch.cat([c.reshape(rows, -1, c.shape[-1]) for c in chunks], dim=1))
        return torch.stack(per_layer, dim=-2)


def route_mismatch(recorded: Tensor, own: Tensor) -> float:
    """Share of the recorded expert choices [..., top_k] that are not among the model's own [..., top_k]
    (rows whose first recorded index is negative are skipped)."""
    use = recorded[..., 0] >= 0
    same = (recorded[..., :, None] == own[..., None, :]).any(-1)
    total = int(use.sum()) * recorded.shape[-1]
    return float(((~same) & use[..., None]).sum()) / total if total else 0.0


# ─────────────────────────────────────────────────────────────── tiny models (tests, ablations)

_TINY_COMMON = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=1024,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

# family -> (transformers model type, config overrides): 8 experts, 2 per token
TINY_MOE: dict[str, tuple[str, dict]] = {
    "qwen3_moe": (
        "qwen3_moe",
        {"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "norm_topk_prob": True},
    ),
    "qwen3_5_moe": (
        "qwen3_5_moe_text",
        {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
    ),
    "mixtral": ("mixtral", {"num_local_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32}),
    "gpt_oss": ("gpt_oss", {"num_local_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32}),
    "deepseek_v3": (
        "deepseek_v3",
        {
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "moe_intermediate_size": 32,
            "n_group": 2,
            "topk_group": 2,
            "first_k_dense_replace": 1,
            "kv_lora_rank": 16,
            "q_lora_rank": 32,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "head_dim": 8,
            "v_head_dim": 16,
        },
    ),
}


def tiny_moe(family: str = "qwen3_moe", seed: int = 0, router_std: float = 0.05, **overrides) -> nn.Module:
    """A small randomly initialised MoE model of a family in `TINY_MOE` (transformers' own classes
    and checkpoint layout), for tests and ablations: `tiny_moe("mixtral", num_hidden_layers=2)`.
    The routers get random weights of standard deviation `router_std` (some families initialise
    them to zero, which routes every token alike)."""
    import transformers
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    if family not in TINY_MOE:
        raise ValueError(f"unknown MoE family {family!r}; one of: {', '.join(TINY_MOE)}")
    model_type, family_overrides = TINY_MOE[family]
    config_cls = transformers.CONFIG_MAPPING[model_type]
    config = config_cls(**{**_TINY_COMMON, **family_overrides, **overrides})
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        model = getattr(transformers, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type])(config)
        for layer in find_moe_layers(model):
            nn.init.normal_(layer.router.weight, std=router_std)
    finally:
        torch.random.set_rng_state(state)
    return model
