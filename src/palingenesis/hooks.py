"""Trainer plugins: user code that runs inside `pgs train` (SFT, off-policy distillation).

A plugin is a class with any of the hook methods of `TrainerPlugin`; every hook is optional
and a no-op by default. Plugins are listed in the config (`plugins.custom`) or passed to
`train(config, plugins=[...])`, and run in the order given.

    plugins:
      custom:
        - my_package.qat:TernaryQAT                    # module:Class
        - {class: ./my_plugin.py:MyPlugin, args: {lam: 0.1}}
        - my-registered-plugin                         # entry point (group "palingenesis.plugins")

Where each hook runs in a training step, in order:

    setup(ctx)                         once, after the config is validated, before any model loads
    transform_model(model, ctx)        once, after the student loads, before float8 / activation
                                       checkpointing / FSDP / compile (substitute layers, add modules)
    feature_requests(ctx)              once: student/teacher module outputs to capture and match
    param_groups(model, ctx)           once: optimizer overrides for some parameters (lr, eps, weight decay)
    on_train_start(model, ctx)         once, before the first step (after resume)
    on_forward_start(model, batch, ctx)   every micro-batch, before the forward
    transform_targets(targets, batch, ctx)  every micro-batch with a teacher: change its targets
    loss(inputs, ctx)                  every micro-batch: extra loss terms
    on_before_optimizer_step(model, optimizer, ctx)  after the last backward of a step, before clipping
    on_after_optimizer_step(model, optimizer, ctx)   after optimizer.step()
    metrics(model, ctx)                at logging steps: scalars to log (as plugins/<name>/<key>)
    export_state_dict(state, model, ctx)   when a model is exported (final, best): fold / rename weights
    state_dict() / load_state_dict()   plugin state saved with every checkpoint, restored on resume
    on_train_end(model, ctx)           once, before the final save

Scaling of loss terms: `loss()` returns a scalar (or a dict of named scalars) for this micro-batch,
computed on this rank, e.g. a mean over the rank's tokens. The trainer adds it to the micro-batch
loss multiplied by 1 / (gradient accumulation steps x data-parallel ranks): gradients are summed over
the accumulation window and over ranks, so a per-rank mean becomes a mean over the step.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "palingenesis.plugins"


# ─────────────────────────────────────────────────────────────── context objects


@dataclass
class PluginContext:
    """What a plugin can see about the run. `step` is the number of optimizer steps done so far."""

    config: Any
    total_steps: int = 0
    step: int = 0
    rank: int = 0
    world_size: int = 1
    dp_world_size: int = 1
    device: torch.device | None = None
    is_fsdp: bool = False
    output_dir: str = ""
    # set by the trainer once the teachers exist (off-policy distillation), else None
    distiller: Any = None
    _logged: dict[str, float] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Fraction of training done, in [0, 1]."""
        return min(1.0, self.step / self.total_steps) if self.total_steps > 0 else 0.0

    def log(self, key: str, value: float) -> None:
        """Record a scalar for the next logging step (the last value wins)."""
        self._logged[key] = float(value)

    def pop_logged(self) -> dict[str, float]:
        out, self._logged = self._logged, {}
        return out


@dataclass
class FeatureSpec:
    """Match a student module's output to a teacher module's output at the trained positions.

    `student` / `teacher`: a decoder-layer index (int, negative counts from the end) or a module
    path (`model.layers.3.mlp`; matched as a suffix of the module names, so wrappers added by
    activation checkpointing or FSDP do not matter). The module's output (its first element when it
    returns a tuple) must be [batch, seq, dim].

    loss: "mse" (mean squared error over the feature dimension), "nmse" (MSE of the L2-normalised
    vectors), "cosine" (1 - cosine similarity), "lens_kl" (forward KL between the two outputs read through
    each model's own final norm and output head, frozen for this term). `projection`: a learnable linear map from the student
    dimension to the teacher's, created when the dimensions differ ("auto"), always ("linear") or never
    ("none"; the dimensions must match). `weight` may be a callable of training progress (a ramp).
    `via`: the online teacher whose module is read (default: the first online teacher).
    """

    student: int | str
    teacher: int | str
    weight: float | Callable[[float], float] = 1.0
    loss: str = "mse"
    projection: str = "auto"
    name: str = ""
    via: str = ""

    def weight_at(self, progress: float) -> float:
        return float(self.weight(progress)) if callable(self.weight) else float(self.weight)


@dataclass
class LossInputs:
    """What `TrainerPlugin.loss` receives for one micro-batch.

    hidden:      the student's final hidden states [B, T, H] (with gradient), None on paths that do
                 not expose them
    labels:      shifted labels [B, T] (position t predicts token t+1; -100 = no loss)
    rows:        flat indices into hidden.reshape(-1, H) of the trained positions
    features:    {name: (student [N, Ds], teacher [N, Dt])} at `rows`, for the FeatureSpecs requested
                 (student side with gradient; teacher side frozen)
    teacher:     the micro-batch's teacher targets (palingenesis.distill.TeacherTargets) or None
    loss_denom:  the global trained-token count x gradient accumulation steps: a term summed over this
                 rank's tokens and divided by it matches the trainer's own normalisation (then divide the
                 returned value by ctx's scaling back: see `token_sum_scale`)
    """

    model: nn.Module
    batch: dict
    hidden: torch.Tensor | None
    labels: torch.Tensor
    rows: torch.Tensor
    loss_denom: float
    grad_accum: int
    features: dict = field(default_factory=dict)
    teacher: Any = None
    head: nn.Module | None = None

    @property
    def token_sum_scale(self) -> float:
        """Multiply a sum over this rank's trained tokens by this to get a value that, after the
        trainer's 1 / (grad accumulation x ranks) scaling, is normalised exactly like the main loss."""
        return self.grad_accum / self.loss_denom


# ─────────────────────────────────────────────────────────────── the plugin base


class TrainerPlugin:
    """Base class of trainer plugins: override the hooks you need (see the module docstring)."""

    #: name used in logs and metric keys (default: the class name)
    name: str = ""

    def setup(self, ctx: PluginContext) -> None:
        """Validate or adjust the config (ctx.config) before anything loads."""

    def transform_model(self, model: nn.Module, ctx: PluginContext) -> nn.Module | None:
        """Change the freshly loaded student: substitute layers, wrap the embedding or the head,
        freeze or unfreeze parameters, add modules. Return a replacement model or None (in place).
        Runs before FSDP wrapping, so new parameters are sharded and checkpointed like the rest."""

    def feature_requests(self, ctx: PluginContext) -> list[FeatureSpec]:
        """Student / teacher module outputs to capture (needs a teacher: distill.enabled)."""
        return []

    def param_groups(self, model: nn.Module, ctx: PluginContext) -> list[dict]:
        """Optimizer overrides: [{"params": [tensors] or "names": regex, "lr": ..., "eps": ...,
        "weight_decay": ..., "betas": ...}]. Matching parameters move to a group of their own with
        these settings (the learning-rate schedule still applies, relative to the group's lr)."""
        return []

    def on_train_start(self, model: nn.Module, ctx: PluginContext) -> None:
        """Before the first step (after a resume restored the weights): calibration, statistics."""

    def on_forward_start(self, model: nn.Module, batch: dict, ctx: PluginContext) -> None:
        """Before each micro-batch's forward."""

    def transform_targets(self, targets: Any, batch: dict, ctx: PluginContext) -> Any:
        """Change the teacher targets of a micro-batch (palingenesis.distill.TeacherTargets): reweight
        tokens, sharpen, add targets. Return the new targets or None (changed in place)."""

    def loss(self, inputs: LossInputs, ctx: PluginContext) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Extra loss terms for this micro-batch (scaling: see the module docstring)."""

    def on_before_optimizer_step(self, model: nn.Module, optimizer, ctx: PluginContext) -> None:
        """Gradients of the step are complete (not clipped yet): mask, add regulariser gradients."""

    def on_after_optimizer_step(self, model: nn.Module, optimizer, ctx: PluginContext) -> None:
        """After optimizer.step(): clamp latent weights, update statistics."""

    def metrics(self, model: nn.Module, ctx: PluginContext) -> dict[str, float]:
        """Scalars to log at a logging step."""
        return {}

    def export_state_dict(self, state: dict, model: nn.Module, ctx: PluginContext) -> dict | None:
        """Change the state dict of an exported model (final / best; not the resume checkpoints):
        fold quantised weights, drop training-only tensors. `state` is a full (gathered) state dict
        under the architecture's names. Return the new dict or None (changed in place)."""

    def state_dict(self) -> dict:
        """Plugin state to save with each checkpoint (picklable)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore what `state_dict` returned."""

    def on_train_end(self, model: nn.Module, ctx: PluginContext) -> None:
        """After the last step, before the final save."""

    # -- helpers

    @property
    def plugin_name(self) -> str:
        return self.name or type(self).__name__

    @classmethod
    def overrides(cls, hook: str) -> bool:
        """Whether this plugin class implements `hook` (the trainer refuses hooks a path cannot run)."""
        return getattr(cls, hook) is not getattr(TrainerPlugin, hook)


# ─────────────────────────────────────────────────────────────── schedules


@dataclass
class Ramp:
    """A value that moves from `start_value` to `end_value` between two fractions of training.

    shape: "linear", "cosine" (smooth start and end) or "step" (jumps at `end`). Before `start` the
    value is `start_value`, after `end` it is `end_value`.

        strength = Ramp(start=0.0, end=0.3)            # 0 -> 1 over the first 30% of training
        lam = Ramp(0.1, 0.5, 0.0, 1e-3, "cosine")
        lam(ctx.progress)
    """

    start: float = 0.0
    end: float = 1.0
    start_value: float = 0.0
    end_value: float = 1.0
    shape: str = "linear"

    def __post_init__(self):
        if self.shape not in ("linear", "cosine", "step"):
            raise ValueError(f"Ramp shape must be linear, cosine or step, got {self.shape!r}")
        if not 0.0 <= self.start <= self.end <= 1.0:
            raise ValueError(f"Ramp needs 0 <= start <= end <= 1, got start={self.start} end={self.end}")

    def __call__(self, progress: float) -> float:
        if progress <= self.start and self.end > self.start:
            return self.start_value
        if progress >= self.end:
            return self.end_value
        if self.shape == "step":
            return self.start_value
        x = (progress - self.start) / (self.end - self.start)
        if self.shape == "cosine":
            x = 0.5 - 0.5 * math.cos(math.pi * x)
        return self.start_value + (self.end_value - self.start_value) * x

    @classmethod
    def parse(cls, spec) -> Ramp:
        """A Ramp from a number (constant), a dict of its fields, or a Ramp."""
        if isinstance(spec, Ramp):
            return spec
        if isinstance(spec, (int, float)):
            return cls(0.0, 0.0, float(spec), float(spec))
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError(f"expected a number, a dict or a Ramp, got {spec!r}")


# ─────────────────────────────────────────────────────────────── loading


class PluginError(Exception):
    """A plugin cannot be loaded or used with this configuration."""


def load_object(spec: str) -> Any:
    """`module:attr` (importable), `path/to/file.py:attr`, or an entry point name in the
    `palingenesis.plugins` group."""
    module_name, sep, attr = spec.rpartition(":")
    if not sep:
        from importlib.metadata import entry_points

        found = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == spec]
        if not found:
            raise PluginError(
                f"{spec!r}: not 'module:Class', 'file.py:Class' or an installed entry point of group "
                f"{ENTRY_POINT_GROUP!r}"
            )
        return found[0].load()
    if not module_name or not attr:
        raise PluginError(f"{spec!r}: expected 'module:name' or 'path/to/file.py:name'")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        if not path.exists():
            raise PluginError(f"{spec!r}: {path} does not exist")
        mod_name = f"pgs_plugin_{path.stem}"
        module = sys.modules.get(mod_name)
        if module is None or getattr(module, "__file__", None) != str(path):
            loader_spec = importlib.util.spec_from_file_location(mod_name, path)
            if loader_spec is None or loader_spec.loader is None:
                raise PluginError(f"cannot load {path}")
            module = importlib.util.module_from_spec(loader_spec)
            sys.modules[mod_name] = module
            loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise PluginError(f"{module_name} has no {attr!r}") from None


def build_plugin(spec) -> TrainerPlugin:
    """A plugin from a config entry: "module:Class", {"class": "module:Class", "args": {...}},
    or an already built object."""
    if isinstance(spec, TrainerPlugin):
        return spec
    args: dict = {}
    if isinstance(spec, dict):
        unknown = set(spec) - {"class", "args"}
        if "class" not in spec or unknown:
            raise PluginError(f"plugin entry {spec!r}: expected {{class: module:Class, args: {{...}}}}")
        args = dict(spec.get("args") or {})
        spec = spec["class"]
    if not isinstance(spec, str):
        raise PluginError(f"plugin entry {spec!r}: expected a string or a mapping")
    obj = load_object(spec)
    plugin = obj(**args) if isinstance(obj, type) or callable(obj) else obj
    if not isinstance(plugin, TrainerPlugin):
        # duck typing is fine, but every hook must exist: wrap missing ones with the defaults
        missing = [h for h in _HOOKS if not hasattr(plugin, h)]
        if missing:
            raise PluginError(
                f"{spec}: {type(plugin).__name__} is not a palingenesis.hooks.TrainerPlugin and lacks {missing}"
            )
    return plugin


_HOOKS = (
    "setup",
    "transform_model",
    "feature_requests",
    "param_groups",
    "on_train_start",
    "on_forward_start",
    "transform_targets",
    "loss",
    "on_before_optimizer_step",
    "on_after_optimizer_step",
    "metrics",
    "export_state_dict",
    "state_dict",
    "load_state_dict",
    "on_train_end",
)


class PluginManager:
    """The run's plugins, called in order."""

    def __init__(self, plugins: list[TrainerPlugin], ctx: PluginContext):
        self.plugins = list(plugins)
        self.ctx = ctx
        names = [self._name(p) for p in self.plugins]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise PluginError(f"two plugins share the name(s) {sorted(dupes)}: set `name` on one of them")

    @staticmethod
    def _name(plugin) -> str:
        return getattr(plugin, "name", "") or type(plugin).__name__

    def __bool__(self) -> bool:
        return bool(self.plugins)

    def __iter__(self):
        return iter(self.plugins)

    def implementing(self, hook: str) -> list:
        """The plugins whose class overrides `hook`."""
        out = []
        for p in self.plugins:
            fn = getattr(type(p), hook, None)
            if fn is not None and fn is not getattr(TrainerPlugin, hook, None):
                out.append(p)
        return out

    # -- dispatch

    def setup(self) -> None:
        for p in self.plugins:
            p.setup(self.ctx)

    def transform_model(self, model: nn.Module) -> nn.Module:
        for p in self.plugins:
            out = p.transform_model(model, self.ctx)
            if out is not None:
                model = out
        return model

    def feature_requests(self) -> list[FeatureSpec]:
        specs = []
        for p in self.plugins:
            for i, s in enumerate(p.feature_requests(self.ctx) or []):
                if not s.name:
                    s.name = f"{self._name(p)}/{i}"
                specs.append(s)
        return specs

    def param_groups(self, model: nn.Module) -> list[dict]:
        out = []
        for p in self.plugins:
            out.extend(p.param_groups(model, self.ctx) or [])
        return out

    def on_train_start(self, model) -> None:
        for p in self.plugins:
            p.on_train_start(model, self.ctx)

    def on_forward_start(self, model, batch) -> None:
        for p in self.plugins:
            p.on_forward_start(model, batch, self.ctx)

    def transform_targets(self, targets, batch):
        for p in self.plugins:
            out = p.transform_targets(targets, batch, self.ctx)
            if out is not None:
                targets = out
        return targets

    def loss(self, inputs: LossInputs) -> tuple[torch.Tensor | None, dict[str, float]]:
        """The sum of the plugins' loss terms (unscaled) and their values for logging."""
        total, logged = None, {}
        for p in self.plugins:
            out = p.loss(inputs, self.ctx)
            if out is None:
                continue
            terms = out if isinstance(out, dict) else {"loss": out}
            for key, value in terms.items():
                if not torch.is_tensor(value) or value.numel() != 1:
                    raise PluginError(f"plugin {self._name(p)}: loss term {key!r} must be a scalar tensor")
                total = value.reshape(()) if total is None else total + value.reshape(())
                logged[f"plugins/{self._name(p)}/{key}"] = float(value.detach())
        return total, logged

    def on_before_optimizer_step(self, model, optimizer) -> None:
        for p in self.plugins:
            p.on_before_optimizer_step(model, optimizer, self.ctx)

    def on_after_optimizer_step(self, model, optimizer) -> None:
        for p in self.plugins:
            p.on_after_optimizer_step(model, optimizer, self.ctx)

    def metrics(self, model) -> dict[str, float]:
        out = {}
        for p in self.plugins:
            for k, v in (p.metrics(model, self.ctx) or {}).items():
                out[f"plugins/{self._name(p)}/{k}"] = float(v)
        out.update(self.ctx.pop_logged())
        return out

    def export_state_dict(self, state: dict, model) -> dict:
        for p in self.plugins:
            out = p.export_state_dict(state, model, self.ctx)
            if out is not None:
                state = out
        return state

    def state_dict(self) -> dict:
        return {self._name(p): p.state_dict() for p in self.plugins}

    def load_state_dict(self, state: dict) -> None:
        for p in self.plugins:
            if self._name(p) in state:
                p.load_state_dict(state[self._name(p)])

    def on_train_end(self, model) -> None:
        for p in self.plugins:
            p.on_train_end(model, self.ctx)


def load_plugins(config_specs: list, extra: list | None = None) -> list[TrainerPlugin]:
    """Plugins from the config's `plugins.custom` entries, then the ones passed in Python."""
    plugins = [build_plugin(s) for s in (config_specs or [])]
    plugins.extend(build_plugin(p) for p in (extra or []))
    return plugins


# ─────────────────────────────────────────────────────────────── optimizer groups


def apply_param_groups(optimizer: torch.optim.Optimizer, model: nn.Module, overrides: list[dict]) -> int:
    """Move the parameters each override selects into a new group with its settings.

    Call before the first optimizer step (the state must be empty). A moved parameter keeps every
    setting of the group it came from except those the override sets. Returns the number of
    parameters moved. Raises when an override selects nothing (a typo would silently do nothing).
    """
    import re

    if not overrides:
        return 0
    if any(optimizer.state.values()):
        raise PluginError("parameter groups can only change before the first optimizer step")
    names = {id(p): n for n, p in model.named_parameters()}
    owner = {}
    for gi, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            owner[id(p)] = gi
    moved_total = 0
    for override in overrides:
        settings = {k: v for k, v in override.items() if k not in ("params", "names")}
        if "params" in override:
            chosen = [p for p in override["params"] if id(p) in owner]
        elif "names" in override:
            pattern = re.compile(override["names"])
            chosen = [
                p
                for pid, p in ((id(p), p) for g in optimizer.param_groups for p in g["params"])
                if pattern.search(names.get(pid, ""))
            ]
        else:
            raise PluginError(f"param group override {override!r} needs 'params' or 'names'")
        if not chosen:
            raise PluginError(f"param group override {settings} selects no trainable parameter")
        by_group: dict[int, list] = {}
        for p in chosen:
            by_group.setdefault(owner[id(p)], []).append(p)
        for gi, params in by_group.items():
            source = optimizer.param_groups[gi]
            ids = {id(p) for p in params}
            source["params"] = [p for p in source["params"] if id(p) not in ids]
            new = {k: v for k, v in source.items() if k != "params"}
            new.update(settings)
            if "lr" in settings:
                new["initial_lr"] = settings["lr"]
            new["params"] = params
            optimizer.add_param_group(new)
            for p in params:
                owner[id(p)] = len(optimizer.param_groups) - 1
            moved_total += len(params)
    optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
    return moved_total
