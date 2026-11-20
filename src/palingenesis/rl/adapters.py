"""LoRA in the RL trainer (the config's `lora:` section): the policy's base weights stay frozen and low-rank
adapters train (palingenesis.lora; the same adapted layer, folding and export as `pgs train`).

    lora:
      rank: 16            # 0 = off: full-weight RL
      alpha: 32

What changes in a run with adapters:

  optimizer    steps the adapters only, which stay float32 parameters (megabytes: no bf16 copy with an fp32 master
               as full weights have, so DoRA's magnitudes and small updates are never rounded in the module); the
               base keeps no gradient and no optimizer state, and on one GPU it is held in bf16 only.
  rollouts     the engine generates with the folded weights W + scaling * B A (DoRA: rows rescaled), pushed through
               the same weight path as full-weight RL after every step (`engine_parameters`). The engine needs no
               LoRA support and runs at its plain-model speed. Folding makes new tensors: the trainer's base weights
               are never written. The fold reads the parameters the trainer computes with and rounds once into the
               engine's bf16, so the part of the delta below half an ulp of its weight is not in the sampler: the
               importance ratio (and policy/mismatch_k3) sees that difference like any other sampler/trainer
               mismatch, and `lora/push_survival` reports how much of the delta the pushed weights carry (a
               warning is logged once when it is still below SURVIVAL_WARN after SURVIVAL_WARN_AFTER pushes).
  reference    loss.kl_ref: start is the policy with its adapters switched off (`disabled`): no second copy of the
               model. Another checkpoint as reference, or kl_ref_reset, keeps the frozen copy.
  checkpoints  every checkpoint holds the folded model (a plain checkpoint of the policy's architecture, float32)
               and `adapter/` (PEFT layout, float32). A run resumes from the policy's base plus the adapter.

Per-role adapters (several adapters on one base, one per agent role) would build on the same three operations:
fold for the engine, disable for the reference, load to resume.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from palingenesis.checkpoint import hf_state_dict
from palingenesis.hooks import PluginContext
from palingenesis.lora import ADAPTER_KEYS, LoRA, LoRALinear, fold_weight
from palingenesis.precision import SurvivalMeter, cast_state_dict

logger = logging.getLogger(__name__)

NAMED_DIR = "adapters"  # <checkpoint>/adapters/<name>: the named adapters of a run with several
ADAPTER_DIR = "adapter"  # inside every checkpoint of a run with adapters
SURVIVAL_EVERY = 50  # weight pushes between measurements of what the engine's dtype keeps of the delta
# A delta the engine's bf16 weights mostly round away means the engine still samples close to the base while the
# trainer scores the adapted policy. That is expected in the first steps (B starts at 0). 50 optimizer steps at an
# adapter learning rate move B by about 50 x the rate, which at the usual scalings is several half-ulps of a
# typical weight: a share still under one half by then says the rate or the scaling is too small for the push.
SURVIVAL_WARN = 0.5
SURVIVAL_WARN_AFTER = 50
PEFT_CONFIG_FILE = "adapter_config.json"


def hf_state_dict_items(named: Iterable[tuple[str, Tensor]]) -> Iterator[tuple[str, Tensor]]:
    """`named` with the wrapper prefixes removed from every name (checkpoint.hf_state_dict, one pair at a time)."""
    for name, tensor in named:
        yield next(iter(hf_state_dict({name: tensor}).items()))


def is_adapter_checkpoint(path: str | Path) -> bool:
    """Whether the checkpoint at `path` was saved by a run with adapters."""
    return (Path(path) / ADAPTER_DIR / "adapter_model.safetensors").exists()


class PolicyAdapters:
    """The policy's LoRA adapters for one RL run. `config` is the RLConfig; rank / world / fsdp describe the
    run's layout (palingenesis.rl.parallel)."""

    def __init__(self, config: Any, device: str = "cpu", rank: int = 0, world: int = 1, fsdp: bool = False):
        c = config.lora
        self.plugin = LoRA(
            rank=c.rank,
            alpha=c.alpha,
            rslora=c.rslora,
            dora=c.dora,
            lr_ratio=c.lr_ratio,
            targets=list(c.targets) or None,
            exclude=list(c.exclude),
        )
        self.ctx = PluginContext(
            config=config,
            total_steps=config.train.steps,
            rank=rank,
            world_size=world,
            dp_world_size=world,
            device=torch.device(device),
            is_fsdp=fsdp,
            output_dir=config.train.output_dir,
        )
        self.lr_ratio = float(c.lr_ratio)
        # further adapters on the same frozen base, one per name: roles of a scenario that have their own policy
        # (roles.<name>.adapter). "" is the default adapter; `default_trains`: some role (or a plain run) uses it.
        self.names: list[str] = []
        self.default_trains = True
        self.resumed = False  # the adapters came from a checkpoint (apply)
        self.warned = False  # the low push_survival warning, once
        self.pushes = 0
        self.push_survival: float | None = None

    # ------------------------------------------------------------------ model

    def apply(self, model: nn.Module, resume: str | Path | None = None) -> nn.Module:
        """Freeze `model` (the policy's base, as loaded) and wrap its targeted linears; with `resume`, load that
        checkpoint's adapters. Before the model is moved, wrapped for checkpointing or sharded."""
        model = self.plugin.transform_model(model, self.ctx)
        for name in self.names:
            self.plugin.add_adapter(name, int(self.ctx.config.train.seed))
        if not self.default_trains:  # every trained role has its own adapter: the default one stays at zero
            for m in self.plugin.modules.values():
                m.lora_A.requires_grad_(False)
                m.lora_B.requires_grad_(False)
        if resume is not None:
            self._check_saved_settings(Path(resume) / ADAPTER_DIR)
            self.plugin.load_adapter(Path(resume) / ADAPTER_DIR)
            for name in self.names:
                self.plugin.load_adapter(Path(resume) / NAMED_DIR / name, name)
        self.resumed = resume is not None
        return model

    def _check_saved_settings(self, path: Path) -> None:
        """Refuse a checkpoint whose adapters were trained with other settings or on another base: the tensors
        would load (same shapes) and compute another policy."""
        from palingenesis.rl.config import RLConfigError

        file = path / PEFT_CONFIG_FILE
        if not file.exists():
            raise RLConfigError(f"{path} has no {PEFT_CONFIG_FILE}: not a checkpoint of a run with adapters.")
        saved = json.loads(file.read_text())
        p = self.plugin
        now = {
            "r": p.rank,
            "lora_alpha": p.alpha,
            "use_rslora": p.rslora,
            "use_dora": p.dora,
            "base_model_name_or_path": self.ctx.config.model.policy,
        }
        names = {
            "r": "lora.rank",
            "lora_alpha": "lora.alpha",
            "use_rslora": "lora.rslora",
            "use_dora": "lora.dora",
            "base_model_name_or_path": "model.policy",
        }
        changed = [f"{names[k]} (saved {saved.get(k)!r}, now {v!r})" for k, v in now.items() if saved.get(k) != v]
        if changed:
            raise RLConfigError(
                f"the adapters at {path} were trained with other settings: {'; '.join(changed)}. Resume with the "
                "settings of the checkpoint (its rl_config.json), or start a new run."
            )

    @torch.no_grad()
    def lower_base_precision(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
        """Hold the frozen base in `dtype` (one GPU; the adapters stay float32). DoRA's starting magnitude is the
        row norm of the base as it is computed with, so a fresh run takes it again: the adapted model is then
        exactly the base at step 0."""
        for p in model.parameters():
            if not p.requires_grad and p.is_floating_point():
                p.data = p.data.to(dtype)
        if self.plugin.dora and not self.resumed:
            for m in self.plugin.modules.values():
                norm = torch.linalg.vector_norm(m.weight.float(), dim=1)
                m.lora_magnitude.copy_(norm.to(m.lora_magnitude.dtype))

    def start(self, model: nn.Module) -> None:
        """Log the trainable-parameter count and the memory the adapters take (once, before the first step)."""
        self.plugin.on_train_start(model, self.ctx)

    def param_groups(self, params: list, names: list[str]) -> tuple[list, list[float]]:
        """(what the optimizer takes, each group's learning-rate multiplier). One group, unless: lora.lr_ratio
        (LoRA+) puts the B matrices in a group at lr_ratio x the rate; train.weight_decay with DoRA puts the
        magnitudes in a group without decay (decay would pull every row's norm towards 0, away from the base's;
        A and B are decayed as configured, which shrinks the delta towards the base). `params` are the tensors
        the optimizer steps under the module names `names`."""
        split_b = self.lr_ratio != 1.0
        split_m = self.plugin.dora and self.ctx.config.train.weight_decay > 0
        if self.names:  # one group per adapter: each is clipped by itself (adapter_of names a parameter's adapter)
            order = ([""] if self.default_trains else []) + self.names
            groups = [{"params": [p for p, n in zip(params, names) if adapter_of(n) == name]} for name in order]
            return groups, [1.0] * len(groups)
        if not split_b and not split_m:
            return params, [1.0]
        kinds = [
            "b" if split_b and n.endswith(".lora_B") else "m" if split_m and n.endswith(".lora_magnitude") else "rest"
            for n in names
        ]
        groups: list = [{"params": [p for p, kind in zip(params, kinds) if kind == "rest"]}]
        scales = [1.0]
        if split_b:
            groups.append({"params": [p for p, kind in zip(params, kinds) if kind == "b"]})
            scales.append(self.lr_ratio)
        if split_m:
            groups.append({"params": [p for p, kind in zip(params, kinds) if kind == "m"], "weight_decay": 0.0})
            scales.append(1.0)
        return groups, scales

    def disabled(self):
        """A context in which the model is its frozen base (the policy the run started from)."""
        return self.plugin.disabled()

    # ---------------------------------------------------------------- weights

    def _meter(self) -> SurvivalMeter | None:
        """Every SURVIVAL_EVERY-th fold measures what the base's dtype keeps of the delta (the first holds none)."""
        self.pushes += 1
        return SurvivalMeter() if self.pushes % SURVIVAL_EVERY == 2 else None

    def _measured(self, meter: SurvivalMeter | None) -> None:
        if meter is None or not meter.dd > 0:
            return
        self.push_survival = meter.result()["projection"]
        if self.push_survival < SURVIVAL_WARN and self.pushes > SURVIVAL_WARN_AFTER and not self.warned:
            self.warned = True
            logger.warning(
                "lora/push_survival %.2f after %d weight pushes: the rollout engine's bf16 weights hold less than "
                "half of the adapters' delta (the rest is below half an ulp of its weight), so the engine still "
                "samples close to the base while the trainer scores the adapted policy. The importance ratio "
                "corrects for it like staleness, but the policy the rollouts explore lags. Raise "
                "train.learning_rate or lora.alpha, or set lora.rslora.",
                self.push_survival,
                self.pushes,
            )

    @staticmethod
    def _fold(m: LoRALinear, weight: Tensor, a: Tensor, b: Tensor, magnitude: Tensor | None, meter) -> Tensor:
        """A new tensor in `weight`'s dtype: weight + scaling * b a in float32 (DoRA: rows rescaled), rounded once."""
        folded = fold_weight(weight, a, b, m.scaling, magnitude)
        cast = folded.to(weight.dtype)
        if meter is not None:
            meter.update(weight, folded, cast)
        return cast

    def fold_stream(self, named: Iterable[tuple[str, Tensor]], measure: bool = False) -> Iterator[tuple[str, Tensor]]:
        """`named` (the model's parameters or state under its own names, adapters included, in any order) with
        every adapter folded into its base weight and the adapter tensors left out. One pair at a time: a base
        weight is held until its adapter has arrived and then leaves as a new tensor, so a consumer that keeps
        what it gets (FSDP's gathered parameters) never holds a weight both folded and not. Nothing in `named` is
        written. `measure`: this fold is a push to the rollout engine (it counts for lora/push_survival)."""
        modules = self.plugin.modules
        meter = self._meter() if measure else None
        pending: dict[str, dict[str, Tensor]] = {}
        for name, tensor in hf_state_dict_items(named):
            module, _, leaf = name.rpartition(".")
            m = modules.get(module)
            if adapter_of(name):  # a named adapter's tensors: not part of this (the default adapter's) policy
                continue
            if m is None or leaf not in ("weight", *ADAPTER_KEYS):
                yield name, tensor
                continue
            parts = pending.setdefault(module, {})
            parts[leaf] = tensor.detach()
            needed = ("weight", "lora_A", "lora_B") + (("lora_magnitude",) if m.lora_magnitude is not None else ())
            if all(k in parts for k in needed):
                del pending[module]
                magnitude = parts.get("lora_magnitude")
                yield (
                    f"{module}.weight",
                    self._fold(m, parts["weight"], parts["lora_A"], parts["lora_B"], magnitude, meter),
                )
        if pending:
            raise RuntimeError(f"lora: {sorted(pending)[:3]} came without their weight or adapter: nothing to fold")
        self._measured(meter)

    def folded_state(self, model: nn.Module) -> dict[str, Tensor]:
        """The model's state with the adapters folded in: a plain model's state (a reference reset loads it)."""
        return dict(self.fold_stream(model.state_dict().items()))

    def engine_parameters(self, model: nn.Module, adapter: str = "") -> Iterator[tuple[str, Tensor]]:
        """(checkpoint name, tensor) of the policy as a rollout engine loads it (one process, not sharded; with
        `adapter`, the policy of that named adapter): what
        opd.rollout.checkpoint_named_parameters gives for a plain model, with each adapted weight replaced by its
        fold. The fold reads the model's own parameters, the values the trainer computes with (on one GPU the
        published bf16 copies of the adapters, as a full-weight push reads the published bf16 weights), so the
        engine and the trainer differ only by the fold's rounding. Lazy: each folded weight is made when the
        engine asks for it, so the in-process engine, which copies one tensor at a time, never holds a second copy
        of the model (a vLLM server's push keeps every tensor until the server has loaded them: one extra copy of
        the adapted weights for the time of the push). The trainer's weights are read, never written."""
        from palingenesis.opd.rollout import checkpoint_named_parameters

        suffixes = tuple(f".{a}" for a in ADAPTER_KEYS)
        # an adapted weight is recognised by its storage: checkpoint names may differ from the module's
        by_weight = {m.weight.data_ptr(): m for m in self.plugin.modules.values()}
        if len(by_weight) != len(self.plugin.modules):
            raise RuntimeError(
                "lora: two adapted modules share one weight: their adapters cannot both be folded into it"
            )
        base = ((name, p) for name, p in model.named_parameters() if not name.endswith(suffixes))
        meter = self._meter()
        folded = 0
        for name, tensor in checkpoint_named_parameters(model, base):
            m = by_weight.get(tensor.data_ptr()) if tensor.numel() else None
            if m is None or tensor.shape != m.weight.shape or tensor.stride() != m.weight.stride():
                yield name, tensor
                continue
            magnitude = None if m.lora_magnitude is None else m.lora_magnitude.detach()
            a, b = m.factors(adapter)
            folded += 1
            yield name, self._fold(m, tensor, a.detach(), b.detach(), magnitude, meter)
        if folded != len(by_weight):
            # a weight the checkpoint stores in another form would reach the engine without its adapter
            raise RuntimeError(
                f"lora: {len(by_weight) - folded} adapted weight(s) were not found among the parameters the engine "
                "loads (the checkpoint stores them under a transformation): the engine cannot be given this policy"
            )
        self._measured(meter)

    # ------------------------------------------------------------ checkpoints

    def export(self, path: str | Path):
        """The `export` argument of checkpoint._save_gathered_or_local for a checkpoint at `path`: the state with
        the adapters folded in, in float32 (a bf16 base plus the adapters' float32 delta, exactly), and the adapter
        alone written to <path>/adapter in the PEFT layout."""

        def export(state: dict[str, Tensor]) -> dict[str, Tensor]:
            # on the CPU in float32, tied weights still one tensor (so they are written once), then the fold
            moved: dict[tuple, Tensor] = {}
            host = {}
            for name, tensor in state.items():
                key = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.dtype,
                )
                if key not in moved:
                    moved[key] = tensor.detach().cpu()
                host[name] = moved[key]
            del moved
            wide = cast_state_dict(host, torch.float32)
            unnamed = [name for name in self.plugin.modules if f"{name}.weight" not in wide]
            if unnamed:  # the fold goes by name: a checkpoint without its adapters folded would be the base
                raise RuntimeError(
                    f"lora: {len(unnamed)} adapted module(s) are not in the state to save under their own names "
                    f"({unnamed[:3]}): the checkpoint cannot be folded"
                )
            self.ctx.export_dir = str(path)
            try:
                for name in self.names:  # each named adapter by itself (PEFT layout), read from the state saved
                    parts = {
                        module: {
                            "lora_A": wide[f"{module}.roles.{name}.lora_A"],
                            "lora_B": wide[f"{module}.roles.{name}.lora_B"],
                        }
                        for module in self.plugin.modules
                    }
                    self.plugin._write_peft_adapter(Path(path) / NAMED_DIR / name, parts, self.ctx)
                return self.plugin.export_state_dict(wide, None, self.ctx)
            finally:
                self.ctx.export_dir = ""

        return export

    # ---------------------------------------------------------------- metrics

    def metrics(self, model: nn.Module, step: int) -> dict[str, float]:
        """lora/trainable_params, lora/trainable_share, lora/delta_rel (||scaling B A|| / ||W|| over the adapted
        weights), lora/magnitude_change (DoRA) and lora/push_survival (the share of the delta the engine's
        weights carry, once measured)."""
        self.ctx.step = step
        out = {f"lora/{k}": v for k, v in self.plugin.metrics(model, self.ctx).items()}
        if self.push_survival is not None:
            out["lora/push_survival"] = self.push_survival
        return out


def adapter_of(parameter_name: str) -> str:
    """The named adapter a parameter belongs to ("" for the default adapter and for base weights)."""
    found = re.search(r"\.roles\.([A-Za-z0-9_]+)\.lora_[AB]$", parameter_name)
    return found.group(1) if found else ""


def validate(config: Any, sharded: bool) -> tuple[list[str], list[str]]:
    """(errors, warnings) of the `lora:` section of an RLConfig."""
    c = config.lora
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(c.rank, bool) or not isinstance(c.rank, int) or c.rank < 0:
        return [f"lora.rank must be an integer >= 0 (0 = full-weight RL), got {c.rank!r}."], warnings
    if c.rank == 0:
        defaults = type(c)()
        changed = [
            f
            for f in ("alpha", "rslora", "dora", "lr_ratio", "targets", "exclude")
            if getattr(c, f) != getattr(defaults, f)
        ]
        if changed:
            errors.append(
                f"lora.{', lora.'.join(changed)} set without lora.rank: adapters are off (set lora.rank, e.g. 16)."
            )
        return errors, warnings
    if not c.alpha > 0:
        errors.append(f"lora.alpha must be > 0, got {c.alpha!r}.")
    if not c.lr_ratio > 0:
        errors.append(f"lora.lr_ratio must be > 0 (1 = plain LoRA), got {c.lr_ratio!r}.")
    for option in ("targets", "exclude"):
        for pattern in getattr(c, option):
            try:
                re.compile(str(pattern))
            except re.error as error:
                errors.append(f"lora.{option}: {pattern!r} is not a regular expression ({error}).")
    if config.train.optimizer == "muon":
        errors.append(
            "lora with train.optimizer muon is not supported: Muon orthogonalises full weight matrices, which is "
            "not defined for low-rank factors. Use adamw."
        )
    if config.model.multimodal:
        errors.append(
            "lora with model.multimodal is not supported yet (the adapters would also wrap the vision tower's linears)."
        )
    if config.cmr.enabled:
        errors.append("lora with cmr is not supported: the correction is defined on the full weight matrices.")
    if sharded and c.lr_ratio != 1.0:
        errors.append("lora.lr_ratio (LoRA+) is not supported with FSDP (train.fsdp, train.cpu_offload, torchrun).")
    if sharded and c.dora:
        errors.append(
            "lora.dora is not supported with FSDP (train.fsdp, train.cpu_offload, torchrun): FSDP computes with "
            "bf16 copies of the trained parameters, and a rounded magnitude makes the starting policy differ from "
            "the base."
        )
    return errors, warnings
