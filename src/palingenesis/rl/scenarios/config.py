"""Validation of the RL config's `scenario:` and `roles:` sections (palingenesis.rl.config)."""

from __future__ import annotations

import os
import re
from typing import Any

from palingenesis.rl.scenarios.base import build_scenario

BASE = "base"  # roles.<name>.partner: the policy's frozen base


def adapter_names(config: Any) -> list[str]:
    """The named adapters the roles ask for, in order of first use."""
    names: list[str] = []
    for role in config.roles.values():
        if role.adapter and not role.partner and role.adapter not in names:
            names.append(role.adapter)
    return names


def validate(config: Any, sharded: bool) -> list[str]:
    """Errors of the scenario and roles sections. `sharded`: the policy is sharded with FSDP."""
    s, e, loss = config.scenario, config.env, config.loss
    errors: list[str] = []
    if not s.type:
        return ["roles: and scenario options need scenario.type (a built-in scenario or 'module:Class')."]
    try:
        scenario = build_scenario(s.type, s.args)
    except Exception as error:  # noqa: BLE001 — whatever the scenario's import or constructor says
        return [f"scenario.type {s.type!r}: {type(error).__name__}: {error}"]
    roles = ", ".join(scenario.roles)
    unknown = sorted(set(config.roles) - set(scenario.roles))
    if unknown:
        errors.append(f"roles: {unknown} are not roles of scenario {scenario.name} ({roles}).")
    if s.outcome and s.outcome not in scenario.roles:
        errors.append(f"scenario.outcome {s.outcome!r} is not a role of {scenario.name} ({roles}).")
    if not s.timeout > 0:
        errors.append("scenario.timeout must be > 0 seconds.")
    if e.type != "single_turn" or e.tools:
        errors.append(
            "scenario with env.type / env.tools: a scenario gives each role its tools; leave env at its default."
        )
    if config.rewards:
        errors.append(
            "scenario with rewards: the scenario's verifier sets the rewards (weights: scenario.args.weights)."
        )
    if config.model.multimodal:
        errors.append("scenario with model.multimodal is not supported yet.")
    if config.rollout.think_skip:
        errors.append("scenario with rollout.think_skip is not supported.")
    if loss.overlong_buffer or loss.group_shaping or loss.observation_weight:
        errors.append(
            "scenario with loss.overlong_buffer, loss.group_shaping or loss.observation_weight is not supported."
        )
    if loss.advantage_std_floor < 0:
        errors.append("loss.advantage_std_floor must be >= 0.")
    table = scenario.partners()
    trained = [name for name in scenario.roles if name not in config.roles]
    for name, role in config.roles.items():
        if name not in scenario.roles:
            continue
        if role.share < 0:
            errors.append(f"roles.{name}.share must be >= 0.")
        if any(getattr(role, k) < 0 for k in ("max_turns", "max_new_tokens", "max_completion_tokens")):
            errors.append(f"roles.{name}: max_turns, max_new_tokens and max_completion_tokens must be >= 0.")
        if role.adapter and not re.fullmatch(r"[A-Za-z0-9_]+", role.adapter):
            errors.append(f"roles.{name}.adapter {role.adapter!r}: letters, digits and underscores only.")
        if role.adapter == BASE:
            errors.append(f"roles.{name}.adapter cannot be called 'base' (the frozen base is partner: base).")
        if role.adapter == "policy":
            errors.append(f"roles.{name}.adapter cannot be called 'policy' (the name of the default adapter).")
        if role.partner and role.adapter:
            errors.append(f"roles.{name}: partner and adapter are alternatives (a partner is not the policy).")
        if role.partner_args and not role.partner:
            errors.append(f"roles.{name}.partner_args without roles.{name}.partner.")
        if role.partner.startswith("scripted:") and role.partner.split(":", 1)[1] not in table.get(name, {}):
            errors.append(
                f"roles.{name}.partner {role.partner!r}: scenario {scenario.name} has no such scripted partner for "
                f"this role (it has {sorted(table.get(name, {}))})."
            )
        if role.partner == BASE and not config.lora.rank:
            errors.append(f"roles.{name}.partner base is the policy with its adapters off: it needs lora.rank.")
        if not role.partner and role.train:
            trained.append(name)
    if not trained:
        errors.append("no role trains: at least one role must be the policy with train: true (the default).")
    names = adapter_names(config)
    based = any(role.partner == BASE for role in config.roles.values())
    if names:
        if not config.lora.rank:
            errors.append(
                f"roles with an adapter ({names}) need lora.rank: they are LoRA adapters on the policy's base."
            )
        if config.lora.dora or config.lora.lr_ratio != 1.0:
            errors.append("roles with their own adapter are not supported with lora.dora or lora.lr_ratio.")
        if sharded:
            errors.append(
                "roles with their own adapter are not supported with FSDP (train.fsdp, train.cpu_offload, torchrun) yet."
            )
        if loss.kl_ref_reset:
            errors.append(
                "roles with their own adapter are not supported with loss.kl_ref_reset (kl_ref: start works)."
            )
    if (names or based) and config.rollout.backend == "remote" and config.rollout.push in ("full", "fold"):
        errors.append(
            "roles with their own adapter (or partner: base) on remote engines are served as named adapters next "
            "to the base in one engine: rollout.push must be native (or auto). An engine that folds, or receives "
            "full weights, holds the trained policy alone."
        )
    if based and config.rollout.backend == "remote" and sharded:
        errors.append(
            "roles.<name>.partner base on remote engines is not supported with FSDP (the engines would hold the "
            "trained policy alone): use train.replicate_base."
        )
    if based and config.rollout.backend == "remote" and config.lora.dora:
        errors.append("roles.<name>.partner base on remote engines needs rollout.push native, which has no lora.dora.")
    if (names or based) and config.rollout.backend not in ("hf", "remote") and os.environ.get("WORLD_SIZE", "1") != "1":
        errors.append(
            "roles with their own adapter (or partner: base) run on one GPU for now: a vLLM server per adapter is "
            "single-process. Under torchrun, share the weights between the roles."
        )
    if (names or based) and config.rollout.backend == "vllm":
        errors.append(
            "roles with their own adapter (or partner: base) need a rollout engine each: use rollout.backend "
            "vllm_server (one server per adapter, each fed its folded weights) or hf; the in-process vllm engine "
            "holds one model."
        )
    if (names or based) and config.rollout.backend == "vllm_server" and config.rollout.url:
        errors.append(
            "several engines with rollout.url: the trainer launches its own servers (leave rollout.url empty)."
        )
    return errors
