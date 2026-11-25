"""Environments from other ecosystems as catalogue entries (experimental).

  openenv_spec(...)       an OpenEnv server (Meta PyTorch / Hugging Face), remote or in-process, through
                          palingenesis.rl.envs.openenv.OpenEnvAdapter
  verifiers_spec(...)     a `verifiers` environment (the Prime Intellect Environments Hub's library): its dataset,
                          system prompt, Python tools and rubric, for SingleTurnEnv and ToolEnv

Both return an EnvSpec, to register in a package's entry point or by hand:

    from palingenesis.rl.catalog import register
    from palingenesis.rl.catalog.adapters import verifiers_spec

    register(verifiers_spec("acme/reverse-text", "reverse_text:load_environment", license="MIT",
                            summary="Reverse a string."))

Licences: an imported environment's code and data keep their own licence, which this library cannot read off the
object. `license` is therefore required, and a licence outside spec.PERMISSIVE is refused unless
allow_non_permissive=True (the entry then shows as not permissive in `pgs env list`).

Audits: imported environments come without a solver, so `pgs env test` runs the degenerate policies only (an empty
reply, a refusal, the prompt repeated, tool spam) and says so; pass oracle=... (a policy(row, messages)) for the
full audit. A rubric that pays any of those is a hole whoever wrote it.

Tested against stand-ins of both libraries' interfaces (neither is a dependency of palingenesis); OpenEnv servers
are exercised by the OpenEnvAdapter's own tests.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from palingenesis.rl.catalog.audit import audit_without_oracle
from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec, Policy, SpecError, is_permissive


def _check_license(name: str, license: str, allow_non_permissive: bool) -> None:
    if not license:
        raise SpecError(f"{name}: license is required (the imported environment's own, e.g. MIT)")
    if not is_permissive(license) and not allow_non_permissive:
        raise SpecError(
            f"{name}: licence {license!r} is not a permissive one; pass allow_non_permissive=True to import it anyway "
            "(check that its terms allow your use)"
        )


# -------------------------------------------------------------------- OpenEnv


def seed_rows(n: int, seed: int = 0, prompt: str = "") -> list[dict]:
    """Rows that select a task by seed: every rollout of a group resets the server with the same one."""
    return [{"prompt": prompt, "seed": seed * 100_003 + i} for i in range(n)]


def openenv_spec(
    name: str,
    *,
    license: str,
    summary: str,
    base_url: str = "",
    env_class: str = "",
    family: str = "other",
    version: str = "0.1.0",
    rows: str = "",
    reset_fields: tuple[str, ...] = ("seed",),
    prompt: str = "observation",
    env_args: dict | None = None,
    max_turns: int = 16,
    oracle: Policy | None = None,
    degenerate: dict[str, Policy] | None = None,
    citation: str = "",
    trainable: bool = True,
    allow_non_permissive: bool = False,
) -> EnvSpec:
    """An OpenEnv environment as a catalogue entry. base_url: a running server; env_class: "module:Class" of the
    server class, in-process. Tasks are selected by the row columns in `reset_fields` (default: a seed per row,
    generated); `rows`: a file of your own rows instead. prompt "observation": the reset observation is the
    prompt."""
    _check_license(name, license, allow_non_permissive)
    arguments = {"reset_fields": list(reset_fields), "prompt": prompt, **(env_args or {})}
    arguments.update({"base_url": base_url} if base_url else {"env_class": env_class})
    dataset = DatasetSpec(path=rows) if rows else DatasetSpec(generator=seed_rows, license=license)
    return EnvSpec(
        name=name,
        version=version,
        family=family,
        summary=summary,
        env="palingenesis.rl.envs.openenv:OpenEnvAdapter",
        env_args=arguments,
        license=license,
        verifier="The OpenEnv server's own step rewards (combined as env_args.reward: last, sum or max).",
        datasets={"rows" if rows else "seeds": dataset},
        citation=citation,
        trainable=trainable,
        max_turns=max_turns,
        oracle=oracle,
        degenerate=degenerate or {},
        audit=None if oracle is not None else audit_without_oracle,
        requires=("openenv",) if base_url else (),
        tags=("openenv", "imported"),
        notes="Imported through OpenEnvAdapter; see palingenesis.rl.envs.openenv for its options.",
    )


# ------------------------------------------------------------------ verifiers

_loaded: dict[str, Any] = {}


def load_verifiers(env: Any, load_args: dict | None = None) -> Any:
    """The verifiers environment object: itself, or "module:load_environment" called once and cached."""
    if not isinstance(env, str):
        return env
    key = f"{env}/{sorted((load_args or {}).items())}"
    if key not in _loaded:
        from palingenesis.rl.rewards import load_object

        loader = load_object(env)
        _loaded[key] = loader(**(load_args or {})) if callable(loader) and not hasattr(loader, "rubric") else loader
    return _loaded[key]


def _kind(vf_env: Any) -> str:
    names = {cls.__name__ for cls in type(vf_env).__mro__}
    if "ToolEnv" in names or getattr(vf_env, "tools", None):
        if "StatefulToolEnv" in names or "SandboxEnv" in names:
            raise NotImplementedError(
                f"{type(vf_env).__name__}: stateful verifiers tool environments keep per-rollout state the adapter "
                "does not carry; write the environment against palingenesis.rl.env instead"
            )
        return "tools"
    if "SingleTurnEnv" in names or not hasattr(vf_env, "env_response"):
        return "single_turn"
    raise NotImplementedError(
        f"{type(vf_env).__name__}: only verifiers SingleTurnEnv and ToolEnv are imported (a MultiTurnEnv with its own "
        "env_response is a program of its own: write it against palingenesis.rl.env)"
    )


def _rubric_functions(rubric: Any) -> list[tuple[Callable, float]]:
    def first(*names: str) -> Any:
        for attr in names:
            value = getattr(rubric, attr, None)
            if value is not None:
                return value() if callable(value) else value
        return None

    functions = list(first("get_reward_funcs", "reward_funcs", "funcs") or [])
    weights = list(first("get_reward_weights", "reward_weights", "weights") or [])
    weights += [1.0] * (len(functions) - len(weights))
    return list(zip(functions, weights))


class VerifiersEnv:
    """A verifiers SingleTurnEnv or ToolEnv behind the palingenesis environment protocol.

    Tools are the environment's Python functions; the reward is the rubric's weighted sum, each function called
    with what it declares among prompt, completion (the messages after the prompt), answer, state, task, info and
    parser, as verifiers does."""

    def __init__(self, env: Any, load_args: dict | None = None):
        self.vf = load_verifiers(env, load_args)
        self.kind = _kind(self.vf)
        self.functions = {fn.__name__: fn for fn in (getattr(self.vf, "tools", None) or [])}
        self.row: dict[str, Any] = {}
        self.prompt_length = 0

    def reset(self, **row) -> None:
        from palingenesis.rl.data import prompt_messages

        self.row = row
        self.prompt_length = len(prompt_messages(row))

    def tool_schemas(self) -> list[dict]:
        from palingenesis.rl.chat import tool_schema

        return [tool_schema(fn, name) for name, fn in self.functions.items()]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from palingenesis.rl.env import call_sync_or_async

        return await call_sync_or_async(self.functions[name], **arguments)

    async def get_reward(self, messages: list[dict]) -> dict[str, float]:
        from palingenesis.rl.env import call_sync_or_async

        prompt, completion = messages[: self.prompt_length], messages[self.prompt_length :]
        rubric = self.vf.rubric
        info = self.row.get("info") or {}
        available = {
            "prompt": prompt,
            "completion": completion,
            "answer": self.row.get("answer", ""),
            "task": self.row.get("task", "default"),
            "info": info,
            "parser": getattr(rubric, "parser", None) or getattr(self.vf, "parser", None),
        }
        available["state"] = {
            **available,
            "responses": [],
            "turn": sum(m.get("role") == "assistant" for m in completion),
        }
        out: dict[str, float] = {}
        total = 0.0
        for fn, weight in _rubric_functions(rubric):
            parameters = inspect.signature(fn).parameters
            everything = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            kwargs = available if everything else {k: v for k, v in available.items() if k in parameters}
            value = float(await call_sync_or_async(fn, **kwargs))
            total += weight * value
            out[f"metric/{getattr(fn, '__name__', 'reward')}"] = value
        return {"reward": total, **out}


def verifiers_rows(
    env: Any, n: int = 0, seed: int = 0, load_args: dict | None = None, split: str = "eval"
) -> list[dict]:
    """The verifiers environment's dataset as palingenesis rows ({"messages", "answer", "info", "task"})."""
    vf_env = load_verifiers(env, load_args)

    def first(*names: str) -> Any:
        for attr in names:
            value = getattr(vf_env, attr, None)
            value = value() if callable(value) else value
            if value is not None:
                return value
        return None

    data = first("get_eval_dataset", "eval_dataset") if split == "eval" else None
    data = data if data is not None else first("get_dataset", "dataset")
    if data is None:
        raise ValueError(f"{type(vf_env).__name__} has no dataset")
    system = getattr(vf_env, "system_prompt", None)
    rows = []
    for i, raw in enumerate(data):
        raw = dict(raw)
        messages = raw.get("prompt")
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(raw.get("question", messages or ""))}]
            if system:
                messages = [{"role": "system", "content": system}] + messages
        rows.append(
            {
                "messages": [dict(m) for m in messages],
                "answer": raw.get("answer", ""),
                "info": raw.get("info") or {},
                "task": raw.get("task", "default"),
                "id": str(raw.get("id", raw.get("example_id", i))),
            }
        )
    if seed:
        import random

        random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def verifiers_spec(
    name: str,
    env: Any,
    *,
    license: str,
    summary: str,
    family: str = "other",
    version: str = "0.1.0",
    load_args: dict | None = None,
    max_turns: int = 0,
    oracle: Policy | None = None,
    degenerate: dict[str, Policy] | None = None,
    citation: str = "",
    trainable: bool = True,
    allow_non_permissive: bool = False,
) -> EnvSpec:
    """A verifiers environment as a catalogue entry. env: the environment object, or "module:load_environment"
    (imported and called with `load_args` when the entry is first used, so listing the catalogue does not import
    the package)."""
    _check_license(name, license, allow_non_permissive)

    def rows(n: int, seed: int = 0, split: str = "eval") -> list[dict]:
        return verifiers_rows(env, n, seed, load_args, split)

    return EnvSpec(
        name=name,
        version=version,
        family=family,
        summary=summary,
        env=VerifiersEnv,
        env_args={"env": env, **({"load_args": load_args} if load_args else {})},
        license=license,
        verifier="The verifiers rubric's weighted sum of reward functions, as the environment's author wrote it.",
        datasets={
            "verifiers": DatasetSpec(generator=rows, license=license, description="The environment's own dataset.")
        },
        citation=citation,
        trainable=trainable,
        max_turns=max_turns or int(getattr(env, "max_turns", 0) or 0) or (1 if isinstance(env, str) else 8),
        oracle=oracle,
        degenerate=degenerate or {},
        audit=None if oracle is not None else audit_without_oracle,
        requires=("verifiers",) if isinstance(env, str) else (),
        tags=("verifiers", "imported"),
        notes="Imported through catalog.adapters.VerifiersEnv (SingleTurnEnv and ToolEnv).",
    )
