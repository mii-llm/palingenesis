"""HarnessProfile: the dialect of one agent harness (prompt, tool contracts, action protocol, observation wording,
stop and answer rules, context policy), and the per-episode Session that runs it on a HarborEnv container.

    ToolSpec(schema, bind)       an exact tool contract (OpenAI function schema) and its binding: bind(session, args)
                                 runs ops (palingenesis.rl.envs.harness.ops) and returns the harness's own output text
    HarnessProfile(...)          name@version, system / user templates, tools, protocol, stop, answer, context
    Session(profile, runtime)    one episode: conversation, tool_schemas, call_tool, text_actions, observation_text,
                                 done, final_message — what HarborEnv delegates to
    HarnessMix(profile|profiles) the profile of an episode: drawn deterministically per GRPO group (same for every
                                 sample of a group: advantages compare policies, not harnesses), varied across groups

Templates fill {instruction}, {workdir}, {tools} (a text protocol's instructions and tool list), {system_information}
(uname of the container), {date} and {model} by plain replacement, so JSON examples need no brace escaping.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable

from palingenesis.rl.envs.harness import ops
from palingenesis.rl.envs.harness.context import Context
from palingenesis.rl.envs.harness.protocols import FormatError, Native, Protocol
from palingenesis.rl.envs.harness.rules import FLAG, AnswerRule, LastText, NoCall, StopRule

FORMAT_ERROR = "__format_error__"  # the synthetic action that answers a turn breaking the protocol


def fn_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """An OpenAI function tool schema."""
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": required}}}


@dataclass
class ToolSpec:
    schema: dict
    bind: Callable[["Session", dict], str] | None = None

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


def _default_format_error(reason: str, n: int) -> str:
    return f"Format error: {reason}"


def _default_unknown_tool(name: str, names: list[str], arguments: dict | None = None) -> str:
    return f"Error: unknown tool {name!r}. Available tools: {', '.join(names)}."


def render(template: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


@dataclass
class HarnessProfile:
    """One harness dialect. legacy=True is HarborEnv's own behaviour (palingenesis-default): the row's conversation
    and HarborEnv's tools run unchanged."""

    name: str
    version: str
    system: str | None = None
    user: str = "{instruction}"
    tools: list[ToolSpec] = field(default_factory=list)
    protocol: Protocol = field(default_factory=Native)
    stop: StopRule = field(default_factory=NoCall)
    answer: AnswerRule = field(default_factory=LastText)
    context: Context = field(default_factory=Context)
    format_error: Callable[[str, int], str] = _default_format_error
    unknown_tool: Callable[[str, list[str], dict], str] = _default_unknown_tool  # (name, tool names, arguments)
    alias: Callable[[str, dict, list[str]], tuple[str, dict] | None] | None = None  # the harness's own repair of an
    # unknown tool name (opencode lowercases, OpenHands maps bash -> terminal): (name, arguments) to run instead
    start: Callable[["Session"], dict[str, str]] | None = None  # per-episode setup (e.g. a tmux session) returning
    # extra template values for the first turn (e.g. {terminal_state})
    exec_timeout: float = 60  # default shell timeout (seconds) when the call sets none
    runtime_max_output: int = 8_000_000  # the episode runtime's own output cap (the profile truncates first)
    chat_template_kwargs: dict | None = None  # applied to the episode's chat template (e.g. enable_thinking); None:
    # the model's default, as a harness that sends no chat_template_kwargs to vLLM gets
    emulates: str = ""  # what was read to write the profile
    legacy: bool = False
    values: dict[str, str] = field(default_factory=dict)  # extra template values (date, model)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def by_name(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    def schemas(self) -> list[dict]:
        return [t.schema for t in self.tools]

    def messages(self, instruction: str, **values: str) -> list[dict]:
        vals = {"workdir": "/workdir", "date": "", "model": "", "system_information": "Linux", "git_repo": "no", "git_root": "/",
                **self.values, **values}
        vals["tools"] = "" if self.protocol.native else self.protocol.instructions(self.schemas())
        vals["instruction"] = instruction
        out = [{"role": "system", "content": render(self.system, vals)}] if self.system else []
        out.append({"role": "user", "content": render(self.user.replace("{instruction}", "\0"), vals)
                    .replace("\0", instruction)})
        return out


class Session:
    """One episode of a profile on a container (HarborEnv delegates its protocol methods here)."""

    def __init__(self, profile: HarnessProfile, rt: Any):
        self.profile = profile
        self.ws = ops.Workspace(rt) if rt is not None else None
        self.done = False
        self.answer: str | None = None
        self.flags: dict[str, Any] = {}
        self.last_exec: ops.ExecOut | None = None
        self.calls = 0

    # ---------------------------------------------------------------- episode start

    def conversation(self, instruction: str) -> list[dict] | None:
        """The starting conversation (None for the legacy profile: the row's own)."""
        p = self.profile
        if p.legacy:
            return None
        values = {"workdir": self.workdir}
        templates = (p.system or "") + p.user
        if self.ws is not None and "{system_information}" in templates:
            r = self.ws.sh("uname -srvm", timeout=30)
            values["system_information"] = r.stdout.strip() if r.exit_code == 0 else "Linux"
        if self.ws is not None and "{git_repo}" in templates:
            r = self.ws.sh(f"git -C {shlex.quote(self.workdir)} rev-parse --is-inside-work-tree 2>/dev/null", timeout=30)
            values["git_repo"] = "yes" if r.stdout.strip() == "true" else "no"
        if self.ws is not None and "{git_root}" in templates:
            r = self.ws.sh(f"git -C {shlex.quote(self.workdir)} rev-parse --show-toplevel 2>/dev/null", timeout=30)
            values["git_root"] = r.stdout.strip() if r.exit_code == 0 and r.stdout.strip() else "/"
        if p.start is not None:
            values.update(p.start(self))
        return p.messages(instruction, **values)

    @property
    def workdir(self) -> str:
        return self.ws.workdir if self.ws is not None else "/workdir"

    def tool_schemas(self) -> list[dict]:
        """Native schemas (descriptions may name {workdir}); none for a text protocol (tools are in the prompt)."""
        if not self.profile.protocol.native:
            return []
        schemas = self.profile.schemas()
        raw = json.dumps(schemas)
        return json.loads(raw.replace("{workdir}", self.workdir)) if "{workdir}" in raw else schemas

    # ---------------------------------------------------------------- actions

    def finish(self, answer: str | None) -> None:
        self.done = True
        if answer is not None:
            self.answer = answer

    def text_actions(self, content: str) -> list[tuple[str, dict]] | None:
        """Actions from a turn without native tool calls: the text protocol's, a format error the harness answers, or
        None (the episode ends)."""
        p = self.profile
        if p.legacy:
            return None
        try:
            actions = [] if p.protocol.native else p.protocol.parse(content)
        except FormatError as e:
            return [(FORMAT_ERROR, {"message": p.format_error(str(e), e.n)})]
        if actions:
            return actions
        if p.stop.requires_action:
            reason = "No tool calls found in the response. Every response MUST include at least one tool call." \
                if p.protocol.native else "Expected exactly 1 action, found 0."
            return [(FORMAT_ERROR, {"message": p.format_error(reason, 0)})]
        return None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        p = self.profile
        if name == FORMAT_ERROR:
            return str(arguments.get("message", ""))
        self.last_exec = None
        self.calls += 1
        if name == FLAG:
            out = ""
        else:
            spec = p.by_name.get(name)
            if spec is None and p.alias is not None:  # the harness repairs the name itself
                routed = p.alias(name, dict(arguments or {}), list(p.by_name))
                if routed is not None and routed[0] in p.by_name:
                    name, arguments = routed
                    spec = p.by_name[name]
            if spec is None or spec.bind is None:
                return p.unknown_tool(name, list(p.by_name), dict(arguments or {}))
            try:
                out = spec.bind(self, dict(arguments or {}))
            except ops.OpError as e:  # a binding that does not word this error itself
                out = f"Error: {e.kind}: " + ", ".join(f"{k}={v}" for k, v in e.data.items())
        return p.stop.on_tool(self, name, arguments or {}, out)

    def unknown_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """A call of a tool the episode does not expose: the harness's alias, or its own error message."""
        return self.call_tool(name, arguments)

    def observation_text(self, results: list[tuple[str, str]]) -> str:
        if results and all(n == FORMAT_ERROR for n, _ in results):
            return "\n\n".join(o for _, o in results)
        return self.profile.protocol.observation([(n, o) for n, o in results if n != FORMAT_ERROR])

    def final_message(self, messages: list[dict] | None) -> str | None:
        return self.profile.answer(self, messages)


# ------------------------------------------------------------------ registry and mixing

_REGISTRY: dict[str, Callable[[], HarnessProfile]] = {}


def register(name: str, version: str, factory: Callable[[], HarnessProfile]) -> None:
    """Register a profile factory under "name" and "name@version"."""
    _REGISTRY[name] = factory
    _REGISTRY[f"{name}@{version}"] = factory


def get_profile(key: str) -> HarnessProfile:
    from palingenesis.rl.envs.harness import profiles  # noqa: F401 — registers the built-in profiles

    if key not in _REGISTRY:
        raise KeyError(f"unknown harness profile {key!r}; known: {', '.join(sorted(k for k in _REGISTRY if '@' in k))}")
    return _REGISTRY[key]()


def available() -> list[str]:
    from palingenesis.rl.envs.harness import profiles  # noqa: F401

    return sorted(k for k in _REGISTRY if "@" in k)


def _unit(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64


class HarnessMix:
    """Which profile an episode runs. profiles={name: weight} (or one profile): drawn from sha256(seed, group,
    task_dir), where group is the GRPO group id the trainer sets on the environment (HarborEnv.episode_group), so
    every sample of a group shares the profile and successive groups vary; without a group id (an environment used
    outside the trainer) the row's content stands in. A row's own "harness_profile" field wins."""

    def __init__(self, profile: str | None = None, profiles: dict[str, float] | None = None, seed: int = 0):
        weights = dict(profiles or {})
        if profile:
            weights.setdefault(profile, 1.0)
        if not weights:
            raise ValueError("HarnessMix needs a profile or profiles={name: weight}")
        if any(float(w) < 0 for w in weights.values()) or sum(float(w) for w in weights.values()) <= 0:
            raise ValueError(f"profile weights must be >= 0 with a positive sum: {weights}")
        self.weights = {k: float(w) for k, w in sorted(weights.items()) if float(w) > 0}
        self.seed = seed
        self.profiles = {k: get_profile(k) for k in self.weights}  # validates the names now, not mid-run

    @property
    def names(self) -> list[str]:
        return sorted({p.name for p in self.profiles.values()})

    def draw(self, row: dict, group: Any = None) -> HarnessProfile:
        if row.get("harness_profile"):
            return get_profile(str(row["harness_profile"]))
        keys = list(self.weights)
        if len(keys) == 1:
            return get_profile(keys[0])
        anchor = group if group is not None else hashlib.sha256(
            json.dumps({k: v for k, v in row.items() if k != "harness_profile"}, sort_keys=True, default=str)
            .encode()).hexdigest()
        u = _unit(f"{self.seed}|{anchor}|{row.get('task_dir', '')}") * sum(self.weights.values())
        acc = 0.0
        for k in keys:
            acc += self.weights[k]
            if u < acc:
                return get_profile(k)
        return get_profile(keys[-1])
