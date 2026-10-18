"""Action protocols: how the policy writes a tool call and how results come back.

    Native()          native function calling: schemas reach the chat template (as vLLM renders a real harness's
                      request with a matching --tool-call-parser); results are tool messages
    FencedCommand()   one shell command in a fenced block with an info string (mini-swe-agent's text mode:
                      ```mswea_bash_command ... ```), exactly one per turn; results are user turns
    XmlTool / ReAct / JsonBlock / Tags
                      generic text protocols with the tools described in the prompt (the ones palingenesis'
                      AgentMixedEnv rows use): <tool name="N">{json}</tool>, Action/Action Input lines, ```json
                      {"action", "args"}``` blocks, <N>{json}</N>

A text protocol renders its instructions and tool list (instructions()), parses actions from the assistant's text
(parse(), raising FormatError when the turn breaks the protocol), renders calls back to text (render(), for round
trips and scripted episodes) and joins results into the next user turn (observation()).
"""

from __future__ import annotations

import json
import re


class FormatError(Exception):
    """The turn breaks the protocol (e.g. two commands where exactly one is allowed); str(e) is the reason and n the
    number of actions found."""

    def __init__(self, reason: str, n: int = 0):
        super().__init__(reason)
        self.n = n


def tool_lines(schemas: list[dict]) -> str:
    out = []
    for s in schemas:
        f = s.get("function", s)
        params = (f.get("parameters") or {}).get("properties") or {}
        req = set((f.get("parameters") or {}).get("required") or [])
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in params.items())
        out.append(f"- {f['name']}({args}): {f.get('description', '')}")
    return "\n".join(out)


def _json_args(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        v = json.loads(text)
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


class Protocol:
    name = "protocol"
    native = False

    def instructions(self, schemas: list[dict]) -> str:
        return ""

    def parse(self, content: str) -> list[tuple[str, dict]]:
        return []

    def render(self, calls: list[tuple[str, dict]]) -> str:
        raise NotImplementedError

    def observation(self, results: list[tuple[str, str]]) -> str:
        return "\n\n".join(out for _, out in results)


class Native(Protocol):
    name = "native"
    native = True


class FencedCommand(Protocol):
    """mini-swe-agent's text mode: ```<lang>\\n<command>\\n``` (regex as upstream), exactly one per turn."""

    name = "fenced"

    def __init__(
        self, lang: str = "mswea_bash_command", tool: str = "bash", arg: str = "command", exactly_one: bool = True
    ):
        self.lang, self.tool, self.arg, self.exactly_one = lang, tool, arg, exactly_one
        self.regex = re.compile(rf"```{re.escape(lang)}\s*\n(.*?)\n```", re.DOTALL)

    def parse(self, content):
        actions = [a.strip() for a in self.regex.findall(content or "")]
        if self.exactly_one and len(actions) != 1:
            raise FormatError(f"Expected exactly 1 action, found {len(actions)}.", len(actions))
        return [(self.tool, {self.arg: a}) for a in actions]

    def render(self, calls):
        return "\n\n".join(f"```{self.lang}\n{args[self.arg]}\n```" for _, args in calls)


class XmlTool(Protocol):
    name = "xml"
    pattern = re.compile(r'<tool\s+name="([^"]+)"\s*>(.*?)</tool>', re.DOTALL)

    def instructions(self, schemas):
        return (
            'Write each action as <tool name="NAME">{"argument": value}</tool>; several actions in one message '
            "if they are independent. When you are done, answer in plain text with no action.\n\n" + tool_lines(schemas)
        )

    def parse(self, content):
        out = []
        for name, body in self.pattern.findall(content or ""):
            args = _json_args(body)
            if args is None:
                raise FormatError(f"The arguments of {name} are not a JSON object.", len(out))
            out.append((name, args))
        return out

    def render(self, calls):
        return "\n".join(f'<tool name="{n}">{json.dumps(a, ensure_ascii=False)}</tool>' for n, a in calls)

    def observation(self, results):
        return "\n".join(f'<tool_result name="{n}">{o}</tool_result>' for n, o in results)


class ReAct(Protocol):
    name = "react"
    pattern = re.compile(r"Action:\s*([\w.-]+)\s*\nAction Input:\s*(\{.*?\}|\S*)\s*(?=\n\S|\Z)", re.DOTALL)

    def instructions(self, schemas):
        return (
            'Write one action per message as\nAction: NAME\nAction Input: {"argument": value}\n'
            "When you are done, answer in plain text with no action.\n\n" + tool_lines(schemas)
        )

    def parse(self, content):
        found = self.pattern.findall(content or "")
        if len(found) > 1:
            raise FormatError("Write one action per message.", len(found))
        out = []
        for name, body in found:
            args = _json_args(body)
            if args is None:
                raise FormatError(f"The Action Input of {name} is not a JSON object.", 1)
            out.append((name, args))
        return out

    def render(self, calls):
        return "\n".join(f"Action: {n}\nAction Input: {json.dumps(a, ensure_ascii=False)}" for n, a in calls)

    def observation(self, results):
        return "\n".join(f"Observation: {o}" for _, o in results)


class JsonBlock(Protocol):
    name = "json"
    pattern = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)

    def instructions(self, schemas):
        return (
            'Write each action as a ```json {"action": "NAME", "args": {...}} ``` block; several blocks if they '
            "are independent. When you are done, answer in plain text with no action.\n\n" + tool_lines(schemas)
        )

    def parse(self, content):
        out = []
        for body in self.pattern.findall(content or ""):
            v = _json_args(body)
            if not v or "action" not in v:
                raise FormatError('A ```json block must be {"action": NAME, "args": {...}}.', len(out))
            args = v.get("args") or {}
            if not isinstance(args, dict):
                raise FormatError("args must be a JSON object.", len(out))
            out.append((str(v["action"]), args))
        return out

    def render(self, calls):
        return "\n".join(
            "```json\n" + json.dumps({"action": n, "args": a}, ensure_ascii=False) + "\n```" for n, a in calls
        )

    def observation(self, results):
        return "\n\n".join(f"Result ({n}):\n{o}" for n, o in results)


class Tags(Protocol):
    name = "tags"

    def __init__(self, names: list[str] | None = None):
        self.names = list(names or [])

    def instructions(self, schemas):
        self.names = [s.get("function", s)["name"] for s in schemas]
        return (
            'Write each action as <TOOL_NAME>{"argument": value}</TOOL_NAME>, one tag per action; several tags '
            "if they are independent. When you are done, answer in plain text with no action.\n\n" + tool_lines(schemas)
        )

    def parse(self, content):
        out = []
        for name in self.names:
            for m in re.finditer(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", content or "", re.DOTALL):
                args = _json_args(m.group(1))
                if args is None:
                    raise FormatError(f"The arguments of {name} are not a JSON object.", len(out))
                out.append((m.start(), name, args))
        return [(n, a) for _, n, a in sorted(out, key=lambda x: x[0])]

    def render(self, calls):
        return "\n".join(f"<{n}>{json.dumps(a, ensure_ascii=False)}</{n}>" for n, a in calls)

    def observation(self, results):
        return "\n".join(f"<{n}_result>{o}</{n}_result>" for n, o in results)


PROTOCOLS = {p.name: p for p in (Native, FencedCommand, XmlTool, ReAct, JsonBlock, Tags)}
