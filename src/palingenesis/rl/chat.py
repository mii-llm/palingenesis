"""Chat-template plumbing for token-in/token-out rollouts.

encode_prompt       a conversation (and its tools) rendered for the policy's first turn
ChatFormat          the tokens the template writes between an assistant turn and the
                    next one when tool results (or any messages) come in between, from
                    the template's own rendering against a dummy prefix. Appending them
                    to the trajectory never re-renders or re-tokenizes history, so
                    templates that rewrite earlier turns (Qwen3 dropping past reasoning)
                    cannot desynchronize what the sampler saw from what is trained.
parse_assistant     an assistant turn's text split into reasoning, content and tool
                    calls: Hermes JSON (Qwen2.5/Qwen3: <tool_call>{"name", "arguments"}</tool_call>)
                    and XML (Qwen3-Coder/Qwen3.5: <tool_call><function=f><parameter=p>v</parameter>...)
tool_schema         OpenAI function schema from a Python callable's type hints and docstring
"""

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_DUMMY_USER = "PgsDummyUserTurn"
_DUMMY_ANSWER = "PgsDummyAssistantTurn"


def encode_prompt(
    tok,
    messages: list[dict],
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """Token ids of `messages` rendered for the assistant's turn (BOS prepended when the
    template leaves it out)."""
    text = tok.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
        **(chat_template_kwargs or {}),
    )
    ids = tok.encode(text, add_special_tokens=False)
    if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    return ids


class ChatFormat:
    """What the policy's chat template writes between assistant turns."""

    def __init__(
        self,
        tok,
        eot_id: int,
        chat_template_kwargs: dict | None = None,
        think_tags: tuple[str, str] = ("<think>", "</think>"),
    ):
        self.tok = tok
        self.eot_id = eot_id
        self.eot_text = tok.decode([eot_id])
        self.kwargs = chat_template_kwargs or {}
        self.think_tags = think_tags
        # Control-token strings a tool result must not smuggle into the context: the special
        # tokens and every added token (<tool_call>, <think>, ... are added but not special).
        added = getattr(tok, "added_tokens_decoder", None) or {}
        self._specials = sorted(
            {t for t in getattr(tok, "all_special_tokens", []) if len(t) > 1}
            | {t.content for t in added.values() if len(t.content) > 1},
            key=len,
            reverse=True,
        )

    def continuation_ids(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        calls: list["ToolCall"] | None = None,
    ) -> list[int]:
        """Tokens after an assistant turn's end-of-turn token when `messages` follow it, up to
        and including the next assistant header (the next turn's generation prompt)."""
        assistant: dict[str, Any] = {"role": "assistant", "content": _DUMMY_ANSWER}
        if calls:
            assistant["tool_calls"] = [c.message() for c in calls]
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": _DUMMY_USER}, assistant, *messages],
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
            **self.kwargs,
        )
        start = text.index(_DUMMY_ANSWER)
        end = text.find(self.eot_text, start)
        if end < 0:
            raise ValueError(f"the chat template does not close an assistant turn with {self.eot_text!r}")
        return self.tok.encode(text[end + len(self.eot_text) :], add_special_tokens=False)

    def sanitize(self, text: str) -> str:
        """Break special-token strings (<|im_start|>, <tool_call>, ...) in untrusted text, so a tool
        result tokenizes as plain text and cannot open turns or fake calls."""
        for special in self._specials:
            if special in text:
                text = text.replace(special, special[0] + "​" + special[1:])
        return text

    def truncate(self, text: str, max_tokens: int) -> str:
        """`text` cut to about `max_tokens` tokens, keeping its head and tail."""
        if max_tokens <= 0:
            return text
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        half = max_tokens // 2
        return (
            self.tok.decode(ids[:half])
            + f"\n...[{len(ids) - 2 * half} tokens truncated]...\n"
            + self.tok.decode(ids[-half:])
        )

    def split_reasoning(self, text: str) -> tuple[str, str]:
        """(reasoning, rest): the text up to the closing think tag is reasoning. The opening
        tag may be in the prompt (Qwen3.5's generation prompt ends with it)."""
        open_tag, close_tag = self.think_tags
        if close_tag not in text:
            return "", text
        head, rest = text.split(close_tag, 1)
        return head.replace(open_tag, "", 1).strip(), rest.strip()


# ----------------------------------------------------------------- tool calls


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""

    def message(self) -> dict[str, Any]:
        """The call in OpenAI's message format (what chat templates render)."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class AssistantTurn:
    reasoning: str
    content: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # malformed calls, reported back to the policy

    def message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.calls:
            msg["tool_calls"] = [c.message() for c in self.calls]
        return msg


_TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_XML_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)


def parse_assistant(
    text: str,
    chat: ChatFormat,
    fmt: str = "auto",
    schemas: dict[str, dict] | None = None,
    call_prefix: str = "call",
) -> AssistantTurn:
    """Split a sampled assistant turn (special tokens removed) into reasoning, content and calls."""
    reasoning, rest = chat.split_reasoning(text)
    first = rest.find("<tool_call>")
    content = (rest if first < 0 else rest[:first]).strip()
    turn = AssistantTurn(reasoning, content)
    for k, block in enumerate(_TOOL_CALL.findall(rest if first >= 0 else "")):
        block = block.strip()
        try:
            if fmt == "xml" or (fmt == "auto" and block.startswith("<function=")):
                call = _parse_xml(block, schemas or {})
            else:
                call = _parse_hermes(block)
        except ValueError as e:
            turn.errors.append(f"Invalid tool call: {e}")
            continue
        call.id = f"{call_prefix}_{k}"
        turn.calls.append(call)
    return turn


def _parse_hermes(block: str) -> ToolCall:
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as e:
        raise ValueError(f"the call is not valid JSON ({e.msg} at position {e.pos})") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        raise ValueError('expected {"name": ..., "arguments": {...}}')
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            raise ValueError("arguments is a string that is not valid JSON") from None
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return ToolCall(obj["name"], arguments)


def _parse_xml(block: str, schemas: dict[str, dict]) -> ToolCall:
    match = _XML_FUNCTION.search(block)
    if match is None:
        raise ValueError("expected <function=NAME> inside <tool_call>")
    name, body = match.group(1), match.group(2)
    properties = schemas.get(name, {}).get("function", {}).get("parameters", {}).get("properties", {})
    arguments = {}
    for param, value in _XML_PARAMETER.findall(body):
        arguments[param] = _coerce(value, properties.get(param, {}).get("type"))
    return ToolCall(name, arguments)


def _coerce(value: str, kind: str | None) -> Any:
    """An XML parameter's text as the type its schema declares (text stays text)."""
    if kind in (None, "string"):
        return value
    try:
        if kind == "integer":
            return int(value.strip())
        if kind == "number":
            return float(value.strip())
        if kind == "boolean":
            return value.strip().lower() in ("true", "1", "yes")
        return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        return value


def tool_schema(fn: Callable, name: str | None = None) -> dict:
    """OpenAI function schema from type hints and a Google-style docstring (transformers'
    get_json_schema); falls back to the bare signature when the docstring is missing."""
    from transformers.utils import get_json_schema

    try:
        schema = get_json_schema(fn)
    except Exception:  # noqa: BLE001 — undocumented functions still get a usable schema
        params = inspect.signature(fn).parameters
        types = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
            dict: "object",
            list: "array",
        }
        schema = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (inspect.getdoc(fn) or "").split("\n\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": {
                        p: {"type": types.get(v.annotation, "string")} for p, v in params.items() if p != "self"
                    },
                    "required": [p for p, v in params.items() if p != "self" and v.default is inspect.Parameter.empty],
                },
            },
        }
    schema["function"].pop("return", None)
    if name:
        schema["function"]["name"] = name
    return schema
