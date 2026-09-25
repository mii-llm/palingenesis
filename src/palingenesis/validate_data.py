"""Data validation: tool-call consistency checks and role normalization.

Two responsibilities:

1. TOOL VALIDATION: When a sample has both `messages` and `tools` fields,
   verify that every tool_call in assistant messages references a tool that
   exists in the tools list, with matching parameter names.

2. ROLE MAPPING: Normalize non-standard role names (gpt→assistant, human→user,
   from/value→role/content) so the training pipeline sees a consistent format.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ROLE MAPPING — Normalize non-standard chat formats
# ══════════════════════════════════════════════════════════════════════════════

# Common role aliases found in the wild (ShareGPT, Alpaca, OpenAI, etc.)
ROLE_MAP: dict[str, str] = {
    # Standard
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    # ShareGPT style
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    # OpenAI function calling
    "function": "tool",
    # Other common aliases
    "ai": "assistant",
    "model": "assistant",
    "chatbot": "assistant",
    "instruction": "user",
    "input": "user",
    "output": "assistant",
    "response": "assistant",
}

# Common field name aliases for the message structure
CONTENT_FIELD_ALIASES = ("content", "value", "text", "message", "response")
ROLE_FIELD_ALIASES = ("role", "from", "sender", "author", "type")


def normalize_messages(
    sample: dict[str, Any],
    messages_field: str = "messages",
    role_map: dict[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    """Normalize a sample's messages to standard {role, content} format.

    Handles:
    - Non-standard role names (gpt→assistant, human→user)
    - Non-standard field names (from→role, value→content)
    - ShareGPT format ({"from": "human", "value": "..."})
    - Single-field conversations (list of dicts with any key combo)

    - A conversation stored as a JSON string (common in exports whose tool-call
      arguments differ in shape from row to row, which Arrow cannot store natively)
    - Tool-call arguments stored as JSON strings (the OpenAI wire format): parsed to
      dicts, which is what chat templates iterate over (Qwen3.x, Llama 3.x, ...)
    - Other per-message keys (`tool_call_id`, `name`, `loss`, ...) are kept; keys whose
      value is None (Arrow fills them in for messages that lack them) are dropped

    Returns normalized messages list, or None if the sample is unparseable.
    """
    if role_map is None:
        role_map = ROLE_MAP

    raw = sample.get(messages_field)
    if raw is None:
        # Try common alternative field names
        for alt in ("conversations", "conversation", "chat", "dialogue", "turns"):
            raw = sample.get(alt)
            if raw is not None:
                break

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not raw or not isinstance(raw, list):
        return None

    normalized = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue

        turn = {k: v for k, v in turn.items() if v is not None}

        # Find the role field
        role_key = next((alias for alias in ROLE_FIELD_ALIASES if alias in turn), None)
        if role_key is None:
            continue
        role_raw = turn[role_key]

        # Normalize role name
        role = role_map.get(str(role_raw).lower().strip(), str(role_raw).lower().strip())

        # Find the content field
        content_key = next((alias for alias in CONTENT_FIELD_ALIASES if alias in turn), None)
        content = turn[content_key] if content_key is not None else None

        if content is None:
            # Maybe the content is the only other key besides role
            non_role_keys = [k for k in turn if k not in ROLE_FIELD_ALIASES]
            if len(non_role_keys) == 1:
                content_key = non_role_keys[0]
                content = turn[content_key]

        if role == "tool" and isinstance(content, (dict, list)) and not _is_content_parts(content):
            content = json.dumps(content, ensure_ascii=False)

        # Build normalized message. Keys the template may read (tool_call_id, name, ...)
        # and per-message training flags (loss) pass through unchanged.
        msg: dict[str, Any] = {
            k: v for k, v in turn.items()
            if k not in (role_key, content_key, "reasoning", "reasoning_content", "think", "function_call")
        }
        msg["role"] = role
        msg["content"] = content or ""

        # `reasoning` is the canonical field (OpenAI/vLLM); `reasoning_content` and
        # `think` are accepted as legacy input. As vLLM does before rendering, the
        # trace is exposed under both keys: templates such as Qwen3.5's still read
        # `message.reasoning_content` in their Jinja.
        reasoning = next((turn[k] for k in ("reasoning", "reasoning_content", "think")
                          if isinstance(turn.get(k), str) and turn[k].strip()), None)
        if role == "assistant" and isinstance(msg["content"], str):
            baked, rest = split_leading_think(msg["content"])
            if baked is not None:
                # The formatting is baked into the content: its LEADING <think> block is the
                # turn's reasoning (an explicit reasoning field wins), never a second block.
                # The content as given is kept for templates that do not render reasoning
                # (restore_baked_think), where it must stay verbatim.
                msg["_baked_content"] = msg["content"]
                msg["content"] = rest
                if reasoning is None and baked.strip():
                    reasoning = baked
        if reasoning is not None:
            msg["reasoning"] = reasoning
            msg["reasoning_content"] = reasoning
        elif role == "assistant" and "</think>" in (msg["content"] if isinstance(msg["content"], str) else ""):
            # Think tags inside the answer are text: an explicit (empty) reasoning field stops
            # templates such as Qwen3.x's from splitting the content at "</think>".
            msg["reasoning"] = msg["reasoning_content"] = ""
        if "function_call" in turn:
            # OpenAI legacy format → normalize to tool_calls
            msg["tool_calls"] = [{"type": "function", "function": turn["function_call"]}]
        if msg.get("tool_calls"):
            msg["tool_calls"] = [_parse_tool_call(call) for call in msg["tool_calls"]]
        else:
            msg.pop("tool_calls", None)

        normalized.append(msg)

    return normalized if normalized else None


_LEADING_THINK = re.compile(r"\A\s*<think>(.*?)</think>[ \t]*\n*", re.DOTALL)


def restore_baked_think(messages: list[dict]) -> list[dict]:
    """Messages for a template that does not render reasoning: a turn whose content came with
    its <think> block baked in gets that content back verbatim (the block is its reasoning
    text, which such a template would otherwise drop)."""
    out = []
    for m in messages:
        if "_baked_content" in m:
            raw = m["_baked_content"]
            m = {k: v for k, v in m.items() if k not in ("_baked_content", "reasoning", "reasoning_content")}
            m["content"] = raw
        out.append(m)
    return out


def split_leading_think(content: str) -> tuple[str | None, str]:
    """(reasoning, rest) when `content` starts with a closed <think>...</think> block, else
    (None, content). Only a leading block counts: think tags later in the text are text."""
    m = _LEADING_THINK.match(content)
    if m is None:
        return None, content
    return m.group(1).strip("\n"), content[m.end():]


def normalize_tools(tools: Any) -> list[dict] | None:
    """Tool definitions for the chat template's `tools=` argument: a list (possibly
    stored as a JSON string), or None when there are none."""
    if isinstance(tools, str):
        if not tools.strip():
            return None
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        return None
    tools = [t for t in tools if isinstance(t, dict)]
    return tools or None


def is_trained_message(msg: dict[str, Any]) -> bool:
    """Per-message training flag: `"loss": false` (or `"weight": 0`) marks a turn that
    stays in the context but receives no loss, e.g. a greeting or a failed attempt."""
    if msg.get("loss") is False:
        return False
    weight = msg.get("weight")
    return not (isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight == 0)


def _is_content_parts(content: Any) -> bool:
    """Multimodal-style content: a list of {"type": ...} parts, which templates render."""
    return isinstance(content, list) and all(isinstance(p, dict) and "type" in p for p in content)


def _parse_tool_call(call: Any) -> Any:
    """Tool call with its `arguments` as a dict. The OpenAI wire format stores them as a
    JSON string; chat templates iterate over them as a mapping (`arguments|items`) and
    fail on a string. Arguments that are not valid JSON are left untouched."""
    if not isinstance(call, dict):
        return call
    call = {k: v for k, v in call.items() if v is not None}
    func = call.get("function")
    target = func if isinstance(func, dict) else call
    args = target.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            target = {**target, "arguments": parsed}
            if isinstance(func, dict):
                call = {**call, "function": target}
            else:
                call = target
    return call


# ══════════════════════════════════════════════════════════════════════════════
# TOOL VALIDATION — Check tool_calls match declared tools
# ══════════════════════════════════════════════════════════════════════════════


def validate_tool_calls(
    sample: dict[str, Any],
    messages_field: str = "messages",
    tools_field: str = "tools",
) -> list[str]:
    """Validate that all tool_calls in messages reference valid tools.

    Checks:
    1. Every tool_call function name exists in the tools list
    2. Every parameter in the tool_call exists in the tool's parameter schema
    3. No required parameters are missing from the tool_call

    Args:
        sample: A data sample with messages and tools fields
        messages_field: Field containing the conversation
        tools_field: Field containing the tool definitions

    Returns:
        List of validation error strings (empty = valid)
    """
    messages = sample.get(messages_field, [])
    tools = sample.get(tools_field, [])

    if not tools:
        return []  # No tools declared — nothing to validate

    # Build tool registry: name → {parameters: {name: {required: bool}}}
    tool_registry = _build_tool_registry(tools)
    if not tool_registry:
        return []  # Tools field exists but couldn't parse it

    errors = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", msg.get("from", ""))
        if str(role).lower() not in ("assistant", "gpt", "ai", "model"):
            continue

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            # Check for function_call (OpenAI legacy)
            fc = msg.get("function_call")
            if fc:
                tool_calls = [{"function": fc}]

        for j, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue

            func = call.get("function", call)
            if not isinstance(func, dict):
                continue

            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")

            # Parse arguments
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"turn {i}, call {j}: arguments is not valid JSON: {args_raw[:100]}")
                    continue
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}

            # Check 1: tool name exists
            if name not in tool_registry:
                available = ", ".join(sorted(tool_registry.keys())[:5])
                errors.append(
                    f"turn {i}, call {j}: tool '{name}' not found in tools list. "
                    f"Available: [{available}{'...' if len(tool_registry) > 5 else ''}]"
                )
                continue

            tool_def = tool_registry[name]

            # Check 2: parameters exist in schema
            if tool_def["parameters"]:
                valid_params = set(tool_def["parameters"].keys())
                for param_name in args:
                    if param_name not in valid_params:
                        errors.append(
                            f"turn {i}, call {j}: tool '{name}' has no parameter '{param_name}'. "
                            f"Valid: {sorted(valid_params)}"
                        )

                # Check 3: required parameters present
                for param_name, param_info in tool_def["parameters"].items():
                    if param_info.get("required", False) and param_name not in args:
                        errors.append(
                            f"turn {i}, call {j}: tool '{name}' missing required parameter '{param_name}'"
                        )

    return errors


def _build_tool_registry(tools: list[dict]) -> dict[str, dict]:
    """Parse tool definitions into a lookup table.

    Supports multiple formats:
    - OpenAI function calling: {"type": "function", "function": {"name": ..., "parameters": ...}}
    - Simple: {"name": ..., "parameters": ...}
    - Anthropic style: {"name": ..., "input_schema": ...}
    """
    registry = {}

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # OpenAI format
        if "function" in tool and isinstance(tool["function"], dict):
            func = tool["function"]
            name = func.get("name", "")
            params = _extract_params(func.get("parameters", {}))
        # Anthropic format
        elif "input_schema" in tool:
            name = tool.get("name", "")
            params = _extract_params(tool.get("input_schema", {}))
        # Simple format
        elif "name" in tool:
            name = tool["name"]
            params = _extract_params(tool.get("parameters", {}))
        else:
            continue

        if name:
            registry[name] = {"parameters": params}

    return registry


def _extract_params(schema: dict) -> dict[str, dict]:
    """Extract parameter names and required status from a JSON schema."""
    if not isinstance(schema, dict):
        return {}

    properties = schema.get("properties", {})
    required_list = set(schema.get("required", []))

    params = {}
    for param_name, param_def in properties.items():
        params[param_name] = {
            "required": param_name in required_list,
            "type": param_def.get("type", "any") if isinstance(param_def, dict) else "any",
        }

    return params


# ══════════════════════════════════════════════════════════════════════════════
# BATCH VALIDATION — Run on a whole dataset
# ══════════════════════════════════════════════════════════════════════════════


def validate_dataset(
    samples: list[dict[str, Any]],
    messages_field: str = "messages",
    tools_field: str = "tools",
    max_errors: int = 50,
) -> dict[str, Any]:
    """Validate an entire dataset for tool-call consistency.

    Returns a report dict with:
    - total: number of samples checked
    - valid: number passing all checks
    - errors: list of (sample_index, error_message) tuples
    - tool_coverage: which tools are actually called vs declared
    """
    errors = []
    tools_called: dict[str, int] = {}
    tools_declared: set[str] = set()
    samples_with_tools = 0

    for idx, sample in enumerate(samples):
        if len(errors) >= max_errors:
            break

        tools = sample.get(tools_field, [])
        if tools:
            samples_with_tools += 1
            registry = _build_tool_registry(tools)
            tools_declared.update(registry.keys())

        sample_errors = validate_tool_calls(sample, messages_field, tools_field)
        for err in sample_errors:
            errors.append((idx, err))

        # Track which tools are actually called
        messages = sample.get(messages_field, [])
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for call in msg.get("tool_calls", []):
                if isinstance(call, dict):
                    func = call.get("function", call)
                    if isinstance(func, dict):
                        name = func.get("name", "")
                        if name:
                            tools_called[name] = tools_called.get(name, 0) + 1

    # Tools declared but never called
    never_called = tools_declared - set(tools_called.keys())

    return {
        "total": len(samples),
        "valid": len(samples) - len(set(idx for idx, _ in errors)),
        "samples_with_tools": samples_with_tools,
        "errors": errors[:max_errors],
        "tools_declared": sorted(tools_declared),
        "tools_called": dict(sorted(tools_called.items(), key=lambda x: -x[1])),
        "tools_never_called": sorted(never_called),
        "truncated": len(errors) >= max_errors,
    }
