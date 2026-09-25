"""Dataset row formats: whatever a dataset ships, converted to palingenesis rows at load time.

A palingenesis RL row is flat: the prompt (`messages`, a chat, or `prompt`, one user turn),
optionally `tools` (tool schemas) and `verifier` (the names of the rewards that apply to
it), and any other columns, which reach the rewards and the environment unchanged:

    {"messages": [...], "tools": [...], "verifier": ["math"], "answer": "42", ...}

Converters (data.format, "auto" detects them):

  nemo_gym   NVIDIA NeMo Gym rows (the Nemotron-RL datasets): the prompt lives in
             responses_create_params (OpenAI Responses API: `instructions` + `input`, a
             string or a list of items; flat tool schemas). The Responses items become chat
             messages: function calls -> assistant tool_calls, their outputs -> tool
             messages, reasoning -> reasoning_content. The verifier's fields (ground_truth,
             expected_answer, ...) stay top-level columns, and so does responses_create_params.
"""

import json
from typing import Any

from palingenesis.rl.env import normalize_tool

FORMATS = ("auto", "chat", "nemo_gym")


def detect(row: dict[str, Any]) -> str:
    return "nemo_gym" if isinstance(row.get("responses_create_params"), (dict, str)) else "chat"


def convert(rows: list[dict[str, Any]], fmt: str = "auto") -> list[dict[str, Any]]:
    """Rows in palingenesis form (a no-op for rows that already are)."""
    if not rows:
        return rows
    fmt = detect(rows[0]) if fmt == "auto" else fmt
    if fmt == "nemo_gym":
        return [from_nemo_gym(row) for row in rows]
    return rows


def from_nemo_gym(row: dict[str, Any]) -> dict[str, Any]:
    params = row["responses_create_params"]
    if isinstance(params, str):
        params = json.loads(params)
    out = dict(row)  # responses_create_params stays: NeMo Gym verifiers read it
    messages = []
    if params.get("instructions"):
        messages.append({"role": "system", "content": params["instructions"]})
    source = params.get("input") or []
    messages += [{"role": "user", "content": source}] if isinstance(source, str) else responses_to_chat(source)
    out["messages"] = messages
    if params.get("tools"):
        out["tools"] = [normalize_tool(t) for t in params["tools"] if t.get("type", "function") == "function"]
    sampling = {k: params[k] for k in ("temperature", "top_p", "max_output_tokens") if k in params}
    if sampling:
        out["sampling"] = sampling  # informational: the run's rollout settings apply
    return out


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def responses_to_chat(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI Responses-API input items as chat-completions messages."""
    messages: list[dict[str, Any]] = []
    reasoning = ""
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            message = {"role": item.get("role", "user"), "content": _text(item.get("content"))}
            if message["role"] == "developer":
                message["role"] = "system"
            if message["role"] == "assistant" and reasoning:
                message["reasoning_content"], reasoning = reasoning, ""
            messages.append(message)
        elif kind == "reasoning":
            reasoning += _text(item.get("summary")) or _text(item.get("content"))
        elif kind == "function_call":
            arguments = item.get("arguments") or "{}"
            call = {
                "id": item.get("call_id") or item.get("id", ""),
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
                },
            }
            last = messages[-1] if messages else None
            if last is not None and last["role"] == "assistant":  # consecutive calls: one assistant turn
                last.setdefault("tool_calls", []).append(call)
            else:
                message = {"role": "assistant", "content": "", "tool_calls": [call]}
                if reasoning:
                    message["reasoning_content"], reasoning = reasoning, ""
                messages.append(message)
        elif kind == "function_call_output":
            output = item.get("output", "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
    return messages
