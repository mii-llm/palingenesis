"""NVIDIA NeMo Gym resources servers as palingenesis environments.

A resources server owns a task family's tools and its verifier (the Nemotron-RL datasets
were built for them). Per episode:

    POST /seed_session  the row                         (a session cookie identifies the rollout)
    POST /<tool>        the tool's arguments            -> the tool's output
    POST /verify        the row + the transcript        -> {"reward", "mask_sample", ...}

    env:
      type: palingenesis.rl.envs.nemo_gym:NemoGymAdapter
      max_concurrent: 64
      args: {base_url: http://localhost:8000}
    data: {format: nemo_gym}              # rows keep responses_create_params, which verify reads

The tools are the row's own (responses_create_params.tools). The transcript is sent back in
the Responses-API shape the verifiers expect; `mask_sample` (an infrastructure failure)
skips the sample. Uses only the standard library.
"""

import asyncio
import http.cookiejar
import json
import urllib.request
from typing import Any

from palingenesis.rl.env import row_tools
from palingenesis.rl.rewards import SkipSample


class NemoGymAdapter:
    def __init__(self, base_url: str, timeout: float = 120.0, tools_field: str = "tools"):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.tools_field = tools_field
        self.row: dict[str, Any] = {}
        self.opener = None

    async def _post(self, path: str, payload: Any) -> Any:
        def post():
            request = urllib.request.Request(
                f"{self.base_url}/{path.lstrip('/')}",
                json.dumps(payload, default=str).encode(),
                {"Content-Type": "application/json"},
            )
            with self.opener.open(request, timeout=self.timeout) as r:
                body = r.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body

        return await asyncio.get_running_loop().run_in_executor(None, post)

    async def reset(self, **row) -> None:
        self.row = row
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        try:
            await self._post("/seed_session", _nemo_row(row))
        except Exception as e:  # noqa: BLE001 — the server is infrastructure
            raise SkipSample(f"NeMo Gym server {self.base_url} unavailable: {e}") from None

    def tool_schemas(self) -> list[dict]:
        return row_tools(self.row, self.tools_field) or []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        output = await self._post(f"/{name}", arguments)
        return output if isinstance(output, str) else json.dumps(output, default=str)

    async def get_reward(self, messages: list[dict]) -> float:
        prompt_length = len(self.row.get("messages") or [])
        payload = {**_nemo_row(self.row), "response": {"output": chat_to_responses(messages[prompt_length:])}}
        try:
            verdict = await self._post("/verify", payload)
        except Exception as e:  # noqa: BLE001
            raise SkipSample(f"NeMo Gym verify failed: {e}") from None
        if verdict.get("mask_sample"):
            raise SkipSample(f"NeMo Gym masked the sample: {verdict.get('failure_reason', '')}")
        return float(verdict["reward"])


def _nemo_row(row: dict[str, Any]) -> dict[str, Any]:
    """The row as the server knows it (palingenesis' own columns removed)."""
    return {k: v for k, v in row.items() if k not in ("messages", "tools", "sampling")}


def chat_to_responses(messages: list[dict]) -> list[dict]:
    """Chat messages (the policy's turns and the tool results) as Responses-API output items."""
    items: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            if m.get("reasoning_content"):
                items.append(
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": m["reasoning_content"]}]}
                )
            if m.get("content"):
                items.append(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": m["content"]}]}
                )
            for call in m.get("tool_calls") or []:
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id", ""),
                        "name": function["name"],
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    }
                )
        elif m.get("role") == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id", ""), "output": m["content"]})
    return items
