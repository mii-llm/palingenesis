"""OpenEnv environments (https://github.com/meta-pytorch/OpenEnv) as palingenesis environments.

    env:
      type: palingenesis.rl.envs.openenv:OpenEnvAdapter
      max_concurrent: 64                          # <= the server's max_concurrent_envs
      args:
        base_url: http://localhost:8001           # a running server (docker, HF Space, uvicorn), or
        # env_class: my_pkg.server:MyEnvironment  # the server class itself, in-process (trusted code only)
        reset_fields: [seed]                      # row columns forwarded to reset() to select the task
        reward: last                              # how step rewards combine: last | sum | max

Tools: MCP environments list theirs (list_tools); for a step environment, `step_tool` names
one tool whose arguments are the environment's Action (JSON schema from the server). The
reward arrives on each step's observation and is combined per `reward`; an episode that never
reaches `done` scores `unfinished`. An environment that ends the episode (done) stops the
trajectory.

GRPO needs every rollout of a group to see the same task: forward the task selection with
`reset_fields`. Environments that pick their own task in reset() cannot be trained on as
groups. OpenEnv servers run one session each unless they declare concurrency; keep
env.max_concurrent within the server's capacity. `pip install openenv`.
"""

import json
from typing import Any

from palingenesis.rl.rewards import SkipSample, load_object

_REWARD_MODES = ("last", "sum", "max")


class OpenEnvAdapter:
    _schemas: dict[str, list[dict]] = {}  # discovered tool schemas, per server / class

    def __init__(
        self,
        base_url: str = "",
        env_class: str = "",
        action_class: str = "",
        reset_fields: list[str] | tuple[str, ...] = (),
        prompt: str = "row",
        reward: str = "last",
        unfinished: float | None = 0.0,
        step_tool: str = "",
        message_timeout_s: float = 120.0,
    ):
        if bool(base_url) == bool(env_class):
            raise ValueError("OpenEnvAdapter: set exactly one of base_url (remote) or env_class (in-process)")
        if reward not in _REWARD_MODES:
            raise ValueError(f"OpenEnvAdapter: reward must be one of {_REWARD_MODES}, got {reward!r}")
        if prompt not in ("row", "observation"):
            raise ValueError("OpenEnvAdapter: prompt must be 'row' or 'observation'")
        self.base_url, self.reset_fields, self.prompt = base_url, tuple(reset_fields), prompt
        self.mode, self.unfinished, self.step_tool, self.timeout = reward, unfinished, step_tool, message_timeout_s
        self.key = base_url or env_class
        self.local = load_object(env_class)() if env_class else None
        self.action_class = load_object(action_class) if action_class else None
        self.client = None
        self.rewards: list[float] = []
        self.done = False

    # ------------------------------------------------------------------- tools

    async def tool_schemas(self) -> list[dict]:
        if self.key not in self._schemas:
            self._schemas[self.key] = await self._discover()
        return self._schemas[self.key]

    async def _discover(self) -> list[dict]:
        if self.step_tool:  # a step environment: one tool taking its Action
            if self.local is not None:
                schema = self._action_cls().model_json_schema()
            else:
                schema = await _http_json(self.base_url.replace("ws", "http", 1).rstrip("/") + "/schema")
                schema = schema["action"]
            schema.get("properties", {}).pop("metadata", None)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": self.step_tool,
                        "description": schema.get("description", ""),
                        "parameters": schema,
                    },
                }
            ]
        observation = (await self._step({"type": "list_tools"})).observation
        tools = observation["tools"] if isinstance(observation, dict) else [t.model_dump() for t in observation.tools]
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        action = arguments if self.step_tool else {"type": "call_tool", "tool_name": name, "arguments": arguments}
        result = await self._step(action)
        if result.reward is not None:
            self.rewards.append(float(result.reward))
        self.done = self.done or bool(result.done)
        observation = result.observation
        if not isinstance(observation, dict):
            observation = observation.model_dump()
        error = observation.get("error")
        if error:  # a ToolError: the policy's mistake, returned to it as the tool's answer
            raise RuntimeError(error.get("message", error) if isinstance(error, dict) else error)
        payload = observation.get("result", observation)
        if isinstance(payload, dict) and "content" in payload:  # MCP CallToolResult
            payload = "".join(c.get("text", "") for c in payload["content"] if isinstance(c, dict)) or payload
        return payload if isinstance(payload, str) else json.dumps(payload, default=str)

    # --------------------------------------------------------------- lifecycle

    async def reset(self, **row) -> str | None:
        kwargs = {k: row[k] for k in self.reset_fields if k in row}
        if self.local is not None:
            observation = await self.local.reset_async(**kwargs)
            first = observation.model_dump()
            done = bool(getattr(observation, "done", False))
        else:
            if self.client is None:
                from openenv.core.generic_client import GenericEnvClient

                self.client = GenericEnvClient(base_url=self.base_url, message_timeout_s=self.timeout)
                try:
                    await self.client.connect()  # async mode, on the rollout loop; kept open across episodes
                except Exception as e:  # noqa: BLE001 — a server at capacity or down is infrastructure
                    self.client = None
                    raise SkipSample(f"OpenEnv server {self.base_url} unavailable: {e}") from None
            result = await self.client.reset(**kwargs)
            first, done = result.observation, bool(result.done)
        self.rewards, self.done = [], done
        if self.prompt == "observation":
            return _observation_text(first)
        return None

    async def _step(self, action: dict):
        if self.local is None:
            return await self.client.step(action)
        from openenv.core.client_types import StepResult
        from openenv.core.env_server.serialization import deserialize_action

        observation = await self.local.step_async(deserialize_action(action, self._action_cls()))
        return StepResult(observation=observation.model_dump(), reward=observation.reward, done=observation.done)

    def _action_cls(self):
        if self.action_class is not None:
            return self.action_class
        from openenv.core.env_server.mcp_types import CallToolAction

        return CallToolAction  # MCP environments (deserialize_action routes list_tools / call_tool by type)

    def get_reward(self) -> float | None:
        if not self.done:
            return self.unfinished
        if not self.rewards:
            return None
        return {"last": self.rewards[-1], "sum": sum(self.rewards), "max": max(self.rewards)}[self.mode]

    async def aclose(self) -> None:
        """At shutdown (between episodes the connection is kept: reset() restarts the episode)."""
        if self.client is not None:
            await self.client.close()
        if self.local is not None and hasattr(self.local, "close"):
            self.local.close()


def _observation_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        for key in ("prompt", "question", "text", "message", "observation"):
            if isinstance(observation.get(key), str):
                return observation[key]
        metadata = observation.get("metadata") or {}
        for key in ("prompt", "question"):
            if isinstance(metadata.get(key), str):
                return metadata[key]
    return json.dumps(observation, default=str)


async def _http_json(url: str) -> dict:
    import asyncio
    import urllib.request

    def get():
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    return await asyncio.get_running_loop().run_in_executor(None, get)
