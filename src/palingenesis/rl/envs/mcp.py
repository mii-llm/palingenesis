"""MCP servers (https://modelcontextprotocol.io) as the tools of an RL environment.

    env:
      type: palingenesis.rl.envs.mcp:MCPEnv
      max_concurrent: 256                       # trajectories in flight against the server(s)
      args:
        server: http://localhost:8000/mcp       # Streamable HTTP; or a command (stdio):
        # server: [python, my_server.py]
        # servers: {search: http://search:8000/mcp, code: [python, code_server.py]}
        tools: [search, fetch]                  # optional allowlist (default: every tool)

The policy sees the server's tools/list as its tool schemas and each tools/call result as the
observation. One connection per server serves every trajectory: MCP (2026-07-28) is a stateless
protocol, so requests from concurrent episodes simply interleave. Servers on the earlier,
session-based revisions are detected and spoken to as well (the SDK's "auto" mode).

Episode state. A stateless server keeps state behind an explicit handle that a creation tool
returns and later calls take as an argument (the spec's "stateful tools" pattern). Set
`state_tool` to have reset() create the episode's handle (called with the row's
`reset_fields`); the handle argument `state_arg` is then hidden from the policy's schemas and
filled in on every call, so the policy does not have to learn to carry it.

Rewards. Rewards are usually palingenesis reward functions over the transcript. A server can
also grade the episode itself: `reward_tool` is called at the end (hidden from the policy) with
the handle, and with `answer` (the last assistant message) and/or `messages` when its input
schema has them. Its structuredContent (a number, {"reward": x}, or a dict of components) or
text (a number) is the environment reward.

Content. Text and embedded text resources go to the policy as text; images, audio and binary
resources are replaced by a short placeholder (the policy reads text). A result with
isError: true is an observation the policy can correct ("Error: ..."), and counts as a tool
error; protocol errors (an unknown tool, a malformed call) are too.

`pip install "palingenesis[mcp]"`
"""

import asyncio
import inspect
import json
import re
from typing import Any

from palingenesis.rl.env import ToolFailed
from palingenesis.rl.rewards import SkipSample, load_object

_SEP = "__"  # server/tool separator when several servers are combined


class _Connection:
    """One MCP client connection, owned by a background task on the event loop that uses it
    (the SDK's client is an async context manager tied to the task that enters it)."""

    def __init__(self, spec: Any, timeout: float, headers: dict[str, str] | None):
        self.spec, self.timeout, self.headers = spec, timeout, headers
        self.client = None
        self.tools: list[Any] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._task = asyncio.get_running_loop().create_task(self._run())

    def _target(self) -> Any:
        spec = self.spec
        if isinstance(spec, (list, tuple)):  # a command: the server runs as our subprocess (stdio)
            from mcp import StdioServerParameters

            return StdioServerParameters(command=str(spec[0]), args=[str(a) for a in spec[1:]])
        if isinstance(spec, str) and spec.startswith(("http://", "https://")):
            if not self.headers:
                return spec
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            return streamable_http_client(spec, http_client=httpx.AsyncClient(headers=self.headers))
        if isinstance(spec, str):  # "module:attr": an MCPServer (or a factory), in-process
            obj = load_object(spec)
            return obj() if callable(obj) and not hasattr(obj, "call_tool") else obj
        return spec  # an MCPServer / Transport object passed programmatically

    async def _run(self) -> None:
        from mcp import Client
        from mcp_types import Implementation

        try:
            async with Client(
                self._target(),
                read_timeout_seconds=self.timeout,
                client_info=Implementation(name="palingenesis", version="rl"),
            ) as client:
                self.client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 — reported to every waiter
            self._error = e
        finally:
            self.client = None
            self._ready.set()

    async def ready(self) -> Any:
        await self._ready.wait()
        if self.client is None:
            raise RuntimeError(f"MCP server {self.spec!r}: could not connect: {self._error!r}") from self._error
        return self.client

    async def list_tools(self) -> list[Any]:
        if self.tools is None:
            client = await self.ready()
            tools, cursor = [], None
            while True:
                page = await client.list_tools(cursor=cursor)
                tools.extend(page.tools)
                cursor = page.next_cursor
                if not cursor:
                    break
            self.tools = tools
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        client = await self.ready()
        return await client.call_tool(name, arguments)

    async def close(self) -> None:
        self._stop.set()
        await asyncio.gather(self._task, return_exceptions=True)


class MCPEnv:
    # (servers, loop) -> connections, shared by every instance on that loop
    _connections: dict[tuple[str, int], dict[str, _Connection]] = {}

    def __init__(
        self,
        server: str | list[str] = "",
        servers: dict[str, str | list[str]] | None = None,
        tools: list[str] | tuple[str, ...] = (),
        state_tool: str = "",
        state_arg: str = "",
        reset_fields: list[str] | tuple[str, ...] = (),
        reward_tool: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        if bool(server) == bool(servers):
            raise ValueError("MCPEnv: set exactly one of server (one MCP server) or servers ({name: server})")
        if bool(state_tool) != bool(state_arg):
            raise ValueError("MCPEnv: state_tool and state_arg go together (the tool creating the handle, its name)")
        self.servers = dict(servers) if servers else {"": server}
        self.allow, self.state_tool, self.state_arg = set(tools), state_tool, state_arg
        self.reset_fields, self.reward_tool = tuple(reset_fields), reward_tool
        self.headers, self.timeout = headers, timeout
        self.key = json.dumps(self.servers, sort_keys=True)
        self.routes: dict[str, tuple[str, str]] = {}  # policy-visible name -> (server, tool)
        self.hidden: dict[str, dict[str, Any]] = {}  # the tool's input schema, per hidden tool
        self.schemas: list[dict] | None = None
        self.handle: Any = None
        self.done = False

    # ------------------------------------------------------------- connection

    def _connected(self) -> dict[str, _Connection]:
        key = (self.key, id(asyncio.get_running_loop()))
        if key not in self._connections:
            self._connections[key] = {
                name: _Connection(spec, self.timeout, self.headers) for name, spec in self.servers.items()
            }
        return self._connections[key]

    async def _call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        return await self._connected()[server].call(tool, arguments)

    # ------------------------------------------------------------------ tools

    async def tool_schemas(self) -> list[dict]:
        if self.schemas is None:
            schemas, routes = [], {}
            for server, connection in self._connected().items():
                for tool in await connection.list_tools():
                    if tool.name in (self.state_tool, self.reward_tool):
                        self.hidden[tool.name] = tool.input_schema or {}
                        continue
                    if self.allow and tool.name not in self.allow:
                        continue
                    name = f"{server}{_SEP}{tool.name}" if server else tool.name
                    routes[name] = (server, tool.name)
                    schemas.append(self._schema(name, tool))
            if self.state_tool and self.state_tool not in self.hidden:
                raise ValueError(f"MCPEnv: state_tool {self.state_tool!r} is not a tool of the server")
            if self.reward_tool and self.reward_tool not in self.hidden:
                raise ValueError(f"MCPEnv: reward_tool {self.reward_tool!r} is not a tool of the server")
            self.routes, self.schemas = routes, schemas
        return self.schemas

    def _schema(self, name: str, tool: Any) -> dict:
        parameters = _untitled(tool.input_schema or {"type": "object", "properties": {}})
        description = inspect.cleandoc(tool.description or tool.title or "")
        if self.state_arg and self.state_arg in parameters.get("properties", {}):
            parameters["properties"] = {k: v for k, v in parameters["properties"].items() if k != self.state_arg}
            parameters["required"] = [r for r in parameters.get("required", []) if r != self.state_arg]
            # the handle's line in a docstring-style description ("basket_id: The basket.")
            description = re.sub(rf"(?m)^[ \t]*{re.escape(self.state_arg)}\b[^:\n]*:.*\n?", "", description)
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        await self.tool_schemas()
        if name not in self.routes:
            raise ToolFailed(f"unknown tool {name!r}. Available tools: {', '.join(self.routes) or 'none'}.")
        server, tool = self.routes[name]
        if self.handle is not None:
            arguments = {**arguments, self.state_arg: self.handle}
        result = await self._call(server, tool, arguments)
        text = _result_text(result)
        if result.is_error:
            raise ToolFailed(text or "the tool failed")
        return text

    # ---------------------------------------------------------------- episode

    async def reset(self, **row) -> None:
        self.done, self.handle = False, None
        if self.state_tool:
            await self.tool_schemas()
            server = self._server_of(self.state_tool)
            fields = {k: row[k] for k in self.reset_fields if k in row}
            result = await self._call(server, self.state_tool, fields)
            if result.is_error:
                raise RuntimeError(f"MCPEnv: {self.state_tool} failed: {_result_text(result)}")
            self.handle = _handle(result, self.state_arg)
        return None

    async def get_reward(self, messages: list[dict]) -> float | dict[str, float] | None:
        if not self.reward_tool:
            return None
        await self.tool_schemas()
        properties = self.hidden[self.reward_tool].get("properties", {})
        arguments: dict[str, Any] = {}
        if self.handle is not None:
            arguments[self.state_arg] = self.handle
        if "answer" in properties:
            arguments["answer"] = next(
                (m.get("content") or "" for m in reversed(messages) if m["role"] == "assistant"), ""
            )
        if "messages" in properties:
            arguments["messages"] = messages
        result = await self._call(self._server_of(self.reward_tool), self.reward_tool, arguments)
        if result.is_error:
            raise SkipSample(f"{self.reward_tool} failed: {_result_text(result)}")
        return _reward(result)

    def _server_of(self, tool: str) -> str:
        """The server exposing a hidden tool (the only server, or the first that lists it)."""
        for server, connection in self._connected().items():
            if connection.tools and any(t.name == tool for t in connection.tools):
                return server
        return next(iter(self.servers))

    async def aclose(self) -> None:
        """Close this loop's connections (the pool calls it once, at shutdown)."""
        key = (self.key, id(asyncio.get_running_loop()))
        for connection in self._connections.pop(key, {}).values():
            await connection.close()


# ---------------------------------------------------------------- results


def _untitled(schema: Any) -> Any:
    """A JSON schema without "title" annotations (pydantic adds one per property and model:
    prompt tokens in every rollout that tell the policy nothing). A property named "title"
    is kept."""
    if isinstance(schema, list):
        return [_untitled(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {k: _untitled(v) for k, v in value.items()}
        else:
            out[key] = _untitled(value)
    return out


def _result_text(result: Any) -> str:
    parts = []
    for item in result.content or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind == "resource":
            resource = item.resource
            text = getattr(resource, "text", None)
            parts.append(text if text is not None else f"[binary resource {resource.uri}]")
        elif kind == "resource_link":
            parts.append(f"[resource {item.uri}]")
        else:  # image, audio: the policy reads text
            parts.append(f"[{kind} omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    return "\n".join(parts)


def _handle(result: Any, arg: str) -> Any:
    """The state handle a creation tool returned: structuredContent[arg], a bare structured
    value, or the text."""
    structured = _structured(result)
    if isinstance(structured, dict) and arg in structured:
        return structured[arg]
    if isinstance(structured, dict) and len(structured) == 1:
        return next(iter(structured.values()))
    if structured is not None and not isinstance(structured, (dict, list)):
        return structured
    text = _result_text(result).strip()
    if not text:
        raise RuntimeError(f"MCPEnv: the state tool returned no handle for {arg!r}")
    return text


def _structured(result: Any) -> Any:
    """structuredContent, or the JSON text a server returned instead (the spec asks servers to
    mirror structured results as text; some return only the text)."""
    if result.structured_content is not None:
        return result.structured_content
    text = _result_text(result).strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return None


def _reward(result: Any) -> float | dict[str, float]:
    structured = _structured(result)
    if isinstance(structured, (int, float)) and not isinstance(structured, bool):
        return float(structured)
    if isinstance(structured, dict):
        if "reward" in structured and len(structured) == 1:
            return float(structured["reward"])
        numbers = {
            k: float(v) for k, v in structured.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numbers:
            return numbers
    text = _result_text(result).strip()
    try:
        return float(text)
    except ValueError:
        raise SkipSample(f"the reward tool returned no number: {text[:200]!r}") from None
