"""Environments: what a multi-turn rollout interacts with.

An environment is any class. Per trajectory, the trainer takes an instance from a pool
("construct once, reset often") and calls:

    reset(self, **row) -> str | list[dict] | None      optional, sync or async
        None: the row's prompt as is; str: appended to the last user message;
        a list: the conversation to start from.
    <public methods>                                    the tools, sync or async
        Their JSON schemas come from type hints and a Google-style docstring, as in
        transformers. A tool's return value (str, or anything JSON-serializable) goes back
        to the policy; an exception goes back as "Error: ..." (the policy's mistake, not
        a crash).
    get_reward(self[, messages]) -> float | dict | None  optional, sync or async
        An environment-owned reward, logged as "env" and added with weight 1 (a dict of
        components is logged as env/<name>, each added). With a parameter it receives the
        conversation, to verify from the transcript.
    close(self)                                          optional, sync or async
    done                                                 optional attribute
        Set it to True (e.g. in a submit tool) to end the episode after this turn's tools.

Tools that are not Python methods (HTTP routes, MCP servers, OpenEnv) come from two
optional methods instead, read after reset() so they may depend on the episode:

    tool_schemas(self) -> list[dict]                     OpenAI (or Responses-API flat) schemas
    call_tool(self, name, arguments) -> Any              sync or async

For example, a Python tool whose final answer is graded by hidden tests:

    class PythonEnv:
        def __init__(self, sandbox):
            self.sandbox = sandbox

        def reset(self, tests, **row):
            self.tests = tests

        async def python(self, code: str) -> str:
            '''Run Python code and return what it prints.

            Args:
                code: The program to run.
            '''
            ...

Stateless tools need no class: env.type "tools" with env.tools: ["module:function", ...].
"""

import asyncio
import inspect
import json
from typing import Any, Callable

from palingenesis.rl.chat import tool_schema

_LIFECYCLE = ("reset", "get_reward", "close", "aclose", "tool_schemas", "call_tool")


class ToolEnv:
    """Stateless tools from plain functions (env.type: tools)."""

    def __init__(self, tools: list[Callable]):
        self._tools = {fn.__name__: fn for fn in tools}

    def tool_functions(self) -> dict[str, Callable]:
        return dict(self._tools)


def env_tools(env: Any) -> dict[str, Callable]:
    """An environment's tools: its public methods other than reset, get_reward and close."""
    if isinstance(env, ToolEnv):
        return env.tool_functions()
    tools = {}
    for name, member in inspect.getmembers(type(env)):
        if name.startswith("_") or name in _LIFECYCLE or not callable(member) or isinstance(member, type):
            continue
        if isinstance(
            inspect.getattr_static(type(env), name),
            (staticmethod, classmethod, property),
        ):
            continue
        tools[name] = getattr(env, name)
    return tools


async def call_sync_or_async(fn: Callable, *args, **kwargs) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kwargs))


def normalize_tool(schema: dict) -> dict:
    """A tool schema in chat-template form ({"type": "function", "function": {...}}), from that
    form or the Responses API's flat one ({"type": "function", "name", "parameters", ...})."""
    if "function" in schema:
        return schema
    fields = {k: schema[k] for k in ("name", "description", "parameters") if k in schema}
    return {"type": "function", "function": fields}


def row_tools(row: dict[str, Any], field: str) -> list[dict] | None:
    """A row's own tool schemas (a list, or a JSON string), normalized."""
    tools = row.get(field)
    if isinstance(tools, str):
        tools = json.loads(tools) if tools.strip() else None
    return [normalize_tool(t) for t in tools] if tools else None


class EnvPool:
    """Instances of one environment class, reused across trajectories when it can reset, at
    most `max_concurrent` live at once (0 = no limit; remote environments have a capacity).

    `factory` is the class (or any callable returning an instance), called with `args`.
    Method tools and their schemas are read once from a first instance; environments with
    tool_schemas() are asked per episode."""

    def __init__(self, factory: Callable[..., Any], args: dict[str, Any] | None = None, max_concurrent: int = 0):
        self.factory = factory
        self.args = args or {}
        self.free: list[Any] = []
        self.slots = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        probe = self._new()
        self.reusable = hasattr(probe, "reset")
        self.dynamic = hasattr(probe, "tool_schemas")
        self.schemas = [] if self.dynamic else [tool_schema(fn, name) for name, fn in env_tools(probe).items()]
        self.has_reward = hasattr(probe, "get_reward")
        self.free.append(probe)

    def _new(self) -> Any:
        return self.factory(**self.args)

    async def acquire(self) -> Any:
        if self.slots is not None:
            await self.slots.acquire()
        return self.free.pop() if self.free else self._new()

    async def release(self, env: Any) -> None:
        try:
            if hasattr(env, "close"):
                await call_sync_or_async(env.close)
            if self.reusable:
                env.done = False
                self.free.append(env)
        finally:
            if self.slots is not None:
                self.slots.release()

    async def shutdown(self) -> None:
        """Close what pooled instances hold open across episodes (their aclose(), if any)."""
        for env in self.free:
            if hasattr(env, "aclose"):
                await call_sync_or_async(env.aclose)
        self.free.clear()

    async def episode_schemas(self, env: Any) -> list[dict]:
        """The tool schemas of `env`'s current episode."""
        if not self.dynamic:
            return self.schemas
        return [normalize_tool(t) for t in await call_sync_or_async(env.tool_schemas)]


async def run_tool(env: Any, name: str, arguments: dict[str, Any], timeout: float) -> tuple[str, bool]:
    """(result text, failed): the tool's output for the policy, or the error it caused."""
    if hasattr(env, "call_tool"):
        call = call_sync_or_async(env.call_tool, name, arguments)
    else:
        tools = env_tools(env)
        if name not in tools:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(tools) or 'none'}.", True
        call = call_sync_or_async(tools[name], **arguments)
    try:
        result = await asyncio.wait_for(call, timeout)
    except asyncio.TimeoutError:
        return f"Error: the tool call did not finish within {timeout:g} s.", True
    except TypeError as e:  # wrong or missing arguments
        return f"Error: {e}", True
    except Exception as e:  # noqa: BLE001 — the tool failed on the policy's input: that is the observation
        return f"Error: {type(e).__name__}: {e}", True
    if isinstance(result, str):
        return result, False
    try:
        return json.dumps(result, ensure_ascii=False, default=str), False
    except (TypeError, ValueError):
        return str(result), False
