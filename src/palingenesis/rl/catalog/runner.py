"""Rows of a catalogue environment, and episodes played by scripted policies (no model).

play() drives an environment the way the RL pipeline's rollout loop does (palingenesis.rl.pipeline): the row's
prompt, reset(**row), the episode's tools, tool calls through run_tool (unknown tools refused, errors returned as
observations), text-protocol actions, `done`, the turn limit, then get_reward. Only the policy differs: a Python
function instead of sampled tokens. It is what the audit (catalog.audit) and the CPU throughput numbers run on.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec, Policy


def _resolve(ref: Any) -> Any:
    if isinstance(ref, str):
        from palingenesis.rl.rewards import load_object

        return load_object(ref)
    return ref


def load_rows(spec: EnvSpec, dataset: str = "", n: int = 0, seed: int = 0, **options) -> list[dict]:
    """`n` rows of one of the environment's datasets (0: every row; 32 from a generator).

    dataset: a key of spec.datasets (default: the first), or a path to a file of rows in the environment's own
    format. Hub datasets are downloaded by `datasets` when first used and converted row by row; rows the
    environment cannot verify are dropped. Every row gets an "id" when it has none."""
    name = dataset or spec.default_dataset
    source = spec.datasets.get(name)
    if source is None:
        from pathlib import Path

        if not Path(name).exists():
            raise KeyError(
                f"{spec.name} has no dataset {name!r} (and no such file); datasets: {', '.join(spec.datasets)}"
            )
        source = DatasetSpec(path=name)
    convert = _resolve(source.convert)
    if source.kind == "generator":
        rows = _resolve(source.generator)(n or 32, seed, **{**source.options, **options})
    elif source.kind == "hub":
        from datasets import load_dataset

        kwargs = {"revision": source.revision} if source.revision else {}
        data = load_dataset(source.hub_id, source.config or None, split=source.split, **kwargs)
        if seed:
            data = data.shuffle(seed=seed)
        rows = []
        for raw in data:
            row = convert(dict(raw)) if convert else dict(raw)
            if row is not None:
                rows.append(row)
            if n and len(rows) >= n:
                break
    else:
        from pathlib import Path

        path = Path(source.path)
        if path.is_dir():  # a directory of Harbor tasks
            from palingenesis.rl.envs.harbor import harbor_rows

            rows = harbor_rows(path)
        else:
            from palingenesis.rl.data import load_rows as read

            rows = read(str(path))
        if convert:
            rows = [r for r in (convert(dict(row)) for row in rows) if r is not None]
        if seed:
            import random

            random.Random(seed).shuffle(rows)
        if n:
            rows = rows[:n]
    for i, row in enumerate(rows):
        row.setdefault("id", f"{spec.name}/{name}/{i}")
    return rows


def make_pool(spec: EnvSpec, max_concurrent: int = 0, env_args: dict | None = None):
    from palingenesis.rl.env import EnvPool

    return EnvPool(spec.env_class(), {**spec.env_args, **(env_args or {})}, max_concurrent)


async def episode_tools(spec: EnvSpec, row: dict | None = None) -> list[dict]:
    """The tool schemas of an episode on `row` (the default dataset's first row when None)."""
    from palingenesis.rl.env import call_sync_or_async, row_tools

    pool = make_pool(spec)
    env = await pool.acquire()
    try:
        if pool.dynamic or row is not None:
            row = row if row is not None else load_rows(spec, n=1)[0]
            if hasattr(env, "reset"):
                await call_sync_or_async(env.reset, **row)
        return await pool.episode_schemas(env) or (row_tools(row, "tools") if row else None) or []
    finally:
        await pool.release(env)
        await pool.shutdown()


@dataclass
class Episode:
    """One scripted episode: what a Trajectory carries after scoring, without tokens."""

    row: dict
    messages: list[dict] = field(default_factory=list)
    reward: float | None = None  # None: the environment could not score it
    rewards: dict[str, float | None] = field(default_factory=dict)
    finish: str = ""  # stop | turns | env_done | error
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    error: str = ""
    seconds: float = 0.0

    @property
    def steps(self) -> int:
        """Environment steps: tool calls, plus the final reply the verifier scores."""
        return self.tool_calls + 1


def _turn(reply: Any, turn: int) -> tuple[str, list[tuple[str, dict]]]:
    if reply is None:
        return "", []
    if isinstance(reply, str):
        return reply, []
    if isinstance(reply, dict):
        return str(reply.get("content") or ""), [(n, dict(a or {})) for n, a in reply.get("calls") or []]
    if isinstance(reply, tuple) and len(reply) == 2 and isinstance(reply[0], str):
        return "", [(reply[0], dict(reply[1] or {}))]
    return "", [(n, dict(a or {})) for n, a in reply]


async def play(
    pool, row: dict, policy: Policy, max_turns: int = 16, tool_timeout: float = 60.0, group: int = 0
) -> Episode:
    """One episode of `row` in an environment from `pool` (an rl.env.EnvPool), played by `policy`."""
    from palingenesis.rl.data import prompt_messages
    from palingenesis.rl.env import ToolOutput, call_sync_or_async, row_tools, run_tool
    from palingenesis.rl.rewards import SkipSample

    start = time.perf_counter()
    messages = prompt_messages(row)
    episode = Episode(row, messages)
    env = await pool.acquire()
    try:
        if hasattr(env, "reset"):
            if hasattr(env, "episode_group"):
                env.episode_group = group
            opening = await call_sync_or_async(env.reset, **row)
            if isinstance(opening, str) and opening:
                last = next(m for m in reversed(messages) if m["role"] == "user")
                last["content"] = f"{last['content']}\n\n{opening}" if last["content"] else opening
            elif isinstance(opening, list):
                messages[:] = [dict(m) for m in opening]
        schemas = await pool.episode_schemas(env) or row_tools(row, "tools") or []
        exposed = {s["function"]["name"] for s in schemas}
        row_turns = row.get("max_turns")
        max_turns = int(row_turns) if row_turns else max_turns
        for turn in range(max_turns):
            reply = policy(row, messages)
            if inspect.isawaitable(reply):
                reply = await reply
            content, calls = _turn(reply, turn)
            text_protocol = False
            if not calls and hasattr(env, "text_actions"):
                calls = [(n, dict(a or {})) for n, a in await call_sync_or_async(env.text_actions, content) or []]
                text_protocol = bool(calls)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if calls and not text_protocol:
                message["tool_calls"] = [
                    {"id": f"call_{turn}_{k}", "type": "function", "function": {"name": name, "arguments": args}}
                    for k, (name, args) in enumerate(calls)
                ]
            messages.append(message)
            episode.turns += 1
            if not calls:
                episode.finish = "stop"
                break
            if turn == max_turns - 1:
                episode.finish = "turns"
                break
            results = await asyncio.gather(
                *(run_tool(env, n, a, tool_timeout, None if text_protocol else exposed) for n, a in calls)
            )
            texts = [r.text if isinstance(r, ToolOutput) else r for r, _ in results]
            episode.tool_calls += len(calls)
            episode.tool_errors += sum(failed for _, failed in results)
            if text_protocol:
                render = getattr(env, "observation_text", None)
                body = render([(n, t) for (n, _), t in zip(calls, texts)]) if render else "\n\n".join(texts)
                messages.append({"role": "user", "content": body})
            else:
                messages.extend(
                    {"role": "tool", "tool_call_id": f"call_{turn}_{k}", "name": name, "content": text}
                    for k, ((name, _), text) in enumerate(zip(calls, texts))
                )
            if getattr(env, "done", False):
                episode.finish = "env_done"
                break
        if hasattr(env, "get_reward"):
            wants = bool(inspect.signature(env.get_reward).parameters)
            try:
                value = await call_sync_or_async(env.get_reward, *((messages,) if wants else ()))
            except SkipSample as skipped:
                value, episode.error = None, f"skipped: {skipped}"
            if isinstance(value, dict):
                episode.rewards = {k: None if v is None else float(v) for k, v in value.items()}
                paid = [v for k, v in episode.rewards.items() if v is not None and not k.startswith("metric/")]
                episode.reward = sum(paid) if paid else None
            elif value is not None:
                episode.reward = float(value)
                episode.rewards = {"env": episode.reward}
    except Exception as error:  # noqa: BLE001 — a broken episode is a finding of the audit, not a crash
        episode.finish, episode.reward = "error", None
        episode.error = f"{type(error).__name__}: {error}"
    finally:
        await pool.release(env)
        episode.seconds = time.perf_counter() - start
    return episode


async def play_many(
    pool, rows: list[dict], policy: Policy, max_turns: int = 16, concurrency: int = 32, tool_timeout: float = 60.0
) -> list[Episode]:
    """Episodes of every row, `concurrency` at a time, in row order."""
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(i: int, row: dict) -> Episode:
        async with gate:
            return await play(pool, row, policy, max_turns, tool_timeout, group=i)

    return list(await asyncio.gather(*(one(i, row) for i, row in enumerate(rows))))


# ------------------------------------------------------------- policy helpers


def turn_index(messages: list[dict]) -> int:
    """How many assistant turns the conversation already has (0 before the first)."""
    return sum(m.get("role") == "assistant" for m in messages)


def last_user(messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")


def scripted(calls_field: str = "solution", answer_field: str = "solution_answer") -> Policy:
    """The oracle of rows that carry their solution: one call of row[calls_field] per turn, then the final reply
    row[answer_field]."""

    def policy(row: dict, messages: list[dict]) -> Any:
        calls = row.get(calls_field) or []
        turn = turn_index(messages)
        if turn < len(calls):
            name, arguments = calls[turn]
            return [(name, arguments)]
        return str(row.get(answer_field) or "Done.")

    return policy
