"""The scenario runtime: episodes in which several roles interact, each in its own context.

A scenario is `async def run(episode, row)`. It builds an episode from five things:

  role      a name the config maps to a model (the trained policy, one of its adapters, or a frozen partner).
  session   one context window of one role. `await episode.run(role, messages)` runs it to its end;
            `s = await episode.open(role, messages)` keeps it open: `text = await s.say()` is its next assistant
            message and `await s.say(reply)` answers it and waits for the following one, so a referee loop reads
            like the game it runs. A trained session is one token-exact trajectory whatever its number of turns
            (the turns share their prefix in the engine's cache); a partner's session is text only.
  channel   text handed from one session to another: size-limited, redacted and logged, both as sent and as
            delivered.
  branch    a fork: `await episode.branch(n, fn)` runs `fn` in n sub-episodes that share everything before it.
            Sessions started at the same place of sibling branches (and of the group's episodes, which are the
            root branch) saw the same state: they form a comparison group, the unit a baseline is computed over.
  credit    `session.reward(name, value)`: named components per session, combined per role with the scenario's
            weights. A component named `metric/...` is logged only; None means "does not apply".

Work that does not block its caller: `episode.spawn(coroutine)` starts a task the episode waits for before it is
scored (a tool that starts a sub-agent and returns at once), and `live.notify(text)` leaves a message for a
running session, which it reads with the result of its next tool call or with the next answer it gets (delivery
at a turn boundary, as agent harnesses deliver a background agent's completion or a teammate's message). A tool
ends its own session by raising EndSession.

Sessions talk through asyncio queues on the pipeline's event loop: a session waiting for another's answer is
suspended, holds no thread and no engine slot, and the engines batch whatever turns are ready, whichever role
or model they belong to.

The runtime knows nothing of models. A host starts sessions: the RL pipeline (trained roles through its rollout
loop, palingenesis.rl.scenarios.pipeline) or `play` below (every role a scripted partner: verifier audits, tests,
data without a GPU).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class ScenarioError(Exception):
    """A scenario used the runtime wrongly (a programming error in the scenario, not a policy's failure)."""


class EndSession(Exception):
    """Raised by a tool to end the session that called it: its message is the tool's result, and the session
    takes no further turn (a `submit` or `finish` tool, a transfer of the task to another role)."""


CLOSE = object()  # the answer that ends an open session


def stable_seed(*parts: Any) -> int:
    """A seed from values, the same in every process (hash() is salted per process)."""
    return zlib.crc32(repr(parts).encode())


# ------------------------------------------------------------------ records


@dataclass
class Session:
    """One context window of one role, and what it earned."""

    role: str
    name: str  # unique in the episode: "<role>[.<slot>]#<n>"
    path: tuple  # the branches it runs under: ((label, index), ...), the root episode first
    key: tuple  # its comparison group within the row's group: (role, slot, path without the last index)
    model: str = ""  # what played it: "policy", an adapter's name, or the partner's name
    trained: bool = False  # its tokens can train (a policy's own session)
    messages: list[dict] = field(default_factory=list)
    finish: str = ""  # stop | length | turns | env_done | error | cancelled
    rewards: dict[str, float | None] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    trajectory: Any = None  # palingenesis.rl.trajectory.Trajectory of a trained session

    @property
    def text(self) -> str:
        """Its last assistant message."""
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message.get("content") or ""
        return ""

    @property
    def ok(self) -> bool:
        """It ended by its own choice or its partner's (not cut by a limit, an error or a cancellation)."""
        return self.finish in ("stop", "env_done")

    def reward(self, name: str, value: float | bool | None) -> None:
        """Add (or replace) a reward component. None: the component does not apply to this session."""
        self.rewards[name] = None if value is None else float(value)

    def record(self, content: bool = True) -> dict:
        out = {
            "session": self.name,
            "role": self.role,
            "model": self.model,
            "trained": self.trained,
            "key": list(map(str, self.key)),
            "finish": self.finish,
            "rewards": dict(self.rewards),
        }
        if content and not self.trained:  # a trained session's turns are in its own trajectory record
            out["messages"] = self.messages
        return out


class Channel:
    """Text handed between sessions. `send` returns what the receiver gets: the text with every `redact` string
    replaced and cut to `max_chars` (0: no limit), and logs both forms in the episode."""

    def __init__(self, episode: "Episode", name: str, max_chars: int = 0, redact: tuple[str, ...] = ()):
        self.episode, self.name, self.max_chars = episode, name, max_chars
        self.redact = tuple(str(r) for r in redact if str(r))

    def send(self, text: str, sender: str = "", receiver: str = "") -> str:
        delivered = text or ""
        redacted = 0
        for secret in self.redact:
            redacted += delivered.count(secret)
            delivered = delivered.replace(secret, "[redacted]")
        cut = bool(self.max_chars) and len(delivered) > self.max_chars
        if cut:
            delivered = delivered[: self.max_chars]
        self.episode.log.append(
            {
                "channel": self.name,
                "from": sender,
                "to": receiver,
                "sent": text or "",
                "delivered": delivered,
                "redacted": redacted,
                "cut": cut,
            }
        )
        return delivered


# ------------------------------------------------------------- live sessions


class LiveSession:
    """A session that is running. `say()` gives its next assistant message (None once it has ended);
    `say(reply)` answers the last one first. `close()` ends it; `done()` gives its record."""

    session: Session

    async def say(self, reply: str | None = None) -> str | None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    def notify(self, text: str) -> None:
        """Leave `text` for the session: it reads it at its next turn boundary (with its next tool result or the
        next answer it gets). A message for a session that takes no further turn is never read."""
        raise NotImplementedError

    async def done(self) -> Session:
        await self.close()
        return self.session


Partner = Callable[..., Any]
"""A frozen role: `partner(messages) -> str` (or async), the next assistant message of a conversation; with a
second parameter, `partner(messages, tools)`, it also gets the session's tools ({name: async callable}) and calls
them itself. A partner that declares a parameter named `row` also gets the episode's row as a keyword (an oracle
reads the gold answer there). Any other state it needs is in `messages` (a partner object may be shared by
concurrent sessions)."""


class PartnerSession(LiveSession):
    """A session played by a partner: text in, text out, nothing to train."""

    def __init__(
        self,
        session: Session,
        partner: Partner,
        tools: dict[str, Callable] | None,
        max_turns: int,
        conversational: bool = True,
        row: dict | None = None,
    ):
        self.session, self.partner, self.tools, self.max_turns = session, partner, tools or {}, max_turns
        self.conversational = conversational  # False: it answers its messages once, by the time it is closed
        self.turns = 0
        parameters = inspect.signature(partner).parameters
        self.takes_tools = len([name for name in parameters if name != "row"]) >= 2
        self.keywords = {"row": row or {}} if "row" in parameters else {}
        self.closed = False
        self.notes: list[str] = []

    def notify(self, text: str) -> None:
        self.notes.append(str(text))

    async def say(self, reply: str | None = None) -> str | None:
        s = self.session
        if self.closed:
            return None
        if reply is not None or self.notes:
            notes, self.notes = self.notes, []
            s.messages.append({"role": "user", "content": "\n\n".join([*([reply] if reply else []), *notes])})
        if self.turns >= self.max_turns:
            self.closed, s.finish = True, "turns"
            return None
        self.turns += 1
        args = (s.messages, self.tools) if self.takes_tools else (s.messages,)
        text = self.partner(*args, **self.keywords)
        if inspect.isawaitable(text):
            text = await text
        text = "" if text is None else str(text)
        s.messages.append({"role": "assistant", "content": text})
        return text

    async def close(self) -> None:
        if not self.closed:
            if not self.conversational and self.turns == 0:
                await self.say()
            self.closed = True
            self.session.finish = self.session.finish or "stop"


class SessionPort:
    """The environment of a trained session (the protocol of palingenesis.rl.env), through which the rollout
    loop reaches the scenario: the session's tools, and for an open session the exchange of turns. Each assistant
    message without a tool call is put on `outbox` and the loop waits on `inbox` for the answer, which comes back
    as the next user message; CLOSE ends the session."""

    def __init__(self, tools: dict[str, Callable] | None, conversational: bool, client: Any = None):
        from palingenesis.rl.chat import tool_schema

        self.tools = dict(tools or {})
        # a tool that carries its own schema (a `schema` attribute: an environment's tool behind a function) keeps it
        self.schemas = [getattr(fn, "schema", None) or tool_schema(fn, name) for name, fn in self.tools.items()]
        self.conversational = conversational
        self.client = client  # the generation client: a session waiting for an answer is not waiting for a turn
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.done = False
        self.closing = False
        self.notes: list[str] = []  # messages left for the session (LiveSession.notify), read at its next turn boundary

    def _with_notes(self, result: Any) -> Any:
        if not self.notes or self.done or not isinstance(result, str | type(None)):
            return result  # (a result with images keeps them: the notes wait for the next text result)
        notes, self.notes = self.notes, []
        return "\n\n".join([*([str(result)] if result not in (None, "") else []), *notes])

    def tool_schemas(self) -> list[dict]:
        return self.schemas

    def text_actions(self, content: str) -> list[tuple[str, dict]] | None:
        return [("reply", {"text": content})] if self.conversational else None

    def observation_text(self, results: list[tuple[str, str]]) -> str:
        return "\n\n".join(out for _, out in results)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "reply" and self.conversational and "reply" not in self.tools:
            return await self._exchange(str(arguments.get("text", "")))
        if name not in self.tools:
            from palingenesis.rl.env import ToolFailed

            raise ToolFailed(f"unknown tool {name!r}. Available tools: {', '.join(self.tools) or 'none'}.")
        try:
            result = self.tools[name](**arguments)
            if inspect.isawaitable(result):
                result = await result
        except EndSession as end:
            self.done = True
            return str(end)
        return self._with_notes(result)

    async def _exchange(self, text: str) -> str:
        if self.closing:
            self.done = True
            return ""
        if self.client is not None:
            self.client.exit()
        try:
            self.outbox.put_nowait(text)
            reply = await self.inbox.get()
        finally:
            if self.client is not None:
                self.client.enter()
        if reply is CLOSE:
            self.done = True
            return ""
        return self._with_notes(reply)


class RolloutSession(LiveSession):
    """A trained session: `rollout` is the coroutine of the pipeline's turn loop on `port`, returning the
    trajectory. Its turns are one token sequence; `say` only passes messages in and out of it."""

    def __init__(self, session: Session, port: SessionPort, rollout: Awaitable):
        self.session, self.port = session, port
        self.task = asyncio.ensure_future(rollout)
        self.awaiting_reply = False
        self.delivered_last = False

    async def say(self, reply: str | None = None) -> str | None:
        if self.awaiting_reply:
            if reply is None:
                raise ScenarioError(f"{self.session.name}: say() again without an answer to its last message")
            self.port.inbox.put_nowait(reply)
            self.awaiting_reply = False
        elif reply is not None and not self.task.done():
            raise ScenarioError(f"{self.session.name}: an answer was given before the session said anything")
        if not self.task.done():
            get = asyncio.ensure_future(self.port.outbox.get())
            try:
                await asyncio.wait({get, self.task}, return_when=asyncio.FIRST_COMPLETED)
            finally:  # also when the caller is cancelled: a waiter left behind would take the next message
                if not get.done():
                    get.cancel()
            if get.done() and not get.cancelled():
                self.awaiting_reply = True
                return get.result()
        return self._last()

    def _last(self) -> str | None:
        """After the loop ended: the final assistant message if it was never handed out (a session stopped by
        its turn limit ends on a message the loop does not pass on), else None."""
        self._finish()
        if self.delivered_last or self.session.finish not in ("turns", "stop"):
            return None
        self.delivered_last = True
        return self.session.text

    def _finish(self) -> None:
        s = self.session
        if s.trajectory is not None or not self.task.done():
            return
        if self.task.cancelled() or self.task.exception() is not None:
            s.finish = "cancelled" if self.task.cancelled() else "error"
            return
        trajectory = self.task.result()
        messages = trajectory.messages
        if messages and messages[-1].get("role") == "user" and not messages[-1].get("content"):
            messages.pop()  # the empty answer that closed it
        s.trajectory, s.messages, s.finish = trajectory, messages, trajectory.finish

    async def close(self) -> None:
        if not self.task.done():
            self.port.closing = True
            # always: the loop may be waiting for an answer to a message nobody read (an unread CLOSE is harmless)
            self.port.inbox.put_nowait(CLOSE)
            self.awaiting_reply = False
            try:
                await asyncio.shield(self.task)
            except asyncio.CancelledError:
                if not self.task.cancelled():  # the caller was cancelled, not the session: pass it on
                    raise
            except Exception:  # noqa: BLE001 — the session's own failure: recorded by _finish
                pass
        self._finish()

    def cancel(self) -> None:
        self.task.cancel()

    def notify(self, text: str) -> None:
        self.port.notes.append(str(text))


# ------------------------------------------------------------------ episode


class Host:
    """What starts an episode's sessions: `open(episode, session, messages, tools, conversational, limits)`
    returns a LiveSession. `limits`: max_turns, max_new_tokens, max_completion_tokens (None: the role's)."""

    def partner(self, role: str) -> Partner | None:
        """The partner that plays `role`, or None when the policy does."""
        raise NotImplementedError

    def model(self, role: str) -> str:
        return role

    async def open(self, episode, session, messages, tools, conversational, limits) -> LiveSession:
        raise NotImplementedError


class Episode:
    """One episode of a scenario on one row. `shared` is the scenario's own state (a blackboard every branch
    sees); `log` the channel messages; `rng` is seeded from the run seed, the row and the episode's place among
    its siblings, so an episode's random choices are the same wherever and whenever it runs."""

    def __init__(
        self,
        host: Host,
        row: dict,
        group: int = 0,
        index: int = 0,
        seed: int = 0,
        training: bool = True,
        path: tuple = (),
        root: "Episode | None" = None,
    ):
        self.host, self.row, self.group, self.index, self.seed, self.training = host, row, group, index, seed, training
        self.path = path or (("", index),)
        self.root = root or self
        if root is None:
            self.sessions: list[Session] = []
            self.live: list[LiveSession] = []
            self.log: list[dict] = []
            self.shared: dict[str, Any] = {}
            self.counter: dict[str, int] = {}
            self.background: list[asyncio.Task] = []
        self.rng = random.Random(stable_seed(seed, row.get("id", ""), self.path))

    def __getattr__(self, name: str):  # a branch shares its root's records
        if name in ("sessions", "live", "log", "shared", "counter", "background"):
            return getattr(self.__dict__["root"], name)
        raise AttributeError(name)

    def channel(self, name: str, max_chars: int = 0, redact: tuple[str, ...] = ()) -> Channel:
        return Channel(self, name, max_chars, redact)

    async def open(
        self,
        role: str,
        messages: list[dict],
        *,
        tools: dict[str, Callable] | None = None,
        slot: str = "",
        max_turns: int | None = None,
        max_new_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        conversational: bool = True,
    ) -> LiveSession:
        """Start a session of `role` on `messages` (system and user messages). `slot` separates sessions of one
        role that play different parts at the same place of the episode (they are not compared with each other)."""
        label = f"{role}.{slot}" if slot else role
        n = self.counter[label] = self.counter.get(label, 0) + 1
        key = (role, slot, self.path[:-1] + (self.path[-1][0],))
        session = Session(role, f"{label}#{n}", self.path, key, model=self.host.model(role))
        session.messages = [dict(m) for m in messages]
        self.sessions.append(session)
        limits = {
            "max_turns": max_turns,
            "max_new_tokens": max_new_tokens,
            "max_completion_tokens": max_completion_tokens,
        }
        live = await self.host.open(self, session, session.messages, tools, conversational, limits)
        self.live.append(live)
        return live

    async def run(self, role: str, messages: list[dict], **options) -> Session:
        """A session run to its end: the role answers `messages`, calling its tools as it needs."""
        options.setdefault("conversational", False)
        live = await self.open(role, messages, **options)
        return await live.done()

    async def branch(self, n: int, fn: Callable[["Episode"], Awaitable], label: str = "branch") -> list:
        """Run `fn(sub_episode)` n times from this point, concurrently. The sessions the sub-episodes start at the
        same place form one comparison group; what `fn` returns comes back in order."""
        if n < 1:
            raise ScenarioError("branch needs n >= 1")
        tag = f"{label}{self.counter.setdefault('branch', 0)}"
        self.counter["branch"] += 1
        subs = [
            Episode(
                self.host,
                self.row,
                self.group,
                self.index,
                self.seed,
                self.training,
                self.path + ((tag, i),),
                self.root,
            )
            for i in range(n)
        ]
        return list(await asyncio.gather(*(fn(sub) for sub in subs)))

    def spawn(self, work: Awaitable) -> asyncio.Task:
        """Start `work` (a coroutine: a sub-agent's session and what follows it) without waiting for it. The
        episode waits for it before it ends (`join`), so whatever it rewards counts, and cancels it if the
        episode fails. A tool returns right after spawning; the result reaches the caller later, e.g. through
        `live.notify`."""
        task = asyncio.ensure_future(work)
        self.background.append(task)
        return task

    async def join(self) -> None:
        """Wait for everything spawned (including what spawned work spawns). A failure there fails the episode."""
        done = 0
        while done < len(self.background):
            pending = self.background[done:]
            done = len(self.background)
            await asyncio.gather(*pending)

    async def abort(self) -> None:
        """Cancel the sessions still running (the episode failed or ran out of time)."""
        tasks = [task for task in self.background if not task.done()]
        for task in tasks:
            task.cancel()
        for live in self.live:
            if isinstance(live, RolloutSession) and not live.task.done():
                live.cancel()
                tasks.append(live.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for live in self.live:
            if isinstance(live, RolloutSession):
                live._finish()
            elif not live.session.finish:
                live.session.finish = "cancelled"

    def record(self, content: bool = True) -> dict:
        """The episode for the rollout dumps and the exports: its sessions and the channel messages."""
        return {
            "row": self.row.get("id"),
            "scenario": self.row.get("scenario"),
            "episode": self.index,
            "sessions": [s.record(content) for s in self.sessions],
            "channels": self.log if content else len(self.log),
        }


# ------------------------------------------------------------- without a model


class PartnerHost(Host):
    """Every role played by a partner: `partners` maps role names to partners."""

    def __init__(self, partners: dict[str, Partner], max_turns: int = 64):
        self.partners, self.max_turns = partners, max_turns

    def partner(self, role: str) -> Partner | None:
        if role not in self.partners:
            raise ScenarioError(f"no partner for role {role!r} (have {sorted(self.partners)})")
        return self.partners[role]

    def model(self, role: str) -> str:
        return getattr(self.partners.get(role), "__name__", role)

    async def open(self, episode, session, messages, tools, conversational, limits) -> LiveSession:
        turns = limits.get("max_turns") or self.max_turns
        return PartnerSession(session, self.partner(session.role), tools, turns, conversational, episode.row)


def play(scenario: Any, rows: list[dict], partners: dict[str, Partner], seed: int = 0, episodes: int = 1) -> list:
    """Run `scenario` on `rows` with every role a partner: [[Episode, ...] per row]. No model is involved, so
    this is how a scenario's verifier is audited (oracle partners must earn the reward, degenerate ones must
    not) and how its rules are tested."""

    async def one(row: dict, group: int, index: int) -> Episode:
        episode = Episode(PartnerHost(partners), row, group, index, seed, training=False)
        await scenario.run(episode, row)
        await episode.join()
        for live in episode.live:
            await live.close()
        return episode

    async def all_rows() -> list:
        return [[await one(row, g, k) for k in range(episodes)] for g, row in enumerate(rows)]

    return asyncio.run(all_rows())
