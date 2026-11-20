"""The RL pipeline's side of scenarios: episodes instead of single rollouts.

ScenarioPipeline replaces one method of RLPipeline: `_group`. For a row it runs `rollout.group_size` episodes of
the scenario concurrently. A trained role's session is the pipeline's own turn loop (`_trajectory`: the same
prompt encoding, tool-call parsing, token budgets and truncation as single-agent RL) on a SessionPort, which is
how the loop reaches the scenario; a partner's session is text. The episodes of a row come back as one RowGroup
(palingenesis.rl.scenarios.credit), which the batch logic of RLPipeline treats like a group: collected until
enough rows have something to train, carried over, overlapped with training under rollout.max_staleness.

Every engine keeps its own continuous batch. Sessions are coroutines on one event loop: whichever turns are
ready, of whatever role, episode or row, are in their engine's batch at the next step, and a session waiting for
another session's message holds nothing. A session's turns extend one token sequence, and the sibling sessions
of a branch start from the same tokens, so the engine's prefix cache serves both.

Roles (`roles.<name>` in the config) are resolved once into RoleSpec: the adapter that plays the role ("" for
the policy's own weights), or its partner.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import threading
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from palingenesis.lora import use_adapter
from palingenesis.opd.orchestrator import PublishedWeights
from palingenesis.rl.generation import GenerationClient
from palingenesis.rl.pipeline import RLBatch, RLPipeline
from palingenesis.rl.scenarios.base import Scenario
from palingenesis.rl.scenarios.credit import RowGroup
from palingenesis.rl.scenarios.runtime import (
    Episode,
    Host,
    LiveSession,
    Partner,
    PartnerSession,
    RolloutSession,
    ScenarioError,
    Session,
    SessionPort,
)

logger = logging.getLogger(__name__)

BASE = "base"  # roles.<name>.partner: the policy's frozen base (its adapters off)
PARTNER_TURNS = 64  # assistant turns of a partner's session when the scenario sets no limit


@dataclass
class RoleSpec:
    name: str
    adapter: str = ""  # the named adapter that plays it ("": the policy's own weights / default adapter)
    partner: Partner | None = None  # a text-level partner instead of the policy
    label: str = "policy"  # what the dumps and metrics call its model
    train: bool = True
    system_prompt: str = ""
    limits: dict[str, int] = field(default_factory=dict)  # max_turns / max_new_tokens / max_completion_tokens
    chat_template_kwargs: dict = field(default_factory=dict)
    share: float = 1.0

    @property
    def engine(self) -> str:
        """The key of the engine that generates for it."""
        return BASE if self.label == BASE else self.adapter


class EndpointPartner:
    """A partner behind an OpenAI-compatible chat endpoint (`roles.<name>.partner: http://host:port/v1`, with
    partner_args {model, api_key, temperature, max_tokens, timeout}). Text only: it takes no tools."""

    def __init__(
        self,
        url: str,
        model: str = "",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        self.url, self.model, self.api_key = url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.__name__ = model or url

    async def __call__(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def post() -> str:
            request = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(), headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)["choices"][0]["message"]["content"] or ""

        return await asyncio.get_running_loop().run_in_executor(None, post)


def resolve_roles(config: Any, scenario: Scenario) -> dict[str, RoleSpec]:
    """The scenario's roles as the config sets them (a role the config does not mention is the trained policy)."""
    from palingenesis.rl.rewards import load_object

    unknown = sorted(set(config.roles) - set(scenario.roles))
    if unknown:
        raise ValueError(f"roles: {unknown} are not roles of scenario {scenario.name} ({', '.join(scenario.roles)})")
    table = scenario.partners()
    specs = {}
    for name in scenario.roles:
        c = config.roles.get(name)
        spec = RoleSpec(name)
        if c is not None:
            spec.adapter, spec.train, spec.share = c.adapter, bool(c.train), float(c.share)
            spec.system_prompt, spec.chat_template_kwargs = c.system_prompt, dict(c.chat_template_kwargs)
            spec.limits = {
                k: int(getattr(c, k)) for k in ("max_turns", "max_new_tokens", "max_completion_tokens") if getattr(c, k)
            }
            spec.label = c.adapter or "policy"
            if c.partner == BASE:
                spec.label, spec.train = BASE, False
            elif c.partner.startswith("scripted:"):
                key = c.partner.split(":", 1)[1]
                if key not in table.get(name, {}):
                    raise ValueError(
                        f"roles.{name}.partner: scenario {scenario.name} has no scripted partner {key!r} for this "
                        f"role (it has {sorted(table.get(name, {}))})"
                    )
                spec.partner, spec.label = table[name][key], c.partner
            elif c.partner.startswith(("http://", "https://")):
                spec.partner, spec.label = EndpointPartner(c.partner, **c.partner_args), c.partner
            elif c.partner:
                target = load_object(c.partner)
                spec.partner = target(**c.partner_args) if inspect.isclass(target) else target
                spec.label = c.partner
            if spec.partner is not None:
                spec.train = False
        specs[name] = spec
    return specs


class AdapterEngine:
    """The trainer's own model as the rollout engine of one adapter (the hf backend with several adapters): each
    call generates with that adapter (lora.use_adapter), or with the adapters off for BASE. One model, one call
    at a time: `lock` is shared by the adapters' engines."""

    def __init__(self, engine: Any, adapter: str, lock: threading.Lock, adapters: Any = None):
        self.engine, self.adapter, self.lock, self.adapters = engine, adapter, lock, adapters

    @property
    def version(self) -> int:
        return self.engine.version

    def update_weights(self, named_tensors, version: int) -> None:
        self.engine.update_weights((), version)  # it generates with the trained model itself

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def generate(self, prompts, max_new_tokens, temperature, **kwargs):
        with self.lock:
            if self.adapter == BASE:
                with self.adapters.disabled():
                    return self.engine.generate(prompts, max_new_tokens, temperature, **kwargs)
            with use_adapter(self.adapter):
                return self.engine.generate(prompts, max_new_tokens, temperature, **kwargs)


class WeightsView:
    """The trainer's published weights for one more engine: the same version and lock as `parent`, another set
    of tensors (`named`: one adapter's fold; None: nothing to load, the engine keeps what it has)."""

    def __init__(self, parent: PublishedWeights, named=None, frozen: bool = False):
        self.parent, self.named, self.frozen = parent, named, frozen

    version = property(lambda self: self.parent.version)
    lock = property(lambda self: self.parent.lock)
    model = property(lambda self: self.parent.model)

    def sync(self, engine) -> float:
        if self.frozen:  # a frozen partner's engine: it is at every version
            engine.version = self.version
            return 0.0
        return PublishedWeights.sync(self, engine)


class _PortPool:
    """The environment pool of one session: its port (palingenesis.rl.env.EnvPool's interface, as the turn loop
    uses it)."""

    has_reward = False

    def __init__(self, port: SessionPort):
        self.port = port

    async def acquire(self) -> SessionPort:
        return self.port

    async def release(self, env: Any) -> None:
        return None

    async def episode_schemas(self, env: SessionPort) -> list[dict]:
        return env.tool_schemas()

    async def shutdown(self) -> None:
        return None


class ScenarioPipeline(RLPipeline, Host):
    """RLPipeline whose groups are episodes of `scenario`. `engines`: {engine key: (engine, weights)} for the
    engines besides the policy's own (one per further adapter, one for the frozen base)."""

    def __init__(self, *args, scenario: Scenario, roles: dict[str, RoleSpec], engines: dict[str, tuple] | None = None):
        super().__init__(*args)
        self.scenario, self.roles = scenario, roles
        self.outcome_role = self.config.scenario.outcome or scenario.outcome or scenario.roles[0]
        self.clients: dict[str, GenerationClient] = {"": self.client}
        self.extra: dict[str, tuple] = {}
        for key, (engine, weights) in (engines or {}).items():
            self.clients[key] = GenerationClient(engine, merge=False)
            self.extra[key] = (engine, weights)
        self.client.merge = False  # sessions are multi-turn: an n-sample request returns when its longest ends
        missing = sorted({spec.engine for spec in roles.values() if spec.partner is None} - set(self.clients))
        if missing:
            raise ScenarioError(f"no rollout engine for {missing}")
        self.role_configs = {name: self._role_config(spec) for name, spec in roles.items()}
        self.episode_counts: Counter = Counter()

    def _role_config(self, spec: RoleSpec):
        config = copy.deepcopy(self.config)
        config.data.prompt_field, config.data.system_prompt = "messages", ""  # the scenario wrote the conversation
        # a session waiting for another role's message is inside a "tool call": only the episode's own timeout ends it
        config.env.tool_timeout = float(self.config.scenario.timeout) + 60.0
        return config

    # ------------------------------------------------------------------- host

    def partner(self, role: str) -> Partner | None:
        return self.roles[role].partner

    def model(self, role: str) -> str:
        return self.roles[role].label

    async def open(self, episode: Episode, session: Session, messages, tools, conversational, limits) -> LiveSession:
        spec = self.roles[session.role]
        limits = {**spec.limits, **{k: v for k, v in limits.items() if v}}
        if spec.system_prompt:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{spec.system_prompt}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {"role": "system", "content": spec.system_prompt})
        if spec.partner is not None:
            return PartnerSession(session, spec.partner, tools, limits.get("max_turns") or PARTNER_TURNS)
        session.trained = spec.train
        view = copy.copy(self)  # the turn loop with this session's engine, limits and port
        view.client, view.config = self.clients[spec.engine], self.role_configs[session.role]
        view.observation_loss = False  # another role's text is not the environment's output
        view.reported_errors = 5  # failures are counted per episode (episode/failed), not logged per session
        port = SessionPort(tools, conversational, view.client)
        if port.schemas:
            session.info["tools"] = port.schemas
        view.env_pool = _PortPool(port)
        row = {"messages": messages, **limits}
        if spec.chat_template_kwargs:
            row["chat_template_kwargs"] = spec.chat_template_kwargs
        if "max_turns" not in row and not tools and not conversational:
            row["max_turns"] = 1

        async def rollout():
            trajectory, _ = await view._trajectory(row, episode.group, episode.root.temperature, episode.training)
            return trajectory

        return RolloutSession(session, port, rollout())

    # ----------------------------------------------------------------- groups

    async def _run_episode(self, row: dict, gid: int, index: int, temperature: float, training: bool):
        episode = Episode(self, row, gid, index, self.config.train.seed, training)
        episode.temperature = temperature
        failure = None
        try:
            await asyncio.wait_for(self.scenario.run(episode, row), self.config.scenario.timeout)
            for live in episode.live:
                await live.close()
        except asyncio.CancelledError:
            await episode.abort()
            raise
        except Exception as error:  # noqa: BLE001 — one broken episode must not stop training; it is counted
            failure = f"{type(error).__name__}: {error}"
            await episode.abort()
            if self.episode_counts["logged"] < 5:
                self.episode_counts["logged"] += 1
                logger.exception("scenario episode failed (its sessions are left out of training)")
        return episode, failure

    async def _group(self, index: int, row: dict, temperature: float, size: int, training: bool, revisit: bool = False):
        gid = next(self.group_ids)
        results = await asyncio.gather(*(self._run_episode(row, gid, k, temperature, training) for k in range(size)))
        group = RowGroup()
        outcomes: list[float] = []
        dumping = bool(self.config.logging.dump_trajectories) and self.config.logging.dump_content
        for episode, failure in results:
            self.episode_counts["episodes"] += 1
            self.episode_counts["failed"] += failure is not None
            self.episode_counts["sessions"] += len(episode.sessions)
            self.episode_counts["messages"] += len(episode.log)
            first = True
            for session in episode.sessions:
                value = None if failure else self.scenario.reward(session.role, session.rewards)
                if session.role == self.outcome_role and value is not None:
                    outcomes.append(value)
                t = session.trajectory
                if t is None:
                    continue
                spec = self.roles[session.role]
                t.role, t.model, t.adapter = session.role, spec.label, spec.adapter
                t.episode, t.session = episode.index, session.name
                t.comparison, t.unit = (gid, *session.key), session.path
                t.rewards = dict(session.rewards)
                t.scored = value is not None and t.finish != "error"
                t.reward = float(value) if t.scored else 0.0
                t.info["raw_reward"] = t.reward
                t.trained = t.scored and spec.train
                if self.config.loss.truncation == "mask" and t.finish == "length":
                    t.trained = False
                if session.info.get("tools"):
                    t.info["tools"] = session.info["tools"]
                if failure:
                    t.info["episode_error"] = failure
                if revisit:
                    t.info["revisit"] = True
                if first and dumping:  # the episode's partners and channel messages, once, for dumps and exports
                    t.info["episode_record"] = episode.record()
                first = False
                group.append(t)
        if training:
            if outcomes:
                threshold = self.config.data.success_threshold
                solved = sum(v >= threshold for v in outcomes) / len(outcomes)
                self.sampler.observe(index, sum(outcomes) / len(outcomes), solved, self.weights.version)
            elif hasattr(self.sampler, "release"):
                self.sampler.release(index)
        return group

    # ---------------------------------------------------------------- batches

    def _sync_engines(self) -> None:
        for key, (engine, weights) in self.extra.items():
            with self.clients[key].paused():
                weights.sync(engine)
            engine.wake()

    def run(self, n_prompts: int, temperature: float) -> RLBatch:
        with self.lock:
            self._sync_engines()
            batch = super().run(n_prompts, temperature)
        counts, self.episode_counts = self.episode_counts, Counter(logged=self.episode_counts["logged"])
        episodes = max(counts["episodes"], 1)
        batch.stats.update(
            {
                "episode/failed": counts["failed"] / episodes,
                "episode/sessions": counts["sessions"] / episodes,
                "episode/messages": counts["messages"] / episodes,
            }
        )
        return batch

    def evaluate(self, rows: list[dict], temperature: float) -> dict[str, float]:
        with self.lock:
            self._sync_engines()
            return super().evaluate(rows, temperature)

    def close(self) -> None:
        super().close()
        for key, client in self.clients.items():
            if key:
                client.close()

    # ------------------------------------------------------------- statistics

    def _group_stats(self, trajectories: list) -> dict[str, float]:
        stats = super()._group_stats(trajectories)
        by_role: dict[str, list] = {}
        for t in trajectories:
            by_role.setdefault(t.role, []).append(t)
        for role, ts in by_role.items():
            scored = [t for t in ts if t.scored]
            prefix = f"role/{role}"
            stats[f"{prefix}/sessions"] = float(len(ts))
            stats[f"{prefix}/completion_len"] = sum(t.sampled_tokens for t in ts) / len(ts)
            stats[f"{prefix}/turns"] = sum(t.turns for t in ts) / len(ts)
            stats[f"{prefix}/truncated"] = sum(t.finish == "length" for t in ts) / len(ts)
            stats[f"{prefix}/skipped"] = 1.0 - len(scored) / len(ts)
            if scored:
                stats[f"{prefix}/reward"] = sum(t.reward for t in scored) / len(scored)
            components: dict[str, list[float]] = {}
            for t in ts:
                for name, value in t.rewards.items():
                    if value is not None:
                        components.setdefault(name, []).append(value)
            stats.update({f"{prefix}/rewards/{name}": sum(v) / len(v) for name, v in components.items()})
        if trajectories:
            stats["reward"] = stats.get(f"role/{self.outcome_role}/reward", stats.get("reward", 0.0))
        return stats

    def _outcome_stats(self, groups: list) -> dict[str, float]:
        """pass@1 / pass@G and reward_all of the outcome role (the task), per row."""
        return super()._outcome_stats([[t for t in g if t.role == self.outcome_role] for g in groups])
