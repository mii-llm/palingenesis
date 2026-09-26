"""Producing scored rollout groups: the RL side of the orchestrator's producer thread.

  run(n)     sync the newest weights into the engine, then run groups of rollouts until
             n informative groups are collected (dynamic sampling: a group whose rewards
             are all equal has zero advantage everywhere and is replaced by a fresh
             prompt, within rollout.max_refill), score them, and return the batch.
  evaluate   held-out prompts, one rollout each, rewards averaged.

Every trajectory is a coroutine on one persistent event loop (a daemon thread): turns
are generated through GenerationClient (batched into engine calls), tools and rewards
run concurrently, and a group is scored as soon as its rollouts finish. The engine is
used from this loop's generation thread only, under the pipeline lock, so a rollout
engine is never touched while the trainer updates the weights it loads.
"""

import asyncio
import inspect
import itertools
import logging
import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from palingenesis.opd.orchestrator import PublishedWeights
from palingenesis.opd.rollout import RolloutEngine
from palingenesis.rl.chat import ChatFormat, encode_prompt, parse_assistant
from palingenesis.rl.config import RLConfig
from palingenesis.rl.data import PromptSampler, prompt_messages
from palingenesis.rl.env import EnvPool, call_sync_or_async, row_tools, run_tool
from palingenesis.rl.generation import GenerationClient
from palingenesis.rl.rewards import SKIPPED, Reward, SkipSample
from palingenesis.rl.trajectory import Trajectory, group_is_informative

logger = logging.getLogger(__name__)


@dataclass
class RLBatch:
    samples: list[list[Trajectory]]  # groups (the orchestrator counts len(samples) when it drops a batch)
    version: int
    stats: dict[str, float] = field(default_factory=dict)


class RLPipeline:
    def __init__(
        self,
        tok,
        chat: ChatFormat,
        engine: RolloutEngine,
        weights: PublishedWeights,
        config: RLConfig,
        sampler: PromptSampler,
        rewards: list[Reward],
        env_pool: EnvPool | None,
        sandbox,
        stop_ids,
    ):
        self.tok, self.chat, self.engine, self.weights = tok, chat, engine, weights
        self.config, self.sampler, self.rewards, self.env_pool, self.sandbox = (
            config,
            sampler,
            rewards,
            env_pool,
            sandbox,
        )
        self.stop_ids = set(stop_ids)
        # n-sample merging only helps when a group's rollouts are one turn each
        self.client = GenerationClient(engine, merge=env_pool is None)
        self.lock = threading.RLock()
        self.group_ids = itertools.count()
        self.carry: list[list[Trajectory]] = []  # surplus informative groups (max_staleness > 0)
        self.continuing: set[asyncio.Task] = set()  # groups still generating at the last batch's end
        self.drop_rate = 0.0  # expected fraction of zero-variance groups (running, per batch)
        self.reported_errors = 0
        self.batches = 0  # batches collected (the dump's step numbers)
        self.rollout_start = 0.0  # loop time the current batch's collection started
        self.dropped: list[list[Trajectory]] = []  # zero-variance groups of the batch (dumps)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="pgs-rl-rollouts", daemon=True).start()

    def _await(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    # ----------------------------------------------------------------- batches

    def run(self, n_prompts: int, temperature: float) -> RLBatch:
        """`n_prompts` informative groups sampled with the newest weights (fewer when the refill
        budget runs out). Called by the orchestrator's producer thread."""
        with self.lock:
            with self.client.paused():  # between engine steps: groups in flight continue with the new weights
                sync = self.weights.sync(self.engine)
            self.engine.wake()
            start = time.perf_counter()
            self.client.timeline.clear()
            prefill = (getattr(self.engine, "prompt_tokens", 0), getattr(self.engine, "cached_tokens", 0))
            self.dropped = []
            groups, stats = self._await(self._collect(n_prompts, temperature))
            elapsed = time.perf_counter() - start
            self.engine.sleep()
        self.batches += 1
        stats.update(_timeline_stats(self.client.timeline))
        if hasattr(self.engine, "prompt_tokens"):
            prompt = self.engine.prompt_tokens - prefill[0]
            stats["engine/prefill_tokens"] = prompt - (self.engine.cached_tokens - prefill[1])
            stats["engine/prefix_hit"] = (self.engine.cached_tokens - prefill[1]) / max(prompt, 1)
        every = self.config.logging.dump_trajectories
        if every and self.batches % every == 0:
            self._dump(groups, self.batches)
        trajectories = [t for g in groups for t in g]
        tokens = sum(t.sampled_tokens for t in trajectories) + stats.pop("_dropped_tokens", 0)
        stats.update(
            {
                "time/sync": sync,
                "time/rollout": elapsed,
                "rollout_tokens": tokens,
                "rollout_tok_s": tokens / max(elapsed, 1e-9),
            }
        )
        # The newest group's version: carried groups may be older, and the trainer drops those
        # beyond max_staleness one by one instead of the orchestrator dropping the batch.
        version = max((t.version for t in trajectories), default=self.weights.version)
        return RLBatch(groups, version, stats)

    async def _collect(self, target: int, temperature: float) -> tuple[list[list[Trajectory]], dict[str, float]]:
        """Run groups until `target` informative ones are collected.

        Groups are launched speculatively, `missing / (1 - expected zero-variance rate)` at a
        time, so a batch usually fills in one wave instead of serial refill rounds that leave
        the engine half empty. Surplus informative groups wait for the next batch when
        max_staleness allows it, and are discarded otherwise.

        The batch ends when it is full. On-policy (max_staleness 0) every group launched is
        then awaited, since the engine sleeps next. Otherwise the groups still generating keep
        going (partial rollouts): they overlap the training step and join the next batch, so
        the engine never drains waiting for a batch's slowest trajectories (measured: the last
        60% of an agentic batch's rollout time ran at under a quarter of the peak batch)."""
        r = self.config.rollout
        budget = target + math.ceil(target * r.max_refill)
        keep_surplus = r.max_staleness > 0
        kept: list[list[Trajectory]] = self.carry[:target]
        self.carry = self.carry[target:]
        counts: Counter = Counter()
        dropped_tokens = 0
        # groups continuing from the last batch are a bonus: new launches do not count on them,
        # since they are its slowest groups (whatever finishes first joins the batch)
        in_flight, self.continuing = self.continuing, set()
        adopted = set(in_flight)
        continued = len(in_flight)
        launched = 0
        self.rollout_start = asyncio.get_running_loop().time()
        dumping = bool(self.config.logging.dump_trajectories)

        def account(group: list[Trajectory]) -> bool:
            nonlocal dropped_tokens
            counts["groups"] += 1
            if group_is_informative(group):
                return True
            counts["zero_variance"] += 1
            dropped_tokens += sum(t.sampled_tokens for t in group)
            if dumping:
                self.dropped.append(group)
            return False

        while len(kept) < target:
            missing = target - len(kept) - len(in_flight - adopted)
            if missing > 0:
                missing = math.ceil(missing / (1.0 - min(self.drop_rate, 0.8)))
            for _ in range(max(0, min(missing, budget - launched))):
                index, row = self.sampler.draw()
                in_flight.add(asyncio.create_task(self._group(index, row, temperature, r.group_size, True)))
                launched += 1
            if not in_flight:
                break  # refill budget spent
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group = task.result()
                if account(group):
                    (kept if len(kept) < target else self.carry if keep_surplus else []).append(group)
        if keep_surplus:
            self.continuing = in_flight  # partial rollouts: they finish during the next batch
        else:
            for group in await asyncio.gather(*in_flight):
                account(group)
        if counts["groups"]:
            self.drop_rate = 0.5 * self.drop_rate + 0.5 * counts["zero_variance"] / counts["groups"]
        stats = self._group_stats([t for g in kept for t in g])
        stats.update(
            {
                "groups/launched": launched,
                "groups/kept": len(kept),
                "groups/zero_variance": counts["zero_variance"] / max(counts["groups"], 1),
                "groups/carried": len(self.carry),
                "groups/continued": continued,
                "_dropped_tokens": dropped_tokens,
            }
        )
        return kept, stats

    # ------------------------------------------------------------------ groups

    async def _group(self, index: int, row: dict, temperature: float, size: int, training: bool) -> list[Trajectory]:
        gid = next(self.group_ids)
        results = await asyncio.gather(*(self._trajectory(row, gid, temperature, training) for _ in range(size)))
        group = [t for t, _ in results]
        envs = [env for _, env in results]
        try:
            await self._score(group, envs)
        finally:
            if self.env_pool is not None:
                await asyncio.gather(*(self.env_pool.release(env) for env in envs if env is not None))
        if training:
            scored = [t.reward for t in group if t.scored]
            if scored:
                self.sampler.observe(index, sum(scored) / len(scored))
        return group

    async def _trajectory(self, row: dict, gid: int, temperature: float, training: bool) -> tuple[Trajectory, Any]:
        config = self.config
        r, e = config.rollout, config.env
        env = await self.env_pool.acquire() if self.env_pool is not None else None
        messages = prompt_messages(row, config.data.prompt_field, config.data.system_prompt)
        trajectory = Trajectory(row, gid, [], messages)
        clock = asyncio.get_running_loop().time
        timing = trajectory.info["timing"] = {"start": clock() - self.rollout_start, "turns": []}
        self.client.enter()
        try:
            if env is not None and hasattr(env, "reset"):
                start = await call_sync_or_async(env.reset, **row)
                if isinstance(start, str) and start:
                    last = next(m for m in reversed(messages) if m["role"] == "user")
                    last["content"] = f"{last['content']}\n\n{start}" if last["content"] else start
                elif isinstance(start, list):
                    messages[:] = [dict(m) for m in start]
            trajectory.info["prompt"] = [dict(m) for m in messages]
            # the environment's tools (its methods, or tool_schemas() for this episode), else the row's own
            schemas = await self.env_pool.episode_schemas(env) if env is not None else None
            schemas = schemas or row_tools(row, config.data.tools_field)
            by_name = {s["function"]["name"]: s for s in schemas or []}
            trajectory.prompt_ids = encode_prompt(self.tok, messages, schemas, config.model.chat_template_kwargs)
            budget = config.completion_budget
            max_turns = 1 if env is None else e.max_turns
            for turn in range(max_turns):
                context = len(trajectory.prompt_ids) + len(trajectory.tokens)
                room = min(
                    r.max_new_tokens,
                    budget - trajectory.sampled_tokens,
                    r.max_model_len - context,
                )
                if room <= 0:
                    trajectory.finish = "length"
                    break
                t_generate = clock()
                out = await self.client.generate(trajectory.prompt_ids + trajectory.tokens, room, temperature)
                turn_timing = {"generate_s": clock() - t_generate, "tokens": len(out.ids), "context": context}
                timing["turns"].append(turn_timing)
                logprobs = out.logprobs if out.logprobs or training else [0.0] * len(out.ids)
                trajectory.append_generated(out.ids, logprobs, out.version)
                text = self.tok.decode(
                    [t for t in out.ids if t not in self.stop_ids],
                    skip_special_tokens=True,
                )
                parsed = parse_assistant(text, self.chat, e.tool_parser, by_name, f"call_{turn}")
                trajectory.messages.append(parsed.message())
                if out.finish == "length":
                    trajectory.finish = "length"
                    break
                if env is None or not (parsed.calls or parsed.errors):
                    trajectory.finish = "stop"
                    break
                if turn == max_turns - 1:
                    trajectory.finish = "turns"
                    break
                t_tools = clock()
                results = await asyncio.gather(
                    *(run_tool(env, c.name, c.arguments, e.tool_timeout, set(by_name)) for c in parsed.calls)
                )
                turn_timing["tools_s"] = clock() - t_tools
                observations = [
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "name": c.name,
                        "content": text,
                    }
                    for c, (text, _) in zip(parsed.calls, results)
                ]
                observations += [{"role": "tool", "content": error} for error in parsed.errors]
                for o in observations:
                    o["content"] = self.chat.truncate(self.chat.sanitize(o["content"]), e.max_tool_output_tokens)
                trajectory.tool_calls += len(parsed.calls)
                trajectory.tool_errors += sum(failed for _, failed in results) + len(parsed.errors)
                if getattr(env, "done", False):  # the environment ended the episode (e.g. a submit tool)
                    trajectory.messages.extend(observations)
                    trajectory.finish = "env_done"
                    break
                context_ids = self.chat.continuation_ids(observations, schemas, parsed.calls)
                if out.ids and out.ids[-1] != self.chat.eot_id:  # the turn ended on another stop token
                    context_ids = [self.chat.eot_id] + context_ids
                if len(trajectory.prompt_ids) + len(trajectory.tokens) + len(context_ids) >= r.max_model_len - 1:
                    trajectory.finish = "length"
                    break
                trajectory.append_context(context_ids)
                trajectory.messages.extend(observations)
        except asyncio.CancelledError:  # shutdown with the rollout in flight
            if env is not None and self.env_pool is not None:
                await self.env_pool.release(env)
            raise
        except Exception as error:  # noqa: BLE001 — one broken rollout must not stop training; it is counted
            trajectory.finish, trajectory.scored, trajectory.trained = (
                "error",
                False,
                False,
            )
            trajectory.info["error"] = f"{type(error).__name__}: {error}"
            if self.reported_errors < 5:
                self.reported_errors += 1
                logger.exception("rollout failed (the trajectory is left out of training)")
        finally:
            timing["end"] = clock() - self.rollout_start
            self.client.exit()
        return trajectory, env

    def _dump(self, groups: list[list[Trajectory]], step: int) -> None:
        """Every trajectory of the batch (kept and zero-variance) and the engine's concurrency
        timeline, to <output_dir>/rollouts/step_<n>.jsonl: to read what the policy does."""
        import json
        from pathlib import Path

        path = Path(self.config.train.output_dir) / "rollouts"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"step_{step}.jsonl", "w") as f:
            for kept, batch in ((True, groups), (False, self.dropped)):
                for group in batch:
                    for t in group:
                        record = {
                            "group": t.group,
                            "kept": kept,
                            "finish": t.finish,
                            "reward": t.reward,
                            "rewards": t.rewards,
                            "turns": t.turns,
                            "tool_calls": t.tool_calls,
                            "tool_errors": t.tool_errors,
                            "prompt_tokens": len(t.prompt_ids),
                            "sampled_tokens": t.sampled_tokens,
                            "context_tokens": len(t.tokens) - t.sampled_tokens,
                            "version": t.version,
                            "messages": t.messages,
                            "info": t.info,
                        }
                        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        with open(path / f"step_{step}.timeline.json", "w") as f:
            json.dump(self.client.timeline, f)

    async def _score(self, group: list[Trajectory], envs: list[Any]) -> None:
        """Rewards of a group, their weighted sum, and the length shaping and masking."""
        live = [i for i, t in enumerate(group) if t.finish != "error"]
        if not live:
            return
        samples = [self._sample_kwargs(group[i], envs[i]) for i in live]
        scores: dict[str, list[Any]] = {}
        for reward in self.rewards:
            scores[reward.name] = await reward.score(samples)
        if self.env_pool is not None and self.env_pool.has_reward:
            scores["env"] = list(await asyncio.gather(*(self._env_reward(envs[i], group[i]) for i in live)))
        weights = {r.name: r.weight for r in self.rewards}
        loss = self.config.loss
        budget = self.config.completion_budget
        for k, i in enumerate(live):
            t = group[i]
            values = {}
            for name, s in scores.items():  # an environment may return components: {"env/<name>": value}
                values.update({f"env/{c}": v for c, v in s[k].items()} if isinstance(s[k], dict) else {name: s[k]})
            if any(v is SKIPPED for v in values.values()):
                t.scored = t.trained = False
                t.info["skipped"] = True
                t.rewards = {n: (None if v is SKIPPED else v) for n, v in values.items()}
                continue
            t.rewards = values
            applicable = [(weights.get(n, 1.0), v) for n, v in values.items() if v is not None]
            if not applicable:
                t.scored = t.trained = False
                continue
            t.reward = sum(w * v for w, v in applicable)
            if loss.overlong_buffer:
                over = t.sampled_tokens - (budget - loss.overlong_buffer)
                if over > 0:
                    penalty = loss.overlong_penalty * min(1.0, over / loss.overlong_buffer)
                    t.rewards["overlong"] = -penalty
                    t.reward -= penalty
            if loss.truncation == "mask" and t.finish == "length":
                t.trained = False

    async def _env_reward(self, env: Any, trajectory: Trajectory) -> Any:
        """The environment's reward: a number, None, or {component: number}; get_reward may ask
        for the conversation (get_reward(messages)) to verify from the transcript."""
        try:
            wants_messages = bool(inspect.signature(env.get_reward).parameters)
            args = (trajectory.messages,) if wants_messages else ()
            value = await call_sync_or_async(env.get_reward, *args)
        except SkipSample:
            return SKIPPED
        except Exception:  # noqa: BLE001 — an environment that cannot score skips the sample
            logger.exception("environment get_reward failed")
            return SKIPPED
        if isinstance(value, dict):
            return {k: None if v is None else float(v) for k, v in value.items()}
        return None if value is None else float(value)

    def _sample_kwargs(self, t: Trajectory, env: Any) -> dict[str, Any]:
        last = next((m for m in reversed(t.messages) if m.get("role") == "assistant"), {})
        return {
            **t.row,
            "completion": last.get("content") or "",
            "reasoning": last.get("reasoning_content") or "",
            "prompt": t.info["prompt"],
            "messages": t.messages,
            "completion_ids": [tok for tok, m in zip(t.tokens, t.mask) if m],
            "finish_reason": t.finish,
            "env": env,
            "sandbox": self.sandbox,
        }

    # --------------------------------------------------------------- statistics

    def _group_stats(self, trajectories: list[Trajectory]) -> dict[str, float]:
        if not trajectories:
            return {}
        n = len(trajectories)
        stats: dict[str, float] = {
            "reward": sum(t.reward for t in trajectories if t.scored) / max(1, sum(t.scored for t in trajectories)),
            "completion_len": sum(t.sampled_tokens for t in trajectories) / n,
            "truncated": sum(t.finish == "length" for t in trajectories) / n,
            "skipped": sum(not t.scored for t in trajectories) / n,
            "rollout_errors": sum(t.finish == "error" for t in trajectories) / n,
        }
        if self.env_pool is not None:
            stats.update(
                {
                    "turns": sum(t.turns for t in trajectories) / n,
                    "tool_calls": sum(t.tool_calls for t in trajectories) / n,
                    "tool_errors": sum(t.tool_errors for t in trajectories)
                    / max(1, sum(t.tool_calls for t in trajectories)),
                    "turn_limit": sum(t.finish == "turns" for t in trajectories) / n,
                }
            )
        per_reward: dict[str, list[float]] = defaultdict(list)
        for t in trajectories:
            for name, value in t.rewards.items():
                if value is not None:
                    per_reward[name].append(value)
        stats.update({f"rewards/{name}": sum(v) / len(v) for name, v in per_reward.items()})
        return stats

    # --------------------------------------------------------------- evaluation

    def evaluate(self, rows: list[dict], temperature: float) -> dict[str, float]:
        """One rollout per held-out row with the newest weights; mean rewards and lengths."""
        with self.lock:
            self.weights.sync(self.engine)
            self.engine.wake()
            groups = self._await(self._eval(rows, temperature))
            self.engine.sleep()
        return self._group_stats([t for g in groups for t in g])

    async def _eval(self, rows: list[dict], temperature: float) -> list[list[Trajectory]]:
        return list(await asyncio.gather(*(self._group(-1, row, temperature, 1, False) for row in rows)))

    def close(self) -> None:
        if self.continuing:  # partial rollouts nobody will train on

            async def cancel(tasks):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            self._await(cancel(self.continuing))
            self.continuing = set()
        if self.env_pool is not None:
            try:
                self._await(self.env_pool.shutdown())
            except Exception:  # noqa: BLE001 — best effort at shutdown
                logger.warning("environment shutdown failed", exc_info=True)
        if self.sandbox is not None:
            try:
                self._await(self.sandbox.close())
            except Exception:  # noqa: BLE001 — best effort at shutdown
                logger.warning("sandbox close failed", exc_info=True)
        self.client.close()
        self.loop.call_soon_threadsafe(self.loop.stop)


def _timeline_stats(timeline: list[tuple[float, int, int]]) -> dict[str, float]:
    """Engine utilization over a batch from (time, sequences in flight, finished) per step:
    the time-weighted mean of sequences decoding at once, and the tail, the time from when
    fewer than a quarter of the peak were left until the last step."""
    if len(timeline) < 2:
        return {}
    times = [t for t, _, _ in timeline]
    live = [n for _, n, _ in timeline]
    span = times[-1] - times[0]
    weighted = sum(live[i] * (times[i + 1] - times[i]) for i in range(len(times) - 1))
    peak = max(live)
    tail_start = next(
        (times[i] for i in range(len(live)) if live[i] < peak / 4 and times[i] > times[live.index(peak)]), times[-1]
    )
    return {
        "engine/steps": len(timeline),
        "engine/seqs_mean": weighted / max(span, 1e-9),
        "engine/seqs_peak": peak,
        "engine/tail_s": times[-1] - tail_start,
        "engine/step_ms": 1000 * span / (len(timeline) - 1),
    }
