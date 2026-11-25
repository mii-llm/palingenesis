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
import json
import logging
import math
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from palingenesis.accelerator import on_current_device
from palingenesis.evalstats import pass_at_k
from palingenesis.multimodal import MultimodalError, has_image_parts, load_image, split_images, vision_of
from palingenesis.opd.orchestrator import PublishedWeights
from palingenesis.opd.rollout import RolloutEngine
from palingenesis.rl.chat import (
    TRUNCATION_MARKER,
    ChatFormat,
    Observation,
    ToolCall,
    encode_prompt,
    message_texts,
    parse_assistant,
)
from palingenesis.rl.concurrency import TokenAllowance, gate_for, merge_metrics
from palingenesis.rl.config import RLConfig
from palingenesis.rl.data import PromptSampler, prompt_messages, row_limit
from palingenesis.rl.env import EnvPool, ToolOutput, call_sync_or_async, row_tools, run_tool
from palingenesis.rl.envs.harness.profile import FORMAT_ERROR
from palingenesis.rl.generation import GenerationClient
from palingenesis.rl.rewards import SKIPPED, Reward, SkipSample, load_object, whitespace_defects
from palingenesis.rl.trajectory import Trajectory, group_is_informative
from palingenesis.rollouts import pick

logger = logging.getLogger(__name__)


@dataclass
class RLBatch:
    samples: list[list[Trajectory]]  # groups (the orchestrator counts len(samples) when it drops a batch)
    version: int
    stats: dict[str, float] = field(default_factory=dict)
    # logging.dump_trajectories: the trajectories to dump as (kept, trajectory), zero-variance groups included, and
    # the engine's concurrency timeline. The trainer hands them to the writer once the advantages are assigned.
    dump: list[tuple[bool, Trajectory]] | None = None
    timeline: list | None = None


class RLPipeline:
    vision = None  # a multimodal policy's image processor (palingenesis.multimodal), from the tokenizer
    _warned_images = False

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
        self.vision = vision_of(tok)
        self.group_shaping = load_object(config.loss.group_shaping) if config.loss.group_shaping else None
        # loss.observation_weight: training rollouts record their observation tokens (ChatFormat.continuation)
        self.observation_loss = config.loss.observation_weight > 0
        self.observation_exclude = self.observation_patterns(config.loss.observation_exclude)
        # n-sample merging only helps when a group's rollouts are one turn each
        if hasattr(engine, "make_client"):  # engines in other processes (palingenesis.rl.pool): requests travel
            self.client = engine.make_client(merge=env_pool is None)
        else:  # rollout.concurrency kv: a vLLM server's requests in flight under its KV-sized cap
            self.client = GenerationClient(engine, merge=env_pool is None, gate=gate_for(config, engine))
        r = config.rollout
        self.allowance = TokenAllowance(r.token_allowance, r.group_size) if r.token_allowance > 0 else None
        self.lock = threading.RLock()
        self.group_ids = itertools.count()
        self.carry: list[list[Trajectory]] = []  # surplus informative groups (max_staleness > 0)
        self.continuing: set[asyncio.Task] = set()  # groups still generating at the last batch's end
        self.drop_rate = 0.0  # expected fraction of zero-variance groups (running, per batch)
        self.reported_errors = 0
        self.batches = 0  # batches collected (the dump's step numbers; the trainer sets the resumed step)
        self.rollout_start = 0.0  # loop time the current batch's collection started
        self.dropped: list[list[Trajectory]] = []  # zero-variance groups of the batch (dumps)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=on_current_device(self.loop.run_forever), name="pgs-rl-rollouts", daemon=True
        )
        self.loop_thread.start()

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
        stats.update(self.concurrency_metrics())
        if hasattr(self.engine, "prompt_tokens"):
            prompt = self.engine.prompt_tokens - prefill[0]
            stats["engine/prefill_tokens"] = prompt - (self.engine.cached_tokens - prefill[1])
            stats["engine/prefix_hit"] = (self.engine.cached_tokens - prefill[1]) / max(prompt, 1)
        every = self.config.logging.dump_trajectories
        dump = timeline = None
        if every and self.batches % every == 0:
            sampled = pick(
                [(True, g) for g in groups] + [(False, g) for g in self.dropped],
                self.config.logging.dump_groups,
                self.batches,
            )
            dump = [(kept, t) for kept, group in sampled for t in group]
            timeline = list(self.client.timeline)
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
        return RLBatch(groups, version, stats, dump, timeline)

    def concurrency_metrics(self) -> dict[str, float]:
        """rollout.concurrency kv and rollout.token_allowance: the gates' and the allowance's metrics."""
        clients = getattr(self, "clients", None) or {"": self.client}
        gates = {id(g): g for g in (getattr(c, "gate", None) for c in clients.values()) if g is not None}
        out = merge_metrics([g.metrics() for g in gates.values()])
        if self.allowance is not None:
            out.update(self.allowance.metrics())
        return out

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
        every = self.config.logging.dump_trajectories
        dumping = bool(every) and (self.batches + 1) % every == 0  # zero-variance groups are kept for dumps only

        finished: list[list[Trajectory]] = []  # every finished group (kept or not): unbiased pass@k / family stats

        def account(group: list[Trajectory]) -> bool:
            nonlocal dropped_tokens
            counts["groups"] += 1
            if self.allowance is not None:
                self.allowance.observe([t.context_tokens for t in group])
            finished.append(group)
            if any(t.rewards.get("env/metric/verifier_failed") for t in group):
                counts["verifier_failed"] += 1  # a broken verifier, not a failure of the policy (the queue skips it)
            if group_is_informative(group, pass_k=self.config.loss.pass_k):
                return True
            counts["zero_variance"] += 1
            dropped_tokens += sum(t.sampled_tokens for t in group)
            if dumping:
                self.dropped.append(group)
            return False

        while len(kept) < target:
            needed = target - len(kept) - len(in_flight - adopted)
            missing = math.ceil(needed / (1.0 - min(self.drop_rate, 0.8))) if needed > 0 else needed
            launches = max(0, min(missing, budget - launched))
            if self.allowance is not None:  # rollout.token_allowance: speculative launches within the allowance
                held = sum(t.context_tokens for g in [*kept, *self.carry] for t in g)
                launches = self.allowance.launches(min(needed, launches), launches, target, held, len(in_flight))
            for _ in range(launches):
                # the policy version is the retry queue's clock (delays and the cooldown count training steps)
                index, row = self.sampler.draw(self.weights.version)
                revisit = bool(getattr(self.sampler, "last_revisit", False))
                counts["revisits"] += revisit
                in_flight.add(asyncio.create_task(self._group(index, row, temperature, r.group_size, True, revisit)))
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
        stats.update(self._outcome_stats(finished))
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
        if self.env_pool is not None:
            stats["groups/verifier_failed"] = counts["verifier_failed"]
        if hasattr(self.sampler, "metrics"):
            stats.update(self.sampler.metrics())  # the sampler's cumulative counters
        if getattr(self.sampler, "queue_on", False):
            stats["sampler/revisit_share_drawn"] = counts["revisits"] / max(launched, 1)
        return kept, stats

    # ------------------------------------------------------------------ groups

    async def _group(
        self, index: int, row: dict, temperature: float, size: int, training: bool, revisit: bool = False
    ) -> list[Trajectory]:
        gid = next(self.group_ids)
        # think-skip: a fixed share of each training group starts with a closed think (the row's view decides if it applies)
        n_skip = round(self.config.rollout.think_skip * size) if training else 0
        results = await asyncio.gather(
            *(self._episode(row, gid, temperature, training, k < n_skip) for k in range(size))
        )
        group = [t for t, _ in results]
        envs = [env for _, env in results]
        if revisit:  # served from the retry queue: outcome/*_fresh and *_revisit keep the two mixes apart
            for t in group:
                t.info["revisit"] = True
        try:
            await self._score(group, envs)
        finally:
            if self.env_pool is not None:
                await asyncio.gather(*(self.env_pool.release(env) for env in envs if env is not None))
        if training:
            scored = [t for t in group if t.scored]
            if scored:
                # The queue and the sampler's counters judge the raw task outcome (the weighted reward before group
                # shaping and the overlong penalty: a correct episode in the overlong buffer is still a success), over
                # the rollouts whose verifier worked: a broken task (env metric verifier_failed) is not a failure to
                # retry, and a group where the verifier failed on more than half of the scored rollouts is not judged
                # at all (solved None: it still counts for retire_above, on the rewards it trains on).
                judged = [t for t in scored if not t.rewards.get("env/metric/verifier_failed")]
                solved = None
                if judged and 2 * len(judged) >= len(scored):
                    threshold = self.config.data.success_threshold
                    solved = sum(t.info.get("raw_reward", t.reward) >= threshold for t in judged) / len(judged)
                mean = sum(t.reward for t in scored) / len(scored)
                self.sampler.observe(index, mean, solved, self.weights.version)
            elif hasattr(self.sampler, "release"):
                self.sampler.release(index)  # every rollout errored: the row is no longer in flight
        return group

    @property
    def _early_release(self) -> bool:
        """Score each episode's environment as soon as the episode ends and release it then, instead of holding every
        finished environment (a Harbor container, a remote session) until the slowest episode of its group ends; only
        when no reward function asks for the live environment (env / envs)."""
        pool = self.env_pool
        return (
            pool is not None
            and pool.has_reward
            and not any(r.accepts_all or {"env", "envs"} & r.accepts for r in self.rewards)
        )

    async def _episode(
        self, row: dict, gid: int, temperature: float, training: bool, skip: bool = False
    ) -> tuple[Trajectory, Any]:
        trajectory, env = await self._trajectory(row, gid, temperature, training, skip)
        if env is not None and trajectory.finish != "error" and self._early_release:
            try:
                trajectory.info["_env_reward"] = await self._env_reward(env, trajectory)
            finally:
                await self.env_pool.release(env)
            env = None
        return trajectory, env

    async def _trajectory(
        self, row: dict, gid: int, temperature: float, training: bool, skip: bool = False
    ) -> tuple[Trajectory, Any]:
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
                if hasattr(env, "episode_group"):  # environments that vary per GRPO group (Harbor harness profiles)
                    env.episode_group = gid
                start = await call_sync_or_async(env.reset, **row)
                trajectory.info.update(getattr(env, "episode_info", None) or {})  # e.g. the harness profile drawn
                if isinstance(start, str) and start:
                    last = next(m for m in reversed(messages) if m["role"] == "user")
                    last["content"] = f"{last['content']}\n\n{start}" if last["content"] else start
                elif isinstance(start, list):
                    messages[:] = [dict(m) for m in start]
            # the environment's tools (its methods, or tool_schemas() for this episode), else the row's own
            schemas = await self.env_pool.episode_schemas(env) if env is not None else None
            schemas = schemas or row_tools(row, config.data.tools_field)
            by_name = {s["function"]["name"]: s for s in schemas or []}
            if row.get("replay_messages"):  # a recorded prefix: its calls run again, its messages join the prompt
                if training:
                    raise ValueError("replay_messages is for evaluation and data generation, not for training rows")
                await self._replay(env, row["replay_messages"], set(by_name), messages, trajectory)
            trajectory.info["prompt"] = [dict(m) for m in messages]
            # a row may carry its own template kwargs (thinking or not) and generation budgets
            # (the environment's episode kwargs, e.g. a harness profile's enable_thinking, under the row's own)
            episode = (
                (getattr(env, "episode_info", None) or {}).get("chat_template_kwargs") if env is not None else None
            )
            chat = self.chat.with_kwargs({**(episode or {}), **(row.get("chat_template_kwargs") or {})})
            rendered, shown = self._split_images(messages)
            trajectory.prompt_ids = encode_prompt(self.tok, rendered, schemas, chat.kwargs)
            self._add_images(trajectory, trajectory.prompt_ids, shown)
            vision = self.vision
            image_results = (getattr(env, "episode_info", None) or {}).get("image_results") or e.image_results
            # a row's own limits: positive integers (row_limit raises on anything else; the trainer checks at load)
            budget = row_limit(row, "max_completion_tokens") or config.completion_budget
            row_new_tokens = row_limit(row, "max_new_tokens") or r.max_new_tokens
            trajectory.info["budget"] = budget
            # a row may raise or lower its own turn limit (long agentic tasks next to short ones in one mix); give it
            # max_completion_tokens too, or its turns share the default episode budget (check_row_limits warns)
            max_turns = 1 if env is None else (row_limit(row, "max_turns") or e.max_turns)
            trajectory.info["max_turns"] = max_turns
            forced: list[int] = []
            if skip and chat.thinking and chat.opens_think:
                forced = self.tok.encode(r.think_skip_prefix, add_special_tokens=False)
                trajectory.info["think_skip"] = True
            for turn in range(max_turns):
                if vision is not None and 0 < e.max_images < len(trajectory.images) - e.image_prune_slack:
                    # old screenshots leave the context; what was sampled so far stays trained on what it saw
                    kept = chat.drop_images(
                        trajectory.prompt_ids + trajectory.tokens,
                        vision.image_token_id,
                        e.max_images,
                        e.image_placeholder,
                    )
                    trajectory.rewrite_context(
                        kept,
                        trajectory.images[-e.max_images :],
                        trajectory.image_tokens[-e.max_images :],
                        vision.image_token_id,
                    )
                    trajectory.info["context_rewrites"] = len(trajectory.segments)
                context = trajectory.context_tokens
                room = min(
                    row_new_tokens,
                    budget - trajectory.sampled_tokens,
                    r.max_model_len - context,
                )
                prefix, forced = forced, []  # first turn only
                room -= len(prefix)
                if room <= 0:
                    trajectory.finish = "length"
                    break
                if prefix:
                    trajectory.append_forced(prefix, math.log(r.think_skip))
                t_generate = clock()
                # stop resampling (evaluation only): a turn that would stop the episode is sampled again
                resamples = 0 if training or env is None or turn == max_turns - 1 else r.stop_resample
                for attempt in range(resamples + 1):
                    out = await self.client.generate(
                        trajectory.prompt_ids + trajectory.tokens,
                        room,
                        temperature,
                        *([trajectory.images] if trajectory.images else []),  # only multimodal engines take images
                    )
                    if vision is not None and vision.image_token_id in out.ids:
                        # an image placeholder is an input; sampled, it has no image to stand for
                        raise MultimodalError("the policy sampled an image placeholder token")
                    text = self.tok.decode(
                        prefix + [t for t in out.ids if t not in self.stop_ids],
                        skip_special_tokens=True,
                    )
                    parsed = parse_assistant(text, chat, e.tool_parser, by_name, f"call_{turn}")
                    # text-protocol environments (tools described in the prompt as tags, ReAct lines, JSON actions...):
                    # the environment parses actions from the assistant's text and answers in a user turn
                    text_protocol = False
                    if (
                        out.finish != "length"
                        and env is not None
                        and not (parsed.calls or parsed.errors)
                        and hasattr(env, "text_actions")
                    ):
                        actions = await call_sync_or_async(env.text_actions, parsed.content) or []
                        parsed.calls = [
                            ToolCall(name, dict(args or {}), f"call_{turn}_{k}")
                            for k, (name, args) in enumerate(actions)
                        ]
                        text_protocol = bool(parsed.calls)
                    if out.finish == "length" or parsed.calls or parsed.errors or attempt == resamples:
                        break
                    trajectory.info.setdefault("stop_resamples", []).append({"turn": turn, "message": parsed.message()})
                turn_timing = {
                    "generate_s": clock() - t_generate,
                    "tokens": len(out.ids),
                    "context": context,
                    "finish": out.finish,
                }
                timing["turns"].append(turn_timing)
                logprobs = out.logprobs if out.logprobs or training else [0.0] * len(out.ids)
                trajectory.append_generated(out.ids, logprobs, out.version, out.first_version)
                if training:
                    trajectory.extend_routing(out.routed_experts)
                trajectory.messages.append(parsed.message())
                trajectory.info.setdefault("raw_contents", []).append(parsed.raw_content)
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
                exposed = None if text_protocol else set(by_name)  # text-protocol actions: the environment validates

                async def timed(call: ToolCall, exposed=exposed, started=t_tools) -> tuple[Any, bool, float]:
                    result, failed = await run_tool(env, call.name, call.arguments, e.tool_timeout, exposed)
                    return result, failed, clock() - started  # the calls run at once: seconds to its own result

                results = await asyncio.gather(*(timed(c) for c in parsed.calls))
                turn_timing["tools_s"] = clock() - t_tools
                turn_timing["tools"] = [
                    {"name": c.name, "s": seconds, "failed": failed}
                    for c, (_, failed, seconds) in zip(parsed.calls, results)
                ]
                # a tool may return images next to its text (rl.env.ToolOutput)
                texts = [out.text if isinstance(out, ToolOutput) else out for out, _, _ in results]
                pictures = [list(out.images) if isinstance(out, ToolOutput) else [] for out, _, _ in results]
                observations = [
                    {
                        "role": "tool",
                        "tool_call_id": c.id,
                        "name": c.name,
                        "content": text,
                    }
                    for c, text in zip(parsed.calls, texts)
                ]
                observations += [{"role": "tool", "content": error} for error in parsed.errors]
                # the trainer's own messages (a malformed call's error) are not the environment's output
                harness = [False] * len(parsed.calls) + [True] * len(parsed.errors)
                if text_protocol:  # one user turn with the results, in the environment's own format
                    render = getattr(env, "observation_text", None)
                    body = (
                        render([(c.name, out_text) for c, out_text in zip(parsed.calls, texts)])
                        if render
                        else "\n\n".join(texts)
                    )
                    observations = [{"role": "user", "content": body}]
                    harness = [all(c.name == FORMAT_ERROR for c in parsed.calls)]  # the harness's format error
                harness = [h or not o["content"] for h, o in zip(harness, observations)]  # an image's note fills it
                for o in observations:
                    o["content"] = self.chat.truncate(self.chat.sanitize(o["content"]), e.max_tool_output_tokens)
                if any(pictures):
                    self._attach_images(observations, pictures, image_results, text_protocol)
                trajectory.tool_calls += len(parsed.calls)
                trajectory.tool_errors += sum(failed for _, failed, _ in results) + len(parsed.errors)
                if getattr(env, "done", False):  # the environment ended the episode (e.g. a submit tool)
                    trajectory.messages.extend(observations)
                    trajectory.finish = "env_done"
                    break
                harness += [True] * (len(observations) - len(harness))  # env.image_results "user": the note's message
                rendered, shown = self._split_images(observations)
                calls = None if text_protocol else parsed.calls
                marked: list[tuple[int, int]] = []
                if training and self.observation_loss:
                    notes = self._observation_notes(rendered, harness, env)
                    context_ids, marked = chat.continuation(rendered, schemas, calls, notes)
                else:
                    context_ids = chat.continuation_ids(rendered, schemas, calls)
                if out.ids and out.ids[-1] != self.chat.eot_id:  # the turn ended on another stop token
                    context_ids = [self.chat.eot_id] + context_ids
                    marked = [(i + 1, kind) for i, kind in marked]
                shown, image_tokens = self._measure_images(context_ids, shown)
                added = len(context_ids) + sum(n - 1 for n in image_tokens)
                if trajectory.context_tokens + added >= r.max_model_len - 1:
                    trajectory.finish = "length"
                    break
                if marked:
                    trajectory.append_observations(len(trajectory.tokens), marked)
                trajectory.append_context(context_ids)
                trajectory.append_images(shown, image_tokens)
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

    async def _replay(
        self, env: Any, recorded: Any, exposed: set[str], messages: list[dict], trajectory: Trajectory
    ) -> None:
        """Resume an episode from a recorded prefix (a row's replay_messages): run every recorded call in the
        environment, in order and through the path a sampled turn takes, then append the recorded messages to the
        prompt. Each result, normalised as a live observation is, is compared with the recorded one (trailing
        whitespace aside) and the mismatches counted in trajectory.info["replay"]. Raises ReplayError when a call
        fails where the recording did not, or when the environment ends the episode during the replay."""
        if env is None:
            raise ValueError("replay_messages needs an environment to run the recorded calls in")
        if _text_protocol(env):
            raise ValueError("replay_messages is not supported for text-protocol environments or harness profiles")
        replayed, plan = _replay_plan(recorded)
        e = self.config.env
        info = trajectory.info["replay"] = {
            "messages": len(replayed),
            "calls": 0,
            "mismatches": 0,
            "first_mismatch": None,
        }
        for k, (name, arguments, expected) in enumerate(plan):
            result, failed = await run_tool(env, name, arguments, e.tool_timeout, exposed)
            text = result.text if isinstance(result, ToolOutput) else result
            got = self.chat.truncate(self.chat.sanitize(text), e.max_tool_output_tokens)
            info["calls"] += 1
            matched = _trailing_stripped(got) == _trailing_stripped(expected)
            if not matched:
                info["mismatches"] += 1
                if info["first_mismatch"] is None:
                    info["first_mismatch"] = {"call": name, "expected": expected[:300], "got": got[:300]}
            if failed and not matched:  # the recording did not fail this way: the state it led to is not there
                raise ReplayError(f"replayed call {k + 1} of {len(plan)} ({name}) failed: {got[:300]}")
            if getattr(env, "done", False):
                raise ReplayError(
                    f"the environment ended the episode during the replay (call {k + 1} of {len(plan)}: {name})"
                )
        messages.extend(replayed)

    @staticmethod
    def observation_patterns(exclude: list | None) -> list[re.Pattern]:
        """loss.observation_exclude compiled, and the truncation marker (the trainer's own text)."""
        return [re.compile(str(p), re.DOTALL) for p in exclude or []] + [re.compile(TRUNCATION_MARKER)]

    def _observation_notes(self, messages: list[dict], harness: list[bool], env: Any) -> list[Observation]:
        """Per observation message (as rendered), what of it is not the environment's output: the whole message
        when it is the trainer's or the harness's own, else the spans of each text part that loss.observation_exclude,
        the truncation marker or the environment's observation_notes match."""
        patterns = self.observation_exclude + [
            re.compile(str(p), re.DOTALL) for p in (getattr(env, "observation_notes", None) or ())
        ]
        return [
            Observation(
                [
                    [m.span() for p in patterns for m in p.finditer(text) if m.end() > m.start()]
                    for text in message_texts(o)
                ],
                whole,
            )
            for o, whole in zip(messages, harness)
        ]

    # ------------------------------------------------------------------ images

    def _split_images(self, messages: list[dict]) -> tuple[list[dict], list[Any]]:
        """(messages for the chat template, their images in rendering order). A text-only policy
        cannot take images: they are an error in the prompt."""
        if self.vision is None:
            if has_image_parts(messages):
                raise MultimodalError(
                    "the conversation has images but the policy is text-only: set model.multimodal: true"
                )
            return messages, []
        return split_images(messages)

    def _measure_images(self, ids: list[int], images: list[Any]) -> tuple[list[Any], list[int]]:
        """The images loaded, and the tokens each expands to. The template must have written one
        placeholder per image into `ids`."""
        if not images:
            return [], []
        rendered = self.vision.count_images(ids)
        if rendered != len(images):
            raise MultimodalError(
                f"the chat template rendered {rendered} placeholder(s) for {len(images)} image(s): it does not "
                'render images in this role. For images returned by tools, set env.image_results: "user".'
            )
        loaded = [load_image(image) for image in images]
        return loaded, [self.vision.num_tokens(image.height, image.width) for image in loaded]

    def _add_images(self, trajectory: Trajectory, ids: list[int], images: list[Any]) -> None:
        trajectory.append_images(*self._measure_images(ids, images))

    def _attach_images(self, observations: list[dict], pictures: list[list], mode: str, text_protocol: bool) -> None:
        """Put the tools' images into the observation messages (in place). A text-only policy gets
        the text alone. Text-protocol environments answer in one user message: the images follow its
        text. Native tool calls: env.image_results."""
        if self.vision is None:
            if not self._warned_images:
                self._warned_images = True
                logger.warning("tools returned images but the policy is text-only: the images are dropped")
            return
        e = self.config.env

        def parts(images: list) -> list[dict]:
            return [{"type": "image", "image": image} for image in images]

        def with_text(text: str, images: list) -> list[dict]:
            return ([{"type": "text", "text": text}] if text else []) + parts(images)

        flat = [image for images in pictures for image in images]
        if text_protocol:
            observations[0]["content"] = with_text(observations[0]["content"], flat)
        elif mode == "tool":
            for message, images in zip(observations, pictures):
                if images:
                    message["content"] = with_text(message["content"], images)
        else:
            for message, images in zip(observations, pictures):
                if images and not message["content"]:
                    message["content"] = e.image_tool_note
            observations.append({"role": "user", "content": with_text(e.image_user_note, flat)})

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
            scores["env"] = list(
                await asyncio.gather(
                    *(self._env_reward(envs[i], group[i]) for i in live if "_env_reward" not in group[i].info)
                )
            )
            scores["env"] = [
                group[i].info.pop("_env_reward") if "_env_reward" in group[i].info else scores["env"].pop(0)
                for i in live
            ]
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
            # env/metric/<name>: an environment's statistic, logged with the rewards but not part of the reward
            applicable = [
                (weights.get(n, 1.0), v) for n, v in values.items() if v is not None and not n.startswith("env/metric/")
            ]
            if not applicable:
                t.scored = t.trained = False
                continue
            t.reward = sum(w * v for w, v in applicable)
            t.info["raw_reward"] = t.reward  # before group shaping and the overlong penalty (task success)
        if self.group_shaping is not None:
            scored = [group[i] for i in live if group[i].scored]
            if scored:
                before = [(t.reward, dict(t.rewards)) for t in scored]
                try:
                    await call_sync_or_async(self.group_shaping, scored, scored[0].row)
                except Exception:  # noqa: BLE001 — a broken shaping function must not stop the run
                    logger.exception("loss.group_shaping failed; the group keeps its rewards")
                    for t, (reward, rewards) in zip(scored, before):
                        t.reward, t.rewards = reward, rewards
        for i in live:
            t = group[i]
            if not t.scored:
                continue
            if loss.overlong_buffer:
                over = t.sampled_tokens - (t.info.get("budget", budget) - loss.overlong_buffer)
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
            "raw_contents": list(t.info.get("raw_contents", [])),
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
            # rollouts with a whitespace defect in a reply (leading space, single-space line start): no grader sees
            # it (everything strips), so it is logged on every run
            "format/whitespace": sum(bool(whitespace_defects(t.info.get("raw_contents", []))) for t in trajectories)
            / n,
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
        forced = [t for t in trajectories if t.info.get("think_skip")]
        if forced:  # think-skip: how often it applied and what skipping earned against thinking in the same groups
            groups = {t.group for t in forced}
            free = [t.reward for t in trajectories if t.scored and t.group in groups and not t.info.get("think_skip")]
            skipped = [t.reward for t in forced if t.scored]
            stats["think_skip/share"] = len(forced) / n
            if free and skipped:
                stats["think_skip/reward_gap"] = sum(skipped) / len(skipped) - sum(free) / len(free)
        per_reward: dict[str, list[float]] = defaultdict(list)
        for t in trajectories:
            for name, value in t.rewards.items():
                if value is not None:
                    per_reward[name].append(value)
        stats.update({f"rewards/{name}": sum(v) / len(v) for name, v in per_reward.items()})
        return stats

    def _outcome_stats(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        """Over every finished group, kept or not (the "reward" above covers only the informative groups that train,
        whose mix shifts as groups saturate): reward_all, pass@1 (share of rollouts at or above
        data.success_threshold), pass@G (share of groups with at least one such rollout: falling pass@G under rising
        pass@1 is the sharpening signature) and surprisal (mean -log p of sampled tokens, an entropy proxy). With
        data.metrics_key, the same per family as family/<name>/<metric>. With the retry queue, also pass@1_fresh and
        pass@1_revisit (rows drawn from the epoch order / from the queue)."""
        groups = [[t for t in g if t.scored] for g in groups]
        groups = [g for g in groups if g]
        if not groups:
            return {}
        d = self.config.data
        threshold = d.success_threshold

        k = self.config.loss.pass_k

        def summarize(gs: list[list[Trajectory]]) -> dict[str, float]:
            ts = [t for g in gs for t in g]
            sampled = [lp for t in ts for lp, m in zip(t.logprobs, t.mask) if m]
            extra = {}
            if k > 1:  # loss.pass_k: the trained objective, unbiased, over the groups with at least k scored rollouts
                full = [g for g in gs if len(g) >= k]
                if full:
                    extra[f"pass@{k}"] = sum(
                        pass_at_k(len(g), sum(t.reward >= threshold for t in g), k) for g in full
                    ) / len(full)
            return {
                **extra,
                "reward_all": sum(t.reward for t in ts) / len(ts),
                "pass@1": sum(t.reward >= threshold for t in ts) / len(ts),
                f"pass@{self.config.rollout.group_size}": sum(any(t.reward >= threshold for t in g) for g in gs)
                / len(gs),
                "surprisal": -sum(sampled) / max(1, len(sampled)),
                "completion_len": sum(t.sampled_tokens for t in ts) / len(ts),
                "groups": float(len(gs)),
            }

        out = {f"outcome/{k}": v for k, v in summarize(groups).items() if k != "completion_len"}
        if getattr(self.sampler, "queue_on", False):
            # the retry queue changes the prompt mix (hard and partly solved rows come back), so pass@1 above moves with
            # the mix as well as with the policy: fresh draws alone are comparable with a run without the queue
            for name, revisit in (("fresh", False), ("revisit", True)):
                ts = [t for g in groups if bool(g[0].info.get("revisit")) == revisit for t in g]
                if ts:
                    out[f"outcome/pass@1_{name}"] = sum(t.reward >= threshold for t in ts) / len(ts)
        if d.metrics_key:
            by: dict[str, list[list[Trajectory]]] = defaultdict(list)
            for g in groups:
                by[row_family(g[0].row, d.metrics_key)].append(g)
            for family, gs in by.items():
                out.update({f"family/{family}/{k}": v for k, v in summarize(gs).items()})
        return out

    # --------------------------------------------------------------- evaluation

    def evaluate(self, rows: list[dict], temperature: float) -> dict[str, float]:
        """One rollout per held-out row with the newest weights; mean rewards and lengths. With data.metrics_key, also
        per family (family/<name>/reward, pass@1, completion_len, n): the same fixed prompts at every evaluation, so a
        family's trend is not confounded by which prompts training happened to sample (training-time family curves
        are)."""
        with self.lock:
            with self.client.paused():  # groups still decoding from the last batch: load between engine steps
                self.weights.sync(self.engine)
            self.engine.wake()
            groups = self._await(self._eval(rows, temperature))
            self.engine.sleep()
        trajectories = [t for g in groups for t in g]
        stats = self._group_stats(trajectories)
        d = self.config.data
        if d.metrics_key:
            by: dict[str, list[Trajectory]] = defaultdict(list)
            for t in trajectories:
                if t.scored:
                    by[row_family(t.row, d.metrics_key)].append(t)
            for family, ts in by.items():
                stats[f"family/{family}/reward"] = sum(t.reward for t in ts) / len(ts)
                stats[f"family/{family}/pass@1"] = sum(t.reward >= d.success_threshold for t in ts) / len(ts)
                stats[f"family/{family}/completion_len"] = sum(t.sampled_tokens for t in ts) / len(ts)
                stats[f"family/{family}/n"] = float(len(ts))
        return stats

    async def _eval(self, rows: list[dict], temperature: float, size: int = 1) -> list[list[Trajectory]]:
        return list(await asyncio.gather(*(self._group(-1, row, temperature, size, False) for row in rows)))

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
        # Release the threads (a notebook runs many trainings in one process): the loop's default executor
        # (to_thread / run_in_executor calls), then the loop and its thread.
        if self.loop.is_running():

            async def drain():
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self.loop.shutdown_default_executor()

            try:  # bounded: a tool call stuck in a worker thread must not hang the shutdown
                asyncio.run_coroutine_threadsafe(drain(), self.loop).result(timeout=60)
            except Exception:  # noqa: BLE001 — best effort at shutdown
                logger.warning("rollout executor shutdown failed", exc_info=True)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=10)
        if not self.loop.is_running() and not self.loop.is_closed():
            self.loop.close()


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


def row_family(row: dict, keys: list[str]) -> str:
    """The first of `keys` (dotted paths; a JSON-string field is parsed) present in `row`, else "other"."""
    for key in keys:
        value: Any = row
        for part in key.split("."):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = None
            value = value.get(part) if isinstance(value, dict) else None
            if value is None:
                break
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return "other"


class ReplayError(RuntimeError):
    """A recorded prefix (a row's replay_messages) could not be replayed: the episode is not sampled."""


def _text_protocol(env: Any) -> bool:
    """Whether the environment's actions are parsed from the assistant's text (not native tool calls). An environment
    with a harness session (HarborEnv) defines text_actions for every profile: there the profile's protocol decides."""
    if not hasattr(env, "text_actions"):
        return False
    if hasattr(env, "session"):
        if env.session is None:  # no harness profile: native calls
            return False
        protocol = getattr(getattr(env.session, "profile", None), "protocol", None)
        return protocol is not None and not getattr(protocol, "native", True)
    return True


def _trailing_stripped(text: str) -> str:
    """`text` without trailing whitespace, on each line and at the end."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


def _replay_plan(recorded: Any) -> tuple[list[dict], list[tuple[str, dict, str]]]:
    """A row's replay_messages (a list of OpenAI-format messages, or its JSON string) as the messages to append (tool
    call arguments as objects, as a sampled turn records them) and the calls to run: (name, arguments, the recorded
    result's text) in order. A call's result is the tool message after it with its tool_call_id, else the one in its
    position. Raises ValueError on anything that is not assistant turns followed by one result per call."""
    if isinstance(recorded, str):
        recorded = json.loads(recorded)
    if not isinstance(recorded, list) or not all(isinstance(m, dict) for m in recorded):
        raise ValueError("replay_messages must be a list of messages")
    messages: list[dict] = []
    plan: list[tuple[str, dict, str]] = []
    blocks: list[tuple[list[dict], list[dict]]] = []  # (an assistant turn's calls, the tool messages after it)
    for i, m in enumerate(recorded):
        m = dict(m)
        role = m.get("role")
        if role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                function = dict(c.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments) if arguments.strip() else {}
                if not function.get("name") or not isinstance(arguments or {}, dict):
                    raise ValueError(f"replay_messages[{i}]: a tool call needs a name and object arguments")
                function["arguments"] = dict(arguments or {})
                calls.append({**c, "type": c.get("type") or "function", "function": function})
            if calls:
                m["tool_calls"] = calls
            if m.get("content") is None:
                m["content"] = ""
            blocks.append((calls, []))
        elif role == "tool":
            if not blocks:
                raise ValueError(f"replay_messages[{i}]: a tool message before any assistant turn")
            blocks[-1][1].append(m)
        else:
            raise ValueError(f"replay_messages[{i}]: role {role!r} (only assistant and tool messages are replayed)")
        messages.append(m)
    for calls, results in blocks:
        if len(results) != len(calls):
            raise ValueError(f"replay_messages: {len(calls)} tool call(s) followed by {len(results)} tool message(s)")
        by_id = {r["tool_call_id"]: r for r in results if r.get("tool_call_id")}
        for k, c in enumerate(calls):
            result = by_id.get(c.get("id")) if c.get("id") else None
            plan.append(
                (c["function"]["name"], c["function"]["arguments"], "\n".join(message_texts(result or results[k])))
            )
    return messages, plan
