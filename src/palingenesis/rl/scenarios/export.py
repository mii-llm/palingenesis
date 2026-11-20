"""Scenario episodes from the rollout dumps as training rows for the other trainers.

A run with `logging.dump_trajectories` writes one record per trained session (palingenesis.rollouts), tagged with
its role, episode, session and comparison group; the first record of each episode also carries the episode's
partners and channel messages (info.episode_record). From those records:

  sft_rows   one conversation per session ({"messages", "tools"} plus scenario / role / reward columns), by
             default the sessions that earned at least `min_reward`: data for `pgs train`, and with
             `min_reward=None` the policy's own episodes, failures included, for on-policy distillation
             (`sources.<name>.format: agent_traces`).
  dpo_rows   preference pairs ({"prompt", "chosen", "rejected"}) from comparison groups whose sessions started
             from the same prompt: the best against the worst when their rewards differ by `margin` or more.
             Only such groups give pairs, because only there do the two answers respond to the same thing; for
             the same reason a session that was answered by another role along the way (user messages after the
             prompt) gives none: its later turns respond to what only it was told.

Each row keeps where it came from (step, group, episode, session), so a row can be traced to its episode.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any


def session_messages(record: dict) -> list[dict]:
    """The conversation of a dumped session: its prompt, then each turn with its tool results and the messages
    it was answered with."""
    messages = [dict(m) for m in record.get("prompt") or []]
    for turn in record.get("turns") or []:
        message: dict[str, Any] = {"role": "assistant", "content": turn.get("content") or ""}
        if turn.get("reasoning_content"):
            message["reasoning_content"] = turn["reasoning_content"]
        calls = [c for c in turn.get("tool_calls") or [] if "gen_ai.tool.call.arguments" in c]
        if calls:
            message["tool_calls"] = [
                {
                    "id": c.get("gen_ai.tool.call.id"),
                    "type": "function",
                    "function": {"name": c.get("gen_ai.tool.name"), "arguments": c["gen_ai.tool.call.arguments"]},
                }
                for c in calls
            ]
        messages.append(message)
        for c in calls:
            if "gen_ai.tool.call.result" in c:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.get("gen_ai.tool.call.id"),
                        "name": c.get("gen_ai.tool.name"),
                        "content": c["gen_ai.tool.call.result"],
                    }
                )
        for observation in turn.get("observations") or []:
            if observation.get("content"):
                messages.append({"role": observation.get("role") or "user", "content": observation["content"]})
    return messages


def _source(record: dict) -> dict:
    return {
        "role": record.get("role"),
        "model": record.get("model"),
        "reward": record.get("reward"),
        "rewards": record.get("rewards"),
        "source": {
            "step": record.get("step"),
            "group": record.get("group"),
            "episode": record.get("episode"),
            "session": record.get("session"),
        },
    }


def scenario_records(records: Iterable[dict], roles: Iterable[str] | None = None) -> Iterator[dict]:
    """The dumped sessions of scenario runs (records with a role and content), optionally of some roles only."""
    wanted = set(roles) if roles else None
    for record in records:
        if record.get("role") and "prompt" in record and (wanted is None or record["role"] in wanted):
            yield record


def sft_rows(
    records: Iterable[dict], roles: Iterable[str] | None = None, min_reward: float | None = 1.0
) -> Iterator[dict]:
    for record in scenario_records(records, roles):
        if record.get("finish") == "error" or not record.get("scored", True):
            continue
        if min_reward is not None and not (record.get("reward") or 0.0) >= min_reward:
            continue
        row = {"messages": session_messages(record), **_source(record)}
        tools = (record.get("info") or {}).get("tools")
        if tools:
            row["tools"] = tools
        yield row


def dpo_rows(records: Iterable[dict], roles: Iterable[str] | None = None, margin: float = 0.5) -> Iterator[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in scenario_records(records, roles):
        if record.get("scored", True) and record.get("finish") != "error":
            key = (record.get("step"), record.get("rank", 0), json.dumps(record.get("comparison")))
            groups[key].append(record)
    for members in groups.values():
        prompt = json.dumps(members[0].get("prompt"), sort_keys=True)
        if len(members) < 2 or any(json.dumps(m.get("prompt"), sort_keys=True) != prompt for m in members):
            continue
        best = max(members, key=lambda m: m.get("reward") or 0.0)
        worst = min(members, key=lambda m: m.get("reward") or 0.0)
        if (best.get("reward") or 0.0) - (worst.get("reward") or 0.0) < margin:
            continue
        start = len(best.get("prompt") or [])
        chosen, rejected = session_messages(best)[start:], session_messages(worst)[start:]
        if not chosen or not rejected or chosen == rejected:
            continue
        if any(m["role"] == "user" for m in chosen + rejected):
            continue
        yield {
            "prompt": best["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "role": best.get("role"),
            "reward_chosen": best.get("reward"),
            "reward_rejected": worst.get("reward"),
            "source": _source(best)["source"],
        }
