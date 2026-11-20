"""Credit for multi-role episodes: who is compared with whom, how advantages are scaled, and how much of the
loss each role gets.

Comparison groups. A row's episodes yield sessions of several roles; a baseline is only valid among sessions that
could have been each other: the same role at the same place of the episode (runtime: Session.key). The pipeline
tags every trajectory with that key and the trainer splits each row's trajectories into its comparison groups, so
the group baseline of single-agent RL applies to each of them unchanged.

    advantage = (reward - baseline of its comparison group) / scale

  baseline   the mean reward of the group's episodes ("mean"), or of the other episodes ("leave_one_out"). The
             unit is the episode (the sub-episode, under a branch), not the session: when one episode has several
             sessions in a group (the workers of one lead), each episode counts once, and leave-one-out leaves out
             every session of the same episode, which share its outcome.
  scale      loss.advantage_std: "role" divides by the standard deviation of the centred rewards of the same role
             over the batch, never less than loss.advantage_std_floor; "batch" pools every role (single-agent
             RL's default); "group" and "none" as in single-agent RL. Per role, a role with a wide reward does not
             set the size of another's gradient; the floor keeps a role whose rewards barely differ in a batch
             from having that noise scaled up to unit size.

Shares. With the default "prompt" aggregation every comparison group weighs the same, so a role with more groups
per row (the receiver under a branch, the many workers of one lead) would get that many times the loss of the
others, by accident of the scenario's shape. Here the step's loss is

    L = sum over roles of  share_role / (sum of the shares of the roles in the batch)  x  L_role

with L_role the usual aggregation over that role's groups alone. The shares are `roles.<name>.share` (default 1).
Roles that train separate adapters are stepped by an optimizer that rescales each parameter's gradient (Adam), so
there a share only matters through gradient clipping, which is done per adapter for that reason.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from palingenesis.rl import losses
from palingenesis.rl.trajectory import group_is_informative


class RowGroup(list):
    """The trajectories of one row's episodes, all roles together: what the pipeline hands on as one "group"
    (kept, carried, dropped as stale and dumped as a whole), split into comparison groups for training."""

    @property
    def comparison_groups(self) -> list[list[Any]]:
        parts: dict[tuple, list] = defaultdict(list)
        for t in self:
            parts[t.comparison].append(t)
        return list(parts.values())


def comparison_groups(groups: list) -> list[list[Any]]:
    """The comparison groups of a batch that can train (their rewards differ and their role trains)."""
    out = []
    for group in groups:
        for part in getattr(group, "comparison_groups", None) or [group]:
            if group_is_informative(part):
                out.append(part)
    return out


def _role(group: list) -> str:
    return group[0].role if group else ""


def assign_advantages(
    groups: list[list[Any]],
    std: str = "role",
    floor: float = 0.0,
    eps: float = 1e-6,
    reduce: Callable[[list[float]], list[float]] | None = None,
    baseline: str = "mean",
    roles: tuple[str, ...] = (),
) -> dict[str, float]:
    """Advantages of comparison groups, in place (see the module docstring). `roles`: every role name, in a fixed
    order (the per-role sums are reduced over data-parallel ranks in that order). Returns advantage_abs and, per
    role, role/<name>/advantage_abs and role/<name>/advantage_std."""
    names = list(roles) or sorted({_role(g) for g in groups})
    centred: dict[str, list[float]] = {name: [] for name in names}
    for group in groups:
        scored = [t for t in group if t.scored]
        for t in group:
            t.advantage = 0.0
        if not scored:
            continue
        units: dict[Any, list[float]] = defaultdict(list)
        for t in scored:
            units[t.unit].append(t.reward)
        means = {unit: sum(values) / len(values) for unit, values in units.items()}
        total = sum(means.values())
        for t in scored:
            if baseline == "leave_one_out":
                others = len(means) - 1
                t.advantage = t.reward - (total - means[t.unit]) / others if others else 0.0
            else:
                t.advantage = t.reward - total / len(means)
        if std == "group":
            sd = math.sqrt(sum(t.advantage**2 for t in scored) / len(scored))
            for t in scored:
                t.advantage /= sd + eps
        centred.setdefault(_role(group), []).extend(t.advantage for t in scored)
    names = list(centred)
    sums = [
        x
        for name in names
        for x in (sum(a * a for a in centred[name]), len(centred[name]), sum(map(abs, centred[name])))
    ]
    if reduce is not None:
        sums = reduce(sums)
    per_role = {name: sums[3 * i : 3 * i + 3] for i, name in enumerate(names)}
    squares, count, absolute = (sum(v[k] for v in per_role.values()) for k in range(3))
    pooled = math.sqrt(squares / count) if count else 0.0
    scale = {}
    for name, (sq, n, _) in per_role.items():
        sd = math.sqrt(sq / n) if n else 0.0
        scale[name] = max(sd, floor) + eps if std == "role" else pooled + eps if std == "batch" else 1.0
    if std in ("role", "batch"):
        for group in groups:
            for t in group:
                t.advantage /= scale[_role(group)]
    stats = {"advantage_abs": absolute / max(1, count)}
    for name, (sq, n, ab) in per_role.items():
        if n:
            stats[f"role/{name}/advantage_abs"] = ab / n
            stats[f"role/{name}/advantage_std"] = math.sqrt(sq / n)
    return stats


def token_weights(
    groups: list[list[Any]],
    aggregation: str,
    budget: int,
    shares: dict[str, float],
    sequence_level: bool = False,
    reduce: Callable[[list[float]], list[float]] | None = None,
) -> dict[int, float]:
    """Per-trajectory token weights (losses.token_weights) with each role's part of the loss set by its share:
    the role's groups are aggregated by themselves and their weights multiplied by share / (shares of the roles
    that have something to train in this batch, over every rank). One role: exactly losses.token_weights."""
    names = sorted(shares)
    by_role: dict[str, list] = {name: [] for name in names}
    for group in groups:
        by_role.setdefault(_role(group), []).append(group)
    names = sorted(by_role)
    present = [float(any(t.trained and t.sampled_tokens for g in by_role[name] for t in g)) for name in names]
    if reduce is not None:
        present = reduce(present)
    total = sum(shares.get(name, 1.0) for name, has in zip(names, present) if has)
    weights: dict[int, float] = {}
    for name, has in zip(names, present):
        if not has:
            continue
        factor = shares.get(name, 1.0) / total
        role_weights = losses.token_weights(by_role[name], aggregation, budget, sequence_level, reduce)
        weights.update({key: value * factor for key, value in role_weights.items()})
    return weights
