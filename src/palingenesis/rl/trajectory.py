"""The trajectory record: token-native and append-only.

A trajectory is its prompt's token ids followed by everything after it, in the order
the policy saw it: the tokens it sampled (trained, with the sampler's log-probs) and
the context appended between turns (tool results and the chat template's glue; not
trained). Tokens are appended verbatim and never re-tokenized, so what the trainer
scores is exactly what the sampler produced (token-in, token-out).
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    row: dict[str, Any]  # the dataset row (every column reaches rewards and the env)
    group: int  # rollouts of the same prompt share a group id
    prompt_ids: list[int]
    messages: list[dict[str, Any]]  # the conversation, assistant turns included (for rewards)
    tokens: list[int] = field(default_factory=list)  # everything after the prompt
    mask: list[bool] = field(default_factory=list)  # True: sampled by the policy (trained)
    logprobs: list[float] = field(default_factory=list)  # sampler log-probs (0.0 where not sampled)
    versions: list[int] = field(default_factory=list)  # policy version of each assistant turn
    finish: str = ""  # stop | length | turns | error
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    rewards: dict[str, float | None] = field(default_factory=dict)
    reward: float = 0.0
    advantage: float = 0.0
    scored: bool = True  # False: no reward (infrastructure failure); out of baseline and loss
    trained: bool = True  # False: out of the loss (unscored, or a masked truncation)
    info: dict[str, Any] = field(default_factory=dict)  # environment / reward extras, logged

    def append_generated(self, ids: list[int], logprobs: list[float], version: int) -> None:
        if len(logprobs) != len(ids):
            raise ValueError(
                f"{len(ids)} sampled tokens but {len(logprobs)} log-probs: the rollout engine must "
                "return the sampler's log-prob of every token (temperature > 0)"
            )
        self.tokens += ids
        self.mask += [True] * len(ids)
        self.logprobs += [lp if math.isfinite(lp) else 0.0 for lp in logprobs]
        self.versions.append(version)
        self.turns += 1

    def append_context(self, ids: list[int]) -> None:
        self.tokens += ids
        self.mask += [False] * len(ids)
        self.logprobs += [0.0] * len(ids)

    @property
    def sampled_tokens(self) -> int:
        return sum(self.mask)

    @property
    def version(self) -> int:
        """The oldest policy version that sampled any of its tokens."""
        return min(self.versions) if self.versions else 0

    @property
    def completion(self) -> str:
        """The final assistant message's text."""
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message.get("content") or ""
        return ""


def group_is_informative(group: list[Trajectory], tol: float = 1e-6) -> bool:
    """Whether a group's scored rollouts disagree on reward (and one of them trains): with
    equal rewards every advantage is 0 and the group carries no gradient."""
    rewards = [t.reward for t in group if t.scored]
    return (len(rewards) >= 2 and max(rewards) - min(rewards) > tol) and any(t.trained for t in group)
