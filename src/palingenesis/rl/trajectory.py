"""The trajectory record: token-native and append-only.

A trajectory is its prompt's token ids followed by everything after it, in the order
the policy saw it: the tokens it sampled (trained, with the sampler's log-probs) and
the context appended between turns (tool results and the chat template's glue; not
trained). Tokens are appended verbatim and never re-tokenized, so what the trainer
scores is exactly what the sampler produced (token-in, token-out).

Images. The tokens hold one placeholder per image (what the chat template writes and what vLLM
takes next to the raw images); `images` are the pictures in placeholder order. The trainer expands
the placeholders with the model's own image processor (palingenesis.multimodal).

Context rewrites. Dropping old screenshots changes tokens the policy has already read, so the
sequence is no longer append-only. A rewrite closes the current segment (its prompt and the turns
sampled on it) and starts a new one whose prompt is the rewritten context. Every sampled token
belongs to exactly one segment and is trained on the context it was sampled from: `training_views`.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    """A closed stretch of a trajectory: the context as it was, and the turns sampled on it."""

    prompt_ids: list[int]
    tokens: list[int]
    mask: list[bool]
    logprobs: list[float]
    first_versions: list[int]
    turn_starts: list[int]
    forced: list[int]
    images: list[Any]
    image_tokens: list[int]


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
    # Per assistant turn: the version that sampled its first token (older than `versions` when weights were loaded
    # while it decoded: partial rollouts), and where it starts in `tokens`.
    first_versions: list[int] = field(default_factory=list)
    turn_starts: list[int] = field(default_factory=list)
    finish: str = ""  # stop | length | turns | error
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    rewards: dict[str, float | None] = field(default_factory=dict)
    reward: float = 0.0
    advantage: float = 0.0
    scored: bool = True  # False: no reward (infrastructure failure); out of baseline and loss
    trained: bool = True  # False: out of the loss (unscored, or a masked truncation)
    forced: list[int] = field(default_factory=list)  # token indices trained with a forcing probability (think-skip)
    info: dict[str, Any] = field(default_factory=dict)  # environment / reward extras, logged
    images: list[Any] = field(default_factory=list)  # the context's images, in placeholder order
    image_tokens: list[int] = field(default_factory=list)  # tokens each image expands to
    segments: list[Segment] = field(default_factory=list)  # closed by context rewrites (image pruning)

    def append_generated(
        self, ids: list[int], logprobs: list[float], version: int, first_version: int | None = None
    ) -> None:
        if len(logprobs) != len(ids):
            raise ValueError(
                f"{len(ids)} sampled tokens but {len(logprobs)} log-probs: the rollout engine must "
                "return the sampler's log-prob of every token (temperature > 0)"
            )
        self.first_versions.append(version if first_version is None else first_version)
        self.turn_starts.append(len(self.tokens))
        self.tokens += ids
        self.mask += [True] * len(ids)
        self.logprobs += [lp if math.isfinite(lp) else 0.0 for lp in logprobs]
        self.versions.append(version)
        self.turns += 1

    def append_forced(self, ids: list[int], logprob: float) -> None:
        """A prefix the sampler was made to write (think-skip): its first token is trained with the given behaviour
        log-prob (the forcing probability), the rest is context."""
        self.forced.append(len(self.tokens))
        self.tokens += ids
        self.mask += [True] + [False] * (len(ids) - 1)
        self.logprobs += [logprob] + [0.0] * (len(ids) - 1)

    def append_context(self, ids: list[int]) -> None:
        self.tokens += ids
        self.mask += [False] * len(ids)
        self.logprobs += [0.0] * len(ids)

    def append_images(self, images: list[Any], tokens: list[int]) -> None:
        """Images of the context just appended (one placeholder token each), and their token counts."""
        self.images += images
        self.image_tokens += tokens

    @property
    def context_tokens(self) -> int:
        """Tokens the policy reads now: the sequence with every image placeholder expanded."""
        return len(self.prompt_ids) + len(self.tokens) + sum(n - 1 for n in self.image_tokens)

    def rewrite_context(
        self, prompt_ids: list[int], images: list[Any], image_tokens: list[int], image_token_id: int | None = None
    ) -> None:
        """Replace what the policy reads from now on (old screenshots dropped): the turns sampled so
        far stay with the context they were sampled from, as a closed segment. The segment ends at
        its last sampled token: the context after it (and its images, counted by `image_token_id`)
        conditions nothing that is trained."""
        end = max((j + 1 for j, sampled in enumerate(self.mask) if sampled), default=0)
        shown = len(self.images)
        if image_token_id is not None:
            shown = sum(1 for token in self.prompt_ids + self.tokens[:end] if token == image_token_id)
        self.segments.append(
            Segment(
                self.prompt_ids,
                self.tokens[:end],
                self.mask[:end],
                self.logprobs[:end],
                self.first_versions,
                self.turn_starts,
                self.forced,
                self.images[:shown],
                self.image_tokens[:shown],
            )
        )
        self.prompt_ids, self.images, self.image_tokens = list(prompt_ids), list(images), list(image_tokens)
        self.tokens, self.mask, self.logprobs = [], [], []
        self.first_versions, self.turn_starts, self.forced = [], [], []

    def training_views(self) -> list["Trajectory"]:
        """The sequences to train on: the trajectory itself, or one per segment when its context was
        rewritten (each with the context its tokens were sampled from; reward, advantage and group
        are the trajectory's). Segments without sampled tokens are left out."""
        if not self.segments:
            return [self]
        views = [
            dataclasses.replace(
                self,
                segments=[],
                prompt_ids=segment.prompt_ids,
                tokens=segment.tokens,
                mask=segment.mask,
                logprobs=segment.logprobs,
                first_versions=segment.first_versions,
                turn_starts=segment.turn_starts,
                forced=segment.forced,
                images=segment.images,
                image_tokens=segment.image_tokens,
            )
            for segment in self.segments
        ]
        views.append(dataclasses.replace(self, segments=[]))
        return [v for v in views if any(v.mask)]

    @property
    def sampled_tokens(self) -> int:
        return sum(self.mask) + sum(sum(segment.mask) for segment in self.segments)

    def token_versions(self) -> list[int]:
        """Per token of `tokens`, the oldest policy version that sampled any token of its assistant turn (a turn
        decoded across a weight update counts as sampled by the older weights); -1 outside generated turns (context,
        a forced prefix) and for trajectories built without append_generated."""
        out = [-1] * len(self.tokens)
        ends = self.turn_starts[1:] + [len(self.tokens)]
        for start, end, version in zip(self.turn_starts, ends, self.first_versions):
            for j in range(start, end):
                if not self.mask[j]:  # the turn ended: context follows
                    break
                out[j] = version
        return out

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
