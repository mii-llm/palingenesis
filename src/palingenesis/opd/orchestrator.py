"""Producing scored rollouts: the pipeline, and the background thread that runs it.

Pipeline      prompts in, scored samples out: render -> sync weights -> generate
              -> clean -> align -> teacher scoring. Thread-safe (one lock), used
              by the orchestrator for training batches and by evaluation.
Orchestrator  a background thread that draws prompts and runs the pipeline one
              batch ahead of the trainer, through a bounded queue.

Staleness. The trainer consumes batch k at policy version k (k optimizer steps
done). The orchestrator starts generating batch k only once the trainer has
published version k - max_staleness, and syncs the newest weights first, so a
batch is at most max_staleness versions old when it is trained on:

  max_staleness 0  batch k is generated with the weights of step k: on-policy.
                   The thread still prepares the next prompts (sampling,
                   chat-template rendering) while the trainer trains.
  max_staleness 1  batch k+1 is generated (and teacher-scored) while the trainer
                   trains on batch k. Divergence losses need no correction (the
                   staleness only shifts which states are visited); the
                   policy-gradient losses correct with the importance ratio to
                   the rollout policy's log-probs.

A batch older than max_staleness when it reaches the trainer is dropped and
counted; with the gating above that only happens if the trainer skips versions.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from palingenesis.opd.align import TeacherView
from palingenesis.opd.formatting import encode_prompt
from palingenesis.opd.rollout import RolloutEngine, checkpoint_named_parameters
from palingenesis.opd.teachers import TeacherScores

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """One rollout to produce: the conversation rendered for the student and its teacher."""

    messages: list[dict[str, str]]
    max_new_tokens: int
    meta: dict[str, Any]
    source: str
    teacher: str
    prompt_ids: list[int]
    teacher_prompt_ids: list[int]


@dataclass
class Sample:
    """A scored rollout: the student's completion and the teacher's view and scores of it."""

    request: Request
    completion: list[int]             # cleaned student ids (ending with the stop token if it stopped)
    behaviour_lp: list[float]         # rollout policy's log-prob of each completion token
    version: int                      # policy version that generated it
    view: TeacherView
    scores: TeacherScores | None = None

    @property
    def teacher(self) -> str:
        return self.request.teacher


@dataclass
class Batch:
    samples: list[Sample]
    version: int
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class TeacherRoute:
    """How samples reach one teacher: its tokenizer, aligner, scorer, and what the loss needs."""

    tokenizer: Any
    aligner: Any                      # SharedVocabAligner | ByteChunkAligner
    teacher: Any                      # HFTeacher | VLLMTeacher
    top_k: int                        # 0: token log-probs only
    keep_hidden: bool                 # full_rkl: keep hidden states for the loss
    sample_rounds: int = 0            # rs_kd: tokens drawn from the teacher per position
    sample_temperature: float = 1.0   # rs_kd: the proposal's temperature


class PublishedWeights:
    """The student's weights as the trainer publishes them after each optimizer step.

    `lock` is held by the trainer while it changes the weights and by a rollout
    engine while it copies them, so an engine never loads a half-updated model.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.version = 0
        self.lock = threading.Lock()
        self._changed = threading.Condition()

    def publish(self) -> None:
        with self._changed:
            self.version += 1
            self._changed.notify_all()

    def wait_for(self, version: int, stop: threading.Event) -> bool:
        """Block until `version` is published; False if `stop` was set first."""
        with self._changed:
            while self.version < version and not stop.is_set():
                self._changed.wait(timeout=1.0)
        return not stop.is_set()

    def sync(self, engine: RolloutEngine) -> float:
        """Load the newest weights into `engine` if it is behind; seconds spent."""
        if engine.version >= self.version:
            return 0.0
        start = time.perf_counter()
        cuda = next(self.model.parameters()).is_cuda
        with self.lock:
            if cuda:                  # the optimizer's kernels (another stream) are done
                torch.cuda.synchronize()
            engine.update_weights(checkpoint_named_parameters(self.model), self.version)
            if cuda:                  # the copy is done before the optimizer may run again
                torch.cuda.synchronize()
        return time.perf_counter() - start


class Pipeline:
    """render -> sync -> generate -> clean -> align -> teacher scoring, under one lock.

    With `stream`, all of its GPU work runs on that CUDA stream, whichever thread
    calls, so the trainer's kernels on the default stream overlap it (the trainer
    passes one when a vLLM server generates during training; see OPDTrainer). Each
    call finishes its GPU work before releasing the lock, so what it returns is
    ready on any stream.
    """

    def __init__(self, student_tok, engine: RolloutEngine, routes: dict[str, TeacherRoute],
                 weights: PublishedWeights, chat_template_kwargs: dict, score_micro_seqs: int,
                 stream: torch.cuda.Stream | None = None):
        self.student_tok = student_tok
        self.engine = engine
        self.routes = routes
        self.weights = weights
        self.chat_template_kwargs = chat_template_kwargs
        self.score_micro_seqs = score_micro_seqs
        self.stream = stream
        self.lock = threading.RLock()

    @contextlib.contextmanager
    def _exclusive(self):
        with self.lock, torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext():
            yield
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

    def request(self, messages, max_new_tokens: int, meta: dict, source: str, teacher: str) -> Request:
        return Request(messages, max_new_tokens, meta, source, teacher,
                       encode_prompt(self.student_tok, messages, self.chat_template_kwargs),
                       encode_prompt(self.routes[teacher].tokenizer, messages, self.chat_template_kwargs))

    def generate(self, prompts: list[list[int]], max_new_tokens: list[int], temperature: float):
        """Rollouts with the newest weights; returns (rollouts, stats)."""
        with self._exclusive():
            sync = self.weights.sync(self.engine)
            t0 = time.perf_counter()
            self.engine.wake()
            t1 = time.perf_counter()
            rollouts = self.engine.generate(prompts, max_new_tokens, temperature)
            t2 = time.perf_counter()
            self.engine.sleep()
            t3 = time.perf_counter()
        tokens = sum(len(r.completion_ids) for r in rollouts)
        return rollouts, {"time/sync": sync, "time/rollout": t2 - t1, "time/wake_sleep": (t1 - t0) + (t3 - t2),
                          "rollout_tokens": tokens, "rollout_tok_s": tokens / max(t2 - t1, 1e-9)}

    def run(self, requests: list[Request], temperature: float) -> Batch:
        """Scored samples for `requests` (empty completions are dropped)."""
        with self._exclusive():
            rollouts, stats = self.generate([r.prompt_ids for r in requests],
                                            [r.max_new_tokens for r in requests], temperature)
            start = time.perf_counter()
            samples = []
            for request, rollout in zip(requests, rollouts):
                route = self.routes[request.teacher]
                completion = route.aligner.clean(rollout.completion_ids)
                if completion:
                    samples.append(Sample(request, completion, rollout.logprobs[:len(completion)],
                                          rollout.policy_version,
                                          route.aligner.view(request.teacher_prompt_ids, completion)))
            stats["time/align"] = time.perf_counter() - start
            start = time.perf_counter()
            for name, route in self.routes.items():
                group = [s for s in samples if s.teacher == name]
                if group:
                    scores = route.teacher.score([s.view for s in group], top_k=route.top_k,
                                                 keep_hidden=route.keep_hidden, micro_seqs=self.score_micro_seqs,
                                                 sample_rounds=route.sample_rounds,
                                                 sample_temperature=route.sample_temperature)
                    for s, sc in zip(group, scores):
                        s.scores = sc
            stats["time/teacher"] = time.perf_counter() - start
        stats["stop_rate"] = sum(r.finish_reason == "stop" for r in rollouts) / max(1, len(rollouts))
        version = min((r.policy_version for r in rollouts), default=self.engine.version)
        return Batch(samples, version, stats)


class Orchestrator:
    """Background producer of training batches, at most `max_staleness` versions old.

    `draw()` returns the next batch's requests (called on the producer thread
    only, so sources need no locking).
    """

    def __init__(self, pipeline: Pipeline, draw: Callable[[], list[Request]], temperature: float,
                 max_staleness: int):
        self.pipeline = pipeline
        self.draw = draw
        self.temperature = temperature
        self.max_staleness = max_staleness
        self.queue: queue.Queue = queue.Queue(maxsize=max_staleness + 1)
        self.stop_event = threading.Event()
        self.dropped = 0
        self.thread = threading.Thread(target=self._produce, name="opd-orchestrator", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=600)

    def _produce(self) -> None:
        k = self.pipeline.weights.version       # > 0 when the trainer resumed from a checkpoint
        try:
            while not self.stop_event.is_set():
                requests = self.draw()
                waited = time.perf_counter()
                if not self.pipeline.weights.wait_for(k - self.max_staleness, self.stop_event):
                    return
                waited = time.perf_counter() - waited
                batch = self.pipeline.run(requests, self.temperature)
                batch.stats["time/producer_wait"] = waited
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                k += 1
        except BaseException as e:  # noqa: BLE001 — re-raised on the trainer's thread
            self.queue.put(e)

    def next(self, version: int) -> Batch:
        """The next batch that is at most max_staleness versions behind `version`."""
        while True:
            item = self.queue.get()
            if isinstance(item, BaseException):
                raise RuntimeError("rollout producer failed") from item
            if version - item.version > self.max_staleness:
                self.dropped += len(item.samples)
                logger.warning("dropped a batch %d versions old (max_staleness %d)",
                               version - item.version, self.max_staleness)
                continue
            return item
