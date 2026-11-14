"""Rollout dumps: sampled trajectories of RL and on-policy distillation, written off the training threads.

  RolloutSink         a background writer with a bounded queue. submit() never blocks: when the disk is
                      slower than training the batch is dropped and counted (logging/dump_dropped), so a
                      dump can never stall a rollout or a training step. Nothing runs when dumps are off.
  rl_record           one RL trajectory as a record (run on the writer thread)
  completion_record   one single-turn completion (on-policy distillation)
  read_records        the records of a dump directory, for `pgs rollouts` and scripts

Files. `<output_dir>/rollouts/steps_<a>-<b>.jsonl.gz` holds the dumped steps a..b (SHARD_STEPS of them;
`.rank<r>` before `.jsonl.gz` with several ranks). Each dumped batch is appended as its own gzip member,
so a crash can only cut the last batch short, and the reader keeps what precedes it; a resumed run that
redoes a step appends it again and the reader keeps the newest copy. JSONL because a trajectory is
nested and irregular (tool arguments, environment extras) and stays readable with `zcat | jq` and by an
agent; gzip because it is in the standard library and compresses with the GIL released. (Parquet would
need a schema for those irregular fields and buys nothing for a reader that scans whole steps.)

Record (schema_version 1). Where OpenTelemetry's GenAI semantic conventions name the same thing the
field carries that name (gen_ai.*), so an exporter is a mapping; everything else is the trainer's own.

  schema_version, kind ("trajectory" | "timeline"), trainer ("rl" | "opd"), step, rank, trajectory_id
  finish                         stop | length | turns | env_done | error
  gen_ai.usage.input_tokens      prompt tokens
  gen_ai.usage.output_tokens     sampled tokens
  logprob_mean                   the sampler's mean log-prob of the sampled tokens
  loop, repeated_tool_calls      a repetition loop in the text, or one call (same arguments) made 3+ times
  prompt                         the input messages
  turns                          one record per assistant turn:
      gen_ai.operation.name ("chat"), role, content, reasoning_content,
      gen_ai.usage.input_tokens / output_tokens, gen_ai.response.finish_reasons, duration_s, logprob_mean,
      tools_s, observations (results that answer no call: a malformed call's error, a text-protocol reply),
      tool_calls: gen_ai.operation.name ("execute_tool"), gen_ai.tool.name, gen_ai.tool.call.id,
                  gen_ai.tool.call.arguments, gen_ai.tool.call.result, error, duration_s
  RL only: group, kept (False: a zero-variance group), scored, trained, reward, rewards (components),
           advantage, turn_count, tool_calls, tool_errors, context_tokens, version, start_s, duration_s, info
  OPD only: group, source, teacher, version, teacher_logprob_mean, k1_mean (sampled reverse KL per token),
            meta (agent traces: turn, onset, error, weight)
  kind "timeline" (RL): events, the engine's (time, sequences in flight, finished) per engine step

`logging.dump_content: false` leaves the text out (content, arguments, results, prompt) and keeps the rest.
"""

import atexit
import contextlib
import gzip
import json
import logging
import os
import queue
import re
import signal
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SHARD_STEPS = 100  # dumped steps per file
DROPPED_METRIC = "logging/dump_dropped"  # records dropped because the writer was behind (cumulative)


class RolloutSink:
    """Appends each submitted batch to its step range's file, on one background thread.

    `encode` turns a submitted item into a record on the writer thread (None: the items are records
    already). The queue holds batches, not records: `queue_size` batches wait at most, each one a list
    of references, so what a slow disk can hold back is bounded."""

    def __init__(
        self,
        directory: str | Path,
        trainer: str,
        encode: Callable[[Any], dict] | None = None,
        rank: int = 0,
        world: int = 1,
        queue_size: int = 4,
        shard_steps: int = SHARD_STEPS,
    ):
        self.directory = Path(directory)
        self.trainer, self.encode, self.rank, self.shard_steps = trainer, encode, rank, shard_steps
        self.suffix = f".rank{rank}" if world > 1 else ""
        self.dropped = 0  # records dropped: the queue was full, or their batch could not be written
        self.written = 0
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()  # re-entrant: close() may run in a signal handler during submit()
        self._closed = False
        self._warned = 0

    def submit(self, step: int, items: list, timeline: list | None = None) -> bool:
        """Queue a batch for `step` (and the engine's concurrency timeline); False when it was
        dropped. Never blocks."""
        if not items and timeline is None:
            return True
        with self._lock:
            if self._closed:
                return False
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="pgs-rollout-dump", daemon=True)
                self._thread.start()
                atexit.register(self.close)
            try:
                self._queue.put_nowait((step, items, timeline))
            except queue.Full:
                self.dropped += len(items)
                return False
            return True

    def close(self, timeout: float = 30.0) -> None:
        """Write what is queued and stop the writer (idempotent). Gives up after `timeout` seconds."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        atexit.unregister(self.close)
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("rollout dump: the writer is stuck; %d batch(es) left unwritten", self._queue.qsize())
            return
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("rollout dump: the writer did not finish within %g s", timeout)

    def path(self, step: int) -> Path:
        start = step - step % self.shard_steps
        return self.directory / f"steps_{start:08d}-{start + self.shard_steps - 1:08d}{self.suffix}.jsonl.gz"

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            step, items, timeline = job
            try:
                self._write(step, items, timeline)
                with self._lock:
                    self.written += len(items)
            except Exception as e:  # noqa: BLE001 — a dump must never stop training
                with self._lock:
                    self.dropped += len(items)
                    self._warned += 1
                if self._warned <= 3:
                    logger.warning("rollout dump of step %d failed (continuing training): %s", step, e)

    def _write(self, step: int, items: list, timeline: list | None) -> None:
        head = {"schema_version": SCHEMA_VERSION, "trainer": self.trainer, "step": step, "rank": self.rank}
        lines = []  # encoded first: a batch that fails to encode leaves the file untouched
        for i, item in enumerate(items):
            record = {**head, "kind": "trajectory", "trajectory_id": f"{step}-{self.rank}-{i}"}
            record.update(self.encode(item) if self.encode is not None else item)
            lines.append(json.dumps(record, ensure_ascii=False, default=str))
        if timeline is not None:
            lines.append(json.dumps({**head, "kind": "timeline", "events": timeline}))
        self.directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.path(step), "at", compresslevel=3, encoding="utf-8") as f:  # one gzip member per batch
            f.write("\n".join(lines) + "\n")


@contextlib.contextmanager
def flush_on_signal(sink: RolloutSink | None, signals: tuple[int, ...] = (signal.SIGTERM,)):
    """While active, a signal that would end the process writes the queued dumps first, then takes its
    previous course (the handler that was installed, or the default: termination). For trainers that do
    not handle the signal themselves; a no-op without a sink or off the main thread."""
    if sink is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def handler(signum, frame):
        sink.close(timeout=10.0)
        signal.signal(signum, previous[signum])
        os.kill(os.getpid(), signum)

    for signum in signals:
        previous[signum] = signal.signal(signum, handler) or signal.SIG_DFL
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def pick(items: list, count: int, step: int) -> list:
    """`count` of `items` (all of them when count is 0 or covers them), drawn with the step as the seed:
    the first ones to finish are the fast ones, so a prefix would be a biased sample."""
    if count <= 0 or count >= len(items):
        return items
    import random

    return random.Random(step).sample(items, count)


# ----------------------------------------------------------------- records


def loop_flags(texts: list[str], calls: list[str]) -> dict[str, Any]:
    """`texts`: the assistant turns' text; `calls`: every tool call as one string (name and arguments)."""
    from palingenesis.opd.formatting import looped

    repeated = max(Counter(calls).values(), default=0)
    return {"loop": repeated >= 3 or any(looped(text, "") for text in texts if text), "repeated_tool_calls": repeated}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def chat_turns(messages: list[dict], timings: list[dict], logprobs: list[float | None], content: bool) -> list[dict]:
    """Per-turn records of the messages after the prompt: an assistant message opens a turn, and the
    messages up to the next one are its tool results (matched to their calls by id) and observations.
    A turn's timing lists its tool calls in order (name, seconds, failed); text-protocol calls are
    there only, since the environment parsed them from the text."""
    turns: list[dict] = []
    for message in messages:
        if message.get("role") == "assistant":
            i = len(turns)
            timing = timings[i] if i < len(timings) else {}
            turn: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "role": "assistant",
                "gen_ai.usage.input_tokens": timing.get("context"),
                "gen_ai.usage.output_tokens": timing.get("tokens"),
                "gen_ai.response.finish_reasons": [timing["finish"]] if timing.get("finish") else [],
                "duration_s": timing.get("generate_s"),
                "logprob_mean": logprobs[i] if i < len(logprobs) else None,
            }
            if content:
                turn["content"] = message.get("content")
                if message.get("reasoning_content"):
                    turn["reasoning_content"] = message["reasoning_content"]
            tools = timing.get("tools") or []
            listed = message.get("tool_calls") or []
            calls = []
            for k in range(max(len(listed), len(tools))):
                function = (listed[k].get("function") or {}) if k < len(listed) else {}
                tool = tools[k] if k < len(tools) else {}
                record: dict[str, Any] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": function.get("name") or tool.get("name"),
                    "gen_ai.tool.call.id": listed[k].get("id") if k < len(listed) else None,
                }
                if content and function:
                    record["gen_ai.tool.call.arguments"] = function.get("arguments")
                if tool:
                    record["error"], record["duration_s"] = bool(tool.get("failed")), tool.get("s")
                calls.append(record)
            if calls:
                turn["tool_calls"] = calls
            if "tools_s" in timing:
                turn["tools_s"] = timing["tools_s"]
            turns.append(turn)
        elif turns:
            turn = turns[-1]
            call_id = message.get("tool_call_id")
            call = next(
                (c for c in turn.get("tool_calls", ()) if call_id and c["gen_ai.tool.call.id"] == call_id), None
            )
            if call is None:
                observation = {"role": message.get("role")}
                if content:
                    observation["content"] = message.get("content")
                turn.setdefault("observations", []).append(observation)
            elif content:
                call["gen_ai.tool.call.result"] = message.get("content")
    return turns


def _turn_logprobs(trajectory) -> list[float | None]:
    """The sampler's mean log-prob of each assistant turn's tokens (None for an empty turn)."""
    out: list[float | None] = []
    for part in (*trajectory.segments, trajectory):  # context rewrites close segments; turns keep their order
        ends = part.turn_starts[1:] + [len(part.tokens)]
        for start, end in zip(part.turn_starts, ends):
            out.append(_mean([lp for lp, sampled in zip(part.logprobs[start:end], part.mask[start:end]) if sampled]))
    return out


def rl_record(item: tuple[bool, Any], content: bool = True) -> dict:
    """An RL trajectory (palingenesis.rl.trajectory.Trajectory) and whether its group was kept."""
    kept, t = item
    info = t.info
    timing = info.get("timing") or {}
    prompt = info.get("prompt")
    if prompt is None:  # the rollout failed before its prompt was rendered
        first = next((i for i, m in enumerate(t.messages) if m.get("role") == "assistant"), len(t.messages))
        prompt = t.messages[:first]
    turns = chat_turns(t.messages[len(prompt) :], timing.get("turns") or [], _turn_logprobs(t), content)
    if content:
        for turn, raw in zip(turns, info.get("raw_contents") or []):
            if raw != turn.get("content"):  # as sampled, before the parser's strip
                turn["raw_content"] = raw
    answers = [m for m in t.messages[len(prompt) :] if m.get("role") == "assistant"]
    calls = [
        json.dumps(c.get("function"), sort_keys=True, default=str) for m in answers for c in m.get("tool_calls") or []
    ]
    sampled = [lp for part in (*t.segments, t) for lp, mask in zip(part.logprobs, part.mask) if mask]
    extras = {k: v for k, v in info.items() if k not in ("timing", "prompt", "raw_contents") and not k.startswith("_")}
    if not content:
        extras = {k: v for k, v in extras.items() if isinstance(v, bool | int | float)}
    record = {
        "group": t.group,
        "kept": kept,
        "scored": t.scored,
        "trained": t.trained,
        "finish": t.finish,
        "reward": t.reward,
        "rewards": t.rewards,
        "advantage": t.advantage,
        "turn_count": t.turns,
        "tool_calls": t.tool_calls,
        "tool_errors": t.tool_errors,
        "gen_ai.usage.input_tokens": len(t.prompt_ids),
        "gen_ai.usage.output_tokens": t.sampled_tokens,
        "context_tokens": len(t.tokens) - sum(t.mask),
        "version": t.version,
        "start_s": timing.get("start"),
        "duration_s": timing["end"] - timing["start"] if "end" in timing and "start" in timing else None,
        "logprob_mean": _mean(sampled),
        **loop_flags([m.get("content") or "" for m in answers if isinstance(m.get("content"), str)], calls),
        "turns": turns,
        "info": extras,
    }
    if content:
        record["prompt"] = prompt
    return record


def completion_record(
    text: str,
    finish: str,
    prompt_tokens: int,
    logprobs: list[float],
    prompt: list[dict] | None = None,
    content: bool = True,
    **fields: Any,
) -> dict:
    """A single-turn completion (on-policy distillation): `fields` are the trainer's own columns."""
    turn: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "role": "assistant",
        "gen_ai.usage.input_tokens": prompt_tokens,
        "gen_ai.usage.output_tokens": len(logprobs),
        "gen_ai.response.finish_reasons": [finish] if finish else [],
        "logprob_mean": _mean(logprobs),
    }
    record = {
        **fields,
        "finish": finish,
        "gen_ai.usage.input_tokens": prompt_tokens,
        "gen_ai.usage.output_tokens": len(logprobs),
        "logprob_mean": turn["logprob_mean"],
        **loop_flags([text], []),
        "turns": [turn],
    }
    if content:
        turn["content"] = text
        if prompt is not None:
            record["prompt"] = prompt
    return record


# ------------------------------------------------------------------ reading

_SHARD = re.compile(r"steps_(\d+)-(\d+)(?:\.rank\d+)?\.jsonl\.gz$")
_LEGACY = re.compile(r"step_(\d+)\.jsonl$")  # one plain file per step, written before the schema was versioned


def dump_files(directory: str | Path) -> list[tuple[int, int, Path]]:
    """(first step, last step, file) of every dump file in `directory`, in step order."""
    found = []
    for path in Path(directory).glob("step*.jsonl*"):
        if m := _SHARD.match(path.name):
            found.append((int(m.group(1)), int(m.group(2)), path))
        elif m := _LEGACY.match(path.name):
            found.append((int(m.group(1)), int(m.group(1)), path))
    return sorted(found)


def _lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            yield from f
    except (OSError, EOFError) as e:  # a batch cut short by a crash: what precedes it counts
        logger.warning("%s: %s", path, e)


def read_records(
    directory: str | Path, steps: Iterable[int] | None = None, kind: str | None = "trajectory"
) -> Iterator[dict]:
    """The records of a dump directory, in file order (`steps`: only those; `kind`: only records of
    that kind, None for all). Lines that do not parse are skipped. A step written twice to a file (a
    resumed run redid it) keeps its newest copy; records of the unversioned format get "step" from
    their file's name and kind "trajectory"."""
    wanted = None if steps is None else set(steps)
    for first, last, path in dump_files(directory):
        if wanted is not None and not any(first <= s <= last for s in wanted):
            continue
        batches: dict[tuple, list[dict]] = {}  # (step, rank) -> its newest batch
        current = None
        for line in _lines(path):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            record.setdefault("step", first)
            record.setdefault("kind", "trajectory")
            key = (record["step"], record.get("rank", 0))
            if key != current:  # a new batch starts: a key seen before is a redone step
                batches[key] = []
                current = key
            batches[key].append(record)
        for (step, _), records in batches.items():
            if wanted is not None and step not in wanted:
                continue
            for record in records:
                if kind is None or record["kind"] == kind:
                    yield record
