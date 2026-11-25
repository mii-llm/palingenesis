"""The single-turn executor of `pgs eval`: every request of a task in the engine at once, scoring next to it.

A task without an environment needs none of an RL episode's machinery, so it runs here:

  1. prompts are rendered once per (task, mode) and tokenized in one batched tokenizer call (render_prompts);
  2. every request is handed to the engine at once, in lexicographic token order, so prompts that share a prefix
     (few-shot examples, a system prompt, tool schemas) follow each other into the engine's prefix cache; the k
     samples of an item are one request with n = k (one prefill, k decodes), a greedy item one request whatever k;
  3. as an item's completions come back they are decoded, parsed (palingenesis.rl.chat.parse_assistant, the parser
     of training rollouts) and scored: in scorer processes (ScorePool) when the scorers are plain functions, so a
     symbolic math check or an instruction checker never takes the interpreter from the engine's thread, else on
     the event loop's threads (LLM judges, sandboxes);
  4. each finished item is one line appended to the result files (ResultWriter), so a run can be watched, stopped
     and resumed; the first line is the time to first result.

GenerationCache keeps what the engine returned (token ids, not scores) under a key of the model's fingerprint,
the prompt's tokens, the sampling, k and the seed: a repeated run of the same checkpoint costs no generation, and
a changed scorer re-scores cached completions. Multiple-choice tasks (output: choices) go through run_choices:
palingenesis.eval.loglikelihood on the same client and cache.

What a sample's record and a scorer's arguments contain is what the RL pipeline gives them (tests compare the two
paths), so a task scores the same here and inside `pgs rl`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from palingenesis.eval.config import TaskConfig
from palingenesis.eval.tasks import EvalItem, prompt_field, prompt_text

logger = logging.getLogger(__name__)

SKIPPED_MARK = "__skipped__"  # rl.rewards.SKIPPED across a process boundary


def dumps(value: Any) -> bytes:
    """One JSON line (orjson when it is installed)."""
    return _DUMPS(value)


def _json_dumps(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, default=str) + "\n").encode()


try:  # an optional accelerator; the output is the same JSON
    import orjson

    def _orjson_dumps(value: Any) -> bytes:
        try:
            return orjson.dumps(value, default=str, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_NON_STR_KEYS)
        except TypeError:
            return _json_dumps(value)

    _DUMPS = _orjson_dumps
except ImportError:
    _DUMPS = _json_dumps


# ------------------------------------------------------------------- requests


@dataclass
class Request:
    item: EvalItem
    prompt: list[int]  # the prompt's token ids
    messages: list[dict]  # the prompt's messages (records and scorers)
    schemas: dict  # the row's tool schemas by function name
    chat: Any  # the ChatFormat of this row (its thinking mode), None for completion-style prompts
    room: int  # tokens the completion may take (<= 0: the prompt leaves none)
    key: str = ""  # of everything that determines the engine's output for this item


def model_fingerprint(model: str) -> str:
    """What identifies a checkpoint's weights: a local directory's config and weight files (names, sizes,
    modification times), a Hub id's cached snapshot (its commit), else the name."""
    path = Path(model)
    digest = hashlib.sha1(usedforsecurity=False)
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model, local_files_only=True, allow_patterns=["*.json"]))
            digest.update(path.name.encode())  # the snapshot's commit hash
        except Exception:  # noqa: BLE001 — not in the cache: the name is all there is
            return hashlib.sha1(model.encode(), usedforsecurity=False).hexdigest()
    for file in sorted(path.glob("*")):
        if file.suffix in (".safetensors", ".bin", ".pt", ".gguf") or file.name == "config.json":
            stat = file.stat()
            digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns if path.is_dir() else 0}".encode())
    return digest.hexdigest()


def request_key(fingerprint: str, prompt: list[int], *parts: Any) -> str:
    digest = hashlib.sha1(fingerprint.encode(), usedforsecurity=False)
    digest.update(repr(parts).encode())
    digest.update(memoryview(_ints(prompt)))
    return digest.hexdigest()


def _ints(values: list[int]):
    import array

    return array.array("q", values)


def render_prompts(
    tokenizer, chat, task: TaskConfig, items: list[EvalItem], config, template: str, tool_field: str = ""
) -> list[Request]:
    """The requests of a task's items: prompts rendered per item, tokenized together.

    chat: the mode's ChatFormat. config: the task's RLConfig (budgets, context length). template: chat (the
    model's chat template, with the row's tools and its own chat_template_kwargs) or completion (the raw text)."""
    from palingenesis.rl.data import prompt_messages, row_limit
    from palingenesis.rl.env import row_tools

    r = config.rollout
    texts, drafts = [], []
    for item in items:
        row = item.row
        if template == "completion":
            text = prompt_text(row, task)
            drafts.append(([{"role": "user", "content": text}], {}, None))
        else:
            messages = prompt_messages(row, prompt_field(task), task.system_prompt)
            schemas = row_tools(row, tool_field or config.data.tools_field)
            own = chat.with_kwargs(row.get("chat_template_kwargs") or {})
            text = tokenizer.apply_chat_template(
                messages, tools=schemas or None, add_generation_prompt=True, tokenize=False, **own.kwargs
            )
            drafts.append((messages, {s["function"]["name"]: s for s in schemas or []}, own))
        texts.append(text)
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
    bos = tokenizer.bos_token_id if template != "completion" else None
    requests = []
    for item, ids, (messages, schemas, own) in zip(items, encoded, drafts):
        ids = list(ids)
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        budget = row_limit(item.row, "max_completion_tokens") or config.completion_budget
        new_tokens = row_limit(item.row, "max_new_tokens") or r.max_new_tokens
        if template == "completion" and len(ids) + min(new_tokens, budget) > r.max_model_len:
            ids = ids[-max(1, r.max_model_len - min(new_tokens, budget)) :]  # as lm-eval: the left is cut
        room = min(new_tokens, budget, r.max_model_len - len(ids))
        requests.append(Request(item, ids, messages, schemas, own, room))
    return requests


# ---------------------------------------------------------------------- cache


class GenerationCache:
    """Engine outputs by request key, in one append-only JSON-lines file per model fingerprint."""

    def __init__(self, directory: str | Path, fingerprint: str):
        self.path = Path(directory) / f"{fingerprint[:20]}.jsonl"
        self.entries: dict[str, Any] = {}
        self.hits = 0
        self._file = None
        if self.path.is_file():
            with self.path.open("rb") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self.entries[entry["key"]] = entry["value"]
                    except (ValueError, KeyError, TypeError):
                        continue  # a line cut by an interrupted run

    def get(self, key: str) -> Any:
        value = self.entries.get(key)
        if value is not None:
            self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self.entries[key] = value
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("ab")
        self._file.write(dumps({"key": key, "value": value}))

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# -------------------------------------------------------------------- scoring


async def score_samples(rewards: list, samples: list[dict]) -> list[dict]:
    """{reward name: value} per sample (None: not applicable; SKIPPED_MARK: could not be scored), plus "_loop"."""
    from palingenesis.rl.rewards import SKIPPED
    from palingenesis.rollouts import loop_flags

    out: list[dict] = [{} for _ in samples]
    for reward in rewards:
        for values, value in zip(out, await reward.score(samples)):
            values[reward.name] = SKIPPED_MARK if value is SKIPPED else value
    for values, sample in zip(out, samples):
        calls = [
            json.dumps(c.get("function"), sort_keys=True, default=str)
            for m in sample["messages"][len(sample["prompt"]) :]
            for c in m.get("tool_calls") or []
        ]
        values["_loop"] = bool(loop_flags([sample["completion"]] if sample["completion"] else [], calls)["loop"])
    return out


_WORKER_REWARDS: dict[str, tuple[TaskConfig, list]] = {}


def _score_job(task: TaskConfig, samples: list[dict]) -> list[dict]:
    """In a scorer process: the task's scorers (resolved once per task) on one item's samples."""
    from palingenesis.eval.scorers import resolve_scorers

    cached = _WORKER_REWARDS.get(task.name)
    if cached is None or cached[0] != task:
        cached = _WORKER_REWARDS[task.name] = (task, resolve_scorers(task))
    return asyncio.run(score_samples(cached[1], samples))


def _worker_init() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # scorers never touch the GPU
    import palingenesis.eval.scorers  # noqa: F401 — imported before the first job


class ScorePool:
    """Scorer processes. Started in the background (spawn: nothing of the parent's CUDA state is inherited), so
    they are ready by the time the engine is."""

    def __init__(self, workers: int):
        self.workers = workers
        self.executor = concurrent.futures.ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
        )
        for _ in range(workers):  # start the processes now
            self.executor.submit(int)

    def submit(self, task: TaskConfig, samples: list[dict]) -> concurrent.futures.Future:
        return self.executor.submit(_score_job, task, samples)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def auto_score_workers(requested: int, samples: int) -> int:
    """run.score_workers resolved: -1 = a few processes for runs large enough to pay for starting them."""
    if requested >= 0:
        return requested
    cpus = os.cpu_count() or 1
    return 0 if samples < 256 or cpus < 4 else min(4, cpus // 2)


def process_scorable(rewards: list, sandbox) -> bool:
    """Whether a task's scorers can run in another process: plain functions, no sandbox, no event loop."""
    return sandbox is None and bool(rewards) and not any(r.is_async for r in rewards)


def combine(values: dict, weights: dict) -> tuple[float | None, dict]:
    """(the sample's score, its reward components) from the scorers' values, as the RL pipeline weighs them: the
    weighted sum of the applicable rewards; None when one of them could not score or none applies."""
    if any(v == SKIPPED_MARK for v in values.values() if isinstance(v, str)):
        return None, {name: (None if v == SKIPPED_MARK else v) for name, v in values.items()}
    applicable = [(weights.get(name, 1.0), v) for name, v in values.items() if v is not None]
    if not applicable:
        return None, dict(values)
    return float(sum(w * v for w, v in applicable)), dict(values)


# --------------------------------------------------------------------- writer


@dataclass
class Progress:
    started: float = field(default_factory=time.perf_counter)
    first_result: float | None = None
    done: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    engine_requests: int = 0
    cache_hits: int = 0


class ResultWriter:
    """Appends finished items to the result and sample files as they arrive; flushes about once a second."""

    def __init__(self, results: Path | None, samples: Path | None, total: int, label: str):
        self.results = results.open("ab", buffering=1 << 20) if results else None
        self.samples = samples.open("ab", buffering=1 << 20) if samples else None
        self.total, self.label = total, label
        self.progress = Progress()
        self._flushed = self._logged = time.perf_counter()

    def write(self, row: dict, samples: list[dict]) -> None:
        now = time.perf_counter()
        progress = self.progress
        if progress.first_result is None:
            progress.first_result = now - progress.started
        progress.done += 1
        if self.results is not None and row["scores"]:  # `pgs compare` needs a score in every row
            self.results.write(dumps(row))
        if self.samples is not None:
            for index, sample in enumerate(samples):
                record = {"id": row["id"], "sample": index, **sample}
                if index:
                    record.pop("prompt", None)
                self.samples.write(dumps(record))
        if now - self._flushed > 1.0:
            self.flush()
            self._flushed = now
        if now - self._logged > 15.0 and progress.done < self.total:
            self._logged = now
            rate = progress.done / (now - progress.started)
            logger.info("eval %s: %d / %d items (%.1f items/s)", self.label, progress.done, self.total, rate)

    def flush(self) -> None:
        for handle in (self.results, self.samples):
            if handle is not None:
                handle.flush()

    def close(self) -> None:
        for handle in (self.results, self.samples):
            if handle is not None:
                handle.close()


# ----------------------------------------------------------------- generation


def _sample(request: Request, generation, tokenizer, stop_ids: set, tool_parser: str, stops, think_close: str):
    """(the sample's record without its score, the scorer's keyword arguments) of one completion."""
    ids = list(generation["ids"])
    text = tokenizer.decode([t for t in ids if t not in stop_ids], skip_special_tokens=True)
    finish = "length" if generation["finish"] == "length" else "stop"
    calls, errors = 0, 0
    if request.chat is None:  # a completion-style prompt: the text is the answer, cut at the task's stop strings
        from palingenesis.eval.lmeval import cut_at_stop

        reasoning = ""
        if think_close and think_close in text:
            reasoning, _, text = text.partition(think_close)
        cut = cut_at_stop(text, stops)
        if len(cut) < len(text):
            finish = "stop"
        content, raw = cut, cut
        message = {"role": "assistant", "content": content}
    else:
        from palingenesis.rl.chat import parse_assistant

        parsed = parse_assistant(text, request.chat, tool_parser, request.schemas, "call_0")
        message = parsed.message()
        content, reasoning, raw = message.get("content") or "", message.get("reasoning_content") or "", parsed.raw_content
        calls, errors = len(parsed.calls), len(parsed.errors)
    record = {
        "score": None,
        "rewards": {},
        "finish": finish,
        "tokens": len(ids),
        "loop": False,
        "turns": 1,
        "tool_calls": calls,
        "tool_errors": errors,
        "completion": content,
        "reasoning": reasoning,
        "prompt": request.messages,
    }
    kwargs = {
        **request.item.row,
        "completion": content,
        "reasoning": reasoning,
        "prompt": request.messages,
        "messages": [*request.messages, message],
        "completion_ids": ids,
        "raw_contents": [raw],
        "finish_reason": finish,
        "env": None,
    }
    return record, kwargs


async def run_generate(
    *,
    client,
    tokenizer,
    task: TaskConfig,
    requests: list[Request],
    sampling,
    k: int,
    rewards: list,
    stop_ids,
    tool_parser: str,
    writer: ResultWriter,
    row_of,
    cache: GenerationCache | None = None,
    pool: ScorePool | None = None,
    sandbox=None,
    think_close: str = "",
    decoders: concurrent.futures.Executor | None = None,
) -> list[dict]:
    """Generate, score and write every request; the item rows, in completion order.

    sampling: what the engine takes (a temperature or an opd.rollout.Sampling). row_of(item, samples) builds an
    item's result row. Greedy sampling is one request per item whatever k (its k samples are the same)."""
    from palingenesis.opd.rollout import samplings

    spec = samplings(sampling, 1)[0]
    greedy = spec.temperature <= 0
    stop_set = set(stop_ids)
    weights = {r.name: r.weight for r in rewards}
    loop = asyncio.get_running_loop()
    progress = writer.progress
    rows: list[dict] = []
    in_process = pool is not None and process_scorable(rewards, sandbox)

    async def one(request: Request) -> None:
        samples: list[dict] = []
        try:
            if request.room <= 0:  # the prompt fills the context: nothing can be generated
                generations = []
            else:
                generations = cache.get(request.key) if cache is not None else None
                if generations is None:
                    n = 1 if greedy else k
                    outs = await asyncio.gather(*(client.generate(request.prompt, request.room, sampling) for _ in range(n)))
                    generations = [{"ids": list(o.ids), "finish": o.finish} for o in outs]
                    progress.engine_requests += 1
                    progress.prompt_tokens += len(request.prompt)
                    progress.generated_tokens += sum(len(g["ids"]) for g in generations)
                    if cache is not None:
                        cache.put(request.key, generations)
                else:
                    progress.cache_hits += 1
                if greedy:
                    generations = generations[:1] * k
            if not generations:
                samples = [
                    {"score": None, "rewards": {}, "finish": "length", "tokens": 0, "loop": False, "turns": 0,
                     "tool_calls": 0, "tool_errors": 0, "completion": "", "reasoning": "", "prompt": request.messages}
                    for _ in range(k)
                ]  # fmt: skip
            else:
                built = await loop.run_in_executor(
                    decoders,
                    lambda: [
                        _sample(request, g, tokenizer, stop_set, tool_parser, spec.stop, think_close)
                        for g in (generations[:1] if greedy else generations)
                    ],
                )
                if greedy:
                    built = built * k
                samples = [record for record, _ in built]
                kwargs = [kw for _, kw in built]
                unique = kwargs[:1] if greedy else kwargs
                if in_process:
                    values = await asyncio.wrap_future(pool.submit(task, unique))
                else:
                    values = await score_samples(rewards, [{**kw, "sandbox": sandbox} for kw in unique])
                if greedy:
                    values = values * k
                samples = [dict(record) for record in samples]
                for record, value in zip(samples, values):
                    value = dict(value)
                    record["loop"] = bool(value.pop("_loop", False))
                    record["score"], record["rewards"] = combine(value, weights)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 — one broken item is recorded, the run goes on
            if progress.done < 5:
                logger.exception("eval %s: item %s failed", task.name, request.item.id)
            samples = [
                {"score": None, "rewards": {}, "finish": "error", "tokens": 0, "loop": False, "turns": 0,
                 "tool_calls": 0, "tool_errors": 0, "completion": "", "reasoning": "", "prompt": request.messages,
                 "error": f"{type(error).__name__}: {error}"}
                for _ in range(k)
            ]  # fmt: skip
        row = row_of(request.item, samples)
        if request.key:
            row["key"] = request.key
        rows.append(row)
        writer.write(row, samples)

    enter, leave = getattr(client, "enter", None), getattr(client, "exit", None)
    order = sorted(requests, key=lambda q: q.prompt)  # shared prefixes next to each other
    if enter is not None:
        for _ in order:
            enter()

    async def tracked(request: Request) -> None:
        try:
            await one(request)
        finally:
            if leave is not None:
                leave()

    await asyncio.gather(*(tracked(request) for request in order))
    if cache is not None:
        cache.flush()
    return rows


# -------------------------------------------------------------------- choices


async def run_choices(
    *,
    client,
    engine,
    tokenizer,
    task: TaskConfig,
    items: list[EvalItem],
    contexts: list[str],
    max_model_len: int,
    top: int,
    writer: ResultWriter,
    mode: str,
    fingerprint: str = "",
    cache: GenerationCache | None = None,
    category_field: str = "",
) -> list[dict]:
    """Loglikelihood multiple choice: every item's choices scored after its context (see eval.loglikelihood)."""
    from palingenesis.eval import loglikelihood as ll
    from palingenesis.eval.tasks import lookup

    progress = writer.progress
    choice_texts = []
    for item in items:
        choices = lookup(item.row, task.choices_field)
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{item.id}: `{task.choices_field}` must be a non-empty list of choices, got {choices!r}")
        choice_texts.append([str(c) for c in choices])
    pairs = ll.encode_pairs(tokenizer, contexts, [[task.target_delimiter + c for c in cs] for cs in choice_texts])
    fallback = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    encoded = []
    for context, choices in pairs:
        context = context or [fallback]
        room = max_model_len - 1 - max(len(c) for c in choices)
        encoded.append((context[-max(1, room) :], choices))
    rows: list[dict] = []

    def finish(item: EvalItem, texts: list[str], scores: ll.ChoiceScores) -> None:
        gold = lookup(item.row, task.gold_field)
        gold = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(gold.strip().upper()) if isinstance(gold, str) else int(gold)
        metrics = ll.choice_metrics(scores, texts, gold)
        row = {
            "id": item.id,
            "scores": [metrics["acc"]],
            "task": task.name,
            "mode": mode,
            "finish": ["stop"],
            "tokens": [0],
            "loop": [False],
            "metrics": {"acc": [metrics["acc"]], "acc_norm": [metrics["acc_norm"]]},
            "pred": metrics["pred"],
            "gold": gold,
            "prob_gold": metrics["prob_gold"],
        }
        if item.cluster is not None:
            row["cluster"] = item.cluster
        if category_field:
            value = lookup(item.row, category_field)
            row["category"] = "" if value is None or not isinstance(value, str | int | float) else str(value)
        sample = {
            "score": metrics["acc"],
            "rewards": {"acc": metrics["acc"], "acc_norm": metrics["acc_norm"]},
            "finish": "stop",
            "tokens": 0,
            "loop": False,
            "completion": texts[metrics["pred"]],
            "loglikelihoods": metrics["loglikelihoods"],
            "greedy": metrics["greedy"],
            "requests": scores.requests,
        }
        rows.append(row)
        writer.write(row, [sample])

    def key_of(context, choices) -> str:
        return request_key(fingerprint, context, "choices", choices) if cache is not None else ""

    model = getattr(engine, "model", None)
    if model is not None and hasattr(model, "parameters"):  # the model is here: padded forwards, no trie
        flat = [(context, choice) for context, choices in encoded for choice in choices]
        pad = getattr(engine, "pad_id", None)
        pad = fallback if pad is None else pad
        scored = await asyncio.get_running_loop().run_in_executor(
            None, lambda: ll.forward_loglikelihoods(model, pad, flat, getattr(engine, "micro_seqs", 8))
        )
        at = 0
        for item, texts, (context, choices) in zip(items, choice_texts, encoded):
            part = scored[at : at + len(choices)]
            at += len(choices)
            progress.prompt_tokens += len(context) + sum(len(c) for c in choices)
            progress.engine_requests += len(choices)
            finish(item, texts, ll.ChoiceScores([p[0] for p in part], [p[1] for p in part], len(choices)))
        return rows

    peek = ll.client_peek(client, top)

    async def one(item: EvalItem, texts: list[str], context: list[int], choices: list[list[int]]) -> None:
        key = key_of(context, choices)
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            progress.cache_hits += 1
            scores = ll.ChoiceScores(cached["loglikelihoods"], cached["greedy"], 0)
        else:
            scores = await ll.trie_loglikelihoods(peek, context, choices, top)
            progress.engine_requests += scores.requests
            progress.prompt_tokens += len(context)
            progress.generated_tokens += scores.requests
            if cache is not None:
                finite = [x if x > -1e30 else -1e30 for x in scores.loglikelihoods]
                cache.put(key, {"loglikelihoods": finite, "greedy": scores.greedy})
        finish(item, texts, scores)

    order = sorted(range(len(items)), key=lambda i: encoded[i][0])
    enter, leave = getattr(client, "enter", None), getattr(client, "exit", None)

    async def tracked(i: int) -> None:
        if enter is not None:
            enter()
        try:
            await one(items[i], choice_texts[i], *encoded[i])
        finally:
            if leave is not None:
                leave()

    await asyncio.gather(*(tracked(i) for i in order))
    if cache is not None:
        cache.flush()
    return rows
