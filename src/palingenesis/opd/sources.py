"""Prompt sources — the task-specific data layer behind the task-agnostic OPD engine.

The trainer knows how to sample on-policy, score with a teacher, and take
distillation steps; everything task-shaped lives behind the PromptSource
protocol:

  - what conversation to roll out next (templates, shot mixing, length budget)
  - how to evaluate the student on held-out prompts
  - which per-batch quality stats to log

Built-ins (``sources.<name>.format`` in the config):

  - "messages": generic chat — JSONL of {"messages": [...]} ending with a user
                turn. Dev metric: the held-out reverse KL to the teacher (you can't
                auto-grade free-form answers, but distance-to-teacher on unseen
                prompts is exactly the trained quantity, measured out of sample).
                Rows with an "answer" field also get greedy dev accuracy: the
                completion's final number ("Answer: 42", else the last number)
                against the answer.
  - "mcqa":     multiple-choice pools — pool-row JSONL, shot regimes, letter
                accuracy as the dev metric

The configured sources are combined by MixedSource, which also records which
source (hence which teacher) each prompt came from. A custom source is any
object with the three methods; pass it as ``OPDTrainer(config, source=...)``.
"""

import json
import logging
import random
from collections.abc import Callable
from typing import Any, Protocol

from palingenesis.opd.config import OPDConfig, SourceConfig
from palingenesis.opd.formatting import (
    PromptRenderer,
    build_messages,
    extract_letter,
    extract_number,
    load_reference_shots,
    looped,
)
from palingenesis.opd.pool import load_pool, question_hash, split_pool
from palingenesis.privileged import privileged_template_error, with_privileged

logger = logging.getLogger(__name__)


class Engine(Protocol):
    """The trainer services a source may use during evaluate()."""

    def greedy_generate(
        self, messages_list: list[list[dict[str, str]]], max_new_tokens: int, metas: list[dict] | None = None
    ) -> list[str]:
        """Greedy-decode one completion per conversation, stop token removed, decoded. `metas`: each
        row's rollout meta (tools, chat_template_kwargs), as sample() returns it."""

    def dev_kl(
        self, messages_list: list[list[dict[str, str]]], max_new_tokens: int, metas: list[dict] | None = None
    ) -> dict[str, float]:
        """Sample on-policy and teacher-score without grad: {"dev_kl": ..., "dev_len": ...}. `metas`: each
        row's rollout meta (tools, chat_template_kwargs, teacher_messages), as sample() returns it."""


class PromptSource(Protocol):
    def sample(self) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
        """One training rollout: (messages, max_new_tokens, meta)."""

    def evaluate(self, engine: Engine) -> dict[str, float]:
        """Held-out metrics, e.g. {"dev_acc": 0.41} or {"dev_kl": 0.83}."""

    def batch_stats(self, rollouts: list[tuple[dict[str, Any], str]]) -> dict[str, float]:
        """Per-batch stats from (meta, decoded_completion) pairs. May be {}."""


def build_source(config: OPDConfig, rng: random.Random) -> "MixedSource":
    """All configured sources, mixed by weight."""
    subs = []
    for name, source in config.sources.items():
        cls = {"mcqa": McqaPoolSource, "messages": ChatMessagesSource}.get(source.format)
        if cls is None:
            raise ValueError(f"sources.{name}.format: unknown format {source.format!r} (expected 'mcqa' or 'messages')")
        subs.append((name, source.weight, cls(source, config.train.eval_samples, config.train.seed, rng)))
    return MixedSource(subs, rng)


class McqaPoolSource:
    """Multiple-choice pools: shot regimes for training, letter accuracy for dev."""

    def __init__(self, config: SourceConfig, eval_samples: int, seed: int, rng: random.Random):
        self.config = config
        self.eval_samples = eval_samples
        logger.info("Loading MCQA pool from %s", config.path)
        if config.dev_path:
            self.train_rows, self.dev_rows = load_pool(config.path), load_pool(config.dev_path)
        else:
            self.train_rows, self.dev_rows = split_pool(load_pool(config.path), config.dev_size, seed)
        if not self.train_rows:
            raise ValueError(f"{config.path}: no training rows left after holding out dev_size={config.dev_size}")
        logger.info("Pool: %d train / %d dev", len(self.train_rows), len(self.dev_rows))
        self.reference_shots = load_reference_shots(config.shots_path) if config.shots_path else []
        self.fast_template = config.fast_template or None
        self.cot_template = config.cot_template or None
        self.renderer = PromptRenderer(
            self.train_rows,
            self.reference_shots,
            p_reference_shots=config.p_reference_shots,
            p_pool_shots=config.p_pool_shots,
            pool_shots_max_k=config.pool_shots_max_k,
            cot_fraction=config.cot_fraction,
            system_message=config.system_message or None,
            fast_template=self.fast_template,
            cot_template=self.cot_template,
            rng=rng,
        )

    def sample(self):
        messages, row, fast = self.renderer.sample()
        mnt = self.config.max_new_tokens if fast else self.config.cot_max_new_tokens
        return messages, mnt, {"row": row, "fast": fast}

    def evaluate(self, engine: Engine) -> dict[str, float]:
        """Greedy few-shot dev accuracy: fast mode always, CoT mode too when trained.

        CoT answers are read as the LAST standalone letter (reasoning text
        contains incidental capitals; models conclude with their answer).
        """
        rows = self.dev_rows[: self.eval_samples]

        def _acc(fast: bool, template, max_new_tokens: int, last: bool) -> float:
            prompts = [
                build_messages(
                    r,
                    few_shots=self.reference_shots,
                    fast=fast,
                    system_message=self.config.system_message or None,
                    template=template,
                )
                for r in rows
            ]
            texts = engine.greedy_generate(prompts, max_new_tokens=max_new_tokens)
            correct = sum(1 for r, t in zip(rows, texts) if extract_letter(t, last=last) == r["answer"])
            return correct / max(1, len(rows))

        metrics = {"dev_acc": _acc(True, self.fast_template, 8, last=False)}
        if self.config.cot_fraction > 0:
            metrics["dev_acc_cot"] = _acc(False, self.cot_template, self.config.cot_max_new_tokens, last=True)
        return metrics

    def batch_stats(self, rollouts):
        if not rollouts:
            return {}
        ok = sum(
            1
            for meta, text in rollouts
            if (letter := extract_letter(text)) and letter in {le for le, _ in meta["row"]["options"]}
        )
        return {"format_ok": ok / len(rollouts)}


class ChatMessagesSource:
    """Generic chat prompts: JSONL of {"messages": [...], "tools"?: [...], "chat_template_kwargs"?: {...},
    "answer"?: ...}, ending with a user turn or a tool result (a mid-episode agent state: the student
    writes the next assistant turn). `tools` (OpenAI schemas) are rendered into the prompt by the chat
    template, for student and teacher alike: function-calling prompts, where the student learns when and
    how to call. A row's `chat_template_kwargs` override the model's for that prompt (e.g. one mix of
    thinking and non-thinking prompts).

    Rows whose last message is neither a user turn nor a tool result are skipped (the student must
    have something to complete).
    """

    def __init__(self, config: SourceConfig, eval_samples: int, seed: int, rng: random.Random):
        self.config = config
        self.eval_samples = eval_samples
        self.rng = rng
        rows = self._load(config.path)
        if config.dev_path:
            self.train_rows, self.dev_rows = rows, self._load(config.dev_path)
        else:
            # deterministic dev split by content hash (same idea as split_pool)
            def key(row):
                return question_hash(json.dumps(row["messages"], ensure_ascii=False))

            by_hash = {key(r): r for r in rows}
            dev_hashes = set(sorted(by_hash)[: config.dev_size])
            self.dev_rows = [by_hash[h] for h in sorted(dev_hashes)]
            self.train_rows = [r for r in rows if key(r) not in dev_hashes]
        if not self.train_rows:
            raise ValueError(f"{config.path}: no training rows left after holding out dev_size={config.dev_size}")
        logger.info("Chat prompts: %d train / %d dev", len(self.train_rows), len(self.dev_rows))
        self.views = config.privileged_view_list()
        self._check_row_templates()
        self._check_rows()

    def _applicable(self, row: dict) -> list[dict]:
        """The privileged views that apply to a row: a view's field and tools_field must be filled in the row, and a
        view whose only change is dropping the tools needs a row that has tools."""
        return [
            v
            for v in self.views
            if (not v["field"] or _present(row.get(v["field"])))
            and (not v["tools_field"] or _present(row.get(v["tools_field"])))
            and (v["template"] or v["tools_field"] or row.get("tools"))
        ]

    @staticmethod
    def _tool_list(value, where: str) -> list[dict]:
        """A tools_field value: a list of tool schemas (dicts), or its JSON string."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: not a JSON list of tool schemas ({e})") from None
        if not isinstance(value, list) or not all(isinstance(t, dict) for t in value):
            raise ValueError(f"{where}: must be a list of tool schemas (objects), got {type(value).__name__}")
        return value

    def _check_rows(self) -> None:
        """Every row's tools_field value of the views that may apply to it is a tool list."""
        for split, rows in (("train", self.train_rows), ("dev", self.dev_rows)):
            for i, row in enumerate(rows):
                for view in self._applicable(row):
                    if view["tools_field"]:
                        self._tool_list(
                            row[view["tools_field"]],
                            f"{self.config.path}: {split} row {i}: {view['tools_field']} (privileged view "
                            f"{view['name']!r})",
                        )

    def _check_row_templates(self) -> None:
        """A row's own template (privileged_template_field) must be valid for every view that may draw it."""
        name = self.config.privileged_template_field
        if not name:
            return
        for split, rows in (("train", self.train_rows), ("dev", self.dev_rows)):
            for i, row in enumerate(rows):
                template = row.get(name)
                if not _present(template):
                    continue
                for view in self._applicable(row):
                    problem = privileged_template_error(template, with_field=bool(view["field"]))
                    if problem:
                        raise ValueError(
                            f"{self.config.path}: {split} row {i}: its {name} (for the privileged view "
                            f"{view['name']!r}) {problem}"
                        )

    @staticmethod
    def _load(path: str) -> list[dict]:
        logger.info("Loading chat prompts from %s", path)
        rows, skipped = [], 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["messages"] and row["messages"][-1]["role"] in ("user", "tool"):
                    rows.append(row)
                else:
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d rows of %s whose last message is not a user turn or tool result", skipped, path)
        if not rows:
            raise ValueError(f"No usable rows in {path}")
        return rows

    def meta(self, row: dict, rng: random.Random | None = None) -> dict:
        """A row's rollout meta: its tools, its template kwargs and, with privileged context, the teacher's copy of
        the conversation (teacher_messages) and the view that wrote it (_view). The view is drawn by weight with
        `rng` (training rollouts), else fixed per row by its content (held-out rows: the same view at every
        evaluation)."""
        meta = {"tools": row["tools"]} if row.get("tools") else {}
        kwargs = {**self.config.chat_template_kwargs, **(row.get("chat_template_kwargs") or {})}
        if kwargs:
            meta["chat_template_kwargs"] = kwargs
        views = self._applicable(row) if self.views else []
        if views:
            if len(views) == 1:
                view = views[0]
            else:
                if rng is None:
                    rng = random.Random(question_hash(json.dumps(row["messages"], ensure_ascii=False, default=str)))
                view = rng.choices(views, weights=[float(v["weight"]) for v in views], k=1)[0]
            own = row.get(self.config.privileged_template_field) if self.config.privileged_template_field else None
            template = own if _present(own) else view["template"]
            if template:
                meta["teacher_messages"] = with_privileged(
                    row["messages"], str(row[view["field"]]) if view["field"] else "", template, view["role"]
                )
            if view["tools_field"]:  # the teacher's own tool list ([]: none)
                meta["teacher_tools"] = self._tool_list(row[view["tools_field"]], view["tools_field"]) or None
            elif view["tools"] == "drop":
                meta["teacher_tools"] = None
            meta["_view"] = view["name"]
        return meta

    def sample(self):
        row = self.rng.choice(self.train_rows)
        return row["messages"], self.config.max_new_tokens, self.meta(row, self.rng)

    def evaluate(self, engine: Engine) -> dict[str, float]:
        rows = self.dev_rows[: self.eval_samples]
        metrics = engine.dev_kl([r["messages"] for r in rows], self.config.max_new_tokens, [self.meta(r) for r in rows])
        deploy = getattr(engine, "deploy_generate", None)
        answers = (
            deploy([r["messages"] for r in rows], self.config.max_new_tokens, [self.meta(r) for r in rows])
            if deploy
            else None
        )
        answers = [a for a in answers or [] if a[1] != "skipped"]  # prompts too long for rollout.max_model_len
        if answers:
            metrics["dev_loop"] = sum(looped(text, finish) for text, finish, _ in answers) / len(answers)
            metrics["dev_len_deploy"] = sum(n for _, _, n in answers) / len(answers)
        graded = [r for r in rows if "answer" in r]
        if graded:
            texts = engine.greedy_generate(
                [r["messages"] for r in graded], self.config.max_new_tokens, [self.meta(r) for r in graded]
            )
            correct = sum(_same_number(extract_number(t), str(r["answer"])) for r, t in zip(graded, texts))
            metrics["dev_acc"] = correct / len(graded)
        return metrics

    def batch_stats(self, rollouts):
        """With privileged views: the share of the batch's rollouts each view wrote the teacher's context for
        (privileged/<view>; privileged/none: rows no view applied to)."""
        if not self.views or not rollouts:
            return {}
        counts: dict[str, int] = {}
        for meta, _ in rollouts:
            name = meta.get("_view", "none")
            counts[name] = counts.get(name, 0) + 1
        return {f"privileged/{name}": n / len(rollouts) for name, n in sorted(counts.items())}


def _present(value) -> bool:
    """A row field that holds something: not missing, not None, not "" or whitespace only, not a NaN (a missing cell
    of a pandas-loaded column). 0 and False are values."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    return not (isinstance(value, str) and not value.strip())


def _same_number(pred: str | None, gold: str) -> bool:
    try:
        return pred is not None and abs(float(pred) - float(gold.replace(",", ""))) < 1e-6
    except ValueError:
        return False


class MixedSource:
    """Compose several PromptSources with sampling weights into ONE OPD run.

    Each draw picks a sub-source by weight and delegates ``sample()``, tagging the
    meta with ``_src`` (the trainer routes the prompt to that source's teacher and
    groups rollouts by their per-sample ``max_new_tokens``, so heterogeneous
    objectives coexist). ``evaluate()`` and ``batch_stats()`` merge each
    sub-source's metrics under ``metric/<name>``, so one on-policy run can lift
    several objectives jointly — avoiding the forgetting that sequential OPD
    stages cause.
    """

    def __init__(self, sources: list[tuple[str, float, Any]], rng: random.Random):
        if not sources:
            raise ValueError("MixedSource needs at least one sub-source")
        names = [n for n, _, _ in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate sub-source names: {names}")
        weights = [w for _, w, _ in sources]
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"weights must be non-negative and sum > 0: {weights}")
        self.rng = rng
        self.names = names
        self.weights = weights
        self.subs = {n: s for n, _, s in sources}
        logger.info("Sources: %s", ", ".join(f"{n}={w / sum(weights):.0%}" for n, w in zip(names, weights)))

    def sample(self) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        messages, mnt, meta = self.subs[name].sample()
        return messages, mnt, {**meta, "_src": name}

    def evaluate(self, engine: Engine | Callable[[str], Engine]) -> dict[str, float]:
        """Metrics of every sub-source, keyed metric/<name>.

        `engine` may be a function of the source name, for per-source services
        (the trainer scores each source against its own teacher)."""
        out: dict[str, float] = {}
        for name, sub in self.subs.items():
            for k, v in sub.evaluate(engine(name) if callable(engine) else engine).items():
                out[f"{k}/{name}"] = v
        return out

    def batch_stats(self, rollouts: list[tuple[dict[str, Any], str]]) -> dict[str, float]:
        by_src: dict[str, list] = {}
        for meta, text in rollouts:
            by_src.setdefault(meta.get("_src"), []).append((meta, text))
        out: dict[str, float] = {}
        for name, sub in self.subs.items():
            if rs := by_src.get(name):
                for k, v in sub.batch_stats(rs).items():
                    out[f"{k}/{name}"] = v
        return out
