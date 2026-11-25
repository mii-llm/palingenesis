"""The evaluation configuration: tasks, modes, sampling, the run. YAML and the standard library only.

A task (one YAML file, or an entry of `tasks:`):

    name: gsm8k
    dataset: openai/gsm8k            # a Hugging Face dataset id, or a local .jsonl / .json / .parquet file
    subset: main                     # the dataset's configuration name (Hub datasets)
    split: test
    fields:                          # columns the prompt and the scorer read, from the dataset's columns
      question: question
      answer: {from: answer, regex: '####\\s*(.+)$', replace: {',': ''}}
    prompt: |                        # {{field}} placeholders; other braces (LaTeX) are left alone
      {{question}}
    scorer: numeric                  # a built-in scorer, an RL reward (math, code, ...) or module:function
    scorer_args: {field: answer}
    sampling: {max_tokens: 1024}

A run (pgs eval --config eval.yaml):

    model: {path: runs/sft/final}
    run: {k: 4, modes: [plain, think], output_dir: eval/sft}
    sampling: {temperature: 0.7, top_p: 0.8, top_k: 20}
    tasks: [gsm8k, my_task.yaml, {task: mmlu_pro, limit: 500}]

Unknown options, wrong types and impossible combinations raise EvalConfigError naming the file and the option.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class EvalConfigError(ValueError):
    """An evaluation config that cannot run as written."""


# Thinking modes by name: chat template arguments over model.chat_template_kwargs. "default" leaves the template alone.
MODES: dict[str, dict] = {
    "default": {},
    "plain": {"enable_thinking": False},
    "think": {"enable_thinking": True},
}
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
# What a `fields:` entry may say: one source (from / template / value) and transforms, applied in this order.
FIELD_SOURCES = ("from", "template", "value", "columns", "index_of")
FIELD_TRANSFORMS = (
    "json", "regex", "group", "replace", "strip", "map", "choices", "letter", "tools", "default", "shuffle", "in",
)  # fmt: skip
# rollout.* options an `engine:` section may not set: they are the task's sampling, not the engine's.
ENGINE_RESERVED = {
    "temperature": "sampling.temperature",
    "top_p": "sampling.top_p",
    "top_k": "sampling.top_k",
    "max_new_tokens": "sampling.max_tokens",
    "max_completion_tokens": "sampling.max_completion_tokens",
}
ENGINE_DEFAULTS = {"backend": "vllm", "gpu_memory_utilization": 0.9, "max_model_len": 8192, "prefix_caching": True}
# `engine:` options of the evaluation itself (palingenesis.eval.engines), not rollout.* options:
#   replicas               engines per GPU (data parallel on one GPU: small models leave it idle; 0 = 1)
#   devices                GPU indices, one engine each (default: every visible GPU when there are several)
#   tensor_parallel_size   one engine over this many GPUs (a model too big for one)
#   speculative_config     vLLM's speculative decoding configuration, passed through
#   max_logprobs           next-token candidates a loglikelihood request may read at once (default 32)
EVAL_ENGINE_KEYS = ("replicas", "devices", "tensor_parallel_size", "speculative_config", "max_logprobs")
OUTPUTS = ("generate", "choices")
TEMPLATES = ("chat", "completion")
BACKENDS = ("vllm", "sglang", "hf")
DEFAULT_MAX_TOKENS = 2048


@dataclass(slots=True)
class SamplingConfig:
    """How completions are sampled. Unset options fall through: a mode's over the task's over the run's. A
    temperature left unset everywhere is the model's own generation_config.json (what a server uses by default);
    without one, greedy at k = 1 and 1.0 at k > 1."""

    temperature: float | None = None  # 0 = greedy
    top_p: float | None = None  # 1 = off
    top_k: int | None = None  # 0 = off
    presence_penalty: float | None = None  # 0 = off
    max_tokens: int | None = None  # generated tokens per assistant turn, thinking included (default 2048)
    max_completion_tokens: int | None = None  # of a whole episode (agentic tasks; default max_tokens x env.max_turns)
    stop: list | None = None  # stop strings: the completion is cut at the first one (completion-style prompts)


@dataclass(slots=True)
class TaskConfig:
    name: str = ""
    description: str = ""
    # Data: a Hub dataset id (with subset / split / revision), or a local .jsonl / .json / .parquet file.
    dataset: str = ""
    subset: str = ""
    split: str = "test"
    revision: str = ""
    # One file of a Hub dataset repository (JSON lines, JSON or parquet) instead of the dataset's loader.
    data_files: str = ""
    # Columns from a second file of the same source, matched by a key: {data_files: answers.json, key: id}.
    join: dict = field(default_factory=dict)
    # At most this many items (0 = all): a fixed-seed sample (sample_seed) kept in file order, the same for every
    # checkpoint, so results stay paired.
    limit: int = 0
    sample_seed: int = 0
    # name: column | {from: column, regex: ..., replace: {...}, choices: true, letter: true, ...} (see the guide)
    fields: dict = field(default_factory=dict)
    id_field: str = ""  # the item's id column (default: its row number); ids are "<task>/<value>"
    cluster_field: str = ""  # items that are not independent (one passage's questions) resample together
    # The prompt: one user turn (`prompt`), a chat (`messages`: [{role, content}]) or a column (`prompt_field`).
    prompt: str = ""
    messages: list = field(default_factory=list)
    prompt_field: str = ""
    system_prompt: str = ""
    # The score: one scorer, or weighted RL rewards as in an RL config ({name: {fn, weight, args}}); an environment
    # with get_reward scores too.
    scorer: str = ""
    scorer_args: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=dict)
    threshold: float = 0.5  # a sample passes at or above this score (pass@k)
    # An LLM judge is run on labelled examples first: {file, min_agreement, label_field, ...} (see the guide).
    judge_validation: dict = field(default_factory=dict)
    k: int = 0  # samples per item (0 = the run's)
    modes: list = field(default_factory=list)  # thinking modes (empty = the run's)
    sampling: dict = field(default_factory=dict)  # SamplingConfig options
    # Agentic tasks: the RL config's env: and sandbox: sections (type, args, max_turns, tools, ...).
    env: dict = field(default_factory=dict)
    sandbox: dict = field(default_factory=dict)
    # generate: completions are sampled and scored. choices: loglikelihood multiple choice; no text is generated,
    # each row's `choices` (continuations) are scored after its prompt and the most likely one is the answer.
    output: str = "generate"
    choices_field: str = "choices"  # output choices: the row's list of continuation strings
    gold_field: str = "gold"  # output choices: the index of the correct one
    target_delimiter: str = " "  # output choices: between the prompt and a choice
    # chat: the prompt goes through the model's chat template. completion: it is the raw text a base model
    # continues (few-shot prompts of lm-evaluation-harness tasks). Empty: the run's (run.template, default chat).
    template: str = ""
    version: str = ""  # of the task definition: part of the result cache key and of the report
    category_field: str = ""  # a column the report breaks scores down by (subject, difficulty, language)
    subsets: list = field(default_factory=list)  # several dataset configurations as one task (column `subset`)
    # A task written for lm-evaluation-harness (palingenesis.eval.lmeval): its parsed YAML. Set by load_task.
    lmeval: dict = field(default_factory=dict)


@dataclass(slots=True)
class ModelSection:
    path: str = ""  # a Hugging Face id or a local checkpoint
    stop_tokens: list = field(default_factory=list)  # tokens that end a turn besides the template's and eos
    chat_template_kwargs: dict = field(default_factory=dict)  # under every mode's own


@dataclass(slots=True)
class RunSection:
    k: int = 1  # samples per item
    modes: list = field(default_factory=lambda: ["default"])  # each mode is a separate result
    limit: int = 0  # items per task (0 = the task's own limit)
    head: bool = False  # a limit keeps the first items (lm-eval's --limit) instead of a fixed-seed sample
    chunk: int = 512  # agentic tasks: episodes in flight at once (single-turn tasks are submitted all at once)
    template: str = "chat"  # chat | completion: tasks that do not set their own
    num_fewshot: int = -1  # lm-evaluation-harness tasks: few-shot examples (-1 = the task's own)
    # Scorer processes (verifiers run next to generation, off the engine's thread): -1 = automatic, 0 = threads.
    score_workers: int = -1
    cache_dir: str = ""  # the generation cache ("" = off; the command line defaults to ~/.cache/palingenesis/eval)
    contamination: list = field(default_factory=list)  # training data files: n-gram overlap with the prompts
    contamination_ngram: int = 13
    output_dir: str = "eval_results"
    save_samples: bool = True  # <task>.<mode>.samples.jsonl: every completion with its score
    resume: bool = False  # keep the items already in the result files
    resamples: int = 10000  # bootstrap resamples
    confidence: float = 0.95
    seed: int = 0  # the engine's and the bootstrap's


@dataclass(slots=True)
class EvalTasksConfig:
    """The `eval_tasks:` section of a training config (pgs train / rl / distill): evaluation tasks run on the
    in-training weights every `every` steps, with the trainer's rollout engine where it has one."""

    tasks: list = field(default_factory=list)  # built-in names, task YAML paths or inline tasks; empty = off
    every: int = 0  # steps between runs (0 = the trainer's eval_every)
    k: int = 1
    modes: list = field(default_factory=list)  # empty: the trainer's own chat template arguments
    limit: int = 200  # items per task (0 = all of them)
    sampling: dict = field(default_factory=dict)
    save_samples: bool = False
    final: bool = True  # also at the last step

    @property
    def enabled(self) -> bool:
        return bool(self.tasks)

    def errors(self) -> list[str]:
        """What is wrong with the section (no data is loaded)."""
        if not self.tasks:
            return []
        out = []
        if self.every < 0 or self.k < 1 or self.limit < 0:
            out.append("eval_tasks: every and limit must be >= 0 and k >= 1")
        try:
            _apply(SamplingConfig(), "eval_tasks.sampling", self.sampling, "eval_tasks")
            modes = resolve_modes(self.modes or ["default"], {}, "eval_tasks.modes")
            for ref in self.tasks:
                task = load_task(ref)
                task_errors(task, set(modes))
        except EvalConfigError as e:
            out.append(str(e))
        return out


@dataclass(slots=True)
class EvalConfig:
    model: ModelSection = field(default_factory=ModelSection)
    # The rollout engine: the RL config's rollout.* engine options (backend vllm | sglang | hf, max_model_len,
    # gpu_memory_utilization, max_num_seqs, vllm_args, ...).
    engine: dict = field(default_factory=dict)
    run: RunSection = field(default_factory=RunSection)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    # Modes beyond default / plain / think: name -> {chat_template_kwargs: {...}, sampling: {...}}.
    modes: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)  # TaskConfig objects once loaded

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw = _read_mapping(path)
        config = cls()
        base = Path(path).parent
        for section, options in raw.items():
            where = str(path)
            if section == "tasks":
                if not isinstance(options, list):
                    raise EvalConfigError(f"{where}: `tasks:` must be a list of task names, files or mappings.")
                config.tasks = [load_task(ref, base) for ref in options]
            elif section in ("engine", "modes"):
                if not isinstance(options, dict | type(None)):
                    raise EvalConfigError(f"{where}: `{section}:` must be a mapping.")
                setattr(config, section, dict(options or {}))
            elif section in ("model", "run", "sampling"):
                if not isinstance(options, dict | type(None)):
                    raise EvalConfigError(f"{where}: `{section}:` must be a mapping of options.")
                _apply(getattr(config, section), section, options or {}, where)
            else:
                names = [f.name for f in dataclasses.fields(cls)]
                raise EvalConfigError(f"{where}: unknown section `{section}`{_hint(section, names)}")
        return config

    def set(self, key: str, value: Any, where: str = "command line") -> None:
        """One option by dotted key: model.path, run.k, sampling.temperature, engine.<rollout option>."""
        section, _, option = key.partition(".")
        if section == "engine" and option:
            self.engine[option] = yaml.safe_load(value) if isinstance(value, str) else value
        elif section in ("model", "run", "sampling") and option and "." not in option:
            _apply(getattr(self, section), section, {option: value}, where)
        else:
            raise EvalConfigError(
                f"{where}: {key}: use model.<option>, run.<option>, sampling.<option> or engine.<option>."
            )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # --------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Raise EvalConfigError on what cannot run; return warnings. No data and no model are loaded."""
        errors: list[str] = []
        warnings: list[str] = []
        r = self.run
        if not self.model.path:
            errors.append("model.path is empty: give the checkpoint to evaluate (--model).")
        if not self.tasks:
            errors.append(f"no tasks: give --tasks or `tasks:` (built-in: {', '.join(builtin_tasks())}).")
        if r.k < 1 or r.chunk < 1 or r.limit < 0 or r.resamples < 1 or not 0 < r.confidence < 1:
            errors.append("run: k, chunk and resamples must be >= 1, limit >= 0 and confidence in (0, 1).")
        errors += sampling_errors(self.sampling, "sampling")
        backend = self.engine.get("backend", ENGINE_DEFAULTS["backend"])
        if backend not in BACKENDS:
            errors.append(f"engine.backend must be one of {', '.join(BACKENDS)}, got {backend!r}.")
        if r.template not in TEMPLATES:
            errors.append(f"run.template must be one of {', '.join(TEMPLATES)}, got {r.template!r}.")
        for key in ("replicas", "tensor_parallel_size", "max_logprobs"):
            value = self.engine.get(key, 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"engine.{key} must be an integer >= 0, got {value!r}.")
        if self.engine.get("replicas", 1) > 1 and self.engine.get("tensor_parallel_size", 1) > 1:
            errors.append("engine: replicas (several engines) and tensor_parallel_size (one engine) exclude each other.")
        for key in self.engine:
            if key in ENGINE_RESERVED:
                errors.append(f"engine.{key}: set {ENGINE_RESERVED[key]} instead (sampling is per task and mode).")
        modes: dict[str, dict] = {}
        try:
            modes = resolve_modes(_all_modes(self), self.modes, "modes")
            resolve_modes(r.modes, self.modes, "run.modes")
        except EvalConfigError as e:
            errors.append(str(e))
        if not r.modes:
            errors.append("run.modes is empty: give at least one of " + ", ".join(sorted({**MODES, **self.modes})))
        names = [t.name for t in self.tasks]
        for name in sorted({n for n in names if names.count(n) > 1}):
            errors.append(f"task {name!r} appears twice: give one of them another `name`.")
        for task in self.tasks:
            try:
                warnings += task_errors(task, set(modes) | set(MODES))
            except EvalConfigError as e:
                errors.append(str(e))
            k = task.k or r.k
            layers = [self.sampling, _sampling(task.sampling, f"task {task.name}: sampling")]
            temperature = next((s.temperature for s in reversed(layers) if s.temperature is not None), None)
            if k > 1 and temperature == 0:
                warnings.append(
                    f"task {task.name}: k = {k} at temperature 0 draws the same greedy sample {k} times (pass@k "
                    "equals pass@1); set sampling.temperature."
                )
        if errors:
            raise EvalConfigError("\n".join(errors))
        return warnings


# ---------------------------------------------------------------------- tasks


def task_dirs() -> list[Path]:
    """Where built-in tasks live: inside an installed package, and configs/eval of a checkout."""
    here = Path(__file__).resolve()
    return [d for d in (here.parent / "task_pack", here.parents[3] / "configs" / "eval") if d.is_dir()]


def builtin_tasks() -> dict[str, Path]:
    """{name: YAML file} of the built-in task pack."""
    found: dict[str, Path] = {}
    for directory in task_dirs():
        for path in sorted(directory.glob("*.yaml")):
            found.setdefault(path.stem, path)
    return dict(sorted(found.items()))


def load_task(ref: Any, base: Path | None = None) -> TaskConfig:
    """A task from a built-in name, a YAML file, a mapping ({name, dataset, ...}) or a mapping over another task
    ({task: gsm8k, limit: 100, ...}). A TaskConfig is returned as it is."""
    if isinstance(ref, TaskConfig):
        return ref
    if isinstance(ref, str):
        path = _task_path(ref, base)
        from palingenesis.eval import lmeval

        if lmeval.is_lmeval_task(path):  # a task of lm-evaluation-harness's zoo, as it is
            return _named(lmeval.load(path), str(path))
        task = TaskConfig(name=path.stem)
        _apply(task, "task", _read_mapping(path), str(path))
        return _named(task, str(path))
    if isinstance(ref, dict):
        options = dict(ref)
        if "task" in options:
            task = load_task(options.pop("task"), base)
            where = f"task {task.name}"
        else:
            task, where = TaskConfig(), f"task {options.get('name') or '(unnamed)'}"
        _apply(task, "task", options, where)
        return _named(task, where)
    raise EvalConfigError(f"a task is a name, a YAML file or a mapping, got {ref!r}")


def builtin_suites() -> dict[str, Path]:
    """{name: YAML file} of the built-in suites (configs/eval/suites): named lists of tasks."""
    found: dict[str, Path] = {}
    for directory in task_dirs():
        for path in sorted((directory / "suites").glob("*.yaml")):
            found.setdefault(path.stem, path)
    return dict(sorted(found.items()))


def load_suite(ref: str) -> list[TaskConfig]:
    """The tasks of a suite: a built-in name (core, quick, multilingual, ...) or a YAML file with `tasks:`."""
    suites = builtin_suites()
    path = Path(ref) if ref.endswith((".yaml", ".yml")) else suites.get(ref)
    if path is None or not path.is_file():
        raise EvalConfigError(f"unknown suite {ref!r}{_hint(ref, list(suites))}; suites: {', '.join(suites) or 'none'}")
    raw = _read_mapping(path)
    if not isinstance(raw.get("tasks"), list) or not raw["tasks"]:
        raise EvalConfigError(f"{path}: a suite is a mapping with a `tasks:` list.")
    return [load_task(entry, path.parent) for entry in raw["tasks"]]


def _task_path(ref: str, base: Path | None) -> Path:
    if ref.endswith((".yaml", ".yml")) or "/" in ref:
        for candidate in (Path(ref), *([base / ref] if base else [])):
            if candidate.is_dir() and (candidate / f"{candidate.name}.yaml").is_file():  # lm-eval: tasks/<name>/
                return candidate / f"{candidate.name}.yaml"
            if candidate.is_file():
                return candidate
        raise EvalConfigError(f"task file {ref} not found")
    tasks = builtin_tasks()
    if ref not in tasks:
        raise EvalConfigError(
            f"unknown task {ref!r}{_hint(ref, list(tasks))}; built-in tasks: {', '.join(tasks) or 'none found'} "
            "(or give a task YAML file)"
        )
    return tasks[ref]


def _named(task: TaskConfig, where: str) -> TaskConfig:
    if not task.name or not NAME.fullmatch(task.name):
        raise EvalConfigError(
            f"{where}: `name` must be letters, digits, '_' or '-' (it names the result files), got {task.name!r}"
        )
    return task


def task_errors(task: TaskConfig, modes: set[str]) -> list[str]:
    """Raise EvalConfigError on a task that cannot run; return warnings."""
    where = f"task {task.name}"
    errors: list[str] = []
    if not task.dataset:
        errors.append(f"{where}: `dataset` is empty (a Hugging Face dataset id or a .jsonl / .json / .parquet file).")
    if task.limit < 0 or task.k < 0:
        errors.append(f"{where}: limit and k must be >= 0.")
    if task.output not in OUTPUTS:
        errors.append(f"{where}: output must be one of {', '.join(OUTPUTS)}, got {task.output!r}.")
    if task.template and task.template not in TEMPLATES:
        errors.append(f"{where}: template must be one of {', '.join(TEMPLATES)}, got {task.template!r}.")
    if task.output == "choices" and (task.env or task.k > 1):
        errors.append(f"{where}: output choices scores log-likelihoods: it takes no env and no k.")
    if sum(bool(x) for x in (task.prompt, task.messages, task.prompt_field)) > 1:
        errors.append(f"{where}: give one of prompt, messages and prompt_field.")
    if task.scorer and task.rewards:
        errors.append(f"{where}: give `scorer` (one) or `rewards` (several, weighted), not both.")
    if task.output == "choices":
        pass  # scored by the log-likelihoods
    elif not (task.scorer or task.rewards or task.env.get("type", "single_turn") not in ("single_turn", "tools")):
        errors.append(
            f"{where}: no score: set `scorer` (built-in: exact_match, numeric, mcq, regex_match, json_schema, "
            "function_call, ifeval; RL rewards: math, code, judge, ...; or module:function), `rewards`, or an "
            "environment with get_reward."
        )
    for i, message in enumerate(task.messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message.get("role"):
            errors.append(f"{where}: messages[{i}] must be a mapping with `role` and a `content` string.")
    for name, spec in task.fields.items():
        errors += _field_errors(f"{where}: fields.{name}", spec)
    if task.join and not (isinstance(task.join.get("data_files"), str) and isinstance(task.join.get("key"), str)):
        errors.append(f"{where}: join needs `data_files` (the second file) and `key` (the column both files share).")
    for name, entry in task.rewards.items():
        if not (isinstance(entry, str) or (isinstance(entry, dict) and entry.get("fn"))):
            errors.append(f"{where}: rewards.{name} must be a function name or a mapping with `fn`.")
    for mode in task.modes:
        if mode not in modes:
            errors.append(f"{where}: unknown mode {mode!r}{_hint(str(mode), sorted(modes))}")
    try:
        errors += sampling_errors(_sampling(task.sampling, f"{where}: sampling"), f"{where}: sampling")
    except EvalConfigError as e:
        errors.append(str(e))
    if errors:
        raise EvalConfigError("\n".join(errors))
    return []


def _field_errors(where: str, spec: Any) -> list[str]:
    if isinstance(spec, str):
        return [] if spec else [f"{where}: an empty column name."]
    if not isinstance(spec, dict):
        return [f"{where}: a column name or a mapping ({', '.join(FIELD_SOURCES)} plus transforms), got {spec!r}."]
    errors = []
    unknown = [k for k in spec if k not in FIELD_SOURCES + FIELD_TRANSFORMS]
    for key in unknown:
        errors.append(f"{where}: unknown key {key!r}{_hint(key, list(FIELD_SOURCES + FIELD_TRANSFORMS))}")
    if sum(k in spec for k in FIELD_SOURCES) != 1:
        errors.append(f"{where}: give exactly one of {', '.join(FIELD_SOURCES)}.")
    if "regex" in spec:
        try:
            re.compile(spec["regex"])
        except (re.error, TypeError) as e:
            errors.append(f"{where}: regex {spec['regex']!r} does not compile ({e}).")
    for key in ("replace", "map"):
        if key in spec and not isinstance(spec[key], dict):
            errors.append(f"{where}: {key} must be a mapping.")
    return errors


# --------------------------------------------------------------- modes, sampling


def resolve_modes(names: list, custom: dict, where: str) -> dict[str, dict]:
    """{mode name: {"chat_template_kwargs": {...}, "sampling": SamplingConfig}} for `names`, from the built-in
    modes and the config's `modes:` definitions."""
    out: dict[str, dict] = {}
    for name in names:
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise EvalConfigError(f"{where}: a mode name must be letters, digits, '_' or '-', got {name!r}")
        if name in custom:
            entry = custom[name] or {}
            if not isinstance(entry, dict) or set(entry) - {"chat_template_kwargs", "sampling"}:
                raise EvalConfigError(
                    f"modes.{name}: a mapping with chat_template_kwargs and / or sampling, got {entry!r}"
                )
            kwargs = entry.get("chat_template_kwargs") or {}
            if not isinstance(kwargs, dict):
                raise EvalConfigError(f"modes.{name}.chat_template_kwargs must be a mapping.")
            sampling = _sampling(entry.get("sampling") or {}, f"modes.{name}.sampling")
            problems = sampling_errors(sampling, f"modes.{name}.sampling")
            if problems:
                raise EvalConfigError("\n".join(problems))
            out[name] = {"chat_template_kwargs": dict(kwargs), "sampling": sampling}
        elif name in MODES:
            out[name] = {"chat_template_kwargs": dict(MODES[name]), "sampling": SamplingConfig()}
        else:
            known = sorted({**MODES, **custom})
            raise EvalConfigError(
                f"{where}: unknown mode {name!r}{_hint(name, known)}; modes: {', '.join(known)} "
                "(define others under `modes:`)"
            )
    return out


def _all_modes(config: EvalConfig) -> list:
    return list(dict.fromkeys([*config.run.modes, *config.modes, *(m for t in config.tasks for m in t.modes)]))


def _sampling(options: dict, where: str) -> SamplingConfig:
    sampling = SamplingConfig()
    if not isinstance(options, dict):
        raise EvalConfigError(f"{where} must be a mapping of sampling options.")
    _apply(sampling, where, options, where, prefix=False)
    return sampling


def sampling_errors(s: SamplingConfig, where: str) -> list[str]:
    errors = []
    if s.temperature is not None and s.temperature < 0:
        errors.append(f"{where}.temperature must be >= 0, got {s.temperature}.")
    if s.top_p is not None and not 0 < s.top_p <= 1:
        errors.append(f"{where}.top_p must be in (0, 1], got {s.top_p}.")
    if s.top_k is not None and s.top_k < 0:
        errors.append(f"{where}.top_k must be >= 0, got {s.top_k}.")
    for name in ("max_tokens", "max_completion_tokens"):
        value = getattr(s, name)
        if value is not None and value < 1:
            errors.append(f"{where}.{name} must be >= 1, got {value}.")
    return errors


def merge_sampling(*layers: SamplingConfig) -> SamplingConfig:
    """The layers' options, later layers over earlier ones where they set a value."""
    out = SamplingConfig()
    for layer in layers:
        for f in dataclasses.fields(SamplingConfig):
            value = getattr(layer, f.name)
            if value is not None:
                setattr(out, f.name, value)
    return out


# ------------------------------------------------------------------- plumbing

_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "dict": dict, "list": list}


def _apply(section, name: str, options: dict, where: str, prefix: bool = True) -> None:
    """Set a dataclass section's options from a mapping, checked against the fields' types."""
    fields = {f.name: f for f in dataclasses.fields(section)}
    for key, value in options.items():
        label = f"{name}.{key}" if prefix else f"{name}: {key}"
        if key not in fields:
            raise EvalConfigError(f"{where}: unknown option {label}{_hint(str(key), list(fields))}")
        setattr(section, key, _coerce(str(fields[key].type), value, f"{where}: {label}"))


def _coerce(annotation: str, value: Any, where: str) -> Any:
    optional = "None" in annotation
    kind = _TYPES[annotation.replace("| None", "").strip()]
    if isinstance(value, str) and kind is not str:
        text = value.strip()
        if optional and text.lower() in ("none", "null", ""):
            return None
        try:
            if kind is bool:
                if text.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError
                return text.lower() in ("true", "1", "yes")
            if kind in (int, float):  # YAML reads 2e-5 as a string; command-line values are strings
                return kind(text)
            value = yaml.safe_load(text)  # mappings and lists given on the command line
        except (ValueError, yaml.YAMLError):
            raise EvalConfigError(f"{where}={value!r} is not a valid {kind.__name__}.") from None
    if value is None:
        if optional:
            return None
        if kind in (dict, list, str):
            return kind()
        raise EvalConfigError(f"{where} needs a value ({kind.__name__}).")
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is list and isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, kind) or (kind in (int, float) and isinstance(value, bool)):
        raise EvalConfigError(f"{where} must be a {kind.__name__}, got {value!r}.")
    return value


def _hint(name: str, names: list[str]) -> str:
    close = difflib.get_close_matches(name, names, n=1)
    return f" (did you mean {close[0]}?)" if close else ""


def _read_mapping(path: str | Path) -> dict:
    try:
        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
    except OSError as e:
        raise EvalConfigError(f"{path}: {e.strerror or e}") from None
    except yaml.YAMLError as e:
        raise EvalConfigError(f"{path}: not valid YAML ({e})") from None
    if not isinstance(raw, dict):
        raise EvalConfigError(f"{path}: expected a mapping of options.")
    return raw
