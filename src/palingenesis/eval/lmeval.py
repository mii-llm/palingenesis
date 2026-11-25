"""Tasks written for lm-evaluation-harness, read as they are: `pgs eval --tasks path/to/lm_eval/tasks/gsm8k/gsm8k.yaml`.

The common subset of the task format (https://github.com/EleutherAI/lm-evaluation-harness, MIT):

    dataset_path / dataset_name / dataset_kwargs.data_files     the dataset
    test_split | validation_split, fewshot_split | training_split
    output_type            generate_until | multiple_choice
    doc_to_text / doc_to_target / doc_to_choice                 a column, a Jinja template, a literal or !function
    description, target_delimiter (" "), fewshot_delimiter ("\\n\\n"), num_fewshot
    fewshot_config         sampler: default (random, seed 1234) | first_n; doc_to_text / doc_to_target / doc_to_choice
    generation_kwargs      until, max_gen_toks (256), temperature, do_sample, top_p, top_k
    filter_list            regex (regex_pattern, group_select, fallback), take_first, remove_whitespace, lowercase,
                           uppercase
    metric_list            exact_match (ignore_case, ignore_punctuation, ignore_numbers, regexes_to_ignore);
                           acc, acc_norm (multiple_choice); any metric a process_results !function returns
    process_docs, process_results                               !function (a module next to the YAML)
    include                another YAML of the same directory, under this one
    metadata.version

One extension: fewshot_config.group_by: <column> draws a document's examples from the few-shot rows that share
its value of the column (one MMLU file for all subjects instead of 57).

What a harness does with these fields is reproduced exactly: the prompt is description + examples
(text + target_delimiter + target + fewshot_delimiter) + text; a multiple-choice continuation is target_delimiter +
choice; the default sampler draws with random.Random(1234) in document order; filters and exact_match are the
harness's. Everything else (loglikelihood_rolling, custom filters and aggregations, images, groups) is refused with
an error that names the key. `!function` runs Python from the task's directory, as the harness does: read a task
before you run it.
"""

from __future__ import annotations

import ast
import importlib.util
import random
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from palingenesis.eval.config import EvalConfigError, TaskConfig

MARKERS = ("doc_to_text", "dataset_path", "output_type", "doc_to_target")
SAMPLER_SEED = 1234  # lm-evaluation-harness's few-shot seed
DEFAULT_MAX_GEN_TOKS = 256
UNSUPPORTED = ("doc_to_image", "doc_to_audio", "use_prompt", "gen_prefix", "unsafe_code")
FILTERS = ("regex", "take_first", "remove_whitespace", "lowercase", "uppercase")
CHOICE_METRICS = ("acc", "acc_norm")


class _Loader(yaml.SafeLoader):
    """YAML with the harness's `!function module.name` tag (kept as a reference, loaded when called)."""


_Loader.add_constructor("!function", lambda loader, node: {"!function": loader.construct_scalar(node)})


def read_yaml(path: str | Path) -> dict:
    """The task mapping of `path` with its `include` chain merged under it."""
    path = Path(path)
    try:
        raw = yaml.load(path.read_text(), Loader=_Loader) or {}  # noqa: S506 — a SafeLoader subclass
    except (OSError, yaml.YAMLError) as e:
        raise EvalConfigError(f"{path}: {e}") from None
    if not isinstance(raw, dict):
        raise EvalConfigError(f"{path}: expected a mapping of options.")
    include = raw.pop("include", None)
    if include:
        base = read_yaml(path.parent / str(include))
        raw = {**base, **raw}
    return raw


def is_lmeval_task(path: str | Path) -> bool:
    """Whether a YAML file is in lm-evaluation-harness's task format (its keys, not ours)."""
    try:
        raw = yaml.load(Path(path).read_text(), Loader=_Loader) or {}  # noqa: S506
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(raw, dict) and ("include" in raw or any(k in raw for k in MARKERS)) and "scorer" not in raw


def load(path: str | Path) -> TaskConfig:
    """The TaskConfig of an lm-evaluation-harness task file. No data is read and nothing is imported."""
    path = Path(path).resolve()
    raw = read_yaml(path)
    where = str(path)
    name = raw.get("task")
    if isinstance(name, list) or "group" in raw and "doc_to_text" not in raw:
        raise EvalConfigError(f"{where}: a group of tasks: give its task files (a glob works: --tasks 'dir/*.yaml').")
    for key in UNSUPPORTED:
        if raw.get(key):
            raise EvalConfigError(f"{where}: `{key}` is outside the supported subset of the lm-eval task format.")
    output = raw.get("output_type", "generate_until")
    if output not in ("generate_until", "multiple_choice"):
        raise EvalConfigError(
            f"{where}: output_type {output!r} is not supported (generate_until and multiple_choice are)."
        )
    if "doc_to_text" not in raw or not raw.get("dataset_path"):
        raise EvalConfigError(f"{where}: an lm-eval task needs dataset_path and doc_to_text.")
    split = raw.get("test_split") or raw.get("validation_split")
    if not split:
        raise EvalConfigError(f"{where}: no test_split or validation_split.")
    task = TaskConfig(
        name=re.sub(r"[^A-Za-z0-9_\-]", "_", str(name or path.stem)),
        description=f"lm-evaluation-harness task {name or path.stem} ({output})",
        dataset=str(raw["dataset_path"]),
        subset=str(raw.get("dataset_name") or ""),
        split=str(split),
        data_files=_data_file(raw, str(split)),
        template="completion",
        version=str((raw.get("metadata") or {}).get("version", "")),
        category_field=str(raw.get("category_field") or ""),
        lmeval={**raw, "_dir": str(path.parent), "_file": where},
    )
    if output == "multiple_choice":
        if raw.get("doc_to_choice") is None:
            raise EvalConfigError(f"{where}: multiple_choice needs doc_to_choice.")
        for entry in raw.get("metric_list") or []:
            if entry.get("metric") not in CHOICE_METRICS:
                raise EvalConfigError(f"{where}: multiple_choice metric {entry.get('metric')!r} (acc, acc_norm are).")
        task.output = "choices"
        task.target_delimiter = str(raw.get("target_delimiter", " "))
        return task
    kwargs = dict(raw.get("generation_kwargs") or {})
    until = kwargs.get("until", [raw.get("fewshot_delimiter", "\n\n")])
    until = [until] if isinstance(until, str) else list(until or [])
    sampled = bool(kwargs.get("do_sample")) and float(kwargs.get("temperature") or 0) > 0
    task.sampling = {
        "max_tokens": int(kwargs.get("max_gen_toks", DEFAULT_MAX_GEN_TOKS)),
        "temperature": float(kwargs["temperature"]) if sampled else 0.0,
        "stop": until,
        **({"top_p": float(kwargs["top_p"])} if sampled and kwargs.get("top_p") else {}),
        **({"top_k": int(kwargs["top_k"])} if sampled and kwargs.get("top_k") else {}),
    }
    task.rewards = _rewards(raw, where)
    return task


def _data_file(raw: dict, split: str) -> str:
    files = (raw.get("dataset_kwargs") or {}).get("data_files")
    if isinstance(files, dict):
        files = files.get(split)
    if isinstance(files, list):
        files = files[0] if len(files) == 1 else None
    return str(files or "")


def _rewards(raw: dict, where: str) -> dict:
    """One reward per (metric, filter), named as the harness names them ("exact_match,strict-match"); the first
    one is the item's score, the others are reported next to it (weight 0)."""
    filters = raw.get("filter_list") or [{"name": "none", "filter": [{"function": "take_first"}]}]
    for pipeline in filters:
        for step in pipeline.get("filter") or []:
            if step.get("function") not in FILTERS:
                raise EvalConfigError(
                    f"{where}: filter {step.get('function')!r} is not supported ({', '.join(FILTERS)} are)."
                )
    metrics = raw.get("metric_list") or [{"metric": "exact_match"}]
    custom = raw.get("process_results")
    rewards: dict[str, dict] = {}
    for metric in metrics:
        name = str(metric.get("metric"))
        if isinstance(metric.get("aggregation"), dict):  # a !function aggregation: not a mean over items
            if custom and len(metrics) > 1:
                continue  # other metrics of the same process_results are still reported
            raise EvalConfigError(f"{where}: metric {name}: a custom aggregation is not supported (means are).")
        if not custom and name != "exact_match":
            raise EvalConfigError(
                f"{where}: metric {name!r} is not built in (exact_match is; others need a process_results function)."
            )
        options = {k: v for k, v in metric.items() if k not in ("metric", "aggregation", "higher_is_better")}
        for pipeline in filters:
            key = name if len(filters) == 1 and pipeline.get("name", "none") == "none" else f"{name},{pipeline['name']}"
            rewards[key] = {
                "fn": "palingenesis.eval.lmeval:generation_metric",
                "weight": 0.0 if rewards else 1.0,
                "args": {"metric": name, "options": options, "filters": pipeline.get("filter") or []},
            }
    if not rewards:
        raise EvalConfigError(f"{where}: no metric of metric_list can be computed here.")
    return rewards


# ------------------------------------------------------------------ templates


@lru_cache(maxsize=1)
def _jinja():
    try:
        from jinja2 import BaseLoader, Environment, StrictUndefined
    except ImportError:
        raise EvalConfigError("lm-eval tasks need jinja2 (pip install jinja2).") from None
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, keep_trailing_newline=True)  # noqa: S701
    env.filters["regex_replace"] = lambda text, pattern, repl, count=0: re.sub(pattern, repl, text, count=count)
    return env


@lru_cache(maxsize=4096)
def _template(source: str):
    return _jinja().from_string(source)


@lru_cache(maxsize=256)
def _function(directory: str, spec: str):
    module_name, _, attribute = spec.rpartition(".")
    path = Path(directory) / (module_name.replace(".", "/") + ".py")
    if not module_name or not path.is_file():
        raise EvalConfigError(f"!function {spec}: {path} not found")
    module_spec = importlib.util.spec_from_file_location(f"pgs_lmeval_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, attribute)


def _resolve(field: Any, doc: dict, directory: str) -> Any:
    """A doc_to_* field on a document: a !function's result, a column's value, a rendered template, or the
    literal itself."""
    if isinstance(field, dict) and "!function" in field:
        return _function(directory, field["!function"])(doc)
    if isinstance(field, str):
        if field in doc:
            return doc[field]
        return _template(field).render(**doc) if "{" in field else field
    return field


def _literal(text: Any) -> Any:
    if isinstance(text, str) and text.strip()[:1] in "[{(" and text.strip():
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return text


class _Fields:
    """doc_to_text / doc_to_choice / doc_to_target of one side (the task's, or its few-shot examples')."""

    def __init__(self, raw: dict, directory: str, base: "_Fields | None" = None):
        self.dir = directory
        self.text = raw.get("doc_to_text", base.text if base else None)
        self.choice = raw.get("doc_to_choice", base.choice if base else None)
        self.target = raw.get("doc_to_target", base.target if base else None)

    def question(self, doc: dict) -> str:
        value = _resolve(self.text, doc, self.dir)
        if not isinstance(value, str):
            raise EvalConfigError(f"doc_to_text gave {type(value).__name__} (a string is needed): {value!r}")
        return value

    def choices(self, doc: dict) -> list[str] | None:
        if self.choice is None:
            return None
        value = self.choice
        if isinstance(value, list):
            return [str(_resolve(v, doc, self.dir)) for v in value]
        value = _literal(_resolve(value, doc, self.dir))
        if isinstance(value, dict):
            value = list(value.values())
        if not isinstance(value, list | tuple):
            raise EvalConfigError(f"doc_to_choice gave {type(value).__name__} (a list is needed): {value!r}")
        return [str(v) for v in value]

    def gold(self, doc: dict) -> Any:
        value = _resolve(self.target, doc, self.dir)
        if isinstance(value, str):
            stripped = value.strip()
            if self.choice is not None and stripped.lstrip("-").isdigit():
                return int(stripped)
            return _literal(value) if stripped[:1] == "[" else value
        return value

    def answer(self, doc: dict) -> str:
        """The text an example's answer shows."""
        gold, choices = self.gold(doc), self.choices(doc)
        if choices and isinstance(gold, int) and not isinstance(gold, bool):
            return choices[gold]
        if isinstance(gold, list):
            gold = gold[0]
        return str(gold)


def _description(raw: dict, doc: dict, directory: str) -> str:
    return str(_resolve(raw.get("description") or "", doc, directory) or "")


# ---------------------------------------------------------------------- items


def build_rows(task: TaskConfig, docs: list[dict], fewshot_docs: list[dict] | None, num_fewshot: int = -1) -> list:
    """The task's rows from its documents, in document order: each document plus `prompt_text` (the context a
    harness builds), `target` (the gold text or texts) and, for multiple choice, `choices` and `gold`."""
    raw = task.lmeval
    directory = raw["_dir"]
    fields = _Fields(raw, directory)
    fewshot_config = dict(raw.get("fewshot_config") or {})
    shots = _Fields(fewshot_config, directory, fields)
    n = int(raw.get("num_fewshot") or 0) if num_fewshot < 0 else num_fewshot
    delimiter = str(raw.get("fewshot_delimiter", "\n\n"))
    target_delimiter = str(raw.get("target_delimiter", " "))
    sampler = str(fewshot_config.get("sampler", "default"))
    if sampler not in ("default", "first_n"):
        raise EvalConfigError(f"task {task.name}: fewshot_config.sampler {sampler!r} is not supported.")
    group_by = fewshot_config.get("group_by")
    pools: dict[Any, list[dict]] = {}
    if n > 0:
        if not fewshot_docs:
            raise EvalConfigError(f"task {task.name}: num_fewshot {n} without a fewshot_split / training_split.")
        for doc in fewshot_docs:
            pools.setdefault(doc.get(group_by) if group_by else None, []).append(doc)
    same_split = fewshot_split(task) == task.split
    rnd = random.Random(SAMPLER_SEED)
    rows = []
    for index, doc in enumerate(docs):
        where = f"task {task.name}, row {index}"
        try:
            examples = _examples(pools.get(doc.get(group_by) if group_by else None, []), n, sampler, rnd, doc, same_split)
            parts = [_description(raw, doc, directory)]
            for example in examples:
                parts.append(shots.question(example) + target_delimiter + shots.answer(example) + delimiter)
            parts.append(fields.question(doc))
            row = {**doc, "prompt_text": "".join(parts)}
            choices, gold = fields.choices(doc), fields.gold(doc)
            if task.output == "choices":
                if isinstance(gold, str):
                    gold = choices.index(gold) if gold in choices else int(gold)
                if not isinstance(gold, int) or not -len(choices) <= gold < len(choices):
                    raise EvalConfigError(f"doc_to_target gave {gold!r}: not an index into {len(choices)} choices")
                row.update(choices=choices, gold=gold % len(choices), target=choices[gold])
            else:
                if choices is not None and isinstance(gold, int):
                    gold = choices[gold]
                row["target"] = [str(g) for g in gold] if isinstance(gold, list) else str(gold)
        except EvalConfigError as e:
            raise EvalConfigError(f"{where}: {e}") from None
        except Exception as e:  # noqa: BLE001 — a template or a task function failing on this document
            raise EvalConfigError(f"{where}: {type(e).__name__}: {e}") from None
        row["_doc"] = doc
        if raw.get("process_results"):
            row["_lmeval"] = {"_dir": directory, "process_results": raw["process_results"]}
        rows.append(row)
    return rows


def _examples(pool: list[dict], n: int, sampler: str, rnd: random.Random, doc: dict, same_split: bool) -> list[dict]:
    """The few-shot examples of one document, as the harness's samplers draw them."""
    if n <= 0:
        return []
    if sampler == "first_n":
        pool = [x for x in pool if x != doc] if same_split else pool
        if n > len(pool):
            raise EvalConfigError(f"{n} few-shot examples requested, {len(pool)} available")
        return pool[:n]
    if not same_split:
        return rnd.sample(pool, n)
    drawn = [x for x in rnd.sample(pool, n + 1) if x != doc][:n]
    return drawn if len(drawn) == n else rnd.sample([x for x in pool if x != doc], n)


def fewshot_split(task: TaskConfig) -> str:
    raw = task.lmeval
    return str(raw.get("fewshot_split") or raw.get("training_split") or raw.get("validation_split") or "")


def process_docs(task: TaskConfig, rows: list[dict]) -> list[dict]:
    """The task's process_docs function on the rows (it takes and returns a datasets.Dataset)."""
    spec = task.lmeval.get("process_docs")
    if not spec:
        return rows
    try:
        from datasets import Dataset
    except ImportError:
        raise EvalConfigError(f"task {task.name}: process_docs needs the datasets library.") from None
    return [dict(r) for r in _function(task.lmeval["_dir"], spec["!function"])(Dataset.from_list(rows))]


# -------------------------------------------------------------------- scoring


def cut_at_stop(text: str, stops) -> str:
    """`text` up to the first stop string (the harness cuts every generation again, whatever the engine did)."""
    for stop in stops or ():
        if stop:
            text = text.split(stop)[0]
    return text


def apply_filters(response: str, filters: list[dict]) -> str:
    """The harness's filters on one response (a one-element list of responses, to it)."""
    for step in filters:
        function = step.get("function")
        if function == "regex":
            match = re.compile(step.get("regex_pattern", r"#### (\-?[0-9\.\,]+)")).findall(response)
            fallback = step.get("fallback", "[invalid]")
            if match:
                match = match[int(step.get("group_select", 0))]
                if isinstance(match, tuple):
                    match = next((m for m in match if m), fallback)
                response = match.strip()
            else:
                response = fallback
        elif function == "remove_whitespace":
            response = response.lstrip()
        elif function == "lowercase":
            response = response.lower()
        elif function == "uppercase":
            response = response.upper()
        # take_first: there is one response
    return response


def exact_match(
    prediction: str,
    reference: str,
    regexes_to_ignore: list | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> float:
    """The harness's exact_match (Hugging Face evaluate's): the options are applied to both strings."""
    for pattern in regexes_to_ignore or []:
        prediction, reference = re.sub(pattern, "", prediction), re.sub(pattern, "", reference)
    if ignore_case:
        prediction, reference = prediction.lower(), reference.lower()
    if ignore_punctuation:
        table = str.maketrans("", "", string.punctuation)
        prediction, reference = prediction.translate(table), reference.translate(table)
    if ignore_numbers:
        table = str.maketrans("", "", string.digits)
        prediction, reference = prediction.translate(table), reference.translate(table)
    return float(prediction == reference)


def generation_metric(
    completion: str, metric: str = "exact_match", options: dict | None = None, filters: list | None = None, **row
) -> float:
    """One (metric, filter) of a generate_until task on one completion. With a process_results function, the
    metric it returns under that name."""
    response = apply_filters(completion, filters or [])
    raw = row.get("_lmeval") or {}
    custom = raw.get("process_results")
    if custom:
        values = _function(raw["_dir"], custom["!function"])(row.get("_doc", row), [response])
        value = values.get(metric)
        if isinstance(value, bool | int | float):
            return float(value)
        raise EvalConfigError(f"process_results gave {metric} = {value!r}: only numbers (averaged over items) work")
    target = row.get("target", "")
    targets = target if isinstance(target, list) else [target]
    return float(any(exact_match(response, str(t), **(options or {})) for t in targets))
