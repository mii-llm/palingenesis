"""Built-in evaluation scorers, and how a task's `scorer` resolves to RL rewards.

A scorer is an RL reward function (palingenesis.rl.rewards): it declares the keyword arguments it wants (completion,
reasoning, messages, prompt, finish_reason, sandbox, env, any dataset column) and returns a number, or None when it
does not apply to the sample. A task names one of:

    exact_match     the answer text equals the gold text (or one of several), normalised
    numeric         the answer's number equals the gold number (\\boxed{}, "answer is", else the last number)
    mcq             the option letter the completion concludes with equals the gold letter (or index)
    regex_match     a pattern's capture equals the gold text
    json_schema     the completion is JSON that validates against a schema (and equals an expected value)
    function_call   the assistant's tool calls match the expected calls (BFCL-style possible answers)
    ifeval          the response follows every instruction of an IFEval item (palingenesis.eval.ifeval)
    any RL reward   math (\\boxed{} with symbolic equivalence), exact, choice, boxed_choice, qa_match, regex, code
                    (hidden tests in the sandbox), judge, equivalence (LLM judges), ...
    module:function your own (or path/to/file.py:function)

LLM judges (`judge`, `equivalence`, or any scorer with `judge_validation:`) are run on labelled examples before the
evaluation and refused below the agreement the task asks for (validate_judge).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import string
from typing import Any, Callable

from palingenesis.eval.config import EvalConfigError, TaskConfig

# ----------------------------------------------------------------- extraction

_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.I)
_NUMBER = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?")
# "answer is (C)", "Answer: C", "the correct option is **C**"; the letter itself is upper case
_LETTER_STATED = re.compile(r"(?i:answer|option|choice)\s*(?i:is|:)?\s*[:\-]?\s*\**\s*\(?([A-Z])\)?(?![A-Za-z])")
_FENCED_JSON = re.compile(r"```(?:json|JSON)?[ \t]*\n(.*?)```", re.S)


def _golds(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list | tuple) else [str(value)]


def _field(row: dict, field: str, scorer: str) -> Any:
    if field not in row:
        raise KeyError(f"{scorer}: the row has no {field!r} column (set scorer_args.field or map it in fields:)")
    return row[field]


def normalize_text(text: str, ignore_case: bool = True, ignore_punctuation: bool = False) -> str:
    text = " ".join(str(text).split())
    if ignore_case:
        text = text.lower()
    if ignore_punctuation:
        text = " ".join(text.translate(str.maketrans("", "", string.punctuation)).split())
    return text


def extract_text(completion: str, extract: str = "full", pattern: str = "") -> str | None:
    """The answer span of a completion: "full" (all of it), "last_line", "boxed" (the last \\boxed{}), "answer"
    (after the last "Answer:" / "answer is", else the last line) or "regex" (`pattern`'s last match, group 1)."""
    from palingenesis.rl.grading import last_boxed

    text = completion.strip()
    if extract == "full":
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if extract == "last_line":
        return lines[-1] if lines else None
    if extract == "boxed":
        return last_boxed(text)
    if extract == "answer":
        stated = _ANSWER_LINE.findall(text)
        if stated:
            return stated[-1].strip(" *\t").rstrip(".").strip(" *\t")  # "**Paris**." -> "Paris"
        return lines[-1] if lines else None
    if extract == "regex":
        matches = list(re.finditer(pattern, text, re.S | re.M))
        if not matches:
            return None
        return matches[-1].group(1 if matches[-1].re.groups else 0)
    raise ValueError(f"extract must be full, last_line, boxed, answer or regex, got {extract!r}")


def parse_number(text: str | None) -> float | None:
    """The last number in `text` (thousands separators, currency and percent signs aside; a/b fractions)."""
    if text is None:
        return None
    matches = _NUMBER.findall(str(text).replace("\\!", "").replace("{,}", ""))
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/")
            return float(numerator) / float(denominator)
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


def extract_number(completion: str, extract: str = "auto") -> float | None:
    """The completion's final number: "boxed" (the last \\boxed{} only), "last_number", or "auto" (\\boxed{}, else
    after "answer is" / "Answer:", else the last number)."""
    from palingenesis.rl.grading import last_boxed

    if extract == "last_number":
        return parse_number(completion)
    boxed = last_boxed(completion)
    if extract == "boxed":
        return parse_number(boxed)
    if extract != "auto":
        raise ValueError(f"extract must be auto, boxed or last_number, got {extract!r}")
    if boxed is not None and parse_number(boxed) is not None:
        return parse_number(boxed)
    stated = _ANSWER_LINE.findall(completion)
    if stated and parse_number(stated[-1]) is not None:
        return parse_number(stated[-1])
    return parse_number(completion)


def extract_choice(completion: str, letters: str = "ABCDEFGHIJ") -> str | None:
    """The option letter a completion concludes with: the last "answer is (X)" / "Answer: X", else a boxed letter,
    else a completion that is only a letter, else the last standalone letter."""
    from palingenesis.opd.formatting import extract_letter
    from palingenesis.rl.grading import last_boxed

    stated = [m.upper() for m in _LETTER_STATED.findall(completion) if m.upper() in letters]
    if stated:
        return stated[-1]
    boxed = last_boxed(completion)
    if boxed is not None:
        inner = re.sub(r"\\text\{([^}]*)\}", r"\1", boxed).strip().strip("().").strip()
        if len(inner) == 1 and inner.upper() in letters:
            return inner.upper()
    bare = completion.strip().strip("*().").strip()
    if len(bare) == 1 and bare.upper() in letters:
        return bare.upper()
    letter = extract_letter(completion, last=True)
    return letter if letter is not None and letter in letters else None


def extract_json(completion: str) -> Any:
    """The JSON value of a completion: the last fenced block, else the whole text, else its outermost {...} or
    [...] span. Raises ValueError when there is none."""
    text = completion.strip()
    candidates = [block.strip() for block in _FENCED_JSON.findall(text)][::-1] + [text]
    for opener, closer in ("{}", "[]"):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON value in the completion")


# -------------------------------------------------------------------- scorers


def exact_match(
    completion: str,
    field: str = "answer",
    extract: str = "full",
    pattern: str = "",
    ignore_case: bool = True,
    ignore_punctuation: bool = False,
    **row,
) -> float:
    """1.0 when the answer text (`extract`: full, last_line, boxed, answer, regex with `pattern`) equals the row's
    `field` (a text, or a list of accepted texts) after normalisation (whitespace; case and punctuation on option)."""
    answer = extract_text(completion, extract, pattern)
    if answer is None:
        return 0.0
    norm = normalize_text(answer, ignore_case, ignore_punctuation)
    golds = _golds(_field(row, field, "exact_match"))
    return float(any(norm == normalize_text(g, ignore_case, ignore_punctuation) for g in golds))


def numeric(
    completion: str,
    field: str = "answer",
    extract: str = "auto",
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    **row,
) -> float:
    """1.0 when the completion's final number (`extract`: auto, boxed, last_number) equals the number in the row's
    `field` within the tolerances."""
    prediction = extract_number(completion, extract)
    if prediction is None:
        return 0.0
    for gold in _golds(_field(row, field, "numeric")):
        value = parse_number(gold)
        if value is None:
            raise ValueError(f"numeric: the gold answer {gold!r} holds no number")
        if math.isclose(prediction, value, rel_tol=rel_tol, abs_tol=abs_tol):
            return 1.0
    return 0.0


def mcq(completion: str, field: str = "answer", letters: str = "ABCDEFGHIJ", **row) -> float:
    """1.0 when the option letter the completion concludes with (extract_choice) is the row's `field`: a letter, or
    an option index (0-based)."""
    gold = _field(row, field, "mcq")
    if isinstance(gold, int) and not isinstance(gold, bool):
        gold = letters[gold]
    gold = str(gold).strip().strip("().").upper()[:1]
    return float(extract_choice(completion, letters) == gold)


def regex_match(
    completion: str, pattern: str, field: str = "", group: int = 1, ignore_case: bool = True, **row
) -> float:
    """1.0 when `pattern` matches the completion and, with `field`, its last match's `group` equals the row's
    `field` (normalised); without `field`, a match is enough."""
    matches = list(re.finditer(pattern, completion, re.S | re.M))
    if not matches:
        return 0.0
    if not field:
        return 1.0
    found = matches[-1].group(group if matches[-1].re.groups else 0) or ""
    norm = normalize_text(found, ignore_case)
    return float(any(norm == normalize_text(g, ignore_case) for g in _golds(_field(row, field, "regex_match"))))


def json_schema(
    completion: str, schema: dict | None = None, schema_field: str = "", expected_field: str = "", **row
) -> float:
    """1.0 when the completion holds a JSON value (extract_json) that validates against `schema` (or the row's
    `schema_field`) and, with `expected_field`, equals the row's expected value."""
    try:
        value = extract_json(completion)
    except ValueError:
        return 0.0
    rule = schema if schema is not None else (_field(row, schema_field, "json_schema") if schema_field else None)
    if isinstance(rule, str):
        rule = json.loads(rule)
    if rule is not None and schema_errors(value, rule):
        return 0.0
    if expected_field:
        expected = _field(row, expected_field, "json_schema")
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                pass
        return float(value == expected)
    if rule is None:
        raise ValueError("json_schema: give schema, schema_field or expected_field")
    return 1.0


def schema_errors(value: Any, schema: dict) -> list[str]:
    """Why `value` does not validate against a JSON Schema ([] when it does): the jsonschema package when it is
    installed, else a validator of the common keywords (type, properties, required, additionalProperties, items,
    enum, const, minimum / maximum, minLength / maxLength, pattern, minItems / maxItems, anyOf / oneOf / allOf)."""
    try:
        import jsonschema
    except ImportError:
        return _schema_errors(value, schema, "$")
    validator = jsonschema.validators.validator_for(schema)(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '$'}: {e.message}" for e in validator.iter_errors(value)]


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
}


def _schema_errors(value: Any, schema: Any, path: str) -> list[str]:
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: nothing is allowed here"]
    errors: list[str] = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_JSON_TYPES.get(kind, lambda v: False)(value) for kind in kinds):
            return [f"{path}: expected {' or '.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above the maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            errors.append(f"{path}: length {len(value)} out of bounds")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            errors.append(f"{path}: {len(value)} items out of bounds")
        if isinstance(schema.get("items"), dict | bool):
            for i, item in enumerate(value):
                errors += _schema_errors(item, schema["items"], f"{path}/{i}")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: missing {name!r}")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                errors += _schema_errors(item, properties[name], f"{path}/{name}")
            elif extra is False:
                errors.append(f"{path}: unexpected {name!r}")
            elif isinstance(extra, dict):
                errors += _schema_errors(item, extra, f"{path}/{name}")
    for sub in schema.get("allOf") or []:
        errors += _schema_errors(value, sub, path)
    if "anyOf" in schema and all(_schema_errors(value, sub, path) for sub in schema["anyOf"]):
        errors.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema and sum(not _schema_errors(value, sub, path) for sub in schema["oneOf"]) != 1:
        errors.append(f"{path}: must match exactly one of oneOf")
    return errors


# ------------------------------------------------------------ function calling

_LOOSE = str.maketrans("", "", " ,./-_*^")


def _standard(value: Any) -> Any:
    """Values compared the way BFCL's checker does: strings without case, spaces and ,./-_*^ and with one quote
    style; integers equal to their floats; containers element by element."""
    if isinstance(value, str):
        return value.translate(_LOOSE).lower().replace("'", '"')
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple):
        return [_standard(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _standard(v) for k, v in value.items()}
    return value


def _value_allowed(value: Any, allowed: list) -> bool:
    if isinstance(value, dict):  # a nested object: each of its keys against that key's own allowed values
        for option in allowed:
            if isinstance(option, dict) and all(isinstance(v, list) for v in option.values()):
                if _arguments_match(value, option):
                    return True
    norm = _standard(value)
    return any(norm == _standard(option) for option in allowed if option != "")


def _arguments_match(arguments: dict, expected: dict) -> bool:
    for name, value in arguments.items():
        if name not in expected or not _value_allowed(value, expected[name]):
            return False
    return all(name in arguments or "" in allowed for name, allowed in expected.items())  # "" = may be left out


def assistant_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    """(name, arguments) of the tool calls in the final assistant message."""
    last = next((m for m in reversed(messages) if m.get("role") == "assistant"), {})
    calls = []
    for call in last.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        calls.append((str(function.get("name")), arguments if isinstance(arguments, dict) else {"": arguments}))
    return calls


def function_call(messages: list[dict], field: str = "expected_calls", ordered: bool = False, **row) -> float:
    """1.0 when the final assistant message's tool calls are the expected ones. The row's `field` lists them as
    [{function name: {argument: [accepted values]}}] (BFCL's possible answers): an argument's value must be one of
    its accepted values, "" among them makes the argument optional, and arguments that are not listed fail. Calls
    are matched in any order unless `ordered`. An empty list expects no call (irrelevance)."""
    expected = _field(row, field, "function_call")
    if isinstance(expected, str):
        expected = json.loads(expected)
    wanted: list[tuple[str, dict]] = []
    for entry in expected or []:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"function_call: an expected call is {{name: {{argument: [values]}}}}, got {entry!r}")
        ((name, arguments),) = entry.items()
        wanted.append((str(name), {k: v if isinstance(v, list) else [v] for k, v in (arguments or {}).items()}))
    calls = assistant_calls(messages)
    if len(calls) != len(wanted):
        return 0.0
    if ordered:
        return float(all(n == wn and _arguments_match(a, wa) for (n, a), (wn, wa) in zip(calls, wanted)))
    remaining = list(calls)
    for name, arguments in wanted:
        match = next((c for c in remaining if c[0] == name and _arguments_match(c[1], arguments)), None)
        if match is None:
            return 0.0
        remaining.remove(match)
    return 1.0


def ifeval(
    completion: str,
    instruction_id_list: list,
    kwargs: list,
    loose: bool = False,
    level: str = "prompt",
    checker: str = "auto",
    **row,
) -> float | None:
    """IFEval: 1.0 when the response follows every instruction of the item (`level: instruction`: the share it
    follows); `loose`: the benchmark's loose criterion. None (the item is left out) when an instruction cannot be
    checked here (see palingenesis.eval.ifeval)."""
    from palingenesis.eval.ifeval import follows

    results = follows(completion, instruction_id_list, kwargs, loose=loose, checker=checker)
    if results is None:
        return None
    if level == "instruction":
        return sum(results) / len(results)
    if level != "prompt":
        raise ValueError(f"ifeval: level must be prompt or instruction, got {level!r}")
    return float(all(results))


SCORERS: dict[str, Callable] = {
    "exact_match": exact_match,
    "numeric": numeric,
    "mcq": mcq,
    "regex_match": regex_match,
    "json_schema": json_schema,
    "function_call": function_call,
    "ifeval": ifeval,
}
JUDGES = ("judge", "equivalence")  # RL rewards that ask an LLM: validated before use


def scorer_names() -> list[str]:
    from palingenesis.rl.rewards import BUILTINS

    return sorted({*SCORERS, *BUILTINS})


def resolve_scorers(task: TaskConfig) -> list:
    """The task's score as RL Reward objects: its `scorer` (weight 1), or its weighted `rewards`."""
    from palingenesis.rl.rewards import BUILTINS, Reward, load_object

    def function(spec: str, where: str) -> Callable:
        if spec in SCORERS:
            return SCORERS[spec]
        if spec in BUILTINS:
            return BUILTINS[spec]
        if ":" not in spec:
            import difflib

            close = difflib.get_close_matches(spec, scorer_names(), n=1)
            raise EvalConfigError(
                f"task {task.name}: {where} {spec!r} is not a scorer"
                + (f" (did you mean {close[0]}?)" if close else "")
                + f"; built-in: {', '.join(scorer_names())}, or module:function"
            )
        try:
            return load_object(spec)
        except (ImportError, AttributeError, ValueError, OSError) as e:
            raise EvalConfigError(f"task {task.name}: {where} {spec!r}: {e}") from None

    if task.scorer:
        return [Reward(task.scorer.rpartition(":")[2], function(task.scorer, "scorer"), 1.0, dict(task.scorer_args))]
    rewards = []
    for name, entry in task.rewards.items():
        entry = {"fn": entry} if isinstance(entry, str) else dict(entry)
        fn = function(str(entry.get("fn")), f"rewards.{name}.fn")
        rewards.append(Reward(str(name), fn, float(entry.get("weight", 1.0)), dict(entry.get("args") or {})))
    return rewards


# ------------------------------------------------------------- judge validation


def is_judge(reward) -> bool:
    from palingenesis.rl.rewards import BUILTINS

    return any(reward.fn is BUILTINS[name] for name in JUDGES)


def validate_judge(reward, spec: dict, task: str = "") -> dict:
    """Run a judge scorer on labelled examples and compare its verdicts with the labels.

    spec (the task's judge_validation):
        file            JSONL of labelled examples: the scorer's inputs (prompt or question, completion, the
                        reference column) and a label
        label_field     the label's column (default "label"): a number or a boolean, the score a correct judge gives
        threshold       scores at or above it count as "pass" on both sides (default 0.5)
        min_agreement   the share of examples where judge and label agree (default 0.8)
        min_examples    fewer labelled examples than this is an error (default 20)
        skip            true: use the judge unvalidated (recorded in the summary)

    Returns {examples, agreement, kappa, mean_absolute_error, validated}; raises EvalConfigError below min_agreement.
    Agreement with too few labels is noise: min_examples guards the check itself."""
    from palingenesis.rl.rewards import SKIPPED

    where = f"task {task}: judge_validation"
    known = {"file", "label_field", "threshold", "min_agreement", "min_examples", "skip"}
    if set(spec) - known:
        raise EvalConfigError(f"{where}: unknown keys {sorted(set(spec) - known)} (known: {sorted(known)})")
    if spec.get("skip"):
        return {"validated": False, "skipped": True}
    if not spec.get("file"):
        raise EvalConfigError(
            f"{where}: an LLM judge must be checked against human labels before its scores mean anything: give "
            "`file` (JSONL: the judge's inputs plus a `label`) and `min_agreement`, or `skip: true` to use it "
            "unvalidated."
        )
    label_field = spec.get("label_field", "label")
    threshold = float(spec.get("threshold", 0.5))
    minimum = float(spec.get("min_agreement", 0.8))
    try:
        with open(spec["file"]) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as e:
        raise EvalConfigError(f"{where}: {spec['file']}: {e}") from None
    if len(rows) < int(spec.get("min_examples", 20)):
        raise EvalConfigError(
            f"{where}: {spec['file']} has {len(rows)} labelled examples, fewer than min_examples "
            f"({spec.get('min_examples', 20)}): agreement on so few is noise."
        )
    samples = []
    for i, row in enumerate(rows):
        if label_field not in row or "completion" not in row:
            raise EvalConfigError(f"{where}: {spec['file']}:{i + 1}: needs `completion` and {label_field!r}")
        prompt = row.get("prompt") or row.get("messages") or row.get("question") or ""
        prompt = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        samples.append({**row, "prompt": prompt, "messages": prompt, "reasoning": "", "finish_reason": "stop"})
    scores = asyncio.run(reward.score(samples))
    pairs = [(s, float(r[label_field])) for s, r in zip(scores, rows) if s is not SKIPPED and s is not None]
    if len(pairs) < len(rows) / 2:
        raise EvalConfigError(f"{where}: the judge scored {len(pairs)} of {len(rows)} examples (is it reachable?)")
    verdicts = [(s >= threshold, label >= threshold) for s, label in pairs]
    agreement = sum(a == b for a, b in verdicts) / len(verdicts)
    p_judge = sum(a for a, _ in verdicts) / len(verdicts)
    p_label = sum(b for _, b in verdicts) / len(verdicts)
    chance = p_judge * p_label + (1 - p_judge) * (1 - p_label)
    report = {
        "validated": True,
        "examples": len(pairs),
        "agreement": agreement,
        "kappa": (agreement - chance) / (1 - chance) if chance < 1 else 1.0,
        "mean_absolute_error": sum(abs(s - label) for s, label in pairs) / len(pairs),
        "min_agreement": minimum,
    }
    if agreement < minimum:
        raise EvalConfigError(
            f"{where}: the judge agrees with the labels on {agreement:.1%} of {len(pairs)} examples, below "
            f"min_agreement {minimum:.0%} (kappa {report['kappa']:.2f}): fix the judge prompt or model before "
            "trusting its scores."
        )
    return report
