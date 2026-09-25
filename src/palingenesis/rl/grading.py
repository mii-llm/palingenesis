"""Verifiers behind the built-in rewards: final answers of math problems, and code
against hidden tests.

Math: the answer is the last \\boxed{...} of the completion (else "Answer: ..." / "the
answer is ...", else its last number) and is compared with math_verify when it is
installed (symbolic equivalence), else by normalized text and numeric value.

Code: the program is the last fenced block of the final answer. Tests of every common
shape are normalized to three kinds:
  stdio   stdin in, stdout compared (APPS/TACO/CodeContests/LiveCodeBench stdin tests)
  call    a function (or Solution method) called with JSON arguments, return value
          compared (LeetCode / LiveCodeBench functional tests)
  assert  assert statements run against the program (MBPP / HumanEval / KodCode)
Programs only receive inputs: expected outputs stay here and are compared here. A call
or assert test counts only when its harness reports, after the call or the asserts, a
per-job nonce the program never sees, so exiting early (sys.exit(0), os._exit(0)) never
passes a test.
"""

import json
import math
import re
import secrets
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from palingenesis.rl.sandbox import ExecJob, ExecResult, Sandbox

# ------------------------------------------------------------------------ math


def last_boxed(text: str) -> str | None:
    """The content of the last \\boxed{...} (or \\fbox{...}), braces balanced."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


_ANSWER_IS = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*\$?([^\n$]+?)\$?\s*(?:\.|$)", re.I | re.M)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    stated = _ANSWER_IS.findall(text)
    if stated:
        return stated[-1].strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(
        r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}|\\textbf\{([^}]*)\}",
        lambda m: next(g for g in m.groups() if g is not None),
        s,
    )
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\ ", ""),
        ("^\\circ", ""),
        ("^{\\circ}", ""),
        ("\\%", ""),
        ("%", ""),
        ("\\$", ""),
        ("dfrac", "frac"),
    ):
        s = s.replace(old, new)
    s = re.sub(r"\s+", "", s).rstrip(".")
    s = re.sub(r"^[a-zA-Z]\w*=(?!=)", "", s)  # "x=3" -> "3"
    s = re.sub(r"^\(([A-Za-z])\)$", r"\1", s)  # "(B)" -> "B"
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _number(s: str) -> float | None:
    s = s.replace("(", "").replace(")", "")
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(prediction: str | None, gold: str) -> bool:
    """Whether a predicted final answer equals the gold one."""
    if prediction is None:
        return False
    try:
        from math_verify import parse, verify

        gold_parsed = parse(
            gold if "$" in gold or "\\boxed" in gold else f"${gold}$",
            parsing_timeout=None,
        )
        pred_parsed = parse(f"${prediction}$", parsing_timeout=None)
        if gold_parsed and pred_parsed and verify(gold_parsed, pred_parsed, timeout_seconds=None):
            return True
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 — math_verify on odd input: fall back to the text comparison
        pass
    a, b = _normalize(prediction), _normalize(str(gold))
    if a == b:
        return True
    x, y = _number(a), _number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


# ------------------------------------------------------------------------ code

_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(text: str, languages: tuple[str, ...] = ("python", "py", "python3", "")) -> str | None:
    """The last fenced code block in one of `languages` (untagged blocks count)."""
    blocks = [code for lang, code in _FENCE.findall(text) if lang.lower() in languages]
    return blocks[-1] if blocks else None


@dataclass
class CodeTest:
    kind: str  # stdio | call | assert
    input: Any  # stdin text | list of arguments | assert source
    expected: Any = None  # stdout text | return value | None
    fn_name: str | None = None


def load_tests(value: Any, fn_name: str | None = None, entry_point: str | None = None) -> list[CodeTest]:
    """Normalize a dataset's tests: a JSON string or object in APPS/TACO form
    ({"inputs", "outputs", "fn_name"?}), a list of {"input", "output", "testtype"?}
    (LiveCodeBench), a list of assert strings (MBPP), or test source (HumanEval, with
    `entry_point` for its check(candidate) function)."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                pass
        if isinstance(value, str):
            source = value + (f"\ncheck({entry_point})\n" if entry_point and "def check(" in value else "")
            return [CodeTest("assert", source)]
    if isinstance(value, dict):
        fn = value.get("fn_name") or fn_name
        inputs, outputs = value.get("inputs", []), value.get("outputs", [])
        if fn:
            return [CodeTest("call", _args(i), _json(o), fn) for i, o in zip(inputs, outputs)]
        return [CodeTest("stdio", _text(i), _text(o)) for i, o in zip(inputs, outputs)]
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return [CodeTest("assert", v) for v in value]
        tests = []
        for v in value:
            if v.get("testtype") == "functional" or fn_name:
                args = (
                    [json.loads(line) for line in str(v["input"]).splitlines() if line.strip()]
                    if isinstance(v["input"], str)
                    else _args(v["input"])
                )
                tests.append(CodeTest("call", args, _json(v["output"]), v.get("fn_name") or fn_name))
            else:
                tests.append(CodeTest("stdio", _text(v["input"]), _text(v["output"])))
        return tests
    raise ValueError(f"unrecognized tests of type {type(value).__name__}")


def _text(x: Any) -> str:
    return "\n".join(map(str, x)) if isinstance(x, list) else str(x)


def _json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def _args(x: Any) -> list:
    x = _json(x)
    return x if isinstance(x, list) else [x]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def outputs_match(got: str, expected: str) -> bool:
    """Judge-style stdout comparison: exact after strip, then per stripped non-empty line,
    then per whitespace token with numeric tokens compared at 1e-6."""
    if got.strip() == expected.strip():
        return True
    if _lines(got) == _lines(expected):
        return True
    a, b = got.split(), expected.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True


def values_match(got: Any, expected: Any) -> bool:
    if isinstance(got, float) or isinstance(expected, float):
        try:
            return math.isclose(float(got), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(got, list) and isinstance(expected, list):
        return len(got) == len(expected) and all(values_match(a, b) for a, b in zip(got, expected))
    if isinstance(got, dict) and isinstance(expected, dict):
        return got.keys() == expected.keys() and all(values_match(got[k], expected[k]) for k in got)
    return got == expected


_CALL_HARNESS = """import json, os
def _pgs_main():
    nonce, report = os.environ.pop("PGS_NONCE"), os.environ.pop("PGS_RESULT")
    with open("args.json") as f:
        args = json.load(f)
    os.remove("args.json")
    ns = {"__name__": "solution"}
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    fn = ns.get(%(fn)r)
    if fn is None and "Solution" in ns:
        fn = getattr(ns["Solution"](), %(fn)r)
    out = json.dumps(fn(*args))
    with open(report, "w") as f:
        f.write(nonce + out)
_pgs_main()
"""

_ASSERT_HARNESS = """import ast, operator, os
_type, _all, _bool = type, all, bool
_PLAIN = (int, float, complex, str, bytes, bool, type(None))
_OPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}

def _plain(x):
    t = _type(x)
    if t in _PLAIN:
        return True
    if t in (list, tuple, set, frozenset):
        return _all(_plain(v) for v in x)
    if t is dict:
        return _all(_plain(k) and _plain(v) for k, v in x.items())
    return False

def _pgs_compare(values, ops):
    for v in values:
        if not _plain(v):
            raise AssertionError("compared a value that is not a plain built-in: " + _type(v).__name__)
    return _all(_OPS[op](a, b) for op, a, b in zip(ops, values, values[1:]))

def _pgs_truth(v):
    if not _plain(v):
        raise AssertionError("asserted a value that is not a plain built-in: " + _type(v).__name__)
    return _bool(v)

class _Checked(ast.NodeTransformer):
    # every comparison and every assert goes through the checks above: returned objects
    # cannot fake a pass with __eq__, __bool__ or operator overloads
    def visit_Compare(self, node):
        self.generic_visit(node)
        ops = ast.List([ast.Constant(type(op).__name__) for op in node.ops], ast.Load())
        values = ast.List([node.left, *node.comparators], ast.Load())
        return ast.Call(ast.Name("_pgs_compare", ast.Load()), [values, ops], [])

    def visit_Assert(self, node):
        self.generic_visit(node)
        node.test = ast.Call(ast.Name("_pgs_truth", ast.Load()), [node.test], [])
        return node

_OPS = {k.__name__: v for k, v in _OPS.items()}          # keyed by the names the rewrite emits

def _pgs_main():
    nonce, report = os.environ.pop("PGS_NONCE"), os.environ.pop("PGS_RESULT")
    ns = {"__name__": "solution"}
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    tree = ast.fix_missing_locations(_Checked().visit(ast.parse(open("tests.py").read())))
    ns["_pgs_compare"], ns["_pgs_truth"] = _pgs_compare, _pgs_truth
    exec(compile(tree, "tests.py", "exec"), ns)
    with open(report, "w") as f:
        f.write(nonce)
_pgs_main()
"""

# Imports competitive-programming judges make available (LiveCodeBench / INTELLECT-3 parity).
PRELUDE = (
    "import sys, os, math, re, heapq, bisect, itertools, functools, collections, string, random\n"
    "from collections import *\nfrom itertools import *\nfrom functools import *\nfrom heapq import *\n"
    "from bisect import *\nfrom math import *\nfrom typing import *\nsys.setrecursionlimit(1 << 16)\n"
)


def _job(code: str, test: CodeTest, timeout: float, memory_mb: int, prelude: str) -> tuple[ExecJob, str]:
    nonce = secrets.token_hex(16)
    source = prelude + code
    if test.kind == "stdio":
        return (
            ExecJob(
                {"main.py": source},
                stdin=test.input,
                timeout=timeout,
                memory_mb=memory_mb,
            ),
            nonce,
        )
    files = {"solution.py": source}
    if test.kind == "call":
        files["main.py"] = _CALL_HARNESS % {"fn": test.fn_name}
        files["args.json"] = json.dumps(test.input)
    else:
        files["main.py"] = _ASSERT_HARNESS
        files["tests.py"] = test.input
    return (
        ExecJob(files, timeout=timeout, memory_mb=memory_mb, env={"PGS_NONCE": nonce}),
        nonce,
    )


def test_passed(test: CodeTest, result: ExecResult, nonce: str) -> bool:
    if result.status != "ok":
        return False
    if test.kind == "stdio":
        return outputs_match(result.stdout, str(test.expected))
    if not result.result.startswith(nonce):
        return False
    if test.kind == "assert":
        return result.result == nonce
    try:
        got = json.loads(result.result[len(nonce) :])
    except json.JSONDecodeError:
        return False
    return values_match(got, test.expected)


@dataclass
class CodeVerdict:
    reward: float | None  # None: the sandbox failed (not a wrong answer)
    passed: int
    run: int
    status: str  # first failure's status (or "passed" / "no_code")


async def grade_code(
    sandbox: Sandbox,
    code: str | None,
    tests: list[CodeTest],
    *,
    timeout: float = 6.0,
    memory_mb: int = 1024,
    max_tests: int = 15,
    partial: bool = False,
    wave: int = 4,
    prelude: str = PRELUDE,
) -> CodeVerdict:
    """Run `code` on up to `max_tests` tests (longest input first) and score it: 1.0 when
    all pass, else 0.0 (`partial`: the passing fraction). Binary grading stops at the first
    wave with a failure, so wrong programs cost a fraction of the tests."""
    if code is None:
        return CodeVerdict(0.0, 0, 0, "no_code")
    if not tests:
        return CodeVerdict(None, 0, 0, "no_tests")
    chosen = sorted(
        tests,
        key=lambda t: len(json.dumps(t.input)) if not isinstance(t.input, str) else len(t.input),
        reverse=True,
    )[:max_tests]
    passed = run = 0
    first_failure = "passed"
    for start in range(0, len(chosen), wave):
        batch = chosen[start : start + wave]
        jobs = [_job(code, t, timeout, memory_mb, prelude) for t in batch]
        results = await sandbox.run([j for j, _ in jobs])
        if any(r.infra_error for r in results):
            return CodeVerdict(None, passed, run, "sandbox_error")
        for test, (_, nonce), result in zip(batch, jobs, results):
            run += 1
            if test_passed(test, result, nonce):
                passed += 1
            elif first_failure == "passed":
                failed_check = result.status == "ok" or "AssertionError" in result.stderr
                first_failure = "wrong_answer" if failed_check else result.status
        if first_failure != "passed" and not partial:
            break
    if partial:
        return CodeVerdict(passed / len(chosen), passed, run, first_failure)
    return CodeVerdict(1.0 if passed == len(chosen) else 0.0, passed, run, first_failure)
