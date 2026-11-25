"""code-unit-tests and code-stdio: Python programs graded by hidden tests in a sandbox with resource limits.

Both are single-turn: the program is the last fenced Python block of the reply (palingenesis.rl.grading).

  code-unit-tests   rows {"prompt", "tests", "entry_point"?}. Tests that are calls with literal arguments and a
                    literal expected value ({"fn_name", "inputs", "outputs"}, or asserts of the form
                    `assert f(1, [2]) == 3`, as MBPP's) run as calls: the sandbox gets the arguments, returns the
                    values as JSON, and they are compared here, so nothing the program does inside the sandbox
                    (exiting early, an object whose __eq__ says yes, rewriting the harness) can turn a wrong value
                    into a pass. Any other test source (HumanEval's check(candidate), asserts on expressions) runs
                    through the library's assert harness, in the program's own process: sound against exits and
                    always-equal objects, not against a program written to rewrite that harness (see the guide).
  code-stdio        rows {"prompt", "tests": {"inputs": [...], "outputs": [...]}}: competitive-programming style.
                    Every input runs in the same sandbox job: the harness points the process's real stdin / stdout
                    (file descriptors 0 and 1) at a fresh file per test and executes the program again, so there
                    is one process per submission instead of one per test. Expected outputs never enter the
                    sandbox; the outputs come back and are compared here (rl.grading.outputs_match).

The sandbox is the RL config's (rollout code runs in the Docker pool by default: `pgs rl`, `pgs screen` and
`pgs env run --sandbox docker` pass it in). Without one, a shared process sandbox is used: the same worker and
limits (CPU time, address space, output size, open files, wall clock; the whole process group is killed), no
isolation from your files. That is for audits and trusted code; it is what `pgs env test` uses.
"""

from __future__ import annotations

import io
import json
import random
import secrets
import sys
import textwrap
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec

_shared_sandbox = None


def process_sandbox(slots: int = 0):
    """The shared process sandbox: one pool of slots for every environment instance of this process."""
    global _shared_sandbox
    if _shared_sandbox is None:
        import os

        from palingenesis.rl.sandbox import SubprocessSandbox

        _shared_sandbox = SubprocessSandbox(slots or min(8, os.cpu_count() or 2))
    return _shared_sandbox


def _reply(messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")


class _CodeEnv:
    tools = ()

    def __init__(self, sandbox=None, timeout: float = 6.0, memory_mb: int = 1024, max_tests: int = 20):
        self.sandbox = sandbox if sandbox is not None else process_sandbox()
        self.timeout, self.memory_mb, self.max_tests = timeout, memory_mb, max_tests
        self.row: dict[str, Any] = {}

    def reset(self, **row) -> None:
        self.row = row


def functional_tests(tests: Any, entry_point: str | None = None) -> tuple[str, list[list], list] | None:
    """(function name, argument lists, expected values) when every test is a call with literal arguments compared
    for equality with a literal, all JSON-plain; else None (the tests need the assert harness)."""
    import ast

    if isinstance(tests, str):
        text = tests.strip()
        if not text.startswith(("{", "[")):
            return None
        try:
            tests = json.loads(text)
        except json.JSONDecodeError:
            return None
    if isinstance(tests, dict):
        name = tests.get("fn_name") or entry_point
        if not name:
            return None
        inputs, outputs = tests.get("inputs", []), tests.get("outputs", [])
        if not all(isinstance(i, list) for i in inputs):
            return None
        return name, inputs, outputs
    if not (isinstance(tests, list) and tests and all(isinstance(t, str) for t in tests)):
        return None
    names, inputs, outputs = set(), [], []
    for source in tests:
        try:
            (node,) = ast.parse(source.strip()).body
            test = node.test
            if not (isinstance(node, ast.Assert) and isinstance(test, ast.Compare) and len(test.ops) == 1):
                return None
            call, (expected,) = test.left, test.comparators
            if not (isinstance(test.ops[0], ast.Eq) and isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                return None
            if call.keywords:
                return None
            arguments, value = [ast.literal_eval(a) for a in call.args], ast.literal_eval(expected)
            if json.loads(json.dumps(arguments)) != arguments:  # tuples, sets, non-string keys: not JSON-plain
                return None
            value = json.loads(json.dumps(value))  # a tuple is compared as the list JSON makes of it
        except (ValueError, SyntaxError, TypeError):
            return None
        names.add(call.func.id)
        inputs.append(arguments)
        outputs.append(value)
    return (names.pop(), inputs, outputs) if len(names) == 1 else None


_CALLS_HARNESS = """import json, os
def _pgs_main():
    nonce, report = os.environ.pop("PGS_NONCE"), os.environ.pop("PGS_RESULT")
    with open("calls.json") as f:
        calls = json.load(f)
    os.remove("calls.json")
    ns = {"__name__": "solution"}
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
    fn = ns.get(%(fn)r)
    if fn is None and "Solution" in ns:
        fn = getattr(ns["Solution"](), %(fn)r)
    outputs = []
    for args in calls:
        try:
            outputs.append(["ok", json.dumps(fn(*args))])
        except BaseException as error:
            outputs.append(["runtime_error", type(error).__name__])
        with open(report, "w") as f:
            f.write(nonce + json.dumps(outputs))
_pgs_main()
"""


async def _batched(sandbox, files: dict[str, str], n: int, timeout: float, memory_mb: int, max_output: int) -> tuple:
    """Run a harness that reports [[status, output], ...] for n tests in one job: (entries | None, job status)."""
    from palingenesis.rl.sandbox import ExecJob

    nonce = secrets.token_hex(16)
    job = ExecJob(
        files,
        timeout=min(120.0, timeout * n),
        memory_mb=memory_mb,
        max_output=max_output * (n + 1),
        env={"PGS_NONCE": nonce},
    )
    result = (await sandbox.run([job]))[0]
    if result.infra_error:
        return None, "sandbox_error"
    entries: list = []
    if result.result.startswith(nonce):
        try:
            entries = json.loads(result.result[len(nonce) :])
        except json.JSONDecodeError:
            entries = []
    entries = [e for e in entries if isinstance(e, list) and len(e) == 2] if isinstance(entries, list) else []
    return entries, result.status


async def grade_calls(
    sandbox,
    code: str | None,
    fn_name: str,
    inputs: list[list],
    expected: list,
    timeout: float = 6.0,
    memory_mb: int = 1024,
) -> tuple[float | None, int, str]:
    """(reward, tests passed, status) of a function on every argument list, in one sandbox job; the returned
    values come back as JSON and are compared here. reward None: the sandbox failed."""
    from palingenesis.rl.grading import PRELUDE, values_match

    if code is None:
        return 0.0, 0, "no_code"
    if not inputs:
        return None, 0, "no_tests"
    files = {
        "solution.py": PRELUDE + code,
        "main.py": _CALLS_HARNESS % {"fn": fn_name},
        "calls.json": json.dumps(inputs),
    }
    entries, status = await _batched(sandbox, files, len(inputs), timeout, memory_mb, 1 << 16)
    if entries is None:
        return None, 0, status
    passed = 0
    for (state, text), want in zip(entries, expected):
        try:
            passed += state == "ok" and values_match(json.loads(text), want)
        except (json.JSONDecodeError, TypeError):
            pass
    if passed == len(expected):
        return 1.0, passed, "passed"
    return 0.0, passed, status if status != "ok" else "wrong_answer"


class UnitTestEnv(_CodeEnv):
    async def get_reward(self, messages: list[dict]) -> dict[str, float]:
        from palingenesis.rl.grading import CodeTest, extract_code, grade_code, load_tests
        from palingenesis.rl.rewards import SkipSample

        code = extract_code(_reply(messages))
        entry_point = self.row.get("entry_point") or None
        calls = functional_tests(self.row["tests"], entry_point)
        if calls is not None and not self.row.get("test_imports"):
            name, inputs, outputs = calls
            pairs = list(zip(inputs, outputs))[: self.max_tests]
            reward, _, status = await grade_calls(
                self.sandbox, code, name, [i for i, _ in pairs], [o for _, o in pairs], self.timeout, self.memory_mb
            )
        else:
            tests = load_tests(self.row["tests"], entry_point=entry_point)
            if all(t.kind == "assert" for t in tests):  # one job for the whole suite
                imports = self.row.get("test_imports") or []
                tests = [CodeTest("assert", "\n".join([*imports, *(t.input for t in tests)]))]
            verdict = await grade_code(
                self.sandbox, code, tests, timeout=self.timeout, memory_mb=self.memory_mb, max_tests=self.max_tests
            )
            reward, status = verdict.reward, verdict.status
        if reward is None:
            raise SkipSample(f"code sandbox: {status}")
        return {"passed": reward, "metric/has_code": float(code is not None)}


_STDIO_HARNESS = """import json, os, sys
def _pgs_main():
    nonce, report = os.environ.pop("PGS_NONCE"), os.environ.pop("PGS_RESULT")
    with open("inputs.json") as f:
        inputs = json.load(f)
    os.remove("inputs.json")
    source = compile(open("solution.py").read(), "solution.py", "exec")
    stdin_fd, stdout_fd = os.dup(0), os.dup(1)
    outputs = []
    for text in inputs:
        with open(".in", "w") as f:
            f.write(text)
        fin = os.open(".in", os.O_RDONLY)
        fout = os.open(".out", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fin, 0), os.dup2(fout, 1), os.close(fin), os.close(fout)
        sys.stdin, sys.stdout = open(0, "r", closefd=False), open(1, "w", closefd=False)
        status = "ok"
        try:
            exec(source, {"__name__": "__main__", "__builtins__": __builtins__})
        except SystemExit as stop:
            status = "ok" if stop.code in (None, 0) else "runtime_error"
        except BaseException:
            status = "runtime_error"
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(stdin_fd, 0), os.dup2(stdout_fd, 1)
        with open(".out", errors="replace") as f:
            outputs.append([status, f.read(%(max_output)d)])
        with open(report, "w") as f:
            f.write(nonce + json.dumps(outputs))
_pgs_main()
"""


async def grade_stdio(
    sandbox,
    code: str | None,
    inputs: list[str],
    expected: list[str],
    timeout: float = 6.0,
    memory_mb: int = 1024,
    max_output: int = 1 << 16,
) -> tuple[float | None, int, str]:
    """(reward, tests passed, status) of a stdin/stdout program on every input, in one sandbox job. reward None:
    the sandbox failed. `timeout` is per test; the job gets their sum (at most 120 s)."""
    from palingenesis.rl.grading import PRELUDE, outputs_match

    if code is None:
        return 0.0, 0, "no_code"
    if not inputs:
        return None, 0, "no_tests"
    files = {
        "solution.py": PRELUDE + code,
        "main.py": _STDIO_HARNESS % {"max_output": max_output},
        "inputs.json": json.dumps(inputs),
    }
    entries, status = await _batched(sandbox, files, len(inputs), timeout, memory_mb, max_output)
    if entries is None:
        return None, 0, status
    passed = sum(state == "ok" and outputs_match(str(text), want) for (state, text), want in zip(entries, expected))
    if passed == len(expected):
        return 1.0, passed, "passed"
    return 0.0, passed, status if status != "ok" else "wrong_answer"


class StdioEnv(_CodeEnv):
    async def get_reward(self, messages: list[dict]) -> dict[str, float]:
        from palingenesis.rl.grading import extract_code
        from palingenesis.rl.rewards import SkipSample

        tests = self.row["tests"]
        if isinstance(tests, str):
            tests = json.loads(tests)
        pairs = sorted(zip(tests["inputs"], tests["outputs"]), key=lambda p: -len(str(p[0])))[: self.max_tests]
        code = extract_code(_reply(messages))
        reward, passed, status = await grade_stdio(
            self.sandbox, code, [str(i) for i, _ in pairs], [str(o) for _, o in pairs], self.timeout, self.memory_mb
        )
        if reward is None:
            raise SkipSample(f"code sandbox: {status}")
        return {
            "passed": reward,
            "metric/tests_passed": passed / len(pairs),
            "metric/has_code": float(code is not None),
        }


# ------------------------------------------------------- generator: unit tests
# Each task: a function name, a description, a reference solution (source) and an argument sampler.


def _words(rng: random.Random, n: int, alphabet: str = "abcdexyz") -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _unit_tasks(rng: random.Random) -> list[dict]:
    a, b = rng.sample([3, 4, 5, 6, 7, 9, 11], 2)
    p, q = rng.randint(1, 3), rng.randint(1, 3)
    a0, a1 = rng.randint(0, 4), rng.randint(1, 5)
    shift = rng.randint(1, 12)
    base = rng.choice([2, 3, 5, 7, 8])
    vowels = rng.choice(["aeiou", "aeiouy", "aei"])
    return [
        {
            "name": "sum_multiples",
            "text": f"Write a function `sum_multiples(n)` that returns the sum of all positive integers below `n` "
            f"that are divisible by {a} or by {b}.",
            "solution": f"def sum_multiples(n):\n    return sum(i for i in range(1, n) if i % {a} == 0 or i % {b} == 0)\n",
            "args": lambda: (rng.randint(5, 400),),
        },
        {
            "name": "sequence_term",
            "text": f"A sequence starts with s(0) = {a0} and s(1) = {a1}, and for n >= 2, "
            f"s(n) = {p} * s(n - 1) + {q} * s(n - 2). Write a function `sequence_term(n)` that returns s(n).",
            "solution": f"def sequence_term(n):\n    x, y = {a0}, {a1}\n    for _ in range(n):\n"
            f"        x, y = y, {p} * y + {q} * x\n    return x\n",
            "args": lambda: (rng.randint(0, 25),),
        },
        {
            "name": "shift_letters",
            "text": f"Write a function `shift_letters(text)` that shifts every lowercase letter a-z forward by "
            f"{shift} positions in the alphabet, wrapping around after z, and leaves every other character unchanged.",
            "solution": "def shift_letters(text):\n    return ''.join(\n"
            f"        chr((ord(c) - 97 + {shift}) % 26 + 97) if 'a' <= c <= 'z' else c for c in text\n    )\n",
            "args": lambda: (_words(rng, rng.randint(3, 12), "abcxyz QZ-1"),),
        },
        {
            "name": "to_base",
            "text": f"Write a function `to_base(n)` that returns the base-{base} representation of the non-negative "
            "integer `n` as a string of digits (\"0\" for 0).",
            "solution": f"def to_base(n):\n    if n == 0:\n        return '0'\n    digits = ''\n    while n:\n"
            f"        digits = str(n % {base}) + digits\n        n //= {base}\n    return digits\n",
            "args": lambda: (rng.choice([0, rng.randint(1, 5000)]),),
        },
        {
            "name": "count_vowels",
            "text": f"Write a function `count_vowels(text)` that returns how many characters of `text` are one of "
            f"the letters \"{vowels}\", in either case.",
            "solution": f"def count_vowels(text):\n    return sum(c in '{vowels}' for c in text.lower())\n",
            "args": lambda: (_words(rng, rng.randint(4, 30), "aeiouyAEIOUYbcdfg "),),
        },
        {
            "name": "run_lengths",
            "text": "Write a function `run_lengths(text)` that returns the run-length encoding of `text` as a list "
            "of `[character, count]` pairs, in order. For example `run_lengths(\"aaxbb\")` is "
            "`[[\"a\", 2], [\"x\", 1], [\"b\", 2]]`.",
            "solution": "def run_lengths(text):\n    out = []\n    for c in text:\n        if out and out[-1][0] == c:\n"
            "            out[-1][1] += 1\n        else:\n            out.append([c, 1])\n    return out\n",
            "args": lambda: ("".join(rng.choice("abc") * rng.randint(1, 4) for _ in range(rng.randint(1, 6))),),
        },
        {
            "name": "merge_intervals",
            "text": "Write a function `merge_intervals(intervals)` that takes a list of `[start, end]` integer "
            "intervals (closed, in any order) and returns the merged intervals sorted by start; intervals that "
            "overlap or touch (`[1, 3]` and `[3, 5]`) are merged.",
            "solution": "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n"
            "        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n"
            "            out.append([s, e])\n    return out\n",
            "args": lambda: (
                [[s, s + rng.randint(0, 6)] for s in (rng.randint(0, 30) for _ in range(rng.randint(1, 7)))],
            ),
        },
        {
            "name": "rotate_left",
            "text": f"Write a function `rotate_left(items)` that returns a new list with the items of `items` "
            f"rotated left by {shift} positions (rotating by the list's length gives the list back; an empty list "
            "stays empty).",
            "solution": f"def rotate_left(items):\n    if not items:\n        return []\n    k = {shift} % len(items)\n"
            "    return items[k:] + items[:k]\n",
            "args": lambda: ([rng.randint(0, 99) for _ in range(rng.randint(0, 9))],),
        },
    ]


def generate_unit_tests(n: int, seed: int = 0, tests_per_task: int = 8) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    while len(rows) < n:
        for task in _unit_tasks(rng):
            if len(rows) >= n:
                break
            namespace: dict[str, Any] = {}
            exec(task["solution"], namespace)  # noqa: S102 — the reference solutions above
            fn = namespace[task["name"]]
            cases, seen = [], set()
            for _ in range(200):
                args = task["args"]()
                key = json.dumps(args)
                if key not in seen:
                    seen.add(key)
                    cases.append((args, fn(*json.loads(key))))
                if len(cases) > tests_per_task:
                    break
            if len({json.dumps(out) for _, out in cases}) < 3:
                continue  # a constant function must not pass
            (shown_args, shown_out), hidden = cases[0], cases[1:]
            example = f"{task['name']}({', '.join(map(repr, shown_args))}) == {shown_out!r}"
            rows.append(
                {
                    "id": f"code-unit-tests/synthetic/{seed}/{len(rows)}",
                    "prompt": f"{task['text']}\n\nExample: `{example}`\n\n"
                    "Reply with the complete function in one ```python code block.",
                    "tests": {
                        "fn_name": task["name"],
                        "inputs": [list(args) for args, _ in hidden],
                        "outputs": [out for _, out in hidden],
                    },
                    "entry_point": task["name"],
                    "example_output": repr(shown_out),
                    "solution": task["solution"],
                    "task": task["name"],
                }
            )
    return rows


def from_mbpp(row: dict) -> dict | None:
    """google-research-datasets/mbpp (config sanitized): the first assert is shown, all of them are run."""
    tests = list(row.get("test_list") or [])
    if not tests or not row.get("code"):
        return None
    return {
        "prompt": f"{row.get('prompt') or row.get('text')}\n\nYour code must pass this test:\n{tests[0]}\n\n"
        "Reply with the complete code in one ```python code block.",
        "tests": tests,
        "test_imports": list(row.get("test_imports") or []),
        "solution": row["code"],
    }


def from_humaneval(row: dict) -> dict | None:
    """openai/openai_humaneval: complete the function; the row's check(candidate) is the test."""
    return {
        "prompt": "Complete this Python function. Reply with the complete function, its signature included, in one "
        f"```python code block.\n\n```python\n{row['prompt']}```",
        "tests": row["test"],
        "entry_point": row["entry_point"],
        "solution": row["prompt"] + row["canonical_solution"],
    }


# ------------------------------------------------------------ generator: stdio


def _stdio_tasks(rng: random.Random) -> list[dict]:
    mod = rng.choice([97, 1000, 1009, 10007])
    k = rng.randint(1, 6)
    letter = rng.choice("abc")

    def numbers(lo: int, hi: int, n_max: int = 12) -> str:
        values = [rng.randint(lo, hi) for _ in range(rng.randint(1, n_max))]
        return f"{len(values)}\n{' '.join(map(str, values))}\n"

    return [
        {
            "name": "sum_mod",
            "text": f"The first line contains an integer n. The second line contains n integers. Print the sum of "
            f"their squares modulo {mod}.",
            "solution": f"n = int(input())\nvalues = list(map(int, input().split()))\n"
            f"print(sum(v * v for v in values) % {mod})\n",
            "input": lambda: numbers(-1000, 1000),
        },
        {
            "name": "pairs_with_difference",
            "text": f"The first line contains an integer n. The second line contains n integers. Print the number "
            f"of pairs of positions i < j whose values differ by exactly {k} (in absolute value).",
            "solution": "n = int(input())\nvalues = list(map(int, input().split()))\n"
            f"print(sum(abs(values[i] - values[j]) == {k} for i in range(n) for j in range(i + 1, n)))\n",
            "input": lambda: numbers(0, 12, 14),
        },
        {
            "name": "longest_run",
            "text": f"The input is one line with a string of lowercase letters. Print two values separated by a "
            f"space: the length of the longest run of consecutive '{letter}' characters, and the number of "
            "maximal runs (of any letter) in the string.",
            "solution": "import sys\ns = sys.stdin.readline().strip()\nbest = current = 0\nruns = 0\nprevious = ''\n"
            f"for c in s:\n    if c != previous:\n        runs += 1\n        current = 0\n    if c == '{letter}':\n"
            "        current += 1\n        best = max(best, current)\n    previous = c\nprint(best, runs)\n",
            "input": lambda: "".join(rng.choice("abc") * rng.randint(1, 5) for _ in range(rng.randint(1, 8))) + "\n",
        },
        {
            "name": "range_sums",
            "text": "The first line contains integers n and q. The second line contains n integers. Each of the "
            "next q lines contains two integers l and r (1-based, l <= r). For each query print, on its own line, "
            "the sum of the values at positions l through r.",
            "solution": "import sys\ndata = sys.stdin.buffer.read().split()\nn, q = int(data[0]), int(data[1])\n"
            "prefix = [0]\nfor v in data[2:2 + n]:\n    prefix.append(prefix[-1] + int(v))\nout = []\n"
            "for i in range(q):\n    l, r = int(data[2 + n + 2 * i]), int(data[3 + n + 2 * i])\n"
            "    out.append(prefix[r] - prefix[l - 1])\nprint('\\n'.join(map(str, out)))\n",
            "input": lambda: _range_queries(rng),
        },
    ]


def _range_queries(rng: random.Random) -> str:
    n, q = rng.randint(1, 10), rng.randint(1, 5)
    lines = [f"{n} {q}", " ".join(str(rng.randint(-50, 50)) for _ in range(n))]
    for _ in range(q):
        left = rng.randint(1, n)
        lines.append(f"{left} {rng.randint(left, n)}")
    return "\n".join(lines) + "\n"


def run_reference(source: str, stdin: str) -> str:
    """Output of a reference program (the generator's own, trusted) on one input, in this process."""
    stdin_, stdout_ = sys.stdin, sys.stdout
    sys.stdin = io.TextIOWrapper(io.BytesIO(stdin.encode()))
    sys.stdout = captured = io.StringIO()
    try:
        exec(compile(source, "reference", "exec"), {"__name__": "__main__"})  # noqa: S102
    finally:
        sys.stdin, sys.stdout = stdin_, stdout_
    return captured.getvalue()


def generate_stdio(n: int, seed: int = 0, tests_per_task: int = 6) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    while len(rows) < n:
        for task in _stdio_tasks(rng):
            if len(rows) >= n:
                break
            inputs = list(dict.fromkeys(task["input"]() for _ in range(60)))[: tests_per_task + 1]
            outputs = [run_reference(task["solution"], text) for text in inputs]
            hidden = [(i, o) for i, o in zip(inputs[1:], outputs[1:]) if o != outputs[0]]
            if len({o for _, o in hidden}) < 2 or any(i.strip() == o.strip() for i, o in hidden):
                continue  # the sample's output, a constant or the input itself must not pass
            rows.append(
                {
                    "id": f"code-stdio/synthetic/{seed}/{len(rows)}",
                    "prompt": f"{task['text']}\n\nExample input:\n```\n{inputs[0]}```\nExample output:\n```\n"
                    f"{outputs[0]}```\n\nWrite a Python program that reads standard input and prints the answer. "
                    "Reply with the complete program in one ```python code block.",
                    "tests": {"inputs": [i for i, _ in hidden], "outputs": [o for _, o in hidden]},
                    "example_output": outputs[0],
                    "solution": task["solution"],
                    "task": task["name"],
                }
            )
    return rows


# ------------------------------------------------------------------- policies


def _block(code: str) -> str:
    return f"Here is the solution.\n\n```python\n{code.rstrip()}\n```"


def oracle(row: dict, messages: list[dict]) -> str:
    return _block(row["solution"])


def exit_zero(row: dict, messages: list[dict]) -> str:
    return _block("import sys\nsys.exit(0)")


def os_exit(row: dict, messages: list[dict]) -> str:
    return _block("import os\nos._exit(0)")


def no_code(row: dict, messages: list[dict]) -> str:
    return "The function is straightforward: it loops over the input and returns the result."


_FORGE = textwrap.dedent(
    """
    import glob, os, sys
    for path in [os.environ.get("PGS_RESULT", "")] + glob.glob(".result*") + glob.glob("*result*"):
        try:
            open(path, "w").write(os.environ.get("PGS_NONCE", "") + "[]")
        except OSError:
            pass
    """
)


def forge_report(row: dict, messages: list[dict]) -> str:
    """Writes to whatever looks like the harness's result file, then defines nothing useful."""
    name = row.get("entry_point") or "solve"
    return _block(f"{_FORGE}\ndef {name}(*args, **kwargs):\n    return None")


def stub_none(row: dict, messages: list[dict]) -> str:
    return _block(f"def {row.get('entry_point') or 'solve'}(*args, **kwargs):\n    return None")


_REWRITE = textwrap.dedent(
    """
    import os, sys
    import __main__
    for name in list(vars(__main__)):
        if name.startswith("_pgs") and name != "_pgs_main":
            setattr(__main__, name, lambda *args, **kwargs: True)
    frame = sys._getframe()
    while frame is not None:
        found = frame.f_locals
        if "nonce" in found and "report" in found:
            with open(found["report"], "w") as f:
                f.write(found["nonce"])
        frame = frame.f_back
    """
)


def rewrite_harness(row: dict, messages: list[dict]) -> str:
    """Replaces the harness's checks from inside the program and reports success with the harness's own nonce."""
    return _block(f"{_REWRITE}\ndef {row.get('entry_point') or 'solve'}(*args, **kwargs):\n    return None")


def always_equal(row: dict, messages: list[dict]) -> str:
    """Returns an object that claims to equal anything."""
    name = row.get("entry_point") or "solve"
    return _block(
        "class Anything:\n    def __eq__(self, other):\n        return True\n    def __ne__(self, other):\n"
        f"        return False\n    def __bool__(self):\n        return True\n\ndef {name}(*args, **kwargs):\n"
        "    return Anything()"
    )


def hardcode_example(row: dict, messages: list[dict]) -> str:
    """The shown example's output, whatever the input."""
    if "entry_point" in row:
        return _block(f"def {row['entry_point']}(*args, **kwargs):\n    return {row['example_output']}")
    return _block(f"print({row['example_output']!r}, end='')")


def echo_input(row: dict, messages: list[dict]) -> str:
    return _block("import sys\nsys.stdout.write(sys.stdin.read())")


def read_harness_files(row: dict, messages: list[dict]) -> str:
    """Prints every file it can find next to the program (the tests' inputs, if the harness left them)."""
    return _block(
        "import os\nfor name in os.listdir('.'):\n    try:\n        print(open(name).read())\n"
        "    except Exception:\n        pass"
    )


_CODE_LICENSE = "Apache-2.0"

UNIT_TESTS = EnvSpec(
    name="code-unit-tests",
    version="1.0.0",
    family="code",
    summary="Write a Python function; hidden unit tests run in a sandbox with resource limits.",
    env=UnitTestEnv,
    license=_CODE_LICENSE,
    verifier=(
        "The last fenced Python block passes every hidden test, run in one sandbox job (CPU, memory, output and "
        "wall-clock limits). Call-shaped tests return their values as JSON and are compared outside the sandbox; "
        "other test source runs through the library's assert harness (plain-value comparisons, a nonce after the "
        "last assert), so exiting early or returning an always-equal object scores 0."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate_unit_tests,
            license="Apache-2.0",
            description="Parametrized function tasks with a reference solution and generated asserts.",
        ),
        "mbpp": DatasetSpec(
            hub_id="google-research-datasets/mbpp",
            config="sanitized",
            split="test",
            convert=from_mbpp,
            license="CC-BY-4.0",
            url="https://huggingface.co/datasets/google-research-datasets/mbpp",
            citation="Austin et al., Program Synthesis with Large Language Models, 2021 (arXiv:2108.07732)",
            description="MBPP sanitized (427 hand-verified problems; the row's asserts are the tests).",
        ),
        "humaneval": DatasetSpec(
            hub_id="openai/openai_humaneval",
            split="test",
            convert=from_humaneval,
            license="MIT",
            url="https://huggingface.co/datasets/openai/openai_humaneval",
            citation="Chen et al., Evaluating Large Language Models Trained on Code, 2021 (arXiv:2107.03374)",
            description="HumanEval (164 function-completion problems).",
        ),
    },
    sandbox="process",
    max_turns=1,
    max_new_tokens=1536,
    oracle=oracle,
    degenerate={
        "exit_zero": exit_zero,
        "os_exit": os_exit,
        "no_code": no_code,
        "stub_none": stub_none,
        "always_equal": always_equal,
        "hardcode_example": hardcode_example,
        "forge_report": forge_report,
        "rewrite_harness": rewrite_harness,
    },
    tags=("single-turn", "python"),
    notes="The process sandbox limits resources but does not isolate: train with sandbox.backend docker.",
)

STDIO = EnvSpec(
    name="code-stdio",
    version="1.0.0",
    family="code",
    summary="Competitive-programming style: a Python program reads stdin and prints the answer; hidden tests.",
    env=StdioEnv,
    license=_CODE_LICENSE,
    verifier=(
        "The last fenced Python block prints the expected output for every hidden input (judge-style comparison: "
        "whitespace and float tolerant). All inputs run in one sandbox job; expected outputs never enter the "
        "sandbox, and the shown example is not among the hidden tests."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate_stdio,
            license="Apache-2.0",
            description="Parametrized stdin/stdout tasks with a reference program and generated tests.",
        ),
    },
    sandbox="process",
    max_turns=1,
    max_new_tokens=1536,
    oracle=oracle,
    degenerate={
        "exit_zero": exit_zero,
        "os_exit": os_exit,
        "no_code": no_code,
        "hardcode_example": hardcode_example,
        "echo_input": echo_input,
        "read_harness_files": read_harness_files,
        "forge_report": forge_report,
        "rewrite_harness": rewrite_harness,
    },
    tags=("single-turn", "python"),
    notes=(
        "Bring your own problems as rows {prompt, tests: {inputs, outputs}} (the APPS / TACO shape). The process "
        "sandbox limits resources but does not isolate: train with sandbox.backend docker."
    ),
)
