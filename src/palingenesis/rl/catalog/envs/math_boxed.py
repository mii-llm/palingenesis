"""math-boxed: problems with one exact final answer in \\boxed{}, compared numerically or symbolically.

Rows: {"prompt", "answer", "form"}; form is how the answer must be written, which is what closes the hole every
symbolic checker has (the unevaluated question is equivalent to its answer):

  number       a number literal (integer, decimal, fraction, \\frac{a}{b}); compared exactly as a rational
  tuple        numbers separated by commas, in order: (2, -3)
  expanded     an expression without parentheses, compared with sympy (simplify(a - b) == 0)
  expression   any expression, compared with sympy (use only when the question is not itself an answer)

Only the last \\boxed{} of the final reply counts: a reply cannot list several candidates. Number forms are decided
in-process (exact rational arithmetic); the sympy comparison runs in the verifier process pool with a time limit, on
text restricted to arithmetic characters and a fixed set of names before it reaches sympy's parser.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec

SUFFIX = "\n\nGive the final answer in \\boxed{}."

_NUMBER = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)")
_THOUSANDS = re.compile(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?")
_ALLOWED_TEXT = re.compile(r"[0-9a-zA-Z+\-*/()., ]*")
_NAMES = {"pi", "sqrt", "sin", "cos", "tan", "log", "exp"}


def normalize(text: str) -> str:
    """LaTeX answer text as plain arithmetic: \\frac{a}{b} -> (a)/(b), ^{n} -> **(n), \\cdot -> *, ..."""
    s = text.strip().strip("$").strip()
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", s)
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\ ", " "),
        ("\\%", ""),
        ("%", ""),
        ("\\$", ""),
        ("\\cdot", "*"),
        ("\\times", "*"),
        ("\\div", "/"),
        ("\\pi", "pi"),
        ("−", "-"),
    ):
        s = s.replace(old, new)
    previous = None
    while previous != s:  # nested fractions and roots, innermost first
        previous = s
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = s.replace("^", "**")
    s = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", s)  # "x = 3" -> "3"
    return s.strip().rstrip(".").strip()


def as_rational(text: str) -> Fraction | None:
    """A number literal as an exact rational, or None when the text is anything else (an expression)."""
    s = text.replace(" ", "")
    if _THOUSANDS.fullmatch(s):
        s = s.replace(",", "")
    while s.startswith("(") and s.endswith(")") and _balanced(s[1:-1]):
        s = s[1:-1]
    sign = 1
    if s.startswith("-(") and s.endswith(")") and _balanced(s[2:-1]):
        sign, s = -1, s[2:-1]
    parts = s.split("/")
    if len(parts) > 2:
        return None
    values = []
    for part in parts:
        while part.startswith("(") and part.endswith(")") and _balanced(part[1:-1]):
            part = part[1:-1]
        if not _NUMBER.fullmatch(part):
            return None
        values.append(Fraction(part))
    if len(values) == 2:
        if values[1] == 0:
            return None
        return sign * values[0] / values[1]
    return sign * values[0]


def _balanced(s: str) -> bool:
    depth = 0
    for c in s:
        depth += (c == "(") - (c == ")")
        if depth < 0:
            return False
    return depth == 0


def split_top_level(text: str) -> list[str]:
    s = text.strip()
    if s[:1] in "([" and s[-1:] in ")]" and _balanced(s[1:-1].replace("[", "(").replace("]", ")")):
        s = s[1:-1]
    parts, depth, current = [], 0, ""
    for c in s:
        depth += (c in "([") - (c in ")]")
        if c == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += c
    return [p.strip() for p in parts + [current]]


def safe_expression(text: str) -> bool:
    """Whether normalized answer text may go to sympy's parser (which evaluates Python): arithmetic characters,
    known names and single letters only, short numbers, small literal exponents."""
    if len(text) > 200 or not _ALLOWED_TEXT.fullmatch(text) or "__" in text:
        return False
    for name in re.findall(r"[A-Za-z_]+", text):
        if len(name) > 1 and name not in _NAMES:
            return False
    if any(len(digits) > 15 for digits in re.findall(r"\d+", text)):
        return False
    if text.count("**") > 6 or re.search(r"\*\*[\s(]*-?\d+[\s)]*\*\*", text):
        return False  # power towers
    for exponent in re.finditer(r"\*\*\s*\(?\s*(-?\d+)?", text):
        if exponent.group(1) is None or abs(int(exponent.group(1))) > 64:
            return False
    return True


def symbolic_equal(prediction: str, gold: str) -> bool:
    """simplify(prediction - gold) == 0 on normalized text; runs in a verifier worker (it can be slow)."""
    if not (safe_expression(prediction) and safe_expression(gold)):
        return False
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    names: dict[str, Any] = {name: getattr(sympy, name) for name in _NAMES}
    for text in (prediction, gold):
        for letter in set(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", text)):
            names.setdefault(letter, sympy.Symbol(letter))
    try:
        a = parse_expr(prediction, local_dict=names, transformations=transformations)
        b = parse_expr(gold, local_dict=names, transformations=transformations)
        return bool(sympy.simplify(a - b) == 0)
    except Exception:  # noqa: BLE001 — text that is not an expression is a wrong answer
        return False


def decide_fast(prediction: str, gold: str, form: str) -> bool | None:
    """The verdict when no symbolic comparison is needed, else None. Arguments are raw (LaTeX) answer texts."""
    p, g = normalize(prediction), normalize(gold)
    if form == "number":
        a, b = as_rational(p), as_rational(g)
        if b is None:
            return p.replace(" ", "") == g.replace(" ", "")
        return a is not None and a == b
    if form == "tuple":
        ps, gs = split_top_level(p), split_top_level(g)
        if len(ps) != len(gs):
            return False
        pairs = [(as_rational(x), as_rational(y)) for x, y in zip(ps, gs)]
        return all(x is not None and x == y for x, y in pairs)
    if form == "expanded" and ("(" in re.sub(r"\*\*\(-?\d+\)", "", p)):
        return False  # the factored question is not its own answer
    if p.replace(" ", "") == g.replace(" ", ""):
        return True
    return None


async def answers_equal(prediction: str | None, gold: str, form: str = "number", timeout: float = 3.0) -> bool:
    if prediction is None or not prediction.strip():
        return False
    fast = decide_fast(prediction, gold, form)
    if fast is not None:
        return fast
    from palingenesis.rl.catalog.verify import VerifyTimeout, verify

    try:
        return bool(await verify(symbolic_equal, normalize(prediction), normalize(gold), timeout=timeout))
    except (VerifyTimeout, RuntimeError):
        return False


class MathEnv:
    """Single-turn: the reply's last \\boxed{} against the row's answer."""

    tools = ()

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.answer, self.form = "", "number"

    def reset(self, answer: Any = "", form: str = "number", **row) -> None:
        self.answer, self.form = str(answer), form

    async def get_reward(self, messages: list[dict]) -> dict[str, float]:
        from palingenesis.rl.grading import last_boxed

        reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        boxed = last_boxed(reply)
        correct = await answers_equal(boxed, self.answer, self.form, self.timeout)
        return {"correct": float(correct), "metric/boxed": float(boxed is not None)}


# ------------------------------------------------------------------ generator


def _latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def _plain(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _polynomial(coefficients: list[int], reverse: bool = False, braces: bool = False) -> str:
    """LaTeX of a polynomial from its coefficients, highest degree first."""
    degree = len(coefficients) - 1
    terms = []
    for i, c in enumerate(coefficients):
        power = degree - i
        if c == 0:
            continue
        magnitude = abs(c)
        body = "" if power == 0 else "x" if power == 1 else f"x^{{{power}}}" if braces else f"x^{power}"
        text = f"{magnitude}{body}" if (magnitude != 1 or power == 0) else body
        terms.append(("-" if c < 0 else "+", text))
    if reverse:
        terms.reverse()
    out = ""
    for k, (sign, text) in enumerate(terms):
        out += (("-" if sign == "-" else "") if k == 0 else f" {sign} ") + text
    return out or "0"


def _arithmetic(rng: random.Random) -> dict:
    while True:
        a, b, c, d = (rng.randint(2, 60) for _ in range(4))
        text, value = rng.choice(
            [
                (f"{a} \\times {b} + {c}", a * b + c),
                (f"{a} \\times {b} - {c} \\times {d}", a * b - c * d),
                (f"({a} + {b}) \\times {c} - {d}", (a + b) * c - d),
                (f"{a * b} \\div {b} + {c} \\times {d}", a + c * d),
            ]
        )
        if abs(value) > 9 and str(abs(value)) not in re.findall(r"\d+", text):
            answer = Fraction(value)
            return {"question": f"Compute ${text}$.", "expression": text, "answer": answer, "form": "number"}


def _fractions(rng: random.Random) -> dict:
    while True:
        a, b = Fraction(rng.randint(1, 9), rng.randint(2, 12)), Fraction(rng.randint(1, 9), rng.randint(2, 12))
        op = rng.choice("+-*")
        value = a + b if op == "+" else a - b if op == "-" else a * b
        symbol = {"+": "+", "-": "-", "*": "\\times"}[op]
        text = f"{_latex(a)} {symbol} {_latex(b)}"
        if value.denominator > 1 and a.denominator > 1 and b.denominator > 1 and value not in (a, b):
            question = f"Compute ${text}$ and give the result as a fraction in lowest terms."
            return {"question": question, "expression": text, "answer": value, "form": "number"}


def _linear(rng: random.Random) -> dict:
    while True:
        a, c = rng.randint(2, 12), rng.randint(-9, 9)
        b, d = rng.randint(-30, 30), rng.randint(-30, 30)
        if a == c or c == 0 or b == 0:
            continue
        value = Fraction(d - b, a - c)
        if value in (0, 1, -1):
            continue

        def side(k: int, m: int) -> str:
            return f"{k}x {'+' if m >= 0 else '-'} {abs(m)}"

        text = f"{side(a, b)} = {side(c, d)}"
        question = f"Solve for $x$: ${text}$. Give $x$ as an integer or a fraction in lowest terms."
        return {"question": question, "expression": text, "answer": value, "form": "number"}


def _percent(rng: random.Random) -> dict:
    while True:
        price, off, qty = rng.randrange(20, 400, 5), rng.choice([10, 15, 20, 25, 30, 40]), rng.randint(2, 9)
        value = Fraction(price * qty * (100 - off), 100)
        question = (
            f"A shop sells a jacket for {price} euros. During a sale every jacket is {off}% off. "
            f"How many euros do {qty} jackets cost during the sale?"
        )
        if str(value.numerator) not in (str(price), str(qty), str(off)):
            return {"question": question, "expression": "", "answer": value, "form": "number"}


def _expand(rng: random.Random) -> dict:
    a, c = rng.randint(1, 6), rng.randint(1, 6)
    b, d = rng.choice([-1, 1]) * rng.randint(1, 9), rng.choice([-1, 1]) * rng.randint(1, 9)
    text = f"({_polynomial([a, b])})({_polynomial([c, d])})"
    coefficients = [a * c, a * d + b * c, b * d]
    return {
        "question": f"Expand and simplify ${text}$.",
        "expression": text,
        "answer": _polynomial(coefficients),
        "solution_answer": _polynomial(coefficients, reverse=True, braces=True),
        "form": "expanded",
    }


def _system(rng: random.Random) -> dict:
    while True:
        x, y = rng.randint(-9, 9), rng.randint(-9, 9)
        a, b, c, d = (rng.choice([-1, 1]) * rng.randint(1, 7) for _ in range(4))
        if a * d - b * c == 0 or x == y or 0 in (x, y):
            continue

        def equation(k: int, m: int) -> str:
            return f"{k}x {'+' if m >= 0 else '-'} {abs(m)}y = {k * x + m * y}"

        text = f"{equation(a, b)}, \\quad {equation(c, d)}"
        question = f"Solve the system ${text}$. Give the solution as the ordered pair $(x, y)$."
        return {
            "question": question,
            "expression": text,
            "answer": f"({x}, {y})",
            "solution_answer": f"\\left({x},\\ {y}\\right)",
            "form": "tuple",
        }


LEVELS = {1: (_arithmetic, _percent), 2: (_fractions, _linear), 3: (_expand, _system)}


def generate(n: int, seed: int = 0, difficulty: int = 0) -> list[dict]:
    """`n` problems; difficulty 1 (integer arithmetic, percentages), 2 (fractions, linear equations), 3 (polynomial
    expansion, 2x2 systems), 0: all of them in turn."""
    rng = random.Random(seed)
    makers = LEVELS[difficulty] if difficulty else [m for level in LEVELS.values() for m in level]
    rows = []
    for i in range(n):
        maker = makers[i % len(makers)]
        task = maker(rng)
        answer = task["answer"]
        level = next(level for level, members in LEVELS.items() if maker in members)
        rows.append(
            {
                "id": f"math-boxed/synthetic/{seed}/{i}",
                "prompt": task["question"] + SUFFIX,
                "answer": _plain(answer) if isinstance(answer, Fraction) else answer,
                "form": task["form"],
                "expression": task["expression"],
                "solution_answer": task.get("solution_answer")
                or (_latex(answer) if isinstance(answer, Fraction) else answer),
                "difficulty": level,
                "task": maker.__name__.lstrip("_"),
            }
        )
    return rows


def from_gsm8k(row: dict) -> dict | None:
    """openai/gsm8k: the number after "####" is the answer."""
    _, marker, answer = str(row.get("answer", "")).rpartition("####")
    answer = answer.strip().replace(",", "")
    if not marker or as_rational(answer) is None:
        return None
    return {"prompt": row["question"].strip() + SUFFIX, "answer": answer, "form": "number", "solution_answer": answer}


# ------------------------------------------------------------------- policies


def oracle(row: dict, messages: list[dict]) -> str:
    return f"Working it out step by step, the result is \\boxed{{{row.get('solution_answer', row['answer'])}}}."


def box_the_question(row: dict, messages: list[dict]) -> str:
    """The unevaluated expression of the question: equivalent to the answer for a naive symbolic checker."""
    return f"\\boxed{{{row.get('expression') or row['prompt']}}}"


def box_prompt_numbers(row: dict, messages: list[dict]) -> str:
    return " ".join(f"\\boxed{{{number}}}" for number in re.findall(r"-?\d+(?:\.\d+)?", row["prompt"])) or "\\boxed{0}"


def box_zero(row: dict, messages: list[dict]) -> str:
    return "\\boxed{0}"


def unboxed_gold(row: dict, messages: list[dict]) -> str:
    """Candidates listed without committing: several boxes, the gold among them but not last."""
    return f"It is \\boxed{{{row['answer']}}} or maybe \\boxed{{12345}}."


SPEC = EnvSpec(
    name="math-boxed",
    version="1.0.0",
    family="math",
    summary="Math problems with one exact final answer in \\boxed{}: numbers, fractions, tuples, expressions.",
    env=MathEnv,
    license="Apache-2.0",
    verifier=(
        "The last \\boxed{} of the reply equals the gold answer: exactly as a rational for number and tuple "
        "answers, by sympy equivalence for expressions. The answer must be in the form the row asks for (a number "
        "literal, an expanded expression), so the unevaluated question does not pass."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="Generated arithmetic, fractions, linear equations, percentages, expansions, 2x2 systems.",
        ),
        "gsm8k": DatasetSpec(
            hub_id="openai/gsm8k",
            config="main",
            split="test",
            convert=from_gsm8k,
            license="MIT",
            url="https://huggingface.co/datasets/openai/gsm8k",
            citation="Cobbe et al., Training Verifiers to Solve Math Word Problems, 2021 (arXiv:2110.14168)",
            description="Grade-school word problems (1319 test rows).",
        ),
    },
    difficulty={"option": "difficulty", "levels": [1, 2, 3], "row_field": "difficulty"},
    max_turns=1,
    max_new_tokens=1024,
    oracle=oracle,
    degenerate={
        "box_the_question": box_the_question,
        "box_prompt_numbers": box_prompt_numbers,
        "box_zero": box_zero,
        "unboxed_gold": unboxed_gold,
    },
    requires=("sympy",),
    tags=("single-turn", "exact-answer"),
)
