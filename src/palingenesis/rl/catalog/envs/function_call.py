"""tools-function-call: single-turn function calling, matched against the acceptable calls (BFCL style).

Rows: {"prompt", "tools": [OpenAI schemas], "answer": [{"name": ..., "arguments": {arg: [acceptable values]}}]}.
An argument whose acceptable values include "" may be left out. An empty answer is an irrelevance row: none of
the offered tools fits and the policy must not call any.

The episode ends with the first assistant turn. Its tool calls (as the trainer parsed them from the sampled text)
must be exactly the expected calls, in any order: the same number of calls, each with the expected name, no
argument the answer does not list, every required argument present, every value acceptable (strings compared
without case or surrounding whitespace, 3 and 3.0 equal, lists and objects element by element). A call described
in prose, the right call plus a spare one, or the right name with other arguments scores 0.
"""

from __future__ import annotations

import json
import random
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec

OPTIONAL = ""


def values_equal(got: Any, want: Any) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(want, bool) and isinstance(got, bool) and got == want
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and abs(got - want) <= 1e-9 * max(1.0, abs(want))
    if isinstance(want, str):
        return isinstance(got, str) and " ".join(got.split()).lower() == " ".join(want.split()).lower()
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(values_equal(g, w) for g, w in zip(got, want))
    if isinstance(want, dict):
        return isinstance(got, dict) and got.keys() == want.keys() and all(values_equal(got[k], want[k]) for k in want)
    return got == want


def call_matches(name: str, arguments: Any, expected: dict) -> bool:
    if name != expected["name"] or not isinstance(arguments, dict):
        return False
    acceptable = expected.get("arguments") or {}
    for key, value in arguments.items():
        if key not in acceptable or not any(values_equal(value, w) for w in acceptable[key] if w != OPTIONAL):
            return False
    return all(key in arguments or OPTIONAL in values for key, values in acceptable.items())


def calls_match(calls: list[tuple[str, Any]], expected: list[dict]) -> bool:
    """Whether the calls are exactly the expected ones, in any order."""
    if len(calls) != len(expected):
        return False
    unused = list(calls)
    for want in expected:
        found = next((c for c in unused if call_matches(c[0], c[1], want)), None)
        if found is None:
            return False
        unused.remove(found)
    return True


def first_turn_calls(messages: list[dict]) -> list[tuple[str, Any]]:
    turn = next((m for m in messages if m.get("role") == "assistant"), {})
    calls = []
    for call in turn.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        calls.append((function.get("name", ""), arguments))
    return calls


class FunctionCallEnv:
    """The row's tools, recorded but not executed; the first assistant turn is the answer."""

    def __init__(self):
        self.schemas: list[dict] = []
        self.expected: list[dict] = []
        self.done = False

    def reset(self, tools: Any = None, answer: Any = None, **row) -> None:
        from palingenesis.rl.env import row_tools

        self.schemas = row_tools({"tools": tools}, "tools") or []
        self.expected = json.loads(answer) if isinstance(answer, str) else list(answer or [])
        self.done = False

    def tool_schemas(self) -> list[dict]:
        return self.schemas

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.done = True  # single-turn: nothing is executed
        return "Call recorded."

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        calls = first_turn_calls(messages)
        names = sorted(c[0] for c in calls) == sorted(e["name"] for e in self.expected)
        return {
            "correct": float(calls_match(calls, self.expected)),
            "metric/called": float(bool(calls)),
            "metric/names": float(names),
        }


# ------------------------------------------------------------------ generator

_CITIES = ["Lisbon", "Osaka", "Nairobi", "Turin", "Montreal", "Hanoi", "Lima", "Tallinn", "Perth", "Bergen"]
_CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "CAD", "SEK", "PLN"]
_NAMES = ["Marta", "Kenji", "Amara", "Luca", "Sofia", "Tariq", "Ines", "Jonas"]
_LANGUAGES = ["Italian", "German", "Japanese", "Portuguese", "Polish", "Swahili"]
_TICKERS = ["ACME", "GLOBX", "INIT", "UMBR", "STRK", "WAYN"]
_PHRASES = ["where is the station", "the meeting is postponed", "thank you for your help", "see you tomorrow"]
_CUISINES = ["Thai", "Ethiopian", "Peruvian", "Georgian", "Korean"]


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _s(description: str, **extra) -> dict:
    return {"type": "string", "description": description, **extra}


def _n(description: str, kind: str = "number") -> dict:
    return {"type": kind, "description": description}


TOOLS = {
    "get_weather": _tool(
        "get_weather",
        "Get the weather forecast for a city.",
        {
            "city": _s("The city name."),
            "days": _n("How many days ahead to forecast (default 1).", "integer"),
            "unit": _s("Temperature unit.", enum=["celsius", "fahrenheit"]),
        },
        ["city"],
    ),
    "convert_currency": _tool(
        "convert_currency",
        "Convert an amount of money from one currency to another.",
        {
            "amount": _n("The amount to convert."),
            "from_currency": _s("ISO code of the source currency."),
            "to_currency": _s("ISO code of the target currency."),
        },
        ["amount", "from_currency", "to_currency"],
    ),
    "send_message": _tool(
        "send_message",
        "Send a text message to a contact.",
        {"recipient": _s("The contact's first name."), "text": _s("The message text.")},
        ["recipient", "text"],
    ),
    "set_alarm": _tool(
        "set_alarm",
        "Set an alarm.",
        {"time": _s("The time, 24-hour HH:MM."), "label": _s("An optional label.")},
        ["time"],
    ),
    "translate_text": _tool(
        "translate_text",
        "Translate text into another language.",
        {"text": _s("The text to translate."), "target_language": _s("The language to translate into.")},
        ["text", "target_language"],
    ),
    "get_stock_price": _tool(
        "get_stock_price",
        "Get the latest price of a stock.",
        {"ticker": _s("The stock ticker symbol.")},
        ["ticker"],
    ),
    "book_table": _tool(
        "book_table",
        "Book a restaurant table.",
        {
            "cuisine": _s("The kind of food."),
            "city": _s("The city."),
            "people": _n("How many people.", "integer"),
            "time": _s("The time, 24-hour HH:MM."),
        },
        ["cuisine", "city", "people", "time"],
    ),
    "search_flights": _tool(
        "search_flights",
        "Search flights between two cities.",
        {
            "origin": _s("The departure city."),
            "destination": _s("The arrival city."),
            "max_price": _n("The highest acceptable price in euros."),
            "direct_only": {"type": "boolean", "description": "Only direct flights (default false)."},
        },
        ["origin", "destination"],
    ),
    "calculate_tip": _tool(
        "calculate_tip",
        "Compute the tip for a bill.",
        {"bill": _n("The bill total."), "percent": _n("The tip percentage.")},
        ["bill", "percent"],
    ),
    "create_timer": _tool(
        "create_timer",
        "Start a countdown timer.",
        {"minutes": _n("The duration in minutes.", "integer")},
        ["minutes"],
    ),
}


def _time(rng: random.Random) -> str:
    return f"{rng.randint(6, 22):02d}:{rng.choice([0, 15, 30, 45]):02d}"


def _request(rng: random.Random, name: str) -> tuple[str, dict]:
    """A user request that needs tool `name`, and the acceptable arguments."""
    if name == "get_weather":
        city, days = rng.choice(_CITIES), rng.randint(2, 6)
        if rng.random() < 0.5:
            return f"What will the weather be like in {city} over the next {days} days, in fahrenheit?", {
                "city": [city],
                "days": [days],
                "unit": ["fahrenheit"],
            }
        return f"Give me tomorrow's forecast for {city}.", {"city": [city], "days": [1, OPTIONAL], "unit": ["celsius", OPTIONAL]}
    if name == "convert_currency":
        amount, (a, b) = rng.randrange(50, 5000, 25), rng.sample(_CURRENCIES, 2)
        return f"How much is {amount} {a} in {b}?", {"amount": [amount], "from_currency": [a], "to_currency": [b]}
    if name == "send_message":
        who, text = rng.choice(_NAMES), rng.choice(_PHRASES)
        return f'Text {who} saying exactly "{text}".', {"recipient": [who], "text": [text, text + "."]}
    if name == "set_alarm":
        when = _time(rng)
        return f"Wake me up at {when}.", {"time": [when], "label": [OPTIONAL, "wake up", "Wake up", "wake me up"]}
    if name == "translate_text":
        text, language = rng.choice(_PHRASES), rng.choice(_LANGUAGES)
        return f'How do you say "{text}" in {language}?', {"text": [text], "target_language": [language]}
    if name == "get_stock_price":
        ticker = rng.choice(_TICKERS)
        return f"What is {ticker} trading at right now?", {"ticker": [ticker]}
    if name == "book_table":
        cuisine, city, people, when = rng.choice(_CUISINES), rng.choice(_CITIES), rng.randint(2, 9), _time(rng)
        return f"Reserve a table for {people} at a {cuisine} place in {city} at {when}.", {
            "cuisine": [cuisine],
            "city": [city],
            "people": [people],
            "time": [when],
        }
    if name == "search_flights":
        (a, b), price = rng.sample(_CITIES, 2), rng.randrange(80, 900, 10)
        if rng.random() < 0.5:
            return f"Find me direct flights from {a} to {b} under {price} euros.", {
                "origin": [a],
                "destination": [b],
                "max_price": [price],
                "direct_only": [True],
            }
        return f"Find me flights from {a} to {b}.", {"origin": [a], "destination": [b], "direct_only": [False, OPTIONAL]}
    if name == "calculate_tip":
        bill, percent = rng.randrange(20, 300), rng.choice([10, 12, 15, 18, 20])
        return f"What is a {percent}% tip on a bill of {bill} euros?", {"bill": [bill], "percent": [percent]}
    minutes = rng.randint(3, 90)
    return f"Start a {minutes}-minute timer.", {"minutes": [minutes]}


def generate(n: int, seed: int = 0, distractors: int = 4) -> list[dict]:
    """Rows in three categories, in turn: simple (one call), parallel (two calls), and every seventh row
    irrelevance (the tool the request needs is not offered: no call)."""
    rng = random.Random(seed)
    names = list(TOOLS)
    rows = []
    for i in range(n):
        category = "irrelevance" if i % 7 == 6 else "parallel" if i % 3 == 1 else "simple"
        needed = rng.sample(names, 2 if category == "parallel" else 1)
        requests = [_request(rng, name) for name in needed]
        others = [name for name in names if name not in needed]
        offered = rng.sample(others, min(distractors, len(others))) + ([] if category == "irrelevance" else needed)
        rng.shuffle(offered)
        prompt = " Also: ".join(text for text, _ in requests) if len(requests) > 1 else requests[0][0]
        answer = [] if category == "irrelevance" else [
            {"name": name, "arguments": arguments} for name, (_, arguments) in zip(needed, requests)
        ]
        rows.append(
            {
                "id": f"tools-function-call/synthetic/{seed}/{i}",
                "prompt": prompt,
                "tools": [TOOLS[name] for name in offered],
                "answer": answer,
                "category": category,
            }
        )
    return rows


def from_bfcl(row: dict) -> dict | None:
    """A Berkeley Function-Calling Leaderboard row with its answer joined in: {"question": [[messages]],
    "function": [...], "ground_truth": [{"name": {"arg": [values]}}]} (single-turn categories: simple, multiple,
    parallel; Apache-2.0). Join the question and possible_answer files on "id" before converting."""
    question, truth = row.get("question"), row.get("ground_truth")
    if not question or truth is None or not isinstance(question[0], list) or len(question) != 1:
        return None  # multi-turn categories need their own executable backends

    def schema(function: dict) -> dict:
        def fix(node: Any) -> Any:
            if isinstance(node, dict):
                kind = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
                return {k: kind.get(v, v) if k == "type" and isinstance(v, str) else fix(v) for k, v in node.items()}
            return [fix(v) for v in node] if isinstance(node, list) else node

        return {"type": "function", "function": {**function, "parameters": fix(function.get("parameters") or {})}}

    answer = []
    for call in truth:
        if not isinstance(call, dict) or len(call) != 1:
            return None
        ((name, arguments),) = call.items()
        answer.append({"name": name, "arguments": {k: v if isinstance(v, list) else [v] for k, v in arguments.items()}})
    return {"messages": question[0], "tools": [schema(f) for f in row["function"]], "answer": answer}


# ------------------------------------------------------------------- policies


def _gold_calls(row: dict) -> list[tuple[str, dict]]:
    return [
        (e["name"], {k: next(v for v in values if v != OPTIONAL) for k, values in e["arguments"].items()})
        for e in row["answer"]
    ]


def oracle(row: dict, messages: list[dict]) -> Any:
    return _gold_calls(row) or "None of the available tools can do that."


def text_only(row: dict, messages: list[dict]) -> str:
    """Never calls: right on irrelevance rows only (spec.chance covers their share)."""
    return "I will take care of it."


def call_in_prose(row: dict, messages: list[dict]) -> str:
    return "I would call: " + json.dumps([{"name": n, "arguments": a} for n, a in _gold_calls(row)])


def right_name_no_arguments(row: dict, messages: list[dict]) -> Any:
    return [(e["name"], {}) for e in row["answer"]] or [(row["tools"][0]["function"]["name"], {})]


def gold_plus_spare(row: dict, messages: list[dict]) -> Any:
    spare = row["tools"][0]["function"]["name"]
    return _gold_calls(row) + [(spare, {})]


def gold_wrong_values(row: dict, messages: list[dict]) -> Any:
    def wrong(value: Any) -> Any:
        return (not value) if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else "other"

    calls = [(name, {k: wrong(v) for k, v in arguments.items()}) for name, arguments in _gold_calls(row)]
    return calls or [(row["tools"][0]["function"]["name"], {})]


SPEC = EnvSpec(
    name="tools-function-call",
    version="1.0.0",
    family="tool-calling",
    summary="Single-turn function calling: the right tools with acceptable arguments, or no call at all.",
    env=FunctionCallEnv,
    license="Apache-2.0",
    verifier=(
        "The first assistant turn's tool calls are exactly the expected calls in any order: names, no unexpected "
        "argument, required arguments present, every value among the acceptable ones. Irrelevance rows (no tool "
        "fits) are right only without a call."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="Requests over ten tools with distractors: simple, parallel and irrelevance rows.",
        ),
    },
    difficulty={"option": "distractors", "levels": [1, 4, 8], "row_field": "category"},
    max_turns=2,
    max_new_tokens=512,
    chance=0.2,  # the irrelevance rows (one in seven) are right for a policy that never calls
    oracle=oracle,
    degenerate={
        "text_only": text_only,
        "call_in_prose": call_in_prose,
        "right_name_no_arguments": right_name_no_arguments,
        "gold_plus_spare": gold_plus_spare,
        "gold_wrong_values": gold_wrong_values,
    },
    tags=("single-turn", "function-calling"),
    notes=(
        "BFCL's single-turn files convert with envs.function_call.from_bfcl (join the question and "
        "possible_answer files on id first)."
    ),
)
