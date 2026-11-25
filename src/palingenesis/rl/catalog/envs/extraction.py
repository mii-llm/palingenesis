"""json-extraction: read a document, return a JSON object that fits a schema and holds the right values.

Rows: {"prompt", "schema": a JSON Schema, "answer": the gold object}. The reply's last ```json block (or a reply
that is nothing but JSON) is parsed, validated against the schema, then compared with the gold field by field:

  valid     the schema holds: types, required keys, no extra keys, enums, patterns, bounds (a subset of JSON
            Schema checked here, without dependencies: type, properties, required, additionalProperties, items,
            enum, const, pattern, minimum, maximum, minLength, minItems, maxItems)
  correct   valid and every value equals the gold: strings without case and surrounding whitespace, numbers to
            1e-6, arrays in order unless the schema marks them "x-unordered", null only where the gold is null

A schema-valid object of blanks, the schema itself, the gold with one extra key, or the gold with numbers written
as strings all fail. The documents are generated with distractors next to every field (an issue date next to the
due date, a subtotal next to the total) and dates written in prose, to be returned as ISO dates.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Violations of `schema` by `value` (the subset of JSON Schema in the module docstring); empty when valid."""
    errors: list[str] = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]

        def is_kind(kind: str) -> bool:
            if isinstance(value, bool):
                return kind == "boolean"
            if kind == "integer":
                return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            return isinstance(value, _TYPES.get(kind, ()))

        if not any(is_kind(kind) for kind in kinds):
            return [f"{path}: expected {' or '.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        for key, item in value.items():
            if key in properties:
                errors += validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key {key!r}")
    return errors


def _text(value: str) -> str:
    return " ".join(value.split()).lower()


def compare(got: Any, want: Any, schema: dict | None = None) -> tuple[int, int]:
    """(leaf values of the gold matched, leaf values in the gold)."""
    schema = schema or {}
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return 0, sum(compare(None, v)[1] for v in want.values()) or 1
        properties = schema.get("properties") or {}
        pairs = [compare(got.get(key), value, properties.get(key)) for key, value in want.items()]
        return sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    if isinstance(want, list):
        total = sum(compare(None, v)[1] for v in want) or 1
        if not isinstance(got, list) or len(got) != len(want):
            return 0, total
        items = schema.get("items") if isinstance(schema.get("items"), dict) else None
        if schema.get("x-unordered"):
            got = sorted(got, key=lambda v: _text(json.dumps(v, sort_keys=True)))
            want = sorted(want, key=lambda v: _text(json.dumps(v, sort_keys=True)))
        pairs = [compare(g, w, items) for g, w in zip(got, want)]
        return (sum(p[0] for p in pairs), sum(p[1] for p in pairs)) if pairs else (1, 1)
    if isinstance(want, bool) or want is None:
        return int(type(got) is type(want) and got == want), 1
    if isinstance(want, (int, float)):
        ok = isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-6 * max(1, abs(want))
        return int(ok), 1
    return int(isinstance(got, str) and _text(got) == _text(str(want))), 1


def extract_json(reply: str) -> Any:
    """The reply's last ```json block, or the whole reply, parsed; ValueError when neither is JSON."""
    from palingenesis.rl.grading import extract_code

    block = extract_code(reply, languages=("json", ""))
    return json.loads(block if block is not None else reply)


class ExtractionEnv:
    tools = ()

    def __init__(self):
        self.schema: dict = {}
        self.answer: Any = None

    def reset(self, schema: Any = None, answer: Any = None, **row) -> None:
        self.schema = json.loads(schema) if isinstance(schema, str) else (schema or {})
        self.answer = json.loads(answer) if isinstance(answer, str) else answer

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        out = {"correct": 0.0, "metric/parsed": 0.0, "metric/valid": 0.0, "metric/fields": 0.0}
        try:
            value = extract_json(reply)
        except ValueError:
            return out
        out["metric/parsed"] = 1.0
        valid = not validate(value, self.schema)
        matched, total = compare(value, self.answer, self.schema)
        out["metric/valid"] = float(valid)
        out["metric/fields"] = matched / total
        out["correct"] = float(valid and matched == total)
        return out


# ------------------------------------------------------------------ generator

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]  # fmt: skip
_VENDORS = ["Northwind Supply", "Calder & Voss", "Brightline Tools", "Tessera Works", "Okapi Freight"]
_ITEMS = ["cable ties", "safety gloves", "packing tape", "steel brackets", "label rolls", "filter cartridges"]
_COMPANIES = ["Lumen Labs", "Harrow Analytics", "Pine Street Robotics", "Quill Health", "Drift Aerospace"]
_JOBS = ["Data Engineer", "Support Lead", "Product Designer", "Site Reliability Engineer", "Technical Writer"]
_SKILLS = ["SQL", "Python", "Kubernetes", "Figma", "Go", "Terraform", "Airflow", "Rust", "Swift"]
_PLACES = ["Rotterdam", "Bilbao", "Leipzig", "Gdansk", "Aarhus"]
_VENUES = ["the Old Granary", "Harbour Hall", "Studio Nine", "the Glasshouse", "Mercer Auditorium"]
_SPEAKERS = ["Ada Okafor", "Luis Brandt", "Mei Tanaka", "Noor Haddad", "Petra Novak", "Sven Aalto"]
_TALKS = ["Night of Small Models", "Maps That Lie", "The Sourdough Summit", "Circuits and Cello", "Open Data Day"]
_ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


def _date(rng: random.Random) -> tuple[str, str]:
    year, month, day = rng.randint(2029, 2033), rng.randint(1, 12), rng.randint(1, 28)
    prose = rng.choice([f"{day} {_MONTHS[month - 1]} {year}", f"{_MONTHS[month - 1]} {day}, {year}"])
    return f"{year:04d}-{month:02d}-{day:02d}", prose


def _obj(properties: dict, **extra) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False, **extra}


def _invoice(rng: random.Random) -> tuple[str, dict, dict, str]:
    (issued, issued_text), (due, due_text) = _date(rng), _date(rng)
    vendor, number, currency = rng.choice(_VENDORS), f"INV-{rng.randint(1000, 9999)}", rng.choice(["EUR", "USD", "GBP"])
    lines = [
        {"description": item, "quantity": rng.randint(2, 40), "unit_price": rng.randrange(150, 4000, 25) / 100}
        for item in rng.sample(_ITEMS, rng.randint(2, 3))
    ]
    subtotal = round(sum(line["quantity"] * line["unit_price"] for line in lines), 2)
    shipping = rng.randrange(500, 3000, 50) / 100
    total = round(subtotal + shipping, 2)
    rows = "\n".join(f"- {line['quantity']} x {line['description']} at {line['unit_price']:.2f} each" for line in lines)
    text = (
        f"From: accounts@{vendor.split()[0].lower()}.example\nSubject: Invoice {number}\n\nDear customer,\n\n"
        f"please find below invoice {number} from {vendor}, issued on {issued_text}. Your previous invoice "
        f"INV-{rng.randint(100, 999)} has been settled, thank you.\n\n{rows}\n\nSubtotal: {subtotal:.2f} {currency}\n"
        f"Shipping: {shipping:.2f} {currency}\nTotal due: {total:.2f} {currency}\n\nPayment is due by {due_text}."
    )
    schema = _obj(
        {
            "vendor": {"type": "string", "minLength": 1},
            "invoice_number": {"type": "string", "pattern": r"^INV-\d+$"},
            "due_date": {"type": "string", "pattern": _ISO_DATE, "description": "ISO date, YYYY-MM-DD"},
            "currency": {"type": "string", "enum": ["EUR", "USD", "GBP"]},
            "total": {"type": "number", "minimum": 0, "description": "the amount to pay, shipping included"},
            "lines": {
                "type": "array",
                "minItems": 1,
                "items": _obj(
                    {
                        "description": {"type": "string", "minLength": 1},
                        "quantity": {"type": "integer", "minimum": 1},
                        "unit_price": {"type": "number", "minimum": 0},
                    }
                ),
            },
        }
    )
    answer = {"vendor": vendor, "invoice_number": number, "due_date": due, "currency": currency, "total": total, "lines": lines}
    return text, schema, answer, "invoice"


def _job(rng: random.Random) -> tuple[str, dict, dict, str]:
    title, company, city = rng.choice(_JOBS), rng.choice(_COMPANIES), rng.choice(_PLACES)
    remote = rng.random() < 0.5
    low = rng.randrange(40, 90, 5) * 1000
    high = low + rng.randrange(10, 40, 5) * 1000
    skills, nice = rng.sample(_SKILLS, 5)[:3], rng.choice(_SKILLS)
    (deadline, deadline_text), (_, posted_text) = _date(rng), _date(rng)
    text = (
        f"{company} is hiring a {title}.\n\nPosted {posted_text}. Our office is in {city}"
        f"{'; the role can be done fully remotely' if remote else ' and the team works on site; this is not a remote role'}. "
        f"The salary band runs from {low:,} to {high:,} euros a year, and last year's bonus pool was {rng.randint(3, 9)}00,000 "
        f"euros.\n\nYou must know {skills[0]}, {skills[1]} and {skills[2]}. Experience with "
        f"{nice if nice not in skills else 'mentoring'} is a plus but not required.\n\nApply by {deadline_text}."
    )
    schema = _obj(
        {
            "title": {"type": "string", "minLength": 1},
            "company": {"type": "string", "minLength": 1},
            "city": {"type": "string", "minLength": 1},
            "remote": {"type": "boolean"},
            "salary_min": {"type": "integer", "minimum": 0},
            "salary_max": {"type": "integer", "minimum": 0},
            "required_skills": {"type": "array", "items": {"type": "string"}, "minItems": 1, "x-unordered": True},
            "apply_by": {"type": "string", "pattern": _ISO_DATE, "description": "ISO date, YYYY-MM-DD"},
        }
    )
    answer = {
        "title": title,
        "company": company,
        "city": city,
        "remote": remote,
        "salary_min": low,
        "salary_max": high,
        "required_skills": skills,
        "apply_by": deadline,
    }
    return text, schema, answer, "job"


def _event(rng: random.Random) -> tuple[str, dict, dict, str]:
    name, venue, city = rng.choice(_TALKS), rng.choice(_VENUES), rng.choice(_PLACES)
    (date, date_text), (_, sales_text) = _date(rng), _date(rng)
    hour, doors = rng.randint(17, 20), rng.choice([30, 45, 60])
    speakers = rng.sample(_SPEAKERS, rng.randint(2, 3))
    host = rng.choice([s for s in _SPEAKERS if s not in speakers])
    free = rng.random() < 0.35
    price = None if free else rng.randrange(8, 60)
    money = "Entry is free." if free else f"Tickets cost {price} euros ({price + 5} at the door is an old rumour: it is {price})."
    text = (
        f"{name} comes to {city}!\n\nJoin us at {venue} on {date_text}. Doors open {doors} minutes before the "
        f"{hour}:00 start. Talks by {', '.join(speakers[:-1])} and {speakers[-1]}; the evening is hosted by {host}. "
        f"{money} Ticket sales open on {sales_text}."
    )
    schema = _obj(
        {
            "name": {"type": "string", "minLength": 1},
            "venue": {"type": "string", "minLength": 1},
            "city": {"type": "string", "minLength": 1},
            "date": {"type": "string", "pattern": _ISO_DATE, "description": "ISO date of the event, YYYY-MM-DD"},
            "start_time": {"type": "string", "pattern": r"^\d{2}:\d{2}$", "description": "24-hour HH:MM"},
            "speakers": {"type": "array", "items": {"type": "string"}, "x-unordered": True, "description": "not the host"},
            "price_eur": {"type": ["number", "null"], "description": "null when entry is free"},
        }
    )
    answer = {
        "name": name,
        "venue": venue,
        "city": city,
        "date": date,
        "start_time": f"{hour:02d}:00",
        "speakers": speakers,
        "price_eur": price,
    }
    return text, schema, answer, "event"


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    makers = [_invoice, _job, _event]
    rows = []
    for i in range(n):
        text, schema, answer, kind = makers[i % len(makers)](rng)
        rows.append(
            {
                "id": f"json-extraction/synthetic/{seed}/{i}",
                "prompt": "Extract the information from the document below as one JSON object that validates "
                f"against this JSON Schema:\n\n```json\n{json.dumps(schema, indent=1)}\n```\n\nDocument:\n\n"
                f"{text}\n\nReply with the JSON object in a ```json code block and nothing after it.",
                "schema": schema,
                "answer": answer,
                "task": kind,
            }
        )
    return rows


# ------------------------------------------------------------------- policies


def _fenced(value: Any) -> str:
    return f"```json\n{json.dumps(value, indent=1)}\n```"


def oracle(row: dict, messages: list[dict]) -> str:
    return "Here is the extracted record.\n\n" + _fenced(row["answer"])


def blank(schema: dict) -> Any:
    """The emptiest value that validates (as far as blanks can)."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    kind = kind[-1] if isinstance(kind, list) else kind
    if kind == "object":
        return {key: blank(sub) for key, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [blank(schema["items"])] * schema.get("minItems", 0) if isinstance(schema.get("items"), dict) else []
    if kind == "string":
        pattern = schema.get("pattern", "")
        return "2030-01-01" if pattern == _ISO_DATE else "00:00" if "\\d{2}:\\d{2}" in pattern else "INV-0" if "INV" in pattern else "x"
    return {"integer": 0, "number": 0, "boolean": False, "null": None}.get(kind)


def empty_object(row: dict, messages: list[dict]) -> str:
    return _fenced({})


def schema_valid_blanks(row: dict, messages: list[dict]) -> str:
    return _fenced(blank(row["schema"]))


def schema_echo(row: dict, messages: list[dict]) -> str:
    return _fenced(row["schema"])


def gold_extra_key(row: dict, messages: list[dict]) -> str:
    return _fenced({**row["answer"], "confidence": 1.0})


def gold_as_strings(row: dict, messages: list[dict]) -> str:
    def stringify(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: stringify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [stringify(v) for v in value]
        return value if isinstance(value, str) else json.dumps(value)

    return _fenced(stringify(row["answer"]))


SPEC = EnvSpec(
    name="json-extraction",
    version="1.0.0",
    family="extraction",
    summary="Extract a record from a document as JSON that validates against a schema and matches the gold values.",
    env=ExtractionEnv,
    license="Apache-2.0",
    verifier=(
        "The reply's JSON validates against the row's JSON Schema (types, required and unexpected keys, enums, "
        "patterns, bounds) and every value equals the gold (strings without case, numbers to 1e-6, arrays in "
        "order unless marked unordered)."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="Generated invoices, job postings and event notices with distractor values and prose dates.",
        )
    },
    difficulty={"row_field": "task"},
    max_turns=1,
    max_new_tokens=768,
    oracle=oracle,
    degenerate={
        "empty_object": empty_object,
        "schema_valid_blanks": schema_valid_blanks,
        "schema_echo": schema_echo,
        "gold_extra_key": gold_extra_key,
        "gold_as_strings": gold_as_strings,
    },
    tags=("single-turn", "structured-output"),
)
