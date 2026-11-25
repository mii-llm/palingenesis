"""text-to-sql: answer a question with one SQLite query; graded by execution on the database and on a hidden twin.

Rows: {"prompt", "schema", "gold_sql", "hidden_schema"?}. `schema` is a script (CREATE TABLE + INSERT) that builds
the database the policy can explore with the run_sql tool (read-only); the final reply's last ```sql block is the
answer. It is executed, with the gold query, on that database and, when the row has one, on `hidden_schema`: the same
tables holding other data. The result sets must be equal on both (row order only where the gold query orders them;
floats to 1e-6; column names ignored, column order not).

The hidden twin is what makes the tool safe to give: a policy can run queries, read the answer off the visible
database and reply `SELECT 42`; that literal is wrong on the twin. Queries run on in-memory databases with an
authorizer that allows reading only (no writes, ATTACH, PRAGMA, extensions) and a cap on virtual-machine steps, so
a runaway recursive query ends as an error. Built databases are cached as serialized images and restored per
episode.
"""

from __future__ import annotations

import random
import re
import sqlite3
import threading
from collections import Counter, OrderedDict
from typing import Any

from palingenesis.rl.catalog.runner import turn_index
from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec
from palingenesis.rl.env import ToolFailed

MAX_STEPS = 2_000_000  # SQLite virtual-machine instructions per query
MAX_ROWS = 10_000
_DENIED_FUNCTIONS = {"load_extension", "randomblob", "zeroblob", "readfile", "writefile", "edit", "fts3_tokenizer"}
_images: OrderedDict[str, bytes] = OrderedDict()


class SqlError(Exception):
    pass


def _authorize(action, arg1, arg2, database, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if str(arg2).lower() in _DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def open_database(script: str) -> sqlite3.Connection:
    """A read-only in-memory database built by `script` (from a cached image after the first time)."""
    image = _images.get(script)
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    if image is None:
        try:
            connection.executescript(script)
        except sqlite3.Error as error:
            connection.close()
            raise SqlError(f"the schema script does not run: {error}") from None
        connection.commit()
        _images[script] = connection.serialize()
        while len(_images) > 512:
            _images.popitem(last=False)
    else:
        _images.move_to_end(script)
        connection.deserialize(image)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
    connection.set_authorizer(_authorize)
    return connection


def run_query(connection: sqlite3.Connection, query: str, max_rows: int = MAX_ROWS) -> tuple[list[str], list[tuple]]:
    """(column names, rows) of one read-only statement; SqlError on anything else."""
    query = query.strip().rstrip(";").strip()
    if not query:
        raise SqlError("empty query")
    steps = [0]

    def tick() -> int:
        steps[0] += 1000
        return int(steps[0] > MAX_STEPS)  # non-zero interrupts the query

    connection.set_progress_handler(tick, 1000)
    try:
        cursor = connection.execute(query)  # execute() refuses more than one statement
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.Warning as error:
        raise SqlError(f"one statement only: {error}") from None
    except sqlite3.Error as error:
        message = str(error)
        if "not authorized" in message or "prohibited" in message:
            message = "the database is read-only: only SELECT queries are allowed"
        elif "interrupted" in message:
            message = "the query ran too long and was stopped"
        raise SqlError(message) from None
    finally:
        connection.set_progress_handler(None, 0)
    if len(rows) > max_rows:
        raise SqlError(f"the query returns more than {max_rows} rows")
    return columns, rows


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) + 0.0
    if isinstance(value, bytes):
        return value.hex()
    return value


def results_equal(got: list[tuple], want: list[tuple], ordered: bool) -> bool:
    a = [tuple(_cell(v) for v in row) for row in got]
    b = [tuple(_cell(v) for v in row) for row in want]
    if len(a) != len(b):
        return False
    return a == b or (not ordered and Counter(map(repr, a)) == Counter(map(repr, b)))


_ORDER_BY = re.compile(r"\border\s+by\b(?![^()]*\))", re.I)


def is_ordered(query: str) -> bool:
    """Whether the query orders its result at the top level (an ORDER BY outside parentheses)."""
    return bool(_ORDER_BY.search(query))


def extract_sql(reply: str) -> str | None:
    """The reply's last ```sql block; a reply that is nothing but a query counts as one."""
    from palingenesis.rl.grading import extract_code

    block = extract_code(reply, languages=("sql", "sqlite", "sqlite3", ""))
    if block is not None:
        return block.strip()
    text = reply.strip()
    return text if re.match(r"(?is)^(select|with)\b", text) else None


def format_rows(columns: list[str], rows: list[tuple], limit: int = 20) -> str:
    lines = [" | ".join(columns)] + [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... ({len(rows)} rows, {limit} shown)")
    return "\n".join(lines) if rows else " | ".join(columns) + "\n(no rows)"


class SqlEnv:
    tools = ("run_sql",)

    def __init__(self):
        self.connection: sqlite3.Connection | None = None
        self.row: dict[str, Any] = {}
        # one query at a time per connection: the calls of one turn run in parallel threads, and SQLite calling
        # back into Python from one while another sets its handlers on the same connection deadlocks
        self.lock = threading.Lock()

    def reset(self, schema: str = "", gold_sql: str = "", hidden_schema: str = "", **row) -> None:
        self.close()
        self.row = {"schema": schema, "gold_sql": gold_sql, "hidden_schema": hidden_schema}
        self.connection = open_database(schema)

    def run_sql(self, query: str) -> str:
        """Run one read-only SQLite query on the database and return its rows (at most 20 are shown).

        Args:
            query: A single SELECT statement.
        """
        try:
            with self.lock:
                columns, rows = run_query(self.connection, str(query), max_rows=1000)
        except SqlError as error:
            raise ToolFailed(str(error)) from None
        return format_rows(columns, rows)

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        query = extract_sql(reply)
        out = {"correct": 0.0, "metric/valid": 0.0, "metric/visible": 0.0}
        if query is None:
            return out
        gold, ordered = self.row["gold_sql"], is_ordered(self.row["gold_sql"])
        scripts = [self.row["schema"]] + ([self.row["hidden_schema"]] if self.row["hidden_schema"] else [])
        verdicts = []
        for script in scripts:
            connection = open_database(script)
            try:
                _, want = run_query(connection, gold)
                try:
                    _, got = run_query(connection, query)
                except SqlError:
                    return out
                verdicts.append(results_equal(got, want, ordered))
            finally:
                connection.close()
        out["metric/valid"] = 1.0
        out["metric/visible"] = float(verdicts[0])
        out["correct"] = float(all(verdicts))
        return out

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None


# ------------------------------------------------------------------ generator

_DEPARTMENTS = ["Research", "Sales", "Support", "Finance", "Design", "Logistics"]
_CITIES = ["Lyon", "Porto", "Graz", "Malmo", "Cork"]
_FIRST = ["Aiko", "Bruno", "Carla", "Dmitri", "Elif", "Farid", "Greta", "Hugo", "Ilse", "Jamal", "Kira", "Leon",
          "Mina", "Nuno", "Olga", "Pavel", "Quinn", "Rosa", "Sami", "Tova", "Ugo", "Vera", "Wim", "Yara"]  # fmt: skip
_COUNTRIES = ["Chile", "Kenya", "Norway", "Japan", "Greece"]
_CATEGORIES = ["tools", "garden", "kitchen", "audio"]
_PRODUCTS = ["Anvil", "Bellows", "Chisel", "Dibber", "Easel", "Funnel", "Gimlet", "Hatchet", "Inkwell", "Jigsaw",
             "Kettle", "Ladle", "Mallet", "Nozzle"]  # fmt: skip

COMPANY_DDL = (
    "CREATE TABLE department (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL);\n"
    "CREATE TABLE employee (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER NOT NULL "
    "REFERENCES department(id), salary INTEGER NOT NULL, hired_year INTEGER NOT NULL);\n"
)
STORE_DDL = (
    "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);\n"
    "CREATE TABLE product (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, price INTEGER NOT NULL);\n"
    "CREATE TABLE purchase (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customer(id), "
    "product_id INTEGER NOT NULL REFERENCES product(id), quantity INTEGER NOT NULL, year INTEGER NOT NULL);\n"
)


def _values(table: str, rows: list[tuple]) -> str:
    def literal(v: Any) -> str:
        return str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'"

    return "".join(f"INSERT INTO {table} VALUES ({', '.join(literal(v) for v in row)});\n" for row in rows)


def _company(rng: random.Random) -> str:
    cities = _CITIES[:3] + [rng.choice(_CITIES) for _ in range(3)]
    rng.shuffle(cities)
    departments = [(i + 1, name, cities[i]) for i, name in enumerate(_DEPARTMENTS)]
    names = rng.sample(_FIRST, 20)
    salaries = rng.sample(range(31, 120), 20)
    staffed = rng.sample(range(1, len(departments) + 1), rng.randint(4, 5))  # one or two departments stay empty
    employees = [
        (i + 1, names[i], rng.choice(staffed), salaries[i] * 1000, rng.randint(2008, 2024)) for i in range(20)
    ]
    return COMPANY_DDL + _values("department", departments) + _values("employee", employees)


def _store(rng: random.Random) -> str:
    customers = [(i + 1, name, rng.choice(_COUNTRIES)) for i, name in enumerate(rng.sample(_FIRST, 12))]
    prices = rng.sample(range(5, 200), len(_PRODUCTS))
    products = [(i + 1, name, _CATEGORIES[i % len(_CATEGORIES)], prices[i]) for i, name in enumerate(_PRODUCTS)]
    purchases = [
        (i + 1, rng.randint(1, 10), rng.randint(1, len(products)), rng.randint(1, 6), rng.randint(2020, 2024))
        for i in range(30)
    ]
    return STORE_DDL + _values("customer", customers) + _values("product", products) + _values("purchase", purchases)


def _company_question(rng: random.Random) -> tuple[str, str, int]:
    department, city = rng.choice(_DEPARTMENTS), rng.choice(_CITIES[:3])
    year, salary, k = rng.randint(2010, 2020), rng.randrange(40, 90, 5) * 1000, rng.randint(2, 4)
    return rng.choice(
        [
            (
                f"How many employees work in the {department} department?",
                "SELECT COUNT(*) FROM employee e JOIN department d ON d.id = e.department_id "
                f"WHERE d.name = '{department}'",
                1,
            ),
            (
                f"List the names of the employees hired after {year} who earn more than {salary}, in alphabetical order.",
                f"SELECT name FROM employee WHERE hired_year > {year} AND salary > {salary} ORDER BY name",
                1,
            ),
            (
                "For each department that has employees, give the department name and the average salary of its "
                "employees, highest average first.",
                "SELECT d.name, AVG(e.salary) FROM employee e JOIN department d ON d.id = e.department_id "
                "GROUP BY d.id ORDER BY AVG(e.salary) DESC",
                2,
            ),
            (
                f"What is the name of the highest-paid employee among the departments located in {city}?",
                "SELECT e.name FROM employee e JOIN department d ON d.id = e.department_id "
                f"WHERE d.city = '{city}' ORDER BY e.salary DESC LIMIT 1",
                2,
            ),
            (
                f"Which departments have more than {k} employees? Return their names.",
                "SELECT d.name FROM department d JOIN employee e ON e.department_id = d.id "
                f"GROUP BY d.id HAVING COUNT(*) > {k}",
                2,
            ),
            (
                "Which departments have no employees? Return their names.",
                "SELECT name FROM department WHERE id NOT IN (SELECT department_id FROM employee)",
                3,
            ),
            (
                "Give the name of every employee who earns more than the average salary of their own department, "
                "in alphabetical order.",
                "SELECT e.name FROM employee e WHERE e.salary > (SELECT AVG(x.salary) FROM employee x "
                "WHERE x.department_id = e.department_id) ORDER BY e.name",
                3,
            ),
        ]
    )


def _store_question(rng: random.Random) -> tuple[str, str, int]:
    category, country, year = rng.choice(_CATEGORIES), rng.choice(_COUNTRIES), rng.randint(2021, 2023)
    return rng.choice(
        [
            (
                f"What is the total revenue (price times quantity, summed over purchases) of the {category} category?",
                "SELECT SUM(p.price * b.quantity) FROM purchase b JOIN product p ON p.id = b.product_id "
                f"WHERE p.category = '{category}'",
                2,
            ),
            (
                f"List the names of the customers who bought at least one {category} product, each once, in "
                "alphabetical order.",
                "SELECT DISTINCT c.name FROM customer c JOIN purchase b ON b.customer_id = c.id "
                f"JOIN product p ON p.id = b.product_id WHERE p.category = '{category}' ORDER BY c.name",
                2,
            ),
            (
                "For each year, how many purchases were made? Return the year and the count, earliest year first.",
                "SELECT year, COUNT(*) FROM purchase GROUP BY year ORDER BY year",
                1,
            ),
            (
                "Which customers have never made a purchase? Return their names in alphabetical order.",
                "SELECT name FROM customer WHERE id NOT IN (SELECT customer_id FROM purchase) ORDER BY name",
                3,
            ),
            (
                f"How many units in total did customers from {country} buy since {year} (that year included)?",
                "SELECT SUM(b.quantity) FROM purchase b JOIN customer c ON c.id = b.customer_id "
                f"WHERE c.country = '{country}' AND b.year >= {year}",
                2,
            ),
            (
                "Give the name of each product and the total quantity sold of it, only for products sold at least "
                "once, largest total first and by name for equal totals.",
                "SELECT p.name, SUM(b.quantity) FROM product p JOIN purchase b ON b.product_id = p.id "
                "GROUP BY p.id ORDER BY SUM(b.quantity) DESC, p.name",
                3,
            ),
        ]
    )


def _ddl(script: str) -> str:
    return "\n".join(line for line in script.splitlines() if line.startswith("CREATE"))


def generate(n: int, seed: int = 0, difficulty: int = 0) -> list[dict]:
    """Questions over two generated databases (a company, a store); difficulty 1 (one table or a count), 2 (joins,
    grouping), 3 (subqueries, anti-joins), 0: all."""
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        build, ask = rng.choice([(_company, _company_question), (_store, _store_question)])
        question, gold, level = ask(rng)
        if difficulty and level != difficulty:
            continue
        schema, hidden = build(rng), build(rng)
        results = []
        for script in (schema, hidden):
            connection = open_database(script)
            results.append(run_query(connection, gold)[1])
            connection.close()
        if any(not r or r in ([(None,)], [(0,)]) for r in results) or results_equal(results[0], results[1], False):
            continue  # an empty or constant result pays a trivial query; equal results would pay a hard-coded one
        if is_ordered(gold) and len(set(map(repr, results[0]))) != len(results[0]):
            continue
        rows.append(
            {
                "id": f"text-to-sql/synthetic/{seed}/{len(rows)}",
                "prompt": f"A SQLite database has these tables:\n\n{_ddl(schema)}\n\n{question}\n\n"
                "You can explore the data with the run_sql tool. Reply with one SQLite query that answers the "
                "question, in a ```sql code block. The query is graded by running it, also on other data.",
                "schema": schema,
                "hidden_schema": hidden,
                "gold_sql": gold,
                "question": question,
                "difficulty": level,
            }
        )
    return rows


def from_gretel(row: dict) -> dict | None:
    """gretelai/synthetic_text_to_sql: rows whose context and query run in SQLite and return something."""
    context, gold = row.get("sql_context") or "", (row.get("sql") or "").strip().rstrip(";")
    if not re.match(r"(?is)^(select|with)\b", gold):
        return None
    try:
        connection = open_database(context)
        try:
            _, result = run_query(connection, gold)
        finally:
            connection.close()
    except SqlError:
        return None
    if not result or result in ([(None,)], [(0,)]):
        return None
    ddl = "\n".join(s.strip() + ";" for s in context.split(";") if s.strip().upper().startswith("CREATE"))
    return {
        "prompt": f"A SQLite database has these tables:\n\n{ddl}\n\n{row['sql_prompt']}\n\n"
        "You can explore the data with the run_sql tool. Reply with one SQLite query that answers the question, "
        "in a ```sql code block.",
        "schema": context,
        "gold_sql": gold,
        "question": row["sql_prompt"],
    }


# ------------------------------------------------------------------- policies


def _fenced(query: str) -> str:
    return f"```sql\n{query}\n```"


def oracle(row: dict, messages: list[dict]) -> Any:
    if turn_index(messages) == 0:
        return [("run_sql", {"query": row["gold_sql"]})]
    return "This query answers it.\n\n" + _fenced(row["gold_sql"])


def _first_table(row: dict) -> str:
    return re.search(r"CREATE TABLE (\w+)", row["schema"]).group(1)


def select_star(row: dict, messages: list[dict]) -> str:
    return _fenced(f"SELECT * FROM {_first_table(row)}")


def select_constant(row: dict, messages: list[dict]) -> str:
    return _fenced("SELECT 1")


def hardcode_visible(row: dict, messages: list[dict]) -> str:
    """Runs the gold query on the visible database and replies with its rows as literals."""
    connection = open_database(row["schema"])
    _, rows = run_query(connection, row["gold_sql"])
    connection.close()

    def literal(v: Any) -> str:
        return "NULL" if v is None else repr(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'"

    return _fenced(" UNION ALL ".join("SELECT " + ", ".join(literal(v) for v in r) for r in rows))


def gold_then_drop(row: dict, messages: list[dict]) -> Any:
    """Tries to change the database through the tool, then stacks a second statement on the gold query."""
    if turn_index(messages) == 0:
        return [("run_sql", {"query": f"DROP TABLE {_first_table(row)}"}), ("run_sql", {"query": f"DELETE FROM {_first_table(row)}"})]
    return _fenced(f"{row['gold_sql']}; DROP TABLE {_first_table(row)}")


def gold_in_prose(row: dict, messages: list[dict]) -> str:
    return f"I would count the rows with a query along the lines of {row['gold_sql']} and read off the result."


def runaway(row: dict, messages: list[dict]) -> str:
    return _fenced("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")


SPEC = EnvSpec(
    name="text-to-sql",
    version="1.0.0",
    family="sql",
    summary="Answer a question with one SQLite query; graded by execution, also on a hidden twin of the database.",
    env=SqlEnv,
    license="Apache-2.0",
    verifier=(
        "The final ```sql block, one read-only statement, returns the same rows as the gold query on the visible "
        "database and on a hidden database with the same tables and other data (order compared where the gold "
        "query orders; floats to 1e-6). A literal read off the visible data fails on the hidden one."
    ),
    datasets={
        "synthetic": DatasetSpec(
            generator=generate,
            license="Apache-2.0",
            description="Two generated schemas (company, store) with joins, grouping, subqueries; a hidden twin per row.",
        ),
        "gretel": DatasetSpec(
            hub_id="gretelai/synthetic_text_to_sql",
            split="test",
            convert=from_gretel,
            license="Apache-2.0",
            url="https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
            citation="Meyer et al., Synthetic-Text-To-SQL, Gretel.ai, 2024",
            description=(
                "Rows whose context and query run in SQLite and return something. No hidden twin: a result read "
                "off the data and hard-coded passes, so use it for evaluation, not as an RL reward."
            ),
        ),
    },
    difficulty={"option": "difficulty", "levels": [1, 2, 3], "row_field": "difficulty"},
    max_turns=8,
    max_new_tokens=768,
    oracle=oracle,
    degenerate={
        "select_star": select_star,
        "select_constant": select_constant,
        "hardcode_visible": hardcode_visible,
        "gold_then_drop": gold_then_drop,
        "gold_in_prose": gold_in_prose,
        "runaway": runaway,
    },
    tags=("multi-turn", "sqlite"),
)
