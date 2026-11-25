"""tools-filesystem, tools-calendar, tools-shop: multi-turn tasks on simulated stateful tool APIs.

Each row carries a world ("state"), a request, and the world the request should leave behind ("goal"), computed by
the generator by running the solution's calls on the same simulator. The policy acts through the environment's
tools; nothing it says changes the world. The reward is on the outcome:

  success = the final world equals the goal (compared canonically: file contents, the set of events, orders and
            stock; never the path taken) and, for requests that ask a question, the final message ends with the
            line "ANSWER: <value>" holding the right value.

So claiming to be done, reading without acting, or acting on more than was asked (cancelling every order instead
of the customer's unshipped ones) all score 0, and a question cannot be answered by listing candidates. The
simulators are plain Python objects reset from the row: no process, no I/O, thousands of steps per second.

Rows: {"prompt", "state", "goal", "expected_answer", "solution": [[tool, arguments], ...], "solution_answer"}.
"""

from __future__ import annotations

import copy
import posixpath
import random
import re
from typing import Any

from palingenesis.rl.catalog.runner import scripted, turn_index
from palingenesis.rl.catalog.spec import DatasetSpec, EnvSpec
from palingenesis.rl.env import ToolFailed

ANSWER_NOTE = '\n\nEnd your final message with a line of the form "ANSWER: <value>".'
_ANSWER = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.M)


def final_answer(messages: list[dict]) -> str | None:
    reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    found = _ANSWER.findall(reply)
    return found[-1] if found else None


def answers_match(got: str | None, want: str) -> bool:
    if got is None:
        return False
    a, b = got.strip().strip(".").lower(), str(want).strip().lower()
    if a == b:
        return True
    try:
        return abs(float(a.replace(",", "").removesuffix("euros").removesuffix("eur").strip()) - float(b)) < 1e-6
    except ValueError:
        return False


class StatefulEnv:
    """A simulated world reset from the row; subclasses add the tools and say how worlds compare."""

    def __init__(self):
        self.state: dict = {}
        self.goal: dict = {}
        self.expected_answer = ""

    def reset(self, state: dict | None = None, goal: dict | None = None, expected_answer: Any = "", **row) -> None:
        self.state = copy.deepcopy(state or {})
        self.goal = goal or {}
        self.expected_answer = "" if expected_answer is None else str(expected_answer)

    def canonical(self, state: dict) -> Any:
        raise NotImplementedError

    def get_reward(self, messages: list[dict]) -> dict[str, float]:
        world = self.canonical(self.state) == self.canonical(self.goal)
        answer = not self.expected_answer or answers_match(final_answer(messages), self.expected_answer)
        return {"success": float(world and answer), "metric/state": float(world), "metric/answer": float(answer)}


# ---------------------------------------------------------------- filesystem


def _path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ToolFailed(f"paths are absolute (start with '/'), got {path!r}")
    return posixpath.normpath(path)


class FilesystemEnv(StatefulEnv):
    """An in-memory file tree: state {"files": {path: content}, "dirs": [path, ...]}."""

    tools = ("list_dir", "read_file", "write_file", "make_dir", "move", "delete")

    def canonical(self, state: dict) -> Any:
        return {path: content.rstrip() for path, content in (state.get("files") or {}).items()}

    def _dirs(self) -> set[str]:
        dirs = {"/"} | set(self.state.setdefault("dirs", []))
        for path in self.state.setdefault("files", {}):
            while path != "/":
                path = posixpath.dirname(path)
                dirs.add(path)
        return dirs

    def list_dir(self, path: str) -> list[str]:
        """List the entries of a directory. Directories end with "/".

        Args:
            path: Absolute path of the directory.
        """
        path = _path(path)
        dirs = self._dirs()
        if path not in dirs:
            raise ToolFailed(f"no such directory: {path}")
        entries = {p for p in self.state["files"] if posixpath.dirname(p) == path}
        entries |= {d + "/" for d in dirs if d != "/" and posixpath.dirname(d) == path}
        return sorted(posixpath.basename(e.rstrip("/")) + ("/" if e.endswith("/") else "") for e in entries)

    def read_file(self, path: str) -> str:
        """Read a file.

        Args:
            path: Absolute path of the file.
        """
        path = _path(path)
        if path not in self.state["files"]:
            raise ToolFailed(f"no such file: {path}")
        return self.state["files"][path]

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file. Its directory must exist.

        Args:
            path: Absolute path of the file.
            content: The new content.
        """
        path = _path(path)
        dirs = self._dirs()
        if path in dirs:
            raise ToolFailed(f"{path} is a directory")
        if posixpath.dirname(path) not in dirs:
            raise ToolFailed(f"no such directory: {posixpath.dirname(path)} (create it with make_dir)")
        self.state["files"][path] = str(content)
        return f"wrote {path}"

    def make_dir(self, path: str) -> str:
        """Create a directory (and its parents).

        Args:
            path: Absolute path of the directory.
        """
        path = _path(path)
        if path in self.state["files"]:
            raise ToolFailed(f"{path} is a file")
        if path not in self._dirs():
            self.state["dirs"].append(path)
        return f"created {path}"

    def move(self, source: str, destination: str) -> str:
        """Move or rename a file. The destination's directory must exist.

        Args:
            source: Absolute path of the file to move.
            destination: Absolute path it should have afterwards.
        """
        source, destination = _path(source), _path(destination)
        files, dirs = self.state["files"], self._dirs()
        if source not in files:
            raise ToolFailed(f"no such file: {source}")
        if destination in dirs:
            destination = posixpath.join(destination, posixpath.basename(source))
        if posixpath.dirname(destination) not in dirs:
            raise ToolFailed(f"no such directory: {posixpath.dirname(destination)} (create it with make_dir)")
        source_dir = posixpath.dirname(source)
        files[destination] = files.pop(source)
        if source_dir not in self._dirs():
            self.state["dirs"].append(source_dir)  # a directory does not disappear with its last file
        return f"moved {source} to {destination}"

    def delete(self, path: str) -> str:
        """Delete a file, or a directory with everything in it.

        Args:
            path: Absolute path to delete.
        """
        path = _path(path)
        files = self.state["files"]
        if path in files:
            parent = posixpath.dirname(path)
            del files[path]
            if parent not in self._dirs():
                self.state["dirs"].append(parent)
            return f"deleted {path}"
        if path == "/" or path not in self._dirs():
            raise ToolFailed(f"no such file or directory: {path}")
        prefix = path + "/"
        for name in [p for p in files if p.startswith(prefix)]:
            del files[name]
        self.state["dirs"] = [d for d in self.state["dirs"] if d != path and not d.startswith(prefix)]
        return f"deleted {path}"


_WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "gamma", "harbor", "indigo", "juniper", "kelp", "lumen"]


def _token(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6))


def _fs_task(rng: random.Random) -> dict:
    kind = rng.choice(["archive", "config", "cleanup", "count", "locate"])
    names = rng.sample(_WORDS, 6)
    files: dict[str, str] = {f"/home/notes/{names[0]}.md": f"# {names[0]}\n{_token(rng)}\n"}
    solution: list = []
    answer = ""
    if kind == "archive":
        logs = [f"/srv/app/{n}.log" for n in names[1:4]]
        files.update({p: f"log {_token(rng)}\n" for p in logs})
        files.update({f"/srv/app/{names[4]}.conf": "mode=fast\n", f"/srv/app/data/{names[5]}.log": "nested\n"})
        prompt = "Move every .log file that is directly inside /srv/app into a new directory /archive/logs. Leave everything else where it is."
        solution = [["list_dir", {"path": "/srv/app"}], ["make_dir", {"path": "/archive/logs"}]]
        solution += [
            ["move", {"source": p, "destination": f"/archive/logs/{posixpath.basename(p)}"}] for p in sorted(logs)
        ]
    elif kind == "config":
        keys = rng.sample(["port", "region", "retries", "owner", "tier"], 4)
        values = {k: _token(rng) for k in keys}
        files["/etc/app/config.ini"] = "".join(f"{k} = {v}\n" for k, v in values.items())
        files["/etc/app/config.ini.bak"] = "".join(f"{k} = {_token(rng)}\n" for k in keys)
        key = keys[rng.randrange(4)]
        prompt = f"Read /etc/app/config.ini and write the value of the `{key}` setting, and nothing else, to a new file /out/{key}.txt."
        solution = [
            ["read_file", {"path": "/etc/app/config.ini"}],
            ["make_dir", {"path": "/out"}],
            ["write_file", {"path": f"/out/{key}.txt", "content": values[key]}],
        ]
    elif kind == "cleanup":
        marked = set(rng.sample(names[1:6], rng.randint(1, 3)))
        for n in names[1:6]:
            files[f"/tmp/{n}.txt"] = f"{_token(rng)} {'EXPIRED' if n in marked else 'active'} {_token(rng)}\n"
        prompt = "Delete every file in /tmp whose content contains the word EXPIRED. Keep the others."
        solution = [["list_dir", {"path": "/tmp"}]]
        solution += [["read_file", {"path": f"/tmp/{n}.txt"}] for n in sorted(names[1:6])]
        solution += [["delete", {"path": f"/tmp/{n}.txt"}] for n in sorted(marked)]
    elif kind == "count":
        total = 0
        for n in names[1:6]:
            folder = rng.choice(["/data", "/data/raw", "/data/raw/old"])
            extension = rng.choice(["csv", "csv", "json"])
            total += extension == "csv"
            files[f"{folder}/{n}.{extension}"] = _token(rng)
        files[f"/data/raw/{names[0]}.csv"], files[f"/data/{names[0]}_b.csv"] = "x", "y"
        files[f"/data/raw/old/{names[0]}.json"] = "{}"
        answer = str(total + 2)
        prompt = "How many .csv files are there under /data, in all of its subdirectories?" + ANSWER_NOTE
        solution = [["list_dir", {"path": p}] for p in ("/data", "/data/raw", "/data/raw/old")]
    else:
        code = _token(rng)
        holder = rng.choice(names[1:5])
        for n in names[1:5]:
            files[f"/inbox/{n}.txt"] = f"ref {code if n == holder else _token(rng)}\n"
        prompt = f"One file in /inbox contains the reference {code}. Move that file into a new directory /done, keeping its name."
        solution = [["list_dir", {"path": "/inbox"}]]
        solution += [["read_file", {"path": f"/inbox/{n}.txt"}] for n in sorted(names[1:5])]
        solution += [
            ["make_dir", {"path": "/done"}],
            ["move", {"source": f"/inbox/{holder}.txt", "destination": f"/done/{holder}.txt"}],
        ]
    return {
        "kind": kind,
        "prompt": prompt,
        "state": {"files": files, "dirs": []},
        "solution": solution,
        "answer": answer,
    }


# ------------------------------------------------------------------ calendar

_PEOPLE = ["Ana", "Bo", "Chidi", "Dara", "Emil", "Farah", "Goran", "Hana"]
_TITLES = ["Budget review", "Design sync", "Hiring panel", "Roadmap", "Retro", "Vendor call", "Onboarding", "Demo prep"]


def _hour(value: Any) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})(?::00)?", text)
    if not match or not 0 <= int(match.group(1)) <= 24:
        raise ToolFailed(f"times are whole hours as HH:00, got {value!r}")
    return int(match.group(1))


class CalendarEnv(StatefulEnv):
    """One person's calendar: state {"events": [{id, title, date, start, end, attendees}], "next_id": n}; events
    are whole hours and may not overlap."""

    tools = ("list_events", "create_event", "reschedule_event", "cancel_event")

    def canonical(self, state: dict) -> Any:
        return sorted(
            (
                e["title"].strip().lower(),
                e["date"],
                e["start"],
                e["end"],
                tuple(sorted(a.lower() for a in e["attendees"])),
            )
            for e in state.get("events") or []
        )

    @staticmethod
    def _show(event: dict) -> dict:
        return {**event, "start": f"{event['start']:02d}:00", "end": f"{event['end']:02d}:00"}

    def _find(self, event_id: str) -> dict:
        event = next((e for e in self.state["events"] if e["id"] == str(event_id)), None)
        if event is None:
            raise ToolFailed(f"no event with id {event_id!r}")
        return event

    def _check_free(self, date: str, start: int, end: int, ignore: str = "") -> None:
        if not 0 <= start < end <= 24:
            raise ToolFailed("an event must end after it starts, within one day")
        for other in self.state["events"]:
            if other["id"] != ignore and other["date"] == date and start < other["end"] and other["start"] < end:
                raise ToolFailed(
                    f"conflicts with {other['id']} ({other['title']}, {other['start']:02d}:00-{other['end']:02d}:00)"
                )

    def list_events(self, date: str) -> list[dict]:
        """List the events of a day, by start time.

        Args:
            date: The day, as YYYY-MM-DD.
        """
        return [self._show(e) for e in sorted(self.state["events"], key=lambda e: e["start"]) if e["date"] == date]

    def create_event(self, title: str, date: str, start: str, end: str, attendees: list[str]) -> dict:
        """Create an event. It may not overlap another event.

        Args:
            title: The event's title.
            date: The day, as YYYY-MM-DD.
            start: Start time, a whole hour as HH:00.
            end: End time, a whole hour as HH:00.
            attendees: First names of the people invited.
        """
        first, last = _hour(start), _hour(end)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            raise ToolFailed("dates are YYYY-MM-DD")
        self._check_free(date, first, last)
        event = {
            "id": f"E{self.state['next_id']}",
            "title": str(title),
            "date": date,
            "start": first,
            "end": last,
            "attendees": [str(a) for a in attendees or []],
        }
        self.state["next_id"] += 1
        self.state["events"].append(event)
        return self._show(event)

    def reschedule_event(self, event_id: str, date: str, start: str, end: str) -> dict:
        """Move an event to another day or time. It may not overlap another event.

        Args:
            event_id: The event's id.
            date: The new day, as YYYY-MM-DD.
            start: New start time, a whole hour as HH:00.
            end: New end time, a whole hour as HH:00.
        """
        event = self._find(event_id)
        first, last = _hour(start), _hour(end)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            raise ToolFailed("dates are YYYY-MM-DD")
        self._check_free(date, first, last, ignore=event["id"])
        event.update(date=date, start=first, end=last)
        return self._show(event)

    def cancel_event(self, event_id: str) -> str:
        """Cancel an event.

        Args:
            event_id: The event's id.
        """
        event = self._find(event_id)
        self.state["events"].remove(event)
        return f"cancelled {event['id']}"


def _calendar_task(rng: random.Random) -> dict:
    kind = rng.choice(["cancel_with", "move_after", "schedule", "hours"])
    day = f"2031-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"
    other_day = day[:-2] + str(int(day[-2:]) + 1)
    starts = sorted(rng.sample(range(8, 17), 5))
    titles = rng.sample(_TITLES, 7)
    events = []
    for k, start in enumerate(starts):
        last = starts[k + 1] if k + 1 < len(starts) else 18
        events.append(
            {
                "id": f"E{k + 1}",
                "title": titles[k],
                "date": day,
                "start": start,
                "end": min(last, start + rng.randint(1, 2)),
                "attendees": sorted(rng.sample(_PEOPLE, rng.randint(1, 3))),
            }
        )
    events.append(
        {
            "id": "E6",
            "title": titles[5],
            "date": other_day,
            "start": 10,
            "end": 11,
            "attendees": sorted(rng.sample(_PEOPLE, 2)),
        }
    )
    state = {"events": events, "next_id": 7}
    listing = ["list_events", {"date": day}]
    answer = ""

    def free_start(first: int, length: int, ignore: str = "") -> int | None:
        for start in range(first, 24 - length + 1):
            if all(
                e["id"] == ignore or e["date"] != day or not (start < e["end"] and e["start"] < start + length)
                for e in events
            ):
                return start
        return None

    if kind == "cancel_with":
        person = events[rng.randrange(5)]["attendees"][0]
        events[5]["attendees"] = sorted(set(events[5]["attendees"]) | {person})  # the same person, another day: stays
        prompt = f"Cancel every event on {day} that {person} attends. Do not touch other days."
        solution = [listing] + [["cancel_event", {"event_id": e["id"]}] for e in events[:5] if person in e["attendees"]]
    elif kind == "move_after":
        event = events[rng.randrange(2)]
        length = event["end"] - event["start"]
        start = free_start(13, length, ignore=event["id"])
        if start is None:
            return _calendar_task(rng)
        prompt = (
            f"Reschedule \"{event['title']}\" on {day} to later the same day: the earliest start at or after 13:00 at "
            "which it overlaps no other event, keeping its length."
        )
        arguments = {
            "event_id": event["id"],
            "date": day,
            "start": f"{start:02d}:00",
            "end": f"{start + length:02d}:00",
        }
        solution = [listing, ["reschedule_event", arguments]]
    elif kind == "schedule":
        people = sorted(rng.sample(_PEOPLE, 2))
        start = free_start(9, 1)
        if start is None:
            return _calendar_task(rng)
        prompt = (
            f'Schedule a one-hour event titled "{titles[6]}" with {people[0]} and {people[1]} on {day}, at the earliest '
            "free start time from 09:00 on."
        )
        arguments = {
            "title": titles[6],
            "date": day,
            "start": f"{start:02d}:00",
            "end": f"{start + 1:02d}:00",
            "attendees": people,
        }
        solution = [listing, ["create_event", arguments]]
    else:
        answer = str(sum(e["end"] - e["start"] for e in events[:5]))
        prompt = f"How many hours of events are on the calendar on {day} in total?" + ANSWER_NOTE
        solution = [listing]
    return {"kind": kind, "prompt": prompt, "state": state, "solution": solution, "answer": answer}


# ---------------------------------------------------------------------- shop

_CATEGORIES = {
    "lamps": ["Arc lamp", "Desk lamp", "Clip lamp", "Floor lamp"],
    "kettles": ["Steel kettle", "Glass kettle", "Travel kettle", "Gooseneck kettle"],
    "backpacks": ["Day pack", "Roll-top pack", "Camera pack", "Trail pack"],
}
_CUSTOMERS = ["c-ana", "c-bo", "c-chidi", "c-dara"]
_STREETS = ["Via Roma", "Rue Verte", "Calle Sol", "Hafenstrasse", "Mill Lane"]


class ShopEnv(StatefulEnv):
    """A shop's back office: state {"products": {id: {name, category, price, stock}}, "orders": {id: {customer,
    product_id, quantity, status, address}}, "next_order": n}. Orders can be changed only while "processing"."""

    tools = (
        "search_products",
        "get_product",
        "list_orders",
        "get_order",
        "place_order",
        "cancel_order",
        "update_address",
    )

    def canonical(self, state: dict) -> Any:
        orders = {
            oid: (o["customer"], o["product_id"], o["quantity"], o["status"], " ".join(o["address"].lower().split()))
            for oid, o in (state.get("orders") or {}).items()
        }
        return orders, {pid: p["stock"] for pid, p in (state.get("products") or {}).items()}

    def _order(self, order_id: str) -> dict:
        order = self.state["orders"].get(str(order_id))
        if order is None:
            raise ToolFailed(f"no order with id {order_id!r}")
        return order

    def search_products(self, category: str) -> list[dict]:
        """List the products of a category with price and stock.

        Args:
            category: The category name.
        """
        found = [
            {"product_id": pid, **p}
            for pid, p in self.state["products"].items()
            if p["category"] == str(category).lower()
        ]
        if not found:
            raise ToolFailed(
                f"no category {category!r}; categories: {', '.join(sorted({p['category'] for p in self.state['products'].values()}))}"
            )
        return found

    def get_product(self, product_id: str) -> dict:
        """Get one product.

        Args:
            product_id: The product's id.
        """
        if product_id not in self.state["products"]:
            raise ToolFailed(f"no product with id {product_id!r}")
        return {"product_id": product_id, **self.state["products"][product_id]}

    def list_orders(self, customer: str) -> list[dict]:
        """List a customer's orders.

        Args:
            customer: The customer's id.
        """
        return [{"order_id": oid, **o} for oid, o in self.state["orders"].items() if o["customer"] == customer]

    def get_order(self, order_id: str) -> dict:
        """Get one order.

        Args:
            order_id: The order's id.
        """
        return {"order_id": str(order_id), **self._order(order_id)}

    def place_order(self, customer: str, product_id: str, quantity: int, address: str) -> dict:
        """Place an order for a customer.

        Args:
            customer: The customer's id.
            product_id: The product to order.
            quantity: How many units.
            address: The shipping address.
        """
        product = self.state["products"].get(product_id)
        if product is None:
            raise ToolFailed(f"no product with id {product_id!r}")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ToolFailed("quantity must be a positive integer")
        if product["stock"] < quantity:
            raise ToolFailed(f"only {product['stock']} in stock")
        product["stock"] -= quantity
        order_id = f"o-{self.state['next_order']}"
        self.state["next_order"] += 1
        self.state["orders"][order_id] = {
            "customer": str(customer),
            "product_id": product_id,
            "quantity": quantity,
            "status": "processing",
            "address": str(address),
        }
        return {"order_id": order_id, **self.state["orders"][order_id]}

    def cancel_order(self, order_id: str) -> str:
        """Cancel an order that has not shipped; its units return to stock.

        Args:
            order_id: The order's id.
        """
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ToolFailed(f"order {order_id} is {order['status']} and can no longer be cancelled")
        order["status"] = "cancelled"
        self.state["products"][order["product_id"]]["stock"] += order["quantity"]
        return f"cancelled {order_id}"

    def update_address(self, order_id: str, address: str) -> str:
        """Change the shipping address of an order that has not shipped.

        Args:
            order_id: The order's id.
            address: The new shipping address.
        """
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ToolFailed(f"order {order_id} is {order['status']}; its address can no longer be changed")
        order["address"] = str(address)
        return f"updated {order_id}"


def _address(rng: random.Random) -> str:
    return f"{rng.randint(1, 120)} {rng.choice(_STREETS)}"


def _shop_task(rng: random.Random) -> dict:
    kind = rng.choice(["cancel_unshipped", "cheapest", "move_house", "delivered_total"])
    products, k = {}, 1
    for category, names in _CATEGORIES.items():
        prices = rng.sample(range(12, 160), len(names))
        for name, price in zip(names, prices):
            products[f"p-{k}"] = {"name": name, "category": category, "price": price, "stock": rng.randint(3, 20)}
            k += 1
    customer, other = rng.sample(_CUSTOMERS, 2)
    orders = {}
    statuses = ["processing", "processing", "shipped", "delivered", "delivered"]
    rng.shuffle(statuses)
    for i, status in enumerate(statuses, 1):
        orders[f"o-{i}"] = {
            "customer": customer,
            "product_id": rng.choice(list(products)),
            "quantity": rng.randint(1, 3),
            "status": status,
            "address": _address(rng),
        }
    for i, status in enumerate(["processing", "delivered"], 6):
        orders[f"o-{i}"] = {
            "customer": other,
            "product_id": rng.choice(list(products)),
            "quantity": rng.randint(1, 3),
            "status": status,
            "address": _address(rng),
        }
    state = {"products": products, "orders": orders, "next_order": 8}
    mine = ["list_orders", {"customer": customer}]
    answer = ""
    open_orders = [oid for oid, o in orders.items() if o["customer"] == customer and o["status"] == "processing"]
    if kind == "cancel_unshipped":
        prompt = (
            f"Customer {customer} wants to cancel all of their orders that have not shipped yet. Cancel exactly those."
        )
        solution = [mine] + [["cancel_order", {"order_id": oid}] for oid in open_orders]
    elif kind == "cheapest":
        category = rng.choice(list(_CATEGORIES))
        quantity, address = rng.randint(1, 3), _address(rng)
        cheapest = min(
            (pid for pid, p in products.items() if p["category"] == category), key=lambda pid: products[pid]["price"]
        )
        prompt = f"Order {quantity} of the cheapest product in the {category} category for customer {customer}, shipped to {address}."
        arguments = {"customer": customer, "product_id": cheapest, "quantity": quantity, "address": address}
        solution = [["search_products", {"category": category}], ["place_order", arguments]]
    elif kind == "move_house":
        address = _address(rng) + " (new)"
        prompt = f'Customer {customer} has moved to "{address}". Update the shipping address of every order of theirs that can still be changed.'
        solution = [mine] + [["update_address", {"order_id": oid, "address": address}] for oid in open_orders]
    else:
        delivered = [o for o in orders.values() if o["customer"] == customer and o["status"] == "delivered"]
        answer = str(sum(products[o["product_id"]]["price"] * o["quantity"] for o in delivered))
        prompt = (
            f"What is the total value in euros of customer {customer}'s delivered orders (price times quantity, summed)?"
            + ANSWER_NOTE
        )
        solution = [mine] + [
            ["get_product", {"product_id": pid}] for pid in sorted({o["product_id"] for o in delivered})
        ]
    return {"kind": kind, "prompt": prompt, "state": state, "solution": solution, "answer": answer}


# ------------------------------------------------------------------ generators


def _generate(name: str, env_class: type, make_task, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        task = make_task(rng)
        env = env_class()
        env.reset(state=task["state"])
        for tool, arguments in task["solution"]:
            getattr(env, tool)(**arguments)  # the simulator itself computes the goal
        changed = env.canonical(env.state) != env.canonical(task["state"])
        if not changed and not task["answer"]:
            continue  # a request that needs nothing done would pay doing nothing
        if task["answer"] in ("0", "1"):
            continue
        rows.append(
            {
                "id": f"{name}/synthetic/{seed}/{len(rows)}",
                "prompt": task["prompt"],
                "state": task["state"],
                "goal": env.state,
                "expected_answer": task["answer"],
                "solution": task["solution"],
                "solution_answer": f"ANSWER: {task['answer']}" if task["answer"] else "Done.",
                "task": task["kind"],
            }
        )
    return rows


def generate_filesystem(n: int, seed: int = 0) -> list[dict]:
    return _generate("tools-filesystem", FilesystemEnv, _fs_task, n, seed)


def generate_calendar(n: int, seed: int = 0) -> list[dict]:
    return _generate("tools-calendar", CalendarEnv, _calendar_task, n, seed)


def generate_shop(n: int, seed: int = 0) -> list[dict]:
    return _generate("tools-shop", ShopEnv, _shop_task, n, seed)


# ------------------------------------------------------------------- policies


def claim_done(row: dict, messages: list[dict]) -> str:
    return "All done, everything is taken care of.\nANSWER: 0"


def answer_listing(row: dict, messages: list[dict]) -> str:
    """Every small number as a candidate answer, the gold among them."""
    return "ANSWER: " + ", ".join(str(i) for i in range(0, 40)) + f", {row.get('expected_answer')}"


def look_then_claim(first_call: tuple[str, dict]):
    def policy(row: dict, messages: list[dict]) -> Any:
        return [first_call] if turn_index(messages) == 0 else "I checked and everything is in order.\nANSWER: unknown"

    return policy


def delete_everything(row: dict, messages: list[dict]) -> Any:
    if turn_index(messages) == 0:
        return [("delete", {"path": path}) for path in row["state"]["files"]]
    return "Cleaned up.\nANSWER: 0"


def cancel_everything(row: dict, messages: list[dict]) -> Any:
    if turn_index(messages) == 0:
        return [("cancel_event", {"event_id": e["id"]}) for e in row["state"]["events"]]
    return "Calendar cleared.\nANSWER: 0"


def cancel_every_order(row: dict, messages: list[dict]) -> Any:
    if turn_index(messages) == 0:
        return [("cancel_order", {"order_id": oid}) for oid in row["state"]["orders"]]
    return "All orders cancelled.\nANSWER: 0"


_VERIFIER = (
    "The world after the episode equals the goal world (computed by the simulator from the solution's calls; "
    "compared on content, not on the calls made), and a request that asks a question ends with the line "
    '"ANSWER: <value>" holding the right value.'
)


def _spec(name: str, summary: str, env: type, generator, attack: dict, first_call: tuple[str, dict]) -> EnvSpec:
    return EnvSpec(
        name=name,
        version="1.0.0",
        family="tool-calling",
        summary=summary,
        env=env,
        license="Apache-2.0",
        verifier=_VERIFIER,
        datasets={
            "synthetic": DatasetSpec(
                generator=generator, license="Apache-2.0", description="Generated worlds and requests."
            )
        },
        difficulty={"row_field": "task"},
        max_turns=16,
        max_new_tokens=512,
        oracle=scripted(),
        degenerate={
            "claim_done": claim_done,
            "answer_listing": answer_listing,
            "look_then_claim": look_then_claim(first_call),
            **attack,
        },
        tags=("multi-turn", "stateful", "function-calling"),
    )


FILESYSTEM = _spec(
    "tools-filesystem",
    "Multi-turn file management on a simulated file tree: move, extract, clean up, count.",
    FilesystemEnv,
    generate_filesystem,
    {"delete_everything": delete_everything},
    ("list_dir", {"path": "/"}),
)
CALENDAR = _spec(
    "tools-calendar",
    "Multi-turn scheduling on a simulated calendar: cancel, reschedule around conflicts, find a free slot.",
    CalendarEnv,
    generate_calendar,
    {"cancel_everything": cancel_everything},
    ("list_events", {"date": "2031-01-10"}),
)
SHOP = _spec(
    "tools-shop",
    "Multi-turn customer service on a simulated shop: orders that can and cannot be changed, stock, prices.",
    ShopEnv,
    generate_shop,
    {"cancel_every_order": cancel_every_order},
    ("search_products", {"category": "lamps"}),
)
