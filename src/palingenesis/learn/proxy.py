"""pgs learn proxy: an OpenAI-compatible front that serves each API key its own scope (stdlib only).

    pgs learn proxy STORE --keys keys.yaml --upstream http://127.0.0.1:8000 [--host 127.0.0.1] [--port 8100]

Per request:
  1. auth       the Bearer key's sha256 -> (scope, role, consent) from keys.yaml; the key is never forwarded
                (the upstream gets serving.api_key_env's key, if set)
  2. signals    on POST /v1/chat/completions from a user key: an explicit command in the NEWEST user message
                ("remember ...", "ricorda ...", as signals.detect_command defines; never assistant, system or
                tool text, earlier turns, quoted, fenced or pasted text) is stored in the key's own scope through
                the PII filter, at most signals.max_notes_per_hour per key; x-learn-note / x-learn-note-id tell
                the client. While a job holds the store lock, the signal is spooled and stored by the next job
  3. notes      the chain's notes the served composite does not hold yet (serving.wake_notes), rendered
                first in the system turn as quoted reference data, ancestors' notes first: the prefix stays
                the same between sessions, so the server's prefix cache keeps hitting
  4. routing    `model` becomes the scope's live published name (<scope>@v<n>, or the nearest published
                ancestor's; the base model when nothing is published); adapter-selection fields a client sends
                (lora_path, ...) are dropped, so a key reaches its own scope only
  5. forward    to the upstream, streaming (SSE) and tool calls passed through unchanged

Other routes: GET /v1/models (the key's scope), GET /health, POST /v1/completions (routed, no notes, no
signals), POST /v1/learn/corrections {prompt, correction}, GET /v1/learn/notes (the scope's notes),
POST /v1/learn/forget {ids}, POST /v1/learn/notes {notes, kind} (admin keys: into the scope's inbox, never
approved by the proxy). Everything else, the servers' LoRA load / unload routes included, is 404.

Nothing is logged by default: no request or response body is written anywhere. --access-log prints method,
path, status, scope and latency only.

keys.yaml:
    keys:
      - sha256: 9f86d081...        # pgs learn key STORE --scope ... prints a new key and appends its hash
        scope: user:acme/support/ana
        role: user                 # user | admin (admin: team / org notes into the inbox)
        name: ana                  # recorded as the origin of captured notes
        remember: true             # consent: explicit commands are captured (default true for users)
        corrections: true          # consent: /v1/learn/corrections (default true for users)
"""

from __future__ import annotations

import collections
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from palingenesis.learn.render import NOTES_TEMPLATE, note_statement, render_notes
from palingenesis.learn.signals import detect_command
from palingenesis.learn.store import ScopeId, StoreError, StoreLocked

ADAPTER_FIELDS = ("lora_path", "lora_id", "lora_name", "lora_request", "lora_int_id", "adapter", "adapter_name")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class KeyEntry:
    name: str
    scope: ScopeId
    role: str = "user"
    remember: bool = True
    corrections: bool = True
    digest: str = ""  # the key's sha256

    @property
    def limit_key(self) -> str:
        """The rate limit's key: the key itself (names repeat across scopes: pgs learn key defaults to the
        user's name)."""
        return self.digest or f"{self.scope}|{self.name}"


def load_keys(path: str | Path, org: str) -> dict[str, KeyEntry]:
    """{sha256 of key: KeyEntry} from keys.yaml; every scope must belong to the store's org."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for i, row in enumerate(raw.get("keys") or []):
        digest = row.get("sha256") or (hash_key(row["key"]) if row.get("key") else None)
        if not digest:
            raise StoreError(f"{path}: key {i} needs sha256 (or key)")
        scope = ScopeId.parse(row["scope"])
        if scope.org != org:
            raise StoreError(f"{path}: key {i} names {scope}, outside this store's org {org!r}")
        role = row.get("role", "user")
        if role not in ("user", "admin"):
            raise StoreError(f"{path}: key {i} role must be user or admin, got {role!r}")
        user = role == "user"
        out[digest] = KeyEntry(
            row.get("name", f"key{i}"),
            scope,
            role,
            bool(row.get("remember", user)) and user,
            bool(row.get("corrections", user)) and user,
            digest,
        )
    return out


def inject_notes(messages: list[dict], notes: list) -> list[dict]:
    """`messages` with the notes block first in the system turn (training's rendering: render.with_notes);
    a system turn given as content parts gets the block as its first text part."""
    if not notes:
        return list(messages)
    block = render_notes(notes)
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, list):
            return [{**messages[0], "content": [{"type": "text", "text": block + "\n\n"}, *content]}] + messages[1:]
        text = NOTES_TEMPLATE.format(privileged=block, content=content or "")
        return [{**messages[0], "content": text}] + list(messages[1:])
    return [{"role": "system", "content": block}] + list(messages)


class RateLimiter:
    """At most `limit` events per key in any `window` seconds (0 = no limit)."""

    def __init__(self, limit: int, window: float = 3600.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.events: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def allow(self, key: str, n: int = 1) -> bool:
        if self.limit <= 0:
            return True
        with self.lock:
            q, t = self.events[key], self.clock()
            while q and t - q[0] >= self.window:
                q.popleft()
            if len(q) + n > self.limit:
                return False
            q.extend([t] * n)
            return True


class LearnProxy:
    """The proxy's state: keys, the notes cache per scope, the rate limiter and the upstream connection."""

    def __init__(
        self,
        learner,
        keys: dict[str, KeyEntry],
        upstream: str,
        upstream_key: str = "",
        max_body_bytes: int = 8 << 20,
        access_log: bool = False,
        timeout: float = 600.0,
        lock_wait: float = 0.5,
    ):
        self.learner, self.store, self.keys = learner, learner.store, keys
        config = learner.config
        self.engine = config.serving.engine
        self.base_name = config.serving.base_name or self.store.state["base"]
        url = urllib.parse.urlsplit(upstream.rstrip("/"))
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"upstream must be an http(s) URL, got {upstream!r}")
        self.upstream, self.upstream_key = url, upstream_key
        self.max_body_bytes, self.access_log, self.timeout, self.lock_wait = (
            max_body_bytes,
            access_log,
            timeout,
            lock_wait,
        )
        self.limiter = RateLimiter(config.signals.max_notes_per_hour)
        self.mutex = threading.Lock()  # one store mutation at a time (the Store's lock is not thread-safe)
        self._cache: dict[str, tuple] = {}
        self._local = threading.local()

    # ------------------------------------------------------------------ routing and notes
    def _stamp(self, scope: ScopeId) -> tuple:
        paths = [self.store.root / "published.json"] + [self.store.scope(s).dir / "notes.jsonl" for s in scope.chain()]
        return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)

    def route(self, scope: ScopeId) -> tuple[str | None, list, int]:
        """(served adapter name or None for the base, wake notes within notes.wake_max_tokens, notes left out)."""
        from palingenesis.learn.serving import served_version, wake_notes

        stamp = self._stamp(scope)
        cached = self._cache.get(str(scope))
        if cached and cached[0] == stamp:
            return cached[1:]
        _, entry = served_version(self.store, scope)
        notes = wake_notes(self.store, scope, entry)
        budget = self.learner.config.notes.wake_max_tokens * 4
        kept, used = [], 0
        for note in notes:
            used += len(note_statement(note)) + 3
            if budget and used > budget:
                break
            kept.append(note)
        value = (entry["name"] if entry else None, kept, len(notes) - len(kept))
        self._cache[str(scope)] = (stamp, *value)
        return value

    def rewrite(self, entry: KeyEntry, body: dict, chat: bool) -> tuple[dict, dict]:
        """The upstream body (model routed, notes injected for chat, adapter fields dropped) and the x-learn
        response headers."""
        name, notes, dropped = self.route(entry.scope)
        body = {k: v for k, v in body.items() if k not in ADAPTER_FIELDS}
        if self.engine == "sglang":
            body["model"] = self.base_name
            if name:
                body["lora_path"] = name
        else:
            body["model"] = name or self.base_name
        headers = {"x-learn-version": name or "base"}
        if chat:
            body["messages"] = inject_notes(body.get("messages") or [], notes)
            headers["x-learn-notes"] = str(len(notes))
            if dropped:
                headers["x-learn-notes-dropped"] = str(dropped)
        return body, headers

    # ------------------------------------------------------------------ signals
    def _mutate(self, fn):
        """Run a store mutation under the store lock; StoreLocked when a job holds it."""
        with self.mutex, self.store.lock(timeout=self.lock_wait):
            return fn()

    def store_signal(self, entry: KeyEntry, kind: str, text: str, prompt: str = "") -> dict:
        origin = f"proxy:{entry.name}"
        try:

            def store():
                self.learner.drain_spool(entry.scope)  # earlier spooled signals first, in order
                if kind == "correction":
                    return self.learner.correct(entry.scope, prompt, text, origin=origin)
                return self.learner.remember(entry.scope, text, origin=origin)

            result = self._mutate(store)
        except StoreLocked:
            with self.mutex:
                result = self.learner.spool(entry.scope, kind, text, origin, prompt=prompt)
        return {"status": result.status, "id": result.note_id or result.inbox_id, "reason": result.reason}

    def capture(self, entry: KeyEntry, body: dict) -> dict:
        """x-learn headers for an explicit command in the newest user message (nothing otherwise)."""
        if not entry.remember or entry.scope.shared:
            return {}
        signals = self.learner.config.signals
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            return {}
        signal = detect_command(messages, signals.command_patterns, signals.max_note_chars)
        if signal is None:
            return {}
        if not self.limiter.allow(entry.limit_key):
            return {"x-learn-note": "rate_limited"}
        out = self.store_signal(entry, "remembered", signal.text)
        headers = {"x-learn-note": out["status"]}
        if out["id"]:
            headers["x-learn-note-id"] = out["id"]
        return headers

    # ------------------------------------------------------------------ upstream
    def _connection(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self.upstream.scheme == "https" else http.client.HTTPConnection
            conn = cls(self.upstream.hostname, self.upstream.port, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def forward(self, method: str, path: str, body: bytes | None) -> http.client.HTTPResponse:
        headers = {"Content-Type": "application/json"}
        if self.upstream_key:
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        prefix = self.upstream.path.rstrip("/")
        for attempt in (0, 1):
            conn = self._connection()
            try:
                conn.request(method, prefix + path, body=body, headers=headers)
                return conn.getresponse()
            except (http.client.HTTPException, ConnectionError, OSError) as e:
                conn.close()
                self._local.conn = None
                # One retry for a kept-alive connection the upstream closed; a timeout means the request was
                # sent and is still generating: retrying would run it twice and double the wait.
                if attempt or isinstance(e, TimeoutError):
                    raise
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ server
    def handler(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "pgs-learn-proxy"

            def log_message(self, *args):  # no request logging by default
                pass

            def _send(self, status: int, payload, headers: dict | None = None) -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)
                self._status = status

            def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
                self._send(status, {"error": {"message": message, "type": kind, "code": status}})

            def _auth(self) -> KeyEntry | None:
                header = self.headers.get("Authorization", "")
                key = header[7:].strip() if header.lower().startswith("bearer ") else ""
                entry = proxy.keys.get(hash_key(key)) if key else None
                if entry is None:
                    self._error(401, "missing or unknown API key", "authentication_error")
                self._scope = str(entry.scope) if entry else "-"
                return entry

            def _body(self) -> dict | None:
                # A chunked or malformed body is refused, and so is an oversized one, without reading it: the
                # connection closes instead (a negative length would read to EOF, a huge one into memory).
                raw_length = (self.headers.get("Content-Length") or "0").strip()
                if self.headers.get("Transfer-Encoding") or not (raw_length.isascii() and raw_length.isdigit()):
                    self.close_connection = True
                    self._error(411, "send the body with a Content-Length (no chunked transfer encoding)")
                    return None
                length = int(raw_length)
                self._consumed = True  # read below, or the connection closes
                if length > proxy.max_body_bytes:
                    self.close_connection = True
                    self._error(413, f"request body over {proxy.max_body_bytes} bytes")
                    return None
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._error(400, "the body is not JSON")
                    return None
                if not isinstance(body, dict):
                    self._error(400, "the body must be a JSON object")
                    return None
                return body

            def _relay(self, path: str, body: dict, headers: dict) -> None:
                data = json.dumps(body, ensure_ascii=False).encode()
                try:
                    resp = proxy.forward("POST", path, data)
                except (http.client.HTTPException, OSError) as e:
                    self._error(502, f"upstream unreachable: {e}", "upstream_error")
                    return
                if not body.get("stream"):
                    self._send(resp.status, resp.read(), headers)
                    return
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self._status = resp.status
                while chunk := resp.read1(65536):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def _dispatch(self, method: str) -> None:
                path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
                if method == "GET" and path == "/health":
                    self._send(200, {"status": "ok"})
                    return
                entry = self._auth()
                if entry is None:
                    return
                if method == "GET" and path == "/v1/models":
                    card = {"id": str(entry.scope), "object": "model", "owned_by": "palingenesis-learn"}
                    self._send(200, {"object": "list", "data": [card]})
                elif method == "GET" and path == "/v1/learn/notes":
                    self._send(200, proxy.list_notes(entry))
                elif method != "POST":
                    self._error(404, f"no route {method} {path}", "not_found_error")
                elif path in ("/v1/chat/completions", "/v1/completions"):
                    body = self._body()
                    if body is None:
                        return
                    chat = path == "/v1/chat/completions"
                    if chat and not isinstance(body.get("messages"), list):
                        self._error(400, "messages must be a list")
                        return
                    headers = proxy.capture(entry, body) if chat else {}
                    body, route_headers = proxy.rewrite(entry, body, chat)
                    self._relay(path, body, {**route_headers, **headers})
                elif path.startswith("/v1/learn/"):
                    body = self._body()
                    if body is not None:
                        status, payload = proxy.learn_route(entry, path, body)
                        self._send(status, payload)
                else:
                    # Adapter admin routes (/v1/load_lora_adapter, /load_lora_adapter, ...) are never forwarded.
                    self._error(404, f"no route {method} {path}", "not_found_error")

            def _run(self, method: str) -> None:
                started, self._status, self._scope, self._consumed = time.perf_counter(), 0, "-", False
                try:
                    self._dispatch(method)
                    if not self._consumed and (
                        self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0").strip() != "0"
                    ):
                        # Answered without reading the body (401, 404): its bytes must not be parsed as the next
                        # request on this connection (a desync behind a reverse proxy that reuses connections).
                        self.close_connection = True
                except (BrokenPipeError, ConnectionResetError):
                    # The client left mid-response: drop the upstream connection with its unread rest.
                    self.close_connection = True
                    conn = getattr(proxy._local, "conn", None)
                    if conn is not None:
                        conn.close()
                        proxy._local.conn = None
                except Exception as e:  # noqa: BLE001 - answer instead of dropping the connection; no body is logged
                    if not self._status:
                        self._error(500, f"proxy error: {type(e).__name__}", "server_error")
                    self.close_connection = True
                if proxy.access_log:
                    ms = (time.perf_counter() - started) * 1000
                    print(f"{method} {self.path} {self._status} {self._scope} {ms:.1f}ms", file=sys.stderr)

            def do_GET(self):
                self._run("GET")

            def do_POST(self):
                self._run("POST")

            def do_PUT(self):
                self._run("PUT")

            def do_DELETE(self):
                self._run("DELETE")

        return Handler

    def serve(self, host: str = "127.0.0.1", port: int = 8100, background: bool = True) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((host, port), self.handler())
        server.daemon_threads = True
        if background:
            threading.Thread(target=server.serve_forever, daemon=True, name="pgs-learn-proxy").start()
        return server

    # ------------------------------------------------------------------ /v1/learn routes
    def list_notes(self, entry: KeyEntry) -> dict:
        ss = self.store.scope(entry.scope)
        if not ss.exists:
            return {"scope": str(entry.scope), "notes": [], "pending": []}
        notes = [
            {"id": n.id, "kind": n.kind, "text": n.text, "consolidated": n.consolidated_in is not None}
            for n in ss.active_notes()
        ]
        pending = [
            {"id": i.id, "kind": i.kind, "text": (i.payload.get("note") or {}).get("text", ""), "reason": i.reason}
            for i in ss.inbox()
        ] + [{"id": r["id"], "kind": r["kind"], "text": r["text"], "reason": "spooled"} for r in ss.spool()]
        return {"scope": str(entry.scope), "notes": notes, "pending": pending}

    def learn_route(self, entry: KeyEntry, path: str, body: dict) -> tuple[int, dict]:
        def error(status, message):
            return status, {"error": {"message": message, "code": status}}

        if path in ("/v1/learn/corrections", "/v1/learn/forget") and entry.scope.shared:
            # Team and org knowledge changes through admins (pgs learn approve / forget), never a shared key.
            return error(403, f"{path} acts on a user scope; this key serves the shared scope {entry.scope}")
        if path == "/v1/learn/corrections":
            if not entry.corrections:
                return error(403, "this key does not send corrections")
            prompt, correction = body.get("prompt", ""), body.get("correction", "")
            if not isinstance(prompt, str) or not isinstance(correction, str) or not correction.strip():
                return error(400, "a correction needs {prompt, correction} strings")
            if len(correction) > self.learner.config.signals.max_note_chars:
                return error(400, f"a correction is at most {self.learner.config.signals.max_note_chars} characters")
            if not self.limiter.allow(entry.limit_key):
                return error(429, "too many notes this hour (signals.max_notes_per_hour)")
            return 200, self.store_signal(entry, "correction", correction, prompt)
        if path == "/v1/learn/forget":
            ids = body.get("ids")
            if entry.role != "user" or not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                return error(400, "forget takes {ids: [...]} of the key's own notes (user keys)")
            try:
                report = self._mutate(lambda: self.learner.forget(entry.scope, ids))
            except StoreLocked:
                return error(409, "a job holds the store; try again when it ends")
            except KeyError as e:
                return error(404, str(e))
            return 200, {"ok": report.ok, "summary": report.summary(), "unloaded": report.unloaded}
        if path == "/v1/learn/notes":
            if entry.role != "admin":
                return error(403, "notes are added by admin keys; users send explicit commands")
            notes, kind = body.get("notes"), body.get("kind", "fact")
            if not isinstance(notes, list) or not notes or not all(isinstance(n, str) for n in notes):
                return error(400, "notes takes {notes: [text, ...], kind?}")
            if kind not in ("fact", "instruction"):
                return error(400, "kind is fact or instruction")
            if any(len(n) > self.learner.config.signals.max_note_chars for n in notes):
                return error(400, f"a note is at most {self.learner.config.signals.max_note_chars} characters")
            if not self.limiter.allow(entry.limit_key, len(notes)):
                return error(429, "too many notes this hour (signals.max_notes_per_hour)")
            origin = f"proxy-admin:{entry.name}"
            try:
                results = self._mutate(
                    lambda: self.learner.add_notes(entry.scope, notes, kind=kind, approve=False, origin=origin)
                )
            except StoreLocked:
                return error(409, "a job holds the store; try again when it ends")
            except StoreError as e:
                return error(400, str(e))
            return 200, {
                "results": [{"status": r.status, "id": r.note_id or r.inbox_id, "reason": r.reason} for r in results]
            }
        return error(404, f"no route POST {path}")
