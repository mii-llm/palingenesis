"""Fakes for the pgs learn serving tests (CPU, no GPU, no vLLM): an in-process OpenAI-compatible server with
vLLM's (or SGLang's) LoRA routes, a character-level HF + PEFT reference, and store versions with tiny adapter
weights and evaluation reports, as run + eval leave them."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFIX = "U:"


def fake_lp(token: int, position: int, adapter: bool, scale: float = 1.0) -> float:
    """The log-probability both fakes assign to `token` at completion `position`; the adapter shifts it."""
    lp = -1.0 - 0.1 * ((token * 7 + position) % 5)
    if adapter:
        lp += scale * 0.4 * (((token * 3 + position) % 4) - 1.5)
    return lp


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


class CharReference:
    """The canary's HF + PEFT side: characters as tokens, answers from a {question: answer} table (adapter on)
    or "I don't know." (adapter off)."""

    def __init__(self, answers: dict[str, str], noise: float = 0.01):
        self.answers, self.noise = answers, noise
        self.loaded = []

    def __call__(self, composite):
        self.loaded.append(str(composite))
        return self

    def encode(self, messages):
        return encode(PREFIX + messages[-1]["content"])

    def decode(self, ids):
        return decode(ids)

    def greedy(self, ids, max_new_tokens, adapter):
        q = decode(ids)[len(PREFIX) :]
        return encode(self.answers.get(q, "I don't know.") if adapter else "I don't know.")[:max_new_tokens]

    def logprobs(self, ids, start, adapter):
        return [
            fake_lp(t, i, adapter) + self.noise * math.sin(i + 3 * adapter) for i, t in enumerate(ids) if i >= start
        ]


class FakeServer:
    """An OpenAI-compatible server with vLLM's runtime LoRA routes (engine vllm) or SGLang's (engine sglang).

    runtime: the LoRA routes exist (VLLM_ALLOW_RUNTIME_LORA_UPDATING); resolver: a callable name -> path or None
    called for unknown model names, like vLLM's resolver plugins; fault: None | "ignore_adapter" (the adapter is
    loaded but never applied) | "half_adapter" (half the modules dropped: the effect on odd positions only) |
    "wrong_answers" (the adapter answers like the base). Every request is recorded in `calls`.
    """

    def __init__(self, base="base-model", engine="vllm", runtime=True, resolver=None, answers=None, fault=None):
        self.base, self.engine, self.runtime, self.resolver = base, engine, runtime, resolver
        self.answers = answers or {}
        self.fault = fault
        self.loaded: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def paths(self, path):
        return [c for p, c in self.calls if p == path]

    # model semantics
    def _adapter(self, model: str | None) -> bool:
        if not model or model == self.base:
            return False
        if model not in self.loaded and self.resolver is not None:
            path = self.resolver(model)
            if path:
                self.loaded[model] = path
        if model not in self.loaded:
            raise KeyError(model)
        return self.fault != "ignore_adapter"

    def _lp(self, ids, adapter):
        out = []
        for i, t in enumerate(ids):
            on = adapter and not (self.fault == "half_adapter" and i % 2 == 0)
            out.append(fake_lp(t, i, on) + 0.01 * math.sin(i + 7 * on))
        return out

    def _answer(self, prompt, adapter):
        text = decode(prompt) if isinstance(prompt, list) else prompt
        q = text[len(PREFIX) :] if text.startswith(PREFIX) else text
        if adapter and self.fault != "wrong_answers":
            return self.answers.get(q, "I don't know.")
        return "I don't know."

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, payload, ctype="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.calls.append((self.path, {"_headers": dict(self.headers)}))
                if self.path == "/v1/models":
                    cards = [{"id": fake.base, "root": fake.base}] + [
                        {"id": n, "root": p} for n, p in fake.loaded.items()
                    ]
                    self._send(200, {"object": "list", "data": cards})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                body["_headers"] = {k.lower(): v for k, v in self.headers.items()}
                with fake.lock:
                    fake.calls.append((self.path, body))
                try:
                    self._route(body)
                except KeyError as e:
                    self._send(404, {"error": {"message": f"The model `{e.args[0]}` does not exist.", "code": 404}})

            def _route(self, body):
                path = self.path
                vllm = fake.engine == "vllm"
                if path in ("/v1/load_lora_adapter", "/load_lora_adapter"):
                    if not fake.runtime or vllm != (path == "/v1/load_lora_adapter"):
                        self._send(404, {"detail": "Not Found"})
                    elif body["lora_name"] in fake.loaded:
                        self._send(400, {"error": {"message": "has already been loaded", "code": 400}})
                    else:
                        fake.loaded[body["lora_name"]] = body["lora_path"]
                        self._send(200, b"Success" if vllm else {"success": True})
                elif path in ("/v1/unload_lora_adapter", "/unload_lora_adapter"):
                    if not fake.runtime or vllm != (path == "/v1/unload_lora_adapter"):
                        self._send(404, {"detail": "Not Found"})
                    elif body["lora_name"] not in fake.loaded:
                        msg = "The lora adapter cannot be found."
                        self._send(404, {"error": {"message": msg, "type": "NotFoundError", "code": 404}})
                    else:
                        del fake.loaded[body["lora_name"]]
                        self._send(200, b"Success" if vllm else {"success": True})
                elif path == "/v1/completions" and vllm:
                    adapter = fake._adapter(body.get("model"))
                    prompt = body["prompt"]
                    choice = {"index": 0, "text": fake._answer(prompt, adapter)[: body.get("max_tokens", 16)]}
                    if body.get("prompt_logprobs") is not None:
                        lps = fake._lp(prompt, adapter)
                        choice["prompt_logprobs"] = [None] + [
                            {str(t): {"logprob": lp, "rank": 1}} for t, lp in zip(prompt[1:], lps[1:])
                        ]
                    self._send(200, {"choices": [choice], "model": body.get("model")})
                elif path == "/generate" and not vllm:
                    name = body.get("lora_path")
                    adapter = fake._adapter(name)
                    prompt = body.get("input_ids") or body.get("text")
                    out = {"text": fake._answer(prompt, adapter), "meta_info": {}}
                    if body.get("return_logprob"):
                        lps = fake._lp(prompt, adapter)
                        out["meta_info"]["input_token_logprobs"] = [[None, prompt[0], None]] + [
                            [lp, t, None] for t, lp in zip(prompt[1:], lps[1:])
                        ]
                    self._send(200, out)
                elif path == "/v1/chat/completions":
                    fake._adapter(body.get("model") if vllm else body.get("lora_path"))
                    self._chat(body)
                else:
                    self._send(404, {"detail": "Not Found"})

            def _chat(self, body):
                message = {"role": "assistant", "content": "ok"}
                if body.get("tools"):
                    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}}
                    message = {"role": "assistant", "content": None, "tool_calls": [call]}
                if not body.get("stream"):
                    self._send(
                        200, {"id": "c1", "model": body.get("model"), "choices": [{"index": 0, "message": message}]}
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for event in stream_events(body.get("model"), message):
                    data = event.encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

        return Handler


def stream_events(model, message) -> list[str]:
    deltas = [{"role": "assistant", "content": ""}]
    if message.get("tool_calls"):
        deltas.append({"tool_calls": [dict(message["tool_calls"][0], index=0)]})
    else:
        deltas += [{"content": "o"}, {"content": "k"}]
    events = [
        "data: " + json.dumps({"id": "c1", "model": model, "choices": [{"index": 0, "delta": d}]}) + "\n\n"
        for d in deltas
    ]
    return events + ["data: [DONE]\n\n"]


# ---------------------------------------------------------------------------- store versions


def make_version(learner, scope, passed=True, rank=None):
    """A trained version of `scope` with tiny adapter weights and an evaluation report against the current live
    chain (passing, or failing retention), as run + eval leave them. Returns the version name."""
    import torch

    from palingenesis.learn import adapter as adapters
    from palingenesis.learn.evaluate import chain_versions
    from palingenesis.learn.generate import StubGenerator
    from palingenesis.learn.recipe import plan_session
    from palingenesis.learn.store import ScopeId, sha256_file
    from palingenesis.learn.trainer import write_session_files

    scope = ScopeId.parse(scope)
    ss = learner.store.scope(scope).ensure()
    rank = rank or learner.config.scopes.ranks[scope.level]
    plan = plan_session(ss, StubGenerator(), learner.config)
    number = ss.next_number()
    session, version = f"s{number:04d}", f"v{number:04d}"
    write_session_files(ss.dir / "sessions" / session, plan, learner.config)
    out = ss.version_dir(version)
    gen = torch.Generator().manual_seed(number + 100 * len(scope.path))
    tensors = {
        f"model.layers.{i}.mlp.{m}": (torch.randn(rank, 8, generator=gen), torch.randn(8, rank, generator=gen))
        for i in range(2)
        for m in ("gate_proj", "up_proj", "down_proj")
    }
    adapters.write_adapter(tensors, out, learner.store.state["base"])
    manifest = {
        "version": version,
        "scope": str(scope),
        "session": session,
        "rank": rank,
        "parent_versions": {s: v for s, v in chain_versions(learner.store, scope)[:-1] if v},
        "warm_start": ss.live,
        "note_ids": [n.id for n in ss.active_notes()],
        "fact_ids": [f.id for f in plan.new_facts],
        "sha256": sha256_file(out / adapters.ADAPTER_WEIGHTS),
        "gated": False,
        "forced": False,
        "pruned": None,
    }
    ss.write_manifest(version, manifest)
    from palingenesis.learn import gates as gating
    from palingenesis.learn import report as reporting
    from palingenesis.learn.evaluate import Answers, Probe, compute_metrics, kl_increment

    probes = [Probe("acquisition", "a?", ["f"], ["v"])] + [Probe("retention", f"r{i}?", ["g"], ["w"]) for i in range(5)]
    cand = ["v"] + ["no" if not passed else "w"] * 5
    metrics, _ = compute_metrics(probes, Answers(cand, ["no"] + ["w"] * 5, ["v"] + [None] * 5), [])
    metrics["drift"] = kl_increment(None, None, 0, 0)
    gates = gating.evaluate(metrics, learner.config.gates)
    report = {
        "scope": str(scope),
        "version": version,
        "session": session,
        "created": "2026-01-01T00:00:00Z",
        "live_chain": chain_versions(learner.store, scope),
        "candidate_chain": [],
        "calibrated": False,
        "metrics": metrics,
        "gates": [g.to_dict() for g in gates],
        "passed": gating.passed(gates),
        "warnings": [],
    }
    assert report["passed"] == passed
    reporting.write(ss.dir / "sessions" / session, report, [])
    manifest["evaluation"] = {"passed": passed, "failed": [] if passed else ["retention"]}
    ss.write_manifest(version, manifest)
    return version


def gold_answers(learner, scopes) -> dict[str, str]:
    """{probe question: an answer stating its facts' values} for the probes of `scopes`."""
    from palingenesis.learn.store import read_jsonl

    out = {}
    for scope in scopes:
        ss = learner.store.scope(scope)
        facts = {f.id: f for f in ss.facts()}
        for session in ss.session_dirs():
            for row in read_jsonl(session / "probes.jsonl"):
                values = [facts[f].value for f in row.get("fact_ids", []) if f in facts]
                if row.get("kind") == "probe" and values:
                    out[row["q"]] = "It is " + " and ".join(values) + "."
    return out
