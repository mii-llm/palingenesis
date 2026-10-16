"""`pgs harness`: record what a real agent harness sends to its model, to check a profile against it.

    pgs harness serve   --script turns.json --out requests.jsonl [--host 127.0.0.1] [--port 8000] [--model NAME]
    pgs harness capture <agent> --task <task_dir> --script turns.json --out <dir> [--model NAME] [--harbor 0.24.0]
                        [--host-ip IP] [--ak key=value ...] [--ae KEY=VALUE ...]
    pgs harness profiles

serve: a stub OpenAI-compatible endpoint (/v1/chat/completions, streaming or not; /v1/models) that answers from a
script and appends every request body, exactly as received, to a JSONL file. capture: the same stub, then the real
agent run once through the Harbor CLI (`uv run --with harbor==<version> harbor run -p <task> -a <agent> ...`) against
it; <dir>/requests.jsonl is the fixture a conformance test diffs a profile against (tests/test_harness_profiles.py
reads research/tiny-reliable-agent/harness/fixtures/<agent>/requests.jsonl when present). Fixtures stay local: some
harness prompts are not ours to redistribute.

Script (JSON): a list of assistant turns, each {"tool_calls": [{"name": ..., "arguments": {...}}]} or {"content":
"..."} (text protocols: the text the harness parses). Turns advance per conversation (requests sharing their first two
messages); a side request without tools (a title, a summary) gets a short text instead of a tool call; past
the end of the script the stub answers {"content": final} (default "Done."). Responses / Anthropic Messages requests
are recorded and refused (501).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Stub:
    """The scripted endpoint's state: the script, the request log, the served model name."""

    def __init__(self, script: list[dict], out: str | Path, model: str = "qwen3.5-0.8b", final: str = "Done."):
        self.script, self.out, self.model, self.final = list(script), Path(out), model, final
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.turns: dict[str, int] = {}
        self.out.parent.mkdir(parents=True, exist_ok=True)

    def record(self, path: str, body) -> None:
        entry = {"t": round(time.time(), 3), "path": path, "body": body}
        with self.lock:
            self.requests.append(entry)
            with open(self.out, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def turn(self, body: dict) -> dict:
        """The next scripted turn of this conversation (keyed by its first two messages; a harness that drops a
        rejected turn from its history, e.g. mini-swe-agent after a format error, still advances)."""
        messages = body.get("messages") or []
        key = json.dumps(messages[:2], sort_keys=True, default=str)
        with self.lock:
            k = self.turns.get(key, 0)
            self.turns[key] = k + 1
        turn = self.script[k] if k < len(self.script) else {"content": self.final}
        if turn.get("tool_calls") and not body.get("tools"):
            return {"content": "Capture"}  # a side request without tools (a title, a summary)
        return turn


def _message(turn: dict) -> tuple[dict, str]:
    calls = [{"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
              "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
             for c in turn.get("tool_calls") or []]
    msg = {"role": "assistant", "content": turn.get("content") or ("" if calls else "Done.")}
    if calls:
        msg["tool_calls"] = calls
    return msg, "tool_calls" if calls else "stop"


def make_handler(stub: Stub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                return self._json(200, {"object": "list", "data": [{"id": stub.model, "object": "model",
                                                                     "owned_by": "pgs-harness-capture"}]})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {"_raw": raw.decode(errors="replace")}
            stub.record(self.path, body)
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._json(501, {"error": {"message": f"{self.path} is not served by the capture stub"}})
            msg, finish = _message(stub.turn(body))
            rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if not body.get("stream"):
                return self._json(200, {"id": rid, "object": "chat.completion", "created": created,
                                        "model": body.get("model", stub.model),
                                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                        "usage": usage})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def chunk(delta: dict, finish_reason=None, **extra):
                obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": body.get("model", stub.model),
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())

            chunk({"role": "assistant", "content": ""})
            if msg["content"]:
                chunk({"content": msg["content"]})
            for i, c in enumerate(msg.get("tool_calls") or []):
                chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                                       "function": {"name": c["function"]["name"], "arguments": ""}}]})
                chunk({"tool_calls": [{"index": i, "function": {"arguments": c["function"]["arguments"]}}]})
            chunk({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                self.wfile.write(f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': body.get('model', stub.model), 'choices': [], 'usage': usage})}\n\n".encode())  # noqa: E501
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def serve(stub: Stub, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Start the stub in a daemon thread; returns the server (server.server_address has the port)."""
    server = ThreadingHTTPServer((host, port), make_handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True, name="pgs-harness-stub").start()
    return server


# ------------------------------------------------------------------ Harbor CLI capture

def harbor_agent_args(agent: str, model: str, base_url: str) -> tuple[list[str], dict[str, str]]:
    """(harbor run arguments, agent environment) that point a Harbor 0.24 agent at the stub (REPORT §1)."""
    key = "pgs-capture"
    env = {"OPENAI_API_KEY": key, "OPENAI_BASE_URL": base_url, "OPENAI_API_BASE": base_url}
    if agent == "opencode":
        config = {"provider": {"vllm": {"npm": "@ai-sdk/openai-compatible", "name": "vllm",
                                        "options": {"baseURL": base_url, "apiKey": key},
                                        "models": {model: {"name": model}}}}}
        return ["-m", f"vllm/{model}", "--ak", f"opencode_config={json.dumps(config)}"], env
    if agent == "pi":
        return ["-m", f"openai/{model}", "--ak", "model_api=openai-completions"], env
    if agent in ("openhands-sdk", "openhands"):
        env.update({"LLM_MODEL": f"openai/{model}", "LLM_BASE_URL": base_url, "LLM_API_KEY": key})
        return ["-m", f"openai/{model}"], env
    return ["-m", f"openai/{model}"], env  # mini-swe-agent, qwen-coder, goose, kimi-cli (with base_url), ...


def capture(agent: str, task: str, script: list[dict], out: str | Path, model: str = "qwen3.5-0.8b",
            harbor: str = "0.24.0", host_ip: str | None = None, extra_ak: list[str] = (), extra_ae: list[str] = (),
            timeout: float = 1800) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "requests.jsonl"
    if log.exists():
        log.unlink()
    stub = Stub(script, log, model)
    server = serve(stub, "0.0.0.0", 0)
    port = server.server_address[1]
    ip = host_ip or subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
    base = f"http://{ip}:{port}/v1"
    args, env = harbor_agent_args(agent, model, base)
    for kv in extra_ae:
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = ["uv", "run", "--with", f"harbor=={harbor}", "harbor", "run", "-p", str(task), "-a", agent, *args,
           *[x for kv in extra_ak for x in ("--ak", kv)], *[x for k, v in env.items() for x in ("--ae", f"{k}={v}")],
           "--allow-agent-host", ip, "-n", "1", "-o", str(out / "jobs"), "-y", "--job-name", f"capture_{agent}_{time.strftime('%Y%m%d_%H%M%S')}"]
    (out / "command.json").write_text(json.dumps({"cmd": cmd, "base_url": base}, indent=1))
    try:
        with open(out / "harbor.log", "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env={**os.environ, **env},
                           check=False)
    finally:
        server.shutdown()
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgs harness", description="harness profiles: list, record a harness's requests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profiles", help="list the built-in harness profiles")
    s = sub.add_parser("serve", help="run the recording stub endpoint")
    c = sub.add_parser("capture", help="run a real agent once (Harbor CLI) against the recording stub")
    c.add_argument("agent")
    c.add_argument("--task", required=True)
    c.add_argument("--harbor", default="0.24.0")
    c.add_argument("--host-ip", default=None)
    c.add_argument("--ak", action="append", default=[])
    c.add_argument("--ae", action="append", default=[])
    for p in (s, c):
        p.add_argument("--script", required=True, help="JSON list of assistant turns")
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="qwen3.5-0.8b")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if a.cmd == "profiles":
        from palingenesis.rl.envs.harness import available, get_profile

        for key in available():
            print(f"{key:32s} {get_profile(key).emulates[:100]}")
        return 0
    script = json.loads(Path(a.script).read_text())
    if a.cmd == "serve":
        server = serve(Stub(script, a.out, a.model), a.host, a.port)
        print(f"recording stub on http://{a.host}:{server.server_address[1]}/v1 -> {a.out}", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.shutdown()
        return 0
    log = capture(a.agent, a.task, script, a.out, a.model, a.harbor, a.host_ip, a.ak, a.ae)
    n = sum(1 for _ in open(log)) if log.exists() else 0
    print(f"{n} requests recorded -> {log}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
