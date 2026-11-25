"""GPU check of rollout.concurrency kv (palingenesis.rl.concurrency) against a real vLLM server.

    $PGS_PY scripts/gpu_checks/rl_concurrency.py --work $WORK [--model Qwen/Qwen3-0.6B] [--repeats 3]

Starts one `vllm serve` (its own process group; only that group is killed at the end) with a deliberately small KV
cache, then sends the same burst of long completions three times per repeat: all at once (rollout.concurrency
fixed), through the KV-sized gate with its defaults (kv), and through a gate that knows the requests' length and
fills the cache further before it cuts (kv_tight: rollout.concurrency_request_tokens, kv_usage_low 0.8,
kv_usage_high 0.95). Every gate starts cold, so each burst includes the cap's ramp. Prints JSON lines {"check", "status": PASS | FAIL, ...numbers}:

  concurrency/metrics      the server's /metrics and /v1/models give what the controller reads
  concurrency/start_cap    the gate's first cap is KV-cache tokens / max_model_len, within its bounds
  concurrency/burst        per mode: completion tokens/s, preemptions, peak KV usage, requests completed (median
                           over repeats); kv also the cap's range and final value
  concurrency/kv_vs_fixed  the caps stayed in bounds, and the better of the two kv settings reaches --min-ratio x
                           fixed's tokens/s with fewer preemptions (`reason` says what failed; both ratios given)

Everything it writes (the server log, results.jsonl) stays in --work. Needs vLLM; HF_HUB_OFFLINE=1 is set.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if os.environ.get("PGS_SRC"):  # the library under test (a checkout's src/), ahead of an installed copy
    sys.path.insert(0, os.environ["PGS_SRC"])

OUT: Path | None = None


def emit(record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if OUT is not None:
        with OUT.open("a") as f:
            f.write(line + "\n")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """`vllm serve` in a process group of its own."""

    def __init__(self, args: argparse.Namespace, work: Path):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = (work / "vllm_server.log").open("w")
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
        ]
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        self.process = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, cwd=work, start_new_session=True
        )
        atexit.register(self.stop)

    def wait(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                self.log.flush()
                tail = Path(self.log.name).read_bytes()[-600:].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"vllm serve exited with code {self.process.returncode}; vllm_server.log ends with: {tail}"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vllm serve not ready after {timeout:.0f} s")

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
            except ProcessLookupError:
                pass
        self.log.close()


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=3600) as r:
        return json.loads(r.read())


async def burst(url: str, model: str, args: argparse.Namespace, seed: int, gate=None) -> dict:
    from palingenesis.rl.concurrency import ServerMetrics

    rng = random.Random(seed)
    prompts = [[rng.randrange(1000, 100000) for _ in range(args.prompt_tokens)] for _ in range(args.requests)]
    metrics = ServerMetrics(url)
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(args.requests)
    before = await loop.run_in_executor(None, metrics.load)
    peak_usage, caps, done = 0.0, [], {"ok": 0, "failed": 0, "tokens": 0}

    async def monitor():
        nonlocal peak_usage
        while True:
            load = await loop.run_in_executor(None, metrics.load)
            if load is not None and load.kv_usage is not None:
                peak_usage = max(peak_usage, load.kv_usage)
            if gate is not None:
                caps.append(gate.slots[""].controller.cap)
            await asyncio.sleep(0.5)

    async def one(prompt):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 1.0,
            "ignore_eos": True,
            "seed": rng.randrange(1 << 30),
        }
        key = await gate.acquire() if gate is not None else None
        held = 0
        try:
            reply = await loop.run_in_executor(pool, post, url, payload)
            done["ok"] += 1
            done["tokens"] += int(reply["usage"]["completion_tokens"])
            held = len(prompt) + int(reply["usage"]["completion_tokens"])
        except Exception as error:  # noqa: BLE001 — counted, reported
            done["failed"] += 1
            done.setdefault("error", f"{type(error).__name__}: {error}"[:300])
        finally:
            if gate is not None:
                gate.release(key, held)  # (the tokens the request held size the cap)

    watcher = asyncio.ensure_future(monitor())
    start = time.perf_counter()
    await asyncio.gather(*(one(p) for p in prompts))
    seconds = time.perf_counter() - start
    watcher.cancel()
    pool.shutdown(wait=False)
    after = await loop.run_in_executor(None, metrics.load)
    preempted = None
    if before is not None and after is not None and before.preemptions is not None and after.preemptions is not None:
        preempted = after.preemptions - before.preemptions
    out = {
        "seconds": seconds,
        "tok_s": done["tokens"] / seconds,
        "completed": done["ok"],
        "failed": done["failed"],
        "first_error": done.get("error"),
        "preemptions": preempted,
        "kv_usage_peak": peak_usage,
    }
    if gate is not None:
        controller = gate.slots[""].controller
        out.update(
            {
                "cap_min": min(caps, default=controller.cap),
                "cap_max": max(caps, default=controller.cap),
                "cap_final": controller.cap,
                "cuts": controller.cuts,
                "raises": controller.raises,
            }
        )
    return out


def median(runs: list[dict], key: str):
    values = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)  # a small KV cache: preemptions
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-ratio", type=float, default=0.8)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    work = Path(args.work).resolve() / "rl_concurrency"
    work.mkdir(parents=True, exist_ok=True)
    OUT = work / "results.jsonl"

    from palingenesis.rl.concurrency import ConcurrencySettings, ServerMetrics, server_gate

    server = Server(args, work)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    failed = False
    try:
        server.wait(args.startup_timeout)
        metrics = ServerMetrics(server.url)
        load, length = metrics.load(), metrics.max_model_len()
        ok = load is not None and load.kv_usage is not None and load.waiting is not None and bool(load.kv_tokens)
        ok = ok and length == args.max_model_len
        emit(
            {
                "check": "concurrency/metrics",
                "status": "PASS" if ok else "FAIL",
                "kv_tokens": load.kv_tokens if load else None,
                "kv_usage": load.kv_usage if load else None,
                "waiting": load.waiting if load else None,
                "preemptions": load.preemptions if load else None,
                "max_model_len": length,
            }
        )
        failed |= not ok
        # Two settings of the controller: the defaults (TRL's thresholds; the cap starts from max_model_len), and a
        # tight one that knows what a request holds when it ends and fills the KV cache further before it cuts.
        request_tokens = args.prompt_tokens + args.max_tokens
        common = {"minimum": 4, "maximum": args.max_num_seqs, "poll_s": 1.0, "max_model_len": args.max_model_len}
        modes = {
            "kv": ConcurrencySettings(low=0.6, high=0.8, **common),
            "kv_tight": ConcurrencySettings(low=0.8, high=0.95, request_tokens=request_tokens, **common),
        }
        runs: dict[str, list[dict]] = {"fixed": [], **{mode: [] for mode in modes}}
        starts: list[int] = []
        for repeat in range(args.repeats):
            runs["fixed"].append(asyncio.run(burst(server.url, args.model, args, seed=repeat)))
            for mode, settings in modes.items():
                gate = server_gate(settings, server.url)

                async def gated(gate=gate, repeat=repeat, mode=mode):
                    key = await gate.acquire()  # the first cap, before the burst
                    if mode == "kv":
                        starts.append(gate.slots[""].controller.cap)
                    gate.release(key)
                    try:
                        return await burst(server.url, args.model, args, seed=repeat, gate=gate)
                    finally:
                        gate.close()

                runs[mode].append(asyncio.run(gated()))
        expected = max(4, min(args.max_num_seqs, (load.kv_tokens or 0) // args.max_model_len)) if load else None
        ok = bool(starts) and all(s == expected for s in starts)
        emit(
            {"check": "concurrency/start_cap", "status": "PASS" if ok else "FAIL", "caps": starts, "expected": expected}
        )
        failed |= not ok
        summary = {}
        for mode, mode_runs in runs.items():
            keys = ["tok_s", "seconds", "preemptions", "kv_usage_peak", "completed", "failed"]
            if mode != "fixed":
                keys += ["cap_min", "cap_max", "cap_final", "cuts", "raises"]
            summary[mode] = {key: median(mode_runs, key) for key in keys}
            complete = all(r["completed"] == args.requests and not r["failed"] for r in mode_runs)
            errors = [r["first_error"] for r in mode_runs if r.get("first_error")]
            emit(
                {
                    "check": "concurrency/burst",
                    "mode": mode,
                    "status": "PASS" if complete else "FAIL",
                    "requests": args.requests,
                    "max_tokens": args.max_tokens,
                    "repeats": args.repeats,
                    **summary[mode],
                    **({"first_error": errors[0]} if errors else {}),
                }
            )
            failed |= not complete
        fixed = summary["fixed"]
        ratios, reasons, bounded = {}, [], True
        for mode, settings in modes.items():
            bounded &= all(settings.minimum <= r["cap_min"] and r["cap_max"] <= settings.maximum for r in runs[mode])
            ratios[mode] = (summary[mode]["tok_s"] or 0.0) / max(fixed["tok_s"] or 0.0, 1e-9)
        # the better setting must reach min_ratio of fixed's tokens/s, with fewer preemptions than fixed when fixed
        # preempted at all
        best = max(ratios, key=ratios.get)
        fewer = not fixed["preemptions"] or (summary[best]["preemptions"] or 0.0) < fixed["preemptions"]
        if not bounded:
            reasons.append("a cap left [minimum, maximum]")
        if ratios[best] < args.min_ratio:
            reasons.append(
                f"the best setting ({best}) reached {ratios[best]:.2f} of fixed's tokens/s, below {args.min_ratio}"
            )
        if not fewer:
            reasons.append(f"{best} preempted as much as fixed")
        emit(
            {
                "check": "concurrency/kv_vs_fixed",
                "status": "PASS" if not reasons else "FAIL",
                **({"reason": "; ".join(reasons)} if reasons else {}),
                "tok_s_ratio": ratios,
                "best": best,
                "caps_in_bounds": bounded,
                "preemptions": {mode: values["preemptions"] for mode, values in summary.items()},
                "request_tokens": request_tokens,
                "kv_tokens": load.kv_tokens if load else None,
                "min_ratio": args.min_ratio,
            }
        )
        failed |= bool(reasons)
    except Exception as error:  # noqa: BLE001 — reported as a failed check
        emit({"check": "concurrency/run", "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
        failed = True
    finally:
        server.stop()
    if failed:  # the cause on stderr too, for whoever reads only the log
        print(f"rl_concurrency: failed checks, see the FAIL lines in {OUT}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
