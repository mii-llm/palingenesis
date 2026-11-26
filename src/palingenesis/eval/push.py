"""eval_tasks transport push: the trainer's weights in memory, sent to the evaluator over TCP.

Trainer side (PushClient, rank 0). At a due step the weights are packed into byte buckets in ONE pinned host
staging area (the wire format of palingenesis.rl.pool.buckets, floats cast to bfloat16 as they are copied): that
copy is the only part on the training thread. A background thread then computes the buckets' checksums and sends
them, paced by eval_tasks.push_gbps; the staging area is free again when the evaluator acknowledges that the push
arrived whole. A due step that finds it still busy (or no evaluator connected) is skipped before anything is
gathered or copied, so the evaluator always gets the newest weights it can take and the trainer never waits. The
connection is retried every RECONNECT_SECONDS; nothing here raises into the training loop.

Evaluator side (serve, `pgs eval serve`). Pushes are received into one of two host areas while the engine
evaluates; one that arrives whole and with the checksums it was sent with becomes the `ready` snapshot, replacing a
ready one that was not started (that step is recorded as skipped: the newest weights win). Between two evaluations
the ready snapshot is loaded into the engine and evaluated (palingenesis.eval.service.EvalService); the result goes
to step_<n>/ in the run's directory and back to the trainer as a `result` frame.

Frames (palingenesis.rl.pool.protocol), trainer -> evaluator: hello {token, spec}, weights {step, buckets} + bucket
tensors, end {final_step}; evaluator -> trainer: hello {ok | error, gpus}, received {step} | rejected {step,
error}, result {step, status, metrics}. The spec travels in hello, so an evaluator on a node without the trainer's
file system (eval_tasks.service: host:port) needs no file from it; with one, results are also files.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import torch

from palingenesis.eval import handoff

logger = logging.getLogger(__name__)

RECONNECT_SECONDS = 10.0
PUSH_TIMEOUT = 600.0  # seconds a push may take before the connection is dropped (and the staging area freed)
BUCKET_BYTES = 256 * 2**20


async def paced_send(writer: asyncio.StreamWriter, header: dict, tensors: dict, gbps: float = 0.0) -> int:
    """protocol.send, at most `gbps` gigabits per second (0: as fast as the socket takes it)."""
    from palingenesis.rl.pool.protocol import CHUNK, pack

    entries, buffers = pack(tensors)
    head = json.dumps({**header, "tensors": entries}).encode()
    writer.write(struct.pack("<I", len(head)) + head)
    start, total = time.monotonic(), 0
    for buffer in buffers:
        for offset in range(0, len(buffer), CHUNK):
            chunk = buffer[offset : offset + CHUNK]
            writer.write(chunk)
            await writer.drain()
            total += len(chunk)
            if gbps > 0:
                ahead = total * 8 / (gbps * 1e9) - (time.monotonic() - start)
                if ahead > 0:
                    await asyncio.sleep(ahead)
    await writer.drain()
    return total


def service_address(output_dir: str | Path, spec) -> str:
    """host:port of the evaluator: eval_tasks.service, or the one service.json names (auto, devices)."""
    if spec.service and spec.service != "auto":
        return spec.service
    for service in handoff.services(output_dir):
        if service.get("mode") == "serve" and service.get("port") and service.get("state") not in ("ended", "broken"):
            return f"{service['host']}:{service['port']}"
    return ""


# ------------------------------------------------------------------ trainer


class PushClient:
    """The trainer's connection to `pgs eval serve` (see the module docstring)."""

    def __init__(self, output_dir: str | Path, spec, info, spec_json: dict | None = None):
        from palingenesis.rl.pool.protocol import Staging

        self.output_dir, self.spec, self.info = Path(output_dir), spec, info
        self.spec_json = spec_json or {}
        self.staging = Staging(pin=True)
        self.lock = threading.Lock()
        self.sending = False  # the staging area holds a push not yet acknowledged
        self.connected = False
        self.refused = ""  # the evaluator refused this trainer (its GPUs, a version): no more pushes
        self.results: list[dict] = []
        self.stats = {"pushes": 0, "push_failures": 0, "push_bytes": 0.0, "push_seconds": 0.0}
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.acks: asyncio.Queue | None = None
        self.next_connect = 0.0
        self.closing = False
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="pgs-eval-push", daemon=True)
        self.thread.start()

    # ------------------------------------------------------- connection

    def ready(self) -> bool:
        """Whether a push could start now (connected, nothing in flight); tries to connect when it is time."""
        if self.refused or self.closing:
            return False
        if not self.connected and time.monotonic() >= self.next_connect and self.loop.is_running():
            self.next_connect = time.monotonic() + RECONNECT_SECONDS
            asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
        with self.lock:
            return self.connected and not self.sending

    def poll(self) -> None:
        """Called by AsyncEval.poll: keeps trying to connect between due steps."""
        self.ready()

    async def _connect(self) -> None:
        from palingenesis.rl.pool.protocol import receive, send

        address = service_address(self.output_dir, self.spec)
        if not address or self.connected:
            return
        host, _, port = address.rpartition(":")
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 10.0)
            token = next((s.get("token") for s in handoff.services(self.output_dir) if s.get("token")), "")
            await send(writer, {"op": "hello", "token": token, "spec": self.spec_json})
            header, _ = await asyncio.wait_for(receive(reader), 60.0)
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as error:
            logger.debug("eval_tasks: no evaluator at %s yet (%s)", address, error)
            return
        if header.get("error"):
            self.refused = header["error"]
            logger.error("eval_tasks: the evaluator at %s refused this run: %s", address, header["error"])
            writer.close()
            return
        self.reader, self.writer, self.acks = reader, writer, asyncio.Queue()
        self.connected = True
        logger.info("eval_tasks: connected to the evaluator at %s", address)
        asyncio.ensure_future(self._listen(reader))

    async def _listen(self, reader) -> None:
        from palingenesis.rl.pool.protocol import receive

        try:
            while True:
                header, _ = await receive(reader)
                op = header.get("op")
                if op == "result":
                    with self.lock:
                        self.results.append(header)
                elif op in ("received", "rejected"):
                    await self.acks.put(header)
        except (OSError, asyncio.IncompleteReadError, ValueError):
            pass
        self._lost("the evaluator closed the connection")

    def _lost(self, reason: str) -> None:
        if self.connected:
            logger.warning("eval_tasks: evaluator connection lost (%s); training goes on, retrying", reason)
        self.connected = False
        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()
        self.writer = None
        with self.lock:
            self.sending = False

    # ------------------------------------------------------------- push

    def submit(self, step: int, named, wait: bool = False) -> bool:
        """Stage the weights of `step` and send them in the background. wait (the final step, after training):
        wait for a push in flight first, up to eval_tasks.final_timeout_s, and for this one to arrive."""
        from palingenesis.rl.pool import buckets as bucketed

        timeout = float(self.spec.final_timeout_s)
        if wait:
            deadline = time.monotonic() + timeout
            while not self.ready() and time.monotonic() < deadline and not self.refused:
                time.sleep(0.5)
        with self.lock:
            if not self.connected or self.sending:
                for _ in named:  # (collectives under FSDP)
                    pass
                return False
            self.sending = True
        try:
            snapshot = bucketed.pack(named, BUCKET_BYTES, self._allocate, torch.bfloat16, checksums=False)
            if self.staging.pin:
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 — a push that cannot be staged is skipped, never a training error
            logger.exception("eval_tasks: the weights of step %d could not be staged", step)
            with self.lock:
                self.sending = False
            self.stats["push_failures"] += 1
            return False
        future = asyncio.run_coroutine_threadsafe(self._push(step, snapshot), self.loop)
        if wait:
            with contextlib.suppress(Exception):
                future.result(timeout=max(1.0, timeout))
        return True

    def _allocate(self, index: int, nbytes: int) -> torch.Tensor:
        return self.staging.buffer(f"bucket/{index}", torch.uint8, [nbytes])

    async def _push(self, step: int, snapshot) -> None:
        from palingenesis.rl.pool import buckets as bucketed

        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            sums = await loop.run_in_executor(None, lambda: [bucketed.checksum(b) for b in snapshot.buffers])
            snapshot.sums = sums
            header = {"op": "weights", "step": step, "buckets": snapshot.spec("tcp")}
            sent = await asyncio.wait_for(
                paced_send(self.writer, header, snapshot.host(), float(self.spec.push_gbps)), PUSH_TIMEOUT
            )
            ack = await asyncio.wait_for(self.acks.get(), PUSH_TIMEOUT)
            if ack.get("op") == "rejected":
                self.stats["push_failures"] += 1
                logger.warning("eval_tasks: the evaluator rejected the push of step %d: %s", step, ack.get("error"))
            else:
                self.stats["pushes"] += 1
                self.stats["push_bytes"] += sent
            self.stats["push_seconds"] = time.monotonic() - started
            with self.lock:
                self.sending = False
        except (OSError, asyncio.TimeoutError, AttributeError) as error:
            self.stats["push_failures"] += 1
            self._lost(f"push of step {step}: {type(error).__name__}: {error}")

    def end(self, final_step: int) -> None:
        """The run ended: the evaluator finishes what it holds, then exits."""
        if not self.connected:
            return
        from palingenesis.rl.pool.protocol import send

        future = asyncio.run_coroutine_threadsafe(send(self.writer, {"op": "end", "final_step": final_step}), self.loop)
        with contextlib.suppress(Exception):
            future.result(timeout=10)

    def take_results(self) -> list[dict]:
        with self.lock:
            out, self.results = self.results, []
        return out

    def metrics(self) -> dict[str, float]:
        return {**{k: float(v) for k, v in self.stats.items()}, "connected": float(self.connected)}

    def close(self) -> None:
        self.closing = True

        async def stop() -> None:
            if self.writer is not None:
                self.writer.close()
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(stop(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


# ---------------------------------------------------------------- evaluator


class Receiver:
    """Two host areas: a push arrives in one while the other may hold the ready snapshot (see the module
    docstring). take() hands the ready snapshot to the evaluation loop."""

    def __init__(self):
        from palingenesis.rl.pool.protocol import Staging

        self.areas = [Staging(pin=False), Staging(pin=False)]
        self.changed = threading.Condition()
        self.ready: tuple[int, int, dict] | None = None  # (area, step, bucket spec)
        self.loading: int | None = None  # the area being loaded into the engine
        self.receiving: int | None = None
        self.ended: int | None = None  # the run's final step, once the trainer said so
        self.skipped: list[int] = []

    def area(self, header: dict):
        """The area a weights frame is received into: one neither ready nor loading (else the ready one: the
        newest weights win)."""
        with self.changed:
            busy = {self.loading, self.ready[0] if self.ready else None}
            free = [i for i in (0, 1) if i not in busy]
            if not free and self.ready is not None:
                self.skipped.append(self.ready[1])
                free, self.ready = [self.ready[0]], None
            self.receiving = free[0]
            return self.areas[free[0]]

    def arrived(self, header: dict, tensors: dict) -> str:
        """A whole weights frame: verified, then ready. Returns "" or what is wrong."""
        from palingenesis.rl.pool import buckets as bucketed

        spec = header["buckets"]
        buffers = [tensors[f"bucket/{i}"] for i in range(len(spec["sizes"])) if f"bucket/{i}" in tensors]
        if os.environ.get("PGS_TEST_CORRUPT") and os.path.exists(os.environ["PGS_TEST_CORRUPT"]) and buffers:
            buffers[-1][-1] ^= 1  # tests: one bit of the push arrives flipped
        wrong = bucketed.verify(spec, buffers)
        with self.changed:
            index, self.receiving = self.receiving, None
            if wrong:
                return wrong
            if self.ready is not None:
                self.skipped.append(self.ready[1])
            self.ready = (index, int(header["step"]), spec)
            self.changed.notify_all()
        return ""

    def take(self, timeout: float) -> tuple[int, Any] | None:
        """(step, a callable giving the weights) of the ready snapshot, or None after `timeout` seconds."""
        from palingenesis.rl.pool import buckets as bucketed

        with self.changed:
            if self.ready is None:
                self.changed.wait(timeout)
            if self.ready is None:
                return None
            index, step, spec = self.ready
            self.ready, self.loading = None, index
        buffers = [self.areas[index].buffers[f"bucket/{i}"] for i in range(len(spec["sizes"]))]
        return step, lambda: bucketed.unpack(spec, buffers).items()

    def loaded(self) -> None:
        with self.changed:
            self.loading = None

    def end(self, final_step: int) -> None:
        with self.changed:
            self.ended = final_step
            self.changed.notify_all()


def serve(
    output_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 0,
    engine_factory=None,
    parent_pid: int = 0,
    idle_timeout: float = 300.0,
    spec_timeout: float = 0.0,
    tokenizer=None,
) -> int:
    """`pgs eval serve`: listen, take the spec from the trainer's hello, evaluate what it pushes (see the module
    docstring). Exit code 0 at the run's end, 1 when refused or broken."""
    from palingenesis.eval.service import (
        EvalService,
        EvaluatorRefused,
        _alive,
        check_placement,
        own_gpus,
        write_service,
    )
    from palingenesis.rl.pool.protocol import receive, send

    output_dir = Path(output_dir)
    gpus = own_gpus()
    token = secrets.token_hex(16)
    receiver = Receiver()
    state: dict[str, Any] = {"spec": None, "writer": None, "refused": "", "last_seen": time.monotonic()}
    lock = threading.Lock()
    loop = asyncio.new_event_loop()

    async def reply(writer, header: dict) -> None:
        async with write_lock:
            await send(writer, header)

    def area(header: dict):
        return receiver.area(header) if header.get("op") == "weights" else None

    async def connection(reader, writer) -> None:
        try:
            while True:
                header, tensors = await receive(reader, area)
                op = header.get("op")
                state["last_seen"] = time.monotonic()
                if op == "hello":
                    error = ""
                    if header.get("token") and header["token"] != token:
                        error = "the token does not match this evaluator's service.json (another run's evaluator)"
                    spec = header.get("spec") or handoff.read_json(handoff.root(output_dir) / handoff.SPEC)
                    if not error and not isinstance(spec, dict):
                        error = "no spec: the trainer sent none and there is no spec.json"
                    if not error:
                        try:
                            check_placement(spec, gpus)
                        except EvaluatorRefused as refused:
                            error = state["refused"] = str(refused)  # on a GPU that trains: this evaluator stops
                    if error:
                        await reply(writer, {"op": "hello", "error": error, "gpus": [g["uuid"] for g in gpus]})
                        writer.close()
                        return
                    with lock:
                        state["spec"] = spec
                        state["writer"] = writer
                    receiver.ended = None
                    await reply(writer, {"op": "hello", "ok": True, "gpus": [g["uuid"] for g in gpus]})
                    with receiver.changed:
                        receiver.changed.notify_all()
                elif op == "weights":
                    wrong = receiver.arrived(header, tensors)
                    answer = {"op": "rejected", "error": wrong} if wrong else {"op": "received"}
                    await reply(writer, {**answer, "step": header.get("step")})
                elif op == "end":
                    receiver.end(int(header.get("final_step", 0)))
                elif op == "shutdown":
                    receiver.end(-1)
                    return
        except (OSError, asyncio.IncompleteReadError, ValueError):
            with receiver.changed:
                if receiver.receiving is not None:  # cut inside a push: nothing of it is used
                    receiver.receiving = None
        finally:
            with lock:
                if state["writer"] is writer:
                    state["writer"] = None
            with contextlib.suppress(Exception):
                writer.close()

    write_lock: asyncio.Lock | None = None
    servers: list = []

    async def start() -> int:
        nonlocal write_lock
        write_lock = asyncio.Lock()
        server = await asyncio.start_server(connection, host, port, limit=2**20)
        servers.append(server)
        return server.sockets[0].getsockname()[1]

    async def stop() -> None:
        for server in servers:
            server.close()
        with lock:
            writer = state["writer"]
        if writer is not None:
            writer.close()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    thread = threading.Thread(target=loop.run_forever, name="pgs-eval-serve", daemon=True)
    thread.start()
    bound = asyncio.run_coroutine_threadsafe(start(), loop).result()
    advertised = socket.gethostname() if host in ("0.0.0.0", "", "::") else host
    service_fields = {"port": bound, "token": token, "gpus": [g["uuid"] for g in gpus]}
    write_service(output_dir, "serve", "listening", host=advertised, **service_fields)
    print(f"eval_tasks evaluator listening on {advertised}:{bound}", flush=True)

    def send_result(result: dict) -> None:
        with lock:
            writer = state["writer"]
        if writer is not None:
            future = asyncio.run_coroutine_threadsafe(reply(writer, {"op": "result", **result}), loop)
            with contextlib.suppress(Exception):
                future.result(timeout=30)

    service = None
    code = 0
    try:
        while True:
            for skipped in receiver.skipped[:]:
                receiver.skipped.remove(skipped)
                send_result(handoff.write_result(output_dir, skipped, "skipped", reason="newer weights came"))
            taken = receiver.take(timeout=1.0)
            if taken is not None:
                step, weights = taken
                if service is None:
                    write_service(output_dir, "serve", "starting", host=advertised, **service_fields)
                    service = EvalService(output_dir, state["spec"], engine_factory, tokenizer)
                write_service(output_dir, "serve", "evaluating", host=advertised, step=step, **service_fields)
                result = service.evaluate(step, weights, loaded=receiver.loaded)
                send_result(result)
                if service.broken:
                    write_service(output_dir, "serve", "broken", host=advertised, **service_fields)
                    code = 1
                    break
                write_service(output_dir, "serve", "ready", host=advertised, **service_fields)
                continue
            if state["refused"]:
                write_service(output_dir, "serve", "refused", host=advertised, error=state["refused"], **service_fields)
                logger.error("%s", state["refused"])
                code = 1
                break
            if receiver.ended is not None and receiver.ready is None and receiver.receiving is None:
                break  # the run ended and nothing is left
            if parent_pid and not _alive(parent_pid):
                logger.warning("the trainer (pid %d) is gone: the evaluator stops", parent_pid)
                break
            with lock:
                connected = state["writer"] is not None
            if (
                idle_timeout
                and not connected
                and time.monotonic() - state["last_seen"] > idle_timeout
                and state["spec"]
            ):
                logger.warning("no trainer for %.0f s: the evaluator stops", idle_timeout)
                break
            if spec_timeout and state["spec"] is None and time.monotonic() - state["last_seen"] > spec_timeout:
                logger.error("no trainer connected within %.0f s", spec_timeout)
                code = 1
                break
    finally:
        if code == 0:
            write_service(output_dir, "serve", "ended", host=advertised, **service_fields)
        if service is not None:
            service.close()
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
    return code
