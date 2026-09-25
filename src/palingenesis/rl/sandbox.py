"""Code execution for rewards and tools: one async interface, three backends.

  docker         a warm pool of long-lived local containers, each running the stdlib
                 worker (sandbox_worker.py) as its only process tree: no network,
                 read-only root, tmpfs work dirs, nobody user, no capabilities, pid and
                 memory ceilings. Jobs are multiplexed over the container's stdin/stdout,
                 so there is no per-job container start (docker run costs 0.5-2 s).
  agent_sandbox  a pool of Kubernetes Agent Sandbox pods (agent-sandbox.sigs.k8s.io),
                 claimed once from a SandboxWarmPool and reused for many jobs (a claim
                 costs seconds). Isolation is the pool's RuntimeClass (gVisor, Kata);
                 the same worker runs each job inside the pod.
  subprocess     the worker's limits without any isolation, in this process's threads.
                 For development and tests of trusted code only.

Every backend runs the same worker, so limits and result semantics are identical, and
programs only ever receive their inputs: outputs are compared by the caller.
"""

import asyncio
import itertools
import json
import logging
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from palingenesis.rl.sandbox_worker import run_job

logger = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("sandbox_worker.py")


@dataclass
class ExecJob:
    """One program run: `files` in a fresh directory, `argv` run there with `stdin`."""

    files: dict[str, str]
    argv: list[str] = field(default_factory=lambda: ["{python}", "-I", "-B", "main.py"])
    stdin: str = ""
    timeout: float = 6.0
    memory_mb: int = 1024
    max_output: int = 1 << 16
    env: dict[str, str] = field(default_factory=dict)

    def payload(self, job_id: int) -> dict:
        return {
            "id": job_id,
            "files": self.files,
            "argv": self.argv,
            "stdin": self.stdin,
            "timeout": self.timeout,
            "memory_mb": self.memory_mb,
            "max_output": self.max_output,
            "env": self.env,
        }


@dataclass
class ExecResult:
    status: str  # ok | runtime_error | timeout | memory | output_limit | sandbox_error
    stdout: str = ""
    stderr: str = ""
    result: str = ""  # what the program reported in its result file ($PGS_RESULT)
    exit_code: int | None = None
    wall: float = 0.0

    @property
    def infra_error(self) -> bool:
        """The sandbox failed, not the program: never score it as a wrong answer."""
        return self.status == "sandbox_error"

    @classmethod
    def from_worker(cls, raw: dict) -> "ExecResult":
        return cls(
            raw.get("status", "sandbox_error"),
            raw.get("stdout", ""),
            raw.get("stderr", ""),
            raw.get("result", ""),
            raw.get("exit_code"),
            raw.get("wall", 0.0),
        )


class Sandbox(Protocol):
    async def run(self, jobs: list[ExecJob]) -> list[ExecResult]:
        """Run the jobs concurrently (up to the backend's capacity); results in order."""

    async def close(self) -> None: ...


def make_sandbox(config) -> Sandbox:
    """The backend `config` (a SandboxConfig) selects."""
    if config.backend == "docker":
        return DockerSandbox(config.image, config.workers, config.slots_per_worker, config.memory_mb)
    if config.backend == "agent_sandbox":
        return AgentSandbox(
            config.warmpool,
            config.namespace,
            config.connection,
            config.workers,
            config.slots_per_worker,
        )
    return SubprocessSandbox(config.workers * config.slots_per_worker)


class SubprocessSandbox:
    """The worker's limits in local processes, no isolation (trusted code only)."""

    def __init__(self, slots: int = 8):
        self.pool = ThreadPoolExecutor(slots, thread_name_prefix="pgs-sandbox")
        self.ids = itertools.count()

    async def run(self, jobs: list[ExecJob]) -> list[ExecResult]:
        loop = asyncio.get_running_loop()
        raw = await asyncio.gather(*(loop.run_in_executor(self.pool, run_job, j.payload(next(self.ids))) for j in jobs))
        return [ExecResult.from_worker(r) for r in raw]

    async def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------- docker


class _Container:
    """One long-lived container running the worker; jobs multiplexed by id."""

    def __init__(self, argv: list[str], slots: int):
        self.argv = argv
        self.slots = asyncio.Semaphore(slots)
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.reader: asyncio.Task | None = None
        self.write_lock = asyncio.Lock()
        self.jobs = 0

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.reader = asyncio.create_task(self._read())
        self.jobs = 0

    async def _read(self) -> None:
        try:
            while True:
                header = await self.proc.stdout.readexactly(4)
                (size,) = struct.unpack(">I", header)
                result = json.loads(await self.proc.stdout.readexactly(size))
                future = self.pending.pop(result.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(result)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:  # the container died: fail what it held
            for future in self.pending.values():
                if not future.done():
                    future.set_result(
                        {
                            "status": "sandbox_error",
                            "stderr": "sandbox container exited",
                        }
                    )
            self.pending.clear()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None and not self.reader.done()

    async def submit(self, payload: dict) -> dict:
        future = asyncio.get_running_loop().create_future()
        self.pending[payload["id"]] = future
        data = json.dumps(payload).encode()
        async with self.write_lock:
            self.proc.stdin.write(struct.pack(">I", len(data)) + data)
            await self.proc.stdin.drain()
        self.jobs += 1
        try:
            return await asyncio.wait_for(future, payload["timeout"] + 30)
        except asyncio.TimeoutError:
            self.pending.pop(payload["id"], None)
            return {
                "status": "sandbox_error",
                "stderr": "no answer from the sandbox container",
            }

    async def stop(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()


class DockerSandbox:
    """A warm pool of locked-down containers; see the module docstring."""

    RECYCLE_AFTER = 5000  # jobs per container before it is replaced (state hygiene)

    def __init__(
        self,
        image: str = "python:3.12-slim",
        workers: int = 4,
        slots: int = 4,
        memory_mb: int = 1024,
    ):
        if shutil.which("docker") is None:
            raise RuntimeError("sandbox.backend docker: the docker CLI is not installed or not on PATH")
        self.image = image
        argv = [
            "docker",
            "run",
            "-i",
            "--rm",
            "--init",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/work:rw,nosuid,nodev,noexec,mode=1777,size=512m",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,mode=1777,size=64m",
            "--user",
            "65534:65534",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(64 * slots),
            "--memory",
            f"{memory_mb * slots + 256}m",
            "--cpus",
            str(slots),
            "--ulimit",
            "core=0",
            "--ipc",
            "none",
            "-e",
            "PGS_SANDBOX_ROOT=/work",
            "-v",
            f"{WORKER}:/pgs/worker.py:ro",
            image,
            "python",
            "-u",
            "/pgs/worker.py",
            "--slots",
            str(slots),
        ]
        self.containers = [_Container(argv, slots) for _ in range(workers)]
        self.ids = itertools.count()
        self.started = False
        self.start_lock: asyncio.Lock | None = None

    async def _ensure_started(self) -> None:
        if self.started:
            return
        self.start_lock = self.start_lock or asyncio.Lock()
        async with self.start_lock:
            if self.started:
                return
            probe = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                self.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            if await probe.wait() != 0:
                logger.info("Pulling sandbox image %s", self.image)
                pull = await asyncio.create_subprocess_exec("docker", "pull", "-q", self.image)
                if await pull.wait() != 0:
                    raise RuntimeError(f"docker pull {self.image} failed")
            await asyncio.gather(*(c.start() for c in self.containers))
            logger.info(
                "Sandbox: %d docker containers of %s ready",
                len(self.containers),
                self.image,
            )
            self.started = True

    async def _one(self, job: ExecJob) -> ExecResult:
        container = min(self.containers, key=lambda c: len(c.pending))
        async with container.slots:
            if not container.alive or (container.jobs >= self.RECYCLE_AFTER and not container.pending):
                await container.stop()
                await container.start()
            return ExecResult.from_worker(await container.submit(job.payload(next(self.ids))))

    async def run(self, jobs: list[ExecJob]) -> list[ExecResult]:
        await self._ensure_started()
        return list(await asyncio.gather(*(self._one(j) for j in jobs)))

    async def close(self) -> None:
        await asyncio.gather(*(c.stop() for c in self.containers))


# ---------------------------------------------------------------- agent sandbox


class AgentSandbox:
    """Kubernetes Agent Sandbox pods, claimed once from a warm pool and reused."""

    def __init__(
        self,
        warmpool: str,
        namespace: str = "default",
        connection: str = "in_cluster",
        workers: int = 4,
        slots: int = 4,
    ):
        try:
            import k8s_agent_sandbox  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sandbox.backend agent_sandbox needs the Agent Sandbox client: pip install 'k8s-agent-sandbox[async]'"
            ) from e
        self.warmpool, self.namespace, self.connection = warmpool, namespace, connection
        self.workers, self.slots = workers, slots
        self.client = None
        self.pods: list[object] = []
        self.busy: list[int] = []
        self.ids = itertools.count()
        self.start_lock: asyncio.Lock | None = None

    def _connection_config(self):
        from k8s_agent_sandbox.models import (
            SandboxDirectConnectionConfig,
            SandboxGatewayConnectionConfig,
            SandboxInClusterConnectionConfig,
        )

        if self.connection.startswith("gateway:"):
            name, _, namespace = self.connection[len("gateway:") :].partition("/")
            return SandboxGatewayConnectionConfig(gateway_name=name, gateway_namespace=namespace or "default")
        if self.connection.startswith("url:"):
            return SandboxDirectConnectionConfig(api_url=self.connection[len("url:") :])
        return SandboxInClusterConnectionConfig()

    async def _ensure_started(self) -> None:
        if self.pods:
            return
        self.start_lock = self.start_lock or asyncio.Lock()
        async with self.start_lock:
            if self.pods:
                return
            from k8s_agent_sandbox import AsyncSandboxClient

            self.client = AsyncSandboxClient(connection_config=self._connection_config())
            pods = await asyncio.gather(
                *(self.client.create_sandbox(self.warmpool, namespace=self.namespace) for _ in range(self.workers))
            )
            source = WORKER.read_text()
            await asyncio.gather(*(pod.files.write("pgs_worker.py", source) for pod in pods))
            self.pods, self.busy = list(pods), [0] * len(pods)
            self.slot_limit = asyncio.Semaphore(self.slots * len(pods))
            logger.info(
                "Sandbox: %d agent-sandbox pods claimed from warm pool %s/%s",
                len(pods),
                self.namespace,
                self.warmpool,
            )

    async def _one(self, job: ExecJob) -> ExecResult:
        job_id = next(self.ids)
        name = f"pgs_job_{job_id}.json"
        async with self.slot_limit:
            k = min(range(len(self.pods)), key=self.busy.__getitem__)  # the least loaded pod
            pod = self.pods[k]
            self.busy[k] += 1
            try:
                await pod.files.write(name, json.dumps(job.payload(job_id)))
                out = await pod.commands.run(
                    f"python3 pgs_worker.py --job-file {name}",
                    timeout=int(job.timeout) + 30,
                )
                return ExecResult.from_worker(json.loads(out.stdout))
            except Exception as e:  # noqa: BLE001 — a transport failure is an infrastructure error
                return ExecResult("sandbox_error", stderr=f"{type(e).__name__}: {e}")
            finally:
                self.busy[k] -= 1

    async def run(self, jobs: list[ExecJob]) -> list[ExecResult]:
        await self._ensure_started()
        return list(await asyncio.gather(*(self._one(j) for j in jobs)))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.delete_all()
            await self.client.close()
