"""The evaluation's engines: one persistent engine per model, tuned for throughput, replicated across GPUs.

    options, layout = engine_layout(config.engine)       # rollout.* options, and how many engines where
    engine = start_engines(model, rl_config, tokenizer, stop_ids, layout)   # in-process, or an EnginePool

One engine serves every task and mode of a run (no start per task). What `engine_options` adds to the RL rollout
defaults, unless the config says otherwise:

    prefix_caching            on: few-shot examples, system prompts and tool schemas are prefilled once
    max_num_seqs              512 sequences decoding at once (vLLM's own default is 256)
    max_num_batched_tokens    16384 tokens per engine step, so prefill is not rationed to 2048 tokens per step
    max_logprobs              32 next-token candidates per loglikelihood request (palingenesis.eval.loglikelihood)
    tensor_parallel_size, speculative_config    passed to vLLM (a model over several GPUs; speculative decoding)

Data parallelism. With several visible GPUs (engine.devices, default all of them) each gets its own engine
process; engine.replicas puts several engines on one GPU, which pays for small models: a sub-2B model leaves an
A100 idle most of the time, because the engine's scheduling and sampling loop is one Python thread, and more
processes share the GPU better than one does. The engines are palingenesis.rl.pool workers on local ports;
requests go to the least loaded one. The parent process holds no model.

GpuMonitor samples nvidia-smi while a task runs (utilisation and memory, mean and peak) for the report.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from palingenesis.eval.config import ENGINE_DEFAULTS, EVAL_ENGINE_KEYS

logger = logging.getLogger(__name__)

EVAL_VLLM_ARGS = {"max_num_batched_tokens": 16384}
EVAL_MAX_NUM_SEQS = 512
DEFAULT_MAX_LOGPROBS = 32


@dataclass
class Layout:
    """Where the engines of a run are: `devices` (GPU indices; empty: the process's own device), `replicas` engines
    on each, and the next-token candidates a loglikelihood request may ask for."""

    devices: list[int] = field(default_factory=list)
    replicas: int = 1
    max_logprobs: int = DEFAULT_MAX_LOGPROBS

    @property
    def engines(self) -> int:
        return max(1, len(self.devices)) * self.replicas

    @property
    def pooled(self) -> bool:
        return self.engines > 1


def visible_gpus() -> list[int]:
    """Indices of the GPUs this process may use (CUDA_VISIBLE_DEVICES order), without initialising CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return list(range(len([d for d in visible.split(",") if d.strip()])))
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        )
        return [int(line) for line in out.stdout.split() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def engine_layout(engine: dict, gpus: list[int] | None = None) -> tuple[dict, Layout]:
    """(the RL rollout options of an `engine:` section with the evaluation's defaults, the engines' layout)."""
    options = {**ENGINE_DEFAULTS, **{k: v for k, v in engine.items() if k not in EVAL_ENGINE_KEYS}}
    backend = options.get("backend", "vllm")
    max_logprobs = int(engine.get("max_logprobs") or DEFAULT_MAX_LOGPROBS)
    tensor_parallel = int(engine.get("tensor_parallel_size") or 1)
    if backend == "vllm":
        options.setdefault("max_num_seqs", EVAL_MAX_NUM_SEQS)
        extra = {**EVAL_VLLM_ARGS, "max_logprobs": max_logprobs, **(options.get("vllm_args") or {})}
        if tensor_parallel > 1:
            extra["tensor_parallel_size"] = tensor_parallel
        if engine.get("speculative_config"):
            extra["speculative_config"] = engine["speculative_config"]
        options["vllm_args"] = extra
        max_logprobs = int(extra["max_logprobs"])
    devices = engine.get("devices")
    if isinstance(devices, str):
        devices = [int(d) for d in devices.split(",") if d.strip()]
    if devices is None:
        gpus = visible_gpus() if gpus is None else gpus
        devices = gpus if len(gpus) > 1 and tensor_parallel == 1 and backend != "hf" else []
    replicas = max(1, int(engine.get("replicas") or 1))
    return options, Layout([int(d) for d in devices], replicas, max_logprobs)


# ----------------------------------------------------------------------- pool


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker_command(model: str, config, port: int, share: float, chat_template_kwargs: dict, stop_tokens) -> list[str]:
    """The palingenesis.rl.pool worker of one engine of the pool."""
    r = config.rollout
    extra = dict(r.vllm_args)
    extra.pop("max_logprobs", None)  # the worker's own flag
    command = [
        sys.executable, "-m", "palingenesis.rl.pool.worker", "--model", model, "--port", str(port),
        "--backend", r.backend, "--gpu-memory-utilization", f"{share:.4f}", "--max-model-len", str(r.max_model_len),
        "--max-num-seqs", str(r.max_num_seqs or EVAL_MAX_NUM_SEQS), "--seed", str(config.train.seed),
        "--chat-template-kwargs", json.dumps(chat_template_kwargs), "--stop-tokens", json.dumps(list(stop_tokens)),
        "--max-logprobs", str(r.vllm_args.get("max_logprobs", DEFAULT_MAX_LOGPROBS)),
    ]  # fmt: skip
    if r.backend == "vllm":
        command += ["--vllm-args", json.dumps(extra)]
    elif r.backend == "sglang":
        command += ["--sglang-args", json.dumps(r.sglang_args)]
    if r.fp32_head:
        command.append("--fp32-head")
    return command


class EnginePool:
    """Engine processes on local ports and the client that spreads requests over them.

    commands: one worker command per engine (worker_command, with its port); environments: each worker's own
    environment variables (CUDA_VISIBLE_DEVICES). Closing the pool stops the processes it started, nothing else.
    updatable: update_weights sends new weights to every engine and returns when all of them applied them (the
    asynchronous evaluator of palingenesis.eval.service); otherwise the pool serves the checkpoint it started with."""

    version = 0

    def __init__(
        self,
        commands: list[list[str]],
        ports: list[int],
        environments: list[dict],
        timeout: float = 1800.0,
        updatable: bool = False,
    ):
        from palingenesis.rl.pool.client import RemotePool

        self.processes: list[subprocess.Popen] = []
        self.pool = None
        try:
            for command, environment in zip(commands, environments):
                self.processes.append(
                    subprocess.Popen(command, env={**os.environ, **environment}, start_new_session=True)  # noqa: S603
                )
            watcher = threading.Thread(target=self._watch, daemon=True)
            watcher.start()
            addresses = [f"127.0.0.1:{port}" for port in ports]
            self.updatable = updatable
            self.pool = RemotePool(
                addresses, push_to=None if updatable else [], blocking=updatable, connect_timeout=timeout
            )
        except BaseException:
            self.close()
            raise
        logger.info("engine pool: %d engines ready", len(ports))

    def _watch(self) -> None:
        """A worker that dies while the pool waits for it (out of memory at start) is reported at once."""
        while self.pool is None and self.processes:
            for process in self.processes:
                if process.poll() not in (None, 0):
                    logger.error("an engine worker exited with code %s while starting", process.returncode)
                    return
            time.sleep(1.0)

    def make_client(self, merge: bool = True, adapter: str = ""):
        return self.pool.make_client(merge=merge, adapter=adapter)

    def update_weights(self, named_tensors, version: int) -> None:
        if not getattr(self, "updatable", False):
            raise NotImplementedError("an evaluation's engine pool serves a fixed checkpoint")
        self.pool.update_weights(named_tensors, version)  # blocking: every engine applied it
        self.version = version

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def close(self) -> None:
        if self.pool is not None:
            try:
                self.pool.close(shutdown=True)
            except Exception:  # noqa: BLE001 — the processes are stopped below either way
                logger.warning("engine pool: close failed", exc_info=True)
            self.pool = None
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, 15)
                except (ProcessLookupError, PermissionError):
                    pass
        deadline = time.monotonic() + 20
        for process in self.processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, 9)
                except (ProcessLookupError, PermissionError):
                    pass
        self.processes = []


def start_engines(
    model: str,
    config,
    tokenizer,
    stop_ids,
    layout: Layout,
    chat_template_kwargs=None,
    stop_tokens=(),
    pooled: bool = False,
    updatable: bool = False,
):
    """The run's engine: one in this process (palingenesis.rl.screen's), or an EnginePool of layout.engines.
    pooled: an EnginePool even for one engine (its failures stay in the worker process); updatable: the pool takes
    weight updates (EnginePool)."""
    if not (layout.pooled or pooled):
        if layout.devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = _device_name(layout.devices[0])
        from palingenesis.eval.runner import build_engine

        return build_engine(config, model, tokenizer, stop_ids)
    share = config.rollout.gpu_memory_utilization / layout.replicas
    commands, ports, environments = [], [], []
    for device in layout.devices or [None]:
        for _ in range(layout.replicas):
            port = _free_port()
            ports.append(port)
            commands.append(worker_command(model, config, port, share, chat_template_kwargs or {}, stop_tokens))
            environments.append({} if device is None else {"CUDA_VISIBLE_DEVICES": _device_name(device)})
    logger.info(
        "engine pool: %d engines (%d per GPU, GPUs %s), %.2f of each GPU's memory per engine",
        len(ports), layout.replicas, layout.devices or "default", share,
    )  # fmt: skip
    return EnginePool(commands, ports, environments, updatable=updatable)


def _device_name(index: int) -> str:
    """The CUDA_VISIBLE_DEVICES entry of the index-th GPU this process sees."""
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    return visible[index] if index < len(visible) else str(index)


# -------------------------------------------------------------------- monitor


class GpuMonitor:
    """GPU utilisation and memory while a block runs, from nvidia-smi (nothing where there is none).

    with GpuMonitor() as gpu: ...
    gpu.stats()  ->  {"gpu_util_mean": 87.0, "gpu_util_max": 100.0, "gpu_memory_max_gb": 31.2}
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMonitor":
        if shutil.which("nvidia-smi") is not None:
            self._thread = threading.Thread(target=self._run, name="pgs-eval-gpu", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and all(d.strip().isdigit() for d in visible.split(",")):
            query += ["-i", visible]
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(query, capture_output=True, text=True, timeout=5).stdout  # noqa: S603
                rows = [[float(x) for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
            except (OSError, subprocess.SubprocessError, ValueError):
                return
            if rows:
                self.samples.append((sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / 1024))

    def stats(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        util = [s[0] for s in self.samples]
        return {
            "gpu_util_mean": round(sum(util) / len(util), 1),
            "gpu_util_max": max(util),
            "gpu_memory_max_gb": round(max(s[1] for s in self.samples), 2),
        }
