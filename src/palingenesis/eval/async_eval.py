"""The trainer's side of an asynchronous evaluator (eval_tasks.placement: async).

    async_eval = AsyncEval(config.eval_tasks, info, main=rank == 0)   # every rank (gathers the ranks' GPUs)
    async_eval.start()                                    # spec.json; the evaluator process with eval_tasks.devices
    ...
    async_eval.saved(step, path)                          # transport checkpoint: after each save (lease)
    async_eval.push(step, named)                          # transport push: at due steps, every rank
    for step, metrics in async_eval.poll(): log(metrics, step)     # every step: results that arrived
    async_eval.finish(final_step, final_path)             # after the last save: the final result, then stop

Nothing here waits for the evaluator during training and nothing raises into the training loop: an evaluator that
is missing, slow, dead or out of memory costs at most the snapshot of a due step (push), and its absence is logged
(eval/async/* metrics). The only waits are in finish(), after training (eval_tasks.final_timeout_s).

Costs on the trainer, at due steps only. checkpoint: none beyond the saves train.save_steps already makes (a lease
file). push: the weights are copied into one pinned host staging area on rank 0 (2 bytes per parameter, allocated
at the first push, checked against the host's available memory) and sent by a background thread; under FSDP that
copy is preceded by one gather of the full weights, one parameter at a time, as a save does. That synchronous part
is logged as eval/async/snapshot_seconds. A due step whose previous snapshot is still being sent is skipped
without gathering or copying anything (the newest weights win).
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from palingenesis.eval import handoff

logger = logging.getLogger(__name__)

MAX_RESTARTS = 3  # an evaluator the trainer started is started again at most this many times
POLL_SECONDS = 10.0  # the result files are looked at no more often than this


@dataclass
class RunInfo:
    """What the evaluator needs from the trainer: written to spec.json."""

    trainer: str  # train | rl | distill
    model: str  # the checkpoint the run started from
    output_dir: str
    every: int  # the evaluation cadence (eval_tasks.every, else the trainer's)
    total_steps: int
    start_step: int = 0
    tokenizer: str = ""
    chat_template_kwargs: dict = field(default_factory=dict)
    stop_tokens: list = field(default_factory=list)
    seed: int = 0
    max_model_len: int = 8192
    weights_bytes: int = 0  # the push staging area: 2 bytes per parameter (bfloat16 on the wire)
    base_step: bool = False  # the run's start is evaluated (step 0 of pgs rl / distill, a fresh run)
    checkpoints: dict = field(default_factory=dict)  # {prefix: "step_" | "step-", complete: marker file}


def gather_training_gpus(local_rank: int | None = None) -> list[dict]:
    """[{host, uuid, label}] of every rank (a collective when torch.distributed is initialized)."""
    from palingenesis.eval.placement import rank_gpu

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    mine = rank_gpu(local_rank)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            out: list = [None] * dist.get_world_size()
            dist.all_gather_object(out, mine)
            return [dict(o, label=f"rank {i} ({o['label']} on {o['host']})") for i, o in enumerate(out)]
    except (ImportError, RuntimeError):
        pass
    return [dict(mine, label=f"rank 0 ({mine['label']} on {mine['host']})")]


class AsyncEval:
    """See the module docstring. `spec`: the eval_tasks section; `main`: rank 0 (files, the evaluator process,
    results); `training_gpus`: what gather_training_gpus returned (every rank must call it); `agree`: under FSDP, a
    callable that makes rank 0's bool every rank's (a push gathers on every rank)."""

    def __init__(self, spec, info: RunInfo, main: bool = True, training_gpus: list[dict] | None = None, agree=None):
        self.spec, self.info, self.main = spec, info, main
        self.output_dir = Path(info.output_dir)
        self.training_gpus = training_gpus or []
        self.agree = agree
        self.every = info.every
        self.process: subprocess.Popen | None = None
        self.restarts = 0
        self.logged: set[int] = set()
        self.counts = {"skipped": 0, "failed": 0, "done": 0, "push_skipped": 0}
        self.lag = 0
        self.snapshot_seconds = 0.0
        self.last_poll = 0.0
        self.last_step = 0
        self.ended = False
        self.client = None  # transport push: palingenesis.eval.push.PushClient

    # ------------------------------------------------------------ start

    @property
    def transport(self) -> str:
        return self.spec.transport

    def due(self, step: int, total: int = 0) -> bool:
        if total and step >= total:
            return self.spec.final
        return bool(self.every) and step % self.every == 0

    def spec_json(self) -> dict:
        from palingenesis.eval.service import spec_section

        i = self.info
        return {
            "version": handoff.SPEC_VERSION,
            "trainer": i.trainer,
            "model": i.model,
            "tokenizer": i.tokenizer or i.model,
            "chat_template_kwargs": i.chat_template_kwargs,
            "stop_tokens": list(i.stop_tokens),
            "seed": i.seed,
            "every": self.every,
            "total_steps": i.total_steps,
            "max_model_len": i.max_model_len,
            "base_step": 0 if i.base_step and i.start_step == 0 else None,
            "transport": self.transport,
            "checkpoints": i.checkpoints,
            "engine": dict(self.spec.engine),
            "eval_tasks": spec_section(self.spec),
            "training_gpus": self.training_gpus,
            "trainer_pid": os.getpid(),
            "started": time.time(),
        }

    def start(self) -> None:
        """Rank 0: write spec.json (a resumed run keeps its results), then start the evaluator when
        eval_tasks.devices says where."""
        if not self.main:
            return
        root = handoff.root(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for stale in (handoff.RUN_END, handoff.SERVICE):
            (root / stale).unlink(missing_ok=True)
        self.logged = {
            s
            for s in handoff.results(self.output_dir)
            if (handoff.step_dir(self.output_dir, s) / handoff.LOGGED).exists()
        }
        spec_json = self.spec_json()
        if self.transport == "push" and self.info.weights_bytes:
            from palingenesis.eval.config import EvalConfigError
            from palingenesis.eval.placement import host_memory_error

            error = host_memory_error(self.info.weights_bytes)
            if error:
                raise EvalConfigError(error)
        handoff.write_json(root / handoff.SPEC, spec_json)
        if self.spec.devices:
            self._launch()
        logger.info(
            "eval_tasks: asynchronous (%s), every %d steps; results in %s; evaluator %s",
            self.transport, self.every, root,
            f"on GPUs {self.spec.device_list()} (started here)" if self.spec.devices else f"service {self.spec.service}",
        )  # fmt: skip
        if self.transport == "push":
            from palingenesis.eval.push import PushClient

            self.client = PushClient(self.output_dir, self.spec, self.info, spec_json)

    def command(self) -> list[str]:
        mode = "watch" if self.transport == "checkpoint" else "serve"
        command = [sys.executable, "-m", "palingenesis.eval.service", mode]
        command += (
            [str(self.output_dir), "--poll", "10"]
            if mode == "watch"
            else ["--dir", str(self.output_dir), "--host", "127.0.0.1"]
        )
        command += ["--parent_pid", str(os.getpid())]
        factory = os.environ.get("PGS_EVAL_ENGINE_FACTORY")  # tests: a fake engine
        if factory:
            command += ["--engine", factory]
        return command

    def _launch(self) -> None:
        from palingenesis.eval.placement import child_environment

        environment = child_environment(self.spec.device_list())
        log = open(handoff.root(self.output_dir) / "service.log", "ab")  # noqa: SIM115 — the child's, kept open
        self.process = subprocess.Popen(  # noqa: S603
            self.command(), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        log.close()
        logger.info(
            "eval_tasks: evaluator started (pid %d, GPUs %s)", self.process.pid, environment["CUDA_VISIBLE_DEVICES"]
        )

    def _supervise(self) -> None:
        """An evaluator this trainer started that died is started again (at most MAX_RESTARTS times)."""
        if self.process is None or self.process.poll() is None or self.ended:
            return
        code = self.process.returncode
        service = handoff.read_json(handoff.root(self.output_dir) / handoff.SERVICE) or {}
        if service.get("state") in ("refused", "broken") or self.restarts >= MAX_RESTARTS:
            if self.process is not None:
                logger.error(
                    "eval_tasks: the evaluator exited (code %s, %s): no more asynchronous evaluation in this run; "
                    "training goes on (log: %s)",
                    code, service.get("error") or service.get("state", "no state"), handoff.root(self.output_dir) / "service.log",
                )  # fmt: skip
            self.process = None
            return
        self.restarts += 1
        logger.warning(
            "eval_tasks: the evaluator exited with code %s; starting it again (%d/%d)",
            code,
            self.restarts,
            MAX_RESTARTS,
        )
        self._launch()

    # ------------------------------------------------------------ checkpoint

    def saved(self, step: int, path: str | Path) -> None:
        """transport checkpoint: a checkpoint of `step` was saved at `path` (rank 0, after the save)."""
        if self.main and self.transport == "checkpoint" and self.due(step):
            handoff.write_lease(path, step)

    # ------------------------------------------------------------ push

    def wants_push(self, step: int, total: int = 0) -> bool:
        """transport push: whether this due step sends a snapshot (rank 0 decides; under FSDP every rank learns it,
        since the snapshot gathers). False when the previous snapshot is still on its way: nothing is gathered or
        copied then, and a later step brings newer weights."""
        if self.transport != "push" or not self.due(step, total):
            return False
        ready = bool(self.main and self.client is not None and self.client.ready())
        if self.main and not ready:
            self.counts["push_skipped"] += 1
        return bool(self.agree(ready)) if self.agree is not None else ready

    def push(self, step: int, named) -> float:
        """Stage `named` ((checkpoint name, tensor) pairs of the weights of `step`, a generator every rank must
        iterate under FSDP) and hand it to the sender; returns the seconds the trainer spent (the snapshot)."""
        started = time.perf_counter()
        if self.main and self.client is not None:
            self.client.submit(step, named)
        else:
            for _ in named:  # the gather's collectives
                pass
        self.snapshot_seconds = time.perf_counter() - started
        return self.snapshot_seconds

    # ------------------------------------------------------------ results

    def poll(self, step: int = 0, force: bool = False) -> list[tuple[int, dict]]:
        """(step, metrics) of evaluations finished since the last call (rank 0; at most every POLL_SECONDS)."""
        if not self.main:
            return []
        self.last_step = max(self.last_step, step)
        if not force and time.monotonic() - self.last_poll < POLL_SECONDS:
            return []
        self.last_poll = time.monotonic()
        self._supervise()
        arrived = handoff.results(self.output_dir)
        if self.client is not None:
            self.client.poll()
            for result in self.client.take_results():  # an evaluator without the trainer's file system
                arrived.setdefault(int(result["step"]), result)
        out = []
        for result_step, result in sorted(arrived.items()):
            if result_step in self.logged:
                continue
            self.logged.add(result_step)
            status = result.get("status")
            self.counts[status] = self.counts.get(status, 0) + 1
            if status == "done":
                out.append((result_step, dict(result.get("metrics") or {})))
                self.lag = max(0, self.last_step - result_step)
            elif status == "failed":
                logger.warning("eval_tasks step %d failed in the evaluator: %s", result_step, result.get("error"))
            try:
                (handoff.step_dir(self.output_dir, result_step) / handoff.LOGGED).touch()
            except OSError:
                pass
        return out

    def metrics(self) -> dict[str, float]:
        """eval/async/* for the trainer's step metrics (rank 0)."""
        if not self.main:
            return {}
        out = {f"eval/async/{k}": float(v) for k, v in self.counts.items()}
        out["eval/async/lag_steps"] = float(self.lag)
        out["eval/async/alive"] = float(self.alive())
        if self.transport == "push":
            out["eval/async/snapshot_seconds"] = self.snapshot_seconds
            if self.client is not None:
                out.update({f"eval/async/{k}": v for k, v in self.client.metrics().items()})
        return out

    def alive(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        service = handoff.read_json(handoff.root(self.output_dir) / handoff.SERVICE)
        return isinstance(service, dict) and service.get("state") not in ("refused", "broken", "ended")

    # ------------------------------------------------------------ end

    def finish(self, final_step: int, final_path: str | Path | None = None, named=None) -> list[tuple[int, dict]]:
        """After the last step: tell the evaluator the run ended (transport checkpoint: with the final checkpoint;
        push: `named`, the final weights, sent once whatever is in flight), wait up to eval_tasks.final_timeout_s
        for the final step's result, and return what arrived meanwhile. Every rank calls it (a push gathers)."""
        if self.ended:
            return []
        self.ended = True
        if self.transport == "push" and named is not None and self.due(final_step, final_step):
            if self.main and self.client is not None:
                self.client.submit(final_step, named, wait=True)
            else:
                for _ in named:
                    pass
        if not self.main:
            return []
        handoff.write_json(
            handoff.root(self.output_dir) / handoff.RUN_END,
            {"final_step": final_step, "final_path": str(final_path) if final_path else "", "ended": time.time()},
        )
        if self.client is not None:
            self.client.end(final_step)
        wanted = self.spec.final and (self.transport == "push" or bool(final_path))
        if self.spec.final and not wanted:
            logger.warning(
                "eval_tasks: no final checkpoint (train.save_final off): step %d is not evaluated", final_step
            )
        timeout = float(self.spec.final_timeout_s)
        deadline = time.monotonic() + timeout
        out = []
        while timeout > 0:
            out += self.poll(final_step, force=True)
            finished = not wanted or final_step in self.logged
            launched = self.process is not None and self.process.poll() is None
            if finished and not launched:
                break  # (an evaluator this trainer started ends by itself once nothing is left)
            if not finished and not launched and self.process is None and not self.alive():
                logger.warning("eval_tasks: the evaluator is not running: step %d is not evaluated", final_step)
                break
            if time.monotonic() > deadline:
                logger.warning(
                    "eval_tasks: no result for step %d within final_timeout_s %.0f s; the evaluator goes on writing "
                    "its files in %s", final_step, timeout, handoff.root(self.output_dir),
                )  # fmt: skip
                break
            time.sleep(1.0)
        self.close(wait=timeout > 0 and time.monotonic() <= deadline)
        return out

    def close(self, wait: bool = True) -> None:
        """Stop the push client; an evaluator this trainer started ends by itself after run_end.json (it is only
        stopped here if it is still running once the final result is in or the timeout passed)."""
        if self.client is not None:
            self.client.close()
            self.client = None
        process, self.process = self.process, None
        if process is None or process.poll() is not None or not wait:
            return
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            import signal

            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)


def broadcast_flag(flag: bool, device) -> bool:
    """Rank 0's flag on every rank (a one-element broadcast)."""
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([1 if flag else 0], device=device)
    dist.broadcast(tensor, 0)
    return bool(tensor.item())


def make_async_eval(
    spec, info: RunInfo, *, main: bool, world: int = 1, device="cpu", extra_gpus=()
) -> AsyncEval | None:
    """The AsyncEval of a trainer, started (every rank calls it: the ranks' GPUs are gathered); None when
    eval_tasks is not asynchronous. extra_gpus: [{uuid, label, what}] of rollout engines the evaluator must avoid."""
    if not spec.is_async:
        return None
    gpus = gather_training_gpus() + list(extra_gpus)
    agree = (lambda flag: broadcast_flag(flag, device)) if world > 1 else None
    evaluator = AsyncEval(spec, info, main=main, training_gpus=gpus, agree=agree)
    evaluator.start()
    return evaluator


def engine_gpus(engine) -> list[dict]:
    """The GPUs of remote rollout engines (palingenesis.rl.pool workers report theirs), to keep the evaluator off."""
    state = getattr(engine, "state", None)
    if state is None or not hasattr(engine, "make_client"):
        return []
    try:
        states = state()
    except Exception:  # noqa: BLE001 — an engine that does not answer is checked by nobody, as before
        return []
    return [
        {"uuid": u, "label": f"engine {s.get('address', '')}", "what": "holds a rollout engine"}
        for s in states or []
        if isinstance(s, dict)
        for u in s.get("gpus") or []
    ]
