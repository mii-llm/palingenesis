"""The asynchronous evaluator of eval_tasks (placement: async): a process of its own on GPUs that do not train.

    pgs eval watch runs/sft            # transport checkpoint: evaluates the run's checkpoints as they are saved
    pgs eval watch runs/sft --once     # after training: what is there, then exit
    pgs eval serve --dir runs/rl       # transport push: receives the trainer's weights over TCP

Both read <output_dir>/eval_tasks/spec.json, which the trainer writes at start (tasks, model, tokenizer, chat
template arguments, stop tokens, engine, cadence, the GPUs the trainer uses), so neither takes a training config.
The trainer starts one itself with eval_tasks.devices; with eval_tasks.service: auto the user, Slurm or another
node does. Results go to <output_dir>/eval_tasks/step_<n>/ in the same files as an inline evaluation, then
step_<n>/result.json; the trainer logs them at step n (palingenesis.eval.handoff).

The evaluator refuses to start on a GPU the trainer uses (palingenesis.eval.placement), before it loads anything.
Its engine runs in worker processes (palingenesis.rl.pool.worker; the hf engine in this process), so an engine
that runs out of memory takes down a worker, not this process: the engine is started again with half the
max_num_seqs, then with 5 points less of gpu_memory_utilization, and the step is evaluated again. A step that
still fails is recorded as failed and the next one is tried; after three failed steps in a row the evaluator stops
(service.json: broken). None of this reaches the trainer, which only reads the files.

Checkpoints (`watch`): the step_* / step-* directories of the run whose step is due (every `every` steps, the
last step when `final`), complete (trainer_state.pt / training_meta.json written), in Hugging Face format or as a
sharded DCP checkpoint of pgs train under FSDP, whose model weights are consolidated here (host memory: the full
model in the precision it was saved in). backlog latest evaluates the newest checkpoint not evaluated yet and
records the older ones as skipped; all evaluates every one, oldest first. A checkpoint is leased (.eval_pending)
until its weights are loaded, so the trainer's purge keeps it. eval_tasks.select names a metric; best_tasks.json
records its best step and the purge keeps that checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import logging
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from palingenesis.eval import handoff

logger = logging.getLogger(__name__)

MAX_FAILURES = 3  # steps in a row that failed: the evaluator stops
OOM_RETRIES = 2  # engine restarts with smaller settings per step


class EvaluatorRefused(RuntimeError):
    """The evaluator may not run where it was started (a GPU that trains)."""


# ----------------------------------------------------------------- weights


def _model_dir(path: Path) -> Path:
    if (path / "model" / "config.json").exists():
        return path / "model"  # pgs train, one process: <step-N>/model
    return path


def checkpoint_format(path: str | Path) -> str:
    """hf | dcp | missing, for a checkpoint directory."""
    path = Path(path)
    model = _model_dir(path)
    if any(model.glob("*.safetensors")) or any(model.glob("*.bin")):
        return "hf"
    if (path / "dcp").is_dir():
        return "dcp"
    return "missing"


def hf_weights(path: str | Path) -> Iterable[tuple[str, Any]]:
    """(name, tensor) of a Hugging Face checkpoint, one safetensors file at a time."""
    from safetensors import safe_open

    files = sorted(_model_dir(Path(path)).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{path}: no .safetensors files")
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def consolidate_dcp(dcp_dir: str | Path, base_model: str = "") -> dict[str, Any]:
    """The model weights of a sharded DCP checkpoint (pgs train under FSDP: {"model": ..., "optimizer": ...}) as
    full tensors under their checkpoint names, without the optimizer state and without a process group.
    base_model: the run's model, for the names transformers maps on load (revert_weight_conversion)."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from palingenesis.checkpoint import hf_state_dict

    reader = FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata()
    state = {
        key: torch.empty(tuple(meta.size), dtype=meta.properties.dtype)
        for key, meta in metadata.state_dict_metadata.items()
        if key.startswith("model.") and isinstance(meta, TensorStorageMetadata)
    }
    if not state:
        raise ValueError(f"{dcp_dir}: no model.* tensors in this checkpoint")
    dcp_load(state, storage_reader=reader, no_dist=True)
    named = hf_state_dict({key[len("model.") :]: tensor for key, tensor in state.items()})
    return _checkpoint_names(named, base_model)


def _checkpoint_names(named: dict, base_model: str) -> dict:
    """Names as a checkpoint has them: transformers loads some under other names (Qwen3.5's language model)."""
    if not base_model:
        return named
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.core_model_loading import revert_weight_conversion

        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base_model))
        return revert_weight_conversion(skeleton, named)
    except Exception:  # noqa: BLE001 — the names as saved are the usual case
        logger.debug("checkpoint names: kept as saved", exc_info=True)
        return named


def checkpoint_weights(path: str | Path, base_model: str = "") -> Callable[[], Iterable]:
    """A callable giving the (name, tensor) pairs of a checkpoint (called again for a retry)."""
    kind = checkpoint_format(path)
    if kind == "hf":
        return lambda: hf_weights(path)
    if kind == "dcp":
        return lambda: consolidate_dcp(Path(path) / "dcp", base_model).items()
    raise FileNotFoundError(f"{path}: neither Hugging Face weights nor a DCP checkpoint")


# ------------------------------------------------------------------ engine


def _is_oom(error: BaseException) -> bool:
    import torch

    if isinstance(error, torch.OutOfMemoryError):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return "out of memory" in text or "engine worker exited" in text or "enginelost" in text


def default_engine(spec: dict, options: dict, tokenizer, stop_ids: tuple):
    """The evaluator's engine: vLLM / SGLang in worker processes (an EnginePool, one engine per GPU and replica,
    taking weight updates), or the hf engine in this process."""
    from palingenesis.eval.config import SamplingConfig, TaskConfig
    from palingenesis.eval.engines import engine_layout, start_engines
    from palingenesis.eval.runner import rl_config

    engine_options, layout = engine_layout(options)
    model = spec["model"]
    if engine_options.get("backend") == "hf":
        import torch

        from palingenesis.opd.teachers import load_causal_lm
        from palingenesis.rl.pool.worker import HFWorkerEngine

        class HFEvalEngine(HFWorkerEngine):
            """Checkpoint files may hold tensors the model has no parameter for (a tied output head, buffers)."""

            @torch.no_grad()
            def update_weights(self, named_tensors, version: int) -> None:
                loaded = 0
                for name, tensor in named_tensors:
                    target = self.targets.get(name)
                    if target is not None:
                        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                        loaded += 1
                if not loaded:
                    raise ValueError("none of the weights matched the evaluator's model")
                self.version = version

        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = load_causal_lm(model, torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
        return HFEvalEngine(policy, stop_ids, pad, int(options.get("micro_seqs", 64)))
    kwargs = dict(spec.get("chat_template_kwargs") or {})
    config = rl_config(
        model,
        TaskConfig(name="engine"),
        kwargs,
        SamplingConfig(temperature=0.0, max_tokens=1),
        engine_options,
        spec.get("stop_tokens") or [],
        int(spec.get("seed", 0)),
    )
    return start_engines(
        model,
        config,
        tokenizer,
        stop_ids,
        layout,
        kwargs,
        spec.get("stop_tokens") or [],
        pooled=True,
        updatable=True,
    )


def load_factory(ref: str) -> Callable:
    """module:function -> the function (custom engines: tests, other servers)."""
    module, _, name = ref.partition(":")
    return getattr(importlib.import_module(module), name)


# ----------------------------------------------------------------- service


def own_gpus() -> list[dict]:
    """The GPUs this process sees: [{uuid, label}]."""
    from palingenesis.eval import placement

    devices = placement.nvidia_devices()
    entries = placement.visible_entries()
    count = len(devices) if entries is None else len(entries)
    return [
        {"uuid": u, "label": f"GPU {placement.physical_entry(i)} on {socket.gethostname()}"}
        for i, u in placement.uuids(list(range(count)), devices=devices).items()
    ]


def check_placement(spec: dict, gpus: list[dict]) -> None:
    """Raise EvaluatorRefused when one of `gpus` is one the trainer uses."""
    from palingenesis.eval.placement import overlap_error

    training = spec.get("training_gpus") or []
    error = overlap_error(training, gpus, single_gpu=len({t.get("uuid") for t in training if t.get("uuid")}) <= 1)
    if error:
        raise EvaluatorRefused(error)


def wait_for_spec(output_dir: str | Path, timeout: float = 0.0, poll: float = 5.0) -> dict:
    """spec.json of the run (written by the trainer at start); waits for it up to `timeout` seconds (0: forever)."""
    path = handoff.root(output_dir) / handoff.SPEC
    start = time.monotonic()
    announced = False
    while True:
        spec = handoff.read_json(path)
        if isinstance(spec, dict):
            if spec.get("version") != handoff.SPEC_VERSION:
                raise ValueError(
                    f"{path}: spec version {spec.get('version')}, this evaluator reads {handoff.SPEC_VERSION}"
                )
            return spec
        if timeout and time.monotonic() - start > timeout:
            raise TimeoutError(f"{path} did not appear within {timeout:.0f} s: is the trainer running?")
        if not announced:
            logger.info("waiting for %s (written by the trainer when it starts)", path)
            announced = True
        time.sleep(poll)


class EvalService:
    """One engine, the run's tasks, and the steps it evaluates. engine_factory(spec, options, tokenizer, stop_ids)
    builds the engine (default_engine); tokenizer: given instead of loaded (tests)."""

    def __init__(self, output_dir: str | Path, spec: dict, engine_factory: Callable | None = None, tokenizer=None):
        from palingenesis.eval.config import EvalTasksConfig
        from palingenesis.eval.training import TrainingEval
        from palingenesis.opd.trainer import student_stop_ids

        self.output_dir = Path(output_dir)
        self.spec = spec
        self.options = dict(spec.get("engine") or {})
        self.engine_factory = engine_factory or default_engine
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer") or spec["model"])
        self.tokenizer = tokenizer
        section = EvalTasksConfig(**spec["eval_tasks"])
        self.hook = TrainingEval(
            section,
            spec["model"],
            tokenizer,
            self.output_dir,
            every=int(spec.get("every", 0)),
            engine_options={
                "max_model_len": int(self.options.get("max_model_len") or spec.get("max_model_len") or 8192)
            },
            chat_template_kwargs=spec.get("chat_template_kwargs") or {},
            stop_tokens=spec.get("stop_tokens") or [],
            seed=int(spec.get("seed", 0)),
        )
        self.stop_ids = student_stop_ids(
            tokenizer, spec["model"], spec.get("stop_tokens") or [], spec.get("chat_template_kwargs") or {}
        )
        self.select = section.select
        self.engine = None
        self.failures = 0
        self.broken = False
        self.attempts = 0  # fault injection (PGS_TEST_EVAL_OOM)

    # ----------------------------------------------------------- engine

    def start_engine(self) -> None:
        if self.engine is None:
            started = time.perf_counter()
            self.engine = self.engine_factory(self.spec, self.options, self.tokenizer, self.stop_ids)
            logger.info(
                "evaluator engine ready in %.0f s (%s)", time.perf_counter() - started, self.options or "defaults"
            )

    def close_engine(self) -> None:
        engine, self.engine = self.engine, None
        close = getattr(engine, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _shrink(self, attempt: int) -> None:
        """Smaller engine settings after an out-of-memory error."""
        from palingenesis.eval.engines import EVAL_MAX_NUM_SEQS

        if attempt == 1:
            self.options["max_num_seqs"] = max(1, int(self.options.get("max_num_seqs") or EVAL_MAX_NUM_SEQS) // 2)
        else:
            utilisation = float(self.options.get("gpu_memory_utilization") or 0.9)
            self.options["gpu_memory_utilization"] = round(max(0.1, utilisation - 0.05), 3)
        logger.warning("evaluator: out of memory; starting the engine again with %s", self.options)

    def _inject(self) -> None:
        """PGS_TEST_EVAL_OOM=once | always: an out-of-memory error at the start of an evaluation (tests, GPU check)."""
        mode = os.environ.get("PGS_TEST_EVAL_OOM", "")
        self.attempts += 1
        if mode == "always" or (mode == "once" and self.attempts == 1):
            import torch

            raise torch.OutOfMemoryError("PGS_TEST_EVAL_OOM: injected out-of-memory error")

    # --------------------------------------------------------- evaluate

    def evaluate(self, step: int, weights: Callable[[], Iterable] | None, source: str = "", loaded=None) -> dict:
        """Load `weights()` (None: the engine's own model, the run's start) and evaluate step `step`; returns its
        result (also written to step_<n>/result.json). loaded(): called once the weights are in the engine."""
        retries = 0
        while True:
            try:
                started = time.perf_counter()
                self._inject()
                if weights is None and self.engine is not None and getattr(self.engine, "version", 0) not in (0, -1):
                    self.close_engine()  # it holds a later step's weights: the start's come with a new engine
                self.start_engine()
                if weights is not None:
                    self.engine.update_weights(weights(), step)
                load_seconds = time.perf_counter() - started
                if loaded is not None:
                    loaded()
                    loaded = None
                metrics = self.hook.evaluate(self.engine, step)
                seconds = time.perf_counter() - started
                result = handoff.write_result(
                    self.output_dir, step, "done", metrics=metrics, seconds=seconds, load_seconds=load_seconds,
                    source=source,
                )  # fmt: skip
                logger.info("eval_tasks step %d done in %.0f s (%s)", step, seconds, source or "pushed weights")
                self.failures = 0
                self._select(step, metrics, source)
                return result
            except Exception as error:  # noqa: BLE001 — recorded, the next step is tried
                if _is_oom(error) and retries < OOM_RETRIES:
                    retries += 1
                    self.close_engine()
                    self._shrink(retries)
                    continue
                logger.exception("eval_tasks step %d failed", step)
                self.failures += 1
                self.broken = self.failures >= MAX_FAILURES
                if _is_oom(error):
                    self.close_engine()
                if loaded is not None:
                    loaded()
                return handoff.write_result(
                    self.output_dir, step, "failed", error=f"{type(error).__name__}: {error}", source=source
                )

    def _select(self, step: int, metrics: dict, source: str) -> None:
        """best_tasks.json: the best step by eval_tasks.select (higher is better)."""
        if not self.select:
            return
        key = f"tasks/{self.select}"
        if key not in metrics:
            logger.warning(
                "eval_tasks.select %s: no such metric at step %d (have %s)", self.select, step, sorted(metrics)
            )
            return
        path = handoff.root(self.output_dir) / handoff.BEST
        best = handoff.read_json(path)
        if isinstance(best, dict) and best.get("value") is not None and float(best["value"]) >= metrics[key]:
            return
        handoff.write_json(path, {"metric": self.select, "step": step, "value": metrics[key], "path": source})

    def close(self) -> None:
        self.close_engine()


# ------------------------------------------------------------------ watch


def due(step: int, spec: dict) -> bool:
    every = int(spec.get("every") or 0)
    return bool(every) and step % every == 0


def checkpoints(output_dir: str | Path, spec: dict) -> dict[int, str]:
    """{step: path} of the run's complete checkpoints at due steps, the run's start (step 0) when it is evaluated,
    and the final model once run_end.json names it."""
    output_dir = Path(output_dir)
    layout = spec.get("checkpoints") or {}
    prefix, marker = layout.get("prefix", "step_"), layout.get("complete", "")
    found: dict[int, str] = {}
    if spec.get("base_step") is not None and due(int(spec["base_step"]), spec):
        found[int(spec["base_step"])] = ""  # the engine's own model
    if output_dir.is_dir():
        for entry in output_dir.iterdir():
            name = entry.name
            if not (entry.is_dir() and name.startswith(prefix) and name[len(prefix) :].isdigit()):
                continue
            step = int(name[len(prefix) :])
            if due(step, spec) and (not marker or (entry / marker).exists()):
                found[step] = str(entry)
    end = handoff.read_json(handoff.root(output_dir) / handoff.RUN_END)
    if isinstance(end, dict) and end.get("final_step") is not None and spec["eval_tasks"].get("final", True):
        final = int(end["final_step"])
        if final not in found and end.get("final_path"):
            found[final] = str(end["final_path"])
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def write_service(output_dir: str | Path, mode: str, state: str, **fields) -> None:
    handoff.write_json(
        handoff.root(output_dir) / handoff.SERVICE,
        {
            "mode": mode,
            "state": state,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated": time.time(),
            **fields,
        },
    )


class Watcher:
    """transport checkpoint: evaluate the run's checkpoints as they appear (see the module docstring)."""

    def __init__(
        self,
        output_dir: str | Path,
        backlog: str = "",
        poll: float = 30.0,
        once: bool = False,
        engine_factory: Callable | None = None,
        parent_pid: int = 0,
        tokenizer=None,
        spec_timeout: float = 0.0,
    ):
        self.output_dir = Path(output_dir)
        self.backlog, self.poll, self.once = backlog, poll, once
        self.engine_factory, self.parent_pid, self.tokenizer = engine_factory, parent_pid, tokenizer
        self.spec_timeout = spec_timeout
        self.id = uuid.uuid4().hex[:8]

    def pending(self, spec: dict) -> list[tuple[int, str]]:
        done = handoff.results(self.output_dir)
        return sorted((step, path) for step, path in checkpoints(self.output_dir, spec).items() if step not in done)

    def run(self) -> int:
        spec = wait_for_spec(self.output_dir, self.spec_timeout, poll=min(self.poll, 5.0))
        gpus = own_gpus()
        try:
            check_placement(spec, gpus)
        except EvaluatorRefused as error:
            write_service(self.output_dir, "watch", "refused", error=str(error), gpus=[g["uuid"] for g in gpus])
            logger.error("%s", error)
            return 1
        backlog = self.backlog or spec["eval_tasks"].get("backlog") or "latest"
        write_service(self.output_dir, "watch", "starting", id=self.id, gpus=[g["uuid"] for g in gpus])
        service = EvalService(self.output_dir, spec, self.engine_factory, self.tokenizer)
        try:
            while True:
                pending = self.pending(spec)
                if pending:
                    if backlog == "latest":
                        for step, path in pending[:-1]:
                            handoff.write_result(self.output_dir, step, "skipped", reason="a newer checkpoint came")
                            if path:
                                handoff.release(path)
                        step, path = pending[-1]
                    else:
                        step, path = pending[0]
                    write_service(self.output_dir, "watch", "evaluating", id=self.id, step=step)
                    try:
                        weights = checkpoint_weights(path, spec["model"]) if path else None
                    except FileNotFoundError as error:
                        handoff.write_result(self.output_dir, step, "failed", error=str(error), source=path)
                        continue
                    service.evaluate(step, weights, path, loaded=(lambda p=path: handoff.release(p)) if path else None)
                    if service.broken:
                        write_service(self.output_dir, "watch", "broken", id=self.id)
                        logger.error("eval_tasks: %d steps in a row failed; the evaluator stops", MAX_FAILURES)
                        return 1
                    continue
                write_service(self.output_dir, "watch", "ready", id=self.id, gpus=[g["uuid"] for g in gpus])
                if self.once or (handoff.root(self.output_dir) / handoff.RUN_END).exists():
                    write_service(self.output_dir, "watch", "ended", id=self.id)
                    return 0
                if self.parent_pid and not _alive(self.parent_pid):
                    logger.warning(
                        "the trainer (pid %d) is gone without ending its run: the evaluator stops", self.parent_pid
                    )
                    write_service(self.output_dir, "watch", "ended", id=self.id)
                    return 0
                time.sleep(self.poll)
        finally:
            service.close()


# -------------------------------------------------------------------- cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pgs eval watch | serve", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watch", help="evaluate a run's checkpoints as they are saved (transport checkpoint)")
    watch.add_argument("output_dir", help="the run's train.output_dir")
    watch.add_argument("--backlog", choices=("latest", "all"), help="default: the run's eval_tasks.backlog")
    watch.add_argument("--poll", type=float, default=30.0, help="seconds between looks for new checkpoints")
    watch.add_argument("--once", action="store_true", help="evaluate what is there, then exit")
    watch.add_argument("--spec_timeout", type=float, default=0.0, help="seconds to wait for spec.json (0: forever)")
    serve = sub.add_parser("serve", help="receive the trainer's weights over TCP (transport push)")
    serve.add_argument("--dir", required=True, help="the run's train.output_dir (rendezvous and results)")
    serve.add_argument("--host", default="0.0.0.0", help="the interface to listen on")
    serve.add_argument("--port", type=int, default=0, help="0: any free port (written to service.json)")
    serve.add_argument("--idle_timeout", type=float, default=300.0, help="seconds without a trainer before exiting")
    serve.add_argument("--spec_timeout", type=float, default=0.0, help="seconds to wait for spec.json (0: forever)")
    for p in (watch, serve):
        p.add_argument("--engine", default="", help="module:factory building the engine (tests, custom servers)")
        p.add_argument("--parent_pid", type=int, default=0, help="exit when this process is gone (set by the trainer)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from palingenesis.logging import setup_logging

    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(rank=0)
    with contextlib.suppress(OSError):
        os.nice(10)  # its CPU work yields to a trainer's data loaders on the same node
    factory = load_factory(args.engine) if args.engine else None
    if args.command == "watch":
        watcher = Watcher(args.output_dir, args.backlog or "", args.poll, args.once, factory, args.parent_pid,
                          spec_timeout=args.spec_timeout)  # fmt: skip
        return watcher.run()
    from palingenesis.eval.push import serve

    return serve(args.dir, args.host, args.port, factory, args.parent_pid, args.idle_timeout, args.spec_timeout)


def spec_section(spec) -> dict:
    """The eval_tasks section as spec.json carries it: task files and local datasets as absolute paths, so an
    evaluator started elsewhere (another directory, another node of the shared file system) finds them."""
    section = dataclasses.asdict(spec)
    section["tasks"] = [_absolute_task(ref) for ref in spec.tasks]
    return section


def _absolute_task(ref):
    if isinstance(ref, str):
        return str(Path(ref).resolve()) if Path(ref).exists() else ref
    if isinstance(ref, dict):
        out = dict(ref)
        for key in ("dataset", "data_files"):
            value = out.get(key)
            if isinstance(value, str) and value and Path(value).exists():
                out[key] = str(Path(value).resolve())
        if (
            isinstance(out.get("judge_validation"), dict)
            and Path(str(out["judge_validation"].get("file", ""))).exists()
        ):
            out["judge_validation"] = {
                **out["judge_validation"],
                "file": str(Path(out["judge_validation"]["file"]).resolve()),
            }
        return out
    return ref


if __name__ == "__main__":
    sys.exit(main())
