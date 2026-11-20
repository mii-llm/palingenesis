"""An engine worker: one rollout engine in its own process, serving a trainer over TCP.

    python -m palingenesis.rl.pool.worker --model Qwen/Qwen3.5-0.8B --port 7000 [--backend vllm]
        [--gpu-memory-utilization 0.9] [--max-model-len 8192] [--lora-rank 16] [--host 0.0.0.0]

Generation. Every request joins the engine's own running batch (palingenesis.rl.generation) the moment it
arrives and is answered the step it finishes, tagged with the version of the weights that sampled it (and the
version its first token was sampled with, when weights changed while it decoded).

Weights. A push is received into a staging area in host memory while the engine keeps generating with the
weights it has: nothing of a push is visible to the engine until all of it has arrived. The complete push is
then applied while the engine does nothing else (between two steps of a streaming engine; between two generate
calls of a batch engine, under one lock with them), and from then on requests are tagged with the new version.
A push whose connection ends before its last byte is dropped whole; the engine keeps its version. Frames:

  full     every weight, by checkpoint name: applied as it is.
  base     every weight, held (not applied, not versioned) for the frame that follows it: the frozen base that a
           trainer with adapters sends once, when it connects.
  fold     LoRA factors per adapted weight; the worker computes base + scaling * B A (palingenesis.lora.fold_weight)
           and loads the base and the folds together. Megabytes on the wire, any engine, one policy per engine.
           The worker keeps the base in host memory for it: 2 bytes per parameter for a bf16 model.
  native   LoRA factors per named adapter, loaded as adapters of the engine (after the held base, if one came):
           requests name the adapter they want and requests of different adapters share the batch. With vLLM the
           adapters are written to `--adapter-dir` (a memory file system) in the PEFT layout and reloaded in
           place, and the versions before the last two are deleted; with the hf engine they are this library's
           own adapter layers. The trainer's default adapter is the adapter "policy".

Identity. A trainer starts with a `hello` that names its model, stop tokens, chat-template arguments, context
length and head precision; the worker answers with its own, and the trainer refuses to go on if they differ.

Custom engines (tests, other servers): `--engine module:factory`, a callable returning a RolloutEngine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch

from palingenesis.rl.generation import GenerationClient
from palingenesis.rl.pool.protocol import Staging, receive, send

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER = "policy"  # the name the trainer's default adapter has in an engine that serves named adapters


class HFWorkerEngine:
    """The hf engine of a worker: a model of its own that weight pushes are copied into."""

    def __init__(self, model, stop_ids, pad_id: int, micro_seqs: int = 64):
        from palingenesis.opd.rollout import HFRollout, checkpoint_named_parameters

        self.model = model
        self.inner = HFRollout(model, stop_ids, pad_id, micro_seqs)
        self.targets = dict(checkpoint_named_parameters(model))  # checkpoint name -> the parameter's own storage
        self.plugin = None  # palingenesis.lora.LoRA, once a native push described the adapters
        self.version = 0

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = ""):
        return self.inner.generate(prompts, max_new_tokens, temperature, images, adapter)

    @torch.no_grad()
    def update_weights(self, named_tensors, version: int) -> None:
        for name, tensor in named_tensors:
            target = self.targets[name]
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        self.version = version

    @torch.no_grad()
    def load_adapters(self, adapters: dict[str, dict[str, dict[str, torch.Tensor]]], version: int, lora: dict) -> None:
        """Named adapters as {adapter: {module: {"lora_A": ..., "lora_B": ...}}} ("": the default adapter)."""
        if self.plugin is None:
            from palingenesis.hooks import PluginContext
            from palingenesis.lora import LoRA

            self.plugin = LoRA(
                rank=lora["rank"],
                alpha=lora["alpha"],
                rslora=lora.get("rslora", False),
                targets=lora.get("targets") or None,
                exclude=lora.get("exclude") or [],
            )
            self.plugin.transform_model(self.model, PluginContext(config=None))
        for adapter, modules in adapters.items():
            if adapter and adapter not in getattr(self.plugin, "names", []):
                self.plugin.add_adapter(adapter)
            for module, parts in modules.items():
                a, b = self.plugin.modules[module].factors(adapter)
                a.copy_(parts["lora_A"].to(a.device, a.dtype))
                b.copy_(parts["lora_B"].to(b.device, b.dtype))
        self.version = version

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass


class LockedEngine:
    """A batch engine (one generate call at a time) behind the lock that weight updates take: a call runs on
    one set of weights from start to end, and its completions carry the version of that set."""

    def __init__(self, engine: Any, lock: threading.Lock):
        self.engine, self.lock = engine, lock

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def generate(self, prompts, max_new_tokens, temperature, **kwargs):
        with self.lock:
            version = self.engine.version
            rollouts = self.engine.generate(prompts, max_new_tokens, temperature, **kwargs)
            for rollout in rollouts:
                rollout.policy_version = version
            return rollouts


class EngineWorker:
    """Serves `engine` on a TCP port. `identity`: what a trainer's `hello` is checked against. `gate`: a path;
    while a file exists there, pushes wait where they are (tests hold a push in the middle)."""

    def __init__(self, engine: Any, identity: dict | None = None, adapter_dir: str = "", gate: str = ""):
        self.lock = threading.Lock()  # held by a weight update, and by every generate call of a batch engine
        self.engine = engine  # weights and adapters are loaded into it; requests reach it through `client`
        serving = engine if hasattr(engine, "stream_add") else LockedEngine(engine, self.lock)
        self.identity, self.gate_path = dict(identity or {}), gate
        self.client = GenerationClient(serving, merge=False)
        self.staging = Staging(pin=True)
        self.base: dict[str, torch.Tensor] = {}  # the frozen base (host memory): folds are computed against it
        self.held: dict[str, torch.Tensor] | None = None  # a base frame waiting for the frame that follows it
        self.adapter_dir = adapter_dir
        self.default_adapter = ""  # the engine's name for the trainer's default adapter, once it was loaded
        self.pushes = self.dropped = 0
        self.receiving = False
        self.server: asyncio.AbstractServer | None = None
        self.stopping = asyncio.Event()

    @property
    def version(self) -> int:
        return self.engine.version

    # ---------------------------------------------------------------- serving

    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self.server = await asyncio.start_server(self._connection, host, port)
        return self.server.sockets[0].getsockname()[1]

    async def run(self, host: str, port: int, announce=print) -> None:
        port = await self.serve(host, port)
        announce(f"engine worker listening on {host}:{port}", flush=True)
        await self.stopping.wait()
        self.server.close()
        self.client.close()

    async def _gate(self, header: dict) -> None:
        if header.get("op") == "weights":
            self.receiving = True
            while self.gate_path and os.path.exists(self.gate_path):
                await asyncio.sleep(0.01)

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lock = asyncio.Lock()  # replies of concurrent requests must not interleave
        seen: set[str] = set()  # the ops this connection carried

        async def reply(header: dict) -> None:
            async with lock:
                await send(writer, header)

        try:
            while True:
                try:
                    header, tensors = await receive(reader, self.staging, self._gate)
                except asyncio.IncompleteReadError as error:
                    frame = getattr(error, "frame", None)
                    pushing = frame is not None and (
                        frame.get("op") == "weights" or (not frame and "weights" in seen and "generate" not in seen)
                    )
                    if pushing or self.held is not None and "weights" in seen:
                        self.dropped += 1  # cut inside a push: nothing of it was applied
                        self.held = None
                        self.receiving = False
                    raise
                op = header.get("op")
                seen.add(op)
                if op == "generate":
                    asyncio.ensure_future(self._generate(header, reply))
                elif op == "weights":
                    try:
                        seconds = await asyncio.get_running_loop().run_in_executor(None, self._apply, header, tensors)
                        await reply({"op": "applied", "version": self.version, "seconds": seconds})
                    except Exception as error:  # noqa: BLE001 — the trainer must know its push was not applied
                        logger.exception("a weight update could not be applied")
                        await reply(
                            {"op": "applied", "version": self.version, "error": f"{type(error).__name__}: {error}"}
                        )
                    finally:
                        self.receiving = False
                elif op == "hello":
                    await reply({"op": "hello", "id": header.get("id"), "identity": self.identity, **self.state()})
                elif op == "state":
                    await reply({"op": "state", "id": header.get("id"), **self.state()})
                elif op == "shutdown":
                    self.stopping.set()
                    return
                else:
                    await reply({"op": "error", "id": header.get("id"), "error": f"unknown op {op!r}"})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # the peer left
        finally:
            writer.close()

    def state(self) -> dict:
        return {"version": self.version, "pushes": self.pushes, "dropped": self.dropped, "receiving": self.receiving}

    async def _generate(self, header: dict, reply) -> None:
        self.client.enter()
        try:
            temperature = header["temperature"]
            if isinstance(temperature, dict):
                from palingenesis.opd.rollout import Sampling

                temperature = Sampling(**temperature)
            adapter = header.get("adapter") or self.default_adapter
            extra = {"adapter": adapter} if adapter else {}
            out = await self.client.generate(header["prompt"], header["max_new_tokens"], temperature, **extra)
            message = {
                "op": "generated",
                "id": header["id"],
                "ids": list(out.ids),
                "logprobs": list(out.logprobs),
                "finish": out.finish,
                "version": out.version,
                "first_version": out.first_version,
            }
        except Exception as error:  # noqa: BLE001 — the requester sees the failure
            message = {"op": "generated", "id": header["id"], "error": f"{type(error).__name__}: {error}"}
        finally:
            self.client.exit()
        try:
            await reply(message)
        except ConnectionError:
            pass

    # ---------------------------------------------------------------- weights

    def _apply(self, header: dict, tensors: dict[str, torch.Tensor]) -> float:
        """Make a complete push the engine's weights while the engine does nothing else. Seconds it was held."""
        version, mode = int(header["version"]), header.get("mode", "full")
        if mode == "base":  # held: the frame that follows applies it together with its own content
            self.held = dict(tensors)
            self.staging.release(tensors)  # (its buffers are the held base now, not reused by the next frame)
            return 0.0
        held, self.held = self.held, None
        adapters: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
        named: Any = None
        if mode == "fold":
            from palingenesis.lora import fold_weight

            if held is not None:
                self.base = held
            if not self.base:
                raise RuntimeError("a fold push came before the base it folds into")
            folded = {}
            for name, scaling in header["scaling"].items():
                base = self.base[name]
                weight = fold_weight(
                    base, tensors[f"{name}|A"], tensors[f"{name}|B"], scaling, tensors.get(f"{name}|m")
                )
                folded[name] = weight.to(base.dtype)
            named = ((name, folded.get(name, tensor)) for name, tensor in self.base.items())
        elif mode == "native":
            adapters = {}
            for key, tensor in tensors.items():
                adapter, module, part = key.split("|")
                adapters.setdefault(adapter, {}).setdefault(module, {})[part] = tensor
            named = held.items() if held is not None else None  # the base once, then only adapters
        else:
            named = tensors.items()
        start = time.perf_counter()
        with self.lock, self.client.paused():  # no generate call, no engine step: one set of weights per sample
            if named is not None:
                self.engine.update_weights(named, version)
            if adapters is not None:
                self._load_adapters(adapters, version, header["lora"])
        self.pushes += 1
        return time.perf_counter() - start

    def _load_adapters(self, adapters: dict, version: int, lora: dict) -> None:
        if hasattr(self.engine, "load_adapters"):
            self.engine.load_adapters(adapters, version, lora)
            return
        # vLLM: each adapter as a PEFT directory, reloaded in place under its name (opd.rollout.VLLMColocateRollout)
        from safetensors.torch import save_file

        root = Path(self.adapter_dir or tempfile.gettempdir()) / f"pgs-adapters-{os.getpid()}"
        for adapter, modules in adapters.items():
            name = adapter or DEFAULT_ADAPTER
            path = root / name / f"v{version}"
            path.mkdir(parents=True, exist_ok=True)
            state = {
                f"base_model.model.{module}.{part}.weight": tensor.float().contiguous()
                for module, parts in modules.items()
                for part, tensor in parts.items()
            }
            save_file(state, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
            config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": lora["rank"],
                "lora_alpha": lora["alpha"],
                "use_rslora": lora.get("rslora", False),
                "use_dora": False,
                "target_modules": sorted(modules),
                "lora_dropout": 0.0,
                "bias": "none",
                "fan_in_fan_out": False,
            }
            (path / "adapter_config.json").write_text(json.dumps(config))
            self.engine.load_adapter(name, str(path), version)
            if not adapter:
                self.default_adapter = name  # requests that name no adapter sample the policy, not the bare base
            for old in (root / name).iterdir():  # keep this version and the one before (requests may still read it)
                if old.name.startswith("v") and old.name[1:].isdigit() and int(old.name[1:]) < version - 1:
                    shutil.rmtree(old, ignore_errors=True)
        self.engine.version = version


def build_engine(args: argparse.Namespace) -> tuple[Any, dict]:
    """(the engine, the identity a trainer is checked against)."""
    if args.engine:
        from palingenesis.rl.rewards import load_object

        return load_object(args.engine)(), {}
    from transformers import AutoConfig, AutoTokenizer

    from palingenesis.opd.trainer import student_stop_ids
    from palingenesis.rl.pool.protocol import model_signature

    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = json.loads(args.chat_template_kwargs)
    stop_ids = student_stop_ids(tok, args.model, json.loads(args.stop_tokens), kwargs)
    identity = {
        "model": model_signature(AutoConfig.from_pretrained(args.model)),
        "stop_ids": sorted(int(i) for i in stop_ids),
        "chat_template_kwargs": kwargs,
        "max_model_len": args.max_model_len,
        "fp32_head": bool(args.fp32_head),
    }
    if args.backend == "hf":
        from palingenesis.opd.teachers import load_causal_lm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_causal_lm(args.model, torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]
        del identity["fp32_head"], identity["max_model_len"]  # (vLLM's settings: the hf engine has neither)
        return HFWorkerEngine(model, stop_ids, pad_id), identity
    from palingenesis.opd.rollout import VLLMColocateRollout, precision_options

    extra = json.loads(args.vllm_args)
    extra.update(precision_options(args.fp32_head, args.ssm_state_dtype, extra.get("hf_overrides")))
    if args.lora_rank:  # native pushes: adapters served next to the base
        extra = {"enable_lora": True, "max_loras": args.max_loras, "max_lora_rank": args.lora_rank, **extra}
    engine = VLLMColocateRollout(
        args.model,
        stop_ids,
        args.gpu_memory_utilization,
        args.max_model_len,
        False,
        args.seed,
        sleep_mode=False,
        prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        engine_kwargs=extra,
    )
    return engine, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m palingenesis.rl.pool.worker", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--engine", default="", help="module:factory of a RolloutEngine, instead of --model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32-head", action="store_true", help="logits in fp32, as rollout.fp32_head")
    parser.add_argument("--ssm-state-dtype", default="float32", help="as rollout.ssm_state_dtype")
    parser.add_argument("--lora-rank", type=int, default=0, help="vllm: serve LoRA adapters up to this rank (native)")
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--adapter-dir", default="/dev/shm", help="vllm: where pushed adapters are written")
    parser.add_argument("--vllm-args", default="{}", help="JSON: extra vllm.LLM arguments")
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--stop-tokens", default="[]")
    parser.add_argument("--gate", default="", help="tests: pushes wait while this file exists")
    args = parser.parse_args(argv)
    if not args.model and not args.engine:
        parser.error("--model (or --engine) is required")
    logging.basicConfig(level=logging.INFO, format="[engine] %(asctime)s | %(levelname)s | %(message)s")
    engine, identity = build_engine(args)
    worker = EngineWorker(engine, identity, args.adapter_dir, args.gate)
    asyncio.run(worker.run(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
