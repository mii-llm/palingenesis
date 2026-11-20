"""An engine worker: one rollout engine in its own process, serving a trainer over TCP.

    python -m palingenesis.rl.pool.worker --model Qwen/Qwen3.5-0.8B --port 7000 [--backend vllm]
        [--gpu-memory-utilization 0.9] [--max-model-len 8192] [--lora-rank 16] [--host 0.0.0.0]

Generation. Every request joins the engine's own running batch (palingenesis.rl.generation) the moment it
arrives and is answered the step it finishes, tagged with the version of the weights that sampled it (and the
version its first token was sampled with, when weights changed while it decoded).

Weights. A push is received into a staging area in host memory while the engine keeps generating with the
weights it has: nothing of a push is visible to the engine until all of it has arrived. The complete push is
then applied between two engine steps, and from that step on requests are tagged with the new version. A push
whose connection ends before its last byte is dropped whole; the engine keeps its version. Four kinds of push:

  full     every weight, by checkpoint name.
  base     every weight, kept also as the frozen base that "fold" pushes are folded into (once, when a trainer
           with adapters connects).
  fold     LoRA factors per adapted weight; the worker computes base + scaling * B A (palingenesis.lora.fold_weight)
           and loads the result. Megabytes on the wire, any engine, one policy per engine.
  native   LoRA factors per named adapter, loaded as adapters of the engine: requests name the adapter they want
           and requests of different adapters share the batch. With vLLM the adapters are written to
           `--adapter-dir` (a memory file system) in the PEFT layout and reloaded in place; with the hf engine
           they are this library's own adapter layers.

Custom engines (tests, other servers): `--engine module:factory`, a callable returning a RolloutEngine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from palingenesis.rl.generation import GenerationClient
from palingenesis.rl.pool.protocol import Staging, receive, send

# (a "base" frame and the "fold" frame after it share one staging area: their tensor names differ, and the base is
# cloned when it is applied)

logger = logging.getLogger(__name__)


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
        rollouts = self.inner.generate(prompts, max_new_tokens, temperature, images, adapter)
        for rollout in rollouts:
            rollout.policy_version = self.version
        return rollouts

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


class EngineWorker:
    """Serves `engine` on a TCP port. `throttle`: bytes per second a push is received at (0: as fast as it comes;
    a slow link in tests)."""

    def __init__(self, engine: Any, throttle: float = 0.0, adapter_dir: str = ""):
        self.engine, self.throttle = engine, throttle
        self.client = GenerationClient(engine, merge=False)
        self.staging = Staging(pin=True)
        self.base: dict[str, torch.Tensor] = {}
        self.adapter_dir = adapter_dir
        self.pushes = self.dropped = 0
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

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lock = asyncio.Lock()  # replies of concurrent requests must not interleave
        try:
            while True:
                header, tensors = await self._frame(reader)
                op = header.get("op")
                if op == "generate":
                    asyncio.ensure_future(self._generate(header, writer, lock))
                elif op == "weights":
                    seconds = await asyncio.get_running_loop().run_in_executor(None, self._apply, header, tensors)
                    async with lock:
                        await send(writer, {"op": "applied", "version": self.version, "seconds": seconds})
                elif op == "state":
                    async with lock:
                        await send(writer, {"op": "state", **self.state()})
                elif op == "shutdown":
                    self.stopping.set()
                    return
                else:
                    async with lock:
                        await send(writer, {"op": "error", "error": f"unknown op {op!r}"})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # the peer left; a push it was sending is dropped whole (nothing of it was applied)
        finally:
            writer.close()

    async def _frame(self, reader: asyncio.StreamReader):
        """One frame; a weights frame goes into the staging area, and counts as dropped if it ends early."""
        try:
            return await receive(reader, self.staging, self.throttle)
        except asyncio.IncompleteReadError as error:
            self.dropped += getattr(error, "frame", {}).get("op") == "weights"
            raise

    def state(self) -> dict:
        return {"version": self.version, "pushes": self.pushes, "dropped": self.dropped}

    async def _generate(self, header: dict, writer: asyncio.StreamWriter, lock: asyncio.Lock) -> None:
        self.client.enter()
        try:
            extra = {"adapter": header["adapter"]} if header.get("adapter") else {}
            out = await self.client.generate(header["prompt"], header["max_new_tokens"], header["temperature"], **extra)
            reply = {
                "op": "generated",
                "id": header["id"],
                "ids": list(out.ids),
                "logprobs": list(out.logprobs),
                "finish": out.finish,
                "version": out.version,
                "first_version": out.first_version,
            }
        except Exception as error:  # noqa: BLE001 — the requester sees the failure
            reply = {"op": "generated", "id": header["id"], "error": f"{type(error).__name__}: {error}"}
        finally:
            self.client.exit()
        try:
            async with lock:
                await send(writer, reply)
        except ConnectionError:
            pass

    # ---------------------------------------------------------------- weights

    def _apply(self, header: dict, tensors: dict[str, torch.Tensor]) -> float:
        """Make a complete push the engine's weights, between two engine steps. Seconds the engine was held."""
        version, mode = int(header["version"]), header.get("mode", "full")
        if mode == "fold":
            from palingenesis.lora import fold_weight

            folded = {}
            for name, scaling in header["scaling"].items():
                base = self.base[name]
                weight = fold_weight(
                    base, tensors[f"{name}|A"], tensors[f"{name}|B"], scaling, tensors.get(f"{name}|m")
                )
                folded[name] = weight.to(base.dtype)
            named: Any = ((name, folded.get(name, tensor)) for name, tensor in self.base.items())
        elif mode == "native":
            adapters: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
            for key, tensor in tensors.items():
                adapter, module, part = key.split("|")
                adapters.setdefault(adapter, {}).setdefault(module, {})[part] = tensor
            named = None
        else:
            if mode == "base":  # kept: later pushes are factors to fold into it
                self.base = {name: tensor.clone() for name, tensor in tensors.items()}
            named = tensors.items()
        start = time.perf_counter()
        with self.client.paused():  # between two engine steps: requests in flight continue with the new weights
            if named is None:
                self._load_adapters(adapters, version, header["lora"])
            else:
                self.engine.update_weights(named, version)
        self.pushes += 1
        return time.perf_counter() - start

    def _load_adapters(self, adapters: dict, version: int, lora: dict) -> None:
        if hasattr(self.engine, "load_adapters"):
            self.engine.load_adapters(adapters, version, lora)
            return
        # vLLM: each adapter as a PEFT directory, reloaded in place under its name (opd.rollout.VLLMColocateRollout)
        from safetensors.torch import save_file

        root = Path(self.adapter_dir or tempfile.gettempdir()) / "pgs-adapters"
        for adapter, modules in adapters.items():
            path = root / f"{adapter or 'policy'}-v{version}"
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
                "target_modules": sorted(modules),
                "lora_dropout": 0.0,
                "bias": "none",
            }
            (path / "adapter_config.json").write_text(json.dumps(config))
            self.engine.load_adapter(adapter or "policy", str(path), version)
        self.engine.version = version


def build_engine(args: argparse.Namespace):
    if args.engine:
        from palingenesis.rl.rewards import load_object

        return load_object(args.engine)()
    from transformers import AutoTokenizer

    from palingenesis.opd.trainer import student_stop_ids

    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = json.loads(args.chat_template_kwargs)
    stop_ids = student_stop_ids(tok, args.model, json.loads(args.stop_tokens), kwargs)
    if args.backend == "hf":
        from palingenesis.opd.teachers import load_causal_lm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_causal_lm(args.model, torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]
        return HFWorkerEngine(model, stop_ids, pad_id)
    from palingenesis.opd.rollout import VLLMColocateRollout

    extra = json.loads(args.vllm_args)
    if args.lora_rank:  # native pushes: adapters served next to the base
        extra = {"enable_lora": True, "max_loras": args.max_loras, "max_lora_rank": args.lora_rank, **extra}
    return VLLMColocateRollout(
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
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="vllm: serve LoRA adapters up to this rank (native pushes)"
    )
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--adapter-dir", default="/dev/shm", help="vllm: where pushed adapters are written")
    parser.add_argument("--vllm-args", default="{}", help="JSON: extra vllm.LLM arguments")
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--stop-tokens", default="[]")
    parser.add_argument("--throttle", type=float, default=0.0, help="bytes per second a push is received at (tests)")
    args = parser.parse_args(argv)
    if not args.model and not args.engine:
        parser.error("--model (or --engine) is required")
    logging.basicConfig(level=logging.INFO, format="[engine] %(asctime)s | %(levelname)s | %(message)s")
    worker = EngineWorker(build_engine(args), args.throttle, args.adapter_dir)
    asyncio.run(worker.run(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
