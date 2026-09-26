"""Rollout engines: the student's sampler, with the rollout policy's own log-probs.

  HFRollout            the trainer's model.generate. Always available, no extra
                       dependency, slow (one sequence batch at a time, padded).
  VLLMColocateRollout  an in-process vLLM engine on the trainer's GPU. It sleeps
                       while the trainer trains (weights and KV cache released) and
                       receives the new weights in place after every step.
  VLLMServerRollout    a separate `vllm serve` process fed weights through vLLM's
                       native weight-transfer API over CUDA IPC (same GPU). Runs
                       concurrently with training (rollout.max_staleness > 0).

Every engine returns, per completion, the log-probability of each sampled token
under the policy that sampled it (after temperature: vLLM's
"processed_logprobs"), which the policy-gradient losses use as the behaviour
policy mu of their importance ratios.

vLLM is imported lazily: without it installed, the hf backend works as before.
"""

import atexit
import base64
import json
import logging
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    completion_ids: list[int]
    logprobs: list[float]  # behaviour log-prob of each sampled token (empty for greedy)
    finish_reason: str  # "stop" (a stop token ended it) or "length"
    policy_version: int


class RolloutEngine(Protocol):
    version: int  # optimizer steps reflected in the engine's weights (-1: none loaded)

    def generate(self, prompts: list[list[int]], max_new_tokens: list[int], temperature: float) -> list[Rollout]:
        """One completion per prompt; temperature 0 = greedy."""

    def update_weights(self, named_tensors: Iterable[tuple[str, Tensor]], version: int) -> None:
        """Load the student's current weights (checkpoint-format names)."""

    def sleep(self) -> None:
        """Release GPU memory until the next update_weights/wake."""

    def wake(self) -> None:
        """Reacquire what sleep released, before generate."""


def checkpoint_named_parameters(
    model: nn.Module, named: Iterable[tuple[str, Tensor]] | None = None
) -> Iterator[tuple[str, Tensor]]:
    """The model's parameters under their checkpoint names, as inference engines load them.

    transformers may load a checkpoint under other names (Qwen3.5's
    ``model.language_model.*`` becomes ``model.*`` in the causal-LM class);
    revert_weight_conversion maps them back, as save_pretrained does. `named` replaces
    model.named_parameters() (e.g. full tensors gathered from FSDP shards).
    """
    from transformers import PreTrainedModel
    from transformers.core_model_loading import revert_weight_conversion

    from palingenesis.checkpoint import hf_state_dict

    named = model.named_parameters() if named is None else named
    params = hf_state_dict({name: p.detach() for name, p in named})
    if isinstance(model, PreTrainedModel):
        params = revert_weight_conversion(model, params)
    yield from params.items()


# ------------------------------------------------------------------------- hf


class _RecordLogprobs:
    """Logits processor: temperature-scales the scores and records each sampled
    token's log-probability. The token sampled at step t is visible at step t+1
    as input_ids[:, -1], so each call records the previous step's choice."""

    def __init__(self, temperature: float):
        self.temperature = temperature
        self.previous: Tensor | None = None
        self.logprobs: list[Tensor] = []

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if self.previous is not None:
            self.logprobs.append(self.previous.gather(1, input_ids[:, -1:]).squeeze(1))
        scores = scores / self.temperature
        self.previous = torch.log_softmax(scores.float(), -1)
        return scores

    def finish(self, sequences: Tensor) -> Tensor:
        """[B, steps] log-probs of the generated tokens."""
        if self.previous is not None:
            self.logprobs.append(self.previous.gather(1, sequences[:, -1:]).squeeze(1))
        return torch.stack(self.logprobs, 1) if self.logprobs else sequences.new_zeros((sequences.shape[0], 0))


class HFRollout:
    """Batched model.generate on the trainer's model (bf16 autocast), left-padded."""

    def __init__(self, model: nn.Module, stop_ids: tuple[int, ...], pad_id: int, micro_seqs: int):
        self.model = model
        self.stop_ids = stop_ids
        self.pad_id = pad_id
        self.micro_seqs = micro_seqs
        self.version = 0

    def update_weights(self, named_tensors, version: int) -> None:
        self.version = version  # it generates with the trained model itself

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, temperature):
        from transformers import GenerationConfig, LogitsProcessorList

        device = next(self.model.parameters()).device
        out: list[Rollout | None] = [None] * len(prompts)
        was_training = self.model.training
        self.model.eval()
        try:
            for budget in sorted(set(max_new_tokens)):
                idx = [i for i, m in enumerate(max_new_tokens) if m == budget]
                for start in range(0, len(idx), self.micro_seqs):
                    chunk = idx[start : start + self.micro_seqs]
                    width = max(len(prompts[i]) for i in chunk)
                    ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for row, i in enumerate(chunk):
                        ids[row, width - len(prompts[i]) :] = torch.tensor(prompts[i])
                        mask[row, width - len(prompts[i]) :] = 1
                    # Temperature is applied by the recorder, so the recorded log-probs are the sampled ones.
                    sampling = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if temperature > 0 else {}
                    config = GenerationConfig(
                        max_new_tokens=budget,
                        eos_token_id=list(self.stop_ids),
                        pad_token_id=self.pad_id,
                        use_cache=True,  # the model config may disable it for training
                        **sampling,
                    )
                    recorder = _RecordLogprobs(temperature) if temperature > 0 else None
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        seqs = self.model.generate(
                            ids.to(device),
                            attention_mask=mask.to(device),
                            generation_config=config,
                            logits_processor=LogitsProcessorList([recorder] if recorder else []),
                        )
                    generated = seqs[:, width:]
                    logprobs = recorder.finish(seqs).tolist() if recorder else None
                    for row, i in enumerate(chunk):
                        tokens = generated[row].tolist()
                        cut = next((k + 1 for k, t in enumerate(tokens) if t in self.stop_ids), None)
                        tokens = tokens[:cut] if cut else tokens
                        out[i] = Rollout(
                            tokens,
                            logprobs[row][: len(tokens)] if logprobs else [],
                            "stop" if cut else "length",
                            self.version,
                        )
        finally:
            self.model.train(was_training)
        return out


# ----------------------------------------------------------------------- vLLM


def _import_vllm():
    try:
        import vllm
    except ImportError as e:
        raise ImportError(
            "rollout.backend / teacher backend 'vllm' needs vLLM: install palingenesis[vllm] "
            "(Linux, NVIDIA driver >= 575)."
        ) from e
    return vllm


class VLLMColocateRollout:
    """An in-process vLLM engine sharing the trainer's GPU.

    Single-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the engine's model
    is reachable from here: new weights are loaded straight into it with the
    model's own load_weights, no copy through disk or another process. With
    sleep mode the engine releases its weights and KV cache while the trainer
    trains (level 2: the weights are dropped, since every wake is preceded by an
    update with newer ones).
    """

    def __init__(
        self,
        model: str,
        stop_ids: tuple[int, ...],
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        seed: int,
        sleep_mode: bool,
        prefix_caching: bool = False,
        max_num_seqs: int | None = None,
        external_launcher: bool = False,
        engine_kwargs: dict | None = None,
    ):
        """`external_launcher`: one engine per torchrun rank (colocated data parallelism; vLLM
        joins the launcher's process group). `engine_kwargs`: extra vllm.LLM arguments, e.g.
        {"kv_cache_dtype": "fp8"} or {"quantization": "fp8"}."""
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        vllm = _import_vllm()
        self.stop_ids = stop_ids
        self.sleep_mode = sleep_mode

        def engine(max_num_seqs):
            return vllm.LLM(
                model=model,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                seed=seed,
                enable_sleep_mode=sleep_mode,
                logprobs_mode="processed_logprobs",
                enable_prefix_caching=prefix_caching,
                max_num_seqs=max_num_seqs,
                **({"distributed_executor_backend": "external_launcher"} if external_launcher else {}),
                **(engine_kwargs or {}),
            )

        try:
            self.llm = engine(max_num_seqs)
        except ValueError as e:
            # Hybrid models (Qwen3.5, ...) hold one Mamba state per decoding sequence: vLLM
            # refuses more sequences than its cache has states, and names the limit.
            limit = re.search(r"exceeds available Mamba cache blocks \((\d+)\)", str(e))
            if limit is None:
                raise
            logger.warning(
                "vLLM: %s concurrent sequences do not fit the Mamba cache of gpu_memory_utilization "
                "%s; using %s (raise gpu_memory_utilization for more)",
                max_num_seqs,
                gpu_memory_utilization,
                limit.group(1),
            )
            self.llm = engine(int(limit.group(1)))
        self.SamplingParams = vllm.SamplingParams
        self.version = 0  # -1 while asleep: the weights are gone until the next update
        self.asleep = False  # KV cache released
        self.prompt_tokens = self.cached_tokens = 0  # streaming: prefill accounting

    def update_weights(self, named_tensors, version: int) -> None:
        if self.version < 0:
            self._wake_up("weights")
        self.llm.apply_model(lambda model: model.load_weights(named_tensors))
        self.llm.reset_prefix_cache()
        self.version = version

    def sleep(self) -> None:
        if self.sleep_mode and not self.asleep:
            self.llm.sleep(level=2)
            self.asleep, self.version = True, -1

    def wake(self) -> None:
        if self.asleep:
            if self.version < 0:
                raise RuntimeError("the vLLM engine dropped its weights while asleep: update_weights before wake")
            self._wake_up("kv_cache")
            self.asleep = False

    def _wake_up(self, tag: str) -> None:
        # The trainer's caching allocator keeps the memory it freed (activations,
        # logits slices) reserved; vLLM maps its own memory, so release it first.
        torch.cuda.empty_cache()
        self.llm.wake_up(tags=[tag])

    def generate(self, prompts, max_new_tokens, temperature):
        """One completion per prompt. Identical requests (a GRPO group's rollouts) become one
        request with n samples: the prompt is prefilled once and its KV cache forked, instead
        of relying on the prefix cache to find it n times."""
        unique: dict[tuple, list[int]] = {}
        for i, (prompt, budget) in enumerate(zip(prompts, max_new_tokens)):
            unique.setdefault((tuple(prompt), budget), []).append(i)
        requests = list(unique.items())
        params = [self._params(len(indices), budget, temperature) for (_, budget), indices in requests]
        outputs = self.llm.generate(
            [{"prompt_token_ids": list(prompt)} for (prompt, _), _ in requests], params, use_tqdm=False
        )
        rollouts: list[Rollout | None] = [None] * len(prompts)
        for (_, indices), out in zip(requests, outputs):
            for i, rollout in zip(indices, self._rollouts(out)):
                rollouts[i] = rollout
        return rollouts

    def _params(self, n: int, budget: int, temperature: float, final_only: bool = False):
        kwargs = {}
        if final_only:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY
        return self.SamplingParams(
            n=n,
            max_tokens=budget,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            logprobs=0 if temperature > 0 else None,
            stop_token_ids=list(self.stop_ids),
            detokenize=False,
            **kwargs,
        )

    def _rollouts(self, out) -> list[Rollout]:
        rollouts = []
        for o in out.outputs:
            tokens = list(o.token_ids)
            logprobs = [step[t].logprob for step, t in zip(o.logprobs, tokens)] if o.logprobs else []
            finish = "stop" if o.finish_reason == "stop" else "length"
            rollouts.append(Rollout(tokens, logprobs, finish, self.version))
        return rollouts

    # Streaming (continuous batching across calls): requests join the running batch the moment
    # they are added and come back the step they finish. One thread must own these calls.

    def stream_add(self, request_id: str, prompt: list[int], max_new_tokens: int, temperature: float, n: int) -> None:
        params = self._params(n, max_new_tokens, temperature, final_only=True)
        self.llm.llm_engine.add_request(request_id, {"prompt_token_ids": list(prompt)}, params)

    def stream_pending(self) -> bool:
        return self.llm.llm_engine.has_unfinished_requests()

    def stream_step(self) -> list[tuple[str, list[Rollout]]]:
        """One engine step: (request id, its n rollouts) for every request that finished.
        Counts the finished requests' prompt tokens and those served by the prefix cache."""
        finished = [out for out in self.llm.llm_engine.step() if out.finished]
        for out in finished:
            self.prompt_tokens += len(out.prompt_token_ids or ())
            self.cached_tokens += out.num_cached_tokens or 0
        return [(out.request_id, self._rollouts(out)) for out in finished]


def _die_with_parent() -> None:
    """In a launched server: receive SIGTERM when the trainer dies, however it dies
    (atexit handlers do not run on a signal, and the server holds GPU memory)."""
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMServer:
    """A `vllm serve` process launched by the trainer, or an existing one at `url`.

    Launched servers get the dev endpoints (pause/resume, weight transfer) and are
    terminated when the trainer exits; their log goes to `log_path`.
    """

    def __init__(
        self, model: str, url: str = "", args: tuple[str, ...] = (), log_path: str = "", startup_timeout: float = 900.0
    ):
        self.model = model
        self.process = None
        if url:
            self.url = url.rstrip("/")
        else:
            port = _free_port()
            self.url = f"http://127.0.0.1:{port}"
            env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model,
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--served-model-name",
                model,
                "--dtype",
                "bfloat16",
                *args,
            ]
            logger.info("Launching %s (log: %s)", " ".join(cmd), log_path or "inherited")
            self._log = open(log_path, "w") if log_path else None
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                preexec_fn=_die_with_parent if sys.platform == "linux" else None,
            )
            atexit.register(self.close)
        self._wait_ready(startup_timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server for {self.model} exited with code {self.process.returncode}; see its log."
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM server for {self.model} not ready after {timeout:.0f}s at {self.url}")

    def post(self, path: str, payload: dict | None = None, timeout: float = 3600.0) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"vLLM server {path}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"
            ) from None
        return json.loads(body) if body else {}

    def complete(self, prompts: list[list[int]], parallel: int = 4, **params) -> list[dict]:
        """POST /v1/completions for token-id prompts, split over `parallel` requests; choices in order."""
        size = max(1, -(-len(prompts) // parallel))
        parts = [prompts[i : i + size] for i in range(0, len(prompts), size)]

        def send(part):
            choices = self.post("/v1/completions", {"model": self.model, "prompt": part, **params})["choices"]
            return sorted(choices, key=lambda c: c["index"])

        with ThreadPoolExecutor(len(parts)) as pool:
            return [c for part in pool.map(send, parts) for c in part]

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


class VLLMServerRollout:
    """Rollouts from a vLLM server; weights pushed over CUDA IPC (the server must be
    on the trainer's GPU and started with --weight-transfer-config '{"backend": "ipc"}').

    Each update pauses the server between rollout batches, sends CUDA IPC handles
    of bf16 copies of the weights (vLLM's native start/update/finish weight
    update endpoints; the server loads them with the model's load_weights), and
    resumes. The copies live until the server has loaded them.
    """

    def __init__(self, server: VLLMServer, stop_ids: tuple[int, ...]):
        self.server = server
        self.stop_ids = stop_ids
        self.version = 0
        self.server.post("/init_weight_transfer_engine", {"init_info": {}})

    def update_weights(self, named_tensors, version: int) -> None:
        from torch.multiprocessing.reductions import reduce_tensor

        names, dtypes, shapes, handles, copies = [], [], [], [], []
        gpu = None
        for name, tensor in named_tensors:
            copy = tensor.detach().to(torch.bfloat16).contiguous()
            gpu = gpu or str(torch.cuda.get_device_properties(copy.device).uuid)
            copies.append(copy)
            names.append(name)
            dtypes.append("bfloat16")
            shapes.append(list(copy.shape))
            handles.append({gpu: reduce_tensor(copy)[1]})
        torch.cuda.synchronize()
        update = {
            "names": names,
            "dtype_names": dtypes,
            "shapes": shapes,
            "packed": False,
            "ipc_handles_pickled": base64.b64encode(pickle.dumps(handles)).decode(),
        }
        self.server.post("/pause?mode=wait")
        try:
            self.server.post("/start_weight_update")
            self.server.post("/update_weights", {"update_info": update})
            self.server.post("/finish_weight_update")
        finally:
            self.server.post("/resume")
        del copies
        torch.cuda.ipc_collect()
        self.version = version

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def generate(self, prompts, max_new_tokens, temperature):
        out: list[Rollout | None] = [None] * len(prompts)
        for budget in sorted(set(max_new_tokens)):
            idx = [i for i, m in enumerate(max_new_tokens) if m == budget]
            choices = self.server.complete(
                [prompts[i] for i in idx],
                max_tokens=budget,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                logprobs=0 if temperature > 0 else None,
                stop_token_ids=list(self.stop_ids),
                return_token_ids=True,
                skip_special_tokens=False,
            )
            for i, choice in zip(idx, choices):
                logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
                out[i] = Rollout(
                    choice["token_ids"],
                    logprobs,
                    "stop" if choice["finish_reason"] == "stop" else "length",
                    self.version,
                )
        return out
