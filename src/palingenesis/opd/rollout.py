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
under the policy that sampled it (after temperature and top-k / top-p: vLLM's
"processed_logprobs"), which the policy-gradient losses use as the behaviour
policy mu of their importance ratios.

`generate(prompts, max_new_tokens, temperature)` takes the sampling as a float
(temperature, no truncation; 0 = greedy), a Sampling, or one Sampling per prompt.

Candidates. A Sampling with `candidates` k > 0 also returns, per sampled token, the sampler's k most likely
tokens with their processed log-probs (after temperature and top-k / top-p), the finite ones only, most likely
first (Rollout.candidates). With top-k / top-p the tokens of finite log-prob are the support the token was drawn
from: RL renormalises the trained policy over it (palingenesis.rl.losses.sampled_logprobs).

vLLM is imported lazily: without it installed, the hf backend works as before.
"""

import atexit
import base64
import contextlib
import json
import logging
import math
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
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn

from palingenesis.accelerator import is_rocm

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sampling:
    """How a rollout is sampled: temperature (0 = greedy), nucleus top_p (1 = off), top_k (0 = off), and
    presence_penalty (0 = off), subtracted from the logits of every token already generated in this completion (as in
    vLLM and the OpenAI API: prompt tokens are not penalised). Qwen3.5 recommends 1.5-2.0 against repetition loops."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    # Per sampled token, also return the sampler's `candidates` most likely tokens and their processed log-probs
    # (Rollout.candidates); 0 = none. vLLM bounds it with its max_logprobs.
    candidates: int = 0

    @property
    def truncated(self) -> bool:
        return self.top_p < 1.0 or self.top_k > 0


# vLLM's OpenAI server writes a log-prob of -inf (a token outside the sampler's support) as -9999.0
VLLM_LOGPROB_FLOOR = -9999.0


@dataclass
class Candidates:
    """The sampler's most likely tokens at each sampled position of a completion, ragged and flat: position t has
    the `counts[t]` tokens ids[o_t : o_t + counts[t]] with processed log-probs logprobs[o_t : o_t + counts[t]]
    (o_t = counts[:t].sum()), most likely first; tokens of log-prob -inf (outside the sampler's support) are left
    out."""

    counts: "np.ndarray"  # int32 [T]
    ids: "np.ndarray"  # int32 [counts.sum()]
    logprobs: "np.ndarray"  # float32 [counts.sum()]

    @classmethod
    def from_rows(cls, rows: list[tuple[list[int], list[float]]]) -> "Candidates":
        """From per-position (ids, log-probs) lists, in any order; non-finite log-probs are dropped."""
        import numpy as np

        counts, ids, logprobs = [], [], []
        for row_ids, row_logprobs in rows:
            kept = sorted(
                ((float(lp), int(i)) for i, lp in zip(row_ids, row_logprobs) if math.isfinite(lp)),
                key=lambda x: -x[0],
            )
            kept = [(lp, i) for lp, i in kept if lp > VLLM_LOGPROB_FLOOR]
            counts.append(len(kept))
            ids += [i for _, i in kept]
            logprobs += [lp for lp, _ in kept]
        return cls(
            np.asarray(counts, dtype=np.int32), np.asarray(ids, dtype=np.int32), np.asarray(logprobs, dtype=np.float32)
        )

    @classmethod
    def empty(cls, positions: int = 0) -> "Candidates":
        """`positions` positions without candidates (context, a forced prefix)."""
        import numpy as np

        return cls(np.zeros(positions, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))

    def __len__(self) -> int:
        return len(self.counts)

    def head(self, positions: int) -> "Candidates":
        """The first `positions` positions."""
        end = int(self.counts[:positions].sum())
        return Candidates(self.counts[:positions], self.ids[:end], self.logprobs[:end])

    @classmethod
    def join(cls, parts: list["Candidates"]) -> "Candidates":
        """The positions of `parts`, one after the other."""
        import numpy as np

        return cls(
            np.concatenate([p.counts for p in parts]).astype(np.int32, copy=False),
            np.concatenate([p.ids for p in parts]).astype(np.int32, copy=False),
            np.concatenate([p.logprobs for p in parts]).astype(np.float32, copy=False),
        )

    def select(self, positions: list[int]) -> "Candidates":
        """The candidates of `positions` (indices into this completion), in that order."""
        import numpy as np

        offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=offsets[1:])
        index = np.asarray(positions, dtype=np.int64)
        counts = self.counts[index]
        starts = np.repeat(offsets[index], counts)
        within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        flat = starts + within
        return Candidates(counts.astype(np.int32), self.ids[flat], self.logprobs[flat])

    def tensors(self, prefix: str = "candidates") -> dict[str, Tensor]:
        """As host tensors (the engine pool's wire format, palingenesis.rl.pool)."""
        return {
            f"{prefix}.counts": torch.from_numpy(self.counts),
            f"{prefix}.ids": torch.from_numpy(self.ids),
            f"{prefix}.logprobs": torch.from_numpy(self.logprobs),
        }

    @classmethod
    def from_tensors(cls, tensors: dict[str, Tensor], prefix: str = "candidates") -> "Candidates | None":
        if f"{prefix}.counts" not in tensors:
            return None
        return cls(*(tensors[f"{prefix}.{key}"].numpy() for key in ("counts", "ids", "logprobs")))


def candidates_from_steps(steps, k: int) -> Candidates:
    """Candidates from vLLM's per-token log-probs ({token id: Logprob} per position; the sampled token is in it even
    outside the top k): the k most likely finite entries of each position."""
    rows = []
    for step in steps:
        ranked = sorted(((v.logprob, int(i)) for i, v in (step or {}).items()), key=lambda x: -x[0])[:k]
        rows.append(([i for _, i in ranked], [lp for lp, _ in ranked]))
    return Candidates.from_rows(rows)


def candidates_from_openai(top_logprobs, k: int) -> Candidates:
    """Candidates from an OpenAI completions `top_logprobs` list ({"token_id:<id>": log-prob} per position, as vLLM
    writes it with return_tokens_as_token_ids; the sampled token may be in it beyond the top k)."""
    rows = []
    for step in top_logprobs or []:
        ranked = sorted(
            ((float(lp), int(str(token).removeprefix("token_id:"))) for token, lp in (step or {}).items()),
            key=lambda x: -x[0],
        )[:k]
        rows.append(([i for _, i in ranked], [lp for lp, _ in ranked]))
    return Candidates.from_rows(rows)


def samplings(spec, n: int) -> list[Sampling]:
    """One Sampling per prompt from a float temperature, a Sampling, or a list of them."""
    if isinstance(spec, (int, float)):
        return [Sampling(float(spec))] * n
    if isinstance(spec, Sampling):
        return [spec] * n
    spec = list(spec)
    if len(spec) != n:
        raise ValueError(f"{len(spec)} samplings for {n} prompts")
    return [x if isinstance(x, Sampling) else Sampling(float(x)) for x in spec]


def truncate_scores(scores: Tensor, top_k: int, top_p: float) -> Tensor:
    """Scores outside the top_k tokens or the top_p nucleus set to -inf (the token that crosses top_p stays)."""
    if top_k > 0 and top_k < scores.shape[-1]:
        kth = torch.topk(scores, top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        sorted_scores, order = torch.sort(scores, dim=-1, descending=True)
        probs = torch.softmax(sorted_scores.float(), -1)
        drop = probs.cumsum(-1) - probs > top_p  # mass before this token already exceeds top_p
        scores = scores.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return scores


@dataclass
class Rollout:
    completion_ids: list[int]
    logprobs: list[float]  # behaviour log-prob of each sampled token (empty for greedy)
    finish_reason: str  # "stop" (a stop token ended it) or "length"
    policy_version: int
    # The version that sampled the first token (None: policy_version). Older when new weights were loaded while the
    # request was decoding (partial rollouts): its tokens then come from more than one policy version.
    first_version: int | None = None
    # MoE policies, with routing replay: the experts the engine routed each token it processed to,
    # [prompt + completion - 1, MoE layers, top_k] (the last sampled token was never forwarded); None otherwise.
    routed_experts: object | None = None
    # Sampling.candidates > 0: the sampler's most likely tokens at each sampled position; None otherwise.
    candidates: Candidates | None = None


class RolloutEngine(Protocol):
    version: int  # optimizer steps reflected in the engine's weights (-1: none loaded)

    def generate(self, prompts: list[list[int]], max_new_tokens: list[int], temperature) -> list[Rollout]:
        """One completion per prompt; `temperature`: a float (0 = greedy), a Sampling or one per prompt.
        Multimodal engines also take images=[per prompt: its images or None]: one image per
        placeholder token of the prompt, in order; the engine expands the placeholders."""

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


def decode_routed_experts(encoded: str | None):
    """A vLLM server's `routed_experts` field (base64 of a .npy file, with --enable-return-routed-experts):
    the array [tokens - 1, MoE layers, top_k], or None when the server did not send one."""
    if not encoded:
        return None
    import io

    import numpy as np

    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


class _RecordLogprobs:
    """Logits processor: temperature-scales (and top-k / top-p truncates) the scores
    and records each sampled token's log-probability. The token sampled at step t is
    visible at step t+1 as input_ids[:, -1], so each call records the previous step's choice."""

    def __init__(self, sampling: Sampling, prompt_width: int = 0):
        self.sampling = sampling
        self.prompt_width = prompt_width  # generated tokens start here; only they get the presence penalty
        self.previous: Tensor | None = None
        self.logprobs: list[Tensor] = []
        self.top: list[tuple[Tensor, Tensor]] = []  # Sampling.candidates: per step ([B, k] log-probs, [B, k] ids)

    def _record(self, sampled: Tensor) -> None:
        self.logprobs.append(self.previous.gather(1, sampled).squeeze(1))
        if self.sampling.candidates:
            values, ids = self.previous.topk(min(self.sampling.candidates, self.previous.shape[-1]), dim=-1)
            self.top.append((values.cpu(), ids.cpu()))

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if self.previous is not None:
            self._record(input_ids[:, -1:])
        if self.sampling.presence_penalty and input_ids.shape[1] > self.prompt_width:
            seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, input_ids[:, self.prompt_width :], True)
            scores = scores - self.sampling.presence_penalty * seen
        scores = truncate_scores(scores / self.sampling.temperature, self.sampling.top_k, self.sampling.top_p)
        self.previous = torch.log_softmax(scores.float(), -1)
        return scores

    def finish(self, sequences: Tensor) -> Tensor:
        """[B, steps] log-probs of the generated tokens."""
        if self.previous is not None:
            self._record(sequences[:, -1:])
            self.previous = None
        return torch.stack(self.logprobs, 1) if self.logprobs else sequences.new_zeros((sequences.shape[0], 0))

    def candidates(self, row: int, steps: int) -> Candidates:
        """Row `row`'s candidates over its first `steps` sampled tokens (after finish)."""
        return Candidates.from_rows([(ids[row].tolist(), values[row].tolist()) for values, ids in self.top[:steps]])


class HFRollout:
    """Batched model.generate on the trainer's model (bf16 autocast), left-padded."""

    def __init__(self, model: nn.Module, stop_ids: tuple[int, ...], pad_id: int, micro_seqs: int, vision=None):
        self.model = model
        self.stop_ids = stop_ids
        self.pad_id = pad_id
        self.micro_seqs = micro_seqs
        self.version = 0
        self.record_routing = False  # MoE routing replay: return the experts every token was routed to
        self.vision = vision  # palingenesis.multimodal.VisionProcessor of a multimodal policy

    def update_weights(self, named_tensors, version: int) -> None:
        self.version = version  # it generates with the trained model itself

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def _expand(self, prompts, images):
        """Prompts with their image placeholders expanded, and per prompt its pixel inputs."""
        if not images or not any(images):
            return prompts, None
        if self.vision is None:
            raise ValueError("images were passed to a text-only rollout engine (model.multimodal is off)")
        sequences = [self.vision.sequence(p, im or []) for p, im in zip(prompts, images)]
        return [s.input_ids.tolist() for s in sequences], sequences

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = ""):
        """`adapter`: the named LoRA adapter to generate with (palingenesis.lora.use_adapter; "base": none)."""
        if not adapter:
            return self._generate(prompts, max_new_tokens, temperature, images)
        from palingenesis.lora import use_adapter

        with use_adapter(adapter):
            return self._generate(prompts, max_new_tokens, temperature, images)

    def _generate(self, prompts, max_new_tokens, temperature, images=None):
        from transformers import GenerationConfig, LogitsProcessorList

        prompts, sequences = self._expand(prompts, images)
        device = next(self.model.parameters()).device
        out: list[Rollout | None] = [None] * len(prompts)
        per_prompt = samplings(temperature, len(prompts))
        was_training = self.model.training
        self.model.eval()
        try:
            for budget, sampling in sorted(set(zip(max_new_tokens, per_prompt)), key=lambda x: (x[0], repr(x[1]))):
                idx = [i for i, (m, sp) in enumerate(zip(max_new_tokens, per_prompt)) if m == budget and sp == sampling]
                temperature = sampling.temperature
                if temperature == 0 and sampling.presence_penalty:
                    raise ValueError("presence_penalty with greedy decoding: use temperature > 0 or the vLLM engine")
                for start in range(0, len(idx), self.micro_seqs):
                    chunk = idx[start : start + self.micro_seqs]
                    width = max(len(prompts[i]) for i in chunk)
                    ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for row, i in enumerate(chunk):
                        ids[row, width - len(prompts[i]) :] = torch.tensor(prompts[i])
                        mask[row, width - len(prompts[i]) :] = 1
                    # Temperature and truncation are applied by the recorder, so the recorded log-probs are the sampled ones.
                    sample_kwargs = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if temperature > 0 else {}
                    config = GenerationConfig(
                        max_new_tokens=budget,
                        eos_token_id=list(self.stop_ids),
                        pad_token_id=self.pad_id,
                        use_cache=True,  # the model config may disable it for training
                        **sample_kwargs,
                    )
                    recorder = _RecordLogprobs(sampling, width) if temperature > 0 else None
                    pixels = {}
                    shown = [im for i in chunk for im in sequences[i].images] if sequences else []
                    if shown:  # in row order, the order the model fills the placeholders in
                        pixels = {
                            "pixel_values": torch.cat([im.pixel_values for im in shown]).to(device),
                            "image_grid_thw": torch.tensor([im.grid for im in shown], device=device),
                            "mm_token_type_ids": (ids == self.vision.image_token_id).long().to(device),
                        }
                    routing = contextlib.ExitStack()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"), routing:
                        if self.record_routing:
                            from palingenesis.moe import RoutingRecorder

                            routed = routing.enter_context(RoutingRecorder(self.model))
                        seqs = self.model.generate(
                            ids.to(device),
                            attention_mask=mask.to(device),
                            generation_config=config,
                            logits_processor=LogitsProcessorList([recorder] if recorder else []),
                            **pixels,
                        )
                    # [rows, width + decoding steps, layers, top_k]: the left-padded prompts, then one token per step
                    experts = routed.indices(rows=len(chunk)) if self.record_routing else None
                    generated = seqs[:, width:]
                    logprobs = recorder.finish(seqs).tolist() if recorder else None
                    for row, i in enumerate(chunk):
                        tokens = generated[row].tolist()
                        cut = next((k + 1 for k, t in enumerate(tokens) if t in self.stop_ids), None)
                        tokens = tokens[:cut] if cut else tokens
                        top = recorder.candidates(row, len(tokens)) if recorder and sampling.candidates else None
                        out[i] = Rollout(
                            tokens,
                            logprobs[row][: len(tokens)] if logprobs else [],
                            "stop" if cut else "length",
                            self.version,
                            routed_experts=(
                                experts[row, width - len(prompts[i]) : width + len(tokens) - 1].numpy()
                                if experts is not None
                                else None
                            ),
                            candidates=top,
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
            "(Linux, NVIDIA driver >= 575; on AMD GPUs vLLM's ROCm build or container, see the AMD GPUs guide)."
        ) from e
    return vllm


# vLLM computes a generation model's logits in its head_dtype (hf_overrides) from 0.26.0 on; before, always in the
# model dtype.
FP32_HEAD_MIN_VLLM = (0, 26, 0)


def installed_vllm_version() -> tuple[int, ...] | None:
    """The installed vLLM's release numbers, e.g. (0, 26, 0); None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        text = version("vllm")
    except PackageNotFoundError:
        return None
    from packaging.version import Version

    return Version(text).release


def precision_options(fp32_head: bool, ssm_state_dtype: str, hf_overrides: dict | None = None) -> dict:
    """vllm.LLM arguments (engine-argument names) for the rollout's numerics:
      fp32_head        logits in fp32 (hf_overrides head_dtype: a bf16 GEMM with fp32 output, as the trainer's)
      ssm_state_dtype  the recurrent state of hybrid models between decoding steps (mamba_ssm_cache_dtype; "auto"
                       leaves vLLM's choice). Ignored by models without recurrent layers.
    `hf_overrides`: other overrides to keep. Raises ValueError when the installed vLLM cannot run an fp32 head."""
    options: dict = {}
    if fp32_head:
        installed = installed_vllm_version()
        if installed is not None and installed < FP32_HEAD_MIN_VLLM:
            minimum = ".".join(map(str, FP32_HEAD_MIN_VLLM))
            raise ValueError(
                f"rollout.fp32_head needs vLLM >= {minimum} (the head_dtype of generation models); vLLM "
                f"{'.'.join(map(str, installed))} is installed. Upgrade vLLM or set rollout.fp32_head false."
            )
        options["hf_overrides"] = {**(hf_overrides or {}), "head_dtype": "float32"}
    if ssm_state_dtype != "auto":
        options["mamba_ssm_cache_dtype"] = ssm_state_dtype
    return options


def serve_flags(options: dict) -> list[str]:
    """Engine arguments (e.g. precision_options) as `vllm serve` flags; mappings as JSON."""
    flags = []
    for key, value in options.items():
        flags += ["--" + key.replace("_", "-"), json.dumps(value) if isinstance(value, dict) else str(value)]
    return flags


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
        self.started: dict[str, int] = {}  # streaming: request id -> the version its decoding started with
        self.candidates: dict[str, int] = {}  # streaming: request id -> its Sampling.candidates (when > 0)
        self.adapters: dict = {}  # LoRA adapters by name (load_adapter), for requests that name one
        self.adapter_ids: dict[str, int] = {}

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

    @staticmethod
    def _request(prompt, images=None) -> dict:
        """A vLLM tokens prompt. Images go next to the un-expanded tokens (one placeholder each):
        vLLM runs the model's processor on them and expands the placeholders itself."""
        request = {"prompt_token_ids": list(prompt)}
        if images:
            from palingenesis.multimodal import load_image

            request["multi_modal_data"] = {"image": [load_image(image) for image in images]}
        return request

    def load_adapter(self, name: str, path: str, version: int) -> None:
        """Make the LoRA adapter saved at `path` (PEFT layout) what requests for the adapter `name` use from now
        on. Each version is an adapter of its own to vLLM (its own name and id), loaded when the first request
        names it: requests in flight keep the version they started with, a new version never reads another's
        files, and cached prefixes of an older version are not reused (the prefix cache is reset as well). The
        engine must have been built with enable_lora (engine_kwargs: enable_lora, max_loras, max_lora_rank), with
        max_loras at least twice the number of adapters. Not run on a GPU yet."""
        from vllm.lora.request import LoRARequest

        self.adapter_ids[name] = number = (
            len(self.adapter_ids) + 1 if name not in self.adapter_ids else self.adapter_ids[name]
        )
        request = LoRARequest(lora_name=f"{name}-v{version}", lora_int_id=number * 1_000_000 + version, lora_path=path)
        self.adapters[name] = request
        self.llm.reset_prefix_cache()

    def _lora(self, adapter: str) -> dict:
        if not adapter or adapter == "base":
            return {}
        if adapter not in self.adapters:
            raise KeyError(
                f"the adapter {adapter!r} was not loaded into this engine (loaded: {sorted(self.adapters)}): "
                "adapters reach an engine with the trainer's weight updates (rollout.push native)."
            )
        return {"lora_request": self.adapters[adapter]}

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = ""):
        """One completion per prompt. Identical requests (a GRPO group's rollouts) become one
        request with n samples: the prompt is prefilled once and its KV cache forked, instead
        of relying on the prefix cache to find it n times. `adapter`: a loaded LoRA adapter (load_adapter)."""
        unique: dict[tuple, list[int]] = {}
        pictures: dict[tuple, list] = {}
        images = images or [None] * len(prompts)
        for i, (prompt, budget, sampling) in enumerate(
            zip(prompts, max_new_tokens, samplings(temperature, len(prompts)))
        ):
            key = (tuple(prompt), budget, sampling, tuple(id(image) for image in images[i] or ()))
            unique.setdefault(key, []).append(i)
            pictures[key] = images[i]
        requests = list(unique.items())
        # greedy decoding is deterministic and vLLM refuses n > 1 without sampling: one sample, shared by the duplicates
        greedy = [samplings(sampling, 1)[0].temperature <= 0 for (_, _, sampling, _), _ in requests]
        params = [
            self._params(1 if g else len(indices), budget, sampling)
            for g, ((_, budget, sampling, _), indices) in zip(greedy, requests)
        ]
        outputs = self.llm.generate(
            [self._request(key[0], pictures[key]) for key, _ in requests],
            params,
            use_tqdm=False,
            **self._lora(adapter),
        )
        rollouts: list[Rollout | None] = [None] * len(prompts)
        for g, ((_, _, sampling, _), indices), out in zip(greedy, requests, outputs):
            got = self._rollouts(out, candidates=0 if g else samplings(sampling, 1)[0].candidates)
            for k, i in enumerate(indices):
                rollouts[i] = got[0] if g else got[k]
        return rollouts

    def _params(self, n: int, budget: int, sampling, final_only: bool = False):
        (sampling,) = samplings(sampling, 1)
        kwargs = {}
        if final_only:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY
        return self.SamplingParams(
            n=n,
            max_tokens=budget,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            presence_penalty=sampling.presence_penalty,
            # the sampled token's log-prob, and Sampling.candidates more (processed: after temperature, top-k, top-p)
            logprobs=sampling.candidates if sampling.temperature > 0 else None,
            stop_token_ids=list(self.stop_ids),
            detokenize=False,
            **kwargs,
        )

    def _rollouts(self, out, first_version: int | None = None, candidates: int = 0) -> list[Rollout]:
        rollouts = []
        for o in out.outputs:
            tokens = list(o.token_ids)
            logprobs = [step[t].logprob for step, t in zip(o.logprobs, tokens)] if o.logprobs else []
            finish = "stop" if o.finish_reason == "stop" else "length"
            routed = getattr(o, "routed_experts", None)  # with enable_return_routed_experts (MoE routing replay)
            top = candidates_from_steps(o.logprobs, candidates) if candidates and o.logprobs else None
            rollouts.append(Rollout(tokens, logprobs, finish, self.version, first_version, routed, top))
        return rollouts

    # Streaming (continuous batching across calls): requests join the running batch the moment
    # they are added and come back the step they finish. One thread must own these calls.

    def stream_add(
        self,
        request_id: str,
        prompt: list[int],
        max_new_tokens: int,
        temperature,
        n: int,
        images=None,
        adapter: str = "",
    ) -> None:
        params = self._params(n, max_new_tokens, temperature, final_only=True)
        self.llm.llm_engine.add_request(request_id, self._request(prompt, images), params, **self._lora(adapter))
        self.started[request_id] = self.version  # when added, not at the first token: an upper bound on staleness
        (sampling,) = samplings(temperature, 1)
        if sampling.candidates and sampling.temperature > 0:
            self.candidates[request_id] = sampling.candidates

    def load_stats(self) -> dict | None:
        """The engine's load (rollout.concurrency kv on an engine worker, palingenesis.rl.concurrency): KV-cache
        usage (a fraction), requests waiting and running, and the KV cache's size in tokens, read from the
        in-process scheduler (vLLM V1 without multiprocessing). None where this vLLM does not expose them."""
        engine = self.llm.llm_engine
        core = getattr(getattr(engine, "engine_core", None), "engine_core", None)  # V1 InprocClient -> EngineCore
        scheduler = getattr(core, "scheduler", None)
        manager = getattr(scheduler, "kv_cache_manager", None)
        if scheduler is None or manager is None or not hasattr(manager, "usage"):
            return None
        out: dict = {
            "kv_usage": float(manager.usage),
            "waiting": float(len(scheduler.waiting)),
            "running": float(len(scheduler.running)),
        }
        cache = getattr(getattr(engine, "vllm_config", None), "cache_config", None)
        if getattr(cache, "num_gpu_blocks", None) and getattr(cache, "block_size", None):
            out["kv_tokens"] = int(cache.num_gpu_blocks) * int(cache.block_size)
        return out

    def stream_pending(self) -> bool:
        return self.llm.llm_engine.has_unfinished_requests()

    def stream_step(self) -> list[tuple[str, list[Rollout]]]:
        """One engine step: (request id, its n rollouts) for every request that finished.
        Counts the finished requests' prompt tokens and those served by the prefix cache."""
        finished = [out for out in self.llm.llm_engine.step() if out.finished]
        for out in finished:
            self.prompt_tokens += len(out.prompt_token_ids or ())
            self.cached_tokens += out.num_cached_tokens or 0
        return [
            (
                out.request_id,
                self._rollouts(out, self.started.pop(out.request_id, None), self.candidates.pop(out.request_id, 0)),
            )
            for out in finished
        ]


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
        log = getattr(self, "_log", None)  # a server reached at `url` has none
        if log is not None:
            log.close()
            self._log = None


class VLLMServerRollout:
    """Rollouts from a vLLM server; weights pushed over CUDA IPC (the server must be
    on the trainer's GPU and started with --weight-transfer-config '{"backend": "ipc"}').

    Each update pauses the server between rollout batches, sends CUDA IPC handles
    of bf16 copies of the weights (vLLM's native start/update/finish weight
    update endpoints; the server loads them with the model's load_weights), and
    resumes. The copies live until the server has loaded them.
    """

    def __init__(self, server: VLLMServer, stop_ids: tuple[int, ...]):
        if is_rocm():
            logger.warning(
                "rollout.backend vllm_server on ROCm is untested: the weight push uses torch's CUDA IPC handles "
                "(HIP IPC on AMD). Check that rollouts after a weight update come from the updated policy (in RL, "
                "policy/abs_ratio_dev stays near 0), or use rollout.backend vllm."
            )
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
        per_prompt = samplings(temperature, len(prompts))
        for budget, sampling in sorted(set(zip(max_new_tokens, per_prompt)), key=lambda x: (x[0], repr(x[1]))):
            idx = [i for i, (m, sp) in enumerate(zip(max_new_tokens, per_prompt)) if m == budget and sp == sampling]
            choices = self.server.complete(
                [prompts[i] for i in idx],
                max_tokens=budget,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                presence_penalty=sampling.presence_penalty,
                logprobs=sampling.candidates if sampling.temperature > 0 else None,
                stop_token_ids=list(self.stop_ids),
                return_token_ids=True,
                skip_special_tokens=False,
                # candidates: top_logprobs keyed by token id ("token_id:<id>"), not by text
                **({"return_tokens_as_token_ids": True} if sampling.candidates else {}),
            )
            want = sampling.candidates if sampling.temperature > 0 else 0
            for i, choice in zip(idx, choices):
                logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
                top = None
                if want:
                    top = candidates_from_openai((choice.get("logprobs") or {}).get("top_logprobs"), want)
                    if len(top) != len(choice["token_ids"]):
                        raise RuntimeError(
                            f"vLLM server: top log-probs for {len(top)} of {len(choice['token_ids'])} sampled tokens "
                            "(candidates need one entry per token)"
                        )
                out[i] = Rollout(
                    choice["token_ids"],
                    logprobs,
                    "stop" if choice["finish_reason"] == "stop" else "length",
                    self.version,
                    routed_experts=decode_routed_experts(choice.get("routed_experts")),
                    candidates=top,
                )
        return out
