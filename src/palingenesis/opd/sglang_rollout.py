"""SGLang rollout engines: the same RolloutEngine protocol as the vLLM ones (opd.rollout), on SGLang.

  SGLangRollout        an sglang.Engine on the trainer's GPU (rollout.backend sglang). The engine's scheduler
                       runs in a child process; weights reach it as CUDA IPC handles (update_weights_from_tensor),
                       in buckets, and with sleep mode it releases its weights, KV cache and CUDA graphs while the
                       trainer trains (release_memory_occupation / resume_memory_occupation, torch_memory_saver).
  SGLangServerRollout  a separate SGLang server (rollout.backend sglang_server): launched by the trainer on its
                       GPU, or a running one at rollout.url (another GPU or node). Weights travel as CUDA IPC
                       handles (same GPU), by NCCL broadcast from the trainer (another GPU or node), or through a
                       checkpoint on a shared file system (rollout.weight_sync).

Both return, per completion, the sampled tokens and the log-probability of each under the sampling distribution
(temperature applied; with top-k / top-p, SGLang's renormalised "sampling mask" log-probs where the installed
SGLang returns them, else the untruncated ones, see below), the finish reason, the policy version, with routing
replay the experts every token was routed to, and with Sampling.candidates the sampler's candidates of every
sampled token (candidates_from_output: with top-k / top-p, the support SGLang sampled from). The radix (prefix) cache is flushed with every weight update, so no
prompt is ever served from KV computed by older weights.

Differences from the vLLM engines (see the RL guide's backend table):
  * A weight update waits for the requests in flight to finish (SGLang's model-update lock): rollouts are never
    continued across weight versions (vLLM's partial rollouts), and every completion has one version.
  * The behaviour log-probs of truncated samplings (top_k / top_p) need SGLang's return_sampling_mask
    (sglang >= 0.5.12, and a top_k: SGLang reports no mask for top_p alone); otherwise they are the untruncated
    temperature-scaled log-probs (a warning), and the support is not known at all: Sampling.candidates with
    top-k / top-p (pgs rl's rollout.top_p / top_k) is then an error, not a rollout scored over another support.
  * The support of a truncated sampling is SGLang's own (its sampling mask), which is not vLLM's: the nucleus is
    cut on the temperature-scaled probabilities as they are, not renormalised after top-k (so top_p can keep
    more tokens than vLLM's, which applies it to the top-k distribution), and at most top_k tokens are kept
    (vLLM keeps every token tied with the k-th logit). The trainer renormalises over the ids the engine reports, so
    either convention gives exact ratios.
  * No images (model.multimodal): the engine runs with skip_tokenizer_init and takes token ids only.

The patterns follow slime (THUDM/slime) and verl (volcengine/verl), both Apache-2.0: bucketed CUDA IPC weight
pushes into a colocated engine, release / resume of the engine's memory around each training step (weights resumed
before the update, KV cache after it), and a custom NCCL group between trainer and server for remote engines.
They are reimplemented here against the RolloutEngine protocol.

sglang is imported lazily: nothing here imports it until an engine is built.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import dataclasses
import difflib
import io
import json
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from palingenesis.opd.rollout import Candidates, Rollout, Sampling, _die_with_parent, _free_port, samplings

logger = logging.getLogger(__name__)

# The SGLang release the GPU checks ran on (scripts/gpu_checks/sglang.sh): the last one built on torch 2.11.
TESTED_SGLANG = "0.5.17"

# rollout.kv_cache_dtype as SGLang names it ("fp8" is vLLM's name for e4m3)
KV_CACHE_DTYPES = {"fp8": "fp8_e4m3"}

# Engine arguments palingenesis sets from its own options: rollout.sglang_args may not override them.
OWNED_OPTIONS = {
    "model_path": "model.policy / model.student",
    "skip_tokenizer_init": "(always true: the engine takes and returns token ids)",
    "dtype": "(always bfloat16, the trainer's compute dtype)",
    "mem_fraction_static": "rollout.gpu_memory_utilization",
    "context_length": "rollout.max_model_len",
    "random_seed": "train.seed",
    "disable_radix_cache": "rollout.prefix_caching",
    "enable_memory_saver": "rollout.sleep",
    "disable_cuda_graph": "rollout.enforce_eager",
    "max_running_requests": "rollout.max_num_seqs",
    "enable_fp32_lm_head": "rollout.fp32_head",
    "mamba_ssm_dtype": "rollout.ssm_state_dtype",
    "kv_cache_dtype": "rollout.kv_cache_dtype",
    "quantization": "rollout.quantization",
    "enable_return_routed_experts": "rollout.routing_replay",
}

WEIGHT_SYNC_MODES = ("auto", "tensor", "distributed", "disk")
SGLANG_BACKENDS = ("sglang", "sglang_server")

# torchrun's variables: the engine's scheduler process sets up its own torch.distributed group (tcp:// on a free
# port), and with TORCHELASTIC_USE_AGENT_STORE torch would wait for torchrun's agent store there instead.
LAUNCHER_ENV = (
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "RANK",
    "LOCAL_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _import_sglang():
    try:
        import sglang
    except ImportError as e:
        raise ImportError(
            "rollout.backend 'sglang' / 'sglang_server' needs SGLang: install it in its own environment, "
            f"pip install 'sglang=={TESTED_SGLANG}' (Linux, NVIDIA; it pins torch 2.11 and its own transformers), "
            "then palingenesis there. See the RL guide, 'Rollout engines'."
        ) from e
    return sglang


def installed_sglang_version() -> tuple[int, ...] | None:
    """The installed SGLang's release numbers, e.g. (0, 5, 17); None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        text = version("sglang")
    except PackageNotFoundError:
        return None
    from packaging.version import Version

    return Version(text).release


# ------------------------------------------------------------------ requests and outputs


def sampling_params(sampling: Sampling, max_new_tokens: int, stop_ids: Iterable[int]) -> dict:
    """SGLang sampling_params for one completion. top_k 0 (off) is SGLang's -1; temperature 0 is greedy (SGLang
    turns it into top_k 1). Stop tokens are kept in the output and special tokens are not skipped: the trainer
    trains on exactly the sampled ids."""
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(sampling.temperature),
        "top_p": float(sampling.top_p),
        "top_k": int(sampling.top_k) if sampling.top_k > 0 else -1,
        "presence_penalty": float(sampling.presence_penalty),
        "stop_token_ids": sorted(int(t) for t in stop_ids),
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "n": 1,
    }


def clamp_budget(prompt_len: int, max_new_tokens: int, max_model_len: int) -> int:
    """The completion budget that fits the context: SGLang refuses a request whose prompt plus max_new_tokens
    exceeds its context length, where vLLM stops the completion there ("length"). Raises for a prompt that
    leaves no room at all, as vLLM does."""
    room = max_model_len - prompt_len
    if room < 1:
        raise ValueError(
            f"a prompt of {prompt_len} tokens leaves no room in rollout.max_model_len {max_model_len}: drop "
            "longer prompts (data.max_prompt_tokens) or raise rollout.max_model_len."
        )
    return max(1, min(int(max_new_tokens), room))


def build_request(
    prompt: list[int],
    max_new_tokens: int,
    sampling: Sampling,
    stop_ids: Iterable[int],
    max_model_len: int,
    routing: bool = False,
    lora: str | None = None,
) -> dict:
    """One SGLang generate request (GenerateReqInput fields) for a token prompt."""
    request: dict[str, Any] = {
        "input_ids": [int(t) for t in prompt],
        "sampling_params": sampling_params(
            sampling, clamp_budget(len(prompt), max_new_tokens, max_model_len), stop_ids
        ),
        "return_logprob": sampling.temperature > 0,
    }
    if sampling.temperature > 0 and sampling.top_k > 0:
        # the renormalised log-probs of the truncated distribution, and its support. SGLang aborts a request for
        # the mask of a top_p-only sampling (no bound on the support): that one returns untruncated log-probs
        request["return_sampling_mask"] = True
    if sampling.temperature > 0 and sampling.candidates:
        # the most likely tokens of every sampled position (temperature-scaled, untruncated): with the sampling
        # mask, the sampler's candidates (candidates_from_output)
        request["top_logprobs_num"] = int(sampling.candidates)
    if routing:
        request["return_routed_experts"] = True
    if lora:
        request["lora_path"] = lora
    return request


def finish_type(meta: dict) -> str:
    """SGLang's finish reason ({"type": "stop" | "length" | "abort", ...}, or a plain string) as "stop" / "length".
    An aborted request raises: it was cut by the engine (out of memory, a refused input), not by its budget."""
    reason = meta.get("finish_reason")
    kind = reason.get("type") if isinstance(reason, dict) else reason
    if kind == "abort":
        message = reason.get("message", "") if isinstance(reason, dict) else ""
        raise RuntimeError(f"SGLang aborted a rollout request: {message or reason}")
    return "stop" if kind == "stop" else "length"


def output_tokens_and_logprobs(
    out: dict, want_logprobs: bool, truncated: bool = False
) -> tuple[list[int], list[float]]:
    """The completion's token ids and, when asked for, each one's log-probability, aligned one to one.

    SGLang reports the sampled tokens twice: output_ids, and the token ids inside meta_info.output_token_logprobs
    ((logprob, token id, text) per sampled token). They agree, except when output_ids leave out a matched stop
    token; the log-probs' ids are then the full completion. Anything else is an error (a log-prob would be
    attributed to another token). `truncated`: the sampling was top-k / top-p truncated; the renormalised
    meta_info.output_token_sampling_logprobs replace the untruncated ones when the engine returned them."""
    tokens = [int(t) for t in out.get("output_ids") or []]
    if not want_logprobs:
        return tokens, []
    meta = out.get("meta_info") or {}
    entries = meta.get("output_token_logprobs")
    if entries is None:
        raise RuntimeError("SGLang returned no output_token_logprobs: the request must set return_logprob")
    ids = [int(entry[1]) for entry in entries]
    logprobs = [float(entry[0]) for entry in entries]
    if truncated:
        renormalised = meta.get("output_token_sampling_logprobs")
        if renormalised is not None and len(renormalised) == len(logprobs):
            logprobs = [float(x) for x in renormalised]
        else:
            _warn_once(
                "truncated",
                "SGLang did not return the truncated sampling's log-probs (return_sampling_mask: sglang >= 0.5.12, "
                "and a top_k, which bounds the support it reports): top-k / top-p rollouts carry the untruncated "
                "temperature-scaled log-probs",
            )
    if ids != tokens:
        if len(ids) == len(tokens) + 1 and ids[:-1] == tokens:
            tokens = ids  # output_ids trimmed the stop token
        elif not tokens:
            tokens = ids
        else:
            raise RuntimeError(
                f"SGLang's output_ids ({len(tokens)} tokens) and its log-probs ({len(ids)} tokens) disagree; "
                "the log-probs cannot be aligned to the sampled tokens"
            )
    return tokens, logprobs


def candidates_from_output(out: dict, tokens: list[int], k: int, truncated: bool) -> Candidates:
    """The sampler's candidates (opd.rollout.Candidates) of the sampled `tokens` of an SGLang output: per position
    its `k` most likely tokens and their log-probs under the distribution the token was drawn from.

    SGLang returns two things. meta_info.output_top_logprobs (top_logprobs_num): the k most likely tokens of the
    temperature-scaled distribution p, before top-k / top-p. With a truncated sampling,
    meta_info.output_token_sampling_mask (return_sampling_mask): the ids of the support S the token was drawn
    from, as the sampler built it (the first top_k tokens whose preceding cumulative p is <= top_p, p not
    renormalised after top-k; plus the sampled token, should a kernel have drawn it outside), and
    output_token_sampling_logprobs, log(p(y) / p(S)) of the sampled token.

    Plain temperature sampling: the candidates are the top log-probs as they are. Truncated: the support's tokens
    with log p(v) - log p(S), where log p(S) = log p(y) - log(p(y) / p(S)) is the engine's own normaliser (the
    sampled token's candidate log-prob is exactly its behaviour log-prob). A support of more than k tokens (top_p
    without top_k) gives its k most likely: k candidates, which the trainer reads as a support it may not have
    seen whole, as with vLLM. A support token whose log-prob was not returned although the support has at most k
    tokens (more tokens tied at the k-th log-prob than k leaves room for) cannot be renormalised over: that
    position's candidates leave out the sampled token, so the trainer leaves the token out of the loss
    (sampling/support_incomplete) rather than train it over part of its support."""
    meta = out.get("meta_info") or {}
    top = meta.get("output_top_logprobs")
    entries = meta.get("output_token_logprobs") or []
    if top is None or len(top) != len(tokens) or len(entries) != len(tokens):
        raise RuntimeError(
            f"SGLang returned top log-probs for {0 if top is None else len(top)} of {len(tokens)} sampled tokens "
            "(top_logprobs_num): the sampler's candidates (rollout.top_p / top_k, loss.score_centering) need one "
            "entry per token"
        )
    masks = renormalised = None
    if truncated:
        masks = meta.get("output_token_sampling_mask")
        renormalised = meta.get("output_token_sampling_logprobs")
        if masks is None or renormalised is None or len(masks) != len(tokens) or len(renormalised) != len(tokens):
            raise RuntimeError(
                "SGLang did not return the support of its top-k / top-p sampling (return_sampling_mask: "
                f"sglang >= 0.5.12, tested {TESTED_SGLANG}; it reports no support for top_p without top_k): the "
                "trainer cannot renormalise the policy over it. Upgrade SGLang and set a top_k, or sample at plain "
                "temperature (rollout.top_p 1, rollout.top_k 0)."
            )
    rows = []
    for t, token in enumerate(tokens):
        known = {int(entry[1]): _logprob(entry[0]) for entry in top[t] or []}
        sampled = _logprob(entries[t][0])
        known.setdefault(token, sampled)
        if masks is None:
            ranked = sorted(known.items(), key=lambda x: -x[1])[:k]
        else:
            if masks[t] is None or renormalised[t] is None:
                raise RuntimeError("SGLang returned no sampling mask for a sampled token (return_sampling_mask)")
            support = [int(i) for i in masks[t]]
            log_mass = sampled - _logprob(renormalised[t])  # log p(S), the engine's normaliser
            if not math.isfinite(log_mass):
                raise RuntimeError(
                    f"SGLang's log-probs of sampled token {token} are not finite ({sampled}, renormalised "
                    f"{renormalised[t]}): its support cannot be renormalised"
                )
            ranked = sorted(((i, known[i] - log_mass) for i in support if i in known), key=lambda x: -x[1])[:k]
            if len(ranked) < min(len(support), k):  # a support token without a log-prob: not this token's support
                ranked = [(i, lp) for i, lp in ranked if i != token]
        rows.append(([i for i, _ in ranked], [lp for _, lp in ranked]))
    return Candidates.from_rows(rows)


def _logprob(value) -> float:
    """A log-prob of an SGLang output as a float (-inf arrives as None through JSON)."""
    return float("-inf") if value is None else float(value)


def moe_layer_indices(config) -> list[int] | None:
    """Indices of the MoE layers among a Hugging Face config's decoder layers (None: not a MoE config). SGLang
    returns routed experts for every decoder layer; the trainer's routing replay takes the MoE layers only."""
    config = getattr(config, "text_config", None) or config
    layers = getattr(config, "num_hidden_layers", None)
    experts = next(
        (getattr(config, a) for a in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(config, a, 0)),
        0,
    )
    if not layers or not experts:
        return None
    if getattr(config, "first_k_dense_replace", None) is not None:  # DeepSeek-V3 family
        first = int(config.first_k_dense_replace)
        every = int(getattr(config, "moe_layer_freq", 1) or 1)
        return [i for i in range(layers) if i >= first and i % every == 0]
    dense = set(getattr(config, "mlp_only_layers", None) or [])
    step = int(getattr(config, "decoder_sparse_step", 1) or 1)
    return [i for i in range(layers) if i not in dense and (i + 1) % step == 0]


def decode_routed_experts(value, positions: int, top_k: int, moe_layers: list[int] | None = None):
    """meta_info.routed_experts (SGLang: base64 of an int32 array [positions, decoder layers, top_k], positions =
    prompt + completion - 1; or an array / nested list) as an int32 array [positions, MoE layers, top_k], keeping
    the MoE layers' rows. None when absent or when the size does not fit that layout (the trajectory then trains
    without replay)."""
    if value is None:
        return None
    import numpy as np

    if isinstance(value, (str, bytes)):
        array = np.frombuffer(base64.b64decode(value), dtype=np.int32)
    else:
        array = np.asarray(value, dtype=np.int32)
    if positions < 1 or top_k < 1 or array.size % (positions * top_k):
        _warn_once("routing_shape", "SGLang's routed experts do not fit [positions, layers, top_k]: no replay")
        return None
    array = array.reshape(positions, -1, top_k)
    if moe_layers is not None and array.shape[1] != len(moe_layers) and array.shape[1] > max(moe_layers, default=-1):
        array = array[:, moe_layers]
    return array


_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


# ------------------------------------------------------------------ engine arguments


def engine_options(
    model: str,
    *,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool = False,
    seed: int = 0,
    sleep_mode: bool = False,
    prefix_caching: bool = True,
    max_num_seqs: int = 0,
    fp32_head: bool = False,
    ssm_state_dtype: str = "float32",
    kv_cache_dtype: str = "",
    quantization: str = "",
    routing_replay: bool = False,
    extra: dict | None = None,
) -> dict:
    """sglang.Engine / ServerArgs arguments for a rollout engine. `extra` (rollout.sglang_args) adds others; it
    may not set the ones palingenesis derives from its own options (OWNED_OPTIONS: ValueError naming the option
    to use instead)."""
    options: dict[str, Any] = {
        "model_path": model,
        "dtype": "bfloat16",
        "skip_tokenizer_init": True,
        "mem_fraction_static": float(gpu_memory_utilization),
        "context_length": int(max_model_len),
        "random_seed": int(seed),
        "disable_radix_cache": not prefix_caching,
        "enable_memory_saver": bool(sleep_mode),
    }
    if enforce_eager:
        options["disable_cuda_graph"] = True
    if max_num_seqs > 0:  # else SGLang sizes it from memory (and from the recurrent-state slots of hybrid models)
        options["max_running_requests"] = int(max_num_seqs)
    if fp32_head:
        options["enable_fp32_lm_head"] = True
    if ssm_state_dtype != "auto":
        options["mamba_ssm_dtype"] = ssm_state_dtype
    if kv_cache_dtype:
        options["kv_cache_dtype"] = KV_CACHE_DTYPES.get(kv_cache_dtype, kv_cache_dtype)
    if quantization:
        options["quantization"] = quantization
    if routing_replay:
        options["enable_return_routed_experts"] = True
    problems = owned_option_errors(extra or {})
    if problems:
        raise ValueError(" ".join(problems))
    options.update(extra or {})
    return options


def owned_option_errors(extra: dict, where: str = "rollout.sglang_args") -> list[str]:
    """Errors for engine arguments in `extra` that palingenesis sets itself."""
    return [
        f"{where}.{key} is set by {OWNED_OPTIONS[key]}; remove it from {where}."
        for key in extra
        if key in OWNED_OPTIONS
    ]


def config_errors(r, section: str, multimodal: bool = False) -> list[str]:
    """Errors in the SGLang options of a rollout config section (pgs rl and pgs distill share them): `r` has
    backend, sglang_args, weight_sync, weight_sync_dir, weight_bucket_mb (and vllm_args where it exists)."""
    errors = []
    sglang = r.backend in SGLANG_BACKENDS
    if r.sglang_args and not sglang:
        errors.append(
            f"{section}.sglang_args are SGLang engine arguments: they need {section}.backend sglang or sglang_server."
        )
    if sglang:
        errors += owned_option_errors(r.sglang_args, f"{section}.sglang_args")
        if getattr(r, "vllm_args", None):
            errors.append(f"{section}.vllm_args are vLLM engine arguments; with SGLang use {section}.sglang_args.")
        if multimodal:
            errors.append(
                f"model.multimodal with {section}.backend {r.backend} is not supported: the SGLang engines take token "
                "ids only. Use backend vllm (in-process) or hf."
            )
    if r.weight_sync not in WEIGHT_SYNC_MODES:
        errors.append(f"{section}.weight_sync must be one of {WEIGHT_SYNC_MODES}, got {r.weight_sync!r}.")
    elif r.weight_sync != "auto" and r.backend != "sglang_server":
        errors.append(f"{section}.weight_sync applies to {section}.backend sglang_server only.")
    elif r.weight_sync == "distributed" and not getattr(r, "url", ""):
        errors.append(
            f"{section}.weight_sync distributed needs a server on another GPU ({section}.url): the server the "
            "trainer launches shares its GPU, where NCCL cannot join the two (use tensor, the default)."
        )
    if r.weight_sync_dir and r.weight_sync != "disk":
        errors.append(f"{section}.weight_sync_dir is used with {section}.weight_sync disk only.")
    if r.weight_bucket_mb < 1:
        errors.append(f"{section}.weight_bucket_mb must be >= 1, got {r.weight_bucket_mb}.")
    return errors


def check_options(options: dict, known: Iterable[str]) -> None:
    """ValueError naming every argument the installed SGLang does not have (with the closest names)."""
    known = set(known)
    unknown = [key for key in options if key not in known]
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, known, n=2)
            hints.append(f"{key}" + (f" (did you mean {' or '.join(close)}?)" if close else ""))
        version = installed_sglang_version()
        raise ValueError(
            f"SGLang {'.'.join(map(str, version)) if version else ''} has no engine argument "
            + ", ".join(hints)
            + f". Arguments the rollout options need: enable_fp32_lm_head (rollout.fp32_head), mamba_ssm_dtype "
            f"(rollout.ssm_state_dtype), enable_return_routed_experts (rollout.routing_replay); tested with "
            f"sglang {TESTED_SGLANG}."
        )


def server_flags(options: dict) -> list[str]:
    """Engine arguments as `python -m sglang.launch_server` flags: booleans as switches (false: left out),
    mappings and lists as JSON."""
    flags: list[str] = []
    for key, value in options.items():
        if key == "model_path" or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif isinstance(value, (dict, list)):
            flags += [flag, json.dumps(value)]
        else:
            flags += [flag, str(value)]
    return flags


# Variables that change what an engine returns: with SGLANG_RETURN_ORIGINAL_LOGPROB SGLang reports the log-probs of
# the unscaled logits (temperature 1), not those of the distribution it sampled from.
SEMANTICS_ENV = ("SGLANG_RETURN_ORIGINAL_LOGPROB",)


def engine_env(environ=None) -> dict[str, str]:
    """The environment an engine process starts with: `environ` (os.environ) without torchrun's variables and
    without the ones that would change the meaning of the returned log-probs (a warning names those set)."""
    environ = os.environ if environ is None else environ
    for key in SEMANTICS_ENV:
        if environ.get(key, "").lower() in ("1", "true", "yes"):
            _warn_once(
                key,
                f"{key} is set: the rollout engine is started without it (it must return the log-probs of the "
                "temperature-scaled distribution it samples from)",
            )
    return {k: v for k, v in environ.items() if k not in LAUNCHER_ENV and k not in SEMANTICS_ENV}


@contextlib.contextmanager
def launcher_env_removed():
    """torchrun's variables (and SEMANTICS_ENV) out of os.environ while engine processes are spawned (they copy
    it)."""
    kept = engine_env()
    saved = {key: os.environ.pop(key) for key in list(os.environ) if key not in kept}
    try:
        yield
    finally:
        os.environ.update(saved)


# ------------------------------------------------------------------ weights


def weight_buckets(
    named: Iterable[tuple[str, Tensor]], bucket_bytes: int, dtype: torch.dtype = torch.bfloat16
) -> Iterator[list[tuple[str, Tensor]]]:
    """The weights in buckets of at most `bucket_bytes` (a larger tensor makes a bucket of its own), floating
    point ones cast to `dtype`. Each bucket's tensors are views of one contiguous buffer when they share a dtype
    and device: one CUDA IPC handle per bucket instead of one per tensor. A bucket is yielded only when it is
    full, so a push holds at most two buckets of copies (the one in flight and the next) besides the weights."""
    pending: list[tuple[str, Tensor]] = []
    size = 0
    for name, tensor in named:
        tensor = tensor.detach()
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        nbytes = tensor.numel() * tensor.element_size()
        if pending and size + nbytes > bucket_bytes:
            yield _packed(pending)
            pending, size = [], 0
        pending.append((name, tensor))
        size += nbytes
    if pending:
        yield _packed(pending)


def _packed(items: list[tuple[str, Tensor]]) -> list[tuple[str, Tensor]]:
    if len({(t.dtype, t.device) for _, t in items}) != 1:
        return [(name, t.contiguous()) for name, t in items]
    flat = torch.empty(sum(t.numel() for _, t in items), dtype=items[0][1].dtype, device=items[0][1].device)
    views, offset = [], 0
    for name, tensor in items:
        view = flat[offset : offset + tensor.numel()]
        view.copy_(tensor.reshape(-1))
        views.append((name, view.view(tensor.shape)))
        offset += tensor.numel()
    return views


def _with_last(items: Iterable) -> Iterator[tuple[Any, bool]]:
    """(item, is_last) pairs, one item of lookahead."""
    iterator = iter(items)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for item in iterator:
        yield previous, False
        previous = item
    yield previous, True


def serialize_named_tensors(named: list[tuple[str, Tensor]]) -> str:
    """A list of (name, CUDA tensor) as SGLang's update_weights_from_tensor payload: pickled with torch's
    multiprocessing reductions (CUDA tensors become IPC handles, no copy) and base64-encoded, the format of
    sglang's MultiprocessingSerializer.serialize(..., output_str=True). The tensors must outlive the request."""
    from multiprocessing.reduction import ForkingPickler

    import torch.multiprocessing  # noqa: F401 — registers the tensor reductions with ForkingPickler

    buffer = io.BytesIO()
    ForkingPickler(buffer).dump(named)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _cuda_sync(named: list[tuple[str, Tensor]]) -> None:
    """The copies are complete before another process reads them."""
    if named and named[0][1].is_cuda:
        torch.cuda.synchronize(named[0][1].device)


def _ipc_collect(named: list[tuple[str, Tensor]]) -> None:
    if named and named[0][1].is_cuda:
        torch.cuda.ipc_collect()


# ------------------------------------------------------------------ in-process engine


class InProcessEngine:
    """The calls SGLangRollout makes, on an sglang.Engine built in this process. Tests replace it with a scripted
    engine of the same shape (loop, agenerate, update_weights_from_tensor, release, resume, flush_cache,
    load_lora_adapter, unload_lora_adapter, close)."""

    def __init__(self, options: dict):
        sglang = _import_sglang()
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.server_args import ServerArgs

        check_options(options, (f.name for f in dataclasses.fields(ServerArgs)))
        self.request_fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
        try:
            from sglang.srt.constants import GPU_MEMORY_ALL_TYPES

            self.memory_tags = list(GPU_MEMORY_ALL_TYPES)
        except ImportError:
            self.memory_tags = ["kv_cache", "weights"]
        self.failed = ""  # why the engine is gone (its scheduler process died); calls raise it
        main = threading.current_thread() is threading.main_thread()
        previous = signal.getsignal(signal.SIGTERM) if main else None
        with launcher_env_removed():
            self.engine = sglang.Engine(**options)
        # sglang.Engine kills every child process of this one at exit (its own and any other); ours kills its own
        with contextlib.suppress(Exception):
            atexit.unregister(self.engine.shutdown)
        atexit.register(self.close)
        self.loop = self.engine.loop
        self._helper: ThreadPoolExecutor | None = None
        # The first call starts SGLang's response loop; from the main thread it also installs a SIGTERM handler on
        # that event loop that shuts the process down. The trainer's own SIGTERM handler (checkpoint, then exit)
        # must stay in charge: the event loop's handler is removed again.
        self.run(self.engine.tokenizer_manager.flush_cache())
        if previous is not None:
            with contextlib.suppress(Exception):
                self.loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, previous)
        if main:
            # SGLang's watchdog sends this process SIGQUIT when the scheduler dies, and its handler kills this
            # process and all of its children. The trainer ends with an error instead (its cleanup runs, and
            # other children, sandboxes for instance, are left to it).
            with contextlib.suppress(Exception):
                self.loop.remove_signal_handler(signal.SIGQUIT)
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGQUIT, self._scheduler_died)

    def _scheduler_died(self, signum=None, frame=None) -> None:
        """SIGQUIT from SGLang's subprocess watchdog (main thread): stop what is left of the engine and raise."""
        if self.engine is None:  # closed: its processes were stopped on purpose
            return
        self.failed = (
            "the SGLang engine's scheduler process died (out of memory, a CUDA error, or it was killed): its "
            "output above has the cause. Lower rollout.gpu_memory_utilization or rollout.max_num_seqs if it ran "
            "out of memory."
        )
        with contextlib.suppress(Exception):
            self.close()
        raise RuntimeError(self.failed)

    def check(self) -> None:
        """Raise when the engine's scheduler died (requests would wait for it forever)."""
        if self.failed:
            raise RuntimeError(self.failed)

    def run(self, coroutine):
        """Run a coroutine on the engine's event loop (callers serialize). From a thread whose own event loop is
        running (an asyncio server's executor callbacks are not), it runs on a helper thread."""
        if self.failed:
            coroutine.close()
            raise RuntimeError(self.failed)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.loop.run_until_complete(coroutine)
        if self._helper is None:
            self._helper = ThreadPoolExecutor(1, thread_name_prefix="pgs-sglang")
        return self._helper.submit(self.loop.run_until_complete, coroutine).result()

    async def agenerate(self, request: dict) -> dict:
        from sglang.srt.managers.io_struct import GenerateReqInput

        # fields an older SGLang lacks are optional extras (the sampling mask): left out, the outputs say so
        obj = GenerateReqInput(**{k: v for k, v in request.items() if k in self.request_fields})
        responses = self.engine.tokenizer_manager.generate_request(obj, None)
        try:
            return await responses.__anext__()
        finally:  # leaves SGLang's model-update read lock now, not when the generator is collected
            await responses.aclose()

    def update_weights_from_tensor(self, named: list[tuple[str, Tensor]], flush_cache: bool) -> tuple[bool, str]:
        self.check()
        result = self.engine.update_weights_from_tensor(named, flush_cache=flush_cache)
        return _ok(result)

    def release(self, tags: list[str]) -> None:
        self.check()
        self.engine.release_memory_occupation(tags=tags)

    def resume(self, tags: list[str]) -> None:
        self.check()
        self.engine.resume_memory_occupation(tags=tags)

    def flush_cache(self) -> bool:
        self.check()
        result = self.engine.flush_cache()
        return bool(getattr(result, "success", result))

    def load_lora_adapter(self, name: str, path: str) -> tuple[bool, str]:
        self.check()
        return _ok(self.engine.load_lora_adapter(name, path))

    def unload_lora_adapter(self, name: str) -> tuple[bool, str]:
        self.check()
        return _ok(self.engine.unload_lora_adapter(name))

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        self.engine = None
        pids = []
        with contextlib.suppress(Exception):
            pids = list(engine.get_all_child_pids())
        if not pids:
            with contextlib.suppress(Exception):
                engine.shutdown()
            return
        with contextlib.suppress(Exception):
            watchdog = engine.tokenizer_manager._subprocess_watchdog
            if watchdog is not None:
                watchdog.stop()
        _kill_trees(pids)


def _ok(result) -> tuple[bool, str]:
    """SGLang's (success, message[, ...]) results, or objects / mappings with success and message (the LoRA
    calls report theirs as error_message)."""
    if isinstance(result, tuple):
        return bool(result[0]), str(result[1]) if len(result) > 1 else ""
    get = result.get if isinstance(result, dict) else lambda key, default=None: getattr(result, key, default)
    return bool(get("success", True)), str(get("message", None) or get("error_message", None) or "")


def _kill_trees(pids: list[int], timeout: float = 30.0) -> None:
    """Terminate these processes and their descendants, then kill what is left after `timeout`."""
    try:
        import psutil
    except ImportError:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        return
    procs = []
    for pid in pids:
        with contextlib.suppress(psutil.Error):
            parent = psutil.Process(pid)
            procs += parent.children(recursive=True) + [parent]
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            proc.terminate()
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    psutil.wait_procs(alive, timeout=10)


class _Completions:
    """Turning SGLang outputs into Rollouts, with the prefill accounting (engine/prefix_hit) and routing replay."""

    prompt_tokens = 0
    cached_tokens = 0
    routing: tuple[int, list[int] | None] | None = None  # (top_k, MoE layer indices) with routing replay

    def _routing_layout(self, model: str) -> None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model)
        text = getattr(config, "text_config", None) or config
        self.routing = (int(getattr(text, "num_experts_per_tok", 0) or 0), moe_layer_indices(config))

    def _rollout(self, out: dict, request: dict, version: int, first_version: int | None = None) -> Rollout:
        meta = out.get("meta_info") or {}
        params = request["sampling_params"]
        truncated = params["top_p"] < 1.0 or params["top_k"] > 0
        finish = finish_type(meta)  # first: an aborted request has no log-probs to align
        tokens, logprobs = output_tokens_and_logprobs(out, request["return_logprob"], truncated)
        self.prompt_tokens += int(meta.get("prompt_tokens") or len(request["input_ids"]))
        self.cached_tokens += int(meta.get("cached_tokens") or 0)
        routed = None
        if self.routing is not None:
            top_k, layers = self.routing
            positions = len(request["input_ids"]) + len(tokens) - 1
            routed = decode_routed_experts(meta.get("routed_experts"), positions, top_k, layers)
        top = None
        if request.get("top_logprobs_num"):  # Sampling.candidates
            top = candidates_from_output(out, tokens, request["top_logprobs_num"], truncated)
        return Rollout(tokens, logprobs, finish, version, first_version, routed, top)

    @staticmethod
    def _no_images(images) -> None:
        if images and any(images):
            raise ValueError(
                "the SGLang rollout engines take token ids only (model.multimodal is not supported): use "
                "rollout.backend vllm or hf for images"
            )


class SGLangRollout(_Completions):
    """An sglang.Engine sharing the trainer's GPU (rollout.backend sglang).

    Weights: after each optimizer step, `update_weights` copies them to bf16 in buckets of `bucket_mb` and sends
    each bucket's CUDA IPC handle (update_weights_from_tensor); the last bucket flushes the radix cache. SGLang
    applies an update once the requests in flight have finished.

    Sleep (sleep_mode, max_staleness 0): `sleep` releases the weights, the KV cache and the CUDA graphs
    (release_memory_occupation); the next `update_weights` resumes the weights' memory before loading the new
    ones, and `wake` resumes the rest. The engine must be built with enable_memory_saver (engine_options does
    that from sleep_mode; it needs the torch_memory_saver package, a dependency of sglang).

    Generation: `generate` (a batch, one completion per prompt) and the streaming calls GenerationClient uses
    (`stream_add`, `stream_pending`, `stream_step`): requests run as tasks on the engine's event loop, which
    advances during `stream_step`, while the engine's scheduler process decodes continuously.
    """

    STEP_SECONDS = 0.02  # stream_step returns after this long when nothing finished (weight updates go between)
    POLL_SECONDS = 1.0  # how often a waiting call checks that the engine's scheduler is still alive

    def __init__(
        self,
        model: str,
        stop_ids: tuple[int, ...],
        *,
        gpu_memory_utilization: float = 0.3,
        max_model_len: int = 8192,
        enforce_eager: bool = False,
        seed: int = 0,
        sleep_mode: bool = False,
        prefix_caching: bool = True,
        max_num_seqs: int = 0,
        fp32_head: bool = False,
        ssm_state_dtype: str = "float32",
        kv_cache_dtype: str = "",
        quantization: str = "",
        routing_replay: bool = False,
        engine_kwargs: dict | None = None,
        bucket_mb: int = 512,
        engine=None,
    ):
        """`engine`: the engine calls (InProcessEngine); built from the other arguments when None (tests pass a
        scripted one). `engine_kwargs`: rollout.sglang_args. Raises ValueError for arguments the installed SGLang
        does not have (routing replay included: the trainer then trains without it under routing_replay auto)."""
        self.stop_ids = tuple(int(t) for t in stop_ids)
        self.sleep_mode = sleep_mode
        self.max_model_len = max_model_len
        self.bucket_bytes = max(1, int(bucket_mb)) * 2**20
        self.routing = None
        if routing_replay:
            self._routing_layout(model)
        if engine is None:
            options = engine_options(
                model,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                seed=seed,
                sleep_mode=sleep_mode,
                prefix_caching=prefix_caching,
                max_num_seqs=max_num_seqs,
                fp32_head=fp32_head,
                ssm_state_dtype=ssm_state_dtype,
                kv_cache_dtype=kv_cache_dtype,
                quantization=quantization,
                routing_replay=routing_replay,
                extra=engine_kwargs,
            )
            if torch.cuda.is_available() and "base_gpu_id" not in options:
                options["base_gpu_id"] = torch.cuda.current_device()  # under torchrun: this rank's GPU
            engine = InProcessEngine(options)
        self.calls = engine
        self.memory_tags = list(getattr(engine, "memory_tags", ["kv_cache", "weights"]))
        self.lock = threading.RLock()  # one call on the engine's event loop at a time
        self.version = 0  # -1 while asleep: the weights are gone until the next update
        self.asleep = False  # KV cache (and CUDA graphs) released
        self.weights_released = False
        self.prompt_tokens = self.cached_tokens = 0  # prefill accounting (engine/prefix_hit)
        self.streams: dict[str, tuple[asyncio.Task, int, dict]] = {}  # request id -> (task, started version, info)
        self.adapters: dict[str, str] = {}  # LoRA adapters by name -> SGLang's name of the loaded version

    # ------------------------------------------------------------- weights

    def _alive(self) -> None:
        check = getattr(self.calls, "check", None)
        if check is not None:
            check()

    async def _all(self, tasks: list) -> None:
        """Wait for `tasks`, or for the first failure; raises when the engine's scheduler died meanwhile."""
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, timeout=self.POLL_SECONDS, return_when=asyncio.FIRST_EXCEPTION)
            self._alive()
            if any(not task.cancelled() and task.exception() is not None for task in done):
                return

    def update_weights(self, named_tensors, version: int) -> None:
        with self.lock:
            # The requests in flight finish first, under the weights that started them. SGLang's model-update lock
            # does this for one call; an update is several (one per bucket), and a request that had not started
            # could otherwise run between two of them, on weights of two versions.
            inflight = [task for task, _, _ in self.streams.values() if not task.done()]
            if inflight:
                self.calls.run(self._all(inflight))
            if self.weights_released:
                torch.cuda.empty_cache()  # the trainer's cached blocks back to the driver for the engine to map
                self.calls.resume(["weights"])
                self.weights_released = False
            sent = 0
            for bucket, last in _with_last(weight_buckets(named_tensors, self.bucket_bytes)):
                _cuda_sync(bucket)
                ok, message = self.calls.update_weights_from_tensor(bucket, flush_cache=last)
                _ipc_collect(bucket)
                if not ok:
                    raise RuntimeError(f"SGLang refused the weight update (bucket {sent}): {message}")
                sent += 1
            self.version = version

    def sleep(self) -> None:
        if not self.sleep_mode or self.asleep:
            return
        with self.lock:
            if any(not task.done() for task, _, _ in self.streams.values()):
                raise RuntimeError("the SGLang engine cannot sleep with requests in flight")
            self.calls.release(self.memory_tags)
            self.asleep, self.weights_released, self.version = True, "weights" in self.memory_tags, -1

    def wake(self) -> None:
        if not self.asleep:
            return
        if self.version < 0:
            raise RuntimeError("the SGLang engine dropped its weights while asleep: update_weights before wake")
        with self.lock:
            torch.cuda.empty_cache()
            self.calls.resume([tag for tag in self.memory_tags if tag != "weights"])
            self.asleep = False

    def load_adapter(self, name: str, path: str, version: int) -> None:
        """Make the LoRA adapter at `path` (PEFT layout) what requests for `name` use from now on; each version is
        an adapter of its own to SGLang (loaded under name-vN), and the version before it is unloaded once its
        requests are done. The engine needs enable_lora, max_lora_rank and lora_target_modules
        (rollout.sglang_args / the pool worker's --lora-rank). Not run on a GPU yet."""
        with self.lock:
            loaded = f"{name}-v{version}"
            ok, message = self.calls.load_lora_adapter(loaded, path)
            if not ok:
                raise RuntimeError(f"SGLang could not load the adapter {name!r} from {path}: {message}")
            previous = self.adapters.get(name)
            self.adapters[name] = loaded
            if previous and previous != loaded:
                self.calls.unload_lora_adapter(previous)

    def _lora(self, adapter: str) -> str | None:
        if not adapter or adapter == "base":
            return None
        if adapter not in self.adapters:
            raise KeyError(
                f"the adapter {adapter!r} was not loaded into this engine (loaded: {sorted(self.adapters)}): "
                "adapters reach an engine with the trainer's weight updates (rollout.push native)."
            )
        return self.adapters[adapter]

    # ------------------------------------------------------------- generation

    def _request(self, prompt, budget: int, sampling: Sampling, adapter: str = "") -> dict:
        return build_request(
            prompt,
            budget,
            sampling,
            self.stop_ids,
            self.max_model_len,
            routing=self.routing is not None,
            lora=self._lora(adapter),
        )

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = "") -> list[Rollout]:
        """One completion per prompt. Identical greedy requests are generated once (greedy decoding is
        deterministic); sampled ones each run as their own request, and the radix cache shares their prefill."""
        self._no_images(images)
        per_prompt = samplings(temperature, len(prompts))
        requests: list[dict] = []
        slots: list[int] = []
        greedy: dict[tuple, int] = {}
        for prompt, budget, sampling in zip(prompts, max_new_tokens, per_prompt):
            key = (tuple(prompt), budget, sampling)
            if sampling.temperature <= 0 and key in greedy:
                slots.append(greedy[key])
                continue
            if sampling.temperature <= 0:
                greedy[key] = len(requests)
            slots.append(len(requests))
            requests.append(self._request(prompt, budget, sampling, adapter))

        async def run_all():
            tasks = [asyncio.ensure_future(self.calls.agenerate(r)) for r in requests]
            try:
                await self._all(tasks)
                for task in tasks:  # the first failure, not an unfinished task's InvalidStateError
                    if task.done() and not task.cancelled() and task.exception() is not None:
                        raise task.exception()
                return [task.result() for task in tasks]
            finally:
                for task in tasks:
                    task.cancel()

        with self.lock:
            outs = self.calls.run(run_all()) if requests else []
            version = self.version
            done = [self._rollout(out, request, version) for out, request in zip(outs, requests)]
        return [done[slot] for slot in slots]

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
        self._no_images(images)
        (sampling,) = samplings(temperature, 1)
        copies = 1 if sampling.temperature <= 0 else n  # greedy: one sample, shared by the duplicates
        request = self._request(prompt, max_new_tokens, sampling, adapter)

        async def run():
            outs = await asyncio.gather(*(self.calls.agenerate(request) for _ in range(copies)))
            return outs, self.version  # the version that finished it (updates wait for requests in flight)

        with self.lock:
            task = self.calls.loop.create_task(run())
            self.streams[request_id] = (task, self.version, {"request": request, "n": n})

    def stream_pending(self) -> bool:
        return any(not task.done() for task, _, _ in self.streams.values())

    def stream_step(self) -> list[tuple[str, list[Rollout]]]:
        """Advance the engine's event loop until a request finishes (or STEP_SECONDS pass): (request id, its n
        rollouts) for every request that finished."""
        with self.lock:
            self._alive()
            pending = [task for task, _, _ in self.streams.values() if not task.done()]
            if pending:
                self.calls.run(asyncio.wait(pending, timeout=self.STEP_SECONDS, return_when=asyncio.FIRST_COMPLETED))
            finished = []
            try:
                for request_id, (task, started, info) in list(self.streams.items()):
                    if not task.done():
                        continue
                    del self.streams[request_id]
                    outs, version = task.result()
                    rollouts = [
                        self._rollout(out, info["request"], version, started if started != version else None)
                        for out in outs
                    ]
                    if len(rollouts) < info["n"]:
                        rollouts = rollouts * info["n"]
                    finished.append((request_id, rollouts[: info["n"]]))
            except BaseException:
                # GenerationClient fails every request in flight on an error: they are dropped here as well
                for task, _, _ in self.streams.values():
                    task.cancel()
                self.streams.clear()
                raise
            return finished

    def close(self) -> None:
        """Stop the engine's processes (also done at exit)."""
        with self.lock:
            for task, _, _ in self.streams.values():
                task.cancel()
            self.streams.clear()
            close = getattr(self.calls, "close", None)
            if close is not None:
                close()


# ------------------------------------------------------------------ server


class SGLangServer:
    """An SGLang server (`python -m sglang.launch_server`) launched by the trainer, or a running one at `url`.

    A launched server gets the engine arguments in `options` (engine_options), runs on the trainer's GPU unless
    told otherwise, writes its log to `log_path`, and is terminated with its process group when the trainer exits
    (on Linux also when the trainer dies)."""

    def __init__(
        self,
        model: str,
        url: str = "",
        options: dict | None = None,
        log_path: str = "",
        startup_timeout: float = 900.0,
    ):
        self.model = model
        self.process = None
        self.log_path = log_path
        self._log = None
        if url:
            self.url = url.rstrip("/")
        else:
            port = _free_port()
            self.url = f"http://127.0.0.1:{port}"
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                *server_flags(options or {}),
            ]
            env = engine_env()
            logger.info("Launching %s (log: %s)", " ".join(cmd), log_path or "inherited")
            self._log = open(log_path, "w") if log_path else None
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # its scheduler and detokenizer go with it (killpg)
                preexec_fn=_die_with_parent if sys.platform == "linux" else None,
            )
            atexit.register(self.close)
        self._wait_ready(startup_timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server for {self.model} exited with code {self.process.returncode}"
                    + (f"; see {self.log_path}." if self.log_path else "; see its output.")
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"SGLang server for {self.model} not ready after {timeout:.0f}s at {self.url}")

    def request(self, path: str, payload: dict | None = None, timeout: float = 3600.0, method: str = "POST"):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data if method == "POST" else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"SGLang server {path}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"
            ) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server for {self.model} exited with code {self.process.returncode} during {path}"
                    + (f"; see {self.log_path}" if self.log_path else "")
                ) from None
            raise RuntimeError(f"SGLang server {path} at {self.url}: {e}") from e
        return json.loads(body) if body else {}

    def post(self, path: str, payload: dict | None = None, timeout: float = 3600.0):
        return self.request(path, payload if payload is not None else {}, timeout)

    def info(self) -> dict:
        try:
            return self.request("/server_info", method="GET", timeout=30)
        except RuntimeError:
            return self.request("/get_server_info", method="GET", timeout=30)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None


def _local_address_towards(url: str) -> str:
    """The address of this host's interface that reaches `url`'s host (the NCCL group's rendezvous address)."""
    host = urllib.parse.urlparse(url).hostname or "127.0.0.1"
    if host in ("127.0.0.1", "localhost", "::1"):
        return "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, 9))  # no packet is sent: this only picks the route
        return s.getsockname()[0]


def init_weights_group(master_address: str, master_port: int, world_size: int, group_name: str, rank: int = 0):
    """A torch.distributed NCCL group of its own between the trainer (rank 0) and an SGLang server's ranks,
    outside any default process group: the counterpart of SGLang's init_weights_update_group (the same rendezvous
    and store prefix as its init_custom_process_group, which follows vLLM's and OpenRLHF's)."""
    from datetime import timedelta

    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        rendezvous,
    )

    timeout = timedelta(minutes=10)
    store, rank, world_size = next(
        rendezvous(f"tcp://{master_address}:{master_port}", rank, world_size, timeout=timeout)
    )
    store.set_timeout(timeout)
    store = PrefixStore(group_name, store)
    group, _ = _new_process_group_helper(
        world_size, rank, [], Backend("nccl"), store, group_name=group_name, backend_options=None, timeout=timeout
    )
    _world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    return group


class SGLangServerRollout(_Completions):
    """Rollouts from an SGLang server (rollout.backend sglang_server), its weights pushed after every step.

    weight_sync:
      tensor       CUDA IPC handles of bf16 copies, bucketed (/update_weights_from_tensor): the server must be on
                   the trainer's GPU (a server the trainer launches is).
      distributed  NCCL broadcast from the trainer over a group of its own (/init_weights_update_group, then
                   /update_weights_from_distributed per bucket): a server on another GPU or node. Needs a network
                   path between the two for NCCL; not on the trainer's own GPU (NCCL refuses two ranks on one GPU).
      disk         a safetensors checkpoint in `sync_dir` (a file system the server reads too), loaded with
                   /update_weights_from_disk; the previous version's files are deleted. The slowest, always works.
      auto         tensor for a launched server, distributed for one at rollout.url.

    The server flushes its radix cache with the last bucket of every update. Sleep and wake do nothing: a server
    keeps its memory (start it with a gpu_memory_utilization that leaves the trainer room)."""

    GROUP = "pgs_weights"

    def __init__(
        self,
        server: SGLangServer,
        stop_ids: tuple[int, ...],
        *,
        model: str,
        max_model_len: int,
        weight_sync: str = "auto",
        bucket_mb: int = 512,
        sync_dir: str = "",
        routing_replay: bool = False,
        parallel: int = 4,
    ):
        if weight_sync not in WEIGHT_SYNC_MODES:
            raise ValueError(f"rollout.weight_sync must be one of {WEIGHT_SYNC_MODES}, got {weight_sync!r}")
        self.server = server
        self.stop_ids = tuple(int(t) for t in stop_ids)
        self.model = model
        self.max_model_len = max_model_len
        self.weight_sync = (
            ("tensor" if server.process is not None else "distributed") if weight_sync == "auto" else weight_sync
        )
        self.bucket_bytes = max(1, int(bucket_mb)) * 2**20
        self.sync_dir = Path(sync_dir) if sync_dir else None
        if self.weight_sync == "disk" and self.sync_dir is None:
            raise ValueError("rollout.weight_sync disk needs a directory the server can read (rollout.weight_sync_dir)")
        self.parallel = max(1, parallel)
        self.version = 0
        self.prompt_tokens = self.cached_tokens = 0
        self.group = None  # the NCCL group of weight_sync distributed (created at the first update)
        self.ranks = 0  # the server's tensor-parallel ranks (read at the first tensor update)
        # generate calls and weight updates exclude each other: an update is several requests (one per bucket),
        # and a completion started between two of them would be sampled by weights of two versions
        self.lock = threading.RLock()
        self.disk_versions: list[Path] = []
        self.routing = None
        if routing_replay:
            self._routing_layout(model)

    # ------------------------------------------------------------- weights

    def update_weights(self, named_tensors, version: int) -> None:
        with self.lock:
            self._update_weights(named_tensors, version)

    def _update_weights(self, named_tensors, version: int) -> None:
        mode = self.weight_sync
        if mode == "disk":
            self._update_from_disk(named_tensors, version)
        else:
            if mode == "tensor" and not self.ranks:
                self.ranks = int(self.server.info().get("tp_size", 1) or 1)
            for bucket, last in _with_last(weight_buckets(named_tensors, self.bucket_bytes)):
                _cuda_sync(bucket)
                if mode == "tensor":
                    payload = serialize_named_tensors(bucket)  # one per tensor-parallel rank (each reads its own)
                    reply = self.server.post(
                        "/update_weights_from_tensor",
                        {"serialized_named_tensors": [payload] * self.ranks, "flush_cache": last},
                    )
                    _ipc_collect(bucket)
                    _check_reply(reply, "/update_weights_from_tensor")
                else:
                    self._broadcast(bucket, flush_cache=last)
        self.version = version

    def _broadcast(self, bucket: list[tuple[str, Tensor]], flush_cache: bool) -> None:
        import torch.distributed as dist

        if self.group is None:
            ranks = int(self.server.info().get("tp_size", 1) or 1)
            address, port = _local_address_towards(self.server.url), _free_port()
            with ThreadPoolExecutor(1) as pool:  # both sides block in the rendezvous
                reply = pool.submit(
                    self.server.post,
                    "/init_weights_update_group",
                    {
                        "master_address": address,
                        "master_port": port,
                        "rank_offset": 1,
                        "world_size": ranks + 1,
                        "group_name": self.GROUP,
                        "backend": "nccl",
                    },
                )
                self.group = init_weights_group(address, port, ranks + 1, self.GROUP)
                _check_reply(reply.result(), "/init_weights_update_group")
        payload = {
            "names": [name for name, _ in bucket],
            "dtypes": [str(t.dtype).removeprefix("torch.") for _, t in bucket],
            "shapes": [list(t.shape) for _, t in bucket],
            "group_name": self.GROUP,
            "flush_cache": flush_cache,
        }
        with ThreadPoolExecutor(1) as pool:  # the server receives while this side broadcasts, tensor by tensor
            reply = pool.submit(self.server.post, "/update_weights_from_distributed", payload)
            for _, tensor in bucket:
                dist.broadcast(tensor, src=0, group=self.group)
            _check_reply(reply.result(), "/update_weights_from_distributed")

    def _update_from_disk(self, named_tensors, version: int) -> None:
        from safetensors.torch import save_file

        target = self.sync_dir / f"v{version}"
        target.mkdir(parents=True, exist_ok=True)
        self.disk_versions.append(target)  # removed with the next version, or at close (also when this one fails)
        tensors = {}
        for bucket in weight_buckets(named_tensors, self.bucket_bytes):
            tensors.update({name: t.detach().to("cpu").contiguous() for name, t in bucket})
        save_file(tensors, str(target / "model.safetensors"), metadata={"format": "pt"})
        _copy_model_config(self.model, target)
        _check_reply(
            self.server.post("/update_weights_from_disk", {"model_path": str(target)}), "/update_weights_from_disk"
        )
        while len(self.disk_versions) > 1:  # the server has loaded this one: the older files go, one by one
            self._remove_version(self.disk_versions.pop(0))

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def close(self) -> None:
        """Leave the server as a run found it: the NCCL group of weight_sync distributed destroyed on both sides,
        the last checkpoint of weight_sync disk deleted (file by file). The server itself is its owner's."""
        with self.lock:
            if self.group is not None:
                import torch.distributed as dist

                group, self.group = self.group, None
                with contextlib.suppress(Exception):
                    self.server.post("/destroy_weights_update_group", {"group_name": self.GROUP}, timeout=60)
                with contextlib.suppress(Exception):
                    dist.destroy_process_group(group)
            while self.disk_versions:
                self._remove_version(self.disk_versions.pop())

    @staticmethod
    def _remove_version(directory: Path) -> None:
        for path in sorted(directory.iterdir()) if directory.is_dir() else ():
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            directory.rmdir()

    # ------------------------------------------------------------- generation

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = "") -> list[Rollout]:
        with self.lock:
            return self._generate(prompts, max_new_tokens, temperature, images, adapter)

    def _generate(self, prompts, max_new_tokens, temperature, images=None, adapter: str = "") -> list[Rollout]:
        self._no_images(images)
        if adapter and adapter != "base":
            raise ValueError("rollout.backend sglang_server serves one policy (no named adapters)")
        per_prompt = samplings(temperature, len(prompts))
        requests = [
            build_request(p, b, s, self.stop_ids, self.max_model_len, routing=self.routing is not None)
            for p, b, s in zip(prompts, max_new_tokens, per_prompt)
        ]
        size = max(1, -(-len(requests) // self.parallel))
        parts = [requests[i : i + size] for i in range(0, len(requests), size)]
        if not parts:
            return []
        with ThreadPoolExecutor(len(parts)) as pool:
            outs = [out for part in pool.map(self._send, parts) for out in part]
        version = self.version
        return [self._rollout(out, request, version) for out, request in zip(outs, requests)]

    def _send(self, requests: list[dict]) -> list[dict]:
        """One batched /generate call: the fields as lists, one entry per request."""
        payload: dict[str, Any] = {
            "input_ids": [r["input_ids"] for r in requests],
            "sampling_params": [r["sampling_params"] for r in requests],
            "return_logprob": [r["return_logprob"] for r in requests],
        }
        if any(r.get("return_sampling_mask") for r in requests):
            payload["return_sampling_mask"] = [bool(r.get("return_sampling_mask")) for r in requests]
        if any(r.get("top_logprobs_num") for r in requests):
            payload["top_logprobs_num"] = [int(r.get("top_logprobs_num") or 0) for r in requests]
        if self.routing is not None:
            payload["return_routed_experts"] = True
        out = self.server.post("/generate", payload)
        return out if isinstance(out, list) else [out]


def _check_reply(reply, path: str) -> None:
    ok, message = _ok(reply)
    if not ok:
        raise RuntimeError(f"SGLang server {path} failed: {message}")


def server_option_errors(options: dict) -> list[str]:
    """Engine arguments the SGLang installed here does not have (none checked without SGLang: a server may run
    in another environment, and reports unknown flags in its own log)."""
    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return []
    try:
        check_options(options, (f.name for f in dataclasses.fields(ServerArgs)))
    except ValueError as e:
        return [str(e)]
    return []


def _copy_model_config(model: str, target: Path) -> None:
    """The model's config.json next to a pushed checkpoint (SGLang's loader reads it with the weights)."""
    try:
        from transformers.utils import cached_file

        source = cached_file(model, "config.json")
    except Exception:  # noqa: BLE001 — a local directory without the helper's cache layout
        source = os.path.join(model, "config.json")
    if source and os.path.exists(source):
        shutil.copyfile(source, target / "config.json")
