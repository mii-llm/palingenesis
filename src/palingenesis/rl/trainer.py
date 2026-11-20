"""RL trainer: rollouts in the background, one policy-gradient step per batch.

Per step:
  1. the orchestrator hands over a batch of scored groups, sampled with weights at most
     rollout.max_staleness versions old (pipeline.py: rollouts, rewards, dynamic sampling)
  2. advantages are assigned (group-relative) and the policy scores its trained tokens:
     one forward/backward over token-budgeted micro-batches, the objective of losses.py
  3. clipped AdamW step on fp32 master weights (bf16 compute), then the new weights are
     published to the rollout engine

Launch:
    pgs rl --config configs/rl_math.yaml                                   # one GPU
    torchrun --nproc_per_node 8 -m palingenesis.rl.trainer --config ...    # data parallel, FSDP2
    (multi-node: examples/rl/slurm.sbatch)

Or from Python, with your own rewards and environment:
    from palingenesis.rl import RLConfig, RLTrainer
    RLTrainer(RLConfig.from_yaml("configs/rl_math.yaml"), rewards=[my_reward]).train()
"""

import contextlib
import functools
import inspect
import json
import logging
import math
import os
import random
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from palingenesis import multimodal
from palingenesis.accelerator import check_visible_devices
from palingenesis.kernels import (
    apply_activation_checkpointing,
    apply_liger_kernel,
    match_vllm_gated_norm,
    model_type_of,
)
from palingenesis.logging import Tracker
from palingenesis.logits import final_hidden_states, output_head, verify_output_head
from palingenesis.lora import set_adapter
from palingenesis.memory import trim_host_memory
from palingenesis.moe import RouterBalancer, config_is_moe
from palingenesis.multimodal import VisionProcessor
from palingenesis.opd.orchestrator import Orchestrator, PublishedWeights
from palingenesis.opd.rollout import (
    HFRollout,
    VLLMColocateRollout,
    VLLMServer,
    VLLMServerRollout,
    checkpoint_named_parameters,
    precision_options,
    serve_flags,
)
from palingenesis.opd.teachers import end_of_turn_id, load_causal_lm, right_pad
from palingenesis.opd.trainer import TRAINER_STATE_FILE, checkpoint_steps, prunable_checkpoints, student_stop_ids
from palingenesis.rl.adapters import PolicyAdapters, is_adapter_checkpoint
from palingenesis.rl.chat import ChatFormat
from palingenesis.rl.config import RLConfig, RLConfigError, max_num_seqs
from palingenesis.rl.data import PromptSampler, check_row_limits, load_rows, prompt_messages, split_rows
from palingenesis.rl.env import EnvPool, ToolEnv, row_tools
from palingenesis.rl.formats import convert
from palingenesis.rl.losses import (
    LowPrecisionWeight,
    assign_advantages,
    mismatch_by_bucket,
    mismatch_metrics,
    observation_weights,
    policy_loss,
    target_logprobs,
    token_weights,
)
from palingenesis.rl.parallel import MasterWeights, Parallel, mean_stats
from palingenesis.rl.pipeline import RLBatch, RLPipeline
from palingenesis.rl.rewards import code_reward, load_object, resolve_rewards
from palingenesis.rl.sandbox import make_sandbox
from palingenesis.rl.trajectory import Trajectory
from palingenesis.rollouts import DROPPED_METRIC, RolloutSink, rl_record
from palingenesis.seco import use_chunk_attention

logger = logging.getLogger(__name__)

# Step 0 trains on rollouts of the initial weights scored by the same weights, so its policy/mismatch_k3 is the rollout
# engine's and the trainer's numerics alone (logged as policy/mismatch_k3_step0). bf16 rounding differences stay well
# below this; above it the two compute different distributions (weights not loaded, a chat template, stop token or
# temperature difference, FP8 rollouts, kernels that disagree), and the importance weights correct a bug, not rounding.
MISMATCH_K3_STEP0_WARN = 1e-2
REPLICA_CHECK_EVERY = 50  # steps between checks that the replicas of a replicated base hold the same adapters

# Statistics summed over data-parallel ranks (the rest are averaged)
_SUMMED = {
    "rollout_tokens",
    "rollout_tok_s",
    "trained_tokens",
    "trained_sequences",
    "observation_tokens",
    "groups/launched",
    "groups/kept",
    "groups/carried",
    "groups/verifier_failed",
    "prompts_retired",
    # the prompt sampler's cumulative counters: each rank's shard is disjoint, so the run's totals are sums
    "sampler/rows_tried",
    "sampler/rows_ever_solved",
    "sampler/rows_failed_once",
    "sampler/rows_solved_after_fail",
    "sampler/revisits_pending",
    "sampler/revisits_served",
    "sampler/cooldown_overrides",
    DROPPED_METRIC,
}


class RLTrainer:
    """Reinforcement learning of a causal LM against rewards, single- or multi-turn.

    rewards        callables (or (callable, weight) pairs, or {name: callable}) added to the
                   config's `rewards:`; see palingenesis.rl.rewards for their signature
    env            an environment class (or factory) replacing env.type; see palingenesis.rl.env
    dataset        rows (a list of dicts or a datasets.Dataset) replacing data.dataset
    eval_dataset   held-out rows replacing data.eval_dataset / data.eval_size
    """

    # lora.rank > 0: the policy's low-rank adapters (palingenesis.rl.adapters); None: every weight trains
    adapters: PolicyAdapters | None = None
    lr_scales: tuple[float, ...] = ()  # per optimizer group, with adapters (LoRA+); empty: one rate for all
    # scenario.type: multi-role episodes (palingenesis.rl.scenarios); None: single-agent RL
    scenario = None
    role_specs: dict = {}
    servers: dict = {}  # further vLLM servers: one per adapter with its own engine, one for the frozen base

    def __init__(
        self,
        config: RLConfig,
        rewards: list | dict | None = None,
        env: Callable | None = None,
        dataset: Any = None,
        eval_dataset: Any = None,
    ):
        if dataset is not None and not config.data.dataset:
            config.data.dataset = "<python>"
        for warning in config.validate(python_rewards=bool(rewards), python_env=env is not None):
            logger.warning(warning)
        self.config = config
        m, r, t = config.model, config.rollout, config.train
        check_visible_devices()
        self.parallel = Parallel(
            t.fsdp, t.cpu_offload, t.reshard_after_forward, replicated=t.replicate_base and bool(config.lora.rank)
        )
        rank, world = self.parallel.rank, self.parallel.world
        self.device = self.parallel.device
        self.rng = random.Random(t.seed + rank)
        torch.manual_seed(t.seed + rank)
        os.makedirs(t.output_dir, exist_ok=True)

        # Tokenizer, data, rewards and environment first: mistakes surface before any weights load.
        self.tok = AutoTokenizer.from_pretrained(m.policy)
        # a multimodal policy: the tokenizer carries its image processor (palingenesis.multimodal), which the
        # rollout pipeline, the HF engine and the train step all expand and position images with
        self.vision = None
        if m.multimodal:
            self.vision = VisionProcessor.from_pretrained(m.policy, self.tok, m.min_pixels, m.max_pixels)
            multimodal.attach(self.tok, self.vision)
        if config.loss.observation_weight > 0 and not getattr(self.tok, "is_fast", False):
            raise RLConfigError(
                "loss.observation_weight needs a fast tokenizer: observation tokens are located by their character "
                "offsets in the rendered tool results."
            )
        kwargs = m.chat_template_kwargs
        self.stop_ids = student_stop_ids(self.tok, m.policy, m.stop_tokens, kwargs)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.chat = ChatFormat(self.tok, end_of_turn_id(self.tok, kwargs), kwargs, _think_tags(self.tok, kwargs))
        self.rewards = resolve_rewards(config.rewards, rewards)
        self.env_pool = self._env_pool(env)
        engine_keys: list[str] = []
        if config.scenario.type:  # multi-role episodes: the scenario's verifier rewards, its roles' models
            from palingenesis.rl.scenarios.base import build_scenario
            from palingenesis.rl.scenarios.config import adapter_names
            from palingenesis.rl.scenarios.pipeline import BASE, resolve_roles

            self.scenario = build_scenario(config.scenario.type, config.scenario.args)
            self.role_specs = resolve_roles(config, self.scenario)
            engine_keys = adapter_names(config)
            if any(spec.label == BASE for spec in self.role_specs.values()):
                engine_keys.append(BASE)
        if self.scenario is None and not self.rewards and not (self.env_pool and self.env_pool.has_reward):
            raise RLConfigError(
                "no reward: configure rewards:, pass rewards=[...], or use an environment with get_reward"
            )
        train_rows, eval_rows = self._rows(dataset, eval_dataset)
        eval_rows = _eval_subset(eval_rows, t.eval_samples, config.data.seed)
        # every rank owns a disjoint shard of the prompts (a group never spans ranks)
        self.eval_rows = eval_rows[rank::world]
        d = config.data
        self.sampler = PromptSampler(
            train_rows[rank::world],
            d.seed + rank,
            d.retire_above,
            revisit_after_steps=d.revisit_after_steps,
            revisit_max_failed=d.revisit_max_failed,
            revisit_partial_after_steps=None if d.revisit_partial_after_steps < 0 else d.revisit_partial_after_steps,
            revisit_share=d.revisit_share,
            revisit_max_partial=d.revisit_max_partial,
            revisit_partial_max_share=d.revisit_partial_max_share,
        )
        self.batch_prompts = r.batch_prompts // world
        self.sandbox = make_sandbox(config.sandbox) if self._needs_sandbox() else None
        logger.info(
            "RL: %d training prompts, %d held out%s; rewards %s; env %s",
            len(train_rows),
            len(eval_rows),
            f" (over {world} ranks)" if world > 1 else "",
            ", ".join(f"{x.name} (x{x.weight:g})" for x in self.rewards),
            config.env.type if env is None else getattr(env, "__name__", "custom"),
        )

        # A vLLM server starts before the policy loads: it claims its share of free GPU memory.
        self.server = None
        precision: dict = {}
        if r.backend != "hf" and not (r.backend == "vllm_server" and r.url):  # a running server keeps its own settings
            try:  # the installed vLLM must support what is asked (an fp32 head)
                overrides = r.vllm_args.get("hf_overrides") if r.backend == "vllm" else None
                precision = precision_options(r.fp32_head, r.ssm_state_dtype, overrides)
            except ValueError as e:
                raise RLConfigError(str(e)) from None
        # MoE routing replay: the engine returns the experts it routed every token to (palingenesis.moe)
        self.routing_replay = r.routing_replay != "off" and config_is_moe(m.policy)
        if r.routing_replay == "on" and not self.routing_replay:
            raise RLConfigError(f"rollout.routing_replay: on, but {m.policy} has no MoE layers.")
        if self.routing_replay and r.backend != "hf" and not r.prefix_caching:
            if r.routing_replay == "on":
                raise RLConfigError("rollout.routing_replay with a vLLM backend needs rollout.prefix_caching: true.")
            logger.warning("MoE policy without rollout.prefix_caching: vLLM cannot return routing; no routing replay")
            self.routing_replay = False
        if r.backend == "vllm_server":
            import importlib.util

            if not r.url and importlib.util.find_spec("ray") is None:
                raise RLConfigError(
                    "rollout.backend vllm_server: vLLM's weight transfer into the server it "
                    "launches needs ray. Install it with: pip install 'palingenesis[vllm-server]'"
                )

            def launch(tag: str = "") -> VLLMServer:
                return VLLMServer(
                    m.policy,
                    url=r.url,
                    log_path=os.path.join(t.output_dir, f"vllm{tag}.log"),
                    args=(
                        "--gpu-memory-utilization",
                        str(r.gpu_memory_utilization),
                        "--max-model-len",
                        str(r.max_model_len),
                        "--logprobs-mode",
                        "processed_logprobs",
                        "--weight-transfer-config",
                        '{"backend": "ipc"}',
                        *(["--enforce-eager"] if r.enforce_eager else []),
                        *(["--enable-prefix-caching"] if r.prefix_caching else []),
                        *(["--max-num-seqs", str(r.max_num_seqs)] if r.max_num_seqs else []),
                        *(["--kv-cache-dtype", r.kv_cache_dtype] if r.kv_cache_dtype else []),
                        *(["--quantization", r.quantization] if r.quantization else []),
                        *(["--enable-return-routed-experts"] if self.routing_replay else []),
                        *serve_flags(precision),
                    ),
                )

            self.server = launch()
            # one more server per engine key: each adapter's folded weights are a model of their own to vLLM
            self.servers = {key: launch(f"_{key}") for key in engine_keys}
        model_type = model_type_of(m.policy) if m.match_vllm_norm or m.use_liger_kernel else None
        if m.use_liger_kernel and self.device.startswith("cuda") and model_type:
            apply_liger_kernel(model_type)  # patches classes: before the model loads
        if m.match_vllm_norm:
            match_vllm_gated_norm(model_type)  # also patches a class
        self.resume_path = _resolve_resume(t.resume_from, t.output_dir)
        fsdp = self.parallel.fsdp
        if config.lora.rank:  # low-rank adapters on a frozen base
            self.adapters = PolicyAdapters(config, self.device, rank, world, fsdp)
            if self.scenario is not None:  # roles with an adapter of their own on the same base
                from palingenesis.rl.scenarios.config import adapter_names

                self.adapters.names = adapter_names(config)
                self.adapters.default_trains = any(
                    spec.partner is None and spec.train and not spec.adapter for spec in self.role_specs.values()
                )
        if self.resume_path and (self.adapters is not None) != is_adapter_checkpoint(self.resume_path):
            raise RLConfigError(
                f"{self.resume_path} was saved "
                + (
                    "without adapters, and this run has lora.rank set"
                    if self.adapters is not None
                    else "by a run with LoRA adapters (lora.rank), and this run trains the full weights"
                )
                + ": resume with the configuration the checkpoint was trained with."
            )
        logger.info(
            "Loading policy %s (fp32 master weights, bf16 compute%s) on %s",
            m.policy if self.adapters is not None else self.resume_path or m.policy,  # adapters: always the base
            ", FSDP2" if fsdp else "",
            "cpu, then sharded" if fsdp else self.device,
        )
        if self.vision is not None:  # the checkpoint's own architecture: language model, vision tower, projector
            self.model = multimodal.load_model(self.resume_path or m.policy, dtype=torch.float32)
            multimodal.check_model(self.model, self.vision, "the policy")
            multimodal.set_vision_trainable(self.model, m.train_vision_tower, m.train_projector)
        elif self.adapters is not None:
            # the base is always the policy's own checkpoint: a resumed run adds the checkpoint's adapters to it
            self.model = self.adapters.apply(load_causal_lm(m.policy, torch.float32), self.resume_path)
        else:
            self.model = load_causal_lm(self.resume_path or m.policy, torch.float32)
        if self.resume_path:
            # exports follow the layout of the checkpoint the model was loaded from (checkpoint.save_hf_model,
            # source_layout): after a resume that would be a step_* dir, which checkpoint rotation deletes
            # mid-run (exports then silently lost the multimodal layout). The policy has the same layout and stays.
            self.model.config._name_or_path = m.policy
        # training forwards need no KV cache: building one costs memory, and a checkpointed layer
        # recomputed in the backward would append to it again (different shapes: a hard error)
        for model_config in (self.model.config, getattr(self.model.config, "text_config", None)):
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = False
        # MoE: router balancing over the global batch, routing statistics, routing replay
        self.router_balancer = RouterBalancer.attach(self.model, config.moe, replay=self.routing_replay)
        self.routing_replay = self.routing_replay and self.router_balancer is not None
        if self.adapters is not None and self.router_balancer is not None and self.parallel.main:
            logger.warning(
                "lora on an MoE policy: the routers and any expert weights that are not nn.Linear layers stay frozen, "
                "so the balancing losses reach the routers only through the adapters before them (untested)."
            )
        if self.routing_replay and self.vision is not None:
            # the engine records routing per expanded position, the trajectory stores one placeholder per image
            if r.routing_replay == "on":
                raise RLConfigError(
                    "rollout.routing_replay is not supported with a multimodal policy (model.multimodal)."
                )
            logger.warning("multimodal MoE policy: no routing replay (router balancing stays on)")
            self.routing_replay = False
        if not fsdp:
            self.model.to(self.device)
        checkpointing = m.gradient_checkpointing
        if checkpointing == "auto":
            checkpointing = "full" if t.cpu_offload else "none"
        if checkpointing != "none":
            apply_activation_checkpointing(self.model, mode=checkpointing)
        use_chunk_attention(self.model)
        if t.cpu_offload:
            _check_host_memory(self.model, self.parallel.world)
        if fsdp:  # verify after sharding: the CPU-loaded model cannot run the (Triton) Liger kernels
            self.model = self.parallel.shard(self.model)
        verify_output_head(self.model, output_head(self.model), device=self.device)
        self.parallel.reshard(self.model)
        # one GPU: bf16 parameters, fp32 masters in the optimizer (FSDP2 does this across ranks)
        if self.adapters is not None and not fsdp and self.device.startswith("cuda"):
            self.adapters.lower_base_precision(self.model)  # the frozen base has no master: bf16 only
        # with adapters there are no masters: the adapters stay float32 parameters (megabytes), so nothing trained
        # is rounded in the module (a bf16 DoRA magnitude would make the starting policy differ from the base)
        cuda = not fsdp and self.device.startswith("cuda")
        self.master = MasterWeights(self.model) if cuda and self.adapters is None else None
        if self.adapters is not None:
            self.adapters.start(self.model)  # logs the trainable-parameter count
        self.head = output_head(self.model)
        self.head_weight = LowPrecisionWeight()  # bf16 copy of the head, cast once per optimizer step
        # The policy is the one that samples: vLLM samples (and reports behaviour log-probs) at rollout.temperature,
        # so the trainer scores logits / T: the fp32 logits of the plain linear head are scaled (target_logprobs'
        # scale). Before, T != 1 compared an unscaled policy with a scaled sampler.
        self.inv_temperature = 1.0 / config.rollout.temperature
        if self.inv_temperature != 1.0 and not _plain_linear(self.head):
            raise RLConfigError(
                f"rollout.temperature={config.rollout.temperature} needs a plain linear output head (no bias, no logit "
                "multiplier or soft-capping) so that the trained policy is the sampled one; use temperature 1."
            )
        self.reference = None
        if config.loss.kl_ref:
            from palingenesis.rl.reference import ReferencePolicy

            ref_path = m.policy if config.loss.kl_ref == "start" else config.loss.kl_ref
            if self.adapters is not None and config.loss.kl_ref == "start" and not config.loss.kl_ref_reset:
                from palingenesis.rl.reference import AdapterFreeReference

                # the frozen base is the starting policy: the reference is the model with its adapters off
                self.reference = AdapterFreeReference(
                    self.model,
                    self.adapters,
                    self.device,
                    self.pad_id,
                    t.micro_tokens,
                    temperature=config.rollout.temperature,
                )
                ref_path = self.reference.path
            else:
                self.reference = ReferencePolicy(
                    ref_path, self.device, self.pad_id, t.micro_tokens, temperature=config.rollout.temperature
                )
            logger.info(
                "KL reference: %s (kl_coef %g, reset every %d steps)",
                ref_path,
                config.loss.kl_coef,
                config.loss.kl_ref_reset,
            )

        if r.backend == "hf":
            self.engine = HFRollout(self.model, self.stop_ids, self.pad_id, r.micro_seqs, vision=self.vision)
            self.engine.record_routing = self.routing_replay
        elif r.backend == "vllm":
            extra = {k: v for k, v in (("kv_cache_dtype", r.kv_cache_dtype), ("quantization", r.quantization)) if v}
            if self.vision is not None:  # vLLM's processor must resize as the trainer's does
                extra["mm_processor_kwargs"] = {
                    "min_pixels": self.vision.min_pixels,
                    "max_pixels": self.vision.max_pixels,
                }
            extra.update(r.vllm_args)
            extra.update(precision)

            def colocated(engine_kwargs):
                return VLLMColocateRollout(
                    m.policy,
                    self.stop_ids,
                    r.gpu_memory_utilization,
                    r.max_model_len,
                    r.enforce_eager,
                    t.seed + rank,
                    sleep_mode=r.sleep and r.max_staleness == 0,
                    prefix_caching=r.prefix_caching,
                    max_num_seqs=max(256, max_num_seqs(config) // world),
                    # under torchrun (any world size) vLLM must join the launcher's rendezvous: its own
                    # tcp:// init would wait on the agent store torchrun tells every process to use
                    external_launcher=world > 1 or "TORCHELASTIC_RUN_ID" in os.environ,
                    engine_kwargs=engine_kwargs,
                )

            if self.routing_replay and r.routing_replay == "auto":
                try:  # a vLLM without the option, or a setup it refuses it for: train without replay
                    self.engine = colocated({**extra, "enable_return_routed_experts": True})
                except (TypeError, ValueError, NotImplementedError) as e:
                    logger.warning("vLLM cannot return the routed experts (%s): no routing replay", e)
                    self.routing_replay = False
                    self.engine = colocated(extra)
            else:
                self.engine = colocated(
                    {**extra, "enable_return_routed_experts": True} if self.routing_replay else extra
                )
        else:
            self.engine = VLLMServerRollout(self.server, self.stop_ids)
        self.weights = PublishedWeights(self.model, self._engine_parameters if self.adapters is not None else None)
        if fsdp:  # the trainer thread pushes the weights (collectives); the producer never gathers
            self.weights.sync = lambda engine: 0.0
        pipeline_args = (
            self.tok,
            self.chat,
            self.engine,
            self.weights,
            config,
            self.sampler,
            self.rewards,
            self.env_pool,
            self.sandbox,
            self.stop_ids,
        )
        if self.scenario is None:
            self.pipeline = RLPipeline(*pipeline_args)
        else:
            self.pipeline = self._scenario_pipeline(pipeline_args, engine_keys)
        self.orchestrator = Orchestrator(self.pipeline, lambda: self.batch_prompts, r.temperature, r.max_staleness)
        if self.adapters is not None:
            groups, scales = self.adapters.param_groups(self._trainable(), self._trainable_names())
            self.lr_scales = tuple(scales)
            self.opt = _optimizer(groups, t, self.device)
        else:
            self.opt = _optimizer(self._trainable(), t, self.device, self._trainable_names())
        self.cmr = None
        if config.cmr.enabled:  # common-mode rejection on the parameters the optimizer steps (fp32 masters)
            from palingenesis.cmr import CommonModeRejection

            self.cmr = CommonModeRejection(config.cmr, self.model, self.tok, list(self._trainable()), self.device)
        self.start_step, self.stale_groups, wandb_id = 0, 0, None
        if self.resume_path:
            wandb_id = self._load_state(self.resume_path)
        self.stop_requested = False
        # metrics: wandb, trackio, logging.trackers (rank 0); the run id is saved with each checkpoint
        self.tracker = Tracker.for_run(config.logging, config.to_dict(), t.output_dir, wandb_id, self.parallel.main)
        self.dumps = None  # logging.dump_trajectories: the trajectories' writer (palingenesis.rollouts)
        if config.logging.dump_trajectories:
            self.dumps = RolloutSink(
                Path(t.output_dir) / "rollouts",
                "rl",
                functools.partial(rl_record, content=config.logging.dump_content),
                self.parallel.rank,
                self.parallel.world,
            )

    # ------------------------------------------------------------------ setup

    def _env_pool(self, env: Callable | None) -> EnvPool | None:
        e = self.config.env
        if env is not None:
            factory = env
        elif e.type == "single_turn":
            return None
        elif e.type == "tools":
            functions = [load_object(spec) for spec in e.tools]
            return EnvPool(lambda: ToolEnv(functions), max_concurrent=e.max_concurrent, allowed=e.allowed_tools)
        else:
            factory = load_object(e.type)
        args = dict(e.args)
        if "sandbox" in _parameters(factory) and "sandbox" not in args:
            self._env_wants_sandbox = True
            args["sandbox"] = _LazySandbox(self)
        return EnvPool(factory, args, max_concurrent=e.max_concurrent, allowed=e.allowed_tools)

    def _needs_sandbox(self) -> bool:
        return getattr(self, "_env_wants_sandbox", False) or any(
            r.fn is code_reward or "sandbox" in _parameters(r.fn) for r in self.rewards
        )

    def _rows(self, dataset, eval_dataset) -> tuple[list[dict], list[dict]]:
        d = self.config.data
        rows = convert([dict(x) for x in dataset] if dataset is not None else load_rows(d.dataset, d.split), d.format)
        if eval_dataset is not None:
            train, held = rows, convert([dict(x) for x in eval_dataset], d.format)
        elif d.eval_dataset:
            train, held = rows, convert(load_rows(d.eval_dataset, d.eval_split), d.format)
        else:
            train, held = split_rows(rows, d.eval_size)
        try:  # a row's own max_turns / budgets: a bad value fails here, naming the row, not as rollout errors later
            check_row_limits(rows, d.dataset or "dataset", self.config.completion_budget)
            if held and (eval_dataset is not None or d.eval_dataset):
                check_row_limits(held, d.eval_dataset or "eval_dataset", self.config.completion_budget)
        except ValueError as e:
            raise RLConfigError(str(e)) from None
        limit = d.max_prompt_tokens or (self.config.rollout.max_model_len - self.config.rollout.max_new_tokens)
        dynamic = self.env_pool is not None and (self.env_pool.reusable or self.env_pool.dynamic)
        if (
            not dynamic and self.scenario is None
        ):  # (a scenario writes its prompts itself, per role)  # reset() or tool_schemas() may change the prompt: rollouts check the room instead
            static = self.env_pool.schemas if self.env_pool else None
            kwargs = self.config.model.chat_template_kwargs
            texts = [
                self.tok.apply_chat_template(
                    prompt_messages(row, d.prompt_field, d.system_prompt),
                    tools=static or row_tools(row, d.tools_field),
                    add_generation_prompt=True,
                    tokenize=False,
                    **{**kwargs, **(row.get("chat_template_kwargs") or {})},
                )
                for row in train
            ]
            lengths = [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]  # one batched call
            before = len(train)
            train = [row for row, n in zip(train, lengths) if n + 1 <= limit]
            if len(train) < before:
                logger.info("Dropped %d prompts longer than %d tokens", before - len(train), limit)
        return train, held

    # -------------------------------------------------------------------- step

    def _lr_at(self, step: int) -> float:
        t = self.config.train
        if step < t.warmup_steps:
            return t.learning_rate * (step + 1) / t.warmup_steps
        if t.lr_scheduler == "constant":
            return t.learning_rate
        progress = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps)
        return t.learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def _trainable(self) -> list[torch.nn.Parameter]:
        """The parameters the optimizer steps: the fp32 masters, or the module's own (with adapters: the adapters
        alone, the frozen base holds no optimizer state)."""
        if self.master is not None:
            return self.master.parameters()
        if self.adapters is not None:
            return [p for p in self.model.parameters() if p.requires_grad]
        return list(self.model.parameters())

    def _trainable_names(self) -> list[str]:
        """Names of _trainable(), in order (the fp32 masters mirror the module's trainable parameters)."""
        if self.master is not None or self.adapters is not None:
            return [n for n, p in self.model.named_parameters() if p.requires_grad]
        return [n for n, _ in self.model.named_parameters()]

    def _scenario_pipeline(self, pipeline_args: tuple, engine_keys: list[str]):
        """The pipeline of a scenario run, with one rollout engine per engine key besides the policy's own: the
        trainer's model itself under the hf backend (one adapter per call), a vLLM server per key otherwise (fed
        that adapter's folded weights; the frozen base's server keeps the weights it loaded)."""
        from palingenesis.rl.scenarios.pipeline import BASE, AdapterEngine, ScenarioPipeline, WeightsView

        engines = {}
        if engine_keys and isinstance(self.engine, HFRollout):
            lock = threading.Lock()  # one model: one generate call at a time, whichever adapter
            self.engine = AdapterEngine(self.engine, "", lock, self.adapters)
            pipeline_args = (*pipeline_args[:2], self.engine, *pipeline_args[3:])
            for key in engine_keys:
                engines[key] = (AdapterEngine(self.engine.engine, key, lock, self.adapters), WeightsView(self.weights))
        else:
            for key in engine_keys:
                named = None if key == BASE else functools.partial(self.adapters.engine_parameters, self.model, key)
                weights = WeightsView(self.weights, named, frozen=key == BASE)
                engines[key] = (VLLMServerRollout(self.servers[key], self.stop_ids), weights)
        self.role_shares = {name: spec.share for name, spec in self.role_specs.items()}
        return ScenarioPipeline(*pipeline_args, scenario=self.scenario, roles=self.role_specs, engines=engines)

    def _engine_parameters(self):
        """With adapters: the policy as a rollout engine loads it, the adapters folded into new weight tensors
        (W + scaling * B A from the float32 adapters; the trainer's weights are not written)."""
        return self.adapters.engine_parameters(self.model)

    def _folded_state(self) -> dict[str, torch.Tensor]:
        """With adapters: the model's state with the adapters folded in (a plain model's state)."""
        return self.adapters.folded_state(self.model)

    def _micro_batches(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
        """Micro-batches of at most train.micro_tokens padded tokens, longest trajectories first,
        each with a power-of-two number of rows. Triton kernels (flash-linear-attention's among
        them) autotune per batch size: arbitrary row counts would re-autotune all run long,
        powers of two do it a handful of times. Similar lengths share a micro-batch, so padding
        stays small."""
        budget = self.config.train.micro_tokens
        ordered = sorted(trajectories, key=lambda t: t.context_tokens, reverse=True)  # images expanded
        batches, i = [], 0
        while i < len(ordered):
            width = ordered[i].context_tokens - 1
            rows = min(len(ordered) - i, max(1, budget // width))
            rows = 1 << (rows.bit_length() - 1)
            batches.append(ordered[i : i + rows])
            i += rows
        return batches

    def _train_step(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        """Accumulate the batch's gradient; returns the step's statistics ({} when this rank had
        nothing to train on)."""
        loss_config = self.config.loss
        reduce = self.parallel.sum if self.parallel.world > 1 else None
        if self.scenario is not None:
            # each row's episodes split into the groups a baseline is valid in (same role, same place of the
            # episode); advantages scaled per role, each role's part of the loss set by its share
            from palingenesis.rl.scenarios import credit

            groups = credit.comparison_groups(groups)
            advantage_stats = credit.assign_advantages(
                groups,
                loss_config.advantage_std,
                loss_config.advantage_std_floor,
                reduce=reduce,
                baseline=loss_config.baseline,
                roles=self.scenario.roles,
            )
            weights = credit.token_weights(
                groups,
                loss_config.aggregation,
                self.config.completion_budget,
                self.role_shares,
                sequence_level=loss_config.type == "gspo",
                reduce=reduce,
            )
        else:
            advantage_stats = assign_advantages(
                groups, loss_config.advantage_std, reduce=reduce, baseline=loss_config.baseline
            )
            weights = token_weights(
                groups,
                loss_config.aggregation,
                self.config.completion_budget,
                sequence_level=loss_config.type == "gspo",
                reduce=reduce,
            )
        trajectories = [t for g in groups for t in g if id(t) in weights]
        observed = observation_weights(trajectories, loss_config.observation_weight, reduce)
        if any(t.segments for t in trajectories):
            # a trajectory whose context was rewritten (old images dropped) trains as its segments, each on
            # the context its tokens were sampled from, with the trajectory's advantage and token weight
            views = []
            for t in trajectories:
                for view in t.training_views():
                    weights[id(view)] = weights[id(t)]
                    if id(t) in observed:
                        observed[id(view)] = observed[id(t)]
                    views.append(view)
            trajectories = views
        totals: dict[str, torch.Tensor] = {}
        reference = getattr(self, "reference", None)
        self.ref_lp = reference.logprobs(trajectories) if reference is not None and trajectories else {}
        if self._share_prompts(groups):
            trajectories = self._train_trees(groups, weights, totals, observed)  # the rest: groups too small to share
        micro_adapters: list[str] = []  # with several adapters: the adapter each micro-batch trains
        if self.adapters is not None and self.adapters.names:
            micro_batches: list[list[Trajectory] | None] = []
            for name in sorted({t.adapter for t in trajectories}):
                part = self._micro_batches([t for t in trajectories if t.adapter == name])
                micro_batches += part
                micro_adapters += [name] * len(part)
        else:
            micro_batches = list(self._micro_batches(trajectories))
        # FSDP's gathers are collectives: the same count on every rank (a replicated base has none in its forward)
        count = len(micro_batches) if self.parallel.replicated else self.parallel.max(len(micro_batches))
        micro_batches += [None] * (count - len(micro_batches))
        device = self.device
        current = self._current_version()
        self.model.train()
        balancer = getattr(self, "router_balancer", None)
        for i, micro in enumerate(micro_batches):
            if micro_adapters and i < len(micro_adapters):
                set_adapter(micro_adapters[i])  # this micro-batch's forward and backward use its role's adapter
            self.parallel.gradient_sync(self.model, i == count - 1)
            if micro is None:  # this rank ran out: a zero-weight pass keeps the collectives in step
                if balancer is not None:  # no real token; the count all-reduce stays in step too
                    balancer.begin(torch.zeros(1, 2, dtype=torch.bool), grad_accum=count)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    hidden = final_hidden_states(self.model, torch.full((1, 2), self.pad_id, device=device), None)
                lp, _ = target_logprobs(
                    hidden[0, -1:], self.head, torch.tensor([self.pad_id], device=device), low=self.head_weight
                )
                if balancer is not None:
                    balancer.sync()
                (lp.sum() * 0.0).backward()
                if balancer is not None:
                    balancer.end()
                continue
            sequences = [t.prompt_ids + t.tokens[:-1] for t in micro]
            image_kwargs, index = {}, None
            if any(t.images for t in micro):
                # placeholders expanded with the processor the engine used; the pixels are processed again here
                ids, image_kwargs, index = multimodal.batch_sequences(
                    self.vision, sequences, [t.images for t in micro], self.pad_id, device
                )
                # a row's real positions end at its last token's expanded position
                positions = torch.arange(ids.shape[1], device=device)
                real = torch.stack([positions <= rows[-1] for rows in index])
            else:
                ids, real = right_pad(sequences, self.pad_id, device)
            if balancer is not None:
                replay = _routing_batch(micro, ids.shape[1]) if self.routing_replay else None
                balancer.begin(real, grad_accum=count, replay=replay)
            positions = torch.zeros(ids.shape, dtype=torch.bool)
            targets, behaviour, advantage, weight, seq, forced = [], [], [], [], [], []
            completion_pos, sampled_by = [], []
            # the observation loss: its targets' positions, tokens and weights (none when it is off)
            echo = any(id(t) in observed and t.observation_targets for t in micro)
            echo_positions = torch.zeros(ids.shape, dtype=torch.bool) if echo else None
            echo_targets, echo_weight = [], []
            for row, t in enumerate(micro):
                if echo and id(t) in observed and t.observation_targets:
                    before = [len(t.prompt_ids) + j - 1 for j in t.observation_targets]
                    echo_positions[row, [index[row][p] for p in before] if index else before] = True
                    echo_targets += [t.tokens[j] for j in t.observation_targets]
                    echo_weight += [observed[id(t)]] * len(t.observation_targets)
                sampled = [j for j, trained in enumerate(t.mask) if trained]
                forced += [j in t.forced for j in sampled] if t.forced else [False] * len(sampled)
                before = [len(t.prompt_ids) + j - 1 for j in sampled]  # the token whose output predicts j
                positions[row, [index[row][p] for p in before] if index else before] = True
                completion_pos += sampled
                sampled_by += _sampled_by(t, sampled, current)
                targets += [t.tokens[j] for j in sampled]
                behaviour += [t.logprobs[j] for j in sampled]
                advantage += [t.advantage] * len(sampled)
                weight += [weights[id(t)]] * len(sampled)
                seq += [row] * len(sampled)
            floats = torch.tensor([behaviour, advantage, weight], dtype=torch.float32).to(device, non_blocking=True)
            longs = torch.tensor([targets, seq, completion_pos, sampled_by], dtype=torch.long).to(
                device, non_blocking=True
            )
            seq_len = torch.tensor([float(t.sampled_tokens) for t in micro]).to(device, non_blocking=True)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                # right-padded rows need no mask: padding comes after every position scored
                states = final_hidden_states(self.model, ids, None, **image_kwargs)
                hidden = states[positions.to(device, non_blocking=True)]
                echo_hidden = states[echo_positions.to(device, non_blocking=True)] if echo else None
                del states
            inv_t = getattr(self, "inv_temperature", 1.0)  # applied to the fp32 logits
            lp, entropy = target_logprobs(
                hidden, self.head, longs[0], loss_config.log_entropy, self.head_weight, scale=inv_t
            )
            forced_t = torch.tensor(forced, dtype=torch.bool).to(device, non_blocking=True) if any(forced) else None
            loss, stats = policy_loss(
                lp, floats[0], floats[1], floats[2], longs[1], len(micro), seq_len, loss_config, forced_t
            )
            stats.update(mismatch_by_bucket(lp, floats[0], longs[2], longs[3], forced_t))
            if self.ref_lp:
                loss = self._kl_term(loss, stats, lp, torch.cat([self.ref_lp[id(t)] for t in micro]), floats[2])
            if echo:
                loss = loss + self._observation_term(stats, echo_hidden, echo_targets, echo_weight)
            if balancer is not None:
                balancer.sync()  # the ranks' expert counts, before the routers' backward
            loss.backward()
            if balancer is not None:
                balancer.end()
            _accumulate(totals, stats, loss, entropy)
        if micro_adapters:
            set_adapter("")
        if not totals:
            return {}
        values = dict(zip(totals, torch.stack(list(totals.values())).tolist()))  # the step's one host sync
        tokens = max(values["tokens"], 1.0)
        metrics = {
            "loss": values["loss"],
            "policy/ratio": values["ratio"] / tokens,
            "policy/ratio_max": values["ratio_max"],
            "policy/abs_ratio_dev": values["abs_ratio_dev"] / tokens,
            "policy/mismatch_k3": values["mismatch_k3"] / tokens,
            "policy/clipped": values["clipped"] / tokens,
            "policy/seq_masked": values["seq_masked"] / max(values["sequences"], 1.0),
            "trained_tokens": values["tokens"],
            "trained_sequences": values["sequences"],
            **mismatch_metrics(values),
            **advantage_stats,
        }
        if "entropy" in values:
            metrics["policy/entropy"] = values["entropy"] / tokens
        if "kl_ref" in values:
            metrics["policy/kl_ref"] = values["kl_ref"] / tokens
            metrics["policy/kl_ref_max"] = values["kl_ref_max"]
        if loss_config.observation_weight > 0:
            metrics["observation_tokens"] = values.get("observation_tokens", 0.0)
            if values.get("observation_tokens"):
                metrics["loss/observation"] = values["observation_nll"] / values["observation_tokens"]
        return metrics

    def _observation_term(self, stats: dict, hidden: torch.Tensor, targets: list[int], weight: list[float]):
        """The observation loss of a micro-batch (or branch): its targets' NLL under the model's own logits (not
        scaled by the sampling temperature: this is the likelihood of the environment's text, which the policy
        did not sample), each times its trajectory's weight (observation_weights). Logs the NLL's sum and count."""
        device = hidden.device
        lp, _ = target_logprobs(hidden, self.head, torch.tensor(targets, device=device), low=self.head_weight)
        nll = -lp
        stats["observation_nll"] = nll.detach().sum()
        stats["observation_tokens"] = torch.tensor(float(nll.numel()), device=device)
        return (nll * torch.tensor(weight, dtype=nll.dtype, device=device)).sum()

    def _kl_term(self, loss: torch.Tensor, stats: dict, lp: torch.Tensor, ref_lp: torch.Tensor, weight: torch.Tensor):
        """Log the per-token k3 KL to the reference; with loss.kl_coef > 0 add it with the policy loss's weights."""
        from palingenesis.rl.reference import kl_k3

        kl = kl_k3(lp, ref_lp)
        stats["kl_ref"] = kl.detach().sum()
        stats["kl_ref_max"] = kl.detach().max()
        coef = self.config.loss.kl_coef
        return loss + coef * (kl * weight).sum() if coef > 0 else loss

    def _share_prompts(self, groups: list[list[Trajectory]]) -> bool:
        mode = self.config.train.prompt_sharing
        # routing replay: the recorded routing is per row, the tree computes a group's prompt once
        if (
            mode == "off"
            or self.parallel.fsdp
            or getattr(self, "routing_replay", False)
            or getattr(self, "vision", None) is not None
        ):
            return False
        if mode == "on":
            return True
        lengths = [len(g[0].prompt_ids) for g in groups if g]
        return bool(lengths) and sum(lengths) / len(lengths) >= 1024 and self.config.rollout.group_size >= 4

    def _train_trees(
        self,
        groups: list[list[Trajectory]],
        weights: dict[int, float],
        totals: dict[str, torch.Tensor],
        observed: dict[int, float] | None = None,
    ) -> list[Trajectory]:
        """Each group's trained rollouts as branches off their shared prompt (the trunk), one
        exact tree forward/backward per group: the prompt is computed once, not once per
        rollout. Returns the trajectories left for the plain path (groups with < 2 to train)."""
        from palingenesis.seco_tree import Branch, tree_forward_backward

        loss_config, device = self.config.loss, self.device
        current = self._current_version()
        rest: list[Trajectory] = []
        self.model.train()
        for group in groups:
            trained = [t for t in group if id(t) in weights]
            prompt = trained[0].prompt_ids if trained else []
            # a tree needs one trunk: a group whose members start from different prompts (a scenario's sessions
            # compared across episodes) goes the plain way
            if len(trained) < 2 or len(prompt) < 2 or any(t.prompt_ids != prompt for t in trained[1:]):
                rest += trained
                continue
            if self.adapters is not None and self.adapters.names:
                set_adapter(trained[0].adapter)
            # the trunk ends one token early: branch position j (the prompt's last token, then the
            # completion) predicts completion token j, exactly as in the full sequence
            trunk = torch.tensor([prompt[:-1]], device=device)
            branches = [
                Branch(len(prompt) - 1, torch.tensor([[prompt[-1]] + t.tokens[:-1]], device=device)) for t in trained
            ]

            def loss_fn(i: int, hidden: torch.Tensor, trained=trained) -> torch.Tensor:
                t = trained[i]
                sampled = [j for j, m in enumerate(t.mask) if m]
                echo = bool(observed) and id(t) in observed and bool(t.observation_targets)
                echo_hidden = hidden[0, t.observation_targets] if echo else None  # branch position j predicts j
                rows = hidden[0, sampled]
                targets = torch.tensor([t.tokens[j] for j in sampled], device=device)
                floats = torch.tensor(
                    [[t.logprobs[j] for j in sampled], [t.advantage] * len(sampled), [weights[id(t)]] * len(sampled)],
                    device=device,
                )
                inv_t = getattr(self, "inv_temperature", 1.0)  # applied to the fp32 logits
                lp, entropy = target_logprobs(
                    rows, self.head, targets, loss_config.log_entropy, self.head_weight, scale=inv_t
                )
                seq = torch.zeros(len(sampled), dtype=torch.long, device=device)
                seq_len = torch.tensor([float(len(sampled))], device=device)
                forced = (
                    torch.tensor([j in t.forced for j in sampled], dtype=torch.bool, device=device)
                    if t.forced
                    else None
                )
                loss, stats = policy_loss(lp, floats[0], floats[1], floats[2], seq, 1, seq_len, loss_config, forced)
                where = torch.tensor([sampled, _sampled_by(t, sampled, current)], dtype=torch.long, device=device)
                stats.update(mismatch_by_bucket(lp, floats[0], where[0], where[1], forced))
                if self.ref_lp:
                    loss = self._kl_term(loss, stats, lp, self.ref_lp[id(t)], floats[2])
                if echo:
                    targets = [t.tokens[j] for j in t.observation_targets]
                    loss = loss + self._observation_term(stats, echo_hidden, targets, [observed[id(t)]] * len(targets))
                _accumulate(totals, stats, loss, entropy)
                return loss

            balancer = getattr(self, "router_balancer", None)
            if balancer is not None:  # averaged over the group's tokens, loads from the trunk
                tree_tokens = trunk.shape[1] + sum(b.input_ids.shape[1] for b in branches)
                balancer.begin(None, grad_accum=len(groups), tokens=tree_tokens)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                tree_forward_backward(
                    self.model,
                    trunk,
                    branches,
                    loss_fn,
                    chunk_size=max(len(prompt), 1),
                    branch_tokens=self.config.train.micro_tokens,
                )
            if balancer is not None:
                balancer.end()
        return rest

    def _current_version(self) -> int:
        """The policy version being trained: rollouts sampled by it differ from the trainer by numerics alone."""
        weights = getattr(self, "weights", None)
        return weights.version if weights is not None else 0

    def _push_weights(self, version: int) -> float:
        """FSDP: gather the new weights (one parameter at a time, on every rank) and load them into
        this rank's engine, on the trainer thread. Seconds spent."""
        start = time.perf_counter()
        if isinstance(self.engine, VLLMColocateRollout):
            # no batch is being collected and the engine does not step: partial rollouts (max_staleness > 0) keep
            # decoding between batches, and continue with the new weights
            with self.pipeline.lock, self.pipeline.client.paused():
                full = self.parallel.full_parameters(self.model)
                if self.adapters is not None:  # gathered on every rank (collectives), folded as they arrive
                    full = self.adapters.fold_stream(full, measure=True)
                named = checkpoint_named_parameters(self.model, full)
                self.engine.update_weights(named, version)
        else:
            self.engine.update_weights((), version)
        return time.perf_counter() - start

    # ------------------------------------------------------------------- train

    def train(self) -> None:
        config = self.config
        t = config.train
        failed = True  # until the run ends normally (a stop request included)
        if self.start_step >= t.steps:
            logger.warning("the checkpoint is at step %d of %d: nothing to train", self.start_step, t.steps)
        if self.parallel.fsdp and self.engine.version < self.weights.version:
            self._push_weights(self.weights.version)  # resumed: the engine holds the initial weights
        previous = {sig: signal.signal(sig, self._request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        self.orchestrator.start(end=t.steps)
        try:
            if t.eval_every and self.start_step == 0:
                self._log("eval", self.evaluate(), 0)
            start_time = time.time()
            t_loss = self.config.loss
            for step in range(self.start_step, t.steps):
                t0 = time.perf_counter()
                batch: RLBatch = self.orchestrator.next(self.weights.version)
                waited = time.perf_counter() - t0
                groups = self._fresh(batch.samples)
                lr = self._lr_at(step)
                for i, group in enumerate(self.opt.param_groups):
                    group["lr"] = lr * self.lr_scales[i] if self.lr_scales else lr
                t1 = time.perf_counter()
                if self.cmr is not None:  # its own forward / backward on the twins; grads cleared before the task's
                    self.cmr.pre_step()
                metrics = self._train_step(groups)
                if self.dumps is not None and batch.dump is not None:  # advantages assigned: written off-thread
                    self.dumps.submit(step + 1, batch.dump, batch.timeline)
                trained = self.parallel.sum([metrics.get("trained_sequences", 0.0)])[0]
                if self.parallel.replicated:  # the batch's gradient: the ranks' summed, the same on every replica
                    self.parallel.sum_gradients(self._trainable())
                with self.weights.lock:
                    if self.adapters is not None and self.adapters.names:
                        # each adapter clipped by itself: one role's large gradient does not shrink another's
                        norms = [
                            torch.nn.utils.clip_grad_norm_(group["params"], t.max_grad_norm)
                            for group in self.opt.param_groups
                        ]
                        grad_norm = torch.stack([n.float() for n in norms]).norm()
                        metrics.update(
                            {f"grad_norm/{name}": float(n) for name, n in zip(self.adapters.group_names, norms)}
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable(), t.max_grad_norm)
                    grad_norm = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
                    if trained:  # every rank steps (or none does): the parameters are sharded
                        self.opt.step()
                        if self.cmr is not None:  # correct the masters before they are published
                            metrics.update(self.cmr.post_step())
                        if self.master is not None:
                            self.master.publish()
                    self.opt.zero_grad(set_to_none=True)
                    if self.router_balancer is not None:
                        metrics.update(self.router_balancer.metrics())
                        self.router_balancer.step()
                    if self.reference is not None and t_loss.kl_ref_reset and (step + 1) % t_loss.kl_ref_reset == 0:
                        self.reference.reset_to(self.model, self._folded_state() if self.adapters is not None else None)
                        logger.info("KL reference reset to the policy at step %d", step + 1)
                    if self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                train_time = time.perf_counter() - t1
                push = self._push_weights(self.weights.version + 1) if self.parallel.fsdp else 0.0
                staleness = self.weights.version - batch.version
                self.weights.publish()
                metrics.update(batch.stats)
                metrics.update(
                    {
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "staleness": staleness,
                        "stale_groups_dropped": self.stale_groups,
                        "time/train": train_time,
                        "time/push": push,
                        "time/wait_batch": waited,
                        "time/step": time.perf_counter() - t0,
                        "prompts_retired": len(self.sampler.retired),
                        "epoch": self.sampler.epoch,
                        "elapsed": time.time() - start_time,
                        **self._memory(),
                        **trim_host_memory(),  # after the step: frees what the batch held, then reads the RSS
                    }
                )
                if self.dumps is not None:
                    metrics[DROPPED_METRIC] = self.dumps.dropped
                if self.adapters is not None:
                    metrics.update(self.adapters.metrics(self.model, step + 1))
                if self.parallel.replicated and (step + 1) % REPLICA_CHECK_EVERY == 0:
                    with self.weights.lock:  # the replicas must stay one policy: checked, and repaired from rank 0
                        metrics["replica/drift"] = self.parallel.replica_drift(self._trainable())
                metrics = self._combine(metrics)
                if step == 0:
                    self._parity_check(metrics)
                if not trained:
                    logger.warning(
                        "step %d: no informative group (every rollout of every prompt got the same "
                        "reward); raise rollout.max_refill or check the rewards",
                        step,
                    )
                if step % config.logging.log_every == 0:
                    self._log("step", metrics, step + 1)
                if self._should_stop():
                    logger.warning("stop requested: saving a resumable checkpoint at step %d and exiting", step + 1)
                    self.save(f"step_{step + 1}", step + 1)
                    failed = False
                    return
                if t.eval_every and (step + 1) % t.eval_every == 0 and step + 1 < t.steps:
                    self._log("eval", self.evaluate(), step + 1)
                if t.save_steps and (step + 1) % t.save_steps == 0:
                    self.save(f"step_{step + 1}", step + 1)
            self._log("eval", self.evaluate(), t.steps)
            if t.save_final:
                self.save("final")
            failed = False
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.orchestrator.stop()
            self.pipeline.close()
            if self.dumps is not None:
                self.dumps.close()
            self.tracker.finish(1 if failed else 0)
            if self.server is not None:
                self.server.close()
            for server in self.servers.values():
                server.close()
            self.parallel.close()

    def _parity_check(self, metrics: dict[str, float]) -> None:
        """Step 0 of a fresh run: log the sampler/trainer mismatch before any update, and warn when it is large."""
        if "policy/mismatch_k3" not in metrics:
            return
        k3 = metrics["policy/mismatch_k3_step0"] = metrics["policy/mismatch_k3"]
        if k3 > MISMATCH_K3_STEP0_WARN and self.parallel.main:
            logger.warning(
                "step 0: sampler/trainer mismatch k3 %.3g (> %g) before any update: the rollout engine and the "
                "trainer compute different distributions. Check that the engine loaded the policy's weights, the "
                "chat template and stop tokens, FP8 rollouts (rollout.kv_cache_dtype / quantization) and the "
                "precision options (rollout.fp32_head, rollout.ssm_state_dtype, model.match_vllm_norm).",
                k3,
                MISMATCH_K3_STEP0_WARN,
            )

    def _memory(self) -> dict[str, float]:
        """Peak GPU memory of the trainer since the last step (the engine's own is not counted)."""
        if not self.device.startswith("cuda"):
            return {}
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        return {"memory/peak_gb": peak}

    def _request_stop(self, signum, frame) -> None:
        """SIGUSR1 (Slurm's warning before the time limit) or SIGTERM: checkpoint after this step."""
        self.stop_requested = True

    def _should_stop(self) -> bool:
        return self.parallel.max(int(self.stop_requested)) > 0  # every rank stops at the same step

    def _fresh(self, groups: list[list[Trajectory]]) -> list[list[Trajectory]]:
        """Groups within rollout.max_staleness of the current policy (carried groups can be older)."""
        limit = self.config.rollout.max_staleness
        fresh = [g for g in groups if self.weights.version - min(t.version for t in g) <= limit]
        self.stale_groups += len(groups) - len(fresh)
        return fresh

    def _combine(self, metrics: dict[str, float]) -> dict[str, float]:
        """One step's statistics over all ranks: summed counts, averaged rates."""
        if self.parallel.world == 1:
            return metrics
        per_rank = self.parallel.gather(metrics)
        combined = mean_stats(per_rank)
        for key in _SUMMED & combined.keys():
            combined[key] = sum(s.get(key, 0.0) for s in per_rank)
        for key in combined:
            if key.endswith("_max"):  # a maximum over the batch (policy/ratio_max, policy/kl_ref_max), not a rate
                combined[key] = max(s[key] for s in per_rank if key in s)
        return combined

    def evaluate(self) -> dict[str, float]:
        """Mean rewards on the held-out prompts (every rank its shard), one rollout each at
        train.eval_temperature."""
        self.model.eval()
        try:
            stats = self.pipeline.evaluate(self.eval_rows, self.config.train.eval_temperature) if self.eval_rows else {}
        finally:
            self.model.train()
        if self.parallel.world == 1:
            return stats
        gathered = self.parallel.gather((stats, len(self.eval_rows)))
        return mean_stats([s for s, _ in gathered], [float(n) for _, n in gathered])

    # ------------------------------------------------------------- bookkeeping

    def _log(self, kind: str, metrics: dict[str, float], step: int) -> None:
        if not self.parallel.main or not metrics:
            return
        logger.info("%s %d | %s", kind, step, " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items())))
        self.tracker.log({(f"eval/{k}" if kind == "eval" else k): v for k, v in metrics.items()}, step)

    def save(self, name: str, step: int | None = None) -> None:
        """The policy in Hugging Face format; with `step`, also the state to resume from: the
        optimizer (sharded DCP under FSDP), each rank's prompt sampler and random states.
        trainer_state.pt is written last, so a checkpoint with it is complete.

        The save's transient GPU memory (gathered tensors, state-dict copies) goes back to the
        device before each barrier: an awake rollout engine (rollout.max_staleness > 0) may hold
        the rest of the GPU, and NCCL allocates for a barrier. On one process there is no
        barrier at all, so trainer_state.pt follows rank0.pt directly."""
        from palingenesis.checkpoint import _save_gathered_or_local, _set_config_dtype
        from palingenesis.distributed import release_cuda_memory

        path = Path(self.config.train.output_dir) / name
        if self.parallel.main:
            logger.info("Saving checkpoint -> %s", path)
        # The rollout engine idle (no generation with the trained model, no weight copy), in the
        # pipeline's lock order: pipeline, then weights.
        with self.pipeline.lock, self.weights.lock:
            self.parallel.reshard(self.model)
            with self.master.fp32() if self.master is not None else contextlib.nullcontext():
                if self.adapters is None:
                    _save_gathered_or_local(self.model, self.tok, path, self.parallel.fsdp)
                elif self.parallel.replicated and not self.parallel.main:
                    pass  # every rank holds the same policy: rank 0 writes it
                else:  # the folded model in float32 (a plain checkpoint) and <path>/adapter
                    export = self.adapters.export(path)
                    _save_gathered_or_local(self.model, self.tok, path, self.parallel.fsdp, export)
                    if self.parallel.main:
                        _set_config_dtype(path / "config.json", torch.float32)  # the live base may be bf16
            if step is not None and self.parallel.fsdp:
                from torch.distributed.checkpoint import save as dcp_save
                from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

                dcp_save({"optimizer": get_optimizer_state_dict(self.model, self.opt)}, checkpoint_id=str(path / "dcp"))
        if step is not None:
            local = {
                "sampler": self.sampler.state(),
                "rng": self.rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else None,
            }
            if not self.parallel.fsdp:
                local["optimizer"] = self.opt.state_dict()
            torch.save(local, path / f"rank{self.parallel.rank}.pt")
            del local
            release_cuda_memory()  # on every rank, before the barrier: no collective in it
            self.parallel.barrier()  # every rank's files written before rank 0 marks the checkpoint complete
        if self.parallel.main:
            with open(path / "rl_config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            if step is not None:
                state = {
                    "step": step,
                    "world": self.parallel.world,
                    "stale_groups": self.stale_groups,
                    "wandb_id": self.tracker.run_id,  # the tracker's run id (key kept for older checkpoints)
                }
                torch.save(state, path / (TRAINER_STATE_FILE + ".tmp"))
                os.replace(path / (TRAINER_STATE_FILE + ".tmp"), path / TRAINER_STATE_FILE)
            for old in prunable_checkpoints(self.config.train.output_dir, self.config.train.keep_checkpoints):
                shutil.rmtree(old, ignore_errors=True)
        self.parallel.barrier()

    def _load_state(self, path: str) -> str | None:
        """Restore the optimizer, samplers and random states (the weights came with the model)."""
        path = Path(path)
        state = torch.load(path / TRAINER_STATE_FILE, map_location="cpu", weights_only=False)
        if state.get("world", 1) != self.parallel.world:
            raise RLConfigError(
                f"{path} was saved by {state.get('world', 1)} rank(s); resume with the same number "
                f"(now {self.parallel.world})"
            )
        local = torch.load(path / f"rank{self.parallel.rank}.pt", map_location="cpu", weights_only=False)
        if self.parallel.fsdp:
            from torch.distributed.checkpoint import load as dcp_load
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

            optimizer = {"optimizer": get_optimizer_state_dict(self.model, self.opt)}
            dcp_load(optimizer, checkpoint_id=str(path / "dcp"))
            set_optimizer_state_dict(self.model, self.opt, optimizer["optimizer"])
        else:
            self.opt.load_state_dict(local["optimizer"])
        self.sampler.load_state(local["sampler"])
        self.rng.setstate(local["rng"])
        torch.set_rng_state(local["torch_rng"])
        if local["cuda_rng"] is not None and self.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(local["cuda_rng"])
        self.start_step = state["step"]
        self.pipeline.batches = state["step"]  # trajectory dumps continue the first leg's step numbers
        self.weights.version = state["step"]  # the engine holds older weights: the first rollout (or push) syncs
        self.stale_groups = state["stale_groups"]
        logger.info("Resumed from %s at step %d", path, self.start_step)
        return state["wandb_id"]


def _plain_linear(head) -> bool:
    """A bias-free nn.Linear head whose logits the model uses as they are (scaling its input scales its logits)."""
    return isinstance(head, torch.nn.Linear) and head.bias is None


def _routing_batch(micro: list[Trajectory], width: int) -> torch.Tensor | None:
    """The rollout engine's expert choices for a micro-batch's rows, [rows, width, MoE layers, top_k] (int64), -1
    where a position has none (padding, positions the engine never forwarded, trajectories without routing); None
    when no row has any."""
    recorded = [t.routing for t in micro]
    shape = next((r.shape[1:] for r in recorded if r is not None and len(r)), None)
    if shape is None:
        return None
    out = torch.full((len(micro), width, *shape), -1, dtype=torch.long)
    for row, r in enumerate(recorded):
        if r is not None and len(r) and r.shape[1:] == shape:
            n = min(len(r), width)
            out[row, :n] = torch.from_numpy(r[:n].astype("int64"))
    return out


def _sampled_by(t: Trajectory, sampled: list[int], current: int) -> list[int]:
    """Per sampled token index: 0 if version `current` (the policy being trained) sampled its whole turn, 1 if an
    older version sampled any of it, -1 when unknown (see Trajectory.token_versions)."""
    versions = t.token_versions()
    return [-1 if versions[j] < 0 else int(versions[j] < current) for j in sampled]


def _accumulate(totals: dict[str, torch.Tensor], stats: dict, loss: torch.Tensor, entropy: torch.Tensor) -> None:
    """Add a micro-batch's (or branch's) statistics to the step's, on the GPU."""
    stats["loss"] = loss.detach()
    if entropy.numel():
        stats["entropy"] = entropy.sum()
    for k, v in stats.items():
        if k not in totals:
            totals[k] = v
        else:
            totals[k] = torch.maximum(totals[k], v) if k.endswith("_max") else totals[k] + v


def _check_host_memory(model: torch.nn.Module, world: int) -> None:
    """Warn when CPU offload will not fit in this node's RAM. The state is 16 bytes per
    parameter (fp32 parameters and gradients, pinned, and AdamW's two moments), split over the
    ranks; pinned staging and load-time copies add more (a 4B policy on one A100 held 83 GB,
    about 21 bytes per parameter, when the kernel killed it), so budget 24."""
    from palingenesis.accelerator import host_available_bytes, is_unified_memory

    if is_unified_memory():
        logger.warning(
            "train.cpu_offload on a GPU that shares the host's RAM (unified memory) saves no memory: the offloaded "
            "state comes out of the same pool. Turn it off."
        )
    available = host_available_bytes()
    if available is None:
        return
    local = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    need = sum(p.numel() for p in model.parameters()) * 24 * local / world
    if need > 0.9 * available:
        logger.warning(
            "train.cpu_offload needs about %.0f GB of host RAM on this node (about 24 bytes per parameter), "
            "%.0f GB is available: the kernel may kill the run at its first optimizer step. Free host "
            "memory (tmpfs files count), spread the ranks over more nodes, or use a smaller model.",
            need / 1e9,
            available / 1e9,
        )


class _LazySandbox:
    """The trainer's sandbox, handed to environment constructors before it exists."""

    def __init__(self, trainer: RLTrainer):
        self._trainer = trainer

    def __getattr__(self, name):
        return getattr(self._trainer.sandbox, name)


def _optimizer(
    params: list[torch.nn.Parameter], t: Any, device: str, names: list[str] | None = None
) -> torch.optim.Optimizer:
    """AdamW over the trained parameters (the fp32 masters on one GPU), with fp32 or 8-bit moments; or Muon on the
    hidden matrices with AdamW on the rest (`names` route the parameters, palingenesis.muon)."""
    eps = t.adam_eps or (1e-15 if t.optimizer in ("adamw", "muon") else 1e-8)
    kwargs = dict(lr=t.learning_rate, betas=(t.adam_beta1, t.adam_beta2), eps=eps, weight_decay=t.weight_decay)
    if t.optimizer == "muon":
        from palingenesis.muon import build_muon

        params = list(params)
        named = (
            list(zip(names, params))
            if names and len(names) == len(params)
            else [(str(i), p) for i, p in enumerate(params)]
        )
        return build_muon(named, t.learning_rate, t.weight_decay, betas=kwargs["betas"], eps=eps, **t.muon_options())
    if t.optimizer == "adamw":
        return torch.optim.AdamW(params, fused=device.startswith("cuda") and not t.cpu_offload, **kwargs)
    import bitsandbytes as bnb

    cls = bnb.optim.PagedAdamW8bit if t.optimizer == "paged_adamw8bit" else bnb.optim.AdamW8bit
    return cls(params, **kwargs)


def _parameters(fn: Callable) -> set[str]:
    try:
        target = fn.__init__ if isinstance(fn, type) else fn
        return set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return set()


def _think_tags(tok, kwargs: dict) -> tuple[str, str]:
    from palingenesis.data import detect_think_tags

    tags = detect_think_tags(lambda m, **kw: tok.apply_chat_template(m, tokenize=False, **{**kwargs, **kw}))
    return tags or ("<think>", "</think>")


def _eval_subset(rows: list[dict], k: int, seed: int) -> list[dict]:
    """At most k held-out rows: a fixed-seed sample in file order (the same rows at every evaluation and on resume), so
    families appended at the end of a dev file are evaluated too (a head slice silently dropped them)."""
    if not k or k >= len(rows):
        return rows
    keep = sorted(random.Random(seed).sample(range(len(rows)), k))
    return [rows[i] for i in keep]


def _resolve_resume(resume_from: str, output_dir: str) -> str | None:
    """None to start fresh; "auto": the newest complete step_* checkpoint in output_dir, if any."""
    if not resume_from:
        return None
    if resume_from == "auto":
        found = checkpoint_steps(output_dir)
        if not found:
            logger.info("resume_from auto: no checkpoint in %s, starting fresh", output_dir)
        return found[-1] if found else None
    if not os.path.exists(os.path.join(resume_from, TRAINER_STATE_FILE)):
        raise RLConfigError(
            f"train.resume_from: {resume_from} is not a complete checkpoint "
            f"(no {TRAINER_STATE_FILE}; `final` holds the model only)"
        )
    return resume_from


def main():
    from palingenesis.logging import setup_logging

    setup_logging(rank=int(os.environ.get("RANK", "0")))
    RLTrainer(RLConfig.from_cli()).train()


if __name__ == "__main__":
    main()
