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
import inspect
import json
import logging
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from palingenesis.kernels import apply_activation_checkpointing, apply_liger_kernel, model_type_of
from palingenesis.logits import final_hidden_states, output_head, verify_output_head
from palingenesis.opd.orchestrator import Orchestrator, PublishedWeights
from palingenesis.opd.rollout import (
    HFRollout,
    VLLMColocateRollout,
    VLLMServer,
    VLLMServerRollout,
    checkpoint_named_parameters,
)
from palingenesis.opd.teachers import end_of_turn_id, load_causal_lm, right_pad
from palingenesis.opd.trainer import TRAINER_STATE_FILE, checkpoint_steps, student_stop_ids
from palingenesis.rl.chat import ChatFormat
from palingenesis.rl.config import RLConfig, RLConfigError, max_num_seqs
from palingenesis.rl.data import PromptSampler, load_rows, prompt_messages, split_rows
from palingenesis.rl.env import EnvPool, ToolEnv, row_tools
from palingenesis.rl.formats import convert
from palingenesis.rl.losses import LowPrecisionWeight, assign_advantages, policy_loss, target_logprobs, token_weights
from palingenesis.rl.parallel import MasterWeights, Parallel, mean_stats
from palingenesis.rl.pipeline import RLBatch, RLPipeline
from palingenesis.rl.rewards import code_reward, load_object, resolve_rewards
from palingenesis.rl.sandbox import make_sandbox
from palingenesis.rl.trajectory import Trajectory
from palingenesis.seco import use_chunk_attention

logger = logging.getLogger(__name__)

# Statistics summed over data-parallel ranks (the rest are averaged)
_SUMMED = {
    "rollout_tokens",
    "rollout_tok_s",
    "trained_tokens",
    "trained_sequences",
    "groups/launched",
    "groups/kept",
    "groups/carried",
    "prompts_retired",
}


class RLTrainer:
    """Reinforcement learning of a causal LM against rewards, single- or multi-turn.

    rewards        callables (or (callable, weight) pairs, or {name: callable}) added to the
                   config's `rewards:`; see palingenesis.rl.rewards for their signature
    env            an environment class (or factory) replacing env.type; see palingenesis.rl.env
    dataset        rows (a list of dicts or a datasets.Dataset) replacing data.dataset
    eval_dataset   held-out rows replacing data.eval_dataset / data.eval_size
    """

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
        self.parallel = Parallel(t.fsdp, t.cpu_offload, t.reshard_after_forward)
        rank, world = self.parallel.rank, self.parallel.world
        self.device = self.parallel.device
        self.rng = random.Random(t.seed + rank)
        torch.manual_seed(t.seed + rank)
        os.makedirs(t.output_dir, exist_ok=True)

        # Tokenizer, data, rewards and environment first: mistakes surface before any weights load.
        self.tok = AutoTokenizer.from_pretrained(m.policy)
        kwargs = m.chat_template_kwargs
        self.stop_ids = student_stop_ids(self.tok, m.policy, m.stop_tokens, kwargs)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.chat = ChatFormat(self.tok, end_of_turn_id(self.tok, kwargs), kwargs, _think_tags(self.tok, kwargs))
        self.rewards = resolve_rewards(config.rewards, rewards)
        self.env_pool = self._env_pool(env)
        if not self.rewards and not (self.env_pool and self.env_pool.has_reward):
            raise RLConfigError(
                "no reward: configure rewards:, pass rewards=[...], or use an environment with get_reward"
            )
        train_rows, eval_rows = self._rows(dataset, eval_dataset)
        eval_rows = eval_rows[: t.eval_samples] if t.eval_samples else eval_rows
        # every rank owns a disjoint shard of the prompts (a group never spans ranks)
        self.eval_rows = eval_rows[rank::world]
        self.sampler = PromptSampler(train_rows[rank::world], config.data.seed + rank, config.data.retire_above)
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
        if r.backend == "vllm_server":
            self.server = VLLMServer(
                m.policy,
                url=r.url,
                log_path=os.path.join(t.output_dir, "vllm.log"),
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
                ),
            )
        if m.use_liger_kernel and self.device.startswith("cuda") and (model_type := model_type_of(m.policy)):
            apply_liger_kernel(model_type)  # patches classes: before the model loads
        self.resume_path = _resolve_resume(t.resume_from, t.output_dir)
        fsdp = self.parallel.fsdp
        logger.info(
            "Loading policy %s (fp32 master weights, bf16 compute%s) on %s",
            self.resume_path or m.policy,
            ", FSDP2" if fsdp else "",
            "cpu, then sharded" if fsdp else self.device,
        )
        self.model = load_causal_lm(self.resume_path or m.policy, torch.float32)
        # training forwards need no KV cache: building one costs memory, and a checkpointed layer
        # recomputed in the backward would append to it again (different shapes: a hard error)
        for model_config in (self.model.config, getattr(self.model.config, "text_config", None)):
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = False
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
        self.master = MasterWeights(self.model) if not fsdp and self.device.startswith("cuda") else None
        self.head = output_head(self.model)
        self.head_weight = LowPrecisionWeight()  # bf16 copy of the head, cast once per optimizer step

        if r.backend == "hf":
            self.engine = HFRollout(self.model, self.stop_ids, self.pad_id, r.micro_seqs)
        elif r.backend == "vllm":
            extra = {k: v for k, v in (("kv_cache_dtype", r.kv_cache_dtype), ("quantization", r.quantization)) if v}
            extra.update(r.vllm_args)
            self.engine = VLLMColocateRollout(
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
                engine_kwargs=extra,
            )
        else:
            self.engine = VLLMServerRollout(self.server, self.stop_ids)
        self.weights = PublishedWeights(self.model)
        if fsdp:  # the trainer thread pushes the weights (collectives); the producer never gathers
            self.weights.sync = lambda engine: 0.0
        self.pipeline = RLPipeline(
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
        self.orchestrator = Orchestrator(self.pipeline, lambda: self.batch_prompts, r.temperature, r.max_staleness)
        self.opt = _optimizer(self._trainable(), t, self.device)
        self.start_step, self.stale_groups, wandb_id = 0, 0, None
        if self.resume_path:
            wandb_id = self._load_state(self.resume_path)
        self.stop_requested = False
        self.wandb, self.wandb_id = None, wandb_id
        if config.logging.use_wandb and self.parallel.main:
            try:
                import wandb

                run = wandb.init(
                    project=config.logging.project,
                    name=config.logging.run_name or None,
                    config=config.to_dict(),
                    dir=t.output_dir,
                    id=wandb_id,
                    resume="allow" if wandb_id else None,
                )
                self.wandb, self.wandb_id = wandb, run.id
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill a training run
                logger.warning("wandb init failed (%s); continuing without it", e)

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
        limit = d.max_prompt_tokens or (self.config.rollout.max_model_len - self.config.rollout.max_new_tokens)
        dynamic = self.env_pool is not None and (self.env_pool.reusable or self.env_pool.dynamic)
        if not dynamic:  # reset() or tool_schemas() may change the prompt: rollouts check the room instead
            static = self.env_pool.schemas if self.env_pool else None
            kwargs = self.config.model.chat_template_kwargs
            texts = [
                self.tok.apply_chat_template(
                    prompt_messages(row, d.prompt_field, d.system_prompt),
                    tools=static or row_tools(row, d.tools_field),
                    add_generation_prompt=True,
                    tokenize=False,
                    **kwargs,
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
        """The parameters the optimizer steps: the fp32 masters, or the module's own."""
        return self.master.parameters() if self.master is not None else list(self.model.parameters())

    def _micro_batches(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
        """Micro-batches of at most train.micro_tokens padded tokens, longest trajectories first,
        each with a power-of-two number of rows. Triton kernels (flash-linear-attention's among
        them) autotune per batch size: arbitrary row counts would re-autotune all run long,
        powers of two do it a handful of times. Similar lengths share a micro-batch, so padding
        stays small."""
        budget = self.config.train.micro_tokens
        ordered = sorted(trajectories, key=lambda t: len(t.prompt_ids) + len(t.tokens), reverse=True)
        batches, i = [], 0
        while i < len(ordered):
            width = len(ordered[i].prompt_ids) + len(ordered[i].tokens) - 1
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
        advantage_stats = assign_advantages(groups, loss_config.advantage_std, reduce=reduce)
        weights = token_weights(
            groups,
            loss_config.aggregation,
            self.config.completion_budget,
            sequence_level=loss_config.type == "gspo",
            reduce=reduce,
        )
        trajectories = [t for g in groups for t in g if id(t) in weights]
        totals: dict[str, torch.Tensor] = {}
        if self._share_prompts(groups):
            trajectories = self._train_trees(groups, weights, totals)  # the rest: groups too small to share
        micro_batches: list[list[Trajectory] | None] = list(self._micro_batches(trajectories))
        count = self.parallel.max(len(micro_batches))  # FSDP's gathers are collectives: same count on every rank
        micro_batches += [None] * (count - len(micro_batches))
        device = self.device
        self.model.train()
        for i, micro in enumerate(micro_batches):
            self.parallel.gradient_sync(self.model, i == count - 1)
            if micro is None:  # this rank ran out: a zero-weight pass keeps the collectives in step
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    hidden = final_hidden_states(self.model, torch.full((1, 2), self.pad_id, device=device), None)
                lp, _ = target_logprobs(hidden[0, -1:], self.head, torch.tensor([self.pad_id], device=device))
                (lp.sum() * 0.0).backward()
                continue
            ids, _ = right_pad([t.prompt_ids + t.tokens[:-1] for t in micro], self.pad_id, device)
            positions = torch.zeros(ids.shape, dtype=torch.bool)
            targets, behaviour, advantage, weight, seq = [], [], [], [], []
            for row, t in enumerate(micro):
                sampled = [j for j, trained in enumerate(t.mask) if trained]
                positions[row, [len(t.prompt_ids) + j - 1 for j in sampled]] = True
                targets += [t.tokens[j] for j in sampled]
                behaviour += [t.logprobs[j] for j in sampled]
                advantage += [t.advantage] * len(sampled)
                weight += [weights[id(t)]] * len(sampled)
                seq += [row] * len(sampled)
            floats = torch.tensor([behaviour, advantage, weight], dtype=torch.float32).to(device, non_blocking=True)
            longs = torch.tensor([targets, seq], dtype=torch.long).to(device, non_blocking=True)
            seq_len = torch.tensor([float(t.sampled_tokens) for t in micro]).to(device, non_blocking=True)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                # right-padded rows need no mask: padding comes after every position scored
                hidden = final_hidden_states(self.model, ids, None)[positions.to(device, non_blocking=True)]
            lp, entropy = target_logprobs(hidden, self.head, longs[0], loss_config.log_entropy, self.head_weight)
            loss, stats = policy_loss(lp, floats[0], floats[1], floats[2], longs[1], len(micro), seq_len, loss_config)
            loss.backward()
            _accumulate(totals, stats, loss, entropy)
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
            **advantage_stats,
        }
        if "entropy" in values:
            metrics["policy/entropy"] = values["entropy"] / tokens
        return metrics

    def _share_prompts(self, groups: list[list[Trajectory]]) -> bool:
        mode = self.config.train.prompt_sharing
        if mode == "off" or self.parallel.fsdp:
            return False
        if mode == "on":
            return True
        lengths = [len(g[0].prompt_ids) for g in groups if g]
        return bool(lengths) and sum(lengths) / len(lengths) >= 1024 and self.config.rollout.group_size >= 4

    def _train_trees(
        self, groups: list[list[Trajectory]], weights: dict[int, float], totals: dict[str, torch.Tensor]
    ) -> list[Trajectory]:
        """Each group's trained rollouts as branches off their shared prompt (the trunk), one
        exact tree forward/backward per group: the prompt is computed once, not once per
        rollout. Returns the trajectories left for the plain path (groups with < 2 to train)."""
        from palingenesis.seco_tree import Branch, tree_forward_backward

        loss_config, device = self.config.loss, self.device
        rest: list[Trajectory] = []
        self.model.train()
        for group in groups:
            trained = [t for t in group if id(t) in weights]
            prompt = trained[0].prompt_ids if trained else []
            if len(trained) < 2 or len(prompt) < 2:
                rest += trained
                continue
            # the trunk ends one token early: branch position j (the prompt's last token, then the
            # completion) predicts completion token j, exactly as in the full sequence
            trunk = torch.tensor([prompt[:-1]], device=device)
            branches = [
                Branch(len(prompt) - 1, torch.tensor([[prompt[-1]] + t.tokens[:-1]], device=device)) for t in trained
            ]

            def loss_fn(i: int, hidden: torch.Tensor, trained=trained) -> torch.Tensor:
                t = trained[i]
                sampled = [j for j, m in enumerate(t.mask) if m]
                rows = hidden[0, sampled]
                targets = torch.tensor([t.tokens[j] for j in sampled], device=device)
                floats = torch.tensor(
                    [[t.logprobs[j] for j in sampled], [t.advantage] * len(sampled), [weights[id(t)]] * len(sampled)],
                    device=device,
                )
                lp, entropy = target_logprobs(rows, self.head, targets, loss_config.log_entropy, self.head_weight)
                seq = torch.zeros(len(sampled), dtype=torch.long, device=device)
                seq_len = torch.tensor([float(len(sampled))], device=device)
                loss, stats = policy_loss(lp, floats[0], floats[1], floats[2], seq, 1, seq_len, loss_config)
                _accumulate(totals, stats, loss, entropy)
                return loss

            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                tree_forward_backward(
                    self.model,
                    trunk,
                    branches,
                    loss_fn,
                    chunk_size=max(len(prompt), 1),
                    branch_tokens=self.config.train.micro_tokens,
                )
        return rest

    def _push_weights(self, version: int) -> float:
        """FSDP: gather the new weights (one parameter at a time, on every rank) and load them into
        this rank's engine, on the trainer thread. Seconds spent."""
        start = time.perf_counter()
        if isinstance(self.engine, VLLMColocateRollout):
            with self.pipeline.lock:  # the engine is between batches
                named = checkpoint_named_parameters(self.model, self.parallel.full_parameters(self.model))
                self.engine.update_weights(named, version)
        else:
            self.engine.update_weights((), version)
        return time.perf_counter() - start

    # ------------------------------------------------------------------- train

    def train(self) -> None:
        config = self.config
        t = config.train
        if self.start_step >= t.steps:
            logger.warning("the checkpoint is at step %d of %d: nothing to train", self.start_step, t.steps)
        if self.parallel.fsdp and self.engine.version < self.weights.version:
            self._push_weights(self.weights.version)  # resumed: the engine holds the initial weights
        previous = {sig: signal.signal(sig, self._request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        self.orchestrator.start()
        try:
            if t.eval_every and self.start_step == 0:
                self._log("eval", self.evaluate(), 0)
            start_time = time.time()
            for step in range(self.start_step, t.steps):
                t0 = time.perf_counter()
                batch: RLBatch = self.orchestrator.next(self.weights.version)
                waited = time.perf_counter() - t0
                groups = self._fresh(batch.samples)
                lr = self._lr_at(step)
                for group in self.opt.param_groups:
                    group["lr"] = lr
                t1 = time.perf_counter()
                metrics = self._train_step(groups)
                trained = self.parallel.sum([metrics.get("trained_sequences", 0.0)])[0]
                with self.weights.lock:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable(), t.max_grad_norm)
                    grad_norm = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
                    if trained:  # every rank steps (or none does): the parameters are sharded
                        self.opt.step()
                        if self.master is not None:
                            self.master.publish()
                    self.opt.zero_grad(set_to_none=True)
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
                    }
                )
                metrics = self._combine(metrics)
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
                    return
                if t.eval_every and (step + 1) % t.eval_every == 0 and step + 1 < t.steps:
                    self._log("eval", self.evaluate(), step + 1)
                if t.save_steps and (step + 1) % t.save_steps == 0:
                    self.save(f"step_{step + 1}", step + 1)
            self._log("eval", self.evaluate(), t.steps)
            if t.save_final:
                self.save("final")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.orchestrator.stop()
            self.pipeline.close()
            if self.server is not None:
                self.server.close()
            self.parallel.close()

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
        if self.wandb:
            try:
                self.wandb.log({(f"eval/{k}" if kind == "eval" else k): v for k, v in metrics.items()}, step=step)
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill training
                logger.warning("wandb.log failed (%s); disabling wandb", e)
                self.wandb = None

    def save(self, name: str, step: int | None = None) -> None:
        """The policy in Hugging Face format; with `step`, also the state to resume from: the
        optimizer (sharded DCP under FSDP), each rank's prompt sampler and random states.
        trainer_state.pt is written last, so a checkpoint with it is complete."""
        from palingenesis.checkpoint import _save_gathered_or_local

        path = Path(self.config.train.output_dir) / name
        if self.parallel.main:
            logger.info("Saving checkpoint -> %s", path)
        # The rollout engine idle (no generation with the trained model, no weight copy), in the
        # pipeline's lock order: pipeline, then weights.
        with self.pipeline.lock, self.weights.lock:
            self.parallel.reshard(self.model)
            with self.master.fp32() if self.master is not None else contextlib.nullcontext():
                _save_gathered_or_local(self.model, self.tok, path, self.parallel.fsdp)
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
            self.parallel.barrier()
        if self.parallel.main:
            with open(path / "rl_config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            if step is not None:
                state = {
                    "step": step,
                    "world": self.parallel.world,
                    "stale_groups": self.stale_groups,
                    "wandb_id": self.wandb_id,
                }
                torch.save(state, path / (TRAINER_STATE_FILE + ".tmp"))
                os.replace(path / (TRAINER_STATE_FILE + ".tmp"), path / TRAINER_STATE_FILE)
            keep = self.config.train.keep_checkpoints
            if keep > 0:
                for old in checkpoint_steps(self.config.train.output_dir, complete=False)[:-keep]:
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
        self.weights.version = state["step"]  # the engine holds older weights: the first rollout (or push) syncs
        self.stale_groups = state["stale_groups"]
        logger.info("Resumed from %s at step %d", path, self.start_step)
        return state["wandb_id"]


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
    try:
        with open("/proc/meminfo") as f:
            available = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
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


def _optimizer(params: list[torch.nn.Parameter], t: Any, device: str) -> torch.optim.Optimizer:
    """AdamW over the trained parameters (the fp32 masters on one GPU), with fp32 or 8-bit moments."""
    eps = t.adam_eps or (1e-15 if t.optimizer == "adamw" else 1e-8)
    kwargs = dict(lr=t.learning_rate, betas=(t.adam_beta1, t.adam_beta2), eps=eps, weight_decay=t.weight_decay)
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
