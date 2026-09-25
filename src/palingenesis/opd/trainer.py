"""On-policy distillation trainer.

Per step:
  1. the orchestrator's batch arrives: prompts drawn from the sources, sampled by
     the student with (at most max_staleness versions old) current weights, each
     completion aligned with and scored by its source's teacher (orchestrator.py)
  2. the student scores its own completions (the only forward with gradient) and
     each teacher group's loss is accumulated (losses.py): full_rkl, topk_kl,
     sampled_rkl or xtok, normalized by the batch's completion tokens
  3. clipped AdamW step on fp32 master weights (bf16 autocast), then the new
     weights are published to the rollout engine

Launch:
    pgs distill --config configs/distill_math.yaml
    python -m palingenesis.opd.trainer --config configs/distill_math.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer

from palingenesis.kernels import apply_liger_kernel, model_type_of
from palingenesis.logits import final_hidden_states, output_head, verify_output_head
from palingenesis.opd import fused_rkl, losses
from palingenesis.opd.align import ByteChunkAligner, SharedVocabAligner
from palingenesis.opd.config import OPDConfig, OPDConfigError, TeacherConfig
from palingenesis.opd.formatting import encode_prompt
from palingenesis.opd.fused_rkl import fused_full_rkl
from palingenesis.opd.orchestrator import Batch, Orchestrator, Pipeline, PublishedWeights, Request, Sample, TeacherRoute
from palingenesis.opd.rollout import HFRollout, VLLMColocateRollout, VLLMServer, VLLMServerRollout
from palingenesis.opd.sources import PromptSource, build_source
from palingenesis.opd.teachers import (
    HFTeacher,
    VLLMTeacher,
    completion_positions,
    end_of_turn_id,
    load_causal_lm,
    right_pad,
)
from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible
from palingenesis.seco import use_chunk_attention

logger = logging.getLogger(__name__)

TRAINER_STATE_FILE = "trainer_state.pt"     # written last: its presence marks a complete checkpoint

SHARED_VOCAB_LOSSES = ("full_rkl", "topk_kl", "sampled_rkl", "rs_kd")


def student_stop_ids(tok, model: str, extra_tokens: list[str], chat_template_kwargs: dict) -> tuple[int, ...]:
    """Tokens that end a student completion; the chat template's end-of-turn token first."""
    from transformers import GenerationConfig

    ids = [end_of_turn_id(tok, chat_template_kwargs), tok.eos_token_id]
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
        ids += eos if isinstance(eos, list) else [eos]
    except OSError:
        pass
    for name in extra_tokens:
        token = tok.convert_tokens_to_ids(name)
        if token is None or token == tok.unk_token_id:
            raise OPDConfigError(f"model.stop_tokens: {name!r} is not a token of {model}")
        ids.append(token)
    return tuple(dict.fromkeys(i for i in ids if i is not None))


def shared_vocab_bridge(student_tok, teacher_tok, teacher: TeacherConfig, name: str) -> TokenBridge | None:
    """The pair's TokenBridge, or None when the tokenizers do not share a vocabulary."""
    try:
        bridge = TokenBridge.from_tokenizers(student_tok, teacher_tok, eos_map=teacher.eos_map)
        check_compatible(student_tok, teacher_tok, bridge, probe_texts=tuple(teacher.probe_texts))
        return bridge
    except TokenBridgeError as e:
        if teacher.eos_map or teacher.loss in SHARED_VOCAB_LOSSES:
            raise OPDConfigError(
                f"teachers.{name}: {teacher.loss or 'eos_map'} needs a teacher that shares the student's "
                f"vocabulary, and {teacher.model}'s tokenizer does not ({e}). Use loss xtok (cross-tokenizer)."
            ) from None
        return None


class OPDTrainer:
    """The task-agnostic OPD engine; task-specific data lives in the sources.

    Pass a custom source (opd.sources.PromptSource) for tasks the built-ins don't
    cover; its prompts go to the first teacher unless its meta carries "_src" of
    a configured source.
    """

    def __init__(self, config: OPDConfig, source: PromptSource | None = None):
        for warning in config.validate():
            logger.warning(warning)
        self.config = config
        # Not MPS: rollouts run on the orchestrator's thread, and Metal command
        # buffers are not safe to use from two threads.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = random.Random(config.train.seed)
        torch.manual_seed(config.train.seed)
        os.makedirs(config.train.output_dir, exist_ok=True)
        chat_kwargs = config.model.chat_template_kwargs

        # Tokenizers and alignment first: pairing errors surface before any weights load.
        self.tok = AutoTokenizer.from_pretrained(config.model.student)
        self.stop_ids = student_stop_ids(self.tok, config.model.student, config.model.stop_tokens, chat_kwargs)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.kinds: dict[str, str] = {}
        aligners, teacher_toks = {}, {}
        for name, teacher in config.teachers.items():
            teacher_toks[name] = AutoTokenizer.from_pretrained(teacher.tokenizer or teacher.model)
            bridge = shared_vocab_bridge(self.tok, teacher_toks[name], teacher, name)
            kind = teacher.loss or ("xtok" if bridge is None else "full_rkl" if teacher.backend == "hf" else "topk_kl")
            if kind == "xtok":
                aligners[name] = ByteChunkAligner(self.tok, teacher_toks[name], self.stop_ids,
                                                  end_of_turn_id(teacher_toks[name], chat_kwargs),
                                                  config.loss.mask_whitespace)
                if bridge is not None:
                    logger.warning("teachers.%s shares the student's vocabulary; xtok works but full_rkl is exact", name)
            else:
                aligners[name] = SharedVocabAligner(bridge, self.stop_ids)
            self.kinds[name] = kind
            logger.info("teacher %s: %s (%s backend), loss %s", name, teacher.model, teacher.backend, kind)

        # vLLM servers start before the trainer's models: they claim their share of free GPU memory.
        self.servers: list[VLLMServer] = []
        teachers = {name: self._vllm_teacher(name, t) for name, t in config.teachers.items() if t.backend == "vllm"}
        rollout = config.rollout
        server = None
        if rollout.backend == "vllm_server":
            import importlib.util

            if not rollout.url and importlib.util.find_spec("ray") is None:
                raise OPDConfigError("rollout.backend vllm_server: vLLM's weight transfer into the server it "
                                     "launches needs ray. Install it with: pip install 'palingenesis[vllm-server]'")
            server = self._server(config.model.student, rollout.url, "rollout", [
                "--gpu-memory-utilization", str(rollout.gpu_memory_utilization),
                "--max-model-len", str(rollout.max_model_len), "--logprobs-mode", "processed_logprobs",
                "--weight-transfer-config", '{"backend": "ipc"}', *(["--enforce-eager"] if rollout.enforce_eager else []),
                *(["--enable-prefix-caching"] if rollout.prefix_caching else [])])

        if config.model.use_liger_kernel and self.device == "cuda":     # patches classes: before any model loads
            for model_type in sorted({model_type_of(m) for m in [config.model.student] + [
                    t.model for t in config.teachers.values() if t.backend == "hf"]} - {None}):
                apply_liger_kernel(model_type)
        self.resume_path = resolve_resume(config.train.resume_from, config.train.output_dir)
        student_path = self.resume_path or config.model.student
        logger.info("Loading student %s (fp32 master weights, bf16 autocast) on %s", student_path, self.device)
        self.student = load_causal_lm(student_path, torch.float32).to(self.device)
        if config.model.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
        self.head = output_head(self.student)
        verify_output_head(self.student, self.head)
        use_chunk_attention(self.student)       # fused attention under autocast (fp32 master weights)
        for name, teacher in config.teachers.items():
            if teacher.backend == "hf":
                teachers[name] = HFTeacher(teacher.model, teacher.device or self.device, teacher.offload,
                                           seed=config.train.seed)
                use_chunk_attention(teachers[name].model)

        if rollout.backend == "hf":
            engine = HFRollout(self.student, self.stop_ids, self.pad_id, rollout.micro_seqs)
        elif rollout.backend == "vllm":
            engine = VLLMColocateRollout(config.model.student, self.stop_ids, rollout.gpu_memory_utilization,
                                         rollout.max_model_len, rollout.enforce_eager, config.train.seed,
                                         sleep_mode=rollout.sleep and rollout.max_staleness == 0,
                                         prefix_caching=rollout.prefix_caching)
        else:
            engine = VLLMServerRollout(server, self.stop_ids)

        self.routes = {
            name: TeacherRoute(teacher_toks[name], aligners[name], teachers[name],
                               top_k=config.loss.top_k if kind == "topk_kl" or (
                                   kind == "xtok" and config.loss.xtok_dense_weight > 0) else 0,
                               keep_hidden=kind == "full_rkl",
                               sample_rounds=config.loss.rs_rounds if kind == "rs_kd" else 0,
                               sample_temperature=config.loss.rs_temperature)
            for name, kind in self.kinds.items()
        }
        # full_rkl through the fused kernels where they compute the same thing (fused_rkl.supported)
        self.fused = {name for name, kind in self.kinds.items()
                      if kind == "full_rkl" and fused_rkl.supported(self.head, teachers[name].head, aligners[name].bridge.swap)
                      and teachers[name].head.weight.device == self.head.weight.device}
        self.teacher_to_student = {name: torch.tensor(a.teacher_to_student, device=self.device)
                                   for name, a in aligners.items() if isinstance(a, ByteChunkAligner)}
        self.weights = PublishedWeights(self.student)
        # Rollouts overlapping training run on their own CUDA stream (see Pipeline), except
        # with the in-process vLLM engine: next to the trainer's default-stream kernels it
        # hit illegal memory accesses (vLLM 0.26) that CUDA_LAUNCH_BLOCKING=1 made vanish,
        # a race between streams. On the default stream only its host work overlaps.
        overlap = rollout.max_staleness > 0 and rollout.backend == "vllm_server" and self.device == "cuda"
        self.pipeline = self._make_pipeline(engine, overlap)
        self.source = source or self._make_source()
        self.orchestrator = Orchestrator(self.pipeline, self._draw, rollout.temperature, rollout.max_staleness)
        self.opt = torch.optim.AdamW(self.student.parameters(), lr=config.train.learning_rate, weight_decay=0.0,
                                     fused=self.device == "cuda")
        self.start_step, wandb_id = 0, None
        if self.resume_path:
            wandb_id = self._load_state(self.resume_path)
        self.wandb, self.wandb_id = None, wandb_id
        if config.logging.use_wandb:
            try:
                import wandb

                run = wandb.init(project=config.logging.project, name=config.logging.run_name or None,
                                 config=dataclasses.asdict(config), dir=config.train.output_dir,
                                 id=wandb_id, resume="allow" if wandb_id else None)
                self.wandb_id = run.id
                self.wandb = wandb
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill a training run
                logger.warning("wandb init failed (%s); continuing without it", e)

    def _make_pipeline(self, engine, overlap: bool):
        """The rollout pipeline (a subclass for other data shapes, e.g. agent traces)."""
        return Pipeline(self.tok, engine, self.routes, self.weights, self.config.model.chat_template_kwargs,
                        self.config.train.score_micro_seqs, torch.cuda.Stream() if overlap else None)

    def _make_source(self):
        return build_source(self.config, self.rng)

    def _server(self, model: str, url: str, role: str, args: list[str]) -> VLLMServer:
        server = VLLMServer(model, url=url, args=tuple(args),
                            log_path=os.path.join(self.config.train.output_dir, f"vllm_{role}.log"))
        self.servers.append(server)
        return server

    def _vllm_teacher(self, name: str, teacher: TeacherConfig) -> VLLMTeacher:
        top_k = max(self.config.loss.top_k, 1)
        return VLLMTeacher(self._server(teacher.model, teacher.url, f"teacher_{name}", [
            "--gpu-memory-utilization", str(teacher.gpu_memory_utilization),
            "--max-model-len", str(self.config.rollout.max_model_len + 512),
            "--max-logprobs", str(top_k + 1), "--enforce-eager"]))

    # ----------------------------------------------------------------- batches

    def _draw(self) -> list[Request]:
        """The next step's requests: batch_prompts prompts x group_size rollouts each."""
        requests = []
        for _ in range(self.config.rollout.batch_prompts):
            messages, max_new_tokens, meta = self.source.sample()
            source = meta.get("_src")
            teacher = self.config.teacher_of(source) if source in self.config.sources else next(iter(self.routes))
            request = self.pipeline.request(messages, max_new_tokens, meta, source, teacher)
            requests += [request] * self.config.rollout.group_size
        return requests

    def _lr_at(self, step: int) -> float:
        train = self.config.train
        if step < train.warmup_steps:
            return train.learning_rate * (step + 1) / train.warmup_steps
        if train.lr_scheduler == "constant":
            return train.learning_rate
        t = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
        return train.learning_rate * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    # -------------------------------------------------------------------- loss

    def _score(self, samples: list[Sample], weights: list[float], train: bool) -> dict[str, float]:
        """Student forward over `samples` (one teacher) and that teacher's loss; backward when training.

        `weights` is each sample's per-token loss weight (the normalization)."""
        name = samples[0].teacher
        kind, route, loss = self.kinds[name], self.routes[name], self.config.loss
        device = self.device
        ids, mask = right_pad([s.request.prompt_ids + s.completion[:-1] for s in samples], self.pad_id, device)
        positions = completion_positions([len(s.request.prompt_ids) for s in samples],
                                         [len(s.completion) for s in samples], ids.shape[1]).to(device)
        targets = torch.tensor([t for s in samples for t in s.completion], device=device)
        token_weights = torch.tensor([w for s, w in zip(samples, weights) for _ in s.completion], device=device)
        behaviour = None
        if all(len(s.behaviour_lp) == len(s.completion) for s in samples):
            behaviour = torch.tensor([x for s in samples for x in s.behaviour_lp], device=device)
        if kind != "xtok":            # one teacher token per student token
            teacher_ids = torch.tensor([t for s in samples for t in s.view.input_ids[s.view.prompt_len:]], device=device)
            vocab = losses.SharedVocab(route.aligner.bridge.shared_vocab_size, route.aligner.bridge.swap)
        if kind in ("topk_kl", "sampled_rkl", "rs_kd"):
            token_lp = torch.cat([s.scores.token_lp for s in samples]).to(device)

        with (torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")),
              torch.set_grad_enabled(train)):
            # right-padded rows: no mask (padding after a row cannot reach it; flash attention)
            hidden = final_hidden_states(self.student, ids, None)[positions]
            token_weights = self._token_weights(token_weights, hidden, behaviour)
            if kind == "full_rkl":
                for s in samples:       # made on the pipeline's CUDA stream: keep until this stream is done
                    if s.scores.hidden.is_cuda:
                        s.scores.hidden.record_stream(torch.cuda.current_stream())
                teacher_hidden = torch.cat([s.scores.hidden for s in samples])
                if name in self.fused:
                    value, stats = fused_full_rkl(hidden, self.head, teacher_hidden, route.teacher.head, teacher_ids,
                                                  token_weights, vocab.size)
                else:
                    value, stats = losses.full_rkl(hidden, self.head, teacher_ids, token_weights, vocab,
                                                   lambda a, b: route.teacher.log_probs(teacher_hidden[a:b], vocab.size))
            elif kind == "topk_kl":
                topk_ids = torch.cat([s.scores.topk_ids for s in samples]).to(device)
                topk_lp = torch.cat([s.scores.topk_lp for s in samples]).to(device)
                realized_in_topk = (topk_ids == teacher_ids[:, None]).any(1, keepdim=True)
                support = torch.cat([topk_ids, teacher_ids[:, None]], 1)
                valid = torch.cat([topk_lp.isfinite(), ~realized_in_topk], 1) & (support < vocab.size)
                value, stats = losses.topk_kl(hidden, self.head, teacher_ids, token_weights, vocab,
                                              support.clamp(max=vocab.size - 1),
                                              torch.cat([topk_lp, token_lp[:, None]], 1), valid, token_lp,
                                              beta=loss.beta)
            elif kind == "rs_kd":
                sample = [torch.cat([getattr(s.scores, f) for s in samples]).to(device)
                          for f in ("sample_ids", "sample_weights", "sample_lp")]
                value, stats = losses.rs_kd(hidden, self.head, teacher_ids, token_weights, vocab, *sample, token_lp)
            elif kind == "sampled_rkl":
                value, stats = losses.sampled_rkl(hidden, self.head, targets, token_weights, token_lp, behaviour,
                                                  loss.is_low, loss.is_high)
            else:
                chunks, dense = self._xtok_targets(samples, name)
                value, stats = losses.xtok(hidden, self.head, targets, token_weights, chunks, behaviour,
                                           spread=loss.xtok_spread, is_low=loss.is_low, is_high=loss.is_high,
                                           dense=dense, dense_weight=loss.xtok_dense_weight, beta=loss.beta)
        if train:
            value.backward()
        stats["loss"] = float(value.detach())
        stats["tokens"] = targets.numel()
        return stats

    def _token_weights(self, weights: Tensor, hidden: Tensor, behaviour: Tensor | None) -> Tensor:
        """`weights` times loss.token_weighting's per-token factors (detached).

        sure: 1 + alpha (1 - p), p the student's probability of the sampled token when it
        was sampled (the rollout's log-probs; on-policy, the current student's).
        entropy: 1 for the entropy_keep fraction of tokens with the highest student
        entropy in this micro-batch, 0 for the rest."""
        loss = self.config.loss
        if loss.token_weighting == "sure":
            if behaviour is None:
                raise RuntimeError("token_weighting sure needs the rollout's log-probs of the sampled tokens")
            return weights * (1 + loss.sure_alpha * (1 - behaviour.exp()))
        if loss.token_weighting == "entropy":
            entropy = losses.token_entropy(hidden.detach(), self.head)
            keep = max(1, math.ceil(loss.entropy_keep * entropy.numel()))
            threshold = entropy.topk(keep).values[-1]
            return weights * (entropy >= threshold)
        return weights

    def _xtok_targets(self, samples: list[Sample], name: str):
        """Chunk ids and log-probs, and the one-to-one dense targets, over a micro-batch."""
        device = self.device
        chunk_ids, keep, teacher_chunk_lp, ends = [], [], [], []
        dense_rows, dense_support, dense_lp = [], [], []
        row = offset = 0
        for s in samples:
            ch = s.view.chunks
            chunk_ids += [c + offset if c >= 0 else -1 for c in ch.student]
            keep += ch.keep
            t_index = torch.tensor(ch.teacher, dtype=torch.long)
            on = t_index >= 0
            teacher_chunk_lp.append(torch.zeros(ch.n_chunks).index_add_(0, t_index[on], s.scores.token_lp[on]))
            # a loss slice may end wherever no chunk continues across (chunk ids only increase)
            last = -1
            for i, c in enumerate(ch.student):
                if c >= 0:
                    if last >= 0 and c != last:
                        ends.append(row + i)
                    last = c
            ends.append(row + len(ch.student))
            if s.scores.topk_ids is not None:
                for s_pos, t_pos in ch.one_to_one:     # teacher top-k plus the teacher's actual token
                    actual = s.view.input_ids[s.view.prompt_len + t_pos]
                    dense_rows.append(row + s_pos)
                    dense_support.append(torch.cat([s.scores.topk_ids[t_pos], torch.tensor([actual])]))
                    dense_lp.append(torch.cat([s.scores.topk_lp[t_pos], s.scores.token_lp[t_pos:t_pos + 1]]))
            row += len(ch.student)
            offset += ch.n_chunks
        chunks = losses.ChunkTargets(torch.tensor(chunk_ids, device=device), torch.cat(teacher_chunk_lp).to(device),
                                     torch.tensor(keep, dtype=torch.bool, device=device), sorted(set(ends)))
        if not dense_rows:
            return chunks, None
        support = torch.stack(dense_support).to(device)
        t2s = self.teacher_to_student[name]
        mapped = torch.where(support < t2s.numel(), t2s[support.clamp(max=t2s.numel() - 1)], -1)
        teacher_lp = torch.stack(dense_lp).to(device)
        valid = (mapped >= 0) & teacher_lp.isfinite()
        valid[:, -1] &= ~(support[:, -1:] == support[:, :-1]).any(1)      # actual token already in the top-k
        return chunks, losses.DenseTargets(torch.tensor(dense_rows, device=device), mapped, teacher_lp, valid)

    def _scores(self, batch: Batch, train: bool) -> dict[str, float]:
        """Loss over a whole batch in micro-batches per teacher; per-teacher sums of the stats."""
        samples = batch.samples
        total_tokens = sum(len(s.completion) for s in samples)
        stats: dict[str, float] = defaultdict(float)
        for name in self.routes:
            group = sorted((s for s in samples if s.teacher == name), key=lambda s: len(s.completion))
            for start in range(0, len(group), self.config.train.score_micro_seqs):
                micro = group[start:start + self.config.train.score_micro_seqs]
                weights = [1.0 / (len(s.completion) * len(samples)) if self.config.loss.length_norm
                           else 1.0 / total_tokens for s in micro]
                for k, v in self._score(micro, weights, train).items():
                    stats[f"{k}/{name}"] += v
                stats[f"samples/{name}"] += len(micro)
        return stats

    # ------------------------------------------------------------------- train

    def train(self) -> None:
        config = self.config
        if self.start_step >= config.train.steps:
            logger.warning("the checkpoint is at step %d of %d: nothing to train", self.start_step, config.train.steps)
        self.orchestrator.start()
        try:
            if config.train.eval_every and self.start_step == 0:
                self._log("eval", self.evaluate(), 0)
            start_time = time.time()
            for step in range(self.start_step, config.train.steps):
                t0 = time.perf_counter()
                batch = self.orchestrator.next(self.weights.version)
                waited = time.perf_counter() - t0
                if not batch.samples:
                    logger.warning("step %d: no non-empty completions, skipping", step)
                    self.weights.publish()      # the producer waits for this step's version
                    continue
                lr = self._lr_at(step)
                for group in self.opt.param_groups:
                    group["lr"] = lr
                t1 = time.perf_counter()
                self.student.train()
                stats = self._scores(batch, train=True)
                with self.weights.lock:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), config.train.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                train_time = time.perf_counter() - t1
                staleness = self.weights.version - batch.version
                self.weights.publish()
                metrics = self._summarize(stats, batch)
                metrics.update({"lr": lr, "grad_norm": float(grad_norm), "staleness": staleness,
                                "dropped_samples": self.orchestrator.dropped, "time/train": train_time,
                                "time/wait_batch": waited, "time/step": time.perf_counter() - t0,
                                "elapsed": time.time() - start_time})
                if step % config.logging.log_every == 0:
                    self._log("step", metrics, step + 1)
                if config.train.eval_every and (step + 1) % config.train.eval_every == 0 and step + 1 < config.train.steps:
                    self._log("eval", self.evaluate(), step + 1)
                if config.train.save_steps and (step + 1) % config.train.save_steps == 0:
                    self.save(f"step_{step + 1}", step + 1)
            if config.train.eval_every:
                self._log("eval", self.evaluate(), config.train.steps)
            self.save("final")
        finally:
            self.orchestrator.stop()
            for server in self.servers:
                server.close()

    def _summarize(self, stats: dict[str, float], batch: Batch) -> dict[str, float]:
        metrics = {k: v for k, v in batch.stats.items()}
        total_tokens = sum(v for k, v in stats.items() if k.startswith("tokens/"))
        metrics["loss"] = sum(v for k, v in stats.items() if k.startswith("loss/"))
        metrics["k1"] = sum(v for k, v in stats.items() if k.startswith("k1/")) / max(total_tokens, 1)
        metrics["completion_len"] = total_tokens / max(len(batch.samples), 1)
        for name in self.routes:
            tokens = stats.get(f"tokens/{name}", 0)
            if tokens:
                metrics[f"kl/{name}"] = stats[f"kl/{name}"] / tokens
                metrics[f"k1/{name}"] = stats[f"k1/{name}"] / tokens
                metrics[f"tokens/{name}"] = tokens
                if f"is_dropped/{name}" in stats:
                    metrics[f"is_dropped/{name}"] = stats[f"is_dropped/{name}"] / tokens
                    metrics[f"abs_log_ratio/{name}"] = stats[f"abs_log_ratio/{name}"] / tokens
                if f"rs_unique/{name}" in stats:
                    metrics[f"rs_unique/{name}"] = stats[f"rs_unique/{name}"] / tokens
                if f"residual/{name}" in stats:
                    metrics[f"residual_mass/{name}"] = stats[f"residual/{name}"] / tokens
                if stats.get(f"dense_tokens/{name}"):
                    metrics[f"dense_kl/{name}"] = stats[f"dense_kl/{name}"] / stats[f"dense_tokens/{name}"]
                    metrics[f"dense_fraction/{name}"] = stats[f"dense_tokens/{name}"] / tokens
        metrics.update(self.source.batch_stats(
            [(s.request.meta, self.tok.decode(s.completion, skip_special_tokens=True)) for s in batch.samples]))
        return metrics

    # -------------------------------------------------------------------- eval

    def evaluate(self) -> dict[str, float]:
        """Held-out metrics of every source, each scored against its own teacher."""
        self.student.eval()
        try:
            return self.source.evaluate(lambda name: _SourceEngine(self, name))
        finally:
            self.student.train()

    def greedy_generate(self, messages_list, max_new_tokens: int) -> list[str]:
        prompts = [encode_prompt(self.tok, m, self.config.model.chat_template_kwargs) for m in messages_list]
        rollouts, _ = self.pipeline.generate(prompts, [max_new_tokens] * len(prompts), 0.0)
        return [self.tok.decode(r.completion_ids, skip_special_tokens=True) for r in rollouts]

    @torch.no_grad()
    def dev_kl(self, messages_list, max_new_tokens: int, source: str | None, teacher: str) -> dict[str, float]:
        """Sample the held-out prompts on-policy, score them against `teacher`, no gradient.

        dev_kl is the sampled estimate sum(log p_S - log p_T) per student token, on one
        scale for every loss; dev_kl_full (full_rkl teachers) the exact per-token KL."""
        requests = [self.pipeline.request(m, max_new_tokens, {}, source, teacher) for m in messages_list]
        with self.pipeline.lock:      # no rollout engine activity (vLLM sleep) while this scores on the GPU
            batch = self.pipeline.run(requests, self.config.rollout.temperature)
            if not batch.samples:
                return {"dev_kl": float("nan"), "dev_len": 0.0}
            stats = self._scores(batch, train=False)
        tokens = stats[f"tokens/{teacher}"]
        metrics = {"dev_kl": stats[f"k1/{teacher}"] / tokens, "dev_len": tokens / len(batch.samples)}
        if self.kinds[teacher] == "full_rkl":
            metrics["dev_kl_full"] = stats[f"kl/{teacher}"] / tokens
        return metrics

    # ------------------------------------------------------------- bookkeeping

    def _log(self, kind: str, metrics: dict[str, float], step: int) -> None:
        logger.info("%s %d | %s", kind, step, " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items())))
        if self.wandb:
            try:
                self.wandb.log({(f"eval/{k}" if kind == "eval" else k): v for k, v in metrics.items()}, step=step)
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill training
                logger.warning("wandb.log failed (%s); disabling wandb", e)
                self.wandb = None

    def save(self, name: str, step: int | None = None) -> None:
        """The student in Hugging Face format; with `step`, also the state to resume from.

        A resumable checkpoint adds the optimizer, the policy version and the random
        states. trainer_state.pt is written last and atomically, so a checkpoint with
        it is complete; `final` is the model export only."""
        from palingenesis.checkpoint import save_hf_model

        path = Path(self.config.train.output_dir) / name
        logger.info("Saving checkpoint -> %s", path)
        with self.weights.lock:       # a rollout engine may be copying the weights (max_staleness > 0)
            save_hf_model(self.student, self.tok, path, source_layout=True)
        with open(path / "opd_config.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)
        if step is not None:
            state = {"step": step, "optimizer": self.opt.state_dict(), "rng": self.rng.getstate(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if self.device == "cuda" else None,
                     "dropped_samples": self.orchestrator.dropped, "wandb_id": self.wandb_id}
            torch.save(state, path / (TRAINER_STATE_FILE + ".tmp"))
            os.replace(path / (TRAINER_STATE_FILE + ".tmp"), path / TRAINER_STATE_FILE)
        keep = self.config.train.keep_checkpoints
        if keep > 0:
            for old in checkpoint_steps(self.config.train.output_dir, complete=False)[:-keep]:
                shutil.rmtree(old, ignore_errors=True)

    def _load_state(self, path: str) -> str | None:
        """Restore the optimizer, the policy version and the random states; the wandb run id.

        The student's weights were loaded from `path` already. Rollout engines hold the
        weights they were built with, older than the policy version restored here, so
        the first rollout syncs them. Prompts continue with the source's random stream
        as it was at the save: the few prompts drawn ahead of training then are skipped,
        none are repeated."""
        # On the CPU: the random states must stay there; load_state_dict moves the optimizer state to the parameters.
        state = torch.load(Path(path) / TRAINER_STATE_FILE, map_location="cpu", weights_only=False)
        self.opt.load_state_dict(state["optimizer"])
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and self.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.start_step = state["step"]
        self.weights.version = state["step"]
        self.orchestrator.dropped = state["dropped_samples"]
        logger.info("Resumed from %s at step %d", path, self.start_step)
        return state["wandb_id"]


def checkpoint_steps(output_dir: str, complete: bool = True) -> list[str]:
    """The step_* checkpoint dirs in `output_dir`, oldest first; with `complete`, only resumable ones."""
    if not os.path.isdir(output_dir):
        return []
    found = []
    for d in os.listdir(output_dir):
        path = os.path.join(output_dir, d)
        if d.startswith("step_") and d[5:].isdigit() and os.path.isdir(path):
            if not complete or os.path.exists(os.path.join(path, TRAINER_STATE_FILE)):
                found.append((int(d[5:]), path))
    return [path for _, path in sorted(found)]


def resolve_resume(resume_from: str, output_dir: str) -> str | None:
    """The checkpoint to resume from: None to start fresh.

    "auto" picks the newest complete step_* checkpoint in output_dir, and starts
    fresh when there is none (so one command both starts and resumes a run)."""
    if not resume_from:
        return None
    if resume_from == "auto":
        found = checkpoint_steps(output_dir)
        if not found:
            logger.info("resume_from: auto: no checkpoint in %s, starting fresh", output_dir)
            return None
        return found[-1]
    if not os.path.exists(os.path.join(resume_from, TRAINER_STATE_FILE)):
        raise OPDConfigError(f"train.resume_from: {resume_from} is not a complete OPD checkpoint "
                             f"(no {TRAINER_STATE_FILE}; `final` holds the model only)")
    return resume_from


class _SourceEngine:
    """The engine services a source's evaluate() uses, bound to that source's teacher."""

    def __init__(self, trainer: OPDTrainer, source: str):
        self.trainer = trainer
        self.source = source
        config = trainer.config
        self.teacher = config.teacher_of(source) if source in config.sources else next(iter(trainer.routes))

    def greedy_generate(self, messages_list, max_new_tokens: int) -> list[str]:
        return self.trainer.greedy_generate(messages_list, max_new_tokens)

    def dev_kl(self, messages_list, max_new_tokens: int) -> dict[str, float]:
        return self.trainer.dev_kl(messages_list, max_new_tokens, self.source, self.teacher)


def main():
    from palingenesis.logging import setup_logging

    setup_logging(rank=0)
    config = OPDConfig.from_cli()
    if any(source.format == "agent_traces" for source in config.sources.values()):
        from palingenesis.opd.trace_trainer import TraceTrainer

        TraceTrainer(config).train()
    else:
        OPDTrainer(config).train()


if __name__ == "__main__":
    main()
