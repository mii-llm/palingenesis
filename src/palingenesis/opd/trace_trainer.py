"""On-policy distillation on agent traces: the pipeline and trainer (see opd.traces).

Per step, `rollout.batch_prompts` traces are drawn (by source weight), each routed
to its teacher (by source, or by topic), and planned on the producer thread while
the GPU trains: the turns to regenerate (`branches_per_trace`, each
`rollout.group_size` times) and the trunk they share. Then, on the GPU:

  1. the rollout engine prefills every trunk once (prefix caching) and samples
     all regenerated turns of all traces in one batch;
  2. each teacher reads each of its traces once (seco_tree.tree_hidden_states),
     returning hidden states at the regenerated turns and, with
     loss.trace_kd_weight, at the recorded turns;
  3. the student trains on each trace with seco_tree.tree_forward_backward: exact
     full-vocabulary reverse KL on every regenerated token (fused_rkl), and on the
     recorded turns, gradients through the whole shared context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from palingenesis.opd import losses
from palingenesis.opd.fused_rkl import fused_full_rkl
from palingenesis.opd.orchestrator import Pipeline
from palingenesis.opd.traces import (
    TracePlan,
    TracePlanner,
    TraceSample,
    branch_inputs,
    completion_rows,
    load_trace_rows,
    select,
    topic_teacher,
)
from palingenesis.opd.trainer import OPDTrainer
from palingenesis.seco_tree import Branch, tree_forward_backward, tree_hidden_states

logger = logging.getLogger(__name__)

PLAN_ATTEMPTS = 20        # traces drawn before giving up on one that has a usable turn


@dataclass
class TraceRequest:
    source: str
    teacher: str
    plan: TracePlan
    max_new_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceBatch:
    samples: list[TraceSample]
    version: int
    stats: dict[str, float]


class TraceSources:
    """The agent-trace sources, mixed by weight, each with a held-out split."""

    def __init__(self, config, rng: random.Random, think_tags: dict[str, tuple[str, str]] | None = None):
        self.config, self.rng = config, rng
        self.names, self.weights, self.train, self.dev = [], [], {}, {}
        for name, source in config.sources.items():
            tags = tuple(source.think_tags) if source.think_tags else (think_tags or {}).get(name)
            rows = load_trace_rows(source.path, source.messages_field, source.tools_field, tags)
            if source.dev_path:
                train, dev = rows, load_trace_rows(source.dev_path, source.messages_field, source.tools_field, tags)
            else:
                def key(row):
                    return hashlib.sha1(json.dumps(row["messages"], sort_keys=True, ensure_ascii=False,
                                                   default=str).encode()).hexdigest()
                ranked = sorted(rows, key=key)
                dev, train = ranked[:source.dev_size], ranked[source.dev_size:]
            if not train:
                raise ValueError(f"sources.{name}: no training traces left after holding out dev_size={source.dev_size}")
            self.names.append(name)
            self.weights.append(source.weight)
            self.train[name], self.dev[name] = train, dev
            logger.info("Agent traces %s: %d train / %d dev", name, len(train), len(dev))

    def sample(self) -> tuple[str, dict]:
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        return name, self.rng.choice(self.train[name])


class TracePipeline(Pipeline):
    """Rollouts of every regenerated turn, then each trace scored by its teacher (see module doc)."""

    def __init__(self, *args, chunk_size: int, branch_tokens: int, min_gap: int, prefix_caching: bool,
                 regenerate: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.regenerate = regenerate          # False: recorded turns only (no rollouts)
        self.chunk_size = chunk_size
        self.branch_tokens = branch_tokens
        self.min_gap = min_gap
        self.prefix_caching = prefix_caching

    def run(self, requests: list[TraceRequest], temperature: float, regenerate: bool | None = None) -> TraceBatch:
        """`regenerate` overrides the pipeline's setting for this call (evaluation always
        regenerates; the background producer uses the setting)."""
        if not (self.regenerate if regenerate is None else regenerate):
            return self._recorded_only(requests)
        with self._exclusive():
            sync = self.weights.sync(self.engine)
            t0 = time.perf_counter()
            self.engine.wake()
            t1 = time.perf_counter()
            if self.prefix_caching:     # prefill each trunk once: every turn of the trace then reuses it
                self.engine.generate([r.plan.trunk for r in requests], [1] * len(requests), 0.0)
            t2 = time.perf_counter()
            prompts = [b.context for r in requests for b in r.plan.branches]
            budgets = [r.max_new_tokens for r in requests for _ in r.plan.branches]
            rollouts = self.engine.generate(prompts, budgets, temperature)
            t3 = time.perf_counter()
            self.engine.sleep()
            samples, pos, finish = [], 0, []
            for r in requests:
                route = self.routes[r.teacher]
                keep, completions, behaviour, ends = [], [], [], []
                for i in range(len(r.plan.branches)):
                    rollout = rollouts[pos + i]
                    completion = route.aligner.clean(rollout.completion_ids)
                    finish.append(rollout.finish_reason)
                    if completion:
                        keep.append(i)
                        completions.append(completion)
                        behaviour.append(rollout.logprobs[:len(completion)])
                        ends.append(rollout.finish_reason)
                pos += len(r.plan.branches)
                if keep:
                    samples.append(TraceSample(select(r.plan, keep), completions, behaviour, ends, r.teacher,
                                               {**r.meta, "_src": r.source}))
            t4 = time.perf_counter()
            for sample in samples:
                self._score(sample)
            t5 = time.perf_counter()
        tokens = sum(len(c) for s in samples for c in s.completions)
        stats = {"time/sync": sync, "time/wake_sleep": (t1 - t0) + (t4 - t3), "time/prefill": t2 - t1,
                 "time/rollout": t3 - t2, "time/teacher": t5 - t4, "rollout_tokens": tokens,
                 "rollout_tok_s": tokens / max(t3 - t2, 1e-9),
                 "stop_rate": sum(f == "stop" for f in finish) / max(1, len(finish)),
                 "trunk_tokens": sum(len(s.plan.trunk) for s in samples) / max(1, len(samples)),
                 "branches": sum(len(s.completions) for s in samples)}
        version = min((rollouts[i].policy_version for i in range(len(rollouts))), default=self.engine.version)
        return TraceBatch(samples, version, stats)

    def _recorded_only(self, requests: list[TraceRequest]) -> TraceBatch:
        """No rollouts: each trace is scored by its teacher on its recorded turns alone."""
        with self._exclusive():
            start = time.perf_counter()
            samples = [TraceSample(TracePlan(r.plan.trunk, [], r.plan.kd_spans), [], [], [], r.teacher,
                                   {**r.meta, "_src": r.source}) for r in requests if r.plan.kd_spans]
            for sample in samples:
                self._score(sample)
            stats = {"time/teacher": time.perf_counter() - start, "branches": 0,
                     "trunk_tokens": sum(len(s.plan.trunk) for s in samples) / max(1, len(samples))}
        return TraceBatch(samples, self.weights.version, stats)

    @torch.no_grad()
    def _score(self, sample: TraceSample) -> None:
        """The teacher's hidden states at the regenerated and (with KD) recorded turns of one trace."""
        teacher = self.routes[sample.teacher].teacher
        device = teacher.device
        if teacher.offload:
            teacher.model.to(device)
        try:
            bridge = self.routes[sample.teacher].aligner.bridge
            trunk = torch.tensor([bridge.to_teacher(sample.plan.trunk)], device=device)
            inputs = [bridge.to_teacher(ids) for ids in branch_inputs(sample.plan, sample.completions)]
            branches = [Branch(b.attach, torch.tensor([ids], device=device))
                        for b, ids in zip(sample.plan.branches, inputs)]
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                hidden, kd = tree_hidden_states(teacher.model, trunk, branches,
                                                trunk_positions=sample.plan.kd_positions(), chunk_size=self.chunk_size,
                                                branch_tokens=self.branch_tokens, min_gap=self.min_gap)
            rows = completion_rows(sample.plan, sample.completions)
            sample.teacher_hidden = [h[0, r] for h, r in zip(hidden, rows)]
            sample.teacher_kd_hidden = kd
        finally:
            if teacher.offload:
                teacher.model.to("cpu")


class TraceTrainer(OPDTrainer):
    """OPDTrainer on agent traces: same models, rollout engines, schedule, checkpoints."""

    def _make_pipeline(self, engine, overlap: bool):
        return TracePipeline(self.tok, engine, self.routes, self.weights, self.config.model.chat_template_kwargs,
                             self.config.train.score_micro_seqs, torch.cuda.Stream() if overlap else None,
                             chunk_size=self.config.train.tree_chunk_size,
                             branch_tokens=self.config.train.tree_branch_tokens,
                             min_gap=self.config.train.tree_min_gap,
                             regenerate=self.config.loss.trace_branch_weight > 0,
                             prefix_caching=self.config.rollout.prefix_caching and self.config.rollout.backend != "hf")

    def _make_source(self):
        config = self.config
        self.planners = {
            name: TracePlanner(self.tok, config.model.chat_template_kwargs, self.stop_ids, source.max_context,
                               # recorded turns only: every turn that fits, so the trunk holds as many as can be
                               source.branches_per_trace if config.loss.trace_branch_weight > 0 else 0,
                               recorded_kd=config.loss.trace_kd_weight > 0)
            for name, source in config.sources.items()}
        # Evaluation regenerates turns in every configuration (the one metric all compare on)
        self.eval_planners = {
            name: TracePlanner(self.tok, config.model.chat_template_kwargs, self.stop_ids, source.max_context,
                               source.branches_per_trace, recorded_kd=False)
            for name, source in config.sources.items()}
        return TraceSources(config, self.rng, {name: p.think_tags for name, p in self.planners.items()})

    def _request(self, name: str, row: dict, rng: random.Random, planners=None) -> TraceRequest | None:
        source = self.config.sources[name]
        plan = (planners or self.planners)[name].plan(row["messages"], row.get("tools"), rng)
        if plan is None or (planners is None and self.config.loss.trace_branch_weight == 0 and not plan.kd_spans):
            return None          # recorded turns only, and this trace has none the trunk could hold
        group = self.config.rollout.group_size
        if group > 1:
            plan = TracePlan(plan.trunk, [b for b in plan.branches for _ in range(group)], plan.kd_spans)
        teacher = topic_teacher(row, source.topic_field, source.topic_teachers,
                                source.teacher or next(iter(self.routes)))
        topic = row.get(source.topic_field) if source.topic_field else None
        return TraceRequest(name, teacher, plan, source.max_new_tokens, {"_topic": topic})

    def _draw(self) -> list[TraceRequest]:
        requests = []
        for _ in range(self.config.rollout.batch_prompts):
            for _ in range(PLAN_ATTEMPTS):
                name, row = self.source.sample()
                request = self._request(name, row, self.rng)
                if request is not None:
                    requests.append(request)
                    break
            else:
                raise RuntimeError(f"{PLAN_ATTEMPTS} traces in a row had no turn whose context fits max_context")
        return requests

    # -------------------------------------------------------------------- loss

    def _scores(self, batch: TraceBatch, train: bool) -> dict[str, float]:
        loss_cfg = self.config.loss
        branch_tokens = sum(len(c) for s in batch.samples for c in s.completions)
        kd_on = loss_cfg.trace_kd_weight > 0
        kd_tokens = sum(len(s.plan.kd_positions()) for s in batch.samples) if kd_on else 0
        stats: dict[str, float] = defaultdict(float)
        branch_weight = loss_cfg.trace_branch_weight
        for sample in batch.samples:
            for k, v in self._score_trace(sample, train, branch_weight / max(branch_tokens, 1),
                                          loss_cfg.trace_kd_weight / max(kd_tokens, 1) if kd_tokens else 0.0).items():
                stats[k] += v
        return stats

    def _kl(self, name: str, hidden, teacher_hidden, targets, weights) -> tuple[torch.Tensor, dict]:
        route = self.routes[name]
        size = route.aligner.bridge.shared_vocab_size
        if name in self.fused:
            return fused_full_rkl(hidden, self.head, teacher_hidden, route.teacher.head, targets, weights, size)
        vocab = losses.SharedVocab(size, route.aligner.bridge.swap)
        return losses.full_rkl(hidden, self.head, targets, weights, vocab,
                               lambda a, b: route.teacher.log_probs(teacher_hidden[a:b], size))

    def _score_trace(self, sample: TraceSample, train: bool, branch_weight: float, kd_weight: float):
        name, device = sample.teacher, self.device
        bridge = self.routes[name].aligner.bridge
        plan = sample.plan
        trunk = torch.tensor([plan.trunk], device=device)
        branches = [Branch(b.attach, torch.tensor([ids], device=device))
                    for b, ids in zip(plan.branches, branch_inputs(plan, sample.completions))]
        rows = completion_rows(plan, sample.completions)
        targets = [torch.tensor(bridge.to_teacher(c), device=device) for c in sample.completions]
        teacher_hidden = [h.to(device) for h in sample.teacher_hidden]
        kd_positions = plan.kd_positions() if kd_weight > 0 else []
        kd_hidden = sample.teacher_kd_hidden.to(device) if kd_positions else None
        kd_index = {p: i for i, p in enumerate(kd_positions)}
        trunk_targets = torch.tensor(bridge.to_teacher(plan.trunk), device=device)
        stats: dict[str, float] = defaultdict(float)

        def branch_loss(i: int, hidden):
            h = hidden[0, rows[i]]
            weights = torch.full((h.shape[0],), branch_weight, device=device)
            value, s = self._kl(name, h, teacher_hidden[i], targets[i], weights)
            stats[f"kl/{name}"] += s["kl"]
            stats[f"k1/{name}"] += s["k1"]
            stats[f"residual/{name}"] += s["residual"]
            stats[f"tokens/{name}"] += h.shape[0]
            stats[f"loss/{name}"] += float(value.detach())
            return value

        def recorded_loss(positions: list[int], h):
            """KL at trunk `positions` (each predicting the next trunk token) from their hidden states h."""
            index = torch.tensor(positions, device=device)
            weights = torch.full((len(positions),), kd_weight, device=device)
            value, s = self._kl(name, h, kd_hidden[[kd_index[p] for p in positions]], trunk_targets[index + 1],
                                weights)
            stats[f"kd_kl/{name}"] += s["kl"]
            stats[f"kd_tokens/{name}"] += len(positions)
            stats[f"loss/{name}"] += float(value.detach())
            return value

        def trunk_loss(lo: int, hi: int, hidden):
            here = [p for p in kd_positions if lo <= p < hi]
            if not here:
                return None
            return recorded_loss(here, hidden[0, torch.tensor(here, device=device) - lo])

        chunk = self.config.train.tree_chunk_size
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            if train:
                tree_forward_backward(self.student, trunk, branches, branch_loss,
                                      trunk_loss_fn=trunk_loss if kd_positions else None, chunk_size=chunk,
                                      branch_tokens=self.config.train.tree_branch_tokens,
                                      min_gap=self.config.train.tree_min_gap)
            else:
                with torch.no_grad():
                    hidden, kd = tree_hidden_states(self.student, trunk, branches, trunk_positions=kd_positions,
                                                    chunk_size=chunk, branch_tokens=self.config.train.tree_branch_tokens,
                                      min_gap=self.config.train.tree_min_gap)
                    for i, h in enumerate(hidden):
                        branch_loss(i, h)
                    if kd_positions:
                        recorded_loss(kd_positions, kd)
        return stats

    def _summarize(self, stats: dict[str, float], batch: TraceBatch) -> dict[str, float]:
        metrics = dict(batch.stats)
        total = sum(v for k, v in stats.items() if k.startswith("tokens/"))
        metrics["loss"] = sum(v for k, v in stats.items() if k.startswith("loss/"))
        metrics["k1"] = sum(v for k, v in stats.items() if k.startswith("k1/")) / max(total, 1)
        metrics["completion_len"] = total / max(1, batch.stats.get("branches", 1))
        for name in self.routes:
            tokens = stats.get(f"tokens/{name}", 0)
            if tokens:
                metrics[f"kl/{name}"] = stats[f"kl/{name}"] / tokens
                metrics[f"k1/{name}"] = stats[f"k1/{name}"] / tokens
                metrics[f"tokens/{name}"] = tokens
                metrics[f"residual_mass/{name}"] = stats[f"residual/{name}"] / tokens
            if stats.get(f"kd_tokens/{name}"):
                metrics[f"kd_kl/{name}"] = stats[f"kd_kl/{name}"] / stats[f"kd_tokens/{name}"]
                metrics[f"kd_tokens/{name}"] = stats[f"kd_tokens/{name}"]
        return metrics

    # -------------------------------------------------------------------- eval

    def evaluate(self) -> dict[str, float]:
        """Held-out traces of every source: the student regenerates their turns (the rollout
        temperature, fixed plans), each scored against its teacher. dev_kl is the sampled
        estimate, dev_kl_full the exact per-token KL; per topic too when routed by topic."""
        self.student.eval()
        metrics: dict[str, float] = {}
        try:
            for name in self.config.sources:
                rows = self.source.dev[name][: self.config.train.eval_samples]
                rng = random.Random(self.config.train.seed)
                requests = [r for r in (self._request(name, row, rng, self.eval_planners) for row in rows)
                            if r is not None]
                if not requests:
                    continue
                with self.pipeline.lock:
                    batch = self.pipeline.run(requests, self.config.rollout.temperature, regenerate=True)
                    by_topic: dict[Any, list[TraceSample]] = defaultdict(list)
                    for s in batch.samples:
                        by_topic[s.meta.get("_topic")].append(s)
                    totals = defaultdict(float)
                    for topic, samples in by_topic.items():
                        part = TraceBatch(samples, batch.version, {})
                        stats = self._scores(part, train=False)
                        tokens = sum(v for k, v in stats.items() if k.startswith("tokens/"))
                        kl = sum(v for k, v in stats.items() if k.startswith("kl/"))
                        k1 = sum(v for k, v in stats.items() if k.startswith("k1/"))
                        totals["tokens"] += tokens
                        totals["kl"] += kl
                        totals["k1"] += k1
                        if topic is not None and tokens:
                            metrics[f"dev_kl_full/{name}/{topic}"] = kl / tokens
                metrics[f"dev_kl/{name}"] = totals["k1"] / max(totals["tokens"], 1)
                metrics[f"dev_kl_full/{name}"] = totals["kl"] / max(totals["tokens"], 1)
                metrics[f"dev_len/{name}"] = totals["tokens"] / max(1, batch.stats["branches"])
                metrics[f"dev_stop_rate/{name}"] = batch.stats["stop_rate"]
        finally:
            self.student.train()
        return metrics
