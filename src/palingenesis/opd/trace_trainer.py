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

A teacher on a vLLM server (teachers.<name>.backend vllm) scores each regenerated
turn and the trunk prefill-only instead (prompt_logprobs: its top-k and the
student token's log-prob at every position), and the student trains on topk_kl
against those sparse targets. It runs quantized next to the student (e.g. a 27B
in fp8), at a price: vLLM recomputes the whole prefill of a request that asks
for prompt log-probs (no prefix cache), so every regenerated turn costs a full
prefill of its context, and the response carries top-k entries for every
position of it.
"""

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from palingenesis.opd import losses
from palingenesis.opd.fused_rkl import fused_full_rkl
from palingenesis.opd.orchestrator import Pipeline
from palingenesis.opd.teachers import TeacherScores, VLLMTeacher
from palingenesis.opd.traces import (
    FailedTurns,
    OnsetScan,
    TeacherCutoff,
    TracePlan,
    TracePlanner,
    TraceSample,
    apply_cutoff,
    branch_inputs,
    completion_rows,
    load_prompt_rows_as_traces,
    load_trace_rows,
    select,
    topic_teacher,
)
from palingenesis.opd.trainer import OPDTrainer
from palingenesis.seco_tree import Branch, tree_forward_backward, tree_hidden_states

logger = logging.getLogger(__name__)

PLAN_ATTEMPTS = 20  # traces drawn before giving up on one that has a usable turn


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
            if source.format == "messages":  # plain prompts: one-turn traces

                def load(path, source=source):
                    return load_prompt_rows_as_traces(
                        path, source.messages_field, source.tools_field, source.chat_template_kwargs
                    )

            else:

                def load(path, source=source, tags=tags):
                    return load_trace_rows(path, source.messages_field, source.tools_field, tags)

            rows = load(source.path)
            if source.dev_path:
                train, dev = rows, load(source.dev_path)
            else:

                def key(row):
                    return hashlib.sha1(
                        json.dumps(row["messages"], sort_keys=True, ensure_ascii=False, default=str).encode()
                    ).hexdigest()

                ranked = sorted(rows, key=key)
                dev, train = ranked[: source.dev_size], ranked[source.dev_size :]
            if not train:
                raise ValueError(
                    f"sources.{name}: no training traces left after holding out dev_size={source.dev_size}"
                )
            self.names.append(name)
            self.weights.append(source.weight)
            self.train[name], self.dev[name] = train, dev
            logger.info("Agent traces %s: %d train / %d dev", name, len(train), len(dev))

    def sample(self) -> tuple[str, dict]:
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        return name, self.rng.choice(self.train[name])


class TracePipeline(Pipeline):
    """Rollouts of every regenerated turn, then each trace scored by its teacher (see module doc)."""

    def __init__(
        self,
        *args,
        chunk_size: int,
        branch_tokens: int,
        min_gap: int,
        prefix_caching: bool,
        regenerate: bool = True,
        cutoffs: dict[str, TeacherCutoff] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regenerate = regenerate  # False: recorded turns only (no rollouts)
        self.cutoffs = {k: v for k, v in (cutoffs or {}).items() if v is not None and v.active}  # per source
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
            if self.prefix_caching:  # prefill each trunk once: every turn of the trace then reuses it
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
                        behaviour.append(rollout.logprobs[: len(completion)])
                        ends.append(rollout.finish_reason)
                pos += len(r.plan.branches)
                if keep:
                    samples.append(
                        TraceSample(
                            select(r.plan, keep), completions, behaviour, ends, r.teacher, {**r.meta, "_src": r.source}
                        )
                    )
            t4 = time.perf_counter()
            for sample in samples:
                self._score(sample)
            t5 = time.perf_counter()
        cut = self._cut_stats(samples)
        samples = [s for s in samples if s.completions or s.plan.kd_spans]
        tokens = sum(len(c) for s in samples for c in s.completions)
        error_tokens = sum(len(c) for s in samples for b, c in zip(s.plan.branches, s.completions) if b.error)
        stats = {
            "time/sync": sync,
            "time/wake_sleep": (t1 - t0) + (t4 - t3),
            "time/prefill": t2 - t1,
            "time/rollout": t3 - t2,
            "time/teacher": t5 - t4,
            "rollout_tokens": tokens,
            "rollout_tok_s": tokens / max(t3 - t2, 1e-9),
            "stop_rate": sum(f == "stop" for f in finish) / max(1, len(finish)),
            "trunk_tokens": sum(len(s.plan.trunk) for s in samples) / max(1, len(samples)),
            "branches": sum(len(s.completions) for s in samples),
            "onset_branches": sum(b.onset for s in samples for b in s.plan.branches),
            **cut,
        }
        if any(b.error for s in samples for b in s.plan.branches):  # failed turns detected (FailedTurns)
            stats["error_branches"] = sum(b.error for s in samples for b in s.plan.branches)
            stats["error_token_share"] = error_tokens / max(tokens, 1)
        version = min((rollouts[i].policy_version for i in range(len(rollouts))), default=self.engine.version)
        return TraceBatch(samples, version, stats)

    def _recorded_only(self, requests: list[TraceRequest]) -> TraceBatch:
        """No rollouts: each trace is scored by its teacher on its recorded turns alone."""
        with self._exclusive():
            start = time.perf_counter()
            samples = [
                TraceSample(replace(r.plan, branches=[]), [], [], [], r.teacher, {**r.meta, "_src": r.source})
                for r in requests
                if r.plan.kd_spans
            ]
            for sample in samples:
                self._score(sample)
            cut = self._cut_stats(samples)
            samples = [s for s in samples if s.plan.kd_spans]
            stats = {
                "time/teacher": time.perf_counter() - start,
                "branches": 0,
                "trunk_tokens": sum(len(s.plan.trunk) for s in samples) / max(1, len(samples)),
                **cut,
            }
        return TraceBatch(samples, self.weights.version, stats)

    @staticmethod
    def _cut_stats(samples: list[TraceSample]) -> dict[str, float]:
        """TeacherCutoff's effect on a batch: the share of recorded-turn tokens and of regenerated turns past the
        cutoff, and the teacher's mean log-prob of the recorded turns it scored (per turn, then averaged)."""
        scored = [s.cutoff for s in samples if s.cutoff]
        if not scored:
            return {}
        recorded = sum(c["recorded_tokens"] for c in scored)
        branches = sum(c["branches"] for c in scored)
        turns = sum(c["turns"] for c in scored)
        return {
            "cut_trace_share": sum(c["cut"] for c in scored) / len(scored),
            "cut_token_share": sum(c["cut_tokens"] for c in scored) / max(recorded, 1),
            "cut_branch_share": sum(c["cut_branches"] for c in scored) / max(branches, 1),
            "teacher_turn_logp": sum(c["logp_sum"] for c in scored) / max(turns, 1),
        }

    @torch.no_grad()
    def _score(self, sample: TraceSample) -> None:
        """The teacher's view of one trace: at the regenerated turns and (with KD) the recorded turns, hidden
        states (hf) or sparse top-k log-probs (vLLM); with a TeacherCutoff, the turns past it dropped or
        down-weighted first."""
        teacher = self.routes[sample.teacher].teacher
        plan = sample.plan
        cutoff = self.cutoffs.get(sample.meta.get("_src")) if sample.meta.get("_cutoff") else None
        groups = cutoff.positions(plan.turns) if cutoff and plan.turns else []
        kd_positions = plan.kd_positions()
        scored = sorted(set().union(*groups))  # the positions whose log-prob the cutoff reads
        positions = sorted(set(kd_positions).union(scored))
        if isinstance(teacher, VLLMTeacher):
            branch_scores, trunk_scores, trunk_lp = self._score_vllm(sample, positions, scored)
        else:
            branch_scores, trunk_scores, trunk_lp = self._score_hf(sample, positions, scored)
        sample.teacher_hidden = branch_scores
        sample.teacher_kd_hidden = trunk_scores if kd_positions else None
        if not groups:
            return
        plan, keep = self._apply_cutoff(sample, cutoff, groups, dict(zip(scored, trunk_lp.tolist())))
        sample.plan = plan
        sample.completions = [sample.completions[i] for i in keep]
        sample.behaviour_lp = [sample.behaviour_lp[i] for i in keep]
        sample.finish = [sample.finish[i] for i in keep]
        sample.teacher_hidden = [branch_scores[i] for i in keep]
        index = {p: i for i, p in enumerate(positions)}
        rows = [index[p] for p in plan.kd_positions()]
        sample.teacher_kd_hidden = _rows(trunk_scores, rows) if rows else None

    @staticmethod
    def _apply_cutoff(sample: TraceSample, cutoff: TeacherCutoff, groups, at: dict[int, float]):
        """Per-turn means of the teacher's log-probs, the cutoff turn, and the plan without what follows it."""
        plan = sample.plan
        means = [sum(at[p] for p in g) / len(g) if g else 0.0 for g in groups]
        i = cutoff.cut_index(means)
        recorded = sum(b - a for _, a, b in plan.turns)
        stats = {
            "cut": 0,
            "cut_tokens": 0,
            "cut_branches": 0,
            "recorded_tokens": recorded,
            "branches": len(plan.branches),
            "turns": len(means),
            "logp_sum": sum(means),
        }
        keep = list(range(len(plan.branches)))
        if i is not None:
            turn = plan.turns[i][0]
            stats["cut"] = 1
            stats["cut_tokens"] = sum(b - a for k, a, b in plan.turns if k > turn)
            stats["cut_branches"] = sum(b.turn > turn for b in plan.branches)
            plan, keep = apply_cutoff(plan, turn, cutoff.weight)
        sample.cutoff = stats
        return plan, keep

    def _score_hf(self, sample: TraceSample, positions: list[int], scored: list[int]):
        teacher = self.routes[sample.teacher].teacher
        device = teacher.device
        if teacher.offload:
            teacher.model.to(device)
        try:
            bridge = self.routes[sample.teacher].aligner.bridge
            trunk_ids = bridge.to_teacher(sample.plan.trunk)
            trunk = torch.tensor([trunk_ids], device=device)
            inputs = [bridge.to_teacher(ids) for ids in branch_inputs(sample.plan, sample.completions)]
            branches = [
                Branch(b.attach, torch.tensor([ids], device=device)) for b, ids in zip(sample.plan.branches, inputs)
            ]
            with torch.autocast(
                torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
            ):
                hidden, kd = tree_hidden_states(
                    teacher.model,
                    trunk,
                    branches,
                    trunk_positions=positions,
                    chunk_size=self.chunk_size,
                    branch_tokens=self.branch_tokens,
                    min_gap=self.min_gap,
                )
            rows = completion_rows(sample.plan, sample.completions)
            lp = None
            if scored:  # the teacher's log-prob of the next trunk token at `scored`, a slice of rows at a time
                size = bridge.shared_vocab_size
                index = {p: i for i, p in enumerate(positions)}
                rows_at = torch.tensor([index[p] for p in scored], device=kd.device)
                targets = torch.tensor([trunk_ids[p + 1] for p in scored], device=kd.device)
                step = max(1, 2**27 // teacher.head.weight.shape[0])
                lp = torch.cat(
                    [
                        teacher.log_probs(kd[rows_at[a : a + step]], size)
                        .gather(1, targets[a : a + step, None])
                        .squeeze(1)
                        for a in range(0, len(scored), step)
                    ]
                ).cpu()
            return [h[0, r] for h, r in zip(hidden, rows)], kd, lp
        finally:
            if teacher.offload:
                teacher.model.to("cpu")

    def _score_vllm(self, sample: TraceSample, positions: list[int], scored: list[int]):
        """Prefill-only scoring: one request per regenerated turn (its context and completion) and, when trunk
        positions are needed, one for the trunk."""
        route = self.routes[sample.teacher]
        bridge = route.aligner.bridge
        sequences, targets = [], []
        for b, c in zip(sample.plan.branches, sample.completions):
            sequences.append(bridge.to_teacher(b.context + c))
            targets.append(list(range(len(b.context), len(b.context) + len(c))))
        if positions:
            sequences.append(bridge.to_teacher(sample.plan.trunk))
            targets.append([p + 1 for p in positions])
        scores = route.teacher.score_targets(sequences, targets, route.top_k) if sequences else []
        trunk = scores[-1] if positions else None
        lp = None
        if scored:
            index = {p: i for i, p in enumerate(positions)}
            lp = trunk.token_lp[torch.tensor([index[p] for p in scored], dtype=torch.long)]
        return scores[: len(sample.plan.branches)], trunk, lp


def _rows(scores, rows: list[int]):
    """Rows of a teacher's per-position output: hidden states [N, H] or sparse TeacherScores."""
    if isinstance(scores, TeacherScores):
        index = torch.tensor(rows, dtype=torch.long)
        return TeacherScores(scores.token_lp[index], scores.topk_ids[index], scores.topk_lp[index])
    return scores[torch.tensor(rows, dtype=torch.long, device=scores.device)]


def _onset_scan(source, kd: bool) -> OnsetScan | None:
    scan = OnsetScan(
        source.onset_branches,
        source.onset_kd_tokens if kd else -1,
        source.onset_ngram,
        source.onset_window,
        source.onset_threshold,
        source.onset_common,
    )
    return scan if scan.active else None


def _failed_turns(source) -> FailedTurns | None:
    rules = FailedTurns(
        tuple(source.error_patterns),
        tuple(source.error_roles),
        source.error_field,
        source.error_invalid_calls,
        source.error_weight,
        source.first_error_weight,
        source.success_field,
        source.error_branches,
    )
    return rules if rules.active else None


def _cutoff(source) -> TeacherCutoff | None:
    cutoff = TeacherCutoff(source.cutoff_logp, source.cutoff_window, source.cutoff_tokens, source.cutoff_weight)
    return cutoff if cutoff.active else None


class TraceTrainer(OPDTrainer):
    """OPDTrainer on agent traces: same models, rollout engines, schedule, checkpoints."""

    def _make_pipeline(self, engine, overlap: bool):
        return TracePipeline(
            self.tok,
            engine,
            self.routes,
            self.weights,
            self.config.model.chat_template_kwargs,
            self.config.train.score_micro_seqs,
            torch.cuda.Stream() if overlap else None,
            chunk_size=self.config.train.tree_chunk_size,
            branch_tokens=self.config.train.tree_branch_tokens,
            min_gap=self.config.train.tree_min_gap,
            regenerate=self.config.loss.trace_branch_weight > 0,
            prefix_caching=self.config.rollout.prefix_caching and self.config.rollout.backend != "hf",
            cutoffs={name: _cutoff(source) for name, source in self.config.sources.items()},
        )

    def _make_source(self):
        config = self.config
        self.planners = {
            name: TracePlanner(
                self.tok,
                {**config.model.chat_template_kwargs, **source.chat_template_kwargs},
                self.stop_ids,
                source.max_context,
                # recorded turns only: every turn that fits, so the trunk holds as many as can be
                source.branches_per_trace if config.loss.trace_branch_weight > 0 else 0,
                recorded_kd=config.loss.trace_kd_weight > 0,
                onsets=_onset_scan(source, kd=config.loss.trace_kd_weight > 0),
                failures=_failed_turns(source),
                cutoff=_cutoff(source),
            )
            for name, source in config.sources.items()
        }
        # Evaluation regenerates turns in every configuration (the one metric all compare on)
        self.eval_planners = {
            name: TracePlanner(
                self.tok,
                {**config.model.chat_template_kwargs, **source.chat_template_kwargs},
                self.stop_ids,
                source.max_context,
                source.branches_per_trace,
                recorded_kd=False,
                onsets=_onset_scan(source, kd=False),  # dev reports the KL at the onsets apart (dev_kl_onset)
                failures=_failed_turns(source),  # and at failed turns (dev_kl_error); dev has no cutoff
            )
            for name, source in config.sources.items()
        }
        return TraceSources(config, self.rng, {name: p.think_tags for name, p in self.planners.items()})

    def _request(self, name: str, row: dict, rng: random.Random, planners=None) -> TraceRequest | None:
        source = self.config.sources[name]
        plan = (planners or self.planners)[name].plan(row["messages"], row.get("tools"), rng, row)
        if plan is None or (planners is None and self.config.loss.trace_branch_weight == 0 and not plan.kd_spans):
            return None  # recorded turns only, and this trace has none the trunk could hold
        group = self.config.rollout.group_size
        if group > 1:
            plan = replace(plan, branches=[b for b in plan.branches for _ in range(group)])
        teacher = topic_teacher(
            row, source.topic_field, source.topic_teachers, source.teacher or next(iter(self.routes))
        )
        topic = row.get(source.topic_field) if source.topic_field else None
        # training only: evaluation scores every held-out turn the same way in every configuration
        meta = {"_topic": topic, "_cutoff": planners is None}
        return TraceRequest(name, teacher, plan, source.max_new_tokens, meta)

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
        # token counts weighted by FailedTurns / TeacherCutoff (all 1 by default): the loss is a weighted mean
        branch_tokens = sum(len(c) * b.weight for s in batch.samples for b, c in zip(s.plan.branches, s.completions))
        kd_on = loss_cfg.trace_kd_weight > 0
        kd_tokens = sum(sum(s.plan.kd_weights()) for s in batch.samples) if kd_on else 0
        stats: dict[str, float] = defaultdict(float)
        branch_weight = loss_cfg.trace_branch_weight
        for sample in batch.samples:
            for k, v in self._score_trace(
                sample,
                train,
                branch_weight / branch_tokens if branch_tokens > 0 else 0.0,
                loss_cfg.trace_kd_weight / kd_tokens if kd_tokens > 0 else 0.0,
            ).items():
                stats[k] += v
        return stats

    def _kl(self, name: str, hidden, teacher_hidden, targets, weights) -> tuple[torch.Tensor, dict]:
        route = self.routes[name]
        size = route.aligner.bridge.shared_vocab_size
        if isinstance(teacher_hidden, TeacherScores):  # a vLLM teacher's top-k: topk_kl on its support
            device = hidden.device
            topk_ids, topk_lp = teacher_hidden.topk_ids.to(device), teacher_hidden.topk_lp.to(device)
            token_lp = teacher_hidden.token_lp.to(device)
            realized_in_topk = (topk_ids == targets[:, None]).any(1, keepdim=True)
            support = torch.cat([topk_ids, targets[:, None]], 1)
            valid = torch.cat([topk_lp.isfinite(), ~realized_in_topk], 1) & (support < size)
            return losses.topk_kl(
                hidden,
                self.head,
                targets,
                weights,
                losses.SharedVocab(size, route.aligner.bridge.swap),
                support.clamp(max=size - 1),
                torch.cat([topk_lp, token_lp[:, None]], 1),
                valid,
                token_lp,
                beta=self.config.loss.beta,
            )
        if name in self.fused:
            return fused_full_rkl(hidden, self.head, teacher_hidden, route.teacher.head, targets, weights, size)
        vocab = losses.SharedVocab(size, route.aligner.bridge.swap)
        return losses.full_rkl(
            hidden, self.head, targets, weights, vocab, lambda a, b: route.teacher.log_probs(teacher_hidden[a:b], size)
        )

    def _score_trace(self, sample: TraceSample, train: bool, branch_weight: float, kd_weight: float):
        name, device = sample.teacher, self.device
        bridge = self.routes[name].aligner.bridge
        plan = sample.plan
        trunk = torch.tensor([plan.trunk], device=device)
        branches = [
            Branch(b.attach, torch.tensor([ids], device=device))
            for b, ids in zip(plan.branches, branch_inputs(plan, sample.completions))
        ]
        rows = completion_rows(plan, sample.completions)
        targets = [torch.tensor(bridge.to_teacher(c), device=device) for c in sample.completions]
        teacher_hidden = [h if isinstance(h, TeacherScores) else h.to(device) for h in sample.teacher_hidden]
        kd_positions = plan.kd_positions() if kd_weight > 0 else []
        kd_hidden = sample.teacher_kd_hidden if kd_positions else None
        if kd_hidden is not None and not isinstance(kd_hidden, TeacherScores):
            kd_hidden = kd_hidden.to(device)
        kd_index = {p: i for i, p in enumerate(kd_positions)}
        kd_scale = dict(zip(kd_positions, plan.kd_weights())) if kd_positions and plan.turn_weights else None
        trunk_targets = torch.tensor(bridge.to_teacher(plan.trunk), device=device)
        stats: dict[str, float] = defaultdict(float)

        def branch_loss(i: int, hidden):
            h = hidden[0, rows[i]]
            weights = torch.full((h.shape[0],), branch_weight * plan.branches[i].weight, device=device)
            value, s = self._kl(name, h, teacher_hidden[i], targets[i], weights)
            stats[f"kl/{name}"] += s["kl"]
            stats[f"k1/{name}"] += s["k1"]
            stats[f"residual/{name}"] += s.get("residual", 0.0)
            stats[f"tokens/{name}"] += h.shape[0]
            stats[f"loss/{name}"] += float(value.detach())
            if plan.branches[i].onset:
                stats[f"onset_kl/{name}"] += s["kl"]
                stats[f"onset_tokens/{name}"] += h.shape[0]
            if plan.branches[i].error:
                stats[f"error_kl/{name}"] += s["kl"]
                stats[f"error_tokens/{name}"] += h.shape[0]
            return value

        def recorded_loss(positions: list[int], h):
            """KL at trunk `positions` (each predicting the next trunk token) from their hidden states h."""
            index = torch.tensor(positions, device=device)
            if kd_scale is None:
                weights = torch.full((len(positions),), kd_weight, device=device)
            else:
                weights = torch.tensor([kd_weight * kd_scale[p] for p in positions], device=device)
            value, s = self._kl(
                name, h, _rows(kd_hidden, [kd_index[p] for p in positions]), trunk_targets[index + 1], weights
            )
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
                tree_forward_backward(
                    self.student,
                    trunk,
                    branches,
                    branch_loss,
                    trunk_loss_fn=trunk_loss if kd_positions else None,
                    chunk_size=chunk,
                    branch_tokens=self.config.train.tree_branch_tokens,
                    min_gap=self.config.train.tree_min_gap,
                )
            else:
                with torch.no_grad():
                    hidden, kd = tree_hidden_states(
                        self.student,
                        trunk,
                        branches,
                        trunk_positions=kd_positions,
                        chunk_size=chunk,
                        branch_tokens=self.config.train.tree_branch_tokens,
                        min_gap=self.config.train.tree_min_gap,
                    )
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
            if stats.get(f"onset_tokens/{name}"):  # the regenerated turns that start where a recorded one repeated
                metrics[f"kl_onset/{name}"] = stats[f"onset_kl/{name}"] / stats[f"onset_tokens/{name}"]
            if stats.get(f"error_tokens/{name}"):  # the regenerated turns whose recorded tool call failed
                metrics[f"kl_error/{name}"] = stats[f"error_kl/{name}"] / stats[f"error_tokens/{name}"]
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
                requests = [
                    r for r in (self._request(name, row, rng, self.eval_planners) for row in rows) if r is not None
                ]
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
                        for key in ("onset_kl", "onset_tokens", "error_kl", "error_tokens"):
                            totals[key] += sum(v for k, v in stats.items() if k.startswith(f"{key}/"))
                        if topic is not None and tokens:
                            metrics[f"dev_kl_full/{name}/{topic}"] = kl / tokens
                metrics[f"dev_kl/{name}"] = totals["k1"] / max(totals["tokens"], 1)
                metrics[f"dev_kl_full/{name}"] = totals["kl"] / max(totals["tokens"], 1)
                metrics[f"dev_len/{name}"] = totals["tokens"] / max(1, batch.stats["branches"])
                metrics[f"dev_stop_rate/{name}"] = batch.stats["stop_rate"]
                if totals["onset_tokens"]:
                    metrics[f"dev_kl_onset/{name}"] = totals["onset_kl"] / totals["onset_tokens"]
                if totals["error_tokens"]:
                    metrics[f"dev_kl_error/{name}"] = totals["error_kl"] / totals["error_tokens"]
        finally:
            self.student.train()
        return metrics
