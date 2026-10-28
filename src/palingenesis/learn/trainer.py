"""One training session of a scope: plan the rows, train the scope's adapter, save a new version.

    1. plan        questions / probes / near-miss prompts for the unconsolidated facts, replay and anchor
                   rows (recipe.plan_session); written to sessions/sNNNN.tmp/
    2. model       the base, the ancestors' live adapters as a frozen delta, the scope's LoRA (warm start
                   from its live version); the fingerprint (base revision, tokenizer, chat template,
                   quantization) must match the store's
    3. preflight   the teacher answers the probes with the notes in context; failing probes are dropped,
                   and the session aborts when it adopts too few (usually a note the model's prior wins over)
    4. train       GKD: recipe.onpolicy_fraction of the rows continue the student's own sample, the rest a
                   teacher sample; full-vocabulary divergence (fkl default, rkl, jsd) on every token. The
                   sft mode trains cross-entropy on teacher samples. Teacher samples that mention the notes
                   ("according to the notes") are dropped
    5. save        adapters/vNNNN/ (PEFT format) + manifest.json, the session renamed to sNNNN
    6. evaluate    (optional, default on) the new version against the live one on the same probes
                   (evaluate.Evaluator, with the model already loaded): report.json / report.md

A version goes live only through Learner.set_live, which requires its gates to pass (or force=True).

Sampling is pluggable: HFSampler (the trained model's generate) now; VLLMColocateSampler is the interface
for an in-process vLLM engine that receives the merged MLP weights W + s B A after every step.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
import yaml

from palingenesis.learn import adapter as adapters
from palingenesis.learn.generate import Generator
from palingenesis.learn.recipe import META_REFERENCE, PARENT, PREVIOUS, NotesIndex, Row, SessionPlan, plan_session
from palingenesis.learn.render import with_notes
from palingenesis.learn.store import (
    ScopeStore,
    StoreError,
    normalize,
    now,
    remove_tree,
    sha256_file,
    sha256_text,
    write_jsonl,
)

logger = logging.getLogger(__name__)
STUDENT = "student"


class PreflightError(RuntimeError):
    """The teacher does not answer enough probes with the notes in context: training would not help."""


class Sampler(Protocol):
    def generate(self, prompts: list[list[int]], max_new_tokens: int, temperature: float, role: str) -> list[list[int]]:
        """One completion per prompt (token ids, ending with a stop token unless cut), sampled by `role`:
        student (P + the scope's adapter), parent (P) or previous (P + the live adapter)."""


class HFSampler:
    """model.generate on the trained model itself (opd.rollout.HFRollout), under the role's adapter state."""

    def __init__(self, model, stop_ids: tuple[int, ...], pad_id: int, roles: dict[str, Callable], micro_seqs: int = 8):
        from palingenesis.opd.rollout import HFRollout

        self.rollout = HFRollout(model, stop_ids, pad_id, micro_seqs)
        self.roles = roles

    def generate(self, prompts, max_new_tokens, temperature, role):
        with self.roles[role]():
            rollouts = self.rollout.generate(prompts, [max_new_tokens] * len(prompts), temperature)
        return [r.completion_ids for r in rollouts]


class VLLMColocateSampler:
    """Interface of the colocated vLLM sampler (serving milestone).

    It will wrap opd.rollout.VLLMColocateRollout: after each optimizer step the merged target weights
    W + s B A of the student (and of P for teacher samples) go through its update_weights path, so the
    engine needs no LoRA support for sampling.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("recipe.sampler vllm_colocate is not available yet; use recipe.sampler hf")

    def generate(self, prompts, max_new_tokens, temperature, role):  # pragma: no cover
        raise NotImplementedError


@dataclass
class SessionResult:
    scope: str
    session: str
    version: str
    adapter_dir: Path
    steps: int
    losses: list = field(default_factory=list)
    new_facts: int = 0
    rows: int = 0
    dropped_probes: int = 0
    warnings: list = field(default_factory=list)
    evaluation: object = None  # evaluate.EvalResult when the session was evaluated

    def summary(self) -> str:
        first = f"{self.losses[0]:.4f}" if self.losses else "-"
        last = f"{self.losses[-1]:.4f}" if self.losses else "-"
        return (
            f"{self.scope}: session {self.session} -> adapter {self.version} ({self.new_facts} new facts, "
            f"{self.rows} rows, {self.steps} steps, loss {first} -> {last})"
        )


def fingerprint(model, tok) -> dict:
    """What an adapter is bound to: base revision, tokenizer vocabulary, chat template, quantization."""
    vocab = json.dumps(sorted(tok.get_vocab().items()), ensure_ascii=False)
    quant = getattr(model.config, "quantization_config", None)
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict()
    return {
        "base_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_sha": sha256_text(vocab),
        "chat_template_sha": sha256_text(tok.chat_template or ""),
        "quantization": (quant or {}).get("quant_method", "none"),
    }


def _device(name: str) -> str:
    if name != "auto":
        return name
    from palingenesis.opd.teachers import pick_device

    return pick_device()


def write_session_files(path: Path, plan: SessionPlan, config) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        path / "questions.jsonl", [{"q": q.text, "kind": q.kind, "fact_ids": q.fact_ids} for q in plan.questions]
    )
    write_jsonl(path / "probes.jsonl", [{"q": q.text, "kind": q.kind, "fact_ids": q.fact_ids} for q in plan.probes])
    write_jsonl(path / "nearmiss.jsonl", [{"q": q.text, "kind": q.kind, "fact_ids": q.fact_ids} for q in plan.nearmiss])
    write_jsonl(path / "abstain.jsonl", [{"q": q.text, "kind": q.kind, "fact_ids": q.fact_ids} for q in plan.abstain])
    (path / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))


def discard_leftovers(ss: ScopeStore) -> None:
    """What a killed session left (taken under the store lock, so no session is running): its temporary
    session directory, and adapter weights saved before the version's manifest was written."""
    for stale in sorted((ss.dir / "sessions").glob("*.tmp")):
        remove_tree(stale)
    for orphan in sorted((ss.dir / "adapters").iterdir()):
        if orphan.is_dir() and not (orphan / "manifest.json").exists():
            remove_tree(orphan)


class SessionRunner:
    """Trains one scope. `model` / `tokenizer` / `sampler` override what the config would load (tests,
    notebooks); the model is wrapped by PEFT in place."""

    def __init__(
        self,
        learner,
        scope,
        generator: Generator | None,
        model=None,
        tokenizer=None,
        sampler: Sampler | None = None,
    ):
        self.learner = learner
        self.store = learner.store
        self.config = learner.config
        self.ss: ScopeStore = self.store.scope(scope).ensure()
        self.generator = generator
        self.model = model
        self.tok = tokenizer
        self.sampler = sampler

    # ------------------------------------------------------------------ model
    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = self.store.state["base"]
        device = _device(self.config.model.device)
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(base)
        if self.model is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
        self.store.check_fingerprint(fingerprint(self.model, self.tok))
        chain = self.ss.scope.chain()
        parents, ancestor_adapters = {}, []
        for scope in chain[:-1]:
            ancestor = self.store.scope(scope)
            if ancestor.live is not None:
                parents[str(scope)] = ancestor.live
                ancestor_adapters.append(adapters.read_adapter(ancestor.version_dir(ancestor.live))[0])
        self.parents = parents
        self.frozen = adapters.FrozenDelta(self.model, adapters.stack(ancestor_adapters) if ancestor_adapters else {})
        rank = self.config.scopes.ranks[self.ss.scope.level]
        self.peft = adapters.attach_lora(
            self.model, rank, rank * self.config.adapter.alpha_over_rank, self.config.adapter.target_regex
        )
        live = self.ss.live
        if live is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(self.peft, load_file(str(self.ss.version_dir(live) / adapters.ADAPTER_WEIGHTS)))
        self.previous = adapters.lora_state(self.peft)
        self.roles = {
            STUDENT: contextlib.nullcontext,
            PARENT: self.peft.disable_adapter,
            PREVIOUS: lambda: adapters.swapped_lora(self.peft, self.previous),
        }
        base_model = self.peft.get_base_model()
        self.decoder = base_model.get_decoder()
        self.head = base_model.get_output_embeddings()
        from palingenesis.opd.trainer import student_stop_ids

        kwargs = self.config.model.chat_template_kwargs
        self.stop_ids = student_stop_ids(self.tok, base, [], kwargs)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        if self.sampler is None and self.config.recipe.sampler == "vllm_colocate":
            self.sampler = VLLMColocateSampler()
        elif self.sampler is None:
            self.sampler = HFSampler(self.peft, self.stop_ids, self.pad_id, self.roles, self.config.recipe.micro_rows)

    def encode(self, messages: list[dict]) -> list[int]:
        from palingenesis.opd.formatting import encode_prompt

        return encode_prompt(self.tok, messages, self.config.model.chat_template_kwargs)

    # ------------------------------------------------------------------ preflight
    @torch.no_grad()
    def preflight(self, plan: SessionPlan, index: NotesIndex) -> int:
        """Drop the probes the teacher fails with the notes in context; abort when too few pass."""
        probes = [p for p in plan.probes if p.kind == "probe"]
        if not probes:
            return 0
        prompts = []
        for p in probes:
            messages = [{"role": "user", "content": p.text}]
            notes = index.for_facts(p.fact_ids, p.text, self.config.notes.teacher_neighbours)
            prompts.append(self.encode(with_notes(messages, notes)))
        answers = self.sampler.generate(prompts, self.config.recipe.max_new_tokens, 0.0, PARENT)
        passed = []
        for probe, ids in zip(probes, answers):
            text = normalize(self.tok.decode(ids, skip_special_tokens=True).split("</think>")[-1])
            facts = index.resolve(probe.fact_ids)
            passed.append(bool(facts) and all(normalize(f.value) in text for f in facts))
        rate = sum(passed) / len(passed)
        if rate < self.config.recipe.preflight_min:
            raise PreflightError(
                f"{self.ss.scope}: the teacher answers {sum(passed)}/{len(passed)} probes with the notes in context "
                f"(recipe.preflight_min {self.config.recipe.preflight_min}). The model's prior likely wins over the "
                "note text: rewrite the failing notes, or disable the check with recipe.preflight false."
            )
        kept = [p for p, ok in zip(probes, passed) if ok]
        plan.probes = kept + [p for p in plan.probes if p.kind != "probe"]
        return len(probes) - len(kept)

    # ------------------------------------------------------------------ loss
    def _hidden(self, prompts: list[list[int]], completions: list[list[int]]) -> torch.Tensor:
        from palingenesis.opd.teachers import completion_positions, right_pad

        device = self.head.weight.device
        seqs = [p + c[:-1] for p, c in zip(prompts, completions)]
        ids, mask = right_pad(seqs, self.pad_id, device)
        positions = completion_positions([len(p) for p in prompts], [len(c) for c in completions], ids.shape[1])
        hidden = self.decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return hidden[positions.to(device)]

    def loss(
        self, batch: list[tuple[Row, list[int], list[int], list[int]]], total_tokens: int
    ) -> tuple[torch.Tensor, float]:
        from palingenesis.opd import losses

        recipe = self.config.recipe
        device = self.head.weight.device
        targets = torch.tensor([t for *_, c in batch for t in c], device=device)
        weights = torch.full((len(targets),), 1.0 / total_tokens, device=device)
        if recipe.mode == "sft":
            hidden = self._hidden([s for _, s, _, _ in batch], [c for *_, c in batch])

            def ce(logits, a, b):
                lp = F.cross_entropy(logits, targets[a:b], reduction="none")
                return (weights[a:b] * lp).sum(), {"kl": lp.detach().sum()}

            value, _ = losses.chunked_head(hidden, self.head, ce)
            return value, float(value.detach())
        # Teacher passes first: the "previous" role copies the live LoRA weights into the trained parameters
        # and back, which must not happen between the student's forward and its backward.
        teacher_hidden = []
        with torch.no_grad():
            for policy in (PARENT, PREVIOUS):
                rows = [i for i, (row, *_) in enumerate(batch) if row.teacher_policy == policy]
                if rows:
                    with self.roles[policy]():
                        h = self._hidden([batch[i][2] for i in rows], [batch[i][3] for i in rows])
                    teacher_hidden.append((rows, h))
        hidden = self._hidden([s for _, s, _, _ in batch], [c for *_, c in batch])
        # Teacher states back in batch order (completion tokens are concatenated per row).
        offsets, total = [], 0
        for *_, c in batch:
            offsets.append(total)
            total += len(c)
        th = hidden.new_empty(hidden.shape).detach()
        for rows, h in teacher_hidden:
            k = 0
            for i in rows:
                n = len(batch[i][3])
                th[offsets[i] : offsets[i] + n] = h[k : k + n]
                k += n

        def teacher_logprobs(a, b):
            with torch.no_grad():
                return F.log_softmax(self.head(th[a:b]).float(), -1)

        vocab = losses.SharedVocab(self.head.weight.shape[0])
        args = (hidden, self.head, targets, weights, vocab, teacher_logprobs)
        if recipe.divergence == "fkl":
            value, _ = losses.full_fkl(*args)
        elif recipe.divergence == "rkl":
            value, _ = losses.full_rkl(*args)
        else:
            value, _ = losses.full_jsd(*args, beta=recipe.jsd_beta)
        return value, float(value.detach())

    def _sample_rows(self, rows: list[Row], rng: random.Random) -> list[tuple[Row, list[int], list[int], list[int]]]:
        recipe = self.config.recipe
        encoded = [(row, self.encode(row.student), self.encode(row.teacher)) for row in rows]
        onpolicy = [recipe.mode == "distill" and rng.random() < recipe.onpolicy_fraction for _ in rows]
        completions: list[list[int] | None] = [None] * len(rows)
        groups: dict[str, list[int]] = {}
        for i, (row, _, _) in enumerate(encoded):
            groups.setdefault(STUDENT if onpolicy[i] else row.teacher_policy, []).append(i)
        for role, idx in groups.items():
            prompts = [encoded[i][1] if role == STUDENT else encoded[i][2] for i in idx]
            outs = self.sampler.generate(prompts, recipe.max_new_tokens, recipe.temperature, role)
            for i, ids in zip(idx, outs):
                if role != STUDENT and META_REFERENCE.search(self.tok.decode(ids, skip_special_tokens=True)):
                    continue  # a teacher sample that talks about its notes is not a target
                completions[i] = ids
        return [(row, s, t, c) for (row, s, t), c in zip(encoded, completions) if c]

    # ------------------------------------------------------------------ run
    def run(self, evaluate: bool = True) -> SessionResult:
        config, ss = self.config, self.ss
        with self.store.lock():
            discard_leftovers(ss)
            rng = random.Random(config.recipe.seed)
            torch.manual_seed(config.recipe.seed)
            plan = plan_session(ss, self.generator, config, rng)
            if plan.empty:
                raise StoreError(f"{ss.scope}: nothing to train (no unconsolidated facts and no replay prompts).")
            for w in plan.warnings:
                logger.warning(w)
            number = ss.next_number()
            session, version = f"s{number:04d}", f"v{number:04d}"
            tmp = ss.dir / "sessions" / f"{session}.tmp"
            write_session_files(tmp, plan, config)
            try:
                self._load()
                index = NotesIndex(ss)
                dropped = self.preflight(plan, index) if config.recipe.preflight else 0
                write_jsonl(
                    tmp / "probes.jsonl", [{"q": q.text, "kind": q.kind, "fact_ids": q.fact_ids} for q in plan.probes]
                )
                result = self._train(plan, rng, session, version, tmp)
                result.dropped_probes = dropped
            except BaseException:
                remove_tree(tmp)
                self.release()
                raise
            # The version is saved: an evaluation failure leaves it for `pgs learn eval`.
            try:
                if evaluate:
                    from palingenesis.learn.evaluate import Evaluator

                    result.evaluation = Evaluator(self).run(version)
            finally:
                self.release()
        return result

    def load_version(self, version: str) -> None:
        """Put `version`'s weights in the trained LoRA (role student) after _load (role previous = live)."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(self.peft, load_file(str(self.ss.version_dir(version) / adapters.ADAPTER_WEIGHTS)))

    def use_baseline(self, chain: list[list]) -> None:
        """Make role previous the composite `chain` ([[scope, version or None], ...], ancestors first) instead of
        the live chain: its ancestors' adapters in place of the live ones underneath, and its version of this
        scope (or no adapter of it). The evaluation pairs a version against what is served, which a parent
        update makes differ from the live chain."""
        ancestors = [
            adapters.read_adapter(self.store.scope(s).version_dir(v))[0] for s, v in chain[:-1] if v is not None
        ]
        delta = self.frozen.place(adapters.stack(ancestors) if ancestors else {})
        own, state = chain[-1][1], None
        if own is not None:
            current = adapters.lora_state(self.peft)
            self.load_version(own)
            state = adapters.lora_state(self.peft)
            params = dict(self.peft.named_parameters())
            with torch.no_grad():
                for name, value in current.items():
                    params[name].copy_(value)

        @contextlib.contextmanager
        def baseline():
            scope_adapter = self.peft.disable_adapter() if state is None else adapters.swapped_lora(self.peft, state)
            with self.frozen.swapped(delta), scope_adapter:
                yield

        self.roles[PREVIOUS] = baseline  # the sampler holds this dict

    def release(self) -> None:
        """Take the frozen ancestor hooks off the model (the PEFT wrapping stays)."""
        if getattr(self, "frozen", None) is not None:
            self.frozen.remove()
            self.frozen = None

    def _train(self, plan: SessionPlan, rng: random.Random, session: str, version: str, tmp: Path) -> SessionResult:
        config, ss, recipe = self.config, self.ss, self.config.recipe
        params = [p for p in self.peft.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=recipe.learning_rate, weight_decay=0.0)
        self.peft.train()
        losses, started = [], time.time()
        cursor = 0
        for step in range(plan.steps):
            rows = [plan.rows[(cursor + k) % len(plan.rows)] for k in range(recipe.rows_per_step)]
            cursor += recipe.rows_per_step
            batch = self._sample_rows(rows, rng)
            if not batch:
                continue
            total = sum(len(c) for *_, c in batch)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for k in range(0, len(batch), recipe.micro_rows):
                value, scalar = self.loss(batch[k : k + recipe.micro_rows], total)
                value.backward()
                step_loss += scalar
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(step_loss)
            logger.info("%s step %d/%d loss %.4f", ss.scope, step + 1, plan.steps, step_loss)
        out = ss.version_dir(version)
        self.peft.save_pretrained(str(out))
        notes = ss.active_notes()
        manifest = {
            "version": version,
            "scope": str(ss.scope),
            "session": session,
            "created": now(),
            "base": self.store.state["base"],
            "rank": config.scopes.ranks[ss.scope.level],
            "alpha": config.scopes.ranks[ss.scope.level] * config.adapter.alpha_over_rank,
            "target_regex": config.adapter.target_regex,
            "parent_versions": self.parents,
            "warm_start": ss.live,
            # Every note active during training: the teacher could see any of them.
            "note_ids": [n.id for n in notes],
            "fact_ids": [f.id for f in plan.new_facts],
            "sha256": sha256_file(out / adapters.ADAPTER_WEIGHTS),
            "mode": recipe.mode,
            "divergence": recipe.divergence if recipe.mode == "distill" else None,
            "steps": plan.steps,
            "rows": len(plan.rows),
            "final_loss": losses[-1] if losses else None,
            "gates": None,
            "gated": False,
            "forced": False,
            "pruned": None,
        }
        (tmp / "train.json").write_text(
            json.dumps(
                {
                    "losses": losses,
                    "seconds": round(time.time() - started, 1),
                    "rows_processed": plan.steps * recipe.rows_per_step,
                }
            )
            + "\n"
        )
        tmp.rename(ss.dir / "sessions" / session)
        # The manifest last: until it exists the weights are not a version (a killed run leaves them for
        # discard_leftovers and forget to delete).
        ss.write_manifest(version, manifest)
        return SessionResult(
            str(ss.scope),
            session,
            version,
            out,
            plan.steps,
            losses,
            len(plan.new_facts),
            len(plan.rows),
            warnings=plan.warnings,
        )
