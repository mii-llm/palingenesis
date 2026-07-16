"""On-policy distillation trainer.

Per step (mirrors tinker-cookbook's on_policy_distillation, self-contained):
  1. draw pool prompts, render with the STUDENT's chat template
  2. student samples completions (temperature 1.0) with its current weights
     -> exactly on-policy, single gradient step per batch, no importance sampling
  3. teacher scores the same completions conditioned on the SAME conversation
     rendered with the TEACHER's chat template (see token_bridge for why)
  4. loss = full-distribution reverse KL over completion tokens
     sum_v p_student(v) * (log p_student(v) - log p_teacher(v))
     ("sampled_rkl" reproduces tinker's sampled-token REINFORCE variant)

Single-process, single-GPU by design: the student must both generate and take
gradients each step, so there is no idle phase to shard away. On an 80 GB GPU
a 0.4B student + 3B teacher fit together; if they don't, lower
train.score_micro_seqs, enable model.gradient_checkpointing, or move the
teacher with model.teacher_device.

Launch:
    pgs distill --config configs/distill_opd.yaml
    python -m palingenesis.opd.trainer --config configs/distill_opd.yaml
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.formatting import PromptRenderer, build_messages, extract_letter, load_reference_shots
from palingenesis.opd.pool import load_pool, split_pool
from palingenesis.opd.token_bridge import TokenBridge, check_compatible

logger = logging.getLogger(__name__)


def load_causal_lm(name: str, dtype: torch.dtype):
    """from_pretrained across the transformers 4.x (torch_dtype) / 5.x (dtype) rename."""
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OPDTrainer:
    def __init__(self, config: OPDConfig):
        self.config = config
        self.device = pick_device()
        self.teacher_device = config.model.teacher_device or self.device
        self.rng = random.Random(config.train.seed)
        torch.manual_seed(config.train.seed)
        os.makedirs(config.train.output_dir, exist_ok=True)

        logger.info("Loading tokenizers (%s / %s)", config.model.student, config.model.teacher)
        self.s_tok = AutoTokenizer.from_pretrained(config.model.student)
        self.t_tok = AutoTokenizer.from_pretrained(config.model.teacher)
        self.bridge = TokenBridge.from_tokenizers(
            self.s_tok,
            self.t_tok,
            eos_map=config.bridge.eos_map,
            extra_stop_tokens=tuple(config.bridge.extra_stop_tokens),
        )
        check_compatible(self.s_tok, self.t_tok, self.bridge, probe_texts=tuple(config.bridge.probe_texts))
        self.s_pad = self.s_tok.pad_token_id or self.bridge.stop_ids[-1]
        self.t_pad = self.t_tok.pad_token_id or self.t_tok.eos_token_id

        logger.info("Loading student (fp32 + autocast) on %s", self.device)
        self.student = load_causal_lm(config.model.student, torch.float32).to(self.device)
        if config.model.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
        logger.info("Loading teacher (bf16, frozen) on %s", self.teacher_device)
        self.teacher = load_causal_lm(config.model.teacher, torch.bfloat16).to(self.teacher_device)
        self.teacher.eval().requires_grad_(False)

        logger.info("Loading prompt pool from %s", config.data.prompts_path)
        pool = load_pool(config.data.prompts_path)
        self.train_rows, self.dev_rows = split_pool(pool, config.data.dev_size, config.train.seed)
        logger.info("Pool: %d train / %d dev", len(self.train_rows), len(self.dev_rows))
        self.reference_shots = load_reference_shots(config.data.shots_path) if config.data.shots_path else []
        self.renderer = PromptRenderer(
            self.train_rows,
            self.reference_shots,
            p_reference_shots=config.data.p_reference_shots,
            p_pool_shots=config.data.p_pool_shots,
            pool_shots_max_k=config.data.pool_shots_max_k,
            cot_fraction=config.sampling.cot_fraction,
            system_message=config.data.system_message or None,
            rng=self.rng,
        )

        self.opt = torch.optim.AdamW(
            self.student.parameters(), lr=config.train.learning_rate, weight_decay=0.0
        )
        # A metrics backend must never kill a training run: init/log are guarded.
        self.wandb = None
        if config.logging.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.logging.project,
                    name=config.logging.run_name or None,
                    config=dataclasses.asdict(config),
                    dir=config.train.output_dir,
                )
                self.wandb = wandb
            except Exception as e:  # noqa: BLE001 — degrade to console logging
                logger.warning("wandb init failed (%s); continuing without it", e)

    # ------------------------------------------------------------------ utils

    def _lr_at(self, step: int) -> float:
        train = self.config.train
        if step < train.warmup_steps:
            return train.learning_rate * (step + 1) / train.warmup_steps
        if train.lr_scheduler == "constant":
            return train.learning_rate
        t = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
        return train.learning_rate * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    @staticmethod
    def _encode_prompt(tok, messages) -> list[int]:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        bos = tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    @staticmethod
    def _right_pad(seqs: list[list[int]], pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(device), mask.to(device)

    def _left_pad(self, prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Left padding so all prompts end at the same position (for generate)."""
        T = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), T), self.s_pad, dtype=torch.long)
        mask = torch.zeros((len(prompts), T), dtype=torch.long)
        for j, p in enumerate(prompts):
            ids[j, T - len(p):] = torch.tensor(p, dtype=torch.long)
            mask[j, T - len(p):] = 1
        return ids, mask, T

    # -------------------------------------------------------------- generation

    @torch.no_grad()
    def _generate(self, prompt_ids: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Sample one completion per prompt (already replicated for group_size)."""
        sampling = self.config.sampling
        self.student.eval()
        completions: list[list[int]] = []
        for i in range(0, len(prompt_ids), sampling.gen_micro_seqs):
            chunk = prompt_ids[i : i + sampling.gen_micro_seqs]
            ids, mask, T = self._left_pad(chunk)
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16,
                                enabled=self.device != "cpu"):
                out = self.student.generate(
                    ids.to(self.device), attention_mask=mask.to(self.device),
                    do_sample=True, temperature=sampling.temperature, top_p=1.0, top_k=0,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=list(self.bridge.stop_ids),
                    pad_token_id=self.s_pad,
                )
            for j in range(len(chunk)):
                raw = out[j, T:].tolist()
                completions.append(self.bridge.clean_completion(raw))
        self.student.train()
        return completions

    # ----------------------------------------------------------------- scoring

    @staticmethod
    def _gather_logits(model, seqs, plens, lens, pad, device, autocast_dev=None):
        """Forward `seqs`, return lm_head logits only at completion positions.

        Position p predicts token p+1, so for a completion of length L starting
        at index P (= prompt length) we need hidden states at P-1 .. P+L-2.
        Full logits over a 128k vocab for every position would not fit; gathering
        hidden states first keeps memory at N_completion_tokens x vocab.

        Assumes the HF causal-LM layout (model.model backbone + model.lm_head),
        which holds for Llama/Qwen/Gemma-family architectures.
        """
        ids, mask = OPDTrainer._right_pad(seqs, pad, device)
        # grad-vs-no-grad is decided by the caller's context, not here
        ctx = (torch.autocast(autocast_dev, dtype=torch.bfloat16)
               if autocast_dev else contextlib.nullcontext())
        with ctx:
            h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            B, T, H = h.shape
            flat = []
            for i, (P, L) in enumerate(zip(plens, lens)):
                flat.extend(range(i * T + P - 1, i * T + P - 1 + L))
            h_sel = h.reshape(B * T, H)[torch.tensor(flat, device=device)]
            logits = model.lm_head(h_sel)  # (N, vocab)
        return logits

    def _loss_on_chunk(self, s_seqs, t_seqs, plens_s, plens_t, lens, targets_t):
        """Loss for a micro-batch of rollouts. Returns (loss, n_tokens, stats)."""
        train = self.config.train
        V = self.bridge.shared_vocab_size

        with torch.no_grad():
            t_logits = self._gather_logits(
                self.teacher, t_seqs, plens_t, lens, self.t_pad, self.teacher_device
            )
            logp_t = F.log_softmax(t_logits.float(), dim=-1).to(self.device)  # (N, V)

        s_logits = self._gather_logits(
            self.student, s_seqs, plens_s, lens, self.s_pad, self.device,
            autocast_dev=self.device.split(":")[0] if self.device != "cpu" else None,
        )
        logp_s_full = F.log_softmax(s_logits.float(), dim=-1)  # (N, student vocab)

        tgt = torch.tensor(targets_t, dtype=torch.long, device=self.device)  # (N,)

        # merge the student's end-of-turn mass into the teacher's terminator slot(s)
        p_full = logp_s_full.exp()
        p_shared = p_full[:, :V].clone()
        for s_id, t_id in self.bridge.swap.items():
            p_shared[:, t_id] += p_full[:, s_id]
        residual = 1.0 - p_shared.sum(-1)  # mass on unmapped student-only tokens, should be ~0
        logp_s = (p_shared + 1e-12).log()

        kl = (p_shared * (logp_s - logp_t)).sum(-1)  # (N,) full reverse KL
        sampled_kl = (logp_s.gather(-1, tgt[:, None]) - logp_t.gather(-1, tgt[:, None])).squeeze(-1)

        if train.loss_fn == "full_kl":
            loss = kl.sum()
        elif train.loss_fn == "sampled_rkl":
            # tinker-style: REINFORCE with per-token advantage = -sampled KL
            logp_s_tgt = logp_s.gather(-1, tgt[:, None]).squeeze(-1)
            loss = (logp_s_tgt * sampled_kl.detach()).sum()
        else:
            raise ValueError(train.loss_fn)

        stats = {
            "kl": kl.detach().mean().item(),
            "sampled_kl": sampled_kl.detach().mean().item(),
            "residual_mass": residual.detach().mean().item(),
        }
        return loss, len(targets_t), stats

    # ------------------------------------------------------------------- train

    def train(self):
        config = self.config
        t0 = time.time()
        for step in range(config.train.steps):
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)

            # 1. draw prompts, render for both models
            batch = []
            for _ in range(config.sampling.batch_prompts):
                messages, row, fast = self.renderer.sample()
                s_prompt = self._encode_prompt(self.s_tok, messages)
                t_prompt = self._encode_prompt(self.t_tok, messages)
                for _ in range(config.sampling.group_size):
                    batch.append({"s_prompt": s_prompt, "t_prompt": t_prompt,
                                  "row": row, "fast": fast})

            # 2. student samples completions (on-policy)
            fast_idx = [i for i, b in enumerate(batch) if b["fast"]]
            cot_idx = [i for i, b in enumerate(batch) if not b["fast"]]
            completions: dict[int, list[int]] = {}
            for idx, mnt in ((fast_idx, config.sampling.max_new_tokens),
                             (cot_idx, config.sampling.cot_max_new_tokens)):
                if idx:
                    comps = self._generate([batch[i]["s_prompt"] for i in idx], mnt)
                    completions.update(dict(zip(idx, comps)))

            rollouts = [(batch[i], completions[i]) for i in range(len(batch))
                        if len(completions[i]) > 0]
            if not rollouts:
                logger.warning("step %d: no non-empty completions, skipping", step)
                continue

            # 3+4. teacher scoring + reverse-KL update (micro-batched, grad accum)
            self.opt.zero_grad(set_to_none=True)
            n_total = sum(len(c) for _, c in rollouts)
            agg = {"kl": 0.0, "sampled_kl": 0.0, "residual_mass": 0.0}
            for i in range(0, len(rollouts), config.train.score_micro_seqs):
                chunk = rollouts[i : i + config.train.score_micro_seqs]
                s_seqs, t_seqs, plens_s, plens_t, lens, targets = [], [], [], [], [], []
                for b, comp in chunk:
                    comp_t = self.bridge.to_teacher(comp)
                    s_seqs.append(b["s_prompt"] + comp[:-1])
                    t_seqs.append(b["t_prompt"] + comp_t[:-1])
                    plens_s.append(len(b["s_prompt"]))
                    plens_t.append(len(b["t_prompt"]))
                    lens.append(len(comp))
                    targets.extend(comp_t)
                loss, n_tok, stats = self._loss_on_chunk(
                    s_seqs, t_seqs, plens_s, plens_t, lens, targets
                )
                (loss / n_total).backward()
                for k in agg:
                    agg[k] += stats[k] * n_tok / n_total

            torch.nn.utils.clip_grad_norm_(self.student.parameters(), config.train.max_grad_norm)
            self.opt.step()

            # ------------------------------------------------------- logging
            if step % config.logging.log_every == 0:
                mean_len = n_total / len(rollouts)
                fmt_ok = sum(
                    1 for b, comp in rollouts
                    if (letter := extract_letter(self.s_tok.decode(comp)))
                    and letter in {le for le, _ in b["row"]["options"]}
                ) / len(rollouts)
                logger.info(
                    "step %d | kl/tok=%.4f sampled_kl=%.4f len=%.1f fmt_ok=%.2f lr=%.2e (%.0fs)",
                    step, agg["kl"], agg["sampled_kl"], mean_len, fmt_ok,
                    self.opt.param_groups[0]["lr"], time.time() - t0,
                )
                self._track({"kl": agg["kl"], "sampled_kl": agg["sampled_kl"],
                             "residual_mass": agg["residual_mass"],
                             "completion_len": mean_len, "format_ok": fmt_ok,
                             "lr": self.opt.param_groups[0]["lr"]}, step)

            if config.train.eval_every and step and step % config.train.eval_every == 0:
                acc = self.eval_dev()
                logger.info("step %d | dev_acc=%.4f", step, acc)
                self._track({"dev_acc": acc}, step)

            if config.train.save_steps and step and step % config.train.save_steps == 0:
                self._save(f"step_{step}")

        acc = self.eval_dev()
        logger.info("final | dev_acc=%.4f", acc)
        self._save("final")

    # -------------------------------------------------------------------- eval

    @torch.no_grad()
    def eval_dev(self) -> float:
        """Greedy few-shot fast-mode accuracy on the held-out dev slice."""
        config = self.config
        rows = self.dev_rows[: config.train.eval_dev_samples]
        self.student.eval()
        correct = 0
        for i in range(0, len(rows), config.sampling.gen_micro_seqs):
            chunk = rows[i : i + config.sampling.gen_micro_seqs]
            prompts = [
                self._encode_prompt(
                    self.s_tok,
                    build_messages(r, few_shots=self.reference_shots, fast=True,
                                   system_message=config.data.system_message or None),
                )
                for r in chunk
            ]
            ids, mask, T = self._left_pad(prompts)
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16,
                                enabled=self.device != "cpu"):
                out = self.student.generate(
                    ids.to(self.device), attention_mask=mask.to(self.device),
                    do_sample=False, max_new_tokens=8,
                    eos_token_id=list(self.bridge.stop_ids), pad_token_id=self.s_pad,
                )
            for j, r in enumerate(chunk):
                text = self.s_tok.decode(
                    self.bridge.clean_completion(out[j, T:].tolist())[:-1]
                    or out[j, T:].tolist()
                )
                if extract_letter(text) == r["answer"]:
                    correct += 1
        self.student.train()
        return correct / max(1, len(rows))

    # ------------------------------------------------------------- bookkeeping

    def _track(self, metrics: dict, step: int):
        if self.wandb:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill training
                logger.warning("wandb.log failed (%s); disabling wandb", e)
                self.wandb = None

    def _save(self, name: str):
        path = os.path.join(self.config.train.output_dir, name)
        logger.info("Saving checkpoint -> %s", path)
        self.student.save_pretrained(path)
        self.s_tok.save_pretrained(path)
        with open(os.path.join(path, "opd_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Keep only the newest keep_checkpoints step_* dirs ("final" is exempt)."""
        keep = self.config.train.keep_checkpoints
        if keep <= 0:
            return
        steps = sorted(
            (d for d in os.listdir(self.config.train.output_dir) if d.startswith("step_")),
            key=lambda d: int(d.split("_")[1]),
        )
        for d in steps[:-keep]:
            victim = os.path.join(self.config.train.output_dir, d)
            logger.info("Pruning old checkpoint %s", victim)
            shutil.rmtree(victim, ignore_errors=True)


def main():
    from palingenesis.logging import setup_logging

    setup_logging(rank=0)
    config = OPDConfig.from_cli()
    for warning in config.validate():
        logger.warning(warning)
    OPDTrainer(config).train()


if __name__ == "__main__":
    main()
