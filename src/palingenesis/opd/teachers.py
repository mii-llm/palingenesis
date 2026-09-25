"""Teachers: score a student's completion in the teacher's own token space.

A teacher receives TeacherViews (its prompt + its tokens for the completion,
align.py) and returns, for every completion token, the log-probability of that
token and optionally its top-k alternatives:

  HFTeacher    frozen bf16 transformers model in the trainer's process. Can also
               keep the final hidden states of the completion so that full_rkl
               projects them to the full distribution slice by slice during the
               loss (log_probs), never storing [N, V] logits.
  VLLMTeacher  a vLLM server run prefill-only: /v1/completions with max_tokens=1
               and prompt_logprobs=k returns the top-k and the actual token's
               log-prob at every prompt position, at temperature 1.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from palingenesis.logits import final_hidden_states, output_head, verify_output_head
from palingenesis.opd.align import TeacherView
from palingenesis.opd.rollout import VLLMServer

logger = logging.getLogger(__name__)


@dataclass
class TeacherScores:
    token_lp: Tensor | None         # [n] log-prob of each completion token (CPU, fp32); None with hidden only
    topk_ids: Tensor | None = None  # [n, k] (CPU)
    topk_lp: Tensor | None = None   # [n, k] (CPU, fp32)
    hidden: Tensor | None = None    # [n, H] final hidden states (teacher's device), for full_rkl
    # rs_kd: tokens drawn from the teacher's distribution per position [n, R] (teacher's device),
    # their importance weights (each row sums to 1) and the teacher's log-probs of them
    sample_ids: Tensor | None = None
    sample_weights: Tensor | None = None
    sample_lp: Tensor | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_causal_lm(name: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)


def end_of_turn_id(tok, chat_template_kwargs: dict | None = None) -> int:
    """The token the chat template closes an assistant turn with (<|im_end|>, <|eot_id|>, ...)."""
    marker = "\x00MARK\x00"
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": marker}],
                                   tokenize=False, **(chat_template_kwargs or {}))
    special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
    for token in tok.encode(text[text.rindex(marker) + len(marker):], add_special_tokens=False):
        if token in special:
            return token
    if tok.eos_token_id is None:
        raise ValueError(f"cannot tell {tok.name_or_path}'s end-of-turn token from its chat template")
    return tok.eos_token_id


def right_pad(seqs: list[list[int]], pad: int, device) -> tuple[Tensor, Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def completion_positions(prompt_lens: list[int], completion_lens: list[int], width: int) -> Tensor:
    """[B, width] mask of the positions whose next-token prediction is a completion token:
    P - 1 .. P + n - 2 for a prompt of length P and n completion tokens."""
    mask = torch.zeros((len(prompt_lens), width), dtype=torch.bool)
    for i, (p, n) in enumerate(zip(prompt_lens, completion_lens)):
        mask[i, p - 1: p - 1 + n] = True
    return mask


def sample_teacher(logp: Tensor, rounds: int, temperature: float,
                   generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Random Sampling KD's sparse targets (arXiv 2503.16870) from teacher log-probs [R, V].

    Draws `rounds` tokens per row, with replacement, from the proposal r = p^temperature
    (renormalized); each draw's weight is the likelihood ratio p / r, normalized over the
    row. The weighted draws estimate p without bias (temperature 1: plain counts / rounds).
    Returns ids [R, rounds], weights [R, rounds] (rows sum to 1), and log p of the ids.
    """
    if temperature == 1.0:
        proposal_lp = logp
    else:
        proposal_lp = torch.log_softmax(logp * temperature, -1)
    ids = torch.multinomial(proposal_lp.exp(), rounds, replacement=True, generator=generator)
    lp = logp.gather(1, ids)
    return ids, torch.softmax(lp - proposal_lp.gather(1, ids), 1), lp       # ratios, normalized in log space


class HFTeacher:
    """A frozen transformers model scoring in bf16 on `device`.

    With `offload`, the model waits on CPU between scoring calls (the output head
    stays on the device for full_rkl's projections during the loss).
    """

    def __init__(self, model: str, device: str, offload: bool = False, seed: int = 0):
        self.device = device
        self.offload = offload
        logger.info("Loading teacher %s (bf16, frozen) on %s", model, device)
        self.model = load_causal_lm(model, torch.bfloat16).to(device).eval().requires_grad_(False)
        self.head = output_head(self.model)
        verify_output_head(self.model, self.head)
        self.generator = torch.Generator(device=device).manual_seed(seed)     # rs_kd's draws
        if offload:
            self.head = copy.deepcopy(self.head)
            self.model.to("cpu")

    @torch.no_grad()
    def score(self, views: list[TeacherView], top_k: int = 0, keep_hidden: bool = False,
              micro_seqs: int = 16, sample_rounds: int = 0, sample_temperature: float = 1.0) -> list[TeacherScores]:
        """Scores of each view's completion tokens: log-probs, plus top-k, hidden states or
        `sample_rounds` tokens drawn from the teacher's distribution (rs_kd), as asked."""
        if self.offload:
            self.model.to(self.device)
        results: list[TeacherScores | None] = [None] * len(views)
        order = sorted(range(len(views)), key=lambda i: len(views[i].input_ids))
        rows = max(1, 2**27 // self.head.weight.shape[0])
        for start in range(0, len(order), micro_seqs):
            chunk = order[start:start + micro_seqs]
            ids, mask = right_pad([views[i].input_ids[:-1] for i in chunk], 0, self.device)
            positions = completion_positions([views[i].prompt_len for i in chunk],
                                             [views[i].completion_len for i in chunk], ids.shape[1])
            with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
                # right-padded rows: no mask (padding after a row cannot reach it; flash attention)
                hidden = final_hidden_states(self.model, ids, None)[positions.to(ids.device)]
            if keep_hidden and not top_k:     # full_rkl projects the hidden states in its loss: nothing else needed
                offset = 0
                for i in chunk:
                    n = views[i].completion_len
                    results[i] = TeacherScores(None, hidden=hidden[offset:offset + n])
                    offset += n
                continue
            with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
                targets = torch.tensor([t for i in chunk for t in views[i].input_ids[views[i].prompt_len:]],
                                       device=ids.device)
                token_lp, topk_ids, topk_lp, samples = [], [], [], []
                for a in range(0, hidden.shape[0], rows):
                    logp = torch.log_softmax(self.head(hidden[a:a + rows]).float(), -1)
                    token_lp.append(logp.gather(1, targets[a:a + rows, None]).squeeze(1))
                    if top_k:
                        top = logp.topk(top_k, -1)
                        topk_lp.append(top.values)
                        topk_ids.append(top.indices)
                    if sample_rounds:
                        samples.append(sample_teacher(logp, sample_rounds, sample_temperature, self.generator))
            token_lp = torch.cat(token_lp).cpu()
            topk_ids = torch.cat(topk_ids).cpu() if top_k else None
            topk_lp = torch.cat(topk_lp).cpu() if top_k else None
            if sample_rounds:
                sample_ids, sample_w, sample_lp = (torch.cat(x) for x in zip(*samples))
            offset = 0
            for i in chunk:
                n = views[i].completion_len
                sl = slice(offset, offset + n)
                results[i] = TeacherScores(token_lp[sl], topk_ids[sl] if top_k else None,
                                           topk_lp[sl] if top_k else None,
                                           hidden[sl].clone() if keep_hidden else None,
                                           *((sample_ids[sl], sample_w[sl], sample_lp[sl]) if sample_rounds else ()))
                offset += n
        if self.offload:
            self.model.to("cpu")
        return results

    def log_probs(self, hidden: Tensor, size: int) -> Tensor:
        """Log-softmax of the teacher's logits for hidden-state rows, first `size` ids."""
        with torch.no_grad(), torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda):
            return torch.log_softmax(self.head(hidden).float(), -1)[:, :size]


class VLLMTeacher:
    """Prefill-only scoring on a vLLM server (launched with --max-logprobs >= top_k)."""

    def __init__(self, server: VLLMServer):
        self.server = server

    def score(self, views: list[TeacherView], top_k: int = 0, keep_hidden: bool = False,
              micro_seqs: int = 16, sample_rounds: int = 0, sample_temperature: float = 1.0) -> list[TeacherScores]:
        if sample_rounds:
            raise ValueError("a vLLM teacher returns only its top-k: rs_kd needs an hf teacher")
        if keep_hidden:
            raise ValueError("a vLLM teacher has no hidden states for full_rkl; use topk_kl or sampled_rkl")
        choices = self.server.complete([v.input_ids for v in views], max_tokens=1, temperature=1.0,
                                       prompt_logprobs=max(top_k, 1))
        return [self._parse(view, choice["prompt_logprobs"], top_k) for view, choice in zip(views, choices)]

    @staticmethod
    def _parse(view: TeacherView, prompt_logprobs: list, top_k: int) -> TeacherScores:
        token_lp, topk_ids, topk_lp = [], [], []
        for pos in range(view.prompt_len, len(view.input_ids)):
            entries = prompt_logprobs[pos]
            actual = entries.get(str(view.input_ids[pos]))
            if actual is None:
                raise RuntimeError(f"vLLM returned no log-prob for the actual token at position {pos}")
            token_lp.append(_finite(actual["logprob"]))
            if top_k:
                top = sorted(entries.items(), key=lambda kv: kv[1]["rank"])[:top_k]
                pad = top_k - len(top)
                topk_ids.append([int(t) for t, _ in top] + [0] * pad)
                topk_lp.append([_finite(e["logprob"]) for _, e in top] + [-math.inf] * pad)
        return TeacherScores(torch.tensor(token_lp, dtype=torch.float32),
                             torch.tensor(topk_ids, dtype=torch.long).view(-1, top_k) if top_k else None,
                             torch.tensor(topk_lp, dtype=torch.float32).view(-1, top_k) if top_k else None)


def _finite(logprob: float) -> float:
    """vLLM reports impossible tokens as NaN/-9999; keep them very unlikely but finite."""
    return max(-1e4, logprob) if logprob == logprob else -1e4
