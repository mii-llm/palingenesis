"""Direct Preference Optimisation (DPO) and its common variants.

Data
----
Rows use the standard conversational preference format, with an explicit or an
implicit prompt:

    {"prompt":   [{"role": "user", "content": "..."}],
     "chosen":   [{"role": "assistant", "content": "...", "reasoning": "..."}],
     "rejected": [{"role": "assistant", "content": "..."}],
     "chat_template_kwargs": {"enable_thinking": false}}           # optional, per row

    {"chosen": [...full conversation...], "rejected": [...full conversation...]}

`chosen` and `rejected` may each be one assistant message, several messages (a
multi-turn continuation) or a plain string (wrapped as one assistant turn). An
assistant turn carries its reasoning in `reasoning` (the OpenAI/vLLM field),
the legacy `reasoning_content`, or inline as a `<think>...</think>` block.
Like vLLM, normalize_messages exposes the trace to the template under both
keys, so templates whose Jinja still reads `reasoning_content` (e.g. Qwen3.5)
render it.

Each side is rendered as a full conversation by the SAME code path as SFT
(`ChatDataset`), with the model's own chat template, so the scored tokens are
exactly the assistant tokens SFT would train on:

  * `data.last_turn_only` scores only the final assistant turn — the usual DPO
    target when the prompt itself contains earlier assistant turns;
  * `data.train_on_reasoning` decides whether reasoning tokens count;
  * per-row `chat_template_kwargs` reach every render, fallbacks included, so a
    dataset can mix thinking and non-thinking rows.

The chosen answer is never truncated: a pair whose chosen side does not fit
`max_seq_length` is dropped. The rejected side is truncated (start kept) when
`dpo.truncate_rejected` is on, which suits degenerate rejected answers such as
repetition loops; otherwise such pairs are dropped too.

Loss
----
Every supported objective is a function of per-sequence scores

    s_b = sum_t a[b, t] * log pi(y_t | y_<t)

where `a` is the scored-token mask, optionally LD-DPO-weighted (1 on the length
both answers share, `ld_alpha` on the longer answer's tail). By the chain rule

    dL / d log pi(y_t) = (dL / d s_b) * a[b, t]

and dL/ds_b is available after a cheap no-grad pass. So the loss runs in two
chunked passes over the sequence, reusing the chunked-CE machinery:

  1. no grad: per-token log-probs, chunk by chunk (never the full [B, S, V]
     logits), for the policy and the reference;
  2. the DPO loss on the small score tensors, differentiated with autograd to
     get per-sequence weights;
  3. with grad: chunk by chunk, backpropagate sum(weight * log pi) and bridge the
     accumulated hidden-state gradient into the backbone's graph.

Peak memory is the chunked-SFT peak, independent of the vocabulary size; no
full-vocabulary quantity (logits, entropy) is ever held for the whole sequence.

Supported `loss_type`s (definitions as in Hugging Face TRL's DPOTrainer):
    sigmoid       DPO, arXiv:2305.18290                 -log sigma(beta * delta)
    hinge         SLiC-HF, arXiv:2305.10425             relu(1 - beta * delta)
    ipo           IPO, arXiv:2310.12036 (per token)     (delta_avg - 1/(2 beta))^2
    robust        rDPO, arXiv:2403.00409 (label_smoothing = flip rate)
    sigmoid_norm  length-normalised DPO                 -log sigma(beta * delta_avg)
where delta = (policy - reference) log-ratio of chosen minus that of rejected.
`sft_weight` adds the mean NLL over chosen tokens (RPO, arXiv:2404.19733).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from palingenesis.data import IGNORE_INDEX, ChatDataset, _collate_fn, _shard_then_shuffle
from palingenesis.loss import _BackwardBridge, shift_labels

logger = logging.getLogger(__name__)

LOSS_TYPES = ("sigmoid", "hinge", "ipo", "robust", "sigmoid_norm")


# ==============================================================================
# DATA
# ==============================================================================


def _as_messages(value: Any, role: str = "assistant") -> list[dict]:
    """A completion as a message list: a string becomes one `role` turn."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if isinstance(value, dict):
        return [value]
    return list(value)


def _to_conversations(
    example: dict[str, Any],
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
) -> tuple[list[dict], list[dict]] | None:
    """(chosen conversation, rejected conversation), or None if the row is unusable.

    Explicit prompt: prompt + chosen / prompt + rejected. Implicit prompt: chosen
    and rejected are already full conversations. A string prompt is one user turn.
    """
    chosen = _as_messages(example.get(chosen_field))
    rejected = _as_messages(example.get(rejected_field))
    if not chosen or not rejected:
        return None
    prompt = _as_messages(example.get(prompt_field), role="user")
    if prompt and _starts_with(chosen, prompt) and _starts_with(rejected, prompt):
        # Both completions already are full conversations that begin with the
        # prompt (e.g. HuggingFaceH4/ultrafeedback_binarized): prepending it again
        # would repeat the user turn.
        return chosen, rejected
    return prompt + chosen, prompt + rejected


def _starts_with(conversation: list[dict], prefix: list[dict]) -> bool:
    def key(message: dict) -> tuple:
        content = message.get("content")
        return message.get("role"), content.strip() if isinstance(content, str) else content

    return len(conversation) > len(prefix) and all(
        isinstance(m, dict) and isinstance(p, dict) and key(m) == key(p) for m, p in zip(conversation, prefix)
    )


class PreferenceDataset(IterableDataset):
    """Preference pairs tokenised exactly as SFT conversations are.

    Yields {"chosen": {input_ids, attention_mask, labels}, "rejected": {...}}.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length: int,
        *,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        last_turn_only: bool = False,
        train_on_reasoning: bool = True,
        truncate_rejected: bool = True,
        tools_field: str = "tools",
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer: int = 0,
        shuffle_seed: int = 0,
    ):
        self.dataset = dataset
        self.tools_field = tools_field
        self.max_seq_length = max_seq_length
        self.prompt_field = prompt_field
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.truncate_rejected = truncate_rejected
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        # One renderer per side. The rejected one is allowed a very long render so
        # it can be measured and truncated here, instead of being cut by the
        # tokenizer's truncation, which would silently truncate chosen too.
        common = dict(messages_field="messages", last_turn_only=last_turn_only,
                      train_on_reasoning=train_on_reasoning)
        self._chosen = ChatDataset([], tokenizer, max_seq_length=10**7, **common)
        self._rejected = ChatDataset([], tokenizer, max_seq_length=10**7, **common)
        self.stats = {"pairs": 0, "dropped_unusable": 0, "dropped_chosen_too_long": 0,
                      "dropped_rejected_too_long": 0, "dropped_identical": 0, "rejected_truncated": 0}

    def __iter__(self):
        dataset = _shard_then_shuffle(self.dataset, self.rank, self.world_size,
                                      self.shuffle_buffer, self.shuffle_seed)
        for example in dataset:
            pair = self.process(example)
            if pair is not None:
                yield pair

    def process(self, example: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]] | None:
        convs = _to_conversations(example, self.prompt_field, self.chosen_field, self.rejected_field)
        if convs is None:
            self.stats["dropped_unusable"] += 1
            return None
        kwargs = example.get("chat_template_kwargs") or {}
        if isinstance(kwargs, str):
            kwargs = json.loads(kwargs) if kwargs.strip() else {}
        tools = example.get(self.tools_field)
        chosen = self._chosen._process({"messages": convs[0], "chat_template_kwargs": kwargs, "tools": tools})
        rejected = self._rejected._process({"messages": convs[1], "chat_template_kwargs": kwargs, "tools": tools})
        if chosen is None or rejected is None:
            self.stats["dropped_unusable"] += 1
            return None
        if chosen["input_ids"].numel() > self.max_seq_length:
            self.stats["dropped_chosen_too_long"] += 1  # never truncate the reference
            return None
        if rejected["input_ids"].numel() > self.max_seq_length:
            if not self.truncate_rejected:
                self.stats["dropped_rejected_too_long"] += 1
                return None
            rejected = {k: v[: self.max_seq_length] for k, v in rejected.items()}
            if (rejected["labels"] != IGNORE_INDEX).sum() == 0:
                self.stats["dropped_rejected_too_long"] += 1  # nothing scored survives the cut
                return None
            self.stats["rejected_truncated"] += 1
        if (chosen["input_ids"].numel() == rejected["input_ids"].numel()
                and torch.equal(chosen["input_ids"], rejected["input_ids"])):
            self.stats["dropped_identical"] += 1  # no preference to learn
            return None
        self.stats["pairs"] += 1
        return {"chosen": chosen, "rejected": rejected}


def collate_preferences(batch: list[dict], pad_id: int, pad_to_multiple: int = 1) -> dict[str, torch.Tensor]:
    """Chosen rows first, then rejected, padded together: [2P, S]. One forward
    pass then scores both sides, and row b pairs with row b + P."""
    rows = [pair["chosen"] for pair in batch] + [pair["rejected"] for pair in batch]
    out = _collate_fn(rows, pad_id, pad_to_multiple)
    out["num_pairs"] = torch.tensor(len(batch))
    return out


def build_preference_dataloader(
    dataset,
    tokenizer,
    data_config,
    dpo_config,
    rank: int,
    world_size: int,
    batch_size: int,
    streaming_shuffle_buffer: int = 0,
    shuffle_seed: int = 0,
):
    """DataLoader over preference pairs: `batch_size` pairs per micro-batch,
    collated as [2 * batch_size, S] (chosen rows first)."""
    from torch.utils.data import DataLoader

    pairs = PreferenceDataset(
        dataset,
        tokenizer,
        data_config.max_seq_length,
        prompt_field=dpo_config.prompt_field,
        chosen_field=dpo_config.chosen_field,
        rejected_field=dpo_config.rejected_field,
        last_turn_only=data_config.last_turn_only,
        train_on_reasoning=data_config.train_on_reasoning,
        truncate_rejected=dpo_config.truncate_rejected,
        tools_field=getattr(data_config, "tools_field", "tools"),
        rank=rank,
        world_size=world_size,
        shuffle_buffer=streaming_shuffle_buffer,
        shuffle_seed=shuffle_seed,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    num_workers = data_config.num_workers
    return DataLoader(
        pairs,
        batch_size=batch_size,
        collate_fn=lambda b: collate_preferences(b, pad_id, pad_to_multiple=64),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def disable_dropout(model: nn.Module) -> int:
    """Set every dropout probability to 0 (on by default, `dpo.disable_dropout`).

    With dropout the policy's log-probs are noisy while the reference's (eval mode)
    are not, so the implicit reward carries noise and, at initialisation, the
    policy does not even equal the reference. Returns the number of modules changed."""
    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p > 0:
            module.p = 0.0
            changed += 1
    return changed


# ==============================================================================
# LOG-PROBABILITIES (chunked, never the full [B, S, V] logits)
# ==============================================================================


def _loss_dtype(hidden: torch.Tensor) -> torch.dtype:
    """Log-softmax precision: at least float32 (bf16 training), and float64 when
    the model itself runs in float64 (used by the exactness tests)."""
    return torch.promote_types(hidden.dtype, torch.float32)


@torch.no_grad()
def token_logps(
    hidden: torch.Tensor, shifted_labels: torch.Tensor, lm_head: nn.Module, num_chunks: int = 1
) -> torch.Tensor:
    """log pi(y_t | y_<t) at every scored position, 0 elsewhere. [B, S] float32.

    `shifted_labels` must already be shifted (see loss.shift_labels): position t
    holds the token that logits at t predict."""
    dtype = _loss_dtype(hidden)
    out = torch.zeros(shifted_labels.shape, dtype=dtype, device=hidden.device)
    offset = 0
    for h, lab in zip(torch.chunk(hidden, num_chunks, dim=1), torch.chunk(shifted_labels, num_chunks, dim=1)):
        width = h.shape[1]
        logits = lm_head(h).to(dtype)
        valid = lab != IGNORE_INDEX
        picked = logits.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        out[:, offset: offset + width] = torch.where(valid, picked - torch.logsumexp(logits, dim=-1), 0.0)
        offset += width
        del logits
    return out


def score_weights(shifted_labels: torch.Tensor, num_pairs: int, ld_alpha: float | None = None) -> torch.Tensor:
    """a[b, t]: how much token t counts towards sequence b's score.

    Without LD-DPO this is the scored-token mask. With it (arXiv:2409.06411),
    tokens up to the length both answers of a pair share count fully and the
    longer answer's tail counts `ld_alpha`."""
    mask = (shifted_labels != IGNORE_INDEX).float()
    if ld_alpha is None:
        return mask
    position = mask.cumsum(dim=1)                      # 1-based index among scored tokens
    lengths = mask.sum(dim=1)
    shared = torch.minimum(lengths[:num_pairs], lengths[num_pairs:])
    shared = torch.cat([shared, shared]).unsqueeze(1)
    return torch.where(position <= shared, mask, ld_alpha * mask)


# ==============================================================================
# LOSS
# ==============================================================================


def preference_loss(
    policy: torch.Tensor,
    reference: torch.Tensor,
    lengths: torch.Tensor,
    *,
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-pair loss [P] from per-sequence scores [2P] (chosen first)."""
    policy_c, policy_r = policy.chunk(2)
    ref_c, ref_r = reference.chunk(2)
    delta = (policy_c - ref_c) - (policy_r - ref_r)
    if loss_type == "sigmoid":
        return -F.logsigmoid(beta * delta)
    if loss_type == "hinge":
        return torch.relu(1 - beta * delta)
    if loss_type == "robust":
        # Unbiased estimator under label flips (arXiv:2403.00409): the flipped term
        # is SUBTRACTED.
        clean = -(1 - label_smoothing) * F.logsigmoid(beta * delta)
        flipped = -label_smoothing * F.logsigmoid(-beta * delta)
        return (clean - flipped) / (1 - 2 * label_smoothing)
    if loss_type in ("ipo", "sigmoid_norm"):
        len_c, len_r = lengths.clamp(min=1.0).chunk(2)
        delta_avg = (policy_c - ref_c) / len_c - (policy_r - ref_r) / len_r
        if loss_type == "ipo":
            return (delta_avg - 1 / (2 * beta)) ** 2
        return -F.logsigmoid(beta * delta_avg)
    raise ValueError(f"unknown dpo.loss_type {loss_type!r}; expected one of {LOSS_TYPES}")


@dataclass
class PreferenceStep:
    """Outcome of one preference micro-step: a loss tensor wired into the
    backbone's graph (call .backward() as for SFT) plus metrics to log."""

    loss: torch.Tensor
    metrics: dict[str, float]


def preference_step(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    ref_logps: torch.Tensor,
    num_pairs: int,
    *,
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    ld_alpha: float | None = None,
    sft_weight: float = 0.0,
    pair_denom: float | None = None,
    chosen_token_denom: float | None = None,
    num_chunks: int = 1,
) -> PreferenceStep:
    """The DPO loss for one micro-batch, via two chunked passes (see module doc).

    hidden:     [2P, S, D] policy hidden states (graph attached), chosen rows first
    labels:     [2P, S] UNshifted labels, as the data pipeline emits them
    ref_logps:  [2P, S] reference per-token log-probs from `token_logps`
    pair_denom: normaliser of the summed pair losses; defaults to P. Set it to
                (global pairs) * (grad-accum steps) so micro-batches and ranks
                average exactly as SFT's valid-token denominator does.
    chosen_token_denom: normaliser of the SFT term (defaults to local chosen tokens).
    """
    shifted = shift_labels(labels)
    a = score_weights(shifted, num_pairs, ld_alpha)
    lengths = a.sum(dim=1)
    policy_logps = token_logps(hidden.detach(), shifted, lm_head, num_chunks)

    pair_denom = float(pair_denom if pair_denom is not None else num_pairs)
    scores = (policy_logps * a).sum(dim=1).requires_grad_(True)
    ref_scores = (ref_logps * a).sum(dim=1)
    per_pair = preference_loss(scores, ref_scores, lengths, loss_type=loss_type, beta=beta,
                               label_smoothing=label_smoothing)
    dpo_loss = per_pair.sum() / pair_denom
    (grad_scores,) = torch.autograd.grad(dpo_loss, scores)
    weights = grad_scores.unsqueeze(1) * a            # dL/dlog pi(y_t) for every token

    chosen_mask = torch.zeros_like(a)
    chosen_mask[:num_pairs] = (shifted[:num_pairs] != IGNORE_INDEX).float()
    sft_value = torch.zeros((), device=hidden.device)
    if sft_weight:
        denom = float(chosen_token_denom if chosen_token_denom is not None
                      else max(chosen_mask.sum().item(), 1.0))
        sft_value = -(policy_logps * chosen_mask).sum() / denom
        weights = weights - (sft_weight / denom) * chosen_mask   # d(-mean logp)/dlog pi

    total = dpo_loss.detach() + sft_weight * sft_value
    loss = _weighted_logp_backward(hidden, shifted, lm_head, weights, total, num_chunks)

    with torch.no_grad():
        rewards = beta * (scores.detach() - ref_scores)
        r_c, r_r = rewards.chunk(2)
        logps_c, logps_r = scores.detach().chunk(2)
        metrics = {
            "dpo/loss": float(dpo_loss) * pair_denom / max(num_pairs, 1),
            "rewards/chosen": float(r_c.mean()),
            "rewards/rejected": float(r_r.mean()),
            "rewards/margins": float((r_c - r_r).mean()),
            "rewards/accuracies": float((r_c > r_r).float().mean()),
            "logps/chosen": float(logps_c.mean()),
            "logps/rejected": float(logps_r.mean()),
        }
        if sft_weight:
            local_chosen = max(chosen_mask.sum().item(), 1.0)
            metrics["dpo/sft_nll"] = float(-(policy_logps * chosen_mask).sum() / local_chosen)
    return PreferenceStep(loss=loss, metrics=metrics)


def _weighted_logp_backward(
    hidden: torch.Tensor,
    shifted_labels: torch.Tensor,
    lm_head: nn.Module,
    weights: torch.Tensor,
    loss_value: torch.Tensor,
    num_chunks: int,
) -> torch.Tensor:
    """Backpropagate sum(weights * log pi) chunk by chunk and bridge the summed
    hidden-state gradient into the backbone's graph (as chunked CE does).

    Because weights[b, t] = dL/dlog pi(y_t), the gradient this produces is dL/dθ.
    The returned tensor carries `loss_value` for logging."""
    if not hidden.requires_grad:
        return loss_value
    dtype = _loss_dtype(hidden)
    grad_buffer = torch.zeros_like(hidden, dtype=dtype)
    offset = 0
    for h, lab, w in zip(torch.chunk(hidden.detach(), num_chunks, dim=1),
                         torch.chunk(shifted_labels, num_chunks, dim=1),
                         torch.chunk(weights, num_chunks, dim=1)):
        width = h.shape[1]
        h = h.contiguous().requires_grad_(True)
        logits = lm_head(h).to(dtype)
        picked = logits.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        logp = picked - torch.logsumexp(logits, dim=-1)
        surrogate = (logp * w.to(dtype) * (lab != IGNORE_INDEX)).sum()
        surrogate.backward()
        grad_buffer[:, offset: offset + width] = h.grad.to(dtype)
        offset += width
        del logits
    return _BackwardBridge.apply(hidden, grad_buffer.to(hidden.dtype), loss_value)


@torch.no_grad()
def reference_logps(
    ref_model: nn.Module,
    get_hidden,
    get_lm_head,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    num_chunks: int = 1,
) -> torch.Tensor:
    """Per-token reference log-probs [2P, S] for a collated preference batch."""
    hidden = get_hidden(ref_model, input_ids, attention_mask, None)
    return token_logps(hidden, shift_labels(labels), get_lm_head(ref_model), num_chunks)


# ==============================================================================
# EVALUATION
# ==============================================================================


class PreferenceEvaluator:
    """DPO metrics on a fixed set of held-out pairs.

    The reference model never changes, so its sequence scores are computed once,
    on the first call, and reused; later evaluations run the policy only."""

    def __init__(self, batches: list[dict[str, torch.Tensor]], *, loss_type: str, beta: float,
                 label_smoothing: float, ld_alpha: float | None, num_chunks_for):
        self.batches = batches
        self.loss_type = loss_type
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ld_alpha = ld_alpha
        self.num_chunks_for = num_chunks_for   # tokens -> loss chunks
        self._ref_scores: list[torch.Tensor] | None = None

    def _scores(self, model, get_hidden, lm_head, batch, device):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        shifted = shift_labels(batch["labels"].to(device))
        num_pairs = ids.shape[0] // 2
        hidden = get_hidden(model, ids, mask, None)
        logps = token_logps(hidden, shifted, lm_head, self.num_chunks_for(ids.numel()))
        a = score_weights(shifted, num_pairs, self.ld_alpha)
        return (logps * a).sum(dim=1), a.sum(dim=1)

    @torch.no_grad()
    def evaluate(self, model, ref_model, get_hidden, get_lm_head, device, dtype, autocast: bool) -> dict[str, float]:
        was_training = model.training
        model.eval()
        with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast):
            if self._ref_scores is None:
                self._ref_scores = [self._scores(ref_model, get_hidden, get_lm_head(ref_model), b, device)[0]
                                    for b in self.batches]
            losses, chosen, rejected, correct = [], [], [], []
            for batch, ref in zip(self.batches, self._ref_scores):
                policy, lengths = self._scores(model, get_hidden, get_lm_head(model), batch, device)
                losses.append(preference_loss(policy, ref, lengths, loss_type=self.loss_type, beta=self.beta,
                                              label_smoothing=self.label_smoothing))
                r_c, r_r = (self.beta * (policy - ref)).chunk(2)
                chosen.append(r_c)
                rejected.append(r_r)
                correct.append((r_c > r_r).float())
        if was_training:
            model.train()
        r_c, r_r = torch.cat(chosen), torch.cat(rejected)
        return {
            "eval/loss": float(torch.cat(losses).mean()),
            "eval/rewards/chosen": float(r_c.mean()),
            "eval/rewards/rejected": float(r_r.mean()),
            "eval/rewards/margins": float((r_c - r_r).mean()),
            "eval/rewards/accuracies": float(torch.cat(correct).mean()),
        }
