"""Loglikelihood scoring of multiple choice: the context is prefilled once, the choices are leaves of a token trie.

A multiple-choice item asks for log p(choice | context) of every choice. Harnesses send one request per choice,
each a full prefill of context + choice with the log-probs of every prompt position (vLLM's prompt_logprobs). Here
the choices of an item are folded into a trie over their tokens and each trie node that branches or continues is
one next-token query on the generation path:

    context ──► " A" | " B" | " C" | " D"                 one query: 1 prefill, 1 decoded token (MMLU, ARC letters)
    context ──► " Paris" ─► "," ...                       a node per shared prefix: the context's KV blocks come
            └─► " Lon" ─► "don"                           from the engine's prefix cache after the first query

A query (peek) is a 1-token sampling request at temperature 1 that returns the sampler's `top` most likely next
tokens with their log-probs. Wanted tokens outside the top are read with a second request restricted to
{anchor} + the missing tokens (Sampling.allowed): the engine renormalises over that set, and the anchor's known
log-prob restores the scale, log p(t) = log p_S(t) - log p_S(anchor) + log p(anchor). So every value is an exact
log-prob of the model, whatever the rank of the token, and an item with single-token choices costs one request.

Engines whose model is in this process (the hf backend) are scored with one padded forward per batch instead
(forward_loglikelihoods): the reference the trie path is tested against.

Tokenisation follows lm-evaluation-harness (encode_pair): trailing spaces of the context move to the continuation
and the continuation's tokens are those of context + continuation after the context's.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

NEG_INF = float("-inf")


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """(context ids, continuation ids) as lm-evaluation-harness tokenizes a loglikelihood request."""
    spaces = len(context) - len(context.rstrip())
    if spaces > 0:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    head = tokenizer.encode(context, add_special_tokens=False)
    return head, whole[len(head) :]


def encode_pairs(tokenizer, contexts: list[str], continuations: list[list[str]]) -> list[tuple[list[int], list]]:
    """encode_pair for every (context, choice) of every item with two batched tokenizer calls:
    [(context ids, [choice ids, ...]), ...]."""
    heads, wholes, shape = [], [], []
    for context, choices in zip(contexts, continuations):
        spaces = len(context) - len(context.rstrip())
        head = context[:-spaces] if spaces else context
        heads.append(head)
        wholes += [context + choice for choice in choices]
        shape.append(len(choices))
    if not heads:
        return []
    head_ids = tokenizer(heads, add_special_tokens=False)["input_ids"]
    whole_ids = tokenizer(wholes, add_special_tokens=False)["input_ids"] if wholes else []
    out, at = [], 0
    for ids, n in zip(head_ids, shape):
        out.append((list(ids), [list(w[len(ids) :]) for w in whole_ids[at : at + n]]))
        at += n
    return out


@dataclass
class ChoiceScores:
    loglikelihoods: list[float]
    greedy: list[bool]  # whether every token of the choice is the model's most likely next token
    requests: int = 0  # engine requests this item took
    tokens: list[int] = field(default_factory=list)  # tokens of each choice


Peek = Callable[[list[int], tuple[int, ...]], Awaitable[dict[int, float]]]


async def trie_loglikelihoods(peek: Peek, context: list[int], choices: list[list[int]], top: int) -> ChoiceScores:
    """log p(choice | context) for every choice through next-token queries on the trie of the choices.

    peek(prompt, allowed) -> {token: log-prob}: the most likely next tokens after `prompt` (at most `top`), over
    the whole vocabulary (allowed = ()) or renormalised over `allowed`."""
    wanted: dict[tuple[int, ...], set[int]] = {}  # trie node (the path from the context) -> next tokens needed
    for choice in choices:
        for depth, token in enumerate(choice):
            wanted.setdefault(tuple(choice[:depth]), set()).add(token)
    requests = 0

    async def node(path: tuple[int, ...]) -> tuple[dict[int, float], int]:
        nonlocal requests
        prompt = context + list(path)
        requests += 1
        known = dict(await peek(prompt, ()))
        if not known:
            raise RuntimeError("the engine returned no next-token log-probs (Sampling.candidates is not supported)")
        best = max(known, key=known.get)
        missing = sorted(t for t in wanted[path] if t not in known)
        size = max(1, top - 1)
        for start in range(0, len(missing), size):
            part = missing[start : start + size]
            requests += 1
            got = await peek(prompt, (best, *part))
            if best not in got:  # the anchor is the most likely token of the set: it is always reported
                raise RuntimeError("a restricted next-token request did not report its most likely token")
            shift = known[best] - got[best]
            for token in part:
                known[token] = got[token] + shift if token in got else NEG_INF
        return known, best

    paths = list(wanted)
    if not paths:
        return ChoiceScores([0.0] * len(choices), [True] * len(choices), 0, [0] * len(choices))
    # the root first: its prefill puts the context into the prefix cache, which the deeper nodes then reuse
    results = {(): await node(())} if () in wanted else {}
    rest = [p for p in paths if p != ()]
    for path, result in zip(rest, await asyncio.gather(*(node(p) for p in rest))):
        results[path] = result
    scores, greedy = [], []
    for choice in choices:
        total, argmax = 0.0, True
        for depth, token in enumerate(choice):
            known, best = results[tuple(choice[:depth])]
            total += known[token]
            argmax = argmax and token == best
        scores.append(total)
        greedy.append(argmax)
    return ChoiceScores(scores, greedy, requests, [len(c) for c in choices])


def client_peek(client, top: int) -> Peek:
    """peek() over a generation client (palingenesis.rl.generation.GenerationClient, or an engine pool's): one
    sampled token with the sampler's `top` candidates."""
    from palingenesis.opd.rollout import Sampling

    async def peek(prompt: list[int], allowed: tuple[int, ...]) -> dict[int, float]:
        out = await client.generate(prompt, 1, Sampling(1.0, candidates=top, allowed=allowed))
        found: dict[int, float] = {}
        candidates = getattr(out, "candidates", None)
        if candidates is not None and len(candidates):
            n = int(candidates.counts[0])
            found = {int(t): float(lp) for t, lp in zip(candidates.ids[:n], candidates.logprobs[:n])}
        if out.ids and out.logprobs and math.isfinite(out.logprobs[0]):
            found.setdefault(int(out.ids[0]), float(out.logprobs[0]))
        return found

    return peek


def forward_loglikelihoods(model, pad_id: int, pairs: list[tuple[list[int], list[int]]], batch: int = 8):
    """[(log p(continuation | context), is greedy), ...] with padded forwards of the model itself (longest first)."""
    import torch

    device = next(model.parameters()).device
    order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
    out: list[tuple[float, bool]] = [(0.0, True)] * len(pairs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(order), batch):
                chunk = [i for i in order[start : start + batch] if pairs[i][1]]
                if not chunk:
                    continue
                rows = [pairs[i][0] + pairs[i][1] for i in chunk]
                width = max(len(r) for r in rows)
                ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for r, row in enumerate(rows):
                    ids[r, : len(row)] = torch.tensor(row)
                    mask[r, : len(row)] = 1
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                # Only the continuation's positions are normalised (in fp32, on the device): a full-vocabulary
                # log_softmax over every position of the batch is [batch, width, vocab] fp32 (~22 GiB at 8 x 2k x 250k).
                for r, i in enumerate(chunk):
                    context, continuation = pairs[i]
                    first = len(context) - 1
                    steps = torch.log_softmax(logits[r, first : first + len(continuation)].float(), dim=-1)
                    targets = torch.tensor(continuation, device=steps.device)
                    out[i] = (
                        float(steps.gather(1, targets[:, None]).sum()),
                        bool((steps.argmax(-1) == targets).all()),
                    )
                del logits
    finally:
        model.train(was_training)
    return out


def choice_metrics(scores: ChoiceScores, choices: list[str], gold: int) -> dict[str, Any]:
    """An item's multiple-choice outcome, as lm-evaluation-harness computes it: acc (the most likely choice is the
    gold one), acc_norm (log-likelihood per character of the choice), plus the gold choice's probability among the
    choices."""
    lls = scores.loglikelihoods
    pred = max(range(len(lls)), key=lambda i: lls[i])
    norm = max(range(len(lls)), key=lambda i: lls[i] / max(1, len(choices[i])))
    top = max(lls)
    mass = sum(math.exp(x - top) for x in lls) if math.isfinite(top) else 0.0
    return {
        "acc": float(pred == gold),
        "acc_norm": float(norm == gold),
        "pred": pred,
        "gold": gold,
        "prob_gold": math.exp(lls[gold] - top) / mass if mass else 0.0,
        "loglikelihoods": [x if math.isfinite(x) else -1e30 for x in lls],
        "greedy": scores.greedy[gold],
    }
