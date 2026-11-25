"""train.packing for on-policy distillation: scoring forwards as packed rows (palingenesis.tree_packing).

A step's completions come in groups: `rollout.group_size` samples of one prompt, and on agent traces every
regenerated turn of a trace (onset branches included) continues a prefix of the same trunk. Padded scoring
forwards each completion as a row of its own, prompt included, once for the teacher and once for the student.
Packed, a micro-batch is one row [1, N]:

  varlen  the sequences end to end: no padding, every prompt still once per completion
  tree    their prefix trie: a prompt (and whatever its completions start with in common, when that is worth a
          tree level) is forwarded once, for the teacher and for the student, forward and backward

Each model packs its own token sequences: a teacher that reads another prompt than the student (privileged
context, other tool schemas, another tokenizer) shares what its own sequences have in common. The losses read
the hidden states of the completion positions gathered from the row, in the order the padded path yields them,
so every loss computes what it computes on padded rows (equal up to floating-point reordering; gradients reach
a shared token from every completion through it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from palingenesis.tree_packing import PackedRow, PackingSupport, forward_packed, pack_amortized, plan, prepare_model

GATHER_FACTOR = 8  # train.packing_gather_tokens 0: this many times the micro-batch's token budget


@dataclass(frozen=True)
class Packing:
    """train.packing varlen / tree and its budgets."""

    mode: str  # "varlen" | "tree"
    tokens: int = 0  # packed tokens per micro-batch (0: micro_seqs x the longest sequence, the padded worst case)
    gather_tokens: int = 0  # tree: K/V rows a micro-batch's attention gathers per layer (0: GATHER_FACTOR x tokens)

    @property
    def share(self) -> bool:
        return self.mode == "tree"

    def budget(self, sequences: list, micro_seqs: int) -> int:
        return self.tokens or max(1, micro_seqs) * max(len(s) for s in sequences)

    def micro_batches(self, sequences: list, scored: list[int], micro_seqs: int) -> list[list[int]]:
        """Micro-batches (lists of sequence indices), each one packed row within the token budget (tree: the
        trie's tokens; a sequence goes where the longest prefix of it already is, so groups stay together) and
        with at most that many scored positions (the loss's rows)."""
        budget = self.budget(sequences, micro_seqs)
        return plan(
            sequences,
            [0] * len(sequences),
            scored,
            budget,
            share=self.share,
            gather_budget=(self.gather_tokens or GATHER_FACTOR * budget) if self.share else None,
        )

    def row(self, sequences: list) -> PackedRow:
        return pack_amortized(sequences, share=self.share)


def support_for(model, device) -> PackingSupport:
    """`model` checked and routed for packed rows (tree_packing.prepare_model; PackingError when it cannot run
    them exactly)."""
    return prepare_model(model, device)


def gather_index(row: PackedRow, spans: list[tuple[int, int, int]], device) -> torch.Tensor:
    """Packed positions of the hidden states the loss reads: for each (sequence, first position, count), that
    sequence's positions first .. first + count - 1, concatenated in order."""
    parts = [row.positions(i)[first : first + count] for i, first, count in spans]
    return torch.from_numpy(np.concatenate(parts)).to(device)


def packed_states(model, support: PackingSupport, row: PackedRow, device) -> torch.Tensor:
    """The row's final hidden states [N, H]."""
    return forward_packed(model, support, row, device)[0]


@torch.no_grad()
def packed_hidden(
    model, support: PackingSupport, packing: Packing, sequences: list, wanted: list[np.ndarray], micro_seqs: int, device
) -> list[torch.Tensor]:
    """Final hidden states (no grad) of each sequence at its `wanted` positions [len(wanted[i]), H], the sequences
    forwarded as packed rows within the token budget (a teacher reading agent traces: every regenerated turn's
    context and the trunk are prefixes of one another, forwarded once)."""
    out: list[torch.Tensor | None] = [None] * len(sequences)
    kind = torch.device(device).type
    for part in packing.micro_batches(sequences, [len(w) for w in wanted], micro_seqs):
        row = packing.row([sequences[i] for i in part])
        with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
            states = packed_states(model, support, row, device)
        for j, i in enumerate(part):
            index = torch.from_numpy(row.positions(j)[np.asarray(wanted[i], dtype=np.int64)]).to(states.device)
            out[i] = states[index]
    return out
