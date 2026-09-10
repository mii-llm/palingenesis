"""Test the OPD engine math: completion-position logit gathering and reverse-KL loss.

These tests lock in the numerical behavior of the scoring path with tiny
handmade models (no HF downloads, CPU-only) so the data-layer refactor and any
future change to the engine can be checked against a golden reference.
"""

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from palingenesis.opd.config import OPDConfig  # noqa: E402
from palingenesis.opd.token_bridge import TokenBridge  # noqa: E402
from palingenesis.opd.trainer import OPDTrainer  # noqa: E402


class _Backbone(nn.Module):
    """Position-dependent, per-position hidden states (no cross-token mixing),
    so padded-batch outputs at real positions equal unpadded outputs — any
    disagreement with the naive reference is then an indexing bug."""

    def __init__(self, vocab, hidden, max_pos=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_pos, hidden)
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, input_ids=None, attention_mask=None):
        T = input_ids.shape[1]
        h = self.mix(self.emb(input_ids) + self.pos(torch.arange(T)[None, :]))
        return SimpleNamespace(last_hidden_state=h)


class TinyLM(nn.Module):
    def __init__(self, vocab, hidden=16):
        super().__init__()
        self.model = _Backbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


STUDENT_VOCAB = 10
SHARED = 8
SWAP_SRC, SWAP_DST = 9, 3  # student's "<|im_end|>" analog -> teacher's "<|eot_id|>" analog


def make_trainer(loss_fn="full_kl"):
    """A bare OPDTrainer with just the attributes the scoring path uses."""
    torch.manual_seed(0)
    t = OPDTrainer.__new__(OPDTrainer)
    t.config = OPDConfig()
    t.config.train.loss_fn = loss_fn
    t.bridge = TokenBridge(shared_vocab_size=SHARED, swap={SWAP_SRC: SWAP_DST}, stop_ids=(SWAP_SRC,))
    t.device = "cpu"
    t.teacher_device = "cpu"
    t.student = TinyLM(STUDENT_VOCAB)
    t.teacher = TinyLM(SHARED)
    t.s_pad = 0
    t.t_pad = 0
    return t


# ragged rollouts: (student_prompt, completion in student ids)
ROLLOUTS = [
    ([1, 2, 3], [4, 5, SWAP_SRC]),
    ([6, 1], [7, SWAP_SRC]),
    ([2, 4, 6, 1], [5]),
]


def chunk_args(bridge):
    """Build _loss_on_chunk arguments exactly the way the trainer does."""
    s_seqs, t_seqs, plens_s, plens_t, lens, targets = [], [], [], [], [], []
    for s_prompt, comp in ROLLOUTS:
        comp_t = bridge.to_teacher(comp)
        t_prompt = s_prompt[:-1]  # teacher renders the prompt differently (shorter here)
        s_seqs.append(s_prompt + comp[:-1])
        t_seqs.append(t_prompt + comp_t[:-1])
        plens_s.append(len(s_prompt))
        plens_t.append(len(t_prompt))
        lens.append(len(comp))
        targets.extend(comp_t)
    return s_seqs, t_seqs, plens_s, plens_t, lens, targets


def naive_completion_logits(model, seq, plen, comp_len):
    """Reference: full forward of ONE unpadded sequence, slice positions P-1..P+L-2."""
    ids = torch.tensor([seq])
    mask = torch.ones_like(ids)
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    return model.lm_head(h[0, plen - 1 : plen - 1 + comp_len])


def test_gather_logits_matches_naive_per_sequence():
    torch.manual_seed(0)
    model = TinyLM(STUDENT_VOCAB)
    seqs = [s + c[:-1] for s, c in ROLLOUTS]
    plens = [len(s) for s, _ in ROLLOUTS]
    lens = [len(c) for _, c in ROLLOUTS]

    got = OPDTrainer._gather_logits(model, seqs, plens, lens, pad=0, device="cpu")
    want = torch.cat([
        naive_completion_logits(model, seq, plen, comp_len)
        for seq, plen, comp_len in zip(seqs, plens, lens)
    ])
    assert got.shape == (sum(lens), STUDENT_VOCAB)
    torch.testing.assert_close(got, want)


def reference_loss(trainer, loss_fn):
    """Independent recomputation of the reverse-KL objective with plain ops."""
    s_seqs, t_seqs, plens_s, plens_t, lens, targets = chunk_args(trainer.bridge)
    logp_s_full = torch.cat([
        F.log_softmax(naive_completion_logits(trainer.student, seq, plen, L).float(), dim=-1)
        for seq, plen, L in zip(s_seqs, plens_s, lens)
    ])
    logp_t = torch.cat([
        F.log_softmax(naive_completion_logits(trainer.teacher, seq, plen, L).float(), dim=-1)
        for seq, plen, L in zip(t_seqs, plens_t, lens)
    ])
    p_full = logp_s_full.exp()
    p_shared = p_full[:, :SHARED].clone()
    p_shared[:, SWAP_DST] += p_full[:, SWAP_SRC]
    logp_s = (p_shared + 1e-12).log()
    tgt = torch.tensor(targets)[:, None]
    kl = (p_shared * (logp_s - logp_t)).sum(-1)
    sampled_kl = (logp_s.gather(-1, tgt) - logp_t.gather(-1, tgt)).squeeze(-1)
    if loss_fn == "full_kl":
        loss = kl.sum()
    else:
        loss = (logp_s.gather(-1, tgt).squeeze(-1) * sampled_kl.detach()).sum()
    residual = p_full[:, SHARED:STUDENT_VOCAB].sum(-1) - p_full[:, SWAP_SRC]  # unmapped ids only
    return loss, kl, sampled_kl, residual


@pytest.mark.parametrize("loss_fn", ["full_kl", "sampled_rkl"])
def test_loss_on_chunk_matches_reference(loss_fn):
    trainer = make_trainer(loss_fn)
    args = chunk_args(trainer.bridge)
    loss, n_tok, stats = trainer._loss_on_chunk(*args)
    ref_loss, ref_kl, ref_sampled, ref_residual = reference_loss(trainer, loss_fn)

    assert n_tok == sum(len(c) for _, c in ROLLOUTS)
    torch.testing.assert_close(loss, ref_loss)
    assert stats["kl"] == pytest.approx(ref_kl.mean().item(), abs=1e-6)
    assert stats["sampled_kl"] == pytest.approx(ref_sampled.mean().item(), abs=1e-6)
    assert stats["residual_mass"] == pytest.approx(ref_residual.mean().item(), abs=1e-6)


def test_loss_backward_reaches_student_only():
    trainer = make_trainer()
    loss, _, _ = trainer._loss_on_chunk(*chunk_args(trainer.bridge))
    loss.backward()
    assert all(p.grad is not None for p in trainer.student.parameters())
    assert all(p.grad is None for p in trainer.teacher.parameters())
