"""Score the prompt pool with the teacher — one forward per row, no decoding.

Reverse-KL distillation faithfully transfers the teacher's *errors* along with
its knowledge: on a pool where the teacher is right 50% of the time, half the
supervision pulls the student toward wrong answers, and the teacher's accuracy
becomes a hard ceiling. Annotating every row with the teacher's own answer
lets the pool be filtered or reweighted *before* training (the same
score-then-select philosophy as `pgs prepare` on the SFT side).

This module only annotates — each row is written back with two extra fields:

    teacher_answer:  the option letter the teacher assigns the highest logit
    teacher_correct: teacher_answer == row["answer"]

What to do with them (drop incorrect rows, downweight them, rebalance
categories) is downstream policy, decided by the experiment or config.

Method: rows are rendered with the fast-mode template and the benchmark's
reference shots (the teacher at its best, matching the training distribution),
and the teacher's answer is read from the logits of the option-letter tokens
at the final prompt position — in fast mode the first completion token IS the
letter. One batched forward per row: no generation loop, no format parsing,
no unparseable outputs.

Usage:
    python -m palingenesis.opd.score_pool --config configs/distill_opd.yaml \
        --out data/prompts_scored.jsonl
    pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from transformers import AutoTokenizer

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.formatting import build_messages, letter_token_ids, load_reference_shots
from palingenesis.opd.pool import load_pool
from palingenesis.opd.trainer import OPDTrainer, load_causal_lm, pick_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def score_rows(model, tok, rows, shots, letter_ids, batch_size: int, device: str,
               system_message: str | None = None, log_every: int = 50):
    """Yield rows annotated with teacher_answer / teacher_correct."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [
            OPDTrainer._encode_prompt(
                tok, build_messages(r, few_shots=shots, fast=True, system_message=system_message)
            )
            for r in chunk
        ]
        # logits at the last prompt position = distribution over the first
        # completion token; _gather_logits keeps memory at batch x vocab.
        logits = OPDTrainer._gather_logits(
            model, prompts, plens=[len(p) for p in prompts],
            lens=[1] * len(prompts), pad=pad, device=device,
        )
        for row, row_logits in zip(chunk, logits):
            candidates = [letter for letter, _ in row["options"] if letter in letter_ids]
            scores = {letter: row_logits[letter_ids[letter]].item() for letter in candidates}
            answer = max(scores, key=scores.get)
            yield {**row, "options": [list(o) for o in row["options"]],
                   "teacher_answer": answer,
                   "teacher_correct": answer == row["answer"]}
        batch_idx = start // batch_size
        if batch_idx % log_every == 0:
            done = start + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            logger.info("scored %d/%d rows (%.1f rows/s, ETA %.0f min)",
                        done, len(rows), rate, (len(rows) - done) / rate / 60)


def main():
    from palingenesis.logging import setup_logging

    setup_logging(rank=0)
    ap = argparse.ArgumentParser(description="Annotate a prompt pool with the teacher's answers")
    ap.add_argument("--config", required=True, help="OPD config (teacher/pool/shots are read from it)")
    ap.add_argument("--out", required=True, help="Output JSONL (pool rows + teacher_answer/teacher_correct)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N rows (0 = all)")
    args = ap.parse_args()

    config = OPDConfig.from_yaml(args.config)
    device = pick_device()

    logger.info("Loading teacher %s (bf16) on %s", config.model.teacher, device)
    tok = AutoTokenizer.from_pretrained(config.model.teacher)
    model = load_causal_lm(config.model.teacher, torch.bfloat16).to(device)
    model.eval().requires_grad_(False)

    rows = load_pool(config.data.prompts_path)
    if args.limit:
        rows = rows[: args.limit]
    shots = load_reference_shots(config.data.shots_path) if config.data.shots_path else []
    letter_ids = letter_token_ids(tok)
    logger.info("Scoring %d rows (%d-shot fast mode, batch %d)", len(rows), len(shots), args.batch_size)

    n_correct = 0
    with open(args.out, "w") as f:
        for i, scored in enumerate(score_rows(
            model, tok, rows, shots, letter_ids, args.batch_size, device,
            system_message=config.data.system_message or None,
        )):
            n_correct += scored["teacher_correct"]
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                f.flush()

    logger.info("Done: teacher correct on %d/%d rows (%.1f%%) -> %s",
                n_correct, len(rows), 100 * n_correct / max(1, len(rows)), args.out)


if __name__ == "__main__":
    main()
