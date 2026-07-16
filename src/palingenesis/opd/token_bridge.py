"""Bridging a student and a teacher that share a base vocabulary.

The supported setup: both tokenizers share the same base vocabulary
(ids 0..shared_vocab_size-1) and the student *extends* it with template/tool
tokens the teacher cannot embed — e.g. a ChatML student (adds ``<|im_end|>``
at 128256+) distilled from a Llama-3-template teacher (vocab ends at 128256).

Prompts are therefore rendered per-model with each model's own chat template;
only completion tokens (base vocab) are aligned across the two models.

End-of-turn handling: the student's terminator (say ``<|im_end|>``) and the
teacher's (say ``<|eot_id|>``) mean the same thing, so for scoring the
student's terminator probability mass is merged into the teacher's terminator
slot — that way the teacher also supervises *when to stop*, not just what to
say. The mapping is the ``swap`` dict; it is applied both when a completion
token is fed to the teacher and when it is used as a scoring target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tokenization-identity probes for check_compatible. Deliberately mixed:
# plain prose, accented characters (both precomposed forms and ones that
# byte-level BPEs split aggressively), code, and an instruction-style line.
DEFAULT_PROBE_TEXTS = (
    "The quick brown fox jumps over the lazy dog. 12,345.67!",
    "Rispondi alla seguente domanda a scelta multipla sull'argomento 'storia'.",
    "La Divina Commedia è composta da tre cantiche: Inferno, Purgatorio e Paradiso.",
    "Perché l'ossigeno è più elettronegativo dell'azoto? Risposta: A",
    "x = [n**2 for n in range(10)]  # città, però, così",
)


class TokenBridgeError(Exception):
    """Raised when the student/teacher tokenizer pair cannot support exact per-token KL."""


@dataclass(frozen=True)
class TokenBridge:
    """Completion-token alignment between a student and a smaller-vocab teacher."""

    # Ids below this are identical in both tokenizers (checked by check_compatible).
    shared_vocab_size: int
    # student id -> teacher id, applied when a completion token is fed to the
    # teacher or used as a scoring target in the shared vocab.
    swap: dict[int, int] = field(default_factory=dict)
    # Completion is truncated at the first of these (kept, as terminal target).
    stop_ids: tuple[int, ...] = ()

    @classmethod
    def from_tokenizers(
        cls,
        student_tok,
        teacher_tok,
        eos_map: dict[str, str] | None = None,
        extra_stop_tokens: tuple[str, ...] | list[str] = (),
    ) -> "TokenBridge":
        """Build a bridge from a tokenizer pair.

        eos_map maps student token *strings* to teacher token strings, e.g.
        ``{"<|im_end|>": "<|eot_id|>"}``. When empty, falls back to mapping the
        student's eos token to the teacher's eos token if the student's eos id
        lies outside the shared vocab — set eos_map explicitly when the
        teacher's *configured* eos is not its end-of-turn token (base models
        often configure ``<|end_of_text|>`` while the conversational
        terminator is ``<|eot_id|>``).

        extra_stop_tokens are additional student token strings that terminate
        a sampled completion (typically the shared end-of-text token).
        """
        shared = len(teacher_tok)
        if len(student_tok) < shared:
            raise TokenBridgeError(
                f"Student vocab ({len(student_tok)}) is smaller than teacher vocab ({shared}). "
                "OPD requires the teacher vocab to be a prefix of the student's."
            )

        swap: dict[int, int] = {}
        if eos_map:
            for s_name, t_name in eos_map.items():
                s_id = student_tok.convert_tokens_to_ids(s_name)
                t_id = teacher_tok.convert_tokens_to_ids(t_name)
                if s_id is None or t_id is None:
                    raise TokenBridgeError(f"eos_map token not found: {s_name!r} -> {t_name!r}")
                swap[s_id] = t_id
        elif (
            student_tok.eos_token_id is not None
            and teacher_tok.eos_token_id is not None
            and student_tok.eos_token_id >= shared
        ):
            swap[student_tok.eos_token_id] = teacher_tok.eos_token_id
            logger.info(
                "token_bridge: auto eos map %s (%d) -> %s (%d)",
                student_tok.eos_token,
                student_tok.eos_token_id,
                teacher_tok.eos_token,
                teacher_tok.eos_token_id,
            )

        stop_ids: list[int] = list(swap.keys())
        if student_tok.eos_token_id is not None and student_tok.eos_token_id not in stop_ids:
            stop_ids.append(student_tok.eos_token_id)
        for name in extra_stop_tokens:
            tid = student_tok.convert_tokens_to_ids(name)
            if tid is None:
                raise TokenBridgeError(f"extra_stop_tokens token not found in student vocab: {name!r}")
            if tid not in stop_ids:
                stop_ids.append(tid)

        return cls(shared_vocab_size=shared, swap=swap, stop_ids=tuple(stop_ids))

    def clean_completion(self, ids: list[int]) -> list[int]:
        """Truncate a raw sampled completion for scoring.

        Cuts at the first stop token (inclusive: stopping is supervised too).
        Any other out-of-shared-vocab token (tool tokens etc.) truncates the
        completion *before* it — the teacher has no equivalent to score.
        """
        out: list[int] = []
        for t in ids:
            if t in self.stop_ids:
                out.append(t)
                break
            if t >= self.shared_vocab_size and t not in self.swap:
                break
            out.append(t)
        return out

    def to_teacher(self, ids: list[int]) -> list[int]:
        """Map a cleaned student completion to teacher ids (identity outside swap)."""
        return [self.swap.get(t, t) for t in ids]


def check_compatible(
    student_tok,
    teacher_tok,
    bridge: TokenBridge,
    probe_texts: tuple[str, ...] | list[str] = (),
) -> None:
    """Hard preconditions for exact per-token KL. Raises TokenBridgeError on violation.

    Exact KL is only meaningful if both models assign the *same ids* to the
    same text — a near-miss tokenizer pair silently degrades to noise, so this
    check is mandatory and runs before any weights are loaded onto the GPU.
    """
    for text in tuple(DEFAULT_PROBE_TEXTS) + tuple(probe_texts):
        s_ids = student_tok.encode(text, add_special_tokens=False)
        t_ids = teacher_tok.encode(text, add_special_tokens=False)
        if s_ids != t_ids:
            raise TokenBridgeError(
                f"Tokenizers diverge on shared text!\n{text!r}\nstudent={s_ids}\nteacher={t_ids}"
            )

    for s_id, t_id in bridge.swap.items():
        if s_id < bridge.shared_vocab_size:
            raise TokenBridgeError(
                f"swap source id {s_id} lies inside the shared vocab ({bridge.shared_vocab_size}) — "
                "mapping a shared token would corrupt the KL target."
            )
        if not (0 <= t_id < bridge.shared_vocab_size):
            raise TokenBridgeError(f"swap target id {t_id} is outside the teacher vocab.")

    if not bridge.stop_ids:
        raise TokenBridgeError(
            "No stop tokens resolved — generation would never terminate cleanly. "
            "Set bridge.eos_map or bridge.extra_stop_tokens in the config."
        )

    logger.info(
        "token_bridge: OK — shared base vocab (%d ids), swap %s, stop ids %s",
        bridge.shared_vocab_size,
        bridge.swap or "{}",
        list(bridge.stop_ids),
    )
