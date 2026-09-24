"""Aligning a student rollout with the teacher's view of the same text.

The teacher scores the student's completion conditioned on the same conversation,
rendered with the TEACHER's chat template. How the completion itself reaches the
teacher depends on the tokenizer pair:

  SharedVocabAligner  the tokenizers share a base vocabulary (token_bridge): the
                      completion ids are fed as they are (end-of-turn remapped), and
                      student position i is teacher position i.
  ByteChunkAligner    different tokenizers: the completion text is re-tokenized by
                      the teacher, and the two token sequences are cut into chunks at
                      the byte offsets where both tokenizations end a token. A chunk
                      is the same text on both sides, so its total log-probability is
                      comparable across the pair (losses.xtok).

Student byte offsets come from the SAMPLED ids (each token's bytes), never from
re-encoding the decoded text: a sampled sequence is often not the tokenizer's
canonical encoding of its own text ("hel" + "lo" instead of "hello"), and
re-encoding would silently describe other tokens than the ones trained on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from palingenesis.opd.token_bridge import TokenBridge


@dataclass
class ChunkMap:
    """Chunk index of every completion token on both sides.

    ``student[i]`` / ``teacher[j]`` is the chunk of student token i / teacher token j,
    or -1 for tokens outside every chunk (special tokens inside a completion, bytes
    after the valid UTF-8 prefix). ``keep[c]`` is False for chunks the loss ignores
    (whitespace-only text). ``one_to_one`` lists (student, teacher) positions of chunks
    with exactly one token on each side: there both models predict the same next
    token from the same text, so their distributions can be compared directly.
    """

    student: list[int]
    teacher: list[int]
    keep: list[bool]
    one_to_one: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.keep)


@dataclass
class TeacherView:
    """The teacher's input for one rollout: its prompt followed by its completion."""

    input_ids: list[int]
    prompt_len: int
    chunks: ChunkMap | None = None  # None: student position i is teacher position i

    @property
    def completion_len(self) -> int:
        return len(self.input_ids) - self.prompt_len


# ------------------------------------------------------------------ token bytes


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte <-> printable-character table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# str.translate table: byte-level character -> the byte it encodes (as a latin-1 char).
_BYTE_LEVEL = {ord(ch): b for b, ch in _bytes_to_unicode().items()}
_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _is_byte_level(tok) -> bool:
    decoder = json.loads(tok.backend_tokenizer.to_str()).get("decoder") or {}
    kinds = {decoder.get("type")} | {d.get("type") for d in decoder.get("decoders", [])}
    return "ByteLevel" in kinds


def token_bytes(tok) -> list[bytes | None]:
    """The bytes each token id stands for, or None for special tokens.

    Byte-level BPE (GPT-2, Llama 3, Qwen, SmolLM) spells each byte as one printable
    character; SentencePiece-style vocabularies (Mistral, Gemma) use "▁" for a space
    and ``<0xNN>`` for byte fallback. Added non-special tokens (``<think>``) are
    their literal text.
    """
    added = {i: t for i, t in tok.added_tokens_decoder.items()}
    special = set(tok.all_special_ids) | {i for i, t in added.items() if t.special}
    byte_level = _is_byte_level(tok)
    table: list[bytes | None] = []
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
        if i in special or piece is None:
            table.append(None)
        elif i in added:
            table.append(added[i].content.encode())
        elif byte_level:
            table.append(piece.translate(_BYTE_LEVEL).encode("latin-1"))
        elif m := _BYTE_FALLBACK.match(piece):
            table.append(bytes([int(m.group(1), 16)]))
        else:
            table.append(piece.replace("▁", " ").encode())
    return table


def vocab_map(student_bytes: list[bytes | None], teacher_bytes: list[bytes | None]) -> list[int]:
    """For each teacher id, the student id spelling the same bytes (-1 if none).

    Comparing bytes makes the spellings canonical for free: "Ġthe", "▁the" and a
    literal " the" all decode to b" the". The lowest id wins among duplicates.
    """
    by_bytes: dict[bytes, int] = {}
    for i, b in enumerate(student_bytes):
        if b:
            by_bytes.setdefault(b, i)
    return [by_bytes.get(b, -1) if b else -1 for b in teacher_bytes]


def _valid_utf8_prefix(data: bytes) -> int:
    """Length of the longest prefix of `data` that is valid UTF-8."""
    try:
        data.decode()
        return len(data)
    except UnicodeDecodeError as e:
        return e.start


def align_chunks(student_ends: list[float], teacher_ends: list[float]) -> list[tuple[range, range]]:
    """Cut two token sequences over the same bytes into chunks at their common token ends.

    Two pointers: advance the side whose current token ends first; when both end at
    the same byte, close a chunk. Ends must be non-decreasing and both sequences must
    cover the same bytes; anything left over after the last common end forms one
    final chunk.
    """
    chunks = []
    i = j = i0 = j0 = 0
    while i < len(student_ends) and j < len(teacher_ends):
        if student_ends[i] < teacher_ends[j]:
            i += 1
        elif student_ends[i] > teacher_ends[j]:
            j += 1
        else:
            i += 1
            j += 1
            chunks.append((range(i0, i), range(j0, j)))
            i0, j0 = i, j
    if i0 < len(student_ends) or j0 < len(teacher_ends):
        chunks.append((range(i0, len(student_ends)), range(j0, len(teacher_ends))))
    return chunks


# --------------------------------------------------------------------- aligners


class SharedVocabAligner:
    """Student and teacher share a base vocabulary: feed the completion ids as they are."""

    def __init__(self, bridge: TokenBridge, stop_ids: tuple[int, ...]):
        self.bridge = bridge
        self.stop_ids = stop_ids

    def clean(self, ids: list[int]) -> list[int]:
        return self.bridge.clean_completion(ids, self.stop_ids)

    def view(self, teacher_prompt: list[int], completion: list[int]) -> TeacherView:
        return TeacherView(teacher_prompt + self.bridge.to_teacher(completion), len(teacher_prompt))


class ByteChunkAligner:
    """Different tokenizers: re-tokenize the completion text and chunk at shared byte ends.

    The student's end-of-turn token has no text; when a completion ends with it, the
    teacher's end-of-turn token is appended and the pair forms the last chunk, so the
    teacher still supervises when to stop.
    """

    def __init__(self, student_tok, teacher_tok, stop_ids: tuple[int, ...], teacher_eot: int,
                 mask_whitespace: bool = True):
        self.teacher_tok = teacher_tok
        self.stop_ids = stop_ids
        self.teacher_eot = teacher_eot
        self.mask_whitespace = mask_whitespace
        self.student_bytes = token_bytes(student_tok)
        self.teacher_bytes = token_bytes(teacher_tok)
        self.teacher_to_student = vocab_map(self.student_bytes, self.teacher_bytes)
        self.teacher_to_student[teacher_eot] = stop_ids[0]

    def clean(self, ids: list[int]) -> list[int]:
        """Cut at the first stop token (kept: stopping is supervised too), or before an
        id outside the tokenizer: embeddings are often padded past the vocabulary, and
        a sampler can draw those rows, which spell no text."""
        for i, t in enumerate(ids):
            if t >= len(self.student_bytes):
                return ids[:i]
            if t in self.stop_ids:
                return ids[: i + 1]
        return list(ids)

    def view(self, teacher_prompt: list[int], completion: list[int]) -> TeacherView:
        stopped = bool(completion) and completion[-1] in self.stop_ids
        body = completion[:-1] if stopped else completion

        # Student byte ends from the sampled ids; special tokens carry no text.
        pieces = [self.student_bytes[t] for t in body]
        text = b"".join(p for p in pieces if p)
        valid = _valid_utf8_prefix(text)
        student_pos, student_ends, end = [], [], 0
        for i, piece in enumerate(pieces):
            if not piece:
                continue
            end += len(piece)
            if end > valid:
                break
            student_pos.append(i)
            student_ends.append(end)

        teacher_ids, teacher_ends = self._encode(text[:valid].decode())
        chunks = align_chunks(student_ends, teacher_ends)
        student_chunk = [-1] * len(completion)
        teacher_chunk = [-1] * len(teacher_ids)
        keep, one_to_one = [], []
        for c, (s_range, t_range) in enumerate(chunks):
            for i in s_range:
                student_chunk[student_pos[i]] = c
            for j in t_range:
                teacher_chunk[j] = c
            start = student_ends[s_range.start - 1] if s_range.start else 0
            chunk_text = text[start:student_ends[s_range[-1]]] if len(s_range) else b""
            keep.append(bool(len(s_range) and len(t_range)) and not (self.mask_whitespace and chunk_text.isspace()))
            if keep[-1] and len(s_range) == 1 and len(t_range) == 1:
                one_to_one.append((student_pos[s_range[0]], t_range[0]))

        if stopped and valid == len(text):
            c = len(keep)
            student_chunk[-1] = c
            teacher_chunk.append(c)
            one_to_one.append((len(completion) - 1, len(teacher_ids)))
            teacher_ids = teacher_ids + [self.teacher_eot]
            keep.append(True)
        return TeacherView(teacher_prompt + teacher_ids, len(teacher_prompt),
                           ChunkMap(student_chunk, teacher_chunk, keep, one_to_one))

    def _encode(self, text: str) -> tuple[list[int], list[float]]:
        """Teacher ids for `text` and each token's byte end.

        Byte ends come from the tokens' own bytes when they spell the text exactly
        (byte-level BPE always does). Otherwise (normalizers, SentencePiece prefix
        spaces) they come from the character offsets; tokens that split one
        character get a fractional end inside it, so no chunk can close there.
        """
        ids = self.teacher_tok.encode(text, add_special_tokens=False)
        pieces = [self.teacher_bytes[t] for t in ids]
        if all(p is not None for p in pieces) and b"".join(pieces) == text.encode():
            ends, end = [], 0
            for p in pieces:
                end += len(p)
                ends.append(end)
            return ids, ends
        enc = self.teacher_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        char_end = [0]
        for ch in text:
            char_end.append(char_end[-1] + len(ch.encode()))
        spans = enc["offset_mapping"]
        ends = []
        for j, (_, stop) in enumerate(spans):
            shared_with_next = j + 1 < len(spans) and spans[j + 1][1] == stop
            ends.append(char_end[stop] - 0.5 if shared_with_next else char_end[stop])
        return list(enc["input_ids"]), ends
