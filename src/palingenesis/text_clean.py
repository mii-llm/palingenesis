"""Model-free cleaning of raw text for pretraining (`mode: pretrain` sources of `pgs prepare`).

Three checks, each deterministic and none needing a tokenizer or a model:

  * exact duplicates: documents whose normalised text (Unicode NFKC, lower case, whitespace
    collapsed) is the same;
  * near-duplicates: documents whose sets of word shingles have an estimated Jaccard
    similarity above a threshold (MinHash signatures, candidate pairs by locality-sensitive
    hashing; the first document of every group is kept);
  * decontamination: documents that share a word n-gram with an evaluation set.

Everything is held in memory (one 8-byte hash per evaluation n-gram and per document, one
signature per kept document): sized for corpora of up to a few million documents, not for a
web crawl. Near-duplicate detection needs numpy; the rest is standard library only.
"""

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from palingenesis.data_clean import ROW_INDEX, CleanReport

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_MASK = (1 << 64) - 1
_BASE = 0x9E3779B97F4A7C15  # odd multiplier of the rolling n-gram hash (mod 2^64)

SHINGLE_WORDS = 5  # words per shingle of the near-duplicate check
NUM_PERM = 128  # MinHash signature length: the similarity estimate's standard error is about 0.04 at 0.8


def normalize(text: str) -> str:
    """The text exact duplicates are compared on: NFKC, lower case, whitespace collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


def text_key(text: str) -> str:
    """Identity of a document for exact deduplication."""
    return hashlib.blake2b(normalize(text).encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def words(text: str) -> list[str]:
    """The words n-grams and shingles are made of: normalised, punctuation dropped."""
    return _WORD.findall(normalize(text))


class _WordHashes:
    """A stable 64-bit hash per word (Python's own hash() changes between processes)."""

    def __init__(self):
        self._cache: dict[str, int] = {}

    def __call__(self, word: str) -> int:
        h = self._cache.get(word)
        if h is None:
            h = int.from_bytes(hashlib.blake2b(word.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")
            if len(self._cache) < 2_000_000:  # bound the cache on corpora with unbounded vocabularies
                self._cache[word] = h
        return h


def ngram_hashes(tokens: list[str], n: int, word_hash: _WordHashes) -> Iterator[int]:
    """A 64-bit hash of every run of n consecutive words (a polynomial rolling hash: one
    multiplication per n-gram). Fewer than n words give nothing."""
    if len(tokens) < n:
        return
    hashes = [word_hash(t) for t in tokens]
    top = pow(_BASE, n - 1, 1 << 64)
    h = 0
    for i in range(n):
        h = (h * _BASE + hashes[i]) & _MASK
    yield h
    for i in range(n, len(hashes)):
        h = ((h - hashes[i - n] * top) * _BASE + hashes[i]) & _MASK
        yield h


def _strings(value: Any) -> Iterator[str]:
    """Every string in a row, nested lists and mappings included (questions, answers, messages)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def read_rows(path: str | Path) -> Iterator[Any]:
    """Rows of an evaluation file: .jsonl, .json (a list or one object), .parquet, or any other
    file as plain text (one row: the whole file)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decontamination file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data if isinstance(data, list) else [data]
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


SHARED_ROW_SHARE = 0.2  # an n-gram in more than this share of a file's rows is shared scaffolding
SHARED_MIN_ROWS = 10  # ...in files with at least this many rows


@dataclass
class NgramIndex:
    """The word n-grams of the evaluation sets a corpus is decontaminated against.

    What is indexed, per file (from_files): every string of every row, or with `fields` only the
    strings under those top-level columns (nested values included). An n-gram found in more than
    `shared_row_share` of a file's rows is left out: it is the benchmark's scaffolding (a shared
    instruction, a system prompt, a few-shot prefix), not an item, and indexing it would remove
    every document that quotes the instruction."""

    n: int = 13
    hashes: set[int] = field(default_factory=set)
    files: list[str] = field(default_factory=list)
    fields: list[str] | None = None
    strings: int = 0  # strings read from the files
    too_short: int = 0  # strings with fewer than n words: they give no n-gram and protect nothing
    shared: int = 0  # n-grams left out because most rows of their file hold them
    shared_row_share: float = SHARED_ROW_SHARE
    _word_hash: _WordHashes = field(default_factory=_WordHashes, repr=False)

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
        n: int = 13,
        fields: Iterable[str] | None = None,
        shared_row_share: float = SHARED_ROW_SHARE,
    ) -> "NgramIndex":
        """The index of the files' rows. fields: the columns to read (default: all). A file whose
        rows have none of them is an error. shared_row_share: see the class (1.0 keeps every n-gram)."""
        if n < 1:
            raise ValueError(f"the decontamination n-gram must be >= 1 words, got {n}")
        index = cls(n=n, fields=list(fields) if fields else None, shared_row_share=shared_row_share)
        for path in paths:
            index.files.append(str(path))
            index._add_file(path)
        if index.files and not index.hashes:
            logger.warning(
                f"Decontamination: no {n}-word n-gram in {', '.join(index.files)} ({index.strings} strings, "
                f"{index.too_short} shorter than {n} words, {index.shared} shared n-grams left out): nothing will "
                "be removed. Lower the n-gram length for short items."
            )
        return index

    def _row_strings(self, row: Any) -> Iterator[str]:
        if self.fields is None or not isinstance(row, dict):
            yield from _strings(row)
            return
        for name in self.fields:
            if name in row:
                yield from _strings(row[name])

    def _add_file(self, path: str | Path) -> None:
        rows = 0
        matched = self.fields is None
        counts: dict[int, int] = {}  # n-gram -> rows of this file that hold it
        for row in read_rows(path):
            rows += 1
            matched = matched or not isinstance(row, dict) or any(name in row for name in self.fields)
            in_row: set[int] = set()
            for text in self._row_strings(row):
                in_row.update(self._hashes(text))
            for h in in_row:
                counts[h] = counts.get(h, 0) + 1
        if not matched:
            raise ValueError(
                f"decontamination: no row of {path} has any of the fields {self.fields} "
                "(preprocess.decontaminate_fields / --decontaminate_fields)"
            )
        limit = self.shared_row_share * rows if rows >= SHARED_MIN_ROWS else rows
        kept = {h for h, c in counts.items() if c <= limit}
        dropped = len(counts) - len(kept)
        if dropped:
            self.shared += dropped
            logger.info(
                f"Decontamination: {dropped:,} n-grams of {path} are in more than {self.shared_row_share:.0%} of "
                "its rows (a shared instruction or prompt) and are not matched against the corpus"
            )
        self.hashes |= kept

    def _hashes(self, text: str) -> Iterator[int]:
        self.strings += 1
        tokens = words(text)
        if len(tokens) < self.n:
            self.too_short += 1
            return iter(())
        return ngram_hashes(tokens, self.n, self._word_hash)

    def add(self, text: str) -> None:
        """Index one string as it is (no shared-n-gram rule)."""
        self.hashes.update(self._hashes(text))

    def contaminated(self, text: str) -> bool:
        """True when the text shares an n-gram with the evaluation sets."""
        if not self.hashes:
            return False
        hashes = self.hashes
        return any(h in hashes for h in ngram_hashes(words(text), self.n, self._word_hash))

    def summary(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "fields": self.fields,
            "ngram_words": self.n,
            "strings": self.strings,
            "strings_shorter_than_ngram": self.too_short,
            "ngrams": len(self.hashes),
            "shared_ngrams_ignored": self.shared,
        }


def lsh_bands(threshold: float, num_perm: int = NUM_PERM) -> tuple[int, int]:
    """(bands, rows per band) of the locality-sensitive hashing for a Jaccard threshold: the
    split of the signature that minimises the summed probability of missing a pair above the
    threshold and of proposing one below it (the usual S-curve trade-off)."""

    def integrate(f, a: float, b: float, steps: int = 200) -> float:
        width = (b - a) / steps
        return sum(f(a + (i + 0.5) * width) for i in range(steps)) * width

    best, best_error = (num_perm, 1), float("inf")
    for rows in range(1, num_perm + 1):
        bands = num_perm // rows
        false_pos = integrate(lambda s, b=bands, r=rows: 1.0 - (1.0 - s**r) ** b, 0.0, threshold)
        false_neg = integrate(lambda s, b=bands, r=rows: (1.0 - s**r) ** b, threshold, 1.0)
        if false_pos + false_neg < best_error:
            best, best_error = (bands, rows), false_pos + false_neg
    return best


class NearDuplicates:
    """Greedy near-duplicate detection: `check(text)` returns the id of an earlier kept document
    whose estimated Jaccard similarity (word shingles, MinHash) with the text is at least
    `threshold`, or None, in which case the text is kept and indexed under `doc_id`."""

    def __init__(self, threshold: float, num_perm: int = NUM_PERM, shingle_words: int = SHINGLE_WORDS, seed: int = 0):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"the near-duplicate threshold must be in (0, 1], got {threshold}")
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover - numpy comes with the data extras
            raise ImportError("near-duplicate detection (near_dedupe) needs numpy") from e
        self._np = np
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_words = shingle_words
        self.bands, self.rows = lsh_bands(threshold, num_perm)
        rng = np.random.default_rng(seed)
        # multiply-shift hashing: odd 64-bit multipliers, the high 32 bits of a * h + b
        self._a = (rng.integers(0, 1 << 63, size=(num_perm, 1), dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
        self._b = rng.integers(0, 1 << 63, size=(num_perm, 1), dtype=np.uint64)
        self._word_hash = _WordHashes()
        self._buckets: list[dict[bytes, list[int]]] = [{} for _ in range(self.bands)]
        self._signatures: list[Any] = []
        self._ids: list[Any] = []

    def signature(self, text: str):
        """The MinHash signature of the text's shingle set (None for a text without words)."""
        np = self._np
        tokens = words(text)
        if not tokens:
            return None
        n = min(self.shingle_words, len(tokens))  # a text shorter than a shingle is one shingle
        shingles = np.fromiter(set(ngram_hashes(tokens, n, self._word_hash)), dtype=np.uint64)
        signature = np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)
        with np.errstate(over="ignore"):
            for start in range(0, shingles.size, 8192):  # bounded memory on very long documents
                mixed = (self._a * shingles[None, start : start + 8192] + self._b) >> np.uint64(32)
                signature = np.minimum(signature, mixed.min(axis=1).astype(np.uint32))
        return signature

    def check(self, text: str, doc_id: Any) -> Any | None:
        np = self._np
        signature = self.signature(text)
        if signature is None:
            return None
        keys = [signature[b * self.rows : (b + 1) * self.rows].tobytes() for b in range(self.bands)]
        candidates: set[int] = set()
        for bucket, key in zip(self._buckets, keys):
            candidates.update(bucket.get(key, ()))
        for index in sorted(candidates):  # the earliest match, whatever the set's order
            if float(np.mean(self._signatures[index] == signature)) >= self.threshold:
                return self._ids[index]
        index = len(self._signatures)
        self._signatures.append(signature)
        self._ids.append(doc_id)
        for bucket, key in zip(self._buckets, keys):
            bucket.setdefault(key, []).append(index)
        return None


def clean_documents(
    samples: list[dict],
    report: CleanReport,
    text_field: str = "text",
    dedupe: bool = True,
    near_dedupe: float = 0.0,
    bench: NgramIndex | None = None,
    min_chars: int = 0,
    seed: int = 0,
) -> list[dict]:
    """The documents that pass the checks, in their order; every other one is recorded in
    `report` with its reason: empty_text (no string in `text_field`), too_short (fewer than
    min_chars characters), contaminated (an n-gram of `bench`), duplicate, near_duplicate
    (with the row it duplicates). The first document of a set of duplicates is kept."""
    near = NearDuplicates(near_dedupe, seed=seed) if near_dedupe > 0 else None
    first: dict[str, Any] = {}
    kept = []
    for i, sample in enumerate(samples):
        text = sample.get(text_field)
        if not isinstance(text, str) or not text.strip():
            report.drop(sample, "empty_text")
            continue
        if min_chars and len(text) < min_chars:
            report.drop(sample, "too_short")
            continue
        if bench is not None and bench.contaminated(text):
            report.drop(sample, "contaminated")
            continue
        if dedupe:
            key = text_key(text)
            if key in first:
                report.drop(sample, "duplicate", duplicate_of=first[key])
                continue
            first[key] = sample.get(ROW_INDEX)
        if near is not None:
            other = near.check(text, sample.get(ROW_INDEX))
            if other is not None:
                report.drop(sample, "near_duplicate", duplicate_of=other)
                continue
        kept.append(sample)
        if i and i % 50_000 == 0:
            logger.info(f"  Cleaned {i}/{len(samples)} documents ({len(kept)} kept)")
    return kept
