"""A teacher's sparse targets on disk: top-k log-probabilities at every trained position.

Written once by `pgs distill-cache` (build.py), read by the trainer while it trains. The cache is
**content-addressed**: each training sequence (a document of the data pipeline, before packing) is
keyed by a hash of its token ids and its loss mask, so the trainer finds a document's targets
whatever the data order, the shuffling, the packing, the number of ranks or a resume; a document
the cache does not hold is a miss (an error by default: the data or its masking changed).

Layout of a cache directory:

    meta.json                       format, top_k, the student tokenizer's vocabulary fingerprint,
                                    the teacher, its view, the temperature, the dtype of log-probs
    shard-r000-00000.index.npy      per document: key, check, first row, rows, sequence length
    shard-r000-00000.pos.bin        int32 [rows]          position t of each row (predicts token t+1)
    shard-r000-00000.ids.bin        int32 [rows, k]       top-k token ids in the STUDENT's vocabulary
    shard-r000-00000.lp.bin         float16 [rows, k]     their teacher log-probabilities
    shard-r000-00000.gold.bin       float32 [rows]        the teacher's log-prob of the gold token

A shard is complete once its index exists (written last, atomically); an interrupted build resumes
by skipping the documents of complete shards. Each rank of a multi-GPU build writes its own shards.
Reads are memory-mapped: a trainer touches only the rows of the documents it trains on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

FORMAT = "palingenesis-topk-v1"
META = "meta.json"
IGNORE_INDEX = -100
INDEX_DTYPE = np.dtype([("key", "<u8"), ("check", "<u8"), ("start", "<i8"), ("rows", "<i4"), ("length", "<i4")])
LP_DTYPES = {"float16": np.float16, "float32": np.float32}
LP_FLOOR = -1e4  # stored log-probs are floored here (float16 holds down to -65504)


class CacheError(Exception):
    """The cache is missing, incomplete, or does not belong to this student / data."""


# ─────────────────────────────────────────────────────────────── identities


def sequence_key(input_ids, labels) -> tuple[int, int]:
    """(key, check): 128 bits of BLAKE2b over the token ids and which of them are trained."""
    ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int64))
    mask = np.ascontiguousarray(np.asarray(labels) != IGNORE_INDEX, dtype=np.uint8)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(ids.size).tobytes())
    h.update(ids.tobytes())
    h.update(mask.tobytes())
    digest = h.digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")


def vocab_fingerprint(tokenizer) -> str:
    """SHA-256 of the tokenizer's vocabulary (token -> id) and its size: two tokenizers with the same
    fingerprint map every id to the same token."""
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    h = hashlib.sha256()
    h.update(str(len(tokenizer)).encode())
    for token, idx in vocab:
        h.update(f"{idx}\x00{token}\x01".encode())
    return h.hexdigest()


def trained_positions(labels) -> np.ndarray:
    """Positions t (int32) whose next token t+1 is trained: the rows a teacher target exists for."""
    lab = np.asarray(labels)
    return (np.nonzero(lab[1:] != IGNORE_INDEX)[0]).astype(np.int32)


def doc_spans(attention_mask: torch.Tensor, position_ids: torch.Tensor | None) -> list[tuple[int, int]]:
    """(start, length) of each document in one batch row: the row up to its padding, or, packed, the
    stretches whose position ids restart at 0 (the collator's padding is a document of its own, with no
    trained token, so it is never looked up)."""
    n = int(attention_mask.sum())
    if position_ids is None:
        return [(0, n)] if n else []
    starts = torch.nonzero(position_ids[:n] == 0).flatten().tolist() or [0]
    if starts[0] != 0:
        starts = [0] + starts
    ends = starts[1:] + [n]
    return [(s, e - s) for s, e in zip(starts, ends) if e > s]


# ─────────────────────────────────────────────────────────────── writing


@dataclass
class CacheMeta:
    top_k: int
    vocab_hash: str
    vocab_size: int
    teacher: str
    temperature: float = 1.0
    lp_dtype: str = "float16"
    view: dict | None = None
    backend: str = "hf"
    format: str = FORMAT

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_json(cls, d: dict) -> CacheMeta:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def conflicts(self, other: CacheMeta) -> list[str]:
        keys = ("top_k", "vocab_hash", "vocab_size", "teacher", "temperature", "lp_dtype", "view", "format")
        return [k for k in keys if getattr(self, k) != getattr(other, k)]


class CacheWriter:
    """Appends documents' targets to shards of `directory` (this rank's), resumable.

    With an existing cache of the same settings it continues it: `done(key)` tells whether a document
    is already stored. A cache written with other settings is refused (a different teacher, k, vocab).
    """

    def __init__(self, directory: str | Path, meta: CacheMeta, rank: int = 0, shard_rows: int = 1 << 20):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.rank = rank
        self.shard_rows = shard_rows
        self.k = meta.top_k
        self.lp_dtype = LP_DTYPES[meta.lp_dtype]
        meta_path = self.dir / META
        if meta_path.exists():
            existing = CacheMeta.from_json(json.loads(meta_path.read_text()))
            bad = existing.conflicts(meta)
            if bad:
                raise CacheError(
                    f"{self.dir} holds a cache built with other settings ({', '.join(bad)}): "
                    "use another directory or delete it"
                )
        elif rank == 0:
            _atomic_write(meta_path, json.dumps(meta.to_json(), indent=2))
        self._discard_incomplete()
        self.keys: set[int] = set()
        n = -1
        for index_path in sorted(self.dir.glob("shard-r*.index.npy")):
            self.keys.update(int(k) for k in np.load(index_path)["key"])
            if index_path.name.startswith(f"shard-r{rank:03d}-"):
                n = max(n, int(index_path.name.split("-")[2].split(".")[0]))
        self.resumed = len(self.keys)
        self._next = n + 1
        self._open = None
        self.written_docs = 0
        self.written_rows = 0

    def _discard_incomplete(self) -> None:
        prefix = f"shard-r{self.rank:03d}-"
        complete = {p.name.split(".")[0] for p in self.dir.glob(f"{prefix}*.index.npy")}
        for p in self.dir.glob(f"{prefix}*"):
            if p.name.split(".")[0] not in complete:
                p.unlink()

    def done(self, key: int) -> bool:
        return key in self.keys

    def _start_shard(self) -> None:
        stem = f"shard-r{self.rank:03d}-{self._next:05d}"
        self._next += 1
        files = {part: open(self.dir / f"{stem}.{part}.bin.tmp", "wb") for part in ("pos", "ids", "lp", "gold")}
        self._open = {"stem": stem, "files": files, "index": [], "rows": 0}

    def add(self, key: int, check: int, length: int, positions, ids, lp, gold_lp) -> None:
        """One document: positions [n] (int), ids [n, k] (student ids, -1 for none), lp [n, k], gold_lp [n]."""
        if key in self.keys:
            return
        positions = np.asarray(positions, dtype=np.int32)
        n = positions.shape[0]
        ids = np.asarray(ids, dtype=np.int32).reshape(n, self.k)
        lp = np.maximum(np.nan_to_num(np.asarray(lp, dtype=np.float32), nan=LP_FLOOR), LP_FLOOR).reshape(n, self.k)
        gold = np.asarray(gold_lp, dtype=np.float32).reshape(n)
        if self._open is None or (self._open["rows"] and self._open["rows"] + n > self.shard_rows):
            self._finish_shard()
            self._start_shard()
        shard = self._open
        files = shard["files"]
        files["pos"].write(positions.tobytes())
        files["ids"].write(ids.tobytes())
        files["lp"].write(lp.astype(self.lp_dtype).tobytes())
        files["gold"].write(gold.tobytes())
        shard["index"].append((key, check, shard["rows"], n, length))
        shard["rows"] += n
        self.keys.add(key)
        self.written_docs += 1
        self.written_rows += n

    def _finish_shard(self) -> None:
        shard, self._open = self._open, None
        if shard is None:
            return
        for f in shard["files"].values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        stem = shard["stem"]
        if not shard["index"]:
            for part in shard["files"]:
                (self.dir / f"{stem}.{part}.bin.tmp").unlink()
            return
        for part in shard["files"]:
            os.replace(self.dir / f"{stem}.{part}.bin.tmp", self.dir / f"{stem}.{part}.bin")
        index = np.array(shard["index"], dtype=INDEX_DTYPE)
        tmp = self.dir / f"{stem}.index.tmp.npy"
        np.save(tmp, index)
        os.replace(tmp, self.dir / f"{stem}.index.npy")

    def flush(self) -> None:
        """Make everything added so far durable (a shard boundary)."""
        self._finish_shard()

    def close(self) -> None:
        self._finish_shard()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────── reading


@dataclass
class DocTargets:
    positions: np.ndarray  # [n] int32
    ids: np.ndarray  # [n, k] int32
    lp: np.ndarray  # [n, k] float
    gold_lp: np.ndarray  # [n] float32


class TopKCache:
    """Read side of a cache directory. Opened lazily (per process: DataLoader workers each map it)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        meta_path = self.dir / META
        if not meta_path.exists():
            raise CacheError(f"{self.dir}: not a distillation cache (no {META}); build one with pgs distill-cache")
        self.meta = CacheMeta.from_json(json.loads(meta_path.read_text()))
        if self.meta.format != FORMAT:
            raise CacheError(f"{self.dir}: cache format {self.meta.format!r}, this version reads {FORMAT!r}")
        self.k = self.meta.top_k
        self._loaded = False

    def verify_tokenizer(self, tokenizer) -> None:
        """Raise unless the cache was built for this tokenizer's vocabulary."""
        if self.meta.vocab_hash != vocab_fingerprint(tokenizer):
            raise CacheError(
                f"{self.dir} was built for another vocabulary (fingerprint {self.meta.vocab_hash[:12]}..., this "
                f"student's tokenizer {vocab_fingerprint(tokenizer)[:12]}...): rebuild it with this student's tokenizer"
            )

    def _load(self) -> None:
        if self._loaded:
            return
        indexes, shard_of, stems = [], [], []
        for i, path in enumerate(sorted(self.dir.glob("shard-r*.index.npy"))):
            idx = np.load(path)
            indexes.append(idx)
            shard_of.append(np.full(len(idx), i, dtype=np.int32))
            stems.append(path.name[: -len(".index.npy")])
        if indexes:
            index = np.concatenate(indexes)
            shard = np.concatenate(shard_of)
        else:
            index, shard = np.zeros(0, dtype=INDEX_DTYPE), np.zeros(0, dtype=np.int32)
        order = np.argsort(index["key"], kind="stable")
        self._index = index[order]
        self._shard = shard[order]
        self._stems = stems
        self._maps: dict[int, dict] = {}
        self._loaded = True

    def __len__(self) -> int:
        self._load()
        return len(self._index)

    def __getstate__(self):  # DataLoader workers re-map in their own process
        return {"dir": self.dir, "meta": self.meta, "k": self.k, "_loaded": False}

    def _shard_maps(self, i: int) -> dict:
        if i not in self._maps:
            stem = self.dir / self._stems[i]
            lp_dtype = LP_DTYPES[self.meta.lp_dtype]
            self._maps[i] = {
                "pos": np.memmap(f"{stem}.pos.bin", dtype=np.int32, mode="r"),
                "ids": np.memmap(f"{stem}.ids.bin", dtype=np.int32, mode="r").reshape(-1, self.k),
                "lp": np.memmap(f"{stem}.lp.bin", dtype=lp_dtype, mode="r").reshape(-1, self.k),
                "gold": np.memmap(f"{stem}.gold.bin", dtype=np.float32, mode="r"),
            }
        return self._maps[i]

    def get(self, key: int, check: int, length: int | None = None) -> DocTargets | None:
        """A document's targets, or None when the cache does not hold it."""
        self._load()
        keys = self._index["key"]
        i = int(np.searchsorted(keys, np.uint64(key)))
        while i < len(keys) and int(keys[i]) == key:
            rec = self._index[i]
            if int(rec["check"]) == check and (length is None or int(rec["length"]) == length):
                m = self._shard_maps(int(self._shard[i]))
                a, n = int(rec["start"]), int(rec["rows"])
                return DocTargets(m["pos"][a : a + n], m["ids"][a : a + n], m["lp"][a : a + n], m["gold"][a : a + n])
            i += 1
        return None


# ─────────────────────────────────────────────────────────────── batch lookup


def batch_field(name: str, part: str) -> str:
    return f"kd.{name}.{part}"


class CacheLookup:
    """A collate-time transform: adds each cached teacher's targets to a collated batch.

    For teacher `name` with top-k k it adds, aligned with the positions of input_ids (position t holds
    the targets for predicting token t+1):
        kd.<name>.ids    int32 [B, T, k]   (-1: no target)
        kd.<name>.lp     float16/32 [B, T, k]
        kd.<name>.gold   float32 [B, T]    teacher log-prob of the gold token (NaN: none)
        kd.<name>.hit    bool [B, T]       a target exists
    Runs in the DataLoader workers (memory-mapped reads), so it costs the training loop nothing.
    """

    def __init__(self, caches: dict[str, TopKCache], routes: dict[str, set[int]] | None = None):
        self.caches = caches
        # teacher -> data source indices it supervises (None / missing: all)
        self.routes = routes or {}
        self.misses: dict[str, int] = {name: 0 for name in caches}
        self.lookups = 0

    def __call__(self, batch: dict) -> dict:
        ids = batch["input_ids"]
        labels = batch["labels"]
        attn = batch["attention_mask"]
        pos = batch.get("position_ids")
        sources = batch.get("source_ids")
        bsz, seq = ids.shape
        for name, cache in self.caches.items():
            k = cache.k
            out_ids = torch.full((bsz, seq, k), -1, dtype=torch.int32)
            out_lp = torch.zeros(
                (bsz, seq, k), dtype=torch.float16 if cache.meta.lp_dtype == "float16" else torch.float32
            )
            out_gold = torch.full((bsz, seq), float("nan"), dtype=torch.float32)
            out_hit = torch.zeros((bsz, seq), dtype=torch.bool)
            route = self.routes.get(name)
            for b in range(bsz):
                for start, length in doc_spans(attn[b], pos[b] if pos is not None else None):
                    lab = labels[b, start : start + length]
                    if not bool((lab[1:] != IGNORE_INDEX).any()):
                        continue
                    if route is not None and sources is not None and int(sources[b, start]) not in route:
                        continue
                    self.lookups += 1
                    key, check = sequence_key(ids[b, start : start + length].numpy(), lab.numpy())
                    doc = cache.get(key, check, length)
                    if doc is None:
                        self.misses[name] += 1
                        continue
                    p = torch.from_numpy(np.asarray(doc.positions, dtype=np.int64)) + start
                    out_ids[b, p] = torch.from_numpy(np.array(doc.ids))
                    out_lp[b, p] = torch.from_numpy(np.array(doc.lp)).to(out_lp.dtype)
                    out_gold[b, p] = torch.from_numpy(np.array(doc.gold_lp))
                    out_hit[b, p] = True
            batch[batch_field(name, "ids")] = out_ids
            batch[batch_field(name, "lp")] = out_lp
            batch[batch_field(name, "gold")] = out_gold
            batch[batch_field(name, "hit")] = out_hit
        return batch
