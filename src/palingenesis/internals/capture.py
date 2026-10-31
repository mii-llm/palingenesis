"""Capture residuals for a dataset and stream them to a sharded store on disk.

    store = capture_dataset(model, tok, rows, "acts/base_train", layers="all", position="last_prompt")
    x = store.layer(18)             # [N, d] float32, every shard of residual 18
    y, groups = store.labels(), store.groups()

Store layout (one directory, append-only while it is written, readable layer by layer):

    meta.json                  model, fingerprint, position, layers, hidden size, dtype, chat_template_kwargs, shards,
                               dropped rows by reason, the library version
    rows.jsonl                 one line per output row: source row index, id, label, group, n_tokens, token position
    shard_00000.safetensors    {"r<k>": [rows, d]} for every captured residual k

A shard is written every `shard_rows` output rows, so host memory is bounded by one shard (shard_rows x layers x d),
whatever the dataset size; reading loads one residual at a time (safetensors slices), so a 25-layer sweep never holds
more than one layer. Storage dtype defaults to bfloat16: residual streams carry outlier features in the thousands
(massive activations), which float16 can overflow; probe and CKA statistics are far coarser than bf16's resolution.

Rows are processed shortest first in batches of at most `batch_size` rows and `max_batch_tokens` padded tokens, with
right padding (see palingenesis.internals.positions), under torch.no_grad, without a KV cache, and the forward stops
after the deepest residual read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.nn as nn

from palingenesis.internals.hooks import Recorder, StopForward
from palingenesis.internals.positions import Encoded, Encoder, collate
from palingenesis.internals.sites import find_sites, parse_layers, text_config

logger = logging.getLogger(__name__)

STORE_VERSION = 1
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    """JSONL (or a JSON list) of rows: {"messages": [...], "label"?: 0/1 or a float, "group"?: str, "id"?: ...}."""
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: no rows")
    return rows[:limit] if limit else rows


def model_fingerprint(model: nn.Module, tok=None) -> dict:
    """What a probe must match to be applied to another model: architecture sizes always; the name and a hash of the
    first and last decoder layer's first parameter, to tell checkpoints apart; the chat template's hash."""
    cfg = text_config(model)
    sites = find_sites(model)
    digest = hashlib.sha256()
    for layer in (sites.layers[0], sites.layers[-1]):
        p = next(layer.parameters(), None)
        if p is not None and not p.is_meta:
            flat = p.detach().reshape(-1)[:4096].float().cpu()
            digest.update(flat.numpy().tobytes())
    template = getattr(tok, "chat_template", None) if tok is not None else None
    return {
        "name_or_path": getattr(model.config, "_name_or_path", "") or "",
        "model_type": getattr(model.config, "model_type", ""),
        "hidden_size": int(getattr(cfg, "hidden_size", 0) or 0),
        "num_layers": sites.n_layers,
        "layer_types": sites.layer_types,
        "weights_sha": digest.hexdigest()[:16],
        "chat_template_sha": hashlib.sha256(str(template).encode()).hexdigest()[:16] if template else None,
    }


def run_backbone(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    """A forward through the backbone (no output head, no KV cache); a Recorder with stop=True ends it early."""
    from palingenesis.logits import final_hidden_states

    try:
        final_hidden_states(model, input_ids, attention_mask, use_cache=False)
    except StopForward:
        pass


def _batches(items: list[tuple[int, Encoded]], batch_size: int, max_batch_tokens: int) -> Iterator[list]:
    order = sorted(items, key=lambda it: len(it[1].ids))
    batch: list = []
    for item in order:
        width = max([len(item[1].ids)] + [len(b[1].ids) for b in batch])
        if batch and (len(batch) >= batch_size or width * (len(batch) + 1) > max_batch_tokens):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


class StoreWriter:
    """Accumulates output rows and writes a shard every `shard_rows` rows."""

    def __init__(self, path: str | Path, meta: dict, shard_rows: int = 4096, overwrite: bool = False):
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.path} exists and is not empty (overwrite to replace it)")
            for f in self.path.iterdir():  # file by file: never a recursive delete
                if f.is_file() and (f.name in ("meta.json", "rows.jsonl") or f.name.startswith("shard_")):
                    f.unlink()
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta = {**meta, "version": STORE_VERSION, "shards": [], "n": 0}
        self.dtype = DTYPES[meta["dtype"]]
        self.shard_rows = shard_rows
        self._buf: dict[int, list[torch.Tensor]] = {}
        self._buffered = 0
        self._rows = (self.path / "rows.jsonl").open("w")

    def add(self, acts: dict[int, torch.Tensor], rows: list[dict]) -> None:
        for r, x in acts.items():
            self._buf.setdefault(r, []).append(x.to("cpu", self.dtype))
        for row in rows:
            self._rows.write(json.dumps(row) + "\n")
        self._buffered += len(rows)
        if self._buffered >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffered:
            return
        from safetensors.torch import save_file

        name = f"shard_{len(self.meta['shards']):05d}.safetensors"
        save_file({f"r{r}": torch.cat(xs).contiguous() for r, xs in self._buf.items()}, str(self.path / name))
        self.meta["shards"].append({"file": name, "rows": self._buffered})
        self.meta["n"] += self._buffered
        self._buf, self._buffered = {}, 0
        self._rows.flush()

    def close(self) -> ActivationStore:
        self.flush()
        self._rows.close()
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return ActivationStore(self.path)


class ActivationStore:
    """A captured store (see the module docstring), read one residual at a time."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta_path = self.path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{self.path}: not an activation store (no meta.json; was the capture finished?)")
        self.meta = json.loads(meta_path.read_text())
        with (self.path / "rows.jsonl").open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if len(self.rows) != self.meta["n"]:
            raise ValueError(f"{self.path}: rows.jsonl has {len(self.rows)} rows, meta.json {self.meta['n']}")

    def __len__(self) -> int:
        return self.meta["n"]

    @property
    def layers(self) -> list[int]:
        return list(self.meta["layers"])

    @property
    def position(self) -> str:
        return self.meta["position"]

    def layer(self, r: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Residual r of every row, [N, d]."""
        from safetensors import safe_open

        if r not in self.layers:
            raise KeyError(f"{self.path}: residual {r} not captured (have {self.layers})")
        parts = []
        for shard in self.meta["shards"]:
            with safe_open(str(self.path / shard["file"]), framework="pt") as f:
                parts.append(f.get_tensor(f"r{r}").to(dtype))
        return torch.cat(parts)

    def labels(self) -> torch.Tensor:
        """[N] float labels (NaN where a row has none)."""
        return torch.tensor([float("nan") if r.get("label") is None else float(r["label"]) for r in self.rows])

    def groups(self) -> list[str]:
        """Group of each output row: the row's `group`, else its `id`, else its source row index. Every output row of
        one source row (position *_all) shares the group, so a split never separates tokens of one conversation."""
        return [str(r["group"] if r.get("group") is not None else r.get("id", r["row"])) for r in self.rows]

    def n_tokens(self) -> torch.Tensor:
        return torch.tensor([float(r["n_tokens"]) for r in self.rows])


@torch.no_grad()
def capture_dataset(
    model: nn.Module,
    tok,
    rows: Iterable[dict],
    out: str | Path,
    *,
    layers: str | list[int] | None = "all",
    position: str = "last_prompt",
    chat_template_kwargs: dict | None = None,
    batch_size: int = 16,
    max_batch_tokens: int = 32768,
    max_tokens: int | None = None,
    dtype: str = "bfloat16",
    shard_rows: int = 4096,
    overwrite: bool = False,
    extra_meta: dict | None = None,
) -> ActivationStore:
    """Capture residuals `layers` at `position` for every row and write a store at `out` (see the module docstring).
    Rows without anything at the position, or longer than max_tokens, are dropped and counted in meta.json."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, not {dtype!r}")
    sites = find_sites(model)
    rs = parse_layers(layers, sites.n_residuals)
    device = next(model.parameters()).device
    encoder = Encoder(tok, position, chat_template_kwargs, max_tokens)
    rows = list(rows)
    items = []
    for i, row in enumerate(rows):
        enc = encoder.encode(row)
        if enc is not None:
            items.append((i, enc))
    if not items:
        raise ValueError(f"internals capture: no row has anything at position {position!r} ({encoder.dropped})")
    meta = {
        "model": model_fingerprint(model, tok),
        "position": position,
        "layers": rs,
        "mixers": [sites.mixer(r) for r in rs],
        "hidden_size": int(getattr(text_config(model), "hidden_size", 0) or 0),
        "dtype": dtype,
        "chat_template_kwargs": encoder.chat_template_kwargs,
        "source_rows": len(rows),
        "dropped": encoder.dropped,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra_meta or {}),
    }
    from palingenesis import __version__

    meta["palingenesis"] = __version__
    writer = StoreWriter(out, meta, shard_rows=shard_rows, overwrite=overwrite)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    was_training = model.training
    model.eval()
    t0, done = time.time(), 0
    try:
        with Recorder(model, rs, stop=True, sites=sites) as rec:
            for batch_items in _batches(items, batch_size, max_batch_tokens):
                batch = collate([e for _, e in batch_items], pad)
                rec.select = batch.select
                run_backbone(model, batch.input_ids.to(device), batch.attention_mask.to(device))
                acts = rec.take()
                out_rows = []
                for b, p in batch.owners:
                    i, enc = batch_items[b]
                    src = rows[i]
                    out_rows.append(
                        {
                            "row": i,
                            "id": src.get("id"),
                            "label": src.get("label"),
                            "group": src.get("group"),
                            "n_tokens": len(enc.ids),
                            "token": p,
                        }
                    )
                writer.add(acts, out_rows)
                done += len(batch_items)
        logger.info(
            "internals capture: %d rows -> %s in %.1fs (dropped %s)", done, out, time.time() - t0, encoder.dropped
        )
    finally:
        model.train(was_training)
    return writer.close()
