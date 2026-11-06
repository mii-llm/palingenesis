"""`pgs distill-cache`: precompute a teacher's top-k targets for every training document.

    pgs distill-cache --config distill.yaml --teacher big --out caches/big            # HF teacher, one GPU
    torchrun --nproc_per_node 8 -m palingenesis.distill.build --config distill.yaml --teacher big
    pgs distill-cache --config distill.yaml --teacher big --backend vllm --top_k 64   # vLLM prompt log-probs

The documents are the training config's own (data.dataset / data.sources / data.pretokenize, its
masking, its teacher view), so the trainer finds each of them by content (cache.py). The teacher is
`distill.teachers.<name>.model` (with its view: context distillation is baked into the cache). The
cache goes to `--out` or to the teacher's `cache` path; interrupted builds resume, finished documents
are skipped. With torchrun every rank scores its shard of the rows and writes its own shards.

Backends:
  hf     a transformers model on this rank's GPU: full-vocabulary log-softmax, then top-k (any temperature)
  vllm   vLLM's prompt log-probs (prefill only, max_tokens=1): the top-k and the gold token's log-prob at
         every position, at temperature 1. Needs the vllm extra; much faster for large teachers
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Iterator

import numpy as np
import torch

from palingenesis.distill.cache import CacheMeta, CacheWriter, sequence_key, trained_positions, vocab_fingerprint
from palingenesis.distill.config import TeacherSpec, parse_teachers
from palingenesis.distill.views import TeacherView, input_field, positions_field

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


# ─────────────────────────────────────────────────────────────── documents


def iter_documents(config, tokenizer, views: list | None = None, rank: int = 0, world_size: int = 1) -> Iterator[dict]:
    """Every training document of `config`, unpacked, unshuffled, this rank's share: every row of every
    source once (a source's weight decides how often it is drawn in training, not whether it is)."""
    from palingenesis.data import (
        PRETOK_DATA,
        PretrainDataset,
        _load_dataset_source,
        pretokenize_fingerprint,
        pretokenized_cache_valid,
        source_stage,
    )

    data = config.data
    if data.pretokenize:
        from pathlib import Path

        from datasets import load_dataset

        valid, reason = pretokenized_cache_valid(data.pretokenize_path, pretokenize_fingerprint(config, tokenizer))
        if not valid:
            raise RuntimeError(f"data.pretokenize: no valid pre-tokenized cache ({reason}); run pgs train once first")
        if views:
            raise RuntimeError("a teacher view needs the conversations: not available from a pre-tokenized cache")
        rows = load_dataset("parquet", data_files=str(Path(data.pretokenize_path) / PRETOK_DATA), split="train")
        for i, row in enumerate(rows):
            if i % world_size != rank:
                continue
            ids, labels = torch.tensor(row["input_ids"]), torch.tensor(row["labels"])
            pos = row.get("position_ids")
            starts = [0] if pos is None else [j for j, p in enumerate(pos) if p == 0]
            ends = starts[1:] + [len(ids)]
            for a, b in zip(starts, ends):
                yield {"input_ids": ids[a:b], "labels": labels[a:b]}
        return

    stages = []
    if data.sources:
        for src in data.sources:
            raw = _load_dataset_source(src["dataset"], src.get("split", "train"), streaming=data.streaming)
            stages.append(source_stage(raw, tokenizer, data, src, rank, world_size, teacher_views=views))
    else:
        dataset_id = data.dataset
        if config.preprocess.enabled:
            from palingenesis.prepare import find_prepared_dataset

            dataset_id = str(find_prepared_dataset(config.preprocess.output_dir))
        raw = _load_dataset_source(dataset_id, data.dataset_split, data.streaming)
        stages.append(source_stage(raw, tokenizer, data, None, rank, world_size, teacher_views=views))
    if data.pretrain_replay_dataset:
        replay = _load_dataset_source(data.pretrain_replay_dataset, "train", streaming=True)
        stages.append(PretrainDataset(replay, tokenizer, data.max_seq_length, rank=rank, world_size=world_size))
        logger.warning("data.pretrain_replay_dataset: the whole replay corpus is scored (it is a stream)")
    for stage in stages:
        yield from stage


# ─────────────────────────────────────────────────────────────── scoring


def _inverse_swap(bridge) -> dict[int, int]:
    return {t: s for s, t in (bridge.swap or {}).items()}


def _to_student_ids(ids: torch.Tensor, inverse: dict[int, int], student_vocab: int) -> torch.Tensor:
    out = ids.clone()
    for t, s in inverse.items():
        out[ids == t] = s
    out[(out < 0) | (out >= student_vocab)] = -1
    return out


@torch.no_grad()
def score_hf(teacher, docs: list[dict], name: str, k: int, temperature: float, student_vocab: int) -> list[tuple]:
    """(positions, ids, lp, gold_lp) per document from an online teacher's full distribution."""
    from palingenesis.distill.chunked import rows_per_slice
    from palingenesis.distill.teachers import DTYPES

    inverse = _inverse_swap(teacher.bridge)
    swap = teacher.bridge.swap or {}
    seqs, rows_t, positions, golds = [], [], [], []
    for doc in docs:
        s_pos = torch.from_numpy(trained_positions(doc["labels"].numpy())).long()
        if input_field(name) in doc:
            t_ids = doc[input_field(name)]
            q = doc[positions_field(name)][s_pos]
            keep = q >= 0
            s_pos, q = s_pos[keep], q[keep]
        else:
            t_ids, q = doc["input_ids"], s_pos
        gold = doc["input_ids"][s_pos + 1]
        seqs.append(t_ids)
        rows_t.append(q)
        positions.append(s_pos)
        golds.append(torch.tensor([swap.get(int(g), int(g)) for g in gold], dtype=torch.long))
    width = max(int(s.numel()) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : s.numel()] = s
    dtype = DTYPES[teacher.spec.dtype]
    hidden = teacher.hidden_states(
        ids, {"attention_mask": None}, torch.bfloat16 if dtype != torch.float32 else dtype, True
    )
    flat = hidden.reshape(-1, hidden.shape[-1])
    index = torch.cat([i * width + q for i, q in enumerate(rows_t)]).to(flat.device)
    gold_all = torch.cat(golds).to(flat.device)
    h = flat.index_select(0, index)
    step = rows_per_slice(teacher.vocab_size)
    top_ids, top_lp, gold_lp = [], [], []
    for a in range(0, h.shape[0], step):
        lp = torch.log_softmax(teacher.logits(h[a : a + step]) / temperature, -1)
        top = lp.topk(k, -1)
        top_ids.append(top.indices.cpu())
        top_lp.append(top.values.float().cpu())
        g = gold_all[a : a + step]
        valid = g < lp.shape[-1]
        gold_lp.append(
            torch.where(valid, lp.gather(1, g.clamp(max=lp.shape[-1] - 1)[:, None]).squeeze(1), float("nan")).cpu()
        )
    top_ids = _to_student_ids(
        torch.cat(top_ids) if top_ids else torch.zeros((0, k), dtype=torch.long), inverse, student_vocab
    )
    top_lp = torch.cat(top_lp) if top_lp else torch.zeros((0, k))
    gold_lp = torch.cat(gold_lp) if gold_lp else torch.zeros(0)
    out, offset = [], 0
    for s_pos in positions:
        n = s_pos.numel()
        sl = slice(offset, offset + n)
        out.append((s_pos.numpy(), top_ids[sl].numpy(), top_lp[sl].numpy(), gold_lp[sl].numpy()))
        offset += n
    return out


def parse_prompt_logprobs(entries: list, teacher_ids: list[int], rows: list[int], k: int) -> tuple[np.ndarray, ...]:
    """vLLM prompt log-probs -> (ids [n, k], lp [n, k], gold_lp [n]) for the teacher positions `rows`
    (position q predicts token q+1). `entries[j]` is vLLM's {token_id: Logprob(logprob, rank, ...)} for
    token j (None for j = 0); it holds the top-k and the actual token."""
    ids = np.full((len(rows), k), -1, dtype=np.int64)
    lp = np.full((len(rows), k), -1e4, dtype=np.float32)
    gold = np.full(len(rows), np.nan, dtype=np.float32)
    for i, q in enumerate(rows):
        entry = entries[q + 1] or {}
        ranked = sorted(entry.items(), key=lambda kv: getattr(kv[1], "rank", None) or 10**9)
        top = [(t, e) for t, e in ranked if (getattr(e, "rank", None) or 10**9) <= k][:k]
        for j, (t, e) in enumerate(top):
            ids[i, j] = int(t)
            lp[i, j] = max(float(e.logprob), -1e4) if e.logprob == e.logprob else -1e4
        actual = entry.get(teacher_ids[q + 1])
        if actual is not None and actual.logprob == actual.logprob:
            gold[i] = max(float(actual.logprob), -1e4)
    return ids, lp, gold


def score_vllm(llm, docs: list[dict], name: str, k: int, bridge, student_vocab: int) -> list[tuple]:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, prompt_logprobs=k, temperature=1.0, detokenize=False)
    inverse = _inverse_swap(bridge)
    prompts, plans = [], []
    for doc in docs:
        s_pos = trained_positions(doc["labels"].numpy())
        if input_field(name) in doc:
            t_ids = doc[input_field(name)].tolist()
            q = doc[positions_field(name)].numpy()[s_pos]
            keep = q >= 0
            s_pos, q = s_pos[keep], q[keep]
        else:
            t_ids, q = doc["input_ids"].tolist(), s_pos
        prompts.append({"prompt_token_ids": t_ids})
        plans.append((s_pos, q.tolist(), t_ids))
    outputs = llm.generate(prompts, params, use_tqdm=False)
    out = []
    for (s_pos, q, t_ids), result in zip(plans, outputs):
        ids, lp, gold = parse_prompt_logprobs(result.prompt_logprobs, t_ids, q, k)
        ids = _to_student_ids(torch.from_numpy(ids), inverse, student_vocab).numpy()
        out.append((s_pos, ids, lp, gold))
    return out


# ─────────────────────────────────────────────────────────────── the build


def build_cache(
    config,
    teacher: str,
    out: str = "",
    backend: str = "hf",
    top_k: int = 0,
    temperature: float = 1.0,
    max_batch_tokens: int = 32768,
    max_docs: int = 0,
    lp_dtype: str = "float16",
    vllm_options: dict | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    student_tokenizer=None,
) -> dict:
    """Build (or continue) the top-k cache of teacher `teacher` over `config`'s training documents.
    Returns counts: documents written, skipped (already cached), rows, seconds."""
    from transformers import AutoTokenizer

    from palingenesis.distill.teachers import OnlineTeacher, shared_vocabulary

    specs = {s.name: s for s in parse_teachers(config.distill.teachers)}
    if teacher not in specs:
        raise ValueError(f"no teacher {teacher!r} in distill.teachers ({', '.join(specs) or 'none'})")
    spec: TeacherSpec = specs[teacher]
    if not spec.model:
        raise ValueError(f"distill.teachers.{teacher} has no `model` to build its cache with")
    out = out or spec.cache
    if not out:
        raise ValueError(f"no output directory: pass --out or set distill.teachers.{teacher}.cache")
    if backend == "vllm" and temperature != 1.0:
        raise ValueError("the vllm backend returns log-probs at temperature 1")
    k = top_k or spec.top_k
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    tok = student_tokenizer or AutoTokenizer.from_pretrained(
        config.model.name_or_path, trust_remote_code=config.model.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    student_vocab = len(tok)

    online = None
    if backend == "hf":
        online = OnlineTeacher(
            spec,
            tok,
            device,
            attn_implementation=config.model.attn_implementation,
            trust_remote_code=config.model.trust_remote_code,
            model=teacher_model,
            tokenizer=teacher_tokenizer,
        )
        bridge, teacher_tok = online.bridge, online.tokenizer
    elif backend == "vllm":
        teacher_tok = teacher_tokenizer or AutoTokenizer.from_pretrained(spec.tokenizer or spec.model)
        bridge = shared_vocabulary(tok, teacher_tok, spec)
    else:
        raise ValueError(f"unknown backend {backend!r} (hf or vllm)")

    view = TeacherView(spec, bridge) if spec.has_view else None
    views = [(view, teacher_tok, spec.max_seq_length or 2 * config.data.max_seq_length)] if view else None
    meta = CacheMeta(
        top_k=k,
        vocab_hash=vocab_fingerprint(tok),
        vocab_size=student_vocab,
        teacher=spec.model,
        temperature=float(temperature),
        lp_dtype=lp_dtype,
        view=view.describe() if view else None,
        backend=backend,
    )
    writer = CacheWriter(out, meta, rank=rank)
    if writer.resumed and rank == 0:
        logger.info("Resuming the cache in %s: %d documents already done", out, writer.resumed)

    llm = None
    if backend == "vllm":
        from vllm import LLM

        options = {"gpu_memory_utilization": 0.85, **(vllm_options or {})}
        llm = LLM(model=spec.model, tokenizer=spec.tokenizer or spec.model, max_logprobs=k, dtype=spec.dtype, **options)

    def score(batch):
        if llm is not None:
            return score_vllm(llm, batch, teacher, k, bridge, student_vocab)
        return score_hf(online, batch, teacher, k, temperature, student_vocab)

    start = time.perf_counter()
    seen = 0
    pending: list[tuple[dict, int, int]] = []

    def flush():
        if not pending:
            return
        results = score([d for d, _, _ in pending])
        for (doc, key, check), (pos, ids, lp, gold) in zip(pending, results):
            writer.add(key, check, int(doc["input_ids"].numel()), pos, ids, lp, gold)
        pending.clear()

    buffer: list[dict] = []
    budget = max_batch_tokens
    for doc in iter_documents(config, tok, views, rank, world):
        if max_docs and seen >= max_docs:
            break
        seen += 1
        if not bool((doc["labels"][1:] != IGNORE_INDEX).any()):
            continue
        buffer.append(doc)
        if len(buffer) >= 256:
            _drain(buffer, pending, writer, budget, flush, name=teacher)
            buffer = []
        if seen % 1000 == 0:
            logger.info("rank %d: %d documents read, %d written", rank, seen, writer.written_docs)
    _drain(buffer, pending, writer, budget, flush, name=teacher)
    flush()
    skipped = seen - writer.written_docs
    writer.close()
    stats = {
        "documents": writer.written_docs,
        "skipped": skipped,
        "rows": writer.written_rows,
        "seconds": time.perf_counter() - start,
    }
    logger.info("rank %d: cache %s: %s", rank, out, stats)
    return stats


def _drain(buffer, pending, writer, budget, flush, name):
    """Score a buffer of documents in length-sorted batches of at most `budget` padded tokens."""
    buffer.sort(key=lambda d: int(d.get(input_field(name), d["input_ids"]).numel()))
    for doc in buffer:
        key, check = sequence_key(doc["input_ids"].numpy(), doc["labels"].numpy())
        if writer.done(key):
            continue
        length = int(doc.get(input_field(name), doc["input_ids"]).numel())
        widest = max([length] + [int(d.get(input_field(name), d["input_ids"]).numel()) for d, _, _ in pending])
        if pending and (len(pending) + 1) * widest > budget:
            flush()
        pending.append((doc, key, check))


def main(argv: list[str] | None = None) -> int:
    from palingenesis.config import Config

    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="pgs distill-cache", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher", required=True, help="a name in distill.teachers (with `model`)")
    ap.add_argument("--out", default="", help="cache directory (default: the teacher's `cache`)")
    ap.add_argument("--backend", default="hf", choices=("hf", "vllm"))
    ap.add_argument("--top_k", type=int, default=0, help="log-probs kept per position (default: the teacher's top_k)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_batch_tokens", type=int, default=32768, help="hf: padded tokens per teacher forward")
    ap.add_argument("--max_docs", type=int, default=0, help="stop after this many documents (0 = all)")
    ap.add_argument("--lp_dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vllm")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="vllm")
    args, rest = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_cli(["--config", args.config, *rest])
    config.validate()
    stats = build_cache(
        config,
        args.teacher,
        out=args.out,
        backend=args.backend,
        top_k=args.top_k,
        temperature=args.temperature,
        max_batch_tokens=args.max_batch_tokens,
        max_docs=args.max_docs,
        lp_dtype=args.lp_dtype,
        vllm_options={
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    )
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
