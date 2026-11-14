"""Data preparation pipeline: offline cleaning, held-out split and selection.

Default pipeline (tokenizer only, no forward pass; see data_clean):
  1. drop the rows training would drop (cannot be rendered, nothing to train on, no
     trained turn fits max_seq_length) and count the ones it would cut
  2. drop exact duplicates; optionally cap the rows that share a prompt
  3. reserve a held-out eval set that shares no prompt with the training pool
  4. cap rows / tokens per group (source), then select a budget of rows or tokens
  5. write the data, `prepared_meta.json` (what was kept, dropped and why, token counts,
     per-group profile) and `dropped.jsonl` (every removed row with its reason)

Optional scoring (`score`, or any perplexity option): one forward pass with the model you
will fine-tune over the tokens training puts loss on. The score is the perplexity of the
sample's *response* under the model: low means the model would write something very close
to this answer already (familiar), high means the answer is far from what the model would
write (unfamiliar). It is NOT problem difficulty: on 20k NuminaMath problems, the Spearman
correlation between Qwen3.5-0.8B's pass rate and this score was 0.009.

The perplexity filter and the familiarity-based selection strategies are experimental: in
our measurements and in published replications (arXiv:2410.09335, 2410.15225) they do not
beat a random subset of the same size. Compare any of them with `random` on your own eval
battery (see the data guide).

Usage:
    # Standalone flags
    palingenesis prepare --model google/gemma-4-12B-it \\
                        --data your-org/agentic-traces \\
                        --output prepared/ \\
                        --budget 10000

    # Or drive everything from the SAME config used for training
    # (model.name_or_path, data.dataset/split/messages_field/max_seq_length
    #  and the preprocess: section are all read from the YAML):
    palingenesis prepare --config configs/qwen35_4b/a100_80gb.yaml
    # then train with preprocess.enabled=true and the prepared parquet is
    # picked up automatically:
    palingenesis train --config configs/qwen35_4b/a100_80gb.yaml
"""

import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _get_messages(sample: dict, messages_field: str) -> list[dict]:
    """Return messages normalized the same way ChatDataset does.

    This keeps `pgs prepare` aligned with `pgs train`/`pgs inspect` for
    ShareGPT-style datasets (`conversations` with `from`/`value`) and other
    non-standard schemas.
    """
    from palingenesis.validate_data import normalize_messages

    normalized = normalize_messages(sample, messages_field)
    if normalized:
        return normalized
    raw = sample.get(messages_field, [])
    return raw if isinstance(raw, list) else []


class Scorer:
    """The tokenizer and (lazily, once) the model of a prepare run. Cleaning needs only the
    tokenizer; the model is loaded on the first scoring call and shared by every source."""

    def __init__(self, model_name: str, device: str = "auto"):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading model: {self.model_name}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
        return self._model

    def chat(self, max_seq_length: int, messages_field: str, chat_options: dict | None = None):
        """The trainer's row pipeline (chat template, tools, truncation at turn boundaries,
        and the exact tokens that get loss, per-message flags and reasoning settings
        included), so prepare measures what training trains."""
        from palingenesis.data import ChatDataset

        return ChatDataset(None, self.tokenizer, max_seq_length, messages_field, **(chat_options or {}))


def score_samples_with_model(
    model_name: str,
    samples: list[dict],
    messages_field: str = "messages",
    max_seq_length: int = 8192,
    batch_size: int = 4,
    device: str = "auto",
    max_batch_tokens: int = 16384,
    chat_options: dict | None = None,
    hes_top_k_pct: float | None = None,
) -> list[dict]:
    """Score samples by the perplexity of their responses under the model.

    Uses the model you will fine-tune, on exactly the tokens training puts loss
    on. Low perplexity = the response is familiar to the model (close to what it
    would write); high = unfamiliar. This measures the response, not how hard
    the problem is. Rows training would drop are marked unscoreable (infinite
    perplexity). hes_top_k_pct: also write the High-Entropy Sum (see _score_padded_batch).
    """
    from palingenesis.data_clean import CleanReport, tokenize_rows

    scorer = Scorer(model_name, device)
    chat = scorer.chat(max_seq_length, messages_field, chat_options)
    for s in samples:
        _mark_unscoreable(s)
    kept, entries = tokenize_rows(samples, chat, CleanReport(), keep_tokens=True)
    score_entries(scorer, entries, kept, batch_size, max_batch_tokens, hes_top_k_pct)
    return samples


def score_entries(
    scorer: Scorer,
    entries: list[tuple[int, torch.Tensor, torch.Tensor]],
    samples: list[dict],
    batch_size: int = 4,
    max_batch_tokens: int = 16384,
    hes_top_k_pct: float | None = None,
) -> None:
    """Score tokenized rows in place: entries are (index into samples, input_ids, trained mask).

    Truly batched: entries are sorted by length and packed into padded batches
    (≤ batch_size samples AND ≤ max_batch_tokens padded tokens per forward: logits are
    B×S×V, so the token cap is what protects memory), one forward per batch. The
    High-Entropy Sum, when asked for, comes from the same forward.
    """
    entries = sorted(entries, key=lambda e: len(e[1]))
    batches = _group_entries_by_length(entries, batch_size, max_batch_tokens)
    logger.info(
        f"Scoring {len(entries)} samples in {len(batches)} batched forwards "
        f"(batch_size≤{batch_size}, ≤{max_batch_tokens} padded tokens per forward)"
    )
    model = scorer.model
    pad_id = scorer.tokenizer.pad_token_id
    scored_count = 0
    scored_tokens = 0
    total_tokens = sum(len(e[1]) for e in entries)
    t0 = time.perf_counter()
    for batch_num, batch in enumerate(batches):
        _score_padded_batch(model, batch, samples, pad_id, hes_top_k_pct)
        scored_count += len(batch)
        scored_tokens += sum(len(e[1]) for e in batch)
        if batch_num % 20 == 0 or batch_num == len(batches) - 1:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            rate = scored_count / elapsed
            tok_rate = scored_tokens / elapsed
            # ETA from token throughput, not sample throughput: batches are
            # length-sorted so samples/s drops as sequences get longer, while
            # tokens/s stays roughly constant.
            eta_s = (total_tokens - scored_tokens) / max(tok_rate, 1e-6)
            logger.info(
                f"  Scored {scored_count}/{len(entries)} samples "
                f"({rate:.1f} samples/s, {tok_rate / 1000:.1f}K tok/s, ETA {eta_s / 60:.0f}m)"
            )
    if hes_top_k_pct:
        _normalize_hes(samples)


def _mark_unscoreable(sample: dict) -> None:
    sample["_score_ppl"] = float("inf")
    sample["_score_response_ppl"] = float("inf")
    sample["_score_length"] = 0


def _group_entries_by_length(
    entries: list[tuple[int, list[int], torch.Tensor]],
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[tuple[int, list[int], torch.Tensor]]]:
    """Group length-sorted entries into batches.

    A batch holds at most batch_size samples AND at most max_batch_tokens
    padded tokens (n_samples × longest_seq_in_batch). Since entries are sorted
    by length, the longest sequence is always the most recently added one.
    Every entry lands in exactly one batch; a single over-long entry still gets
    its own batch of 1.
    """
    batches: list[list] = []
    current: list = []
    for entry in entries:
        seq_len = len(entry[1])
        if current and (len(current) >= batch_size or (len(current) + 1) * seq_len > max_batch_tokens):
            batches.append(current)
            current = []
        current.append(entry)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def _score_padded_batch(
    model,
    batch: list[tuple[int, list[int], torch.Tensor]],
    samples: list[dict],
    pad_id: int,
    hes_top_k_pct: float | None = None,
) -> None:
    """Forward one padded batch and write per-sample PPL scores in-place.

    Response perplexity is computed only on assistant tokens — the same tokens
    that receive loss during training. With hes_top_k_pct, the High-Entropy Sum
    (arXiv:2605.22389) is written too: the sum of the predictive entropies of the
    top-k% highest-entropy response tokens (`_score_hes`), and their mean
    (`_score_hes_mean`, comparable across response lengths, which the sum is not).
    """
    max_len = max(len(ids) for _, ids, _ in batch)
    n = len(batch)

    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((n, max_len), dtype=torch.long)
    resp_mask = torch.zeros((n, max_len), dtype=torch.bool)
    for row, (_, ids, rmask) in enumerate(batch):
        input_ids[row, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attn_mask[row, : len(ids)] = 1
        resp_mask[row, : len(rmask)] = rmask

    device = model.device
    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    # use_cache=False: a scoring forward never reuses the KV cache, and
    # building it costs both time (per-layer concat) and memory (GBs at
    # large batch sizes).
    logits = model(input_ids, attention_mask=attn_mask, use_cache=False).logits  # [B, S, V]

    # Shifted next-token NLL. Chunk over rows so the fp32 logits copy inside
    # cross_entropy stays bounded (full-batch .float() would double memory),
    # while avoiding one tiny kernel launch per row.
    shift_labels = input_ids[:, 1:]
    nll = torch.empty((n, max_len - 1), dtype=torch.float32, device=device)
    entropy = torch.empty_like(nll) if hes_top_k_pct else None
    rows_per_chunk = max(1, 8192 // max(max_len - 1, 1))
    for start in range(0, n, rows_per_chunk):
        end = min(start + rows_per_chunk, n)
        chunk = logits[start:end, :-1, :].float()
        if entropy is None:
            nll[start:end] = torch.nn.functional.cross_entropy(
                chunk.reshape(-1, chunk.size(-1)),
                shift_labels[start:end].reshape(-1),
                reduction="none",
            ).view(end - start, max_len - 1)
            continue
        # NLL and entropy from one log-softmax, with at most two fp32 copies alive (as above)
        log_probs = torch.nn.functional.log_softmax(chunk, dim=-1)
        del chunk
        nll[start:end] = -log_probs.gather(-1, shift_labels[start:end].unsqueeze(-1)).squeeze(-1)
        entropy[start:end] = -log_probs.exp().mul_(log_probs).sum(dim=-1)
        del log_probs
    del logits

    valid = attn_mask[:, 1:].bool()  # real (non-pad) prediction targets
    resp = resp_mask[:, 1:].to(device) & valid

    # Vectorized per-row masked means, then a single GPU→CPU transfer
    # (per-row .item() calls would sync the device hundreds of times per batch)
    valid_counts = valid.sum(dim=1).clamp(min=1)
    avg_nll_rows = (nll * valid).sum(dim=1) / valid_counts
    resp_counts = resp.sum(dim=1)
    resp_nll_rows = (nll * resp).sum(dim=1) / resp_counts.clamp(min=1)

    stats = torch.stack([avg_nll_rows, resp_nll_rows, resp_counts.float()]).cpu()
    if entropy is not None:
        # response positions first, in decreasing entropy; the rest sorts last
        ranked = entropy.masked_fill(~resp, -1.0).sort(dim=1, descending=True).values.cpu()

    for row, (sample_idx, ids, _) in enumerate(batch):
        sample = samples[sample_idx]
        avg_nll = stats[0, row].item()
        n_resp = int(stats[2, row].item())
        # Fallback to full-sequence NLL if mask detection failed
        response_nll = stats[1, row].item() if n_resp > 0 else avg_nll

        sample["_score_ppl"] = round(math.exp(min(avg_nll, 20.0)), 2)
        sample["_score_response_ppl"] = round(math.exp(min(response_nll, 20.0)), 2)
        sample["_score_length"] = len(ids)
        sample["_score_avg_nll"] = round(avg_nll, 4)
        sample["_score_response_nll"] = round(response_nll, 4)
        sample["_score_response_token_count"] = n_resp
        if entropy is not None:
            k = min(max(1, int(n_resp * hes_top_k_pct / 100.0)), n_resp)
            top = ranked[row, :k]
            sample["_score_hes"] = round(top.sum().item(), 4) if k else 0.0
            sample["_score_hes_mean"] = round(top.mean().item(), 4) if k else 0.0


def _normalize_hes(samples: list[dict]) -> None:
    """`_score_hes_normalized`: the High-Entropy Sum min-max scaled to [0, 1] over this dataset."""
    values = [s["_score_hes"] for s in samples if s.get("_score_hes", 0) > 0]
    if not values:
        return
    low, span = min(values), (max(values) - min(values)) or 1.0
    for s in samples:
        raw = s.get("_score_hes", 0)
        s["_score_hes_normalized"] = round((raw - low) / span, 4) if raw > 0 else 0.0


FAMILIARITY_BUCKETS = ("familiar", "typical", "unfamiliar")


def classify_familiarity(
    samples: list[dict],
    familiar_percentile: float = 25.0,
    unfamiliar_percentile: float = 75.0,
) -> list[dict]:
    """Bucket samples by response perplexity percentile into `_score_familiarity`:
    familiar (bottom 25%), typical, unfamiliar (top 25%). Percentiles adapt to any
    model/dataset pair. The score measures how close the response is to what the
    model would write, not problem difficulty."""
    ppls = [s["_score_response_ppl"] for s in samples if s.get("_score_response_ppl", float("inf")) < float("inf")]
    if not ppls:
        return samples

    ppls_sorted = sorted(ppls)
    n = len(ppls_sorted)
    low = ppls_sorted[int(n * familiar_percentile / 100)]
    high = ppls_sorted[int(n * unfamiliar_percentile / 100)]

    for s in samples:
        ppl = s.get("_score_response_ppl", float("inf"))
        if ppl <= low:
            s["_score_familiarity"] = "familiar"
        elif ppl >= high:
            s["_score_familiarity"] = "unfamiliar"
        else:
            s["_score_familiarity"] = "typical"

    return samples


def profile_groups(samples: list[dict], field: str) -> dict:
    """Per-group profile: count, tokens, mean response tokens and, when the rows are scored,
    median response NLL and familiarity buckets. Rows without the field go to "<none>"."""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(str(s.get(field, "<none>")), []).append(s)
    out = {}
    for name, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        nll = sorted(r["_score_response_nll"] for r in rows if isinstance(r.get("_score_response_nll"), float))
        toks = [r.get("_score_response_token_count", 0) for r in rows]
        fam: dict[str, int] = {}
        for r in rows:
            b = r.get("_score_familiarity", "unknown")
            fam[b] = fam.get(b, 0) + 1
        out[name] = {
            "count": len(rows),
            "tokens": sum(r.get("_score_length", 0) for r in rows),
            "mean_response_tokens": round(sum(toks) / len(toks), 1),
        }
        if nll:
            out[name]["median_response_nll"] = round(nll[len(nll) // 2], 4)
            out[name]["familiarity"] = fam
    return out


def filter_samples(
    samples: list[dict],
    min_ppl: float = 1.5,
    max_ppl: float = 500.0,
    min_length: int = 10,
    max_length: int = 32768,
    score: str = "response",
    on_drop=None,
) -> list[dict]:
    """Remove outliers by perplexity and length.

    - min_ppl: below it the response is near-verbatim what the model writes already
      (<= 0 disables). Short tool calls and templated answers also score this low.
    - max_ppl: above it the response is usually corrupted or in the wrong language
      (<= 0 disables). Absolute perplexity depends on the model, the domain and the
      language: read the distribution before choosing a ceiling.
    - min_length / max_length: tokens of the row as training sees it.

    Rows without a score (scoring was not run) pass the perplexity bounds; rows marked
    unscoreable are removed. on_drop(sample, reason) is called for every removed row.

    For chat SFT, filter on assistant-response perplexity by default. Training
    loss is applied to assistant tokens only, so filtering on full-sequence PPL
    can wrongly reject domain/language-shifted prompts even when responses are
    learnable and masking is healthy.
    """
    filtered = []
    removed = {"unscoreable": 0, "too_easy": 0, "too_hard": 0, "too_short": 0, "too_long": 0}
    score_key = "_score_response_ppl" if score == "response" else "_score_ppl"
    use_max_ppl = max_ppl > 0

    for s in samples:
        ppl = s.get(score_key, s.get("_score_ppl"))
        length = s.get("_score_length", 0)

        if ppl is not None and not math.isfinite(ppl):
            reason = "unscoreable"
        elif ppl is not None and ppl < min_ppl:
            reason = "too_easy"
        elif ppl is not None and use_max_ppl and ppl > max_ppl:
            reason = "too_hard"
        elif length < min_length:
            reason = "too_short"
        elif length > max_length:
            reason = "too_long"
        else:
            filtered.append(s)
            continue
        removed[reason] += 1
        if on_drop is not None:
            on_drop(s, reason)

    finite_ppls = [s.get(score_key, s.get("_score_ppl")) for s in samples]
    finite_ppls = [p for p in finite_ppls if isinstance(p, (int, float)) and math.isfinite(p)]
    ppl_summary = ""
    if finite_ppls:
        sorted_ppls = sorted(finite_ppls)
        ppl_summary = (
            f", ppl_stats=min={sorted_ppls[0]:.2f}, "
            f"p50={sorted_ppls[len(sorted_ppls) // 2]:.2f}, "
            f"p95={sorted_ppls[int(len(sorted_ppls) * 0.95)]:.2f}, max={sorted_ppls[-1]:.2f}"
        )
    max_desc = "disabled" if not use_max_ppl else str(max_ppl)
    logger.info(
        f"Filtered: {len(samples)} → {len(filtered)} "
        f"(score={score_key}, min_ppl={min_ppl}, max_ppl={max_desc}, removed: {removed}{ppl_summary})"
    )
    return filtered


def _optimal_mix(budget: int) -> tuple[float, float, float, float]:
    """(familiar, typical, unfamiliar, most-unfamiliar) fractions of the `optimal` mix.

    A heuristic: the budget thresholds and percentages are ours, not from a paper.
    (2605.12906 trains base math models on single-difficulty subsets and finds the
    best difficulty moves harder as the data budget grows; it tests no mixtures.)
    """
    if budget < 2_000:
        return (0.35, 0.50, 0.15, 0.00)  # small budget: easy-shifted, skip the extreme tail
    if budget < 10_000:
        return (0.25, 0.50, 0.20, 0.05)  # medium budget: transitional
    return (0.20, 0.50, 0.25, 0.05)  # large budget: full J-shape


STRATEGIES = ("optimal", "balanced", "medium_focus", "curriculum", "hard_focus", "flow", "random")


def select_by_budget(
    samples: list[dict],
    budget: int,
    strategy: str = "random",
    seed: int = 42,
) -> list[dict]:
    """Select a subset of `budget` samples.

    Strategies (buckets come from classify_familiarity: response-PPL percentiles
    of THIS dataset under THIS model):
      - "optimal": 20/50/25/5 familiar/typical/unfamiliar/most-unfamiliar mix, shifted familiar
          below 10K samples (_optimal_mix). A heuristic, not a measured optimum: on
          Qwen3.5-0.8B math SFT it scored below "random" (GSM8K -2.4, HumanEval+ -4.0).
      - "balanced": equal parts familiar/typical/unfamiliar
      - "medium_focus": the samples closest to the median PPL
      - "curriculum": a random subset, ordered familiar→unfamiliar (prepare_data sorts it)
      - "hard_focus": the highest-PPL samples
      - "flow": the lowest-PPL samples (the easiest; FLOW's exp(-loss/τ) weights
          are recorded as `_score_flow_weight` but training does not use them)
      - "random": a uniform random subset

    Within a bucket, samples are drawn at random (seeded), never in file order:
    datasets are often grouped by source, and a first-N pick would select sources.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown selection strategy {strategy!r}; choose one of {STRATEGIES}")
    if budget >= len(samples):
        return samples

    import random

    rng = random.Random(seed)

    def shuffled(lst):
        out = list(lst)
        rng.shuffle(out)
        return out

    def bucket(name):
        return shuffled(s for s in samples if s.get("_score_familiarity") == name)

    if strategy == "optimal":
        frac_easy, frac_medium, frac_hard, frac_very_hard = _optimal_mix(budget)
        easy, medium, hard = bucket("familiar"), bucket("typical"), bucket("unfamiliar")

        n_easy = int(budget * frac_easy)
        n_medium = int(budget * frac_medium)
        n_hard = int(budget * frac_hard)
        n_very_hard = budget - n_easy - n_medium - n_hard  # remainder
        logger.info(
            f"Optimal mix for budget={budget}: "
            f"{frac_easy:.0%} familiar / {frac_medium:.0%} typical / "
            f"{frac_hard:.0%} unfamiliar / {n_very_hard / budget:.0%} most-unfamiliar"
        )

        # Very hard: the highest-PPL samples of the hard bucket; hard: random among the rest
        hard_sorted = sorted(hard, key=lambda s: s.get("_score_response_ppl", 0), reverse=True)
        selected_very_hard = hard_sorted[:n_very_hard]
        selected_hard = shuffled(hard_sorted[n_very_hard:])[:n_hard]
        selected = easy[:n_easy] + medium[:n_medium] + selected_hard + selected_very_hard

        # Backfill: if a bucket had fewer samples than its quota, fill the
        # shortfall from unselected samples (medium first, then easy, then hard)
        # instead of silently shrinking the selection.
        if len(selected) < budget:
            chosen = {id(s) for s in selected}
            leftovers = [s for pool in (medium, easy, hard) for s in pool if id(s) not in chosen]
            shortfall = budget - len(selected)
            selected += leftovers[:shortfall]
            logger.info(f"Backfilled {min(shortfall, len(leftovers))} samples (short familiarity buckets)")
        _annotate_flow_weight(selected)

    elif strategy == "flow":
        _annotate_flow_weight(samples)
        selected = sorted(samples, key=lambda s: s.get("_score_flow_weight", 0), reverse=True)[:budget]

    elif strategy == "balanced":
        easy, medium, hard = bucket("familiar"), bucket("typical"), bucket("unfamiliar")
        per_bucket = budget // 3
        selected = easy[:per_bucket] + medium[:per_bucket] + hard[:per_bucket]
        chosen = {id(s) for s in selected}
        leftovers = [s for pool in (medium, easy, hard) for s in pool if id(s) not in chosen]
        selected += leftovers[: budget - len(selected)]

    elif strategy == "medium_focus":
        ppls = sorted(s.get("_score_response_ppl", 0) for s in samples)
        median_ppl = ppls[len(ppls) // 2]
        selected = sorted(samples, key=lambda s: abs(s.get("_score_response_ppl", 0) - median_ppl))[:budget]

    elif strategy == "curriculum":
        # A random subset (all buckets), ordered familiar→unfamiliar by prepare_data.
        # Keeping the `budget` easiest instead would silently drop the hard tail.
        selected = shuffled(samples)[:budget]

    elif strategy == "hard_focus":
        selected = sorted(samples, key=lambda s: s.get("_score_response_ppl", 0), reverse=True)[:budget]

    else:  # random
        selected = shuffled(samples)[:budget]

    logger.info(f"Selected {len(selected)} samples with strategy='{strategy}'")
    return selected


def _annotate_flow_weight(samples: list[dict]) -> None:
    """FLOW (arXiv:2502.02797) sample weight exp(-loss/τ), τ = median loss, from the
    mean response NLL under the scoring model. Metadata only: training does not read it."""
    losses = [s["_score_response_nll"] for s in samples if isinstance(s.get("_score_response_nll"), float)]
    if not losses:
        return
    tau = max(sorted(losses)[len(losses) // 2], 1e-6)
    for s in samples:
        loss = s.get("_score_response_nll")
        if isinstance(loss, float):
            s["_score_flow_weight"] = round(math.exp(-loss / tau), 4)


PREPARED_BASENAME = "scored_data"
EVAL_BASENAME = "eval_data"
PREPARED_META = "prepared_meta.json"


def find_prepared_dataset(output_dir: str | Path) -> Path | None:
    """Locate the prepared dataset file inside a preprocess output directory.

    Prefers parquet over jsonl. Returns None if nothing was prepared yet.
    """
    return _find_by_basename(output_dir, PREPARED_BASENAME)


def find_prepared_eval(output_dir: str | Path) -> Path | None:
    """Locate the held-out eval file written by prepare (eval_holdout > 0)."""
    return _find_by_basename(output_dir, EVAL_BASENAME)


def _find_by_basename(output_dir: str | Path, basename: str) -> Path | None:
    base = Path(output_dir)
    for suffix in (".parquet", ".jsonl"):
        candidate = base / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_eval_holdout(
    samples: list[dict], n_holdout: int, seed: int = 42, messages_field: str = "messages", key=None
) -> tuple[list[dict], list[dict]]:
    """Reserve about n_holdout random samples as a held-out eval set.

    Returns (train_pool, eval_holdout): disjoint, deterministic for a given seed, and
    split by prompt (data_clean.split_holdout): rows that share a system prompt and first
    user turn stay on one side, so the eval set holds no task the model trained on under
    another answer. Random (not familiarity-stratified) so the eval set is an unbiased
    draw from the same distribution the training pool comes from.
    """
    from palingenesis.data_clean import split_holdout

    return split_holdout(samples, n_holdout, seed, messages_field, key)


def save_prepared(
    samples: list[dict],
    output_path: str | Path,
    output_format: str = "parquet",
    basename: str = PREPARED_BASENAME,
) -> Path:
    """Save prepared samples as parquet (preferred) or jsonl.

    Parquet preserves the sample ORDER (important for curriculum strategy),
    loads much faster than jsonl, and is directly consumable by
    datasets.load_dataset at train time.

    Falls back to jsonl if the samples have a schema that Arrow cannot unify
    (e.g. wildly heterogeneous nested message structures).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        try:
            from datasets import Dataset

            out_file = output_path / f"{basename}.parquet"
            Dataset.from_list(samples).to_parquet(str(out_file))
            # Remove a stale jsonl from a previous run so training can't pick it up
            stale = output_path / f"{basename}.jsonl"
            if stale.exists():
                stale.unlink()
            return out_file
        except Exception as e:
            logger.warning(f"Parquet write failed ({e}); falling back to jsonl")

    out_file = output_path / f"{basename}.jsonl"
    with out_file.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # A parquet from a previous run (or the partial file of the failed write above) would win:
    # find_prepared_dataset prefers parquet
    stale = output_path / f"{basename}.parquet"
    if stale.exists():
        stale.unlink()
    return out_file


def load_samples(data_path: str, dataset_split: str = "train", max_samples: int | None = None) -> list[dict]:
    """Rows of a local .jsonl / .json / .parquet file or a Hub dataset, as dicts; at most
    max_samples of them (a seeded random subset)."""
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with path.open() as f:
            samples = [json.loads(line) for line in f if line.strip()]
    elif path.suffix == ".json":
        with path.open() as f:
            samples = json.load(f)
    elif path.suffix == ".parquet":
        from datasets import load_dataset

        samples = [dict(row) for row in load_dataset("parquet", data_files=str(path), split="train")]
    elif max_samples:
        # HuggingFace dataset, capped: stream a shuffled prefix instead of
        # downloading the whole dataset to keep max_samples rows of it.
        import itertools

        from datasets import load_dataset

        ds = load_dataset(str(data_path), split=dataset_split, streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=max(10_000, 2 * max_samples))
        samples = [dict(row) for row in itertools.islice(ds, max_samples)]
    else:
        from datasets import load_dataset

        samples = [dict(row) for row in load_dataset(str(data_path), split=dataset_split)]

    if max_samples and len(samples) > max_samples:
        samples = random.Random(42).sample(samples, max_samples)
    return samples


# What the report calls the rows filter_samples removes.
_FILTER_REASONS = {"too_easy": "response_ppl_below_min", "too_hard": "response_ppl_above_max"}


def prepare_data(
    model_name: str,
    data_path: str,
    output_path: str,
    messages_field: str = "messages",
    max_samples: int | None = None,
    max_seq_length: int = 8192,
    budget: int | None = None,
    strategy: str = "random",
    batch_size: int = 4,
    dataset_split: str = "train",
    output_format: str = "parquet",
    compute_hes: bool = False,
    hes_top_k_pct: float = 0.5,
    eval_holdout: int = 0,
    min_ppl: float = 0.0,
    max_ppl: float = 0.0,
    filter_score: str = "response",
    max_batch_tokens: int = 16384,
    chat_options: dict | None = None,
    group_field: str = "",
    score: bool = False,
    dedupe: bool = True,
    max_per_prompt: int = 0,
    max_per_group: int = 0,
    max_tokens_per_group: int = 0,
    token_budget: int = 0,
    seed: int = 42,
    scorer: Scorer | None = None,
) -> Path:
    """Full data preparation pipeline.

    1. Load data
    2. Drop exact duplicates (dedupe) and the rows training would drop; count the rows
       it would cut to max_seq_length
    3. Optionally score with the model (score, compute_hes, a perplexity bound or a
       familiarity strategy turn it on), bucket by familiarity and filter by perplexity
    4. Reserve a held-out eval set (eval_holdout > 0) that shares no prompt with the
       training pool
    5. Cap the rows that share a prompt (max_per_prompt) and the rows / tokens per
       group (max_per_group, max_tokens_per_group; needs group_field)
    6. Select by budget/strategy, then by token_budget
    7. Save the dataset (parquet or jsonl), prepared_meta.json and dropped.jsonl

    group_field: a per-row column (e.g. "source", "category") to cap and to profile: the
    manifest reports, per group, the sample count, tokens, mean response tokens and, when
    scored, median response NLL and familiarity buckets, for the cleaned pool and for the
    selection. Public datasets mixing sources differ a lot per source, and
    familiarity-based selection mostly moves the source mix.

    chat_options: ChatDataset options (train_on_reasoning, include_observations,
    last_turn_only, tools_field, think_tags, chat_template_kwargs) so prepare renders and
    masks exactly what training trains.

    scorer: a Scorer to reuse (prepare_multi_source shares one model across sources).

    Returns the path to the prepared dataset file.
    """
    from palingenesis.data_clean import (
        ROW_INDEX,
        CleanReport,
        cap_groups,
        drop_exact_duplicates,
        prompt_key,
        take_token_budget,
        token_stats,
        tokenize_rows,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Settings that cannot work are refused before any row is read or scored.
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown selection strategy {strategy!r}; choose one of {STRATEGIES}")
    if token_budget and strategy != "random":
        raise ValueError(f"token_budget selects a random subset: it cannot be combined with strategy={strategy!r}")
    if (max_per_group or max_tokens_per_group) and not group_field:
        raise ValueError("max_per_group / max_tokens_per_group need group_field (the row column that names the group)")
    scoring = score or compute_hes or min_ppl > 0 or max_ppl > 0 or strategy != "random"
    if strategy != "random":
        logger.warning(
            f"strategy={strategy!r} is experimental: selection and ordering by response perplexity have not beaten "
            "a random subset of the same size (see the data guide). Compare with strategy=random on your eval battery."
        )

    logger.info(f"Loading data from: {data_path} (split={dataset_split})")
    samples = load_samples(str(data_path), dataset_split, max_samples)
    for i, s in enumerate(samples):
        s[ROW_INDEX] = i
    report = CleanReport(loaded=len(samples))
    logger.info(f"Loaded {len(samples)} samples")

    scorer = scorer or Scorer(model_name)
    chat = scorer.chat(max_seq_length, messages_field, chat_options)
    if dedupe:
        samples = drop_exact_duplicates(samples, report, messages_field, chat.tools_field)
    samples, entries = tokenize_rows(samples, chat, report, keep_tokens=scoring)

    if scoring:
        logger.info("Scoring samples with model perplexity...")
        score_entries(scorer, entries, samples, batch_size, max_batch_tokens, hes_top_k_pct if compute_hes else None)
        samples = classify_familiarity(samples)
    del entries
    pool_groups = profile_groups(samples, group_field) if group_field else None

    # Rows are cut to max_seq_length already, so the length ceiling never removes any.
    samples = filter_samples(
        samples,
        min_ppl=min_ppl,
        max_ppl=max_ppl,
        max_length=max_seq_length,
        score=filter_score,
        on_drop=lambda s, reason: report.drop(s, _FILTER_REASONS.get(reason, reason)),
    )
    if not samples:
        hint = (
            "For domain-shifted or multilingual chat data, try `preprocess.max_ppl: 0` "
            "to disable the absolute PPL ceiling."
            if max_ppl > 0
            else "Check that messages_field matches the dataset schema and that the rows hold a trained assistant turn."
        )
        raise ValueError(
            f"Preparation produced 0 samples after cleaning and filtering (removed: {dict(report.reasons)}). "
            f"filter_score={filter_score}, min_ppl={min_ppl}, max_ppl={max_ppl}. {hint}"
        )

    prompt_keys = (
        {s[ROW_INDEX]: prompt_key(s, messages_field) for s in samples} if eval_holdout or max_per_prompt else {}
    )

    def by_prompt(s: dict) -> str:
        return prompt_keys[s[ROW_INDEX]]

    # Reserve held-out eval BEFORE caps and selection: a random draw of whole prompts
    # from the cleaned pool, disjoint from anything that gets trained on.
    eval_samples: list[dict] = []
    if eval_holdout:
        samples, eval_samples = split_eval_holdout(samples, eval_holdout, seed, messages_field, key=by_prompt)
        if len(eval_samples) < eval_holdout:
            logger.warning(
                f"Held-out eval set has {len(eval_samples)} of the {eval_holdout} rows asked for "
                f"(pool of {len(samples) + len(eval_samples)} rows; rows that share a prompt are kept together)"
            )
        if eval_samples:
            logger.info(f"Reserved {len(eval_samples)} samples as held-out eval set")

    if max_per_prompt:
        samples = cap_groups(samples, by_prompt, report, "prompt_cap", max_rows=max_per_prompt, seed=seed)
    if max_per_group or max_tokens_per_group:
        samples = cap_groups(
            samples,
            lambda s: str(s.get(group_field, "<none>")),
            report,
            "group_cap",
            max_rows=max_per_group,
            max_tokens=max_tokens_per_group,
            seed=seed,
        )

    # Select by budget
    pool_rows = len(samples)
    if budget:
        samples = select_by_budget(samples, budget, strategy, seed)
    if token_budget:
        samples = take_token_budget(samples, token_budget, seed)

    # Add curriculum rank if not already present
    if strategy == "curriculum":
        samples = sorted(samples, key=lambda s: s.get("_score_response_ppl", 0))
        for i, s in enumerate(samples):
            s["_score_rank"] = i

    # Save data
    out_file = save_prepared(samples, output_path, output_format)
    eval_file: Path | None = None
    if eval_samples:
        eval_file = save_prepared(eval_samples, output_path, output_format, basename=EVAL_BASENAME)
        logger.info(f"Eval holdout: {eval_file} ({len(eval_samples)} samples)")
    dropped_file = report.write(output_path)

    # Stats + manifest (consumed by training to verify provenance)
    ppls = [s["_score_response_ppl"] for s in samples if "_score_response_ppl" in s]
    buckets: dict[str, int] = {}
    if scoring:
        for s in samples:
            b = s.get("_score_familiarity", "unknown")
            buckets[b] = buckets.get(b, 0) + 1

    meta = {
        "model": model_name,
        "source_dataset": str(data_path),
        "dataset_split": dataset_split,
        "messages_field": messages_field,
        "max_seq_length": max_seq_length,
        "strategy": strategy,
        "budget": budget,
        "token_budget": token_budget,
        "seed": seed,
        "num_samples": len(samples),
        "format": out_file.suffix.lstrip("."),
        "output_file": str(out_file),
        "scored": scoring,
        "cleaning": report.summary(),
        "dropped_file": str(dropped_file),
        "selection": {"pool": pool_rows, "selected": len(samples)},
        "tokens": token_stats(samples),
        "familiarity_distribution": buckets,
        "eval_holdout": len(eval_samples),
        "eval_file": str(eval_file) if eval_file else None,
        "filter_score": filter_score,
        "min_ppl": min_ppl,
        "max_ppl": max_ppl,
        "dedupe": dedupe,
        "max_per_prompt": max_per_prompt,
        "max_per_group": max_per_group,
        "max_tokens_per_group": max_tokens_per_group,
    }
    tokens = [s["_score_response_token_count"] for s in samples if "_score_response_token_count" in s]
    if tokens:
        meta["response_tokens"] = {"total": sum(tokens), "mean": round(sum(tokens) / len(tokens), 1)}
    if group_field:
        meta["group_field"] = group_field
        meta["groups_scored_pool"] = pool_groups
        meta["groups_selected"] = profile_groups(samples, group_field)
    if ppls:
        meta["ppl_stats"] = {
            "min": round(min(ppls), 2),
            "median": round(sorted(ppls)[len(ppls) // 2], 2),
            "max": round(max(ppls), 2),
            "mean": round(sum(ppls) / len(ppls), 2),
        }
    with (Path(output_path) / PREPARED_META).open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Cleaning: {report.loaded} loaded, removed {dict(sorted(report.reasons.items())) or 'none'}, "
        f"{report.truncated} cut to max_seq_length={max_seq_length} (every removed row: {dropped_file})"
    )
    logger.info(f"Output: {out_file} ({len(samples)} samples, {meta['tokens'].get('total', 0):,} tokens)")
    if ppls:
        logger.info(
            f"Perplexity stats: min={min(ppls):.1f}, median={sorted(ppls)[len(ppls) // 2]:.1f}, "
            f"max={max(ppls):.1f}, mean={sum(ppls) / len(ppls):.1f}"
        )
        logger.info(f"Familiarity distribution: {buckets}")

    return out_file


def compute_hes_scores(
    model_name: str,
    samples: list[dict],
    messages_field: str = "messages",
    max_seq_length: int = 8192,
    top_k_pct: float = 0.5,
    batch_size: int = 4,
    device: str = "auto",
) -> list[dict]:
    """High-Entropy Sum scores (`_score_hes`, arXiv:2605.22389) next to the perplexity scores.

    Kept for callers of the former two-pass API: the entropies come from the scoring
    forward (see _score_padded_batch), so this is score_samples_with_model with
    hes_top_k_pct set.
    """
    return score_samples_with_model(
        model_name, samples, messages_field, max_seq_length, batch_size, device, hes_top_k_pct=top_k_pct
    )


def prepare_multi_source(
    model_name: str,
    sources: list[dict],
    output_path: str,
    max_seq_length: int = 8192,
    budget_per_source: int | None = None,
    strategy: str = "random",
    compute_hes: bool = False,
    hes_top_k_pct: float = 0.5,
    score: bool = False,
    min_ppl: float = 0.0,
    max_ppl: float = 0.0,
    batch_size: int = 4,
    max_batch_tokens: int = 16384,
):
    """Prepare several data sources, each on its own (prepare_data per source, one model).

    Creates:
      - {output_path}/{source_name}/scored_data.jsonl (+ prepared_meta.json, dropped.jsonl)
      - {output_path}/manifest.json: per source, rows, tokens, what cleaning removed, the
        weight given and the share of rows and of tokens it amounts to. A `data.sources`
        weight is a per-conversation sampling probability, so a source of long
        conversations takes a larger share of the tokens than of the rows.

    Duplicates are removed inside a source, not across sources.

    Args:
        model_name: Model whose tokenizer (and, when scoring, weights) is used
        sources: List of source configs, each with:
            - dataset: path or HF dataset name
            - split: dataset split (default "train")
            - name: human-readable name for this source
            - weight: relative importance (default 1.0)
            - budget: per-source budget override (optional)
            - messages_field: field containing messages (default "messages")
            - tools_field, last_turn_only: as in data.sources
        output_path: Base output directory
        max_seq_length: Max sequence length
        budget_per_source: Default budget per source (None = use all)
        strategy: Selection strategy per source
        compute_hes, hes_top_k_pct, score, min_ppl, max_ppl, batch_size, max_batch_tokens:
            as in prepare_data
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown selection strategy {strategy!r}; choose one of {STRATEGIES}")
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_name,
        "strategy": strategy,
        "sources": [],
    }
    scorer = Scorer(model_name)

    for src_config in sources:
        name = src_config.get("name", src_config.get("dataset", "unknown"))
        dataset_id = src_config.get("dataset")
        weight = src_config.get("weight", 1.0)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing source: {name} (dataset={dataset_id}, weight={weight})")
        if src_config.get("mode", "sft") != "sft":
            logger.warning(
                f"  Source '{name}' is mode={src_config['mode']}: only chat (sft) sources are prepared, skipped"
            )
            continue

        src_output = output_base / name.replace("/", "_").replace(" ", "_")
        out_file = prepare_data(
            model_name,
            dataset_id,
            str(src_output),
            messages_field=src_config.get("messages_field", "messages"),
            max_seq_length=max_seq_length,
            budget=src_config.get("budget", budget_per_source),
            strategy=strategy,
            batch_size=batch_size,
            dataset_split=src_config.get("split", "train"),
            output_format="jsonl",
            compute_hes=compute_hes,
            hes_top_k_pct=hes_top_k_pct,
            min_ppl=min_ppl,
            max_ppl=max_ppl,
            max_batch_tokens=max_batch_tokens,
            chat_options={k: src_config[k] for k in ("tools_field", "last_turn_only") if k in src_config},
            score=score,
            scorer=scorer,
        )
        meta = json.loads((src_output / PREPARED_META).read_text())
        src_meta = {
            "name": name,
            "dataset": dataset_id,
            "output_path": str(out_file),
            "num_samples": meta["num_samples"],
            "weight": weight,
            "tokens": meta["tokens"],
            "cleaning": meta["cleaning"],
        }
        if meta["scored"]:
            src_meta["median_ppl"] = meta.get("ppl_stats", {}).get("median", 0)
            src_meta["familiarity_distribution"] = meta["familiarity_distribution"]
        manifest["sources"].append(src_meta)

    # What the weights amount to: rows drawn per source, and the tokens those rows carry.
    total_weight = sum(s["weight"] for s in manifest["sources"]) or 1.0
    mean_tokens = [s["tokens"].get("total", 0) / max(s["num_samples"], 1) for s in manifest["sources"]]
    token_mass = sum(s["weight"] * t for s, t in zip(manifest["sources"], mean_tokens)) or 1.0
    for src, tokens_per_row in zip(manifest["sources"], mean_tokens):
        src["normalized_weight"] = round(src["weight"] / total_weight, 4)
        src["token_share"] = round(src["weight"] * tokens_per_row / token_mass, 4)
        logger.info(
            f"  {src['name']}: {src['num_samples']} rows, {src['normalized_weight']:.1%} of examples, "
            f"{src['token_share']:.1%} of tokens at weight {src['weight']}"
        )

    manifest_file = output_base / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nManifest saved to {manifest_file}")
    logger.info(f"Total sources: {len(manifest['sources'])}")


def prepare_from_config(config) -> Path:
    """Run preparation driven by the SAME YAML config used for training.

    Reads: model.name_or_path, data.dataset, data.dataset_split, data.messages_field,
    data.max_seq_length, the data options that decide how a row is rendered and masked,
    and the preprocess: section. Training with preprocess.enabled=true then picks the
    output up automatically from preprocess.output_dir.
    """
    pre = config.preprocess
    return prepare_data(
        model_name=config.model.name_or_path,
        data_path=config.data.dataset,
        output_path=pre.output_dir,
        messages_field=config.data.messages_field,
        max_samples=pre.max_samples or None,
        max_seq_length=config.data.max_seq_length,
        budget=pre.budget or None,
        strategy=pre.strategy,
        batch_size=pre.batch_size,
        dataset_split=config.data.dataset_split,
        output_format=pre.format,
        compute_hes=pre.hes,
        hes_top_k_pct=pre.hes_top_k_pct,
        eval_holdout=pre.eval_holdout,
        min_ppl=pre.min_ppl,
        max_ppl=pre.max_ppl,
        filter_score=pre.filter_score,
        max_batch_tokens=pre.max_batch_tokens,
        group_field=pre.group_field,
        chat_options={
            "train_on_reasoning": config.data.train_on_reasoning,
            "include_observations": config.data.include_observations,
            "last_turn_only": config.data.last_turn_only,
            "tools_field": config.data.tools_field,
            "think_tags": config.data.think_tags,
            "chat_template_kwargs": config.data.chat_template_kwargs,
        },
        score=pre.score,
        dedupe=pre.dedupe,
        max_per_prompt=pre.max_per_prompt,
        max_per_group=pre.max_per_group,
        max_tokens_per_group=pre.max_tokens_per_group,
        token_budget=pre.token_budget,
        seed=config.data.seed,
    )


def main():
    """CLI entry point for data preparation.

    Two modes:
      1. Config-driven (recommended): pgs prepare --config train_config.yaml
         Supports the same --section.field overrides as training, e.g.
         pgs prepare --config cfg.yaml --preprocess.budget 5000 --preprocess.eval_holdout 200
      2. Standalone flags: pgs prepare --model M --data D --output prepared/
    """
    if "--config" in sys.argv:
        from palingenesis.config import Config

        config = Config.from_cli()
        prepare_from_config(config)
        return

    import argparse

    parser = argparse.ArgumentParser(
        description="Clean SFT data (broken rows, duplicates), hold out an eval set and select a budget; "
        "optionally score it with the model (response perplexity)"
    )
    parser.add_argument("--model", required=True, help="Model name/path (its tokenizer; its weights only for scoring)")
    parser.add_argument("--data", required=True, help="Data path (JSONL, JSON, parquet, or HF dataset)")
    parser.add_argument("--output", default="./prepared", help="Output directory")
    parser.add_argument("--messages_field", default="messages", help="Field containing messages")
    parser.add_argument("--max_samples", type=int, help="Max samples to process")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--budget", type=int, help="Target number of samples to select")
    parser.add_argument("--token_budget", type=int, default=0, help="Target number of tokens to select (random subset)")
    parser.add_argument(
        "--strategy",
        default="random",
        choices=list(STRATEGIES),
        help="How --budget rows are chosen; every strategy but random is experimental and turns scoring on",
    )
    parser.add_argument("--no_dedupe", action="store_true", help="Keep exact duplicate conversations")
    parser.add_argument("--max_per_prompt", type=int, default=0, help="Keep at most N rows that share a prompt")
    parser.add_argument(
        "--group_field", default="", help="Row column (e.g. source) to cap and to profile in prepared_meta.json"
    )
    parser.add_argument("--max_per_group", type=int, default=0, help="Keep at most N rows per --group_field value")
    parser.add_argument(
        "--max_tokens_per_group", type=int, default=0, help="Keep at most N tokens per --group_field value"
    )
    parser.add_argument(
        "--eval_holdout",
        type=int,
        default=0,
        help="Reserve N samples as a held-out eval set (eval_data.parquet, shares no prompt with the training data)",
    )
    parser.add_argument("--split", default="train", help="Dataset split (for HF datasets)")
    parser.add_argument("--format", default="parquet", choices=["parquet", "jsonl"], help="Output format")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", action="store_true", help="Score every row with the model (response perplexity)")
    parser.add_argument("--batch_size", type=int, default=4, help="Max samples per scoring forward")
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=16384,
        help="Padded-token cap per scoring forward (bounds logits memory; 32768+ OK on 80GB GPUs)",
    )
    parser.add_argument("--hes", action="store_true", help="Also compute HES scores (experimental; turns scoring on)")
    parser.add_argument("--hes_top_k", type=float, default=0.5, help="Top-k%% tokens for HES metric")
    parser.add_argument(
        "--min_ppl", type=float, default=0.0, help="Drop rows below this response PPL (0 = off; turns scoring on)"
    )
    parser.add_argument(
        "--max_ppl", type=float, default=0.0, help="Drop rows above this response PPL (0 = off; turns scoring on)"
    )
    parser.add_argument(
        "--filter_score",
        default="response",
        choices=["response", "full"],
        help="PPL score used by the outlier filter (response=assistant tokens only, full=all tokens)",
    )
    args = parser.parse_args()

    prepare_data(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output,
        messages_field=args.messages_field,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        budget=args.budget,
        strategy=args.strategy,
        batch_size=args.batch_size,
        dataset_split=args.split,
        output_format=args.format,
        compute_hes=args.hes,
        hes_top_k_pct=args.hes_top_k,
        eval_holdout=args.eval_holdout,
        min_ppl=args.min_ppl,
        max_ppl=args.max_ppl,
        filter_score=args.filter_score,
        max_batch_tokens=args.max_batch_tokens,
        group_field=args.group_field,
        score=args.score,
        dedupe=not args.no_dedupe,
        max_per_prompt=args.max_per_prompt,
        max_per_group=args.max_per_group,
        max_tokens_per_group=args.max_tokens_per_group,
        token_budget=args.token_budget,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
