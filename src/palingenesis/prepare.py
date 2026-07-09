"""Data preparation pipeline — offline scoring, filtering, and curriculum ordering.

Scores training samples using the target model's perplexity, then filters and
orders them for optimal SFT. Runs entirely offline (inference only, no training).

Key research findings synthesized:

1. DATA DIFFICULTY IS NOT UNIVERSAL (arxiv:2605.12906, Tsinghua 2026):
   - Small datasets: easier data is optimal (model can't generalize from hard)
   - Large datasets: harder data becomes valuable (easy becomes redundant)
   - There is ALWAYS an optimal difficulty that depends on data budget

2. EASY SAMPLES PREVENT FORGETTING (arxiv:2502.02797, ICML 2025):
   - Upweighting easy samples (low pretrained loss) preserves base capabilities
   - Weight: w = exp(-loss / τ), τ = median(loss). Parameter-free.
   - This is complementary to DEFT (which handles token-level difficulty)

3. IFD = INSTRUCTION INFORMATIVENESS (arxiv:2308.12032):
   - IFD = ppl(response|instruction) / ppl(response|no_instruction)
   - High IFD: instruction is critical for generating the response (USEFUL)
   - Low IFD: model generates same response regardless of instruction (WASTE)
   - With 10% of data selected by IFD, matches full-data performance

4. THE J-SHAPED DISTRIBUTION (synthesis of all findings):
   - Optimal data mix is NOT uniform across difficulty
   - Shape: 20% easy + 50% medium + 25% hard + 5% very hard
   - Easy data: maintains capabilities, provides stable gradients
   - Medium data: maximum information content (InfoSFT sweet spot)
   - Hard data: pushes the capability frontier
   - Very hard: only a few, for exposure without overwhelming

5. DIVERSITY > QUANTITY (arxiv:2603.11076, DIVE 2026):
   - 48K diverse samples >> 200K homogeneous, even with 4× less data
   - Tool/topic/length diversity all matter

6. DISTRIBUTIONAL ALIGNMENT MATTERS (NeurIPS 2025):
   - Data IN the model's distribution but slightly beyond capability = optimal
   - Data FAR from model's distribution = harmful regardless of quality

Usage:
    # Standalone flags
    palingenesis prepare --model google/gemma-4-12B-it \\
                        --data your-org/agentic-traces \\
                        --output prepared/ \\
                        --budget 10000 \\
                        --strategy optimal

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


def score_samples_with_model(
    model_name: str,
    samples: list[dict],
    messages_field: str = "messages",
    max_seq_length: int = 8192,
    batch_size: int = 4,
    device: str = "auto",
    max_batch_tokens: int = 16384,
) -> list[dict]:
    """Score samples by computing model perplexity on responses.

    Uses the target model (same one you'll fine-tune) to compute
    per-sample difficulty. This tells you how hard each sample is
    FOR THIS SPECIFIC MODEL — which is what matters for SFT.

    High perplexity = hard (model can't predict the response)
    Low perplexity = easy (model already knows this)

    For SFT, you want samples in the "medium" range — hard enough
    to be informative, easy enough that the model can learn from them.

    Scoring is truly batched: samples are pre-tokenized, sorted by length,
    packed into padded batches (≤ batch_size samples AND ≤ max_batch_tokens
    padded tokens per forward — logits are B×S×V, so the token cap is what
    protects memory), then scored one forward per batch.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ── Phase 1: tokenize everything on CPU (chat template + response masks) ──
    total = len(samples)
    entries: list[tuple[int, list[int], torch.Tensor]] = []  # (sample_idx, input_ids, response_mask)
    for idx, sample in enumerate(samples):
        messages = _get_messages(sample, messages_field)
        if not messages:
            _mark_unscoreable(sample)
            continue

        try:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            full_text = "\n".join(m.get("content", "") for m in messages)

        ids = tokenizer(full_text, max_length=max_seq_length, truncation=True)["input_ids"]
        if len(ids) < 2:
            _mark_unscoreable(sample)
            continue

        # True for assistant tokens, False for context — same tokens that get loss in training
        response_mask = _build_response_mask(tokenizer, messages, len(ids))
        entries.append((idx, ids, response_mask))

        if idx % 5000 == 0:
            logger.info(f"  Tokenized {idx}/{total}")

    # ── Phase 2: length-sorted batching (minimizes padding waste) ──
    # Batches are capped by BOTH batch_size (samples) and max_batch_tokens
    # (padded tokens per forward), because logits memory is B×S×V.
    entries.sort(key=lambda e: len(e[1]))
    batches = _group_entries_by_length(entries, batch_size, max_batch_tokens)
    logger.info(
        f"Scoring {len(entries)} samples in {len(batches)} batched forwards "
        f"(batch_size≤{batch_size}, ≤{max_batch_tokens} padded tokens per forward)"
    )

    # ── Phase 3: one forward per batch, per-sample masked NLL ──
    pad_id = tokenizer.pad_token_id
    scored_count = 0
    scored_tokens = 0
    total_tokens = sum(len(e[1]) for e in entries)
    t0 = time.perf_counter()
    for batch_num, batch in enumerate(batches):
        _score_padded_batch(model, batch, samples, pad_id)
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

    return samples


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
) -> None:
    """Forward one padded batch and write per-sample PPL scores in-place.

    Response perplexity is computed only on assistant tokens — the same tokens
    that receive loss during training.
    """
    max_len = max(len(ids) for _, ids, _ in batch)
    n = len(batch)

    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((n, max_len), dtype=torch.long)
    resp_mask = torch.zeros((n, max_len), dtype=torch.bool)
    for row, (_, ids, rmask) in enumerate(batch):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
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
    rows_per_chunk = max(1, 8192 // max(max_len - 1, 1))
    for start in range(0, n, rows_per_chunk):
        end = min(start + rows_per_chunk, n)
        chunk = logits[start:end, :-1, :].float()
        nll[start:end] = torch.nn.functional.cross_entropy(
            chunk.reshape(-1, chunk.size(-1)),
            shift_labels[start:end].reshape(-1),
            reduction="none",
        ).view(end - start, max_len - 1)
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


def _build_response_mask(
    tokenizer,
    messages: list[dict],
    total_seq_len: int,
) -> torch.Tensor:
    """Build a boolean mask identifying assistant response tokens.

    Uses the chat template to precisely locate where each assistant turn starts
    and ends within the tokenized sequence. This matches what ChatDataset does
    during training (only assistant tokens get loss).

    Strategy:
    1. For each assistant message, tokenize the conversation UP TO (but not including)
       that message to find the prefix length.
    2. Tokenize the conversation UP TO AND INCLUDING that message for the end position.
    3. Mark tokens in [prefix_len, end_len) as response tokens.

    Falls back to the "everything after first user message" heuristic if
    chat template tokenization fails.
    """
    mask = torch.zeros(total_seq_len, dtype=torch.bool)

    try:
        # Find assistant message indices
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_indices:
            # No assistant messages — everything is context
            return mask

        for asst_idx in assistant_indices:
            # Tokenize up to (not including) this assistant message
            prefix_messages = messages[:asst_idx]
            if prefix_messages:
                prefix_text = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
                prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                start_pos = len(prefix_tokens)
            else:
                start_pos = 0

            # Tokenize up to and including this assistant message
            inclusive_messages = messages[: asst_idx + 1]
            inclusive_text = tokenizer.apply_chat_template(
                inclusive_messages, tokenize=False, add_generation_prompt=False
            )
            inclusive_tokens = tokenizer(inclusive_text, add_special_tokens=False)["input_ids"]
            end_pos = len(inclusive_tokens)

            # Clamp to sequence length
            start_pos = min(start_pos, total_seq_len)
            end_pos = min(end_pos, total_seq_len)

            if end_pos > start_pos:
                mask[start_pos:end_pos] = True

    except Exception:
        # Fallback: mark last 60% as response (crude but better than nothing)
        response_start = max(1, int(total_seq_len * 0.4))
        mask[response_start:] = True

    return mask


def classify_difficulty(
    samples: list[dict],
    easy_percentile: float = 25.0,
    hard_percentile: float = 75.0,
) -> list[dict]:
    """Classify samples into easy/medium/hard based on perplexity distribution.

    Uses percentile-based thresholds (adapts to any model/dataset combination).
    """
    ppls = [s["_score_response_ppl"] for s in samples if s.get("_score_response_ppl", float("inf")) < float("inf")]
    if not ppls:
        return samples

    ppls_sorted = sorted(ppls)
    n = len(ppls_sorted)
    easy_thresh = ppls_sorted[int(n * easy_percentile / 100)]
    hard_thresh = ppls_sorted[int(n * hard_percentile / 100)]

    for s in samples:
        ppl = s.get("_score_response_ppl", float("inf"))
        if ppl <= easy_thresh:
            s["_score_difficulty_bucket"] = "easy"
        elif ppl >= hard_thresh:
            s["_score_difficulty_bucket"] = "hard"
        else:
            s["_score_difficulty_bucket"] = "medium"

    return samples


def filter_samples(
    samples: list[dict],
    min_ppl: float = 1.5,
    max_ppl: float = 500.0,
    min_length: int = 10,
    max_length: int = 32768,
    score: str = "response",
) -> list[dict]:
    """Remove clearly bad samples.

    - min_ppl < 1.5: essentially memorized / trivial (e.g., "Hi" → "Hello!")
    - max_ppl > 500: model has no idea (corrupted, wrong language, noise)
      Set max_ppl <= 0 to disable the upper-bound filter. This is useful for
      multilingual/domain-shifted data where absolute PPL is high but relative
      difficulty ranking is still useful.
    - min_length: too short to be useful
    - max_length: too long (would dominate training)

    For chat SFT, filter on assistant-response perplexity by default. Training
    loss is applied to assistant tokens only, so filtering on full-sequence PPL
    can wrongly reject domain/language-shifted prompts even when responses are
    learnable and masking is healthy.
    """
    filtered = []
    removed = {"too_easy": 0, "too_hard": 0, "too_short": 0, "too_long": 0}
    score_key = "_score_response_ppl" if score == "response" else "_score_ppl"
    use_max_ppl = max_ppl > 0

    for s in samples:
        ppl = s.get(score_key, s.get("_score_ppl", float("inf")))
        length = s.get("_score_length", 0)

        if ppl < min_ppl:
            removed["too_easy"] += 1
        elif use_max_ppl and ppl > max_ppl:
            removed["too_hard"] += 1
        elif length < min_length:
            removed["too_short"] += 1
        elif length > max_length:
            removed["too_long"] += 1
        else:
            filtered.append(s)

    finite_ppls = [s.get(score_key, s.get("_score_ppl")) for s in samples]
    finite_ppls = [p for p in finite_ppls if isinstance(p, (int, float)) and math.isfinite(p)]
    ppl_summary = ""
    if finite_ppls:
        sorted_ppls = sorted(finite_ppls)
        ppl_summary = (
            f", ppl_stats=min={sorted_ppls[0]:.2f}, "
            f"p50={sorted_ppls[len(sorted_ppls)//2]:.2f}, "
            f"p95={sorted_ppls[int(len(sorted_ppls)*0.95)]:.2f}, max={sorted_ppls[-1]:.2f}"
        )
    max_desc = "disabled" if not use_max_ppl else str(max_ppl)
    logger.info(
        f"Filtered: {len(samples)} → {len(filtered)} "
        f"(score={score_key}, min_ppl={min_ppl}, max_ppl={max_desc}, removed: {removed}{ppl_summary})"
    )
    return filtered


def _optimal_mix(budget: int) -> tuple[float, float, float, float]:
    """Budget-adaptive J-shape: (easy, medium, hard, very_hard) fractions.

    2605.12906's key finding is that the optimal difficulty is NOT fixed — it
    shifts harder as the data budget grows. With few samples the model needs
    learnable (easier) data to make any progress; with many samples it can
    afford to spend budget on the hard tail. Since our buckets are percentiles
    of the AVAILABLE PPL range (classify_difficulty), this maps the paper's
    absolute finding onto whatever range this dataset/model pair actually has.
    """
    if budget < 2_000:
        return (0.35, 0.50, 0.15, 0.00)  # small budget: easy-shifted, skip the extreme tail
    if budget < 10_000:
        return (0.25, 0.50, 0.20, 0.05)  # medium budget: transitional
    return (0.20, 0.50, 0.25, 0.05)  # large budget: full J-shape


def select_by_budget(
    samples: list[dict],
    budget: int,
    strategy: str = "optimal",
) -> list[dict]:
    """Select optimal subset within a token/sample budget.

    Strategies:
      - "optimal": J-shaped difficulty distribution, ADAPTIVE to the budget
          (see _optimal_mix). Difficulty buckets are percentile-based, so the
          mix always maps onto the AVAILABLE difficulty range of this dataset
          relative to this model — not absolute PPL thresholds.
          Plus FLOW weighting within each bucket for anti-forgetting.
      - "balanced": equal parts easy/medium/hard (diversity)
      - "medium_focus": prioritize medium-difficulty (InfoSFT-style)
      - "curriculum": order easy→hard for sequential training
      - "hard_focus": prioritize hard (for large datasets / strong models)
      - "flow": FLOW anti-forgetting weighting (easy-heavy)
    """
    if budget >= len(samples):
        return samples

    if strategy == "optimal":
        # J-shaped distribution over PERCENTILE buckets (classify_difficulty),
        # with the mix adapted to the data budget (2605.12906: the optimal
        # difficulty shifts harder as the budget grows).
        frac_easy, frac_medium, frac_hard, frac_very_hard = _optimal_mix(budget)
        easy = [s for s in samples if s.get("_score_difficulty_bucket") == "easy"]
        medium = [s for s in samples if s.get("_score_difficulty_bucket") == "medium"]
        hard = [s for s in samples if s.get("_score_difficulty_bucket") == "hard"]

        n_easy = int(budget * frac_easy)
        n_medium = int(budget * frac_medium)
        n_hard = int(budget * frac_hard)
        n_very_hard = budget - n_easy - n_medium - n_hard  # remainder
        logger.info(
            f"Optimal mix for budget={budget}: "
            f"{frac_easy:.0%} easy / {frac_medium:.0%} medium / "
            f"{frac_hard:.0%} hard / {n_very_hard/budget:.0%} very-hard"
        )

        # Within each bucket, prefer high-IFD samples (instruction informativeness)
        # If IFD not computed, use random selection within bucket
        def sort_by_ifd(lst):
            return sorted(lst, key=lambda s: s.get("_score_ifd", 1.0), reverse=True)

        selected_easy = sort_by_ifd(easy)[:n_easy]
        selected_medium = sort_by_ifd(medium)[:n_medium]
        # Hard: take the hardest from the hard bucket
        hard_sorted = sorted(hard, key=lambda s: s.get("_score_response_ppl", 0), reverse=True)
        selected_very_hard = hard_sorted[:n_very_hard]
        selected_hard = sort_by_ifd(hard_sorted[n_very_hard:])[:n_hard]

        selected = selected_easy + selected_medium + selected_hard + selected_very_hard

        # Backfill: if a bucket had fewer samples than its quota, fill the
        # shortfall from unselected samples (medium first — most informative —
        # then easy, then hard) instead of silently shrinking the selection.
        if len(selected) < budget:
            chosen = {id(s) for s in selected}
            leftovers = [
                s
                for pool in (sort_by_ifd(medium), sort_by_ifd(easy), sort_by_ifd(hard))
                for s in pool
                if id(s) not in chosen
            ]
            shortfall = budget - len(selected)
            selected += leftovers[:shortfall]
            logger.info(f"Backfilled {min(shortfall, len(leftovers))} samples (short difficulty buckets)")

        # Apply FLOW weighting as metadata (for weighted sampling during training)
        ppls = [s.get("_score_response_ppl", 1.0) for s in selected]
        if ppls:
            tau = sorted(ppls)[len(ppls) // 2]  # median
            for s in selected:
                ppl = s.get("_score_response_ppl", tau)
                # FLOW: w = exp(-ppl / tau). Easy → high weight, hard → low weight.
                # This is used as sampling probability during training.
                s["_score_flow_weight"] = round(math.exp(-ppl / max(tau, 1e-6)), 4)

    elif strategy == "flow":
        # Pure FLOW: weight by exp(-loss/median_loss), then sample by weight
        ppls = [s.get("_score_response_ppl", 1.0) for s in samples]
        tau = sorted(ppls)[len(ppls) // 2]
        for s in samples:
            ppl = s.get("_score_response_ppl", tau)
            s["_score_flow_weight"] = math.exp(-ppl / max(tau, 1e-6))
        # Weighted selection without replacement (approximate via sorting by weight)
        weighted = sorted(samples, key=lambda s: s.get("_score_flow_weight", 0), reverse=True)
        selected = weighted[:budget]

    elif strategy == "balanced":
        easy = [s for s in samples if s.get("_score_difficulty_bucket") == "easy"]
        medium = [s for s in samples if s.get("_score_difficulty_bucket") == "medium"]
        hard = [s for s in samples if s.get("_score_difficulty_bucket") == "hard"]
        per_bucket = budget // 3
        selected = easy[:per_bucket] + medium[:per_bucket] + hard[:per_bucket]
        # Fill remaining with medium (most informative)
        remaining = budget - len(selected)
        selected += medium[per_bucket : per_bucket + remaining]

    elif strategy == "medium_focus":
        # InfoSFT-style: prioritize medium-confidence tokens
        ppls = [s.get("_score_response_ppl", 0) for s in samples]
        median_ppl = sorted(ppls)[len(ppls) // 2]
        scored = [(abs(s.get("_score_response_ppl", 0) - median_ppl), s) for s in samples]
        scored.sort(key=lambda x: x[0])
        selected = [s for _, s in scored[:budget]]

    elif strategy == "curriculum":
        # Easy → hard ordering (for multi-epoch progressive training)
        sorted_samples = sorted(samples, key=lambda s: s.get("_score_response_ppl", 0))
        selected = sorted_samples[:budget]
        for i, s in enumerate(selected):
            s["_score_rank"] = i

    elif strategy == "hard_focus":
        # Prioritize hard samples (for large datasets / advanced models)
        sorted_samples = sorted(samples, key=lambda s: s.get("_score_response_ppl", 0), reverse=True)
        selected = sorted_samples[:budget]

    else:
        # Random subset
        import random

        selected = random.sample(samples, budget)

    logger.info(f"Selected {len(selected)} samples with strategy='{strategy}'")
    return selected


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


def split_eval_holdout(samples: list[dict], n_holdout: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Reserve n_holdout random samples as a held-out eval set.

    Returns (train_pool, eval_holdout) — disjoint, deterministic for a given
    seed. Random (not difficulty-stratified) so the eval set is an unbiased
    draw from the same distribution the training pool comes from.
    """
    import random

    if n_holdout <= 0 or n_holdout >= len(samples):
        return samples, []

    rng = random.Random(seed)
    indices = set(rng.sample(range(len(samples)), n_holdout))
    train_pool = [s for i, s in enumerate(samples) if i not in indices]
    holdout = [s for i, s in enumerate(samples) if i in indices]
    return train_pool, holdout


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
    return out_file


def prepare_data(
    model_name: str,
    data_path: str,
    output_path: str,
    messages_field: str = "messages",
    max_samples: int | None = None,
    max_seq_length: int = 8192,
    budget: int | None = None,
    strategy: str = "balanced",
    batch_size: int = 4,
    dataset_split: str = "train",
    output_format: str = "parquet",
    compute_hes: bool = False,
    hes_top_k_pct: float = 0.5,
    eval_holdout: int = 0,
    min_ppl: float = 1.5,
    max_ppl: float = 500.0,
    filter_score: str = "response",
    max_batch_tokens: int = 16384,
) -> Path:
    """Full data preparation pipeline.

    1. Load data
    2. Score with model perplexity
    3. Classify difficulty
    4. Filter outliers
    5. Reserve a held-out eval set (eval_holdout > 0) — never trained on
    6. Select by budget/strategy from the remaining pool
    7. Save scored + ordered dataset (parquet or jsonl) + prepared_meta.json

    Returns the path to the prepared dataset file.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    logger.info(f"Loading data from: {data_path} (split={dataset_split})")
    source_id = str(data_path)
    data_path = Path(data_path)
    if data_path.suffix == ".jsonl":
        with data_path.open() as f:
            samples = [json.loads(line) for line in f if line.strip()]
    elif data_path.suffix == ".json":
        with data_path.open() as f:
            samples = json.load(f)
    elif data_path.suffix == ".parquet":
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=str(data_path), split="train")
        samples = [dict(row) for row in ds]
    else:
        # HuggingFace dataset
        from datasets import load_dataset

        ds = load_dataset(str(data_path), split=dataset_split)
        samples = [dict(row) for row in ds]

    if max_samples and len(samples) > max_samples:
        import random

        random.seed(42)
        samples = random.sample(samples, max_samples)

    logger.info(f"Loaded {len(samples)} samples")

    # Score
    logger.info("Scoring samples with model perplexity...")
    samples = score_samples_with_model(
        model_name, samples, messages_field, max_seq_length, batch_size, max_batch_tokens=max_batch_tokens
    )

    # Optional HES scoring (for reasoning data)
    if compute_hes:
        samples = compute_hes_scores(
            model_name, samples, messages_field, max_seq_length, top_k_pct=hes_top_k_pct, batch_size=batch_size
        )

    # Classify difficulty
    samples = classify_difficulty(samples)

    # Filter
    samples = filter_samples(samples, min_ppl=min_ppl, max_ppl=max_ppl, score=filter_score)
    if not samples:
        hint = (
            "For domain-shifted or multilingual chat data, try `preprocess.max_ppl: 0` "
            "to disable the absolute PPL ceiling and rely on percentile-based selection."
            if max_ppl > 0
            else "All samples were removed by non-PPL filters (usually min_length/max_length). "
            "Check that messages_field matches the dataset schema and that prepare normalizes the conversation format."
        )
        raise ValueError(
            "Preparation produced 0 samples after filtering. "
            f"filter_score={filter_score}, min_ppl={min_ppl}, max_ppl={max_ppl}. "
            f"{hint}"
        )

    # Reserve held-out eval BEFORE selection: an unbiased random draw from the
    # filtered pool, guaranteed disjoint from anything that gets trained on.
    eval_samples: list[dict] = []
    if eval_holdout:
        samples, eval_samples = split_eval_holdout(samples, eval_holdout)
        if eval_samples:
            logger.info(f"Reserved {len(eval_samples)} samples as held-out eval set")

    # Select by budget
    if budget:
        samples = select_by_budget(samples, budget, strategy)

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

    # Stats + manifest (consumed by training to verify provenance)
    ppls = [s["_score_response_ppl"] for s in samples if "_score_response_ppl" in s]
    buckets: dict[str, int] = {}
    for s in samples:
        b = s.get("_score_difficulty_bucket", "unknown")
        buckets[b] = buckets.get(b, 0) + 1

    meta = {
        "model": model_name,
        "source_dataset": source_id,
        "dataset_split": dataset_split,
        "messages_field": messages_field,
        "max_seq_length": max_seq_length,
        "strategy": strategy,
        "budget": budget,
        "num_samples": len(samples),
        "format": out_file.suffix.lstrip("."),
        "output_file": str(out_file),
        "difficulty_distribution": buckets,
        "eval_holdout": len(eval_samples),
        "eval_file": str(eval_file) if eval_file else None,
        "filter_score": filter_score,
        "min_ppl": min_ppl,
        "max_ppl": max_ppl,
    }
    if ppls:
        meta["ppl_stats"] = {
            "min": round(min(ppls), 2),
            "median": round(sorted(ppls)[len(ppls) // 2], 2),
            "max": round(max(ppls), 2),
            "mean": round(sum(ppls) / len(ppls), 2),
        }
    with (Path(output_path) / PREPARED_META).open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Output: {out_file} ({len(samples)} samples)")
    if ppls:
        logger.info(
            f"Perplexity stats: min={min(ppls):.1f}, median={sorted(ppls)[len(ppls)//2]:.1f}, "
            f"max={max(ppls):.1f}, mean={sum(ppls)/len(ppls):.1f}"
        )
        logger.info(f"Difficulty distribution: {buckets}")

    return out_file


# ══════════════════════════════════════════════════════════════════════════════
# HES: HIGH-ENTROPY SUM METRIC (arxiv:2605.22389, May 2026)
# ══════════════════════════════════════════════════════════════════════════════


def compute_hes_scores(
    model_name: str,
    samples: list[dict],
    messages_field: str = "messages",
    max_seq_length: int = 8192,
    top_k_pct: float = 0.5,
    batch_size: int = 4,
    device: str = "auto",
) -> list[dict]:
    """Compute High-Entropy Sum (HES) scores for reasoning quality assessment.

    From arxiv:2605.22389 (Unified Data Selection for LLM Reasoning):
    HES = sum of entropy for only the top-k% highest-entropy tokens in a sample.

    Key insight: Average entropy across all tokens fails to distinguish quality
    because most tokens (function words, punctuation) are low-entropy in both
    good and bad samples. The CRITICAL reasoning tokens (decision points,
    calculations, logical transitions) are the rare high-entropy positions.

    HES is training-free and outperforms perplexity-based scoring for reasoning:
    - Top 20% HES samples match full-dataset SFT performance
    - Low-HES samples actively degrade reasoning when trained on

    The metric captures:
    - Correct reasoning: many high-entropy tokens (model considers alternatives)
    - Rote/template: few high-entropy tokens (model just pattern-matches)
    - Garbage/noise: high average entropy but LOW top-k entropy sum (uniformly confused)

    Args:
        model_name: Model to compute entropy with
        samples: List of sample dicts with messages_field
        messages_field: Key containing chat messages
        max_seq_length: Maximum sequence length
        top_k_pct: Percentage of highest-entropy tokens to sum (default 0.5%)
        batch_size: Batch size for inference
        device: Device for model

    Returns:
        Samples annotated with '_score_hes' field
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Computing HES scores (top_k_pct={top_k_pct}%)")
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    scored = []
    total = len(samples)

    for i in range(0, total, batch_size):
        batch = samples[i : i + batch_size]
        for sample in batch:
            hes = _compute_single_hes(model, tokenizer, sample, messages_field, max_seq_length, top_k_pct)
            sample["_score_hes"] = round(hes, 4)
            scored.append(sample)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  HES scored {min(i + batch_size, total)}/{total}")

    # Normalize HES scores to [0, 1] range for easier comparison
    hes_values = [s["_score_hes"] for s in scored if s["_score_hes"] > 0]
    if hes_values:
        max_hes = max(hes_values)
        min_hes = min(hes_values)
        range_hes = max_hes - min_hes if max_hes > min_hes else 1.0
        for s in scored:
            raw = s["_score_hes"]
            s["_score_hes_normalized"] = round((raw - min_hes) / range_hes, 4) if raw > 0 else 0.0

    return scored


@torch.no_grad()
def _compute_single_hes(
    model,
    tokenizer,
    sample: dict,
    messages_field: str,
    max_seq_length: int,
    top_k_pct: float,
) -> float:
    """Compute HES for a single sample.

    HES = sum of entropy for top-k% highest-entropy tokens.
    """
    messages = _get_messages(sample, messages_field)
    if not messages:
        return 0.0

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)

    encoding = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = encoding["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    if seq_len < 5:
        return 0.0

    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :].float()  # [1, S-1, V]

    # Compute per-token entropy: H = -sum(p * log(p))
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # [S-1]

    # Take top-k% highest entropy tokens and SUM their entropies
    k = max(1, int(len(entropy) * top_k_pct / 100.0))
    top_k_entropy = entropy.topk(k).values
    hes = top_k_entropy.sum().item()

    return hes


# ══════════════════════════════════════════════════════════════════════════════
# MSFT-STYLE PER-SOURCE VALIDATION (arxiv:2603.21606, Mar 2026)
# ══════════════════════════════════════════════════════════════════════════════


def prepare_multi_source(
    model_name: str,
    sources: list[dict],
    output_path: str,
    max_seq_length: int = 8192,
    budget_per_source: int | None = None,
    strategy: str = "optimal",
    compute_hes: bool = False,
    hes_top_k_pct: float = 0.5,
):
    """Prepare multiple data sources independently for MSFT-style training.

    MSFT insight: different sub-datasets overfit at different rates.
    By preparing each source independently:
      1. Each gets its own difficulty scoring (model-relative)
      2. Each gets its own budget allocation (J-shaped within source)
      3. During training, we track per-source val loss to detect overfitting
      4. Sources that overfit early can be dynamically excluded

    This function creates:
      - prepared/{source_name}/scored_data.jsonl per source
      - prepared/manifest.json with source metadata + recommended compute allocation

    Args:
        model_name: Model for scoring
        sources: List of source configs, each with:
            - dataset: path or HF dataset name
            - split: dataset split (default "train")
            - name: human-readable name for this source
            - weight: relative importance (default 1.0)
            - budget: per-source budget override (optional)
            - messages_field: field containing messages (default "messages")
        output_path: Base output directory
        max_seq_length: Max sequence length for scoring
        budget_per_source: Default budget per source (None = use all)
        strategy: Selection strategy per source
        compute_hes: Whether to also compute HES scores (slower, better for reasoning)
        hes_top_k_pct: Top-k% for HES computation
    """
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_name,
        "strategy": strategy,
        "sources": [],
    }

    for src_config in sources:
        name = src_config.get("name", src_config.get("dataset", "unknown"))
        dataset_id = src_config.get("dataset")
        split = src_config.get("split", "train")
        weight = src_config.get("weight", 1.0)
        budget = src_config.get("budget", budget_per_source)
        messages_field = src_config.get("messages_field", "messages")

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing source: {name}")
        logger.info(f"  dataset={dataset_id}, split={split}, weight={weight}")

        # Load
        if dataset_id.endswith(".jsonl"):
            with Path(dataset_id).open() as f:
                samples = [json.loads(line) for line in f if line.strip()]
        else:
            ds = load_dataset(dataset_id, split=split)
            samples = [dict(row) for row in ds]

        logger.info(f"  Loaded {len(samples)} samples")

        # Score with perplexity
        samples = score_samples_with_model(model_name, samples, messages_field, max_seq_length, batch_size=4)

        # Optional HES scoring (for reasoning data)
        if compute_hes:
            samples = compute_hes_scores(
                model_name, samples, messages_field, max_seq_length, top_k_pct=hes_top_k_pct, batch_size=4
            )

        # Classify and filter
        samples = classify_difficulty(samples)
        samples = filter_samples(samples)

        # Select by budget
        if budget and len(samples) > budget:
            samples = select_by_budget(samples, budget, strategy)

        # Save per-source
        src_output = output_base / name.replace("/", "_").replace(" ", "_")
        src_output.mkdir(parents=True, exist_ok=True)
        out_file = src_output / "scored_data.jsonl"
        with out_file.open("w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Compute source statistics for manifest
        ppls = [s.get("_score_response_ppl", 0) for s in samples if "_score_response_ppl" in s]
        median_ppl = sorted(ppls)[len(ppls) // 2] if ppls else 0
        buckets = {}
        for s in samples:
            b = s.get("_score_difficulty_bucket", "unknown")
            buckets[b] = buckets.get(b, 0) + 1

        src_meta = {
            "name": name,
            "dataset": dataset_id,
            "output_path": str(out_file),
            "num_samples": len(samples),
            "weight": weight,
            "median_ppl": round(median_ppl, 2),
            "difficulty_distribution": buckets,
            # MSFT: estimate relative learning speed (lower ppl = faster learning = earlier overfit)
            "estimated_overfit_risk": "high" if median_ppl < 5 else "medium" if median_ppl < 20 else "low",
        }
        if compute_hes:
            hes_vals = [s.get("_score_hes", 0) for s in samples if s.get("_score_hes", 0) > 0]
            src_meta["median_hes"] = round(sorted(hes_vals)[len(hes_vals) // 2], 2) if hes_vals else 0

        manifest["sources"].append(src_meta)
        logger.info(f"  Saved {len(samples)} samples to {out_file}")
        logger.info(f"  Median PPL: {median_ppl:.1f}, Overfit risk: {src_meta['estimated_overfit_risk']}")

    # Compute MSFT-style compute allocation recommendation
    # Sources with higher overfit risk should get LESS compute (fewer epochs)
    total_weight = sum(s["weight"] for s in manifest["sources"])
    for src in manifest["sources"]:
        # Normalize weight
        src["normalized_weight"] = round(src["weight"] / total_weight, 4)
        # MSFT recommendation: overfitting sources get proportionally less training
        risk_factor = {"high": 0.5, "medium": 1.0, "low": 1.5}.get(src["estimated_overfit_risk"], 1.0)
        src["recommended_epoch_multiplier"] = risk_factor

    # Save manifest
    manifest_file = output_base / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nManifest saved to {manifest_file}")
    logger.info(f"Total sources: {len(manifest['sources'])}")


def prepare_from_config(config) -> Path:
    """Run preparation driven by the SAME YAML config used for training.

    Reads: model.name_or_path, data.dataset, data.dataset_split,
    data.messages_field, data.max_seq_length and the preprocess: section.
    Training with preprocess.enabled=true then picks the output up
    automatically from preprocess.output_dir.
    """
    return prepare_data(
        model_name=config.model.name_or_path,
        data_path=config.data.dataset,
        output_path=config.preprocess.output_dir,
        messages_field=config.data.messages_field,
        max_samples=config.preprocess.max_samples or None,
        max_seq_length=config.data.max_seq_length,
        budget=config.preprocess.budget or None,
        strategy=config.preprocess.strategy,
        batch_size=config.preprocess.batch_size,
        dataset_split=config.data.dataset_split,
        output_format=config.preprocess.format,
        compute_hes=config.preprocess.hes,
        hes_top_k_pct=config.preprocess.hes_top_k_pct,
        eval_holdout=config.preprocess.eval_holdout,
        min_ppl=config.preprocess.min_ppl,
        max_ppl=config.preprocess.max_ppl,
        filter_score=config.preprocess.filter_score,
        max_batch_tokens=config.preprocess.max_batch_tokens,
    )


def main():
    """CLI entry point for data preparation.

    Two modes:
      1. Config-driven (recommended): pgs prepare --config train_config.yaml
         Supports the same --section.field overrides as training, e.g.
         pgs prepare --config cfg.yaml --preprocess.budget 5000 --preprocess.strategy curriculum
      2. Standalone flags: pgs prepare --model M --data D --output prepared/
    """
    if "--config" in sys.argv:
        from palingenesis.config import Config

        config = Config.from_cli()
        prepare_from_config(config)
        return

    import argparse

    parser = argparse.ArgumentParser(description="Prepare SFT data with difficulty scoring")
    parser.add_argument("--model", required=True, help="Model name/path for scoring")
    parser.add_argument("--data", required=True, help="Data path (JSONL, JSON, parquet, or HF dataset)")
    parser.add_argument("--output", default="./prepared", help="Output directory")
    parser.add_argument("--messages_field", default="messages", help="Field containing messages")
    parser.add_argument("--max_samples", type=int, help="Max samples to process")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--budget", type=int, help="Target number of samples to select")
    parser.add_argument(
        "--strategy",
        default="optimal",
        choices=["optimal", "balanced", "medium_focus", "curriculum", "hard_focus", "flow", "random"],
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Max samples per scoring forward")
    parser.add_argument(
        "--max_batch_tokens", type=int, default=16384,
        help="Padded-token cap per scoring forward (bounds logits memory; 32768+ OK on 80GB GPUs)",
    )
    parser.add_argument("--split", default="train", help="Dataset split (for HF datasets)")
    parser.add_argument("--format", default="parquet", choices=["parquet", "jsonl"], help="Output format")
    parser.add_argument("--hes", action="store_true", help="Also compute HES reasoning quality scores")
    parser.add_argument("--hes_top_k", type=float, default=0.5, help="Top-k%% tokens for HES metric")
    parser.add_argument(
        "--eval_holdout", type=int, default=0,
        help="Reserve N samples as a held-out eval set (eval_data.parquet, excluded from training)",
    )
    parser.add_argument("--min_ppl", type=float, default=1.5, help="Outlier filter lower PPL bound")
    parser.add_argument("--max_ppl", type=float, default=500.0, help="Outlier filter upper PPL bound (<=0 disables)")
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
    )


if __name__ == "__main__":
    main()
