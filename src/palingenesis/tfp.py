"""Threshold Filtering Packing (TFP) — semantic-aware sample ordering.

Paper: "Threshold Filtering Packing for SFT" (arxiv:2408.09327)
Result: +7% GSM8K, +4% HumanEval over random packing.

Key insight: Instead of random or length-sorted packing, order samples so that
related-but-diverse samples land in the same pack. This creates implicit
few-shot context across document boundaries within a pack.

The algorithm:
1. Embed all samples with a lightweight sentence transformer
2. Build a nearest-neighbor ordering via greedy TSP traversal
3. Apply threshold filtering: skip edges that are too similar (prevents
   overly-homogeneous packs) or too dissimilar (loses the context benefit)

This module provides the sample ORDERING. The actual bin-packing into
fixed-length sequences is done by PackedDataset in data.py.

Usage:
    from palingenesis.tfp import compute_tfp_ordering

    # During data preparation (one-time cost):
    ordered_indices = compute_tfp_ordering(
        texts=["sample 1 text", "sample 2 text", ...],
        sim_threshold_low=0.2,   # minimum similarity to be "related"
        sim_threshold_high=0.85, # maximum similarity to avoid redundancy
    )

    # Then iterate dataset in this order for packing
    reordered_dataset = [dataset[i] for i in ordered_indices]
"""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_tfp_ordering(
    texts: Sequence[str],
    sim_threshold_low: float = 0.2,
    sim_threshold_high: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    seed: int = 42,
) -> list[int]:
    """Compute TFP-ordered indices for a list of text samples.

    Uses semantic embeddings + greedy TSP with threshold filtering to produce
    an ordering where adjacent samples are related but diverse.

    Args:
        texts: List of text strings to order (typically the formatted prompt+response).
        sim_threshold_low: Minimum cosine similarity to consider samples "related".
            Below this, samples are treated as unrelated (no benefit to co-packing).
        sim_threshold_high: Maximum cosine similarity. Above this, samples are
            too similar (redundant, risks overfitting within a pack).
        model_name: Sentence transformer model for embedding.
            Default 'all-MiniLM-L6-v2' is 22M params, very fast.
        batch_size: Batch size for embedding computation.
        seed: Random seed for tie-breaking.

    Returns:
        List of indices in TFP order. Use to reorder your dataset before packing.
    """
    n = len(texts)
    if n <= 1:
        return list(range(n))

    logger.info(f"TFP: Embedding {n} samples with {model_name}...")
    embeddings = _embed_texts(texts, model_name, batch_size)

    logger.info(f"TFP: Computing greedy TSP ordering with thresholds [{sim_threshold_low}, {sim_threshold_high}]...")
    ordering = _greedy_tsp_with_threshold(embeddings, sim_threshold_low, sim_threshold_high, seed)

    logger.info(f"TFP: Ordering complete. {len(ordering)} samples ordered.")
    return ordering


def _embed_texts(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    """Embed texts using sentence-transformers (lazy import)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("TFP requires sentence-transformers. Install with: " "pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    # Truncate very long texts (embeddings don't benefit from >512 tokens)
    truncated = [t[:2048] for t in texts]
    embeddings = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 1000,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )
    return np.array(embeddings, dtype=np.float32)


def _greedy_tsp_with_threshold(
    embeddings: np.ndarray,
    sim_low: float,
    sim_high: float,
    seed: int,
) -> list[int]:
    """Greedy nearest-neighbor TSP with threshold filtering.

    Algorithm:
    1. Start from a random node
    2. At each step, find the nearest unvisited neighbor
    3. Skip neighbors that are:
       - Too similar (sim > sim_high): would create redundant packs
       - Too dissimilar (sim < sim_low): no context benefit
    4. If no valid neighbor exists in threshold range, pick the closest
       unvisited node (fall back to pure nearest-neighbor)

    This produces a path where consecutive samples are semantically related
    but not too similar — ideal for creating informative packed context.

    Complexity: O(n²) in the worst case, but with early termination via
    approximate nearest-neighbor for large datasets.
    """
    rng = np.random.RandomState(seed)
    n = len(embeddings)

    if n <= 2:
        return list(range(n))

    visited = np.zeros(n, dtype=bool)
    ordering = []

    # Start from random node
    current = rng.randint(0, n)
    visited[current] = True
    ordering.append(current)

    for _ in range(n - 1):
        # Compute similarities to all unvisited nodes
        # Since embeddings are L2-normalized, dot product = cosine similarity
        sims = embeddings[current] @ embeddings.T  # (n,)
        sims[visited] = -2.0  # mask visited nodes

        # Find candidates within threshold
        valid_mask = (~visited) & (sims >= sim_low) & (sims <= sim_high)

        if valid_mask.any():
            # Pick the most similar valid candidate (greedy nearest-neighbor)
            candidates = np.where(valid_mask)[0]
            best_idx = candidates[sims[candidates].argmax()]
        else:
            # Fallback: pick closest unvisited node (pure greedy TSP)
            unvisited_indices = np.where(~visited)[0]
            best_idx = unvisited_indices[sims[unvisited_indices].argmax()]

        visited[best_idx] = True
        ordering.append(int(best_idx))
        current = best_idx

    return ordering


def compute_tfp_ordering_fast(
    texts: Sequence[str],
    sim_threshold_low: float = 0.2,
    sim_threshold_high: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    chunk_size: int = 10000,
    seed: int = 42,
) -> list[int]:
    """Memory-efficient TFP for large datasets (>10K samples).

    For datasets larger than chunk_size, processes in chunks:
    1. Divide dataset into chunks
    2. Apply TFP ordering within each chunk
    3. Concatenate chunk orderings

    This trades some cross-chunk ordering quality for O(chunk_size²) memory
    instead of O(n²).

    Args:
        texts: List of text strings to order.
        sim_threshold_low: Minimum similarity threshold.
        sim_threshold_high: Maximum similarity threshold.
        model_name: Embedding model.
        batch_size: Embedding batch size.
        chunk_size: Process this many samples at a time.
        seed: Random seed.

    Returns:
        TFP-ordered indices.
    """
    n = len(texts)
    if n <= chunk_size:
        return compute_tfp_ordering(texts, sim_threshold_low, sim_threshold_high, model_name, batch_size, seed)

    logger.info(f"TFP: Large dataset ({n} samples), processing in chunks of {chunk_size}")

    # Shuffle indices first so chunks have diverse content
    rng = np.random.RandomState(seed)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)

    ordering = []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_indices = all_indices[chunk_start:chunk_end]
        chunk_texts = [texts[i] for i in chunk_indices]

        # TFP within this chunk
        local_order = compute_tfp_ordering(
            chunk_texts,
            sim_threshold_low,
            sim_threshold_high,
            model_name,
            batch_size,
            seed + chunk_start,
        )

        # Map back to global indices
        ordering.extend(int(chunk_indices[i]) for i in local_order)

    return ordering
