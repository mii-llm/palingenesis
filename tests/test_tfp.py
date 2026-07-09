"""Tests for Threshold Filtering Packing (TFP) ordering.

Tests the core algorithm without requiring sentence-transformers (uses mock embeddings).
"""

import sys

sys.path.insert(0, "src")

import numpy as np


def test_greedy_tsp_basic_ordering():
    """Test that greedy TSP produces a valid permutation."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    # Create 10 embeddings in 2D (easy to reason about)
    np.random.seed(42)
    embeddings = np.random.randn(10, 16).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    ordering = _greedy_tsp_with_threshold(embeddings, sim_low=0.0, sim_high=1.0, seed=42)

    # Should be a valid permutation of [0, 9]
    assert len(ordering) == 10
    assert sorted(ordering) == list(range(10)), "Should be a permutation"
    print(f"  Ordering: {ordering}")
    print("✓ test_greedy_tsp_basic_ordering PASSED\n")


def test_threshold_filtering_groups_similar():
    """Test that threshold filtering puts similar samples adjacent."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    # Create 3 clusters of embeddings
    np.random.seed(42)
    cluster_centers = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # cluster A
            [0.0, 1.0, 0.0, 0.0],  # cluster B
            [0.0, 0.0, 1.0, 0.0],  # cluster C
        ],
        dtype=np.float32,
    )

    embeddings = []
    labels = []
    for i, center in enumerate(cluster_centers):
        for _ in range(5):
            noise = np.random.randn(4).astype(np.float32) * 0.1
            vec = center + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
            labels.append(i)

    embeddings = np.array(embeddings)

    ordering = _greedy_tsp_with_threshold(embeddings, sim_low=0.3, sim_high=0.95, seed=42)

    # Check that similar items tend to be adjacent
    ordered_labels = [labels[i] for i in ordering]

    # Count how many adjacent pairs share the same cluster
    same_cluster_pairs = sum(1 for i in range(len(ordered_labels) - 1) if ordered_labels[i] == ordered_labels[i + 1])

    # With good ordering, most adjacent pairs should be same-cluster
    # Random would give ~1/3 * 14 ≈ 4.7 same-cluster pairs
    # TFP should give significantly more
    print(f"  Same-cluster adjacent pairs: {same_cluster_pairs}/14")
    print(f"  Cluster sequence: {ordered_labels}")
    assert same_cluster_pairs >= 8, f"TFP should group clusters, got only {same_cluster_pairs}/14 adjacent"
    print("✓ test_threshold_filtering_groups_similar PASSED\n")


def test_threshold_high_prevents_duplicates():
    """Test that high threshold prevents nearly-identical samples from being adjacent."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    np.random.seed(42)
    # Create embeddings where some are nearly identical
    base = np.random.randn(8).astype(np.float32)
    base = base / np.linalg.norm(base)

    embeddings = []
    # 5 nearly-identical embeddings (sim > 0.99)
    for _ in range(5):
        noise = np.random.randn(8).astype(np.float32) * 0.01
        vec = base + noise
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)

    # 5 different embeddings
    for _ in range(5):
        vec = np.random.randn(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)

    embeddings = np.array(embeddings)

    # With sim_high=0.9, the 5 identical samples should be spread apart
    ordering = _greedy_tsp_with_threshold(embeddings, sim_low=0.1, sim_high=0.9, seed=42)

    # Check that the first 5 (similar) items aren't all grouped together
    first_five_positions = [ordering.index(i) for i in range(5)]
    consecutive_count = sum(
        1
        for i in range(len(first_five_positions) - 1)
        if abs(first_five_positions[i] - first_five_positions[i + 1]) == 1
    )

    print(f"  Positions of similar items: {sorted(first_five_positions)}")
    print(f"  Adjacent similar pairs: {consecutive_count}")
    # Threshold filtering should prevent ALL 5 from being consecutive
    assert consecutive_count < 4, "High threshold should spread similar samples"
    print("✓ test_threshold_high_prevents_duplicates PASSED\n")


def test_ordering_deterministic():
    """Test that ordering is deterministic given same seed."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    np.random.seed(123)
    embeddings = np.random.randn(20, 32).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    order1 = _greedy_tsp_with_threshold(embeddings, 0.1, 0.9, seed=42)
    order2 = _greedy_tsp_with_threshold(embeddings, 0.1, 0.9, seed=42)

    assert order1 == order2, "Same seed should give same ordering"
    print("✓ test_ordering_deterministic PASSED\n")


def test_small_dataset_edge_cases():
    """Test edge cases: 0, 1, 2 samples."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    # Empty
    result = _greedy_tsp_with_threshold(np.zeros((0, 4), dtype=np.float32), 0.1, 0.9, 42)
    assert result == []

    # Single item
    emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = _greedy_tsp_with_threshold(emb, 0.1, 0.9, 42)
    assert result == [0]

    # Two items
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = _greedy_tsp_with_threshold(emb, 0.0, 1.0, 42)
    assert sorted(result) == [0, 1]

    print("✓ test_small_dataset_edge_cases PASSED\n")


def test_ordering_quality_vs_random():
    """Verify TFP produces better locality than random ordering."""
    from palingenesis.tfp import _greedy_tsp_with_threshold

    np.random.seed(42)
    n = 100
    dim = 32

    # Create clustered embeddings (5 clusters of 20)
    embeddings = []
    for cluster in range(5):
        center = np.random.randn(dim).astype(np.float32)
        center = center / np.linalg.norm(center)
        for _ in range(20):
            noise = np.random.randn(dim).astype(np.float32) * 0.2
            vec = center + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

    embeddings = np.array(embeddings)

    # TFP ordering
    tfp_order = _greedy_tsp_with_threshold(embeddings, 0.2, 0.9, seed=42)

    # Random ordering
    random_order = list(range(n))
    np.random.shuffle(random_order)

    # Measure: average similarity between adjacent pairs
    def avg_adjacent_sim(order):
        sims = []
        for i in range(len(order) - 1):
            sim = embeddings[order[i]] @ embeddings[order[i + 1]]
            sims.append(sim)
        return np.mean(sims)

    tfp_sim = avg_adjacent_sim(tfp_order)
    random_sim = avg_adjacent_sim(random_order)

    print(f"  TFP avg adjacent similarity:    {tfp_sim:.4f}")
    print(f"  Random avg adjacent similarity: {random_sim:.4f}")
    print(f"  Improvement: {(tfp_sim - random_sim) / abs(random_sim) * 100:.1f}%")
    assert tfp_sim > random_sim, "TFP should have higher adjacent similarity than random"
    print("✓ test_ordering_quality_vs_random PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TFP (THRESHOLD FILTERING PACKING) TESTS")
    print("=" * 60 + "\n")

    test_greedy_tsp_basic_ordering()
    test_threshold_filtering_groups_similar()
    test_threshold_high_prevents_duplicates()
    test_ordering_deterministic()
    test_small_dataset_edge_cases()
    test_ordering_quality_vs_random()

    print("=" * 60)
    print("ALL TFP TESTS PASSED ✓")
    print("=" * 60)
