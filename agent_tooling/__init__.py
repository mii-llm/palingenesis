"""
Agent tooling for diagnosing agentic SFT training.

Standalone scripts an AI agent or developer can run to check if training is healthy.
Each tool can be invoked independently — no GPU required for most analysis tools.

Tools:
    inspect_batch      — Visualize processed batches: tokens, masks, label alignment
    validate_masking   — Verify assistant-only masking across many samples
    check_loss         — Analyze loss curves for NaN, spikes, plateaus, divergence
    check_gradients    — Per-layer gradient norm analysis (live or from logs)
    profile_memory     — Estimate peak memory for a given config before training
    monitor_run        — Monitor active training: loss, throughput, ETA, health
    diagnose           — All-in-one health check: runs all diagnostics and reports
"""
