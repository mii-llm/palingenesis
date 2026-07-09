"""Autopilot: autonomous training optimization.

Autopilot runs your training with maximum intelligence:
  1. Profiles hardware and auto-sizes config
  2. Sweeps learning rates with collapse detection
  3. Monitors model behavior (KL, entropy, drift)
  4. Validates periodically and stops early if needed
  5. Optionally ablates data mix ratios

Usage:
    palingenesis autopilot --model meta-llama/Llama-3.1-8B-Instruct \\
                          --dataset your-org/data --val_dataset your-org/val \\
                          --output ./best-model --budget 2h
"""

from palingenesis.autopilot.run import autopilot as autopilot
