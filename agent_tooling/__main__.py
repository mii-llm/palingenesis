"""Entry point: python -m agent_tooling <command>

Dispatches to the appropriate diagnostic tool.
"""

import sys


TOOLS = {
    "diagnose": "agent_tooling.diagnose",
    "inspect_batch": "agent_tooling.inspect_batch",
    "validate_masking": "agent_tooling.validate_masking",
    "check_loss": "agent_tooling.check_loss",
    "check_gradients": "agent_tooling.check_gradients",
    "profile_memory": "agent_tooling.profile_memory",
    "monitor_run": "agent_tooling.monitor_run",
}

HELP = """
palingenesis diagnostic tools

Usage:
    python -m agent_tooling <tool> [args...]

Available tools:
    diagnose          All-in-one health check (pre/post/full)
    inspect_batch     Visualize tokenization + masking (colored terminal)
    validate_masking  Verify assistant-only masking across N samples
    check_loss        Analyze loss curves for anomalies
    check_gradients   Per-layer gradient norm analysis (needs GPU)
    profile_memory    Estimate peak GPU memory from config
    monitor_run       Monitor active training from log output

Examples:
    python -m agent_tooling diagnose --config configs/llama3_8b.yaml --mode pre
    python -m agent_tooling check_loss --log_file train.log
    python -m agent_tooling profile_memory --config configs/long_context.yaml
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0)

    tool = sys.argv[1]
    if tool not in TOOLS:
        print(f"Unknown tool: {tool}")
        print(f"Available: {', '.join(TOOLS.keys())}")
        sys.exit(1)

    # Remove the tool name from argv so argparse in the tool sees correct args
    sys.argv = [sys.argv[0] + " " + tool] + sys.argv[2:]

    # Import and run
    import importlib

    module = importlib.import_module(TOOLS[tool])
    module.main()


if __name__ == "__main__":
    main()
