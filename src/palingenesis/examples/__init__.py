"""Example configs shipped with the package: `pgs init <name>` writes one to the current directory.

pgs init --list            the examples and what they train
pgs init sft               ./sft.yaml (edit the model and the dataset, then: pgs train --config sft.yaml)
pgs init rl my_rl.yaml     another file name
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

EXAMPLES = {
    "sft": ("pgs train", "supervised fine-tuning on chat data (any GPU with 24+ GB)"),
    "dpo": ("pgs train", "preference optimization (DPO) on top of an SFT checkpoint"),
    "distill": ("pgs distill", "on-policy distillation from a larger teacher (needs the vllm extra)"),
    "rl": ("pgs rl", "RL with a verifiable math reward (needs the rl and vllm extras)"),
}


def example_text(name: str) -> str:
    """The YAML text of a shipped example config."""
    if name not in EXAMPLES:
        raise KeyError(f"unknown example {name!r}; available: {', '.join(EXAMPLES)}")
    return resources.files(__package__).joinpath(f"{name}.yaml").read_text()


def write_example(name: str, dest: str | Path | None = None, force: bool = False) -> Path:
    """Write example `name` to `dest` (default ./<name>.yaml); refuses to overwrite unless force."""
    text = example_text(name)
    path = Path(dest) if dest else Path(f"{name}.yaml")
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (pass --force to overwrite)")
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgs init", description="Write an example config to start from.")
    ap.add_argument("name", nargs="?", choices=sorted(EXAMPLES), help="the example to write")
    ap.add_argument("dest", nargs="?", help="output file (default: ./<name>.yaml)")
    ap.add_argument("--list", action="store_true", help="list the examples")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.list or a.name is None:
        print("Example configs (pgs init <name> [dest]):")
        for name, (cmd, what) in EXAMPLES.items():
            print(f"  {name:<8} {what}  ->  {cmd} --config {name}.yaml")
        return 0
    try:
        path = write_example(a.name, a.dest, a.force)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {path}. Edit the model and the data, then run: {EXAMPLES[a.name][0]} --config {path}")
    return 0
