"""Shared path setup for agent_tooling scripts.

Ensures `palingenesis` package is importable whether or not it's installed.
Import this at the top of any agent_tooling script that needs the training code.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
