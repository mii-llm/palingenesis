"""Source-checkout shim: the diagnostics now live in `palingenesis.agent_tooling`.

Keeps `python -m agent_tooling.<tool>` working from a repository checkout. This
directory is not shipped in the wheel or the sdist; installed users run
`pgs diagnose ...` or `python -m palingenesis.agent_tooling.<tool>`.
"""

import sys
from pathlib import Path

try:
    import palingenesis.agent_tooling as _real
except ModuleNotFoundError:  # checkout without an installed palingenesis
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import palingenesis.agent_tooling as _real

__path__ = list(_real.__path__)
