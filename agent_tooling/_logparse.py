"""Parse the trainer's step log lines.

The trainer logs one line per optimizer step, as key=value pairs whose order
and set depend on the objective, e.g.

    step=12 loss=0.9027 lr=5.00e-07 tok/s=897 grad_norm=283.6 dt=54.9s
    step=12 loss=0.3121 ce=1.2044 lr=... (DEFT)
    step=12 loss=0.2917 acc=1.000 margin=2.750 lr=... eval=0.3454 (DPO)

Other lines also contain "step=" and "loss" (e.g. "Best model updated: step=10,
eval_loss=0.3454"); only lines with a `loss=` key of their own count.
"""

import re

_STEP = re.compile(r"(?:^|[\s|])step=(\d+)\s")
_KV = re.compile(r"(?<![\w/])([a-z_/]+)=([^\s,]+)")


def _number(value: str) -> float | None:
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


def parse_steps(text: str) -> list[dict]:
    """One dict per logged step ({"step": int, "loss": float, ...}), ordered by
    step. A step logged twice (a resumed run) keeps its last occurrence."""
    by_step: dict[int, dict] = {}
    for line in text.splitlines():
        m = _STEP.search(line)
        if not m:
            continue
        fields = {k: _number(v) for k, v in _KV.findall(line[m.start() :])}
        if fields.get("loss") is None:
            continue
        fields = {k: v for k, v in fields.items() if v is not None}
        fields["step"] = int(m.group(1))
        by_step[fields["step"]] = fields
    return [by_step[s] for s in sorted(by_step)]
