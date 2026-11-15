"""Parse the trainers' step and evaluation log lines.

`pgs train` (SFT, DEFT, DPO, off-policy distillation) logs one line per optimizer step, as
key=value pairs whose order and set depend on the objective, e.g.

    step=12 loss=0.9027 lr=5.00e-07 tok/s=897 grad_norm=283.6 dt=54.9s
    step=12 loss=0.3121 ce=1.2044 lr=... (DEFT)
    step=12 loss=0.2917 acc=1.000 margin=2.750 lr=... eval=0.3454 (DPO)
    step=0 eval=0.3454 (before training)

Other lines also contain "step=" and "loss" (e.g. "Best model updated: step=10,
eval_loss=0.3454"); only lines with a `loss=` key of their own count.

`pgs rl` and `pgs distill` log every metric of a step or an evaluation, sorted by name, after the
kind and the step:

    step 12 | grad_norm=0.41 loss=0.0123 lr=1e-06 reward=0.375 rewards/correct=0.4 time/step=31.2
    eval 12 | reward=0.41 rewards/correct=0.45
"""

import re

_STEP = re.compile(r"(?:^|[\s|])step=(\d+)\s")
_KV = re.compile(r"(?<![\w/])([a-z_/]+)=([^\s,]+)")
# RL / OPD: "<kind> <step> | k=v k=v ..." (the logger's own prefix may precede it)
_KIND = re.compile(r"(?:^|[\s|])(step|train|eval) (\d+) \| (.*)$")


def _number(value: str) -> float | None:
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


def _parse(text: str) -> tuple[dict[int, dict], dict[int, dict]]:
    """(steps, evaluations) by step. A step logged twice (a resumed run) keeps its last occurrence."""
    steps: dict[int, dict] = {}
    evals: dict[int, dict] = {}
    for line in text.splitlines():
        if m := _KIND.search(line):
            fields = {}
            for pair in m.group(3).split():  # names may hold digits, dots, dashes (k1/<teacher>)
                key, sep, value = pair.partition("=")
                number = _number(value) if sep else None
                if number is not None:
                    fields[key] = number
            fields["step"] = int(m.group(2))
            (evals if m.group(1) == "eval" else steps)[fields["step"]] = fields
            continue
        m = _STEP.search(line)
        if not m:
            continue
        fields = {k: v for k, v in ((k, _number(v)) for k, v in _KV.findall(line[m.start() :])) if v is not None}
        step = int(m.group(1))
        if "eval" in fields:
            evals[step] = {"step": step, "loss": fields["eval"]}
        if "loss" in fields:
            fields["step"] = step
            steps[step] = fields
    return steps, evals


def parse_steps(text: str) -> list[dict]:
    """One dict per logged training step ({"step": int, "loss": float, ...}) of any trainer, ordered
    by step. Only steps that logged a loss."""
    steps, _ = _parse(text)
    return [steps[s] for s in sorted(steps) if "loss" in steps[s]]


def parse_evals(text: str) -> list[dict]:
    """One dict per logged evaluation, ordered by step: {"step", "loss"} from `pgs train` (the eval
    loss), every logged metric (reward, dev_kl, ...) from `pgs rl` / `pgs distill`."""
    _, evals = _parse(text)
    return [evals[s] for s in sorted(evals)]
