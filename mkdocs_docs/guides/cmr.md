# Common-Mode Rejection (CMR)

`cmr:` holds a model's overall propensity for a decision fixed while RL or on-policy distillation trains a skill.
It leaves the model's ability to tell when the decision is right free to change. It is experimental: available in
`pgs rl` and the OPD trainer, single GPU.

!!! info "When to use it"
    Use CMR when a run you are about to start (agentic [RL](rl.md), [on-policy distillation](distillation.md)) is
    likely to drag a decision such as "call a tool" up or down across the board, and you want only the model's
    *discrimination* of that decision to change. You need twin prompts for the decision. To repair a shift that already
    happened, without training, use [Margin Surgery](surgery.md) instead.

The minimal setup is one twin file and one decision token, added to an RL or OPD config:

```yaml
cmr:
  enabled: true
  families:
    fc: data/twins_fc.jsonl
  decision_token: "<tool_call>"
```

The full option list is under [Configuration](#configuration).

## The problem it targets

Agentic RL changes when a model calls tools, and the change tends to move along one axis. Needed calls and unneeded
calls go up together, or both go down. On a 0.8B model, an agentic RL run cut When2Call calls-when-needed from .83 to
.54 while hallucinated calls fell from .24 to .09. Preference tuning and weight interpolation slid along the same line.

Anchors that penalize distance from the start (KL, L2) cannot prevent this. Their gradient is zero at the start, so
they act only after the drift has happened.

## What it does

You provide **twins**: the same request in two versions, one where the decision is right ($x^+$, e.g. a matching tool is
available) and one where it is not ($x^-$, e.g. the tool set swapped for irrelevant tools). The decision margin at the
first assistant token is

$$
m(x) = \log p(d \mid x) - \log\big(1 - p(d \mid x)\big)
$$

where $d$ is the decision token (`decision_token`, e.g. `<tool_call>`) and $p$ the model's next-token probability.
{: .pg-where }

- **Common mode**, $m(x^+) + m(x^-)$: the overall propensity.
- **Differential mode**, $m(x^+) - m(x^-)$: the discrimination.

Each step, CMR estimates the common-mode direction of each family $f$ of twins,

$$
u_f = \operatorname{mean}_{(x^+,\,x^-) \in f}\ \nabla_\theta \big[m(x^+) + m(x^-)\big],
$$

over `pairs_per_refresh` twins. The families' directions are orthonormalized. After the optimizer step, CMR removes the
parameter update's component along each $u_f$.

CMR corrects the update rather than the gradient. An adaptive optimizer rescales a projected gradient back off the
subspace, and momentum carries the old common mode forward. To first order the common margins stay where they were,
while the differential margins and everything else train freely.

With `mode: target`, the common margin is steered toward a value of your choice instead of held.

### Decisions after a call

A side can carry `prefix_calls`, a list of `{name, arguments}`. CMR then renders an assistant turn holding those calls
and reads the margin right after the last `</tool_call>`, so the decision is whether to end the turn or make another
call. Set that family's decision token to `<|im_end|>` with `decision_tokens: {cont: "<|im_end|>"}`. `prefix_content`
adds the text the assistant wrote before its calls. `pos_only: [family]` holds only the positive side's margin, which
fits when the negative side is saturated or not a minimal pair.

## Configuration

```yaml
cmr:
  enabled: true
  families:                      # twin files, one per format family
    fc: data/twins_fc.jsonl      # {"pos": {"messages": [...], "tools": [...]}, "neg": {...}} per line
    w2c: data/twins_w2c.jsonl
  decision_token: "<tool_call>"  # must be a single token of the student's tokenizer
  pairs_per_refresh: 16          # twins per family per direction estimate
  refresh_every: 1               # steps between direction estimates
  mode: hold                     # or target, with target: {fc: 0.0} and gain: 0.1
```

Each step logs `cmr/component/<family>` (the update's raw common-mode component), `cmr/correction/<family>`,
`cmr/common/<family>` (the mean common margin of the refresh batch) and `cmr/u_norm/<family>`.

## Check that it worked

- With `mode: hold`, `cmr/common/<family>` stays near its starting value while the run trains; with `mode: target` it
  moves toward the target.
- `cmr/residual/<family>` (what is left of the step along $u_f$ after the correction) stays near 0 with `mode: hold`;
  `cmr/component/<family>` shows how much common-mode change the optimizer tried to make.
- On held-out twins and on your decision benchmark, the needed and the unneeded rate no longer move together.

| Problem | Fix |
|---|---|
| refused at startup with FSDP | CMR is single-GPU only |
| the decision token is not one token | pick a `decision_token` that is a single token of the student's tokenizer |
| margins look meaningless with thinking on | use non-thinking twins: with thinking, the first assistant token is inside the think block |
| out of memory | each family keeps an fp32 copy of the parameters (about 3 GB per family at 0.8B): use fewer families |

## Cost and limits

- **Cost:** one forward and backward per family per refresh, on the twins' prompts, using the last position only. The
  model also keeps one fp32 copy of the parameters per family, about 3 GB per family at 0.8B.
- **Single GPU only:** CMR is refused with FSDP.
- **Non-thinking twins only:** with thinking on, the first assistant token is inside the think block.
- **First-token decision:** the margin is read at the first assistant token. If your model writes text before its tool
  calls, check how often its sampled calls open with the decision token before relying on the margin.
- **Status:** experimental. The mechanism (gradient projection) is old; the common / differential split of twin
  margins as a training constraint is, as far as we know, new. Evaluate it on your own decision benchmarks.

The twin format is shared with [Margin Surgery](surgery.md) (`pgs surgery margins` measures common and differential
margins of any checkpoint). The surrounding run is configured as in [Reinforcement Learning](rl.md) or
[On-Policy Distillation](distillation.md).
