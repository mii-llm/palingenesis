"""A frozen reference policy for KL tracking and an optional KL penalty (loss.kl_ref / kl_coef / kl_ref_reset).

Forgetting under RL follows the KL drift from the starting policy (RL's Razor, 2509.04259); ProRL (2505.24864) keeps a
small KL to a reference that is reset to the current policy when progress stalls. The reference is a bf16 copy of a
causal LM: it scores the sampled tokens of the batch once per step (no gradient), and the trainer logs the per-token
k3 estimator exp(r) - r - 1 with r = log π_ref - log π (Schulman), adding kl_coef x the same token weights as the
policy loss when kl_coef > 0.

With LoRA adapters (lora.rank) and the starting policy as reference, the reference is the policy itself with its
adapters switched off (AdapterFreeReference): the frozen base is the starting policy, so no copy is loaded.
"""

from __future__ import annotations

from typing import Any

import torch

from palingenesis.logits import final_hidden_states, output_head
from palingenesis.opd.teachers import load_causal_lm, right_pad
from palingenesis.rl.losses import target_logprobs


class ReferencePolicy:
    def __init__(self, path: str, device: str, pad_id: int, micro_tokens: int, temperature: float = 1.0):
        self.path = path
        self.inv_temperature = 1.0 / temperature  # scored like the policy: logits / rollout.temperature
        self.device = device
        self.pad_id = pad_id
        self.micro_tokens = micro_tokens
        self.model = load_causal_lm(path, torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.head = output_head(self.model)

    @torch.no_grad()
    def logprobs(self, trajectories: list[Any]) -> dict[int, torch.Tensor]:
        """{id(trajectory): log π_ref of its trained (sampled) tokens, on the device}."""
        out: dict[int, torch.Tensor] = {}
        ordered = sorted(trajectories, key=lambda t: len(t.prompt_ids) + len(t.tokens), reverse=True)
        i = 0
        while i < len(ordered):
            width = len(ordered[i].prompt_ids) + len(ordered[i].tokens) - 1
            rows = max(1, min(len(ordered) - i, self.micro_tokens // max(width, 1)))
            micro = ordered[i : i + rows]
            i += rows
            ids, _ = right_pad([t.prompt_ids + t.tokens[:-1] for t in micro], self.pad_id, self.device)
            positions = torch.zeros(ids.shape, dtype=torch.bool)
            targets, counts = [], []
            for row, t in enumerate(micro):
                sampled = [j for j, trained in enumerate(t.mask) if trained]
                positions[row, [len(t.prompt_ids) + j - 1 for j in sampled]] = True
                targets += [t.tokens[j] for j in sampled]
                counts.append(len(sampled))
            with torch.autocast(
                self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ):
                hidden = final_hidden_states(self.model, ids, None)[positions.to(self.device)]
            lp, _ = target_logprobs(
                hidden, self.head, torch.tensor(targets, device=self.device), scale=self.inv_temperature
            )
            for t, piece in zip(micro, torch.split(lp.detach(), counts)):
                out[id(t)] = piece
        return out

    @torch.no_grad()
    def reset_to(self, model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None) -> None:
        """Make the reference the current policy (ProRL's reference reset). `state`: the policy's state when it is
        not `model.state_dict()` as it stands (adapters folded into their base weights)."""
        state = model.state_dict() if state is None else state
        state = {k: v.detach().to(torch.bfloat16) for k, v in state.items()}
        result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:  # a partial reset would leave a stale anchor, silently
            raise RuntimeError(
                f"reference reset: parameter names differ from the policy's "
                f"(missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]})"
            )


class AdapterFreeReference(ReferencePolicy):
    """The starting policy of a run with LoRA adapters: `model` (the policy being trained) with its adapters
    switched off (`adapters.disabled()`, palingenesis.rl.adapters), in evaluation mode. It scores with the
    trainer's own frozen base weights, so it costs no memory; it cannot be reset to a later policy (the trainer
    keeps a frozen copy for loss.kl_ref_reset)."""

    def __init__(
        self,
        model: torch.nn.Module,
        adapters: Any,
        device: str,
        pad_id: int,
        micro_tokens: int,
        temperature: float = 1.0,
    ):
        self.path = "the policy's frozen base (adapters off)"
        self.inv_temperature = 1.0 / temperature
        self.device = device
        self.pad_id = pad_id
        self.micro_tokens = micro_tokens
        self.model = model
        self.adapters = adapters
        self.head = output_head(model)

    @torch.no_grad()
    def logprobs(self, trajectories: list[Any]) -> dict[int, torch.Tensor]:
        training = self.model.training
        self.model.eval()
        try:
            with self.adapters.disabled():
                return super().logprobs(trajectories)
        finally:
            self.model.train(training)

    def reset_to(self, model: torch.nn.Module, state: dict[str, torch.Tensor] | None = None) -> None:
        raise RuntimeError("the adapter-free reference is the starting policy: it has no weights of its own to reset")


def kl_k3(policy_lp: torch.Tensor, reference_lp: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimate of KL(π || π_ref) on tokens sampled from π: exp(r) - r - 1, r = log π_ref - log π (>= 0)."""
    r = reference_lp - policy_lp
    return torch.exp(r) - r - 1.0
