"""Validation evaluation: compute loss on a held-out dataset.

Supports efficient evaluation without storing full computation graph.
Used by autopilot for:
  - LR sweep scoring
  - Early stopping decisions
  - Data ablation comparisons
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from palingenesis.loss import shift_labels

IGNORE_INDEX = -100


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Evaluate model on validation data.

    Returns dict with:
      - val_loss: average cross-entropy per valid token
      - val_perplexity: exp(val_loss)
      - val_tokens: total tokens evaluated
      - val_accuracy: top-1 token prediction accuracy

    Args:
        model: The model to evaluate
        dataloader: Validation DataLoader
        device: Target device
        max_batches: Maximum batches to evaluate (for speed)
        dtype: Compute dtype for autocast
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # Shift for next-token prediction: logits[t] predicts input_ids[t+1]
        labels = shift_labels(batch["labels"])

        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Compute per-token CE loss
        valid_mask = labels != IGNORE_INDEX
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            continue

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        total_loss += loss.item()
        total_tokens += num_valid

        # Top-1 accuracy on valid tokens
        predictions = logits.argmax(dim=-1)
        correct = ((predictions == labels) & valid_mask).sum().item()
        total_correct += correct

    model.train()

    if total_tokens == 0:
        return {"val_loss": float("inf"), "val_perplexity": float("inf"), "val_tokens": 0, "val_accuracy": 0.0}

    avg_loss = total_loss / total_tokens
    return {
        "val_loss": avg_loss,
        "val_perplexity": min(torch.exp(torch.tensor(avg_loss)).item(), 1e6),
        "val_tokens": total_tokens,
        "val_accuracy": total_correct / total_tokens,
    }
