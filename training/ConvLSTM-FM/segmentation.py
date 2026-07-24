"""Training and evaluation helpers for labeled three-class segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_step(model, tokens, labels, optimizer=None):
    logits = model(tokens)
    loss = F.cross_entropy(logits, labels.long())
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, logits


@torch.no_grad()
def evaluate_segmentation(model, loader, device="cpu"):
    """Return mean cross entropy and a row-normalized confusion matrix."""
    model.eval()
    loss_sum, example_count = 0.0, 0
    predictions, targets = [], []
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        loss, logits = segmentation_step(model, tokens, labels)
        loss_sum += loss.item() * tokens.shape[0]
        example_count += tokens.shape[0]
        predictions.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    if not example_count:
        raise ValueError("evaluation loader is empty")
    confusion = row_normalized_confusion_matrix(torch.cat(predictions), torch.cat(targets))
    return loss_sum / example_count, confusion


def row_normalized_confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, classes: int = 3) -> torch.Tensor:
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    matrix = torch.zeros(classes, classes, dtype=torch.float64)
    for actual, predicted in zip(target.cpu(), prediction.cpu()):
        matrix[int(actual), int(predicted)] += 1
    return matrix / matrix.sum(1, keepdim=True).clamp_min(1)


def signal_noise_labels(labels: torch.Tensor, noise_class: int = 0) -> torch.Tensor:
    """Map labeled multiclass masks to 0=noise and 1=signal."""
    return labels.ne(noise_class).long()
