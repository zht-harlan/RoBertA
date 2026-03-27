from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_macro: float


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> ClassificationMetrics:
    predictions = logits.argmax(dim=-1).detach().cpu()
    labels = labels.detach().cpu()
    accuracy = float((predictions == labels).float().mean().item())
    f1_macro = float(f1_score(labels.numpy(), predictions.numpy(), average="macro"))
    return ClassificationMetrics(accuracy=accuracy, f1_macro=f1_macro)
