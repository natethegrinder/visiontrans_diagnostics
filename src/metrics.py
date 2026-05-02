from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compute_multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(dtype=labels.dtype)

    y_true = _tensor_to_numpy(labels)
    y_prob = _tensor_to_numpy(probabilities)
    y_pred = _tensor_to_numpy(predictions)

    metrics: dict[str, float] = {}
    per_label_auc: list[float] = []

    for index, label_name in enumerate(label_names):
        true_column = y_true[:, index]
        prob_column = y_prob[:, index]
        pred_column = y_pred[:, index]

        if np.unique(true_column).size < 2:
            auc_value = float("nan")
        else:
            auc_value = float(roc_auc_score(true_column, prob_column))
        per_label_auc.append(auc_value)
        metrics[f"auc_{label_name}"] = auc_value

        metrics[f"f1_{label_name}"] = float(
            f1_score(true_column, pred_column, zero_division=0)
        )
        metrics[f"precision_{label_name}"] = float(
            precision_score(true_column, pred_column, zero_division=0)
        )
        metrics[f"recall_{label_name}"] = float(
            recall_score(true_column, pred_column, zero_division=0)
        )
        metrics[f"binary_accuracy_{label_name}"] = float(np.mean(true_column == pred_column))

    metrics["mean_auc"] = float(np.nanmean(per_label_auc)) if per_label_auc else float("nan")
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["exact_match_accuracy"] = float(np.all(y_true == y_pred, axis=1).mean())
    per_label_binary_accuracy = [metrics[f"binary_accuracy_{label_name}"] for label_name in label_names]
    metrics["mean_binary_accuracy"] = float(np.mean(per_label_binary_accuracy))
    return metrics
