from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _nanmean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def _build_threshold_tensor(
    threshold: float | Sequence[float] | torch.Tensor,
    num_labels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.detach().to(device=device, dtype=dtype).flatten()
    elif np.isscalar(threshold):
        threshold_tensor = torch.full((num_labels,), float(threshold), device=device, dtype=dtype)
    else:
        threshold_tensor = torch.as_tensor(list(threshold), device=device, dtype=dtype).flatten()

    if threshold_tensor.numel() != num_labels:
        raise ValueError(
            f"Threshold vector must have one value per label: expected {num_labels}, "
            f"got {threshold_tensor.numel()}."
        )
    return threshold_tensor.view(1, num_labels)


def tune_multilabel_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    thresholds: Sequence[float] | None = None,
    objective: str = "f1",
) -> dict[str, object]:
    if objective != "f1":
        raise ValueError(f"Unsupported threshold tuning objective '{objective}'.")

    threshold_candidates = list(thresholds) if thresholds is not None else np.linspace(0.05, 0.95, 19).tolist()
    probabilities = torch.sigmoid(logits.detach().cpu())
    labels_cpu = labels.detach().cpu()

    y_true = _tensor_to_numpy(labels_cpu)
    y_prob = _tensor_to_numpy(probabilities)

    thresholds_by_label: dict[str, float] = {}
    best_f1_by_label: dict[str, float] = {}
    tuned_thresholds: list[float] = []

    for index, label_name in enumerate(label_names):
        true_column = y_true[:, index]
        prob_column = y_prob[:, index]
        best_threshold = float(threshold_candidates[0])
        best_score = float("-inf")

        for candidate in threshold_candidates:
            pred_column = (prob_column >= float(candidate)).astype(np.float32)
            score = float(f1_score(true_column, pred_column, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(candidate)

        thresholds_by_label[label_name] = best_threshold
        best_f1_by_label[label_name] = best_score
        tuned_thresholds.append(best_threshold)

    threshold_tensor = torch.tensor(tuned_thresholds, dtype=torch.float32)
    return {
        "thresholds_by_label": thresholds_by_label,
        "threshold_tensor": threshold_tensor,
        "best_f1_by_label": best_f1_by_label,
    }


def compute_multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    threshold: float | Sequence[float] | torch.Tensor = 0.5,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    threshold_tensor = _build_threshold_tensor(
        threshold=threshold,
        num_labels=len(label_names),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    predictions = (probabilities >= threshold_tensor).to(dtype=labels.dtype)

    y_true = _tensor_to_numpy(labels)
    y_prob = _tensor_to_numpy(probabilities)
    y_pred = _tensor_to_numpy(predictions)

    metrics: dict[str, float] = {}
    per_label_auc: list[float] = []
    per_label_average_precision: list[float] = []

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

        if np.unique(true_column).size < 2:
            average_precision_value = float("nan")
        else:
            average_precision_value = float(average_precision_score(true_column, prob_column))
        per_label_average_precision.append(average_precision_value)
        metrics[f"average_precision_{label_name}"] = average_precision_value

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

    metrics["mean_auc"] = _nanmean_or_nan(per_label_auc)
    metrics["mean_average_precision"] = _nanmean_or_nan(per_label_average_precision)
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["micro_precision"] = float(
        precision_score(y_true, y_pred, average="micro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["micro_recall"] = float(
        recall_score(y_true, y_pred, average="micro", zero_division=0)
    )
    metrics["exact_match_accuracy"] = float(np.all(y_true == y_pred, axis=1).mean())
    per_label_binary_accuracy = [metrics[f"binary_accuracy_{label_name}"] for label_name in label_names]
    metrics["mean_binary_accuracy"] = float(np.mean(per_label_binary_accuracy))
    return metrics
