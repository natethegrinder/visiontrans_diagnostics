from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import load_config
from data import build_dataloaders
from models import build_model
from train import build_criterion

# Order matches NIH_CHEST_XRAY_LABELS in data.py
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]


def binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC using average ranks, including tied scores."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    num_pos = int(y_true.sum())
    num_neg = int((~y_true).sum())
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = ranks[y_true].sum()
    return float((pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def compute_mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Args:
        y_true: (N, 14) binary ground-truth labels
        y_pred: (N, 14) predicted probabilities (post-sigmoid)
    Returns:
        dict: {label: auc, ..., 'mean': mean_auc}
        Classes with no positive samples get nan and are excluded from the mean.
    """
    results = {}
    aucs = []
    for i, label in enumerate(DISEASE_LABELS):
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == y_true.shape[0]:
            results[label] = float('nan')
            continue
        auc = binary_roc_auc_score(y_true[:, i], y_pred[:, i])
        results[label] = auc
        aucs.append(auc)
    results['mean'] = float(np.mean(aucs)) if aucs else float('nan')
    return results


def _safe_metric(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _compute_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    scores = []
    for index, label in enumerate(DISEASE_LABELS):
        target = y_true[:, index]
        if np.unique(target).size < 2:
            results[label] = None
            continue
        score = average_precision_score_np(target, y_pred[:, index])
        results[label] = score
        scores.append(score)
    results["mean"] = float(np.mean(scores)) if scores else None
    return results


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.float32)
    y_score = np.asarray(y_score)
    num_pos = float(y_true.sum())
    if num_pos <= 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp_cumsum = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1, dtype=np.float32)
    precision_at_k = tp_cumsum / ranks
    return float((precision_at_k * sorted_true).sum() / num_pos)


def binary_counts(y_true: np.ndarray, y_binary: np.ndarray) -> tuple[float, float, float]:
    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_binary).astype(bool)
    tp = float(np.logical_and(true, pred).sum())
    fp = float(np.logical_and(~true, pred).sum())
    fn = float(np.logical_and(true, ~pred).sum())
    return tp, fp, fn


def precision_from_counts(tp: float, fp: float) -> float:
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_from_counts(tp: float, fn: float) -> float:
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def per_class_binary_metrics(y_true: np.ndarray, y_binary: np.ndarray) -> dict[str, dict[str, float]]:
    metrics = {"f1": {}, "precision": {}, "recall": {}}
    for index, label in enumerate(DISEASE_LABELS):
        tp, fp, fn = binary_counts(y_true[:, index], y_binary[:, index])
        precision = precision_from_counts(tp, fp)
        recall = recall_from_counts(tp, fn)
        metrics["precision"][label] = precision
        metrics["recall"][label] = recall
        metrics["f1"][label] = f1_from_precision_recall(precision, recall)
    return metrics


def compute_confusion_matrices(y_true: np.ndarray, y_binary: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, list[list[int]]] = {}
    aggregate = np.zeros((2, 2), dtype=np.int64)
    for index, label in enumerate(DISEASE_LABELS):
        true = y_true[:, index].astype(bool)
        pred = y_binary[:, index].astype(bool)
        tn = int(np.logical_and(~true, ~pred).sum())
        fp = int(np.logical_and(~true, pred).sum())
        fn = int(np.logical_and(true, ~pred).sum())
        tp = int(np.logical_and(true, pred).sum())
        matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        aggregate += matrix
        per_class[label] = matrix.tolist()
    return {
        "aggregate": aggregate.tolist(),
        "per_class": per_class,
    }


def micro_metrics(y_true: np.ndarray, y_binary: np.ndarray) -> dict[str, float]:
    tp, fp, fn = binary_counts(y_true, y_binary)
    precision = precision_from_counts(tp, fp)
    recall = recall_from_counts(tp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_from_precision_recall(precision, recall),
    }


def samples_f1_score(y_true: np.ndarray, y_binary: np.ndarray) -> float:
    scores = []
    for row_true, row_pred in zip(y_true, y_binary):
        tp, fp, fn = binary_counts(row_true, row_pred)
        precision = precision_from_counts(tp, fp)
        recall = recall_from_counts(tp, fn)
        scores.append(f1_from_precision_recall(precision, recall))
    return float(np.mean(scores)) if scores else 0.0


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_binary = (y_pred >= threshold).astype(np.float32)
    auroc = compute_mean_auc(y_true, y_pred)
    pr_auc = _compute_pr_auc(y_true, y_pred)
    per_class = per_class_binary_metrics(y_true, y_binary)
    confusion_matrices = compute_confusion_matrices(y_true, y_binary)
    micro = micro_metrics(y_true, y_binary)
    macro_f1 = float(np.mean(list(per_class["f1"].values())))
    macro_precision = float(np.mean(list(per_class["precision"].values())))
    macro_recall = float(np.mean(list(per_class["recall"].values())))
    return {
        "threshold": threshold,
        "num_examples": int(y_true.shape[0]),
        "mean_auroc": _safe_metric(auroc["mean"]),
        "per_class_auroc": {label: _safe_metric(auroc[label]) for label in DISEASE_LABELS},
        "mean_pr_auc": _safe_metric(pr_auc["mean"]),
        "per_class_pr_auc": {label: pr_auc[label] for label in DISEASE_LABELS},
        "macro_f1": macro_f1,
        "micro_f1": micro["f1"],
        "samples_f1": samples_f1_score(y_true, y_binary),
        "macro_precision": macro_precision,
        "micro_precision": micro["precision"],
        "macro_recall": macro_recall,
        "micro_recall": micro["recall"],
        "label_accuracy": float((y_binary == y_true).mean()),
        "exact_match_accuracy": float(np.all(y_binary == y_true, axis=1).mean()),
        "confusion_matrix": confusion_matrices["per_class"],
        "aggregate_confusion_matrix": confusion_matrices["aggregate"],
        "per_class_f1": per_class["f1"],
        "per_class_precision": per_class["precision"],
        "per_class_recall": per_class["recall"],
        "target_prevalence": {
            label: float(y_true[:, index].mean()) for index, label in enumerate(DISEASE_LABELS)
        },
        "predicted_prevalence": {
            label: float(y_binary[:, index].mean()) for index, label in enumerate(DISEASE_LABELS)
        },
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    all_preds = []
    all_labels = []
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        raise ValueError("Evaluation dataloader produced no batches.")

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_pred, threshold=threshold)
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


def save_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_per_class_csv(metrics: dict[str, Any], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in DISEASE_LABELS:
        rows.append(
            {
                "label": label,
                "auroc": metrics["per_class_auroc"].get(label),
                "pr_auc": metrics["per_class_pr_auc"].get(label),
                "f1": metrics["per_class_f1"].get(label),
                "precision": metrics["per_class_precision"].get(label),
                "recall": metrics["per_class_recall"].get(label),
                "target_prevalence": metrics["target_prevalence"].get(label),
                "predicted_prevalence": metrics["predicted_prevalence"].get(label),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return {}
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def print_auc_table(results: dict[str, float]) -> None:
    print(f"\n  {'Label':<22}  {'AUC':>6}")
    print(f"  {'-'*22}  {'-'*6}")
    for label in DISEASE_LABELS:
        val = results.get(label, float('nan'))
        marker = '  ← ' if val < 0.7 else ''
        print(f"  {label:<22}  {val:.4f}{marker}")
    print(f"  {'='*22}  {'='*6}")
    print(f"  {'Mean AUC':<22}  {results['mean']:.4f}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the config used for the run.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = _resolve_device(args.device)
    threshold = float(args.threshold if args.threshold is not None else config.get("evaluation", {}).get("threshold", 0.5))
    dataloaders = build_dataloaders(config)
    if args.split not in dataloaders:
        raise ValueError(f"Split '{args.split}' is not available.")

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(Path(args.checkpoint), model, device)
    train_frame = dataloaders["train"].dataset.frame
    criterion, _ = build_criterion(config, train_frame, device)
    metrics = evaluate_model(
        model,
        dataloaders[args.split],
        criterion,
        device,
        threshold=threshold,
        use_amp=bool(config.get("training", {}).get("mixed_precision", False)),
    )
    metrics["split"] = args.split
    metrics["checkpoint_path"] = str(Path(args.checkpoint))
    if checkpoint:
        metrics["checkpoint_epoch"] = checkpoint.get("epoch")
        metrics["checkpoint_val_mean_auc"] = checkpoint.get("val_mean_auc")

    run_name = config.get("run", {}).get("name", "model")
    metrics_dir = Path(config.get("artifacts", {}).get("metrics_dir", "artifacts/metrics"))
    output_path = Path(args.output) if args.output else metrics_dir / f"{run_name}_{args.split}_metrics.json"
    per_class_path = output_path.with_name(output_path.stem.replace("_metrics", "") + "_per_class_metrics.csv")
    save_metrics_json(metrics, output_path)
    save_per_class_csv(metrics, per_class_path)

    print(f"{args.split} loss: {metrics['loss']:.4f}")
    print(f"{args.split} mean AUROC: {metrics['mean_auroc']:.4f}")
    print(f"{args.split} mean PR-AUC: {metrics['mean_pr_auc']:.4f}")
    print(f"{args.split} macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved metrics: {output_path}")
    print(f"Saved per-class metrics: {per_class_path}")


if __name__ == "__main__":
    main()
