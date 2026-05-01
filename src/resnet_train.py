from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow

try:
    from .data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
    from .evaluate import evaluate_model, load_config, resolve_device, resolve_threshold
    from .mlflow_utils import configure_mlflow, log_checkpoint_artifact, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
    from .models.resnet import build_resnet_model
except ImportError:
    from data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
    from evaluate import evaluate_model, load_config, resolve_device, resolve_threshold
    from mlflow_utils import configure_mlflow, log_checkpoint_artifact, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
    from models.resnet import build_resnet_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    return float((predictions == targets).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
    )

    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_accuracy += compute_multilabel_accuracy(logits.detach(), targets, threshold)
        num_batches += 1
        progress_bar.set_postfix(
            loss=f"{total_loss / num_batches:.4f}",
            acc=f"{total_accuracy / num_batches:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {
        "loss": total_loss / max(num_batches, 1),
        "label_accuracy": total_accuracy / max(num_batches, 1),
    }


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_config = config.get("training", {})
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
    learning_rate = float(training_config.get("learning_rate", 1e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(training_config.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_config = config.get("training", {})
    scheduler_name = str(training_config.get("scheduler", "none")).lower()

    if scheduler_name in {"none", "", "null"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    if scheduler_name == "step":
        step_size = int(training_config.get("step_size", 10))
        gamma = float(training_config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler '{scheduler_name}'")


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "labels": NIH_CHEST_XRAY_LABELS,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ResNet on the NIH Chest X-ray dataset.")
    parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Where to save the best validation checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold used for reporting multilabel accuracy.",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default=None,
        help="Optional loss override for this run.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None, help="Optional focal loss gamma override.")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Optional focal loss alpha override.")
    return parser


def _strip_loss_suffix(run_name: str) -> str:
    for suffix in ("_bce", "_focal"):
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)]
    return run_name


def apply_loss_overrides(
    config: dict[str, Any],
    loss_override: str | None,
    focal_gamma_override: float | None,
    focal_alpha_override: float | None,
) -> None:
    training_config = config.setdefault("training", {})
    focal_config = training_config.setdefault("focal_loss", {})

    if loss_override is not None:
        training_config["loss"] = loss_override
        run_config = config.setdefault("run", {})
        base_run_name = _strip_loss_suffix(str(run_config.get("name", "resnet_best")))
        run_config["name"] = f"{base_run_name}_{loss_override}"

    if focal_gamma_override is not None:
        focal_config["gamma"] = focal_gamma_override

    if focal_alpha_override is not None:
        focal_config["alpha"] = focal_alpha_override


def resolve_checkpoint_path(config: dict[str, Any], checkpoint_override: str | None) -> Path:
    if checkpoint_override:
        return resolve_nonconflicting_path(Path(checkpoint_override))

    run_name = str(config.get("run", {}).get("name", "resnet_best"))
    return resolve_nonconflicting_path(Path("outputs/checkpoints") / f"{run_name}.pt")


def compute_pos_weight_from_frame(
    frame: DataFrame,
    min_positive_count: float = 1.0,
    max_pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_samples = float(len(frame))
    if total_samples <= 0:
        raise ValueError("Training manifest is empty; cannot compute class imbalance weights.")

    class_stats: dict[str, float] = {}
    weights: list[float] = []
    for label in NIH_CHEST_XRAY_LABELS:
        positive_count = float(frame[label].sum())
        adjusted_positive_count = max(positive_count, min_positive_count)
        negative_count = max(total_samples - positive_count, 0.0)
        pos_weight = negative_count / adjusted_positive_count
        if max_pos_weight is not None:
            pos_weight = min(pos_weight, max_pos_weight)
        weights.append(pos_weight)
        class_stats[label] = positive_count

    return torch.tensor(weights, dtype=torch.float32), class_stats


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_factor

        loss = focal_weight * bce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_criterion(config: dict[str, Any], train_frame: DataFrame, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    training_config = config.get("training", {})
    loss_name = str(training_config.get("loss", "bce_with_logits")).lower()
    imbalance_config = training_config.get("imbalance", {})
    enabled = bool(imbalance_config.get("enabled", False))
    strategy = str(imbalance_config.get("strategy", "none")).lower()

    criterion_summary: dict[str, Any] = {
        "loss_name": loss_name,
        "imbalance_strategy": "none",
    }

    pos_weight: torch.Tensor | None = None

    if enabled and strategy not in {"none", "", "null"}:
        if strategy != "pos_weight":
            raise ValueError(f"Unsupported imbalance strategy '{strategy}'")

        min_positive_count = float(imbalance_config.get("min_positive_count", 1.0))
        max_pos_weight = imbalance_config.get("max_pos_weight")
        max_pos_weight_value = float(max_pos_weight) if max_pos_weight is not None else None
        pos_weight, class_stats = compute_pos_weight_from_frame(
            train_frame,
            min_positive_count=min_positive_count,
            max_pos_weight=max_pos_weight_value,
        )
        pos_weight = pos_weight.to(device)
        criterion_summary.update(
            {
                "imbalance_strategy": strategy,
                "min_positive_count": min_positive_count,
                "max_pos_weight": max_pos_weight_value,
                "pos_weight": {label: float(weight) for label, weight in zip(NIH_CHEST_XRAY_LABELS, pos_weight.tolist())},
                "positive_count": class_stats,
            }
        )

    if loss_name in {"bce", "bce_with_logits", "bcewithlogitsloss"}:
        criterion_summary["loss_name"] = "bce"
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), criterion_summary

    if loss_name in {"focal", "focal_loss", "focalloss"}:
        focal_config = training_config.get("focal_loss", {})
        gamma = float(focal_config.get("gamma", 2.0))
        alpha = focal_config.get("alpha")
        alpha_value = float(alpha) if alpha is not None else None
        criterion_summary.update(
            {
                "loss_name": "focal",
                "focal_gamma": gamma,
                "focal_alpha": alpha_value,
            }
        )
        return BinaryFocalLoss(gamma=gamma, alpha=alpha_value, pos_weight=pos_weight), criterion_summary

    raise ValueError(f"Unsupported loss '{loss_name}'")


def get_gpu_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_peak_memory_allocated_mb": 0.0,
            "gpu_peak_memory_reserved_mb": 0.0,
        }

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    bytes_per_mb = 1024.0 * 1024.0
    return {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device_index) / bytes_per_mb),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device_index) / bytes_per_mb),
        "gpu_peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device_index) / bytes_per_mb),
        "gpu_peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device_index) / bytes_per_mb),
    }


def reset_gpu_peak_memory_stats(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    apply_loss_overrides(config, args.loss, args.focal_gamma, args.focal_alpha)

    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config, args.device)
    threshold = resolve_threshold(config, args.threshold)

    data_module = build_nih_data_module(config)
    dataloaders = cast(dict[str, DataLoader], data_module["dataloaders"])
    train_frame = cast(Any, dataloaders["train"].dataset).frame

    model = build_resnet_model(config).to(device)
    criterion, criterion_summary = build_criterion(config, train_frame, device)
    optimizer = build_optimizer(model, config)
    total_epochs = int(config.get("training", {}).get("epochs", 30))
    scheduler = build_scheduler(optimizer, config, total_epochs)
    patience = int(config.get("training", {}).get("early_stopping_patience", 5))

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    mlflow_config = configure_mlflow(config)
    run_name = str(config.get("run", {}).get("name", checkpoint_path.stem))
    log_artifacts = bool(mlflow_config.get("log_artifacts", True))
    run_start_time = time.perf_counter()

    print(json.dumps({"criterion": criterion_summary}))

    with mlflow.start_run(run_name=run_name):
        log_config_params(config)
        mlflow.log_param("device", str(device))
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        log_dict_artifact("criterion_summary", criterion_summary)

        for epoch in range(1, total_epochs + 1):
            reset_gpu_peak_memory_stats(device)
            epoch_start_time = time.perf_counter()
            train_start_time = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                threshold,
                epoch,
                total_epochs,
            )
            train_duration_sec = time.perf_counter() - train_start_time

            val_start_time = time.perf_counter()
            val_metrics = evaluate_model(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                threshold=threshold,
            )
            val_duration_sec = time.perf_counter() - val_start_time
            epoch_duration_sec = time.perf_counter() - epoch_start_time

            if scheduler is not None:
                scheduler.step()

            epoch_summary: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["label_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
                "val_f1_score": val_metrics["f1_score"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_epoch_time_sec": train_duration_sec,
                "val_epoch_time_sec": val_duration_sec,
                "epoch_time_sec": epoch_duration_sec,
            }
            epoch_summary.update(get_gpu_memory_metrics(device))
            history.append(epoch_summary)
            print(json.dumps(epoch_summary))
            log_metrics({key: value for key, value in epoch_summary.items() if key != "epoch"}, step=epoch)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(checkpoint_path, model, optimizer, epoch, config, best_val_loss)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                mlflow.log_metric("best_epoch", best_epoch, step=epoch)
                log_checkpoint_artifact(checkpoint_path, enabled=log_artifacts)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                early_stop_summary = {
                    "status": "early_stop",
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                }
                print(json.dumps(early_stop_summary))
                log_dict_artifact("early_stop_summary", early_stop_summary)
                break

        completion_summary = {
            "status": "completed",
            "checkpoint": str(checkpoint_path.resolve()),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_epochs_run": len(history),
            "device": str(device),
            "threshold": threshold,
            "imbalance_strategy": criterion_summary.get("imbalance_strategy", "none"),
            "loss_name": criterion_summary.get("loss_name", "bce_with_logits"),
            "total_run_time_sec": time.perf_counter() - run_start_time,
        }
        print(json.dumps(completion_summary))
        log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "num_epochs_run": len(history),
                "total_run_time_sec": float(completion_summary["total_run_time_sec"]),
            }
        )
        log_dict_artifact("run_summary", completion_summary)


if __name__ == "__main__":
    main()