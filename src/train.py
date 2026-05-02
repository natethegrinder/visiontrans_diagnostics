from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from data import build_nih_data_module
from evaluate import evaluate_epoch
from losses import build_loss_function
from mlflow_utils import configure_mlflow, log_epoch_metrics, log_label_statistics, log_params_flat
from models import build_model
from metrics import compute_multilabel_metrics


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text()) or {}
    extends = config.get("extends")
    if extends:
        base_config = load_config(config_path.parent / extends)
        return merge_dicts(base_config, {key: value for key, value in config.items() if key != "extends"})
    return config


def merge_dicts(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_device(config: dict) -> torch.device:
    requested = config.get("project", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    training_config = config.get("training", {})
    optimizer_name = training_config.get("optimizer", "adamw").lower()
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
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_config = config.get("training", {})
    scheduler_name = training_config.get("scheduler")
    if not scheduler_name or scheduler_name == "none":
        return None

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        epochs = int(training_config.get("epochs", 30))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        step_size = int(training_config.get("step_size", 10))
        gamma = float(training_config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


def _peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_names: list[str],
    threshold: float = 0.5,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    epoch_logits: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        epoch_logits.append(logits.detach().cpu())
        epoch_labels.append(labels.detach().cpu())

    stacked_logits = torch.cat(epoch_logits, dim=0)
    stacked_labels = torch.cat(epoch_labels, dim=0)
    metrics = compute_multilabel_metrics(stacked_logits, stacked_labels, label_names, threshold=threshold)
    metrics["loss"] = running_loss / max(sample_count, 1)
    return metrics


def build_run_params(config: dict, pos_weight_stats: dict) -> dict[str, object]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    augmentation_config = data_config.get("augmentation", {})
    mean, std = data_config.get("normalize_mean"), data_config.get("normalize_std")
    if mean is None or std is None:
        from data import default_normalization

        mean, std = default_normalization(int(data_config.get("num_channels", 1)))

    return {
        "model_name": model_config.get("architecture", "vit"),
        "image_size": data_config.get("image_size", 224),
        "patch_size": model_config.get("patch_size", 16),
        "in_channels": data_config.get("num_channels", 1),
        "embed_dim": model_config.get("hidden_dim", 192),
        "nhead": model_config.get("num_heads", 6),
        "num_layers": model_config.get("num_layers", 6),
        "dim_feedforward": model_config.get("dim_feedforward", model_config.get("mlp_dim", 768)),
        "dropout": model_config.get("dropout", 0.1),
        "activation": "gelu",
        "norm_first": model_config.get("norm_first", True),
        "optimizer_name": training_config.get("optimizer", "adamw"),
        "learning_rate": training_config.get("learning_rate", 1e-4),
        "weight_decay": training_config.get("weight_decay", 1e-4),
        "batch_size": training_config.get("batch_size", data_config.get("batch_size", 32)),
        "epochs": training_config.get("epochs", 30),
        "loss_function_name": training_config.get("loss", "bce_with_logits"),
        "use_pos_weight": training_config.get("use_pos_weight", True),
        "pos_weight_clamp": data_config.get("pos_weight_clamp", 50),
        "threshold": training_config.get("threshold", 0.5),
        "seed": config.get("project", {}).get("seed", 42),
        "augmentation_enabled": augmentation_config.get("enabled", False),
        "horizontal_flip_prob": augmentation_config.get("horizontal_flip_prob", 0.0),
        "rotation_degrees": augmentation_config.get("rotation_degrees", 0.0),
        "crop_type": augmentation_config.get("crop_type", "none"),
        "crop_scale": augmentation_config.get("crop_scale"),
        "normalize_mean": mean,
        "normalize_std": std,
    }


def save_checkpoint(state: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def train(config: dict) -> dict[str, float]:
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config)
    data_module = build_nih_data_module(config)
    label_names = list(data_module["labels"])

    model = build_model(config).to(device)
    pos_weight_stats = data_module["pos_weight_stats"]
    use_pos_weight = bool(config.get("training", {}).get("use_pos_weight", True))
    pos_weight_tensor = pos_weight_stats["pos_weight_tensor"].to(device) if use_pos_weight else None
    loss_fn = build_loss_function(config, pos_weight=pos_weight_tensor)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    threshold = float(config.get("training", {}).get("threshold", 0.5))
    gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm")
    gradient_clip_norm = float(gradient_clip_norm) if gradient_clip_norm is not None else None
    epochs = int(config.get("training", {}).get("epochs", 30))

    configure_mlflow(config)
    run_name = config.get("run", {}).get("name")
    best_auc = float("-inf")
    best_macro_f1 = float("-inf")
    summary_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name=run_name):
        log_params_flat(build_run_params(config, pos_weight_stats))
        log_label_statistics(pos_weight_stats)
        mlflow.log_dict(config, "config_resolved.json")

        for epoch in range(epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                data_loader=data_module["dataloaders"]["train"],
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                label_names=label_names,
                threshold=threshold,
                gradient_clip_norm=gradient_clip_norm,
            )
            val_metrics = evaluate_epoch(
                model=model,
                data_loader=data_module["dataloaders"]["val"],
                loss_fn=loss_fn,
                device=device,
                label_names=label_names,
                threshold=threshold,
            )

            if scheduler is not None:
                scheduler.step()

            epoch_time_sec = time.perf_counter() - epoch_start
            learning_rate = float(optimizer.param_groups[0]["lr"])
            peak_gpu_memory_mb = _peak_gpu_memory_mb(device)

            train_metrics["learning_rate"] = learning_rate
            train_metrics["epoch_time_sec"] = epoch_time_sec
            train_metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            val_metrics["learning_rate"] = learning_rate
            val_metrics["epoch_time_sec"] = epoch_time_sec
            val_metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb

            log_epoch_metrics(train_metrics, split="train", epoch=epoch)
            log_epoch_metrics(val_metrics, split="val", epoch=epoch)

            checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "artifacts/models")
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }

            if val_metrics["mean_auc"] > best_auc:
                best_auc = val_metrics["mean_auc"]
                save_checkpoint(checkpoint_state, Path(checkpoint_dir) / "vit_best_auc.pt")
                mlflow.log_metric("best_val_mean_auc", best_auc)
                mlflow.log_metric("best_val_mean_auc_epoch", epoch)

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                save_checkpoint(checkpoint_state, Path(checkpoint_dir) / "vit_best_macro_f1.pt")
                mlflow.log_metric("best_val_macro_f1", best_macro_f1)
                mlflow.log_metric("best_val_macro_f1_epoch", epoch)

            summary_metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mean_auc": train_metrics["mean_auc"],
                "val_mean_auc": val_metrics["mean_auc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }

    return summary_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT/CNN baselines on NIH Chest X-ray manifests.")
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    summary = train(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
