from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from data import build_nih_data_module
from evaluate import evaluate_epoch
from losses import build_loss_function
from mlflow_utils import configure_mlflow, log_epoch_metrics, log_label_statistics, log_params_flat
from metrics import compute_multilabel_metrics, tune_multilabel_thresholds
from models import build_model


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
    requested = str(config.get("project", {}).get("device", "auto")).lower()

    # For experiment runs, prefer CUDA when available. Do not select MPS automatically
    # because prior local Mac testing hit MPS-specific backward stride/view errors.
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "project.device is set to 'cuda', but CUDA is not available. "
                "Check NVIDIA driver, CUDA-enabled PyTorch install, and GPU visibility."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("project.device is set to 'mps', but MPS is not available.")
        return torch.device("mps")

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


def _scheduler_name(config: dict) -> str:
    scheduler_name = config.get("training", {}).get("scheduler")
    if not scheduler_name:
        return "none"
    return str(scheduler_name).lower()


def _scheduler_steps_per_batch(config: dict) -> bool:
    return _scheduler_name(config) == "step_warmup_cosine"


def build_scheduler(
    config: dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_config = config.get("training", {})
    scheduler_name = training_config.get("scheduler")
    if not scheduler_name or scheduler_name == "none":
        return None

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        epochs = int(training_config.get("epochs", 30))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "warmup_cosine":
        epochs = int(training_config.get("epochs", 30))
        warmup_epochs = int(training_config.get("warmup_epochs", 0))

        def lr_lambda(epoch_index: int) -> float:
            if warmup_epochs > 0 and epoch_index < warmup_epochs:
                return float(epoch_index + 1) / float(warmup_epochs)
            progress = float(epoch_index - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_name == "step_warmup_cosine":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for step_warmup_cosine scheduling.")

        epochs = int(training_config.get("epochs", 30))
        warmup_ratio = float(training_config.get("warmup_ratio", 0.05))
        num_training_steps = max(1, int(steps_per_epoch) * epochs)
        if num_training_steps <= 1:
            return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1)

        warmup_steps = max(1, int(num_training_steps * warmup_ratio))
        warmup_steps = min(warmup_steps, num_training_steps - 1)
        cosine_steps = max(1, num_training_steps - warmup_steps)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    if scheduler_name == "step":
        step_size = int(training_config.get("step_size", 10))
        gamma = float(training_config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


def initialize_scheduler_learning_rate(
    config: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    if scheduler is None:
        return

    scheduler_name = _scheduler_name(config)
    if scheduler_name != "warmup_cosine":
        return

    warmup_epochs = int(config.get("training", {}).get("warmup_epochs", 0))
    if warmup_epochs > 0:
        scale = 1.0 / float(warmup_epochs)
    else:
        scale = 1.0

    for param_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        param_group["lr"] = float(base_lr) * scale
    scheduler.last_epoch = 0
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def step_scheduler(
    config: dict,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int | None = None,
    batch_level: bool | None = None,
) -> None:
    if scheduler is None:
        return

    if batch_level is not None and batch_level != _scheduler_steps_per_batch(config):
        return
    if batch_level is None and _scheduler_steps_per_batch(config):
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary and is being deprecated.*",
            category=UserWarning,
        )
        scheduler.step()


def initialize_early_stopping(config: dict) -> dict[str, object] | None:
    training_config = config.get("training", {})
    metric_name = training_config.get("early_stopping_metric")
    patience = training_config.get("early_stopping_patience")
    if metric_name is None or patience is None:
        return None

    return {
        "metric_name": str(metric_name),
        "patience": int(patience),
        "min_delta": float(training_config.get("early_stopping_min_delta", 0.0)),
        "mode": str(training_config.get("early_stopping_mode", "max")).lower(),
        "best_value": None,
        "best_epoch": None,
        "bad_epochs": 0,
        "stopped_epoch": None,
        "should_stop": False,
    }


def resolve_early_stopping_metric_value(
    metric_name: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    tuned_val_metrics: dict[str, float] | None = None,
) -> float:
    if metric_name.startswith("train_"):
        return float(train_metrics[metric_name.removeprefix("train_")])
    if metric_name.startswith("val_tuned_"):
        if tuned_val_metrics is None:
            raise KeyError(f"Requested early stopping metric '{metric_name}' but tuned validation metrics are unavailable.")
        return float(tuned_val_metrics[metric_name.removeprefix("val_tuned_")])
    if metric_name.startswith("val_"):
        return float(val_metrics[metric_name.removeprefix("val_")])
    if metric_name in val_metrics:
        return float(val_metrics[metric_name])
    raise KeyError(f"Unable to resolve early stopping metric '{metric_name}'.")


def update_early_stopping_state(
    state: dict[str, object] | None,
    metric_value: float,
    epoch: int,
) -> dict[str, object] | None:
    if state is None:
        return None

    mode = str(state["mode"])
    min_delta = float(state["min_delta"])
    best_value = state["best_value"]

    is_finite_metric = math.isfinite(metric_value)
    if best_value is None:
        improved = is_finite_metric
    elif not is_finite_metric:
        improved = False
    elif mode == "max":
        improved = metric_value > float(best_value) + min_delta
    elif mode == "min":
        improved = metric_value < float(best_value) - min_delta
    else:
        raise ValueError(f"Unsupported early stopping mode '{mode}'.")

    if improved:
        state["best_value"] = float(metric_value)
        state["best_epoch"] = int(epoch)
        state["bad_epochs"] = 0
        state["should_stop"] = False
        return state

    state["bad_epochs"] = int(state["bad_epochs"]) + 1
    if int(state["bad_epochs"]) >= int(state["patience"]):
        state["should_stop"] = True
        state["stopped_epoch"] = int(epoch)
    return state


def _peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def evaluate_best_checkpoint_on_test_split(
    config: dict,
    data_module: dict[str, object],
    loss_fn: torch.nn.Module,
    device: torch.device,
    label_names: list[str],
    checkpoint_path: str | Path,
    threshold: float | torch.Tensor = 0.5,
    max_batches: int | None = None,
    progress_log_interval: int | None = 50,
    use_amp: bool = False,
) -> dict[str, object]:
    dataloaders = data_module.get("dataloaders", {})
    test_loader = dataloaders.get("test") if isinstance(dataloaders, dict) else None
    if test_loader is None:
        return {
            "skipped": True,
            "reason": "Test dataloader is unavailable for this run.",
            "metrics": None,
        }

    print("[Eval] Running final test evaluation from best val mean AUC checkpoint", flush=True)
    evaluation_model = build_model(config).to(device)
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    evaluation_model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_epoch(
        model=evaluation_model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        label_names=label_names,
        threshold=threshold,
        max_batches=max_batches,
        progress_log_interval=progress_log_interval,
        use_amp=use_amp,
    )
    return {
        "skipped": False,
        "reason": None,
        "metrics": test_metrics,
    }


def train_one_epoch(
    config: dict,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    label_names: list[str],
    threshold: float = 0.5,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    progress_log_interval: int | None = 50,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    epoch_logits: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []
    total_batches = len(data_loader)
    effective_total_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    for batch_index, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if batch_index == 0:
            print("[Train] First train batch loaded", flush=True)
        if progress_log_interval and (
            batch_index == 0
            or (batch_index + 1) % progress_log_interval == 0
            or (batch_index + 1) == effective_total_batches
        ):
            print(f"[Train] Batch {batch_index + 1}/{effective_total_batches}", flush=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda") if scaler is not None else contextlib.nullcontext()
        with amp_context:
            logits = model(images)
            loss = loss_fn(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
        step_scheduler(config, scheduler, batch_level=True)

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


def build_run_params(
    config: dict,
    pos_weight_stats: dict,
    amp_enabled: bool | None = None,
) -> dict[str, object]:
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    augmentation_config = data_config.get("augmentation", {})
    runtime_config = config.get("runtime", {})
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
        "scheduler_name": training_config.get("scheduler", "none"),
        "learning_rate": training_config.get("learning_rate", 1e-4),
        "weight_decay": training_config.get("weight_decay", 1e-4),
        "warmup_epochs": training_config.get("warmup_epochs", 0),
        "warmup_ratio": training_config.get("warmup_ratio"),
        "batch_size": training_config.get("batch_size", data_config.get("batch_size", 32)),
        "epochs": training_config.get("epochs", 30),
        "loss_function_name": training_config.get("loss", "bce_with_logits"),
        "use_pos_weight": training_config.get("use_pos_weight", True),
        "focal_gamma": training_config.get("focal_gamma"),
        "focal_alpha": training_config.get("focal_alpha"),
        "pos_weight_clamp": data_config.get("pos_weight_clamp", 50),
        "threshold": training_config.get("threshold", 0.5),
        "tune_thresholds": training_config.get("tune_thresholds", False),
        "threshold_tuning_objective": training_config.get("threshold_tuning_objective", "f1"),
        "threshold_grid": training_config.get("threshold_grid"),
        "use_amp": training_config.get("use_amp", False),
        "amp_enabled": amp_enabled if amp_enabled is not None else False,
        "early_stopping_patience": training_config.get("early_stopping_patience"),
        "early_stopping_min_delta": training_config.get("early_stopping_min_delta"),
        "early_stopping_metric": training_config.get("early_stopping_metric"),
        "early_stopping_mode": training_config.get("early_stopping_mode"),
        "seed": config.get("project", {}).get("seed", 42),
        "augmentation_enabled": augmentation_config.get("enabled", False),
        "horizontal_flip_prob": augmentation_config.get("horizontal_flip_prob", 0.0),
        "allow_horizontal_flip": augmentation_config.get("allow_horizontal_flip", False),
        "rotation_degrees": augmentation_config.get("rotation_degrees", 0.0),
        "crop_type": augmentation_config.get("crop_type", "none"),
        "crop_scale": augmentation_config.get("crop_scale"),
        "normalize_mean": mean,
        "normalize_std": std,
        "runtime_max_train_batches": runtime_config.get("max_train_batches"),
        "runtime_max_val_batches": runtime_config.get("max_val_batches"),
    }


def save_checkpoint(state: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def train_model(
    config: dict,
    run_name: str | None = None,
    nested_run: bool = False,
) -> dict[str, object]:
    seed = int(config.get("project", {}).get("seed", 42))
    print(f"[Setup] Seed: {seed}", flush=True)
    set_seed(seed)
    device = resolve_device(config)
    print(f"[Setup] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Setup] CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[Setup] CUDA device count: {torch.cuda.device_count()}", flush=True)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Setup] CUDA total memory: {total_memory_gb:.2f} GB", flush=True)
    print("[Data] Building NIH data module...", flush=True)
    data_module = build_nih_data_module(config)
    print("[Data] Built NIH data module", flush=True)
    label_names = list(data_module["labels"])
    print(f"[Data] Labels: {len(label_names)} -> {label_names}", flush=True)
    print(f"[Data] Train batches: {len(data_module['dataloaders']['train'])}", flush=True)
    print(f"[Data] Val batches: {len(data_module['dataloaders']['val'])}", flush=True)

    print("[Model] Building model...", flush=True)
    model = build_model(config).to(device)
    print(f"[Model] Built model: {model.__class__.__name__}", flush=True)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[Model] Total params: {total_params:,}", flush=True)
    print(f"[Model] Trainable params: {trainable_params:,}", flush=True)
    pos_weight_stats = data_module["pos_weight_stats"]
    use_pos_weight = bool(config.get("training", {}).get("use_pos_weight", True))
    pos_weight_tensor = pos_weight_stats["pos_weight_tensor"].to(device) if use_pos_weight else None
    runtime_config = config.get("runtime", {})
    max_train_batches = runtime_config.get("max_train_batches")
    max_val_batches = runtime_config.get("max_val_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    progress_log_interval = runtime_config.get("progress_log_interval", 50)
    progress_log_interval = int(progress_log_interval) if progress_log_interval is not None else None
    train_steps_per_epoch = len(data_module["dataloaders"]["train"])
    if max_train_batches is not None:
        train_steps_per_epoch = min(train_steps_per_epoch, max_train_batches)
    print("[Loss] Building loss function...", flush=True)
    loss_fn = build_loss_function(config, pos_weight=pos_weight_tensor)
    print("[Optim] Building optimizer and scheduler...", flush=True)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch=train_steps_per_epoch)
    initialize_scheduler_learning_rate(config, optimizer, scheduler)
    threshold = float(config.get("training", {}).get("threshold", 0.5))
    gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm")
    gradient_clip_norm = float(gradient_clip_norm) if gradient_clip_norm is not None else None
    epochs = int(config.get("training", {}).get("epochs", 30))
    tune_thresholds = bool(config.get("training", {}).get("tune_thresholds", False))
    threshold_tuning_objective = str(config.get("training", {}).get("threshold_tuning_objective", "f1"))
    threshold_grid = config.get("training", {}).get("threshold_grid")
    use_amp = bool(config.get("training", {}).get("use_amp", False))
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    evaluate_test_after_training = bool(config.get("training", {}).get("evaluate_test_after_training", False))
    early_stopping_state = initialize_early_stopping(config)

    configure_mlflow(config)
    effective_run_name = run_name or config.get("run", {}).get("name")
    print(f"[MLflow] Run name: {effective_run_name}", flush=True)
    best_auc = float("-inf")
    best_macro_f1 = float("-inf")
    summary_metrics: dict[str, float] = {}
    latest_tuned_thresholds: dict[str, float] | None = None
    best_auc_epoch: int | None = None
    best_macro_f1_epoch: int | None = None
    history: list[dict[str, object]] = []
    checkpoint_dir = Path(config.get("training", {}).get("checkpoint_dir", "artifacts/models"))
    best_auc_path = checkpoint_dir / "vit_best_auc.pt"
    best_macro_f1_path = checkpoint_dir / "vit_best_macro_f1.pt"
    final_checkpoint_name = config.get("training", {}).get("final_checkpoint_name")
    final_checkpoint_path = checkpoint_dir / str(final_checkpoint_name) if final_checkpoint_name else None
    stopped_early = False
    stopped_epoch: int | None = None
    final_epoch_index: int | None = None
    test_metrics: dict[str, float] | None = None
    test_evaluation_skipped = not evaluate_test_after_training
    test_evaluation_reason: str | None = (
        "Test evaluation disabled in config." if not evaluate_test_after_training else None
    )

    train_start_time = time.perf_counter()
    with mlflow.start_run(run_name=effective_run_name, nested=nested_run) as run:
        print(f"[MLflow] Started run: {run.info.run_id}", flush=True)
        print("[Train] Starting training loop", flush=True)
        log_params_flat(build_run_params(config, pos_weight_stats, amp_enabled=amp_enabled))
        log_label_statistics(pos_weight_stats)
        mlflow.log_dict(config, "config_resolved.json")

        for epoch in range(epochs):
            print(f"[Train] Epoch {epoch + 1}/{epochs} started", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start = time.perf_counter()
            print(f"[Train] Running train epoch {epoch + 1}/{epochs}", flush=True)
            train_metrics = train_one_epoch(
                config=config,
                model=model,
                data_loader=data_module["dataloaders"]["train"],
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                label_names=label_names,
                threshold=threshold,
                gradient_clip_norm=gradient_clip_norm,
                max_batches=max_train_batches,
                progress_log_interval=progress_log_interval,
                scaler=scaler,
            )
            print(f"[Eval] Running validation epoch {epoch + 1}/{epochs}", flush=True)
            val_metrics = evaluate_epoch(
                model=model,
                data_loader=data_module["dataloaders"]["val"],
                loss_fn=loss_fn,
                device=device,
                label_names=label_names,
                threshold=threshold,
                return_outputs=tune_thresholds,
                max_batches=max_val_batches,
                progress_log_interval=progress_log_interval,
                use_amp=amp_enabled,
            )

            if tune_thresholds:
                val_metrics, val_logits, val_labels = val_metrics
                tuning_result = tune_multilabel_thresholds(
                    logits=val_logits,
                    labels=val_labels,
                    label_names=label_names,
                    thresholds=threshold_grid,
                    objective=threshold_tuning_objective,
                )
                tuned_threshold_tensor = tuning_result["threshold_tensor"]
                latest_tuned_thresholds = dict(tuning_result["thresholds_by_label"])
                tuned_val_metrics = compute_multilabel_metrics(
                    val_logits,
                    val_labels,
                    label_names,
                    threshold=tuned_threshold_tensor,
                )
                tuned_val_metrics["loss"] = val_metrics["loss"]
                tuned_val_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                tuned_val_metrics["epoch_time_sec"] = 0.0
                tuned_val_metrics["peak_gpu_memory_mb"] = 0.0
            else:
                tuned_val_metrics = None

            step_scheduler(config, scheduler, epoch=epoch, batch_level=False)

            epoch_time_sec = time.perf_counter() - epoch_start
            learning_rate = float(optimizer.param_groups[0]["lr"])
            peak_gpu_memory_mb = _peak_gpu_memory_mb(device)

            train_metrics["learning_rate"] = learning_rate
            train_metrics["epoch_time_sec"] = epoch_time_sec
            train_metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            val_metrics["learning_rate"] = learning_rate
            val_metrics["epoch_time_sec"] = epoch_time_sec
            val_metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            if tuned_val_metrics is not None:
                tuned_val_metrics["learning_rate"] = learning_rate
                tuned_val_metrics["epoch_time_sec"] = epoch_time_sec
                tuned_val_metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb

            log_epoch_metrics(train_metrics, split="train", epoch=epoch)
            log_epoch_metrics(val_metrics, split="val", epoch=epoch)
            if tuned_val_metrics is not None:
                log_epoch_metrics(tuned_val_metrics, split="val_tuned", epoch=epoch)
                for label_name, tuned_threshold in latest_tuned_thresholds.items():
                    mlflow.log_metric(f"threshold_{label_name}", float(tuned_threshold), step=epoch)

            current_early_stopping_metric: float | None = None
            if early_stopping_state is not None:
                current_early_stopping_metric = resolve_early_stopping_metric_value(
                    str(early_stopping_state["metric_name"]),
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    tuned_val_metrics=tuned_val_metrics,
                )
                early_stopping_state = update_early_stopping_state(
                    early_stopping_state,
                    metric_value=current_early_stopping_metric,
                    epoch=epoch,
                )

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
                "pos_weight_stats": pos_weight_stats,
                "best_auc": best_auc,
                "best_macro_f1": best_macro_f1,
                "early_stopping_state": dict(early_stopping_state) if early_stopping_state is not None else None,
            }
            if scheduler is not None:
                checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
            if latest_tuned_thresholds is not None:
                checkpoint_state["tuned_thresholds"] = latest_tuned_thresholds

            if val_metrics["mean_auc"] > best_auc:
                best_auc = val_metrics["mean_auc"]
                best_auc_epoch = epoch
                checkpoint_state["best_auc"] = best_auc
                save_checkpoint(checkpoint_state, best_auc_path)
                mlflow.log_metric("best_val_mean_auc", best_auc)
                mlflow.log_metric("best_val_mean_auc_epoch", epoch)
                print(
                    f"[Checkpoint] New best val mean AUC: {best_auc:.4f}; saved to {best_auc_path}",
                    flush=True,
                )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_macro_f1_epoch = epoch
                checkpoint_state["best_macro_f1"] = best_macro_f1
                save_checkpoint(checkpoint_state, best_macro_f1_path)
                mlflow.log_metric("best_val_macro_f1", best_macro_f1)
                mlflow.log_metric("best_val_macro_f1_epoch", epoch)
                print(
                    f"[Checkpoint] New best val macro F1: {best_macro_f1:.4f}; saved to {best_macro_f1_path}",
                    flush=True,
                )

            summary_metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mean_auc": train_metrics["mean_auc"],
                "val_mean_auc": val_metrics["mean_auc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            if tuned_val_metrics is not None:
                summary_metrics["val_tuned_macro_f1"] = tuned_val_metrics["macro_f1"]
                summary_metrics["val_tuned_mean_average_precision"] = tuned_val_metrics["mean_average_precision"]

            history.append(
                {
                    "epoch": epoch,
                    "train_metrics": dict(train_metrics),
                    "val_metrics": dict(val_metrics),
                    "val_tuned_metrics": dict(tuned_val_metrics) if tuned_val_metrics is not None else None,
                    "tuned_thresholds": dict(latest_tuned_thresholds) if latest_tuned_thresholds is not None else None,
                }
            )
            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_mean_auc={val_metrics['mean_auc']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"time_sec={epoch_time_sec:.1f}",
                flush=True,
            )
            final_epoch_index = epoch

            if early_stopping_state is not None and bool(early_stopping_state["should_stop"]):
                stopped_early = True
                stopped_epoch = int(early_stopping_state["stopped_epoch"])
                print(
                    "[EarlyStopping] "
                    f"Stopping after epoch {epoch + 1} because "
                    f"{early_stopping_state['metric_name']} did not improve by at least "
                    f"{float(early_stopping_state['min_delta']):.4f} for "
                    f"{int(early_stopping_state['patience'])} epochs.",
                    flush=True,
                )
                break

        final_checkpoint_state = {
            "epoch": final_epoch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": history[-1]["val_metrics"] if history else {},
            "pos_weight_stats": pos_weight_stats,
            "best_auc": best_auc,
            "best_macro_f1": best_macro_f1,
            "early_stopping_state": dict(early_stopping_state) if early_stopping_state is not None else None,
        }
        if scheduler is not None:
            final_checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
        if latest_tuned_thresholds is not None:
            final_checkpoint_state["tuned_thresholds"] = latest_tuned_thresholds
        if final_checkpoint_path is not None:
            save_checkpoint(final_checkpoint_state, final_checkpoint_path)
            print(f"[Checkpoint] Saved final checkpoint to {final_checkpoint_path}", flush=True)

        if evaluate_test_after_training:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            test_result = evaluate_best_checkpoint_on_test_split(
                config=config,
                data_module=data_module,
                loss_fn=loss_fn,
                device=device,
                label_names=label_names,
                checkpoint_path=best_auc_path,
                threshold=threshold,
                max_batches=max_val_batches,
                progress_log_interval=progress_log_interval,
                use_amp=amp_enabled,
            )
            test_evaluation_skipped = bool(test_result["skipped"])
            test_evaluation_reason = test_result["reason"]
            if not test_evaluation_skipped:
                test_metrics = dict(test_result["metrics"])
                test_metrics["peak_gpu_memory_mb"] = _peak_gpu_memory_mb(device)
                log_epoch_metrics(test_metrics, split="test", epoch=final_epoch_index or 0)
            else:
                print(f"[Eval] Skipping test evaluation: {test_evaluation_reason}", flush=True)

    total_runtime_sec = time.perf_counter() - train_start_time
    print(f"[Train] Finished training in {total_runtime_sec:.1f} sec", flush=True)
    print(f"[Train] Best mean AUC: {best_auc:.4f} at epoch {best_auc_epoch}", flush=True)
    print(f"[Train] Best macro F1: {best_macro_f1:.4f} at epoch {best_macro_f1_epoch}", flush=True)
    return {
        "run_id": run.info.run_id,
        "summary_metrics": summary_metrics,
        "history": history,
        "label_names": label_names,
        "best_auc": best_auc,
        "best_auc_epoch": best_auc_epoch,
        "best_macro_f1": best_macro_f1,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_auc_checkpoint": str(best_auc_path),
        "best_macro_f1_checkpoint": str(best_macro_f1_path),
        "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
        "device": str(device),
        "pos_weight_stats": pos_weight_stats,
        "config": config,
        "total_runtime_sec": total_runtime_sec,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "early_stopping_state": dict(early_stopping_state) if early_stopping_state is not None else None,
        "final_epoch_index": final_epoch_index,
        "test_metrics": test_metrics,
        "test_evaluation_skipped": test_evaluation_skipped,
        "test_evaluation_reason": test_evaluation_reason,
    }


def train(config: dict) -> dict[str, float]:
    report = train_model(config)
    return dict(report["summary_metrics"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ViT baseline on NIH Chest X-ray manifests.")
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    summary = train(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
