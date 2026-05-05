from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import load_config
from data import build_dataloaders, build_nih_manifests
from evaluate import compute_mean_auc, compute_multilabel_metrics, evaluate_model, print_auc_table, save_metrics_json, save_per_class_csv
from models import build_model
from train import ExperimentTrainer, build_criterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a configured model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"], help="Device override.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--force-manifests", action="store_true", help="Regenerate train/val/test manifests.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    requested = requested or "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is not available in this Python environment. "
            "Check your PyTorch install and macOS setup."
        )
    return torch.device(requested)


def ensure_manifests(config: dict[str, Any], force: bool = False) -> None:
    data_config = config["data"]
    manifest_paths = [
        Path(data_config["train_manifest"]),
        Path(data_config["val_manifest"]),
        Path(data_config["test_manifest"]),
    ]
    if not force and all(path.exists() for path in manifest_paths) and _manifest_image_paths_exist(manifest_paths):
        return

    build_nih_manifests(
        raw_dir=data_config["raw_dir"],
        annotations_dir=data_config["annotations_dir"],
        manifest_dir=data_config["manifest_dir"],
        val_fraction=float(data_config.get("val_fraction", 0.1)),
        seed=int(config.get("project", {}).get("seed", 42)),
    )


def _manifest_image_paths_exist(manifest_paths: list[Path]) -> bool:
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            return False
        try:
            frame = pd.read_csv(manifest_path, nrows=20)
        except pd.errors.EmptyDataError:
            return False
        if "image_path" not in frame.columns:
            return False
        paths = [Path(path) for path in frame["image_path"].dropna().astype(str).tolist()]
        if paths and not all(path.exists() for path in paths):
            return False
    return True


def maybe_start_mlflow(config: dict[str, Any], disabled: bool):
    if disabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; continuing without MLflow logging.")
        return None

    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "mlruns"))
    experiment_name = mlflow_config.get("experiment_name", "cnn_vs_vit_medical")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = mlflow_config.get("artifact_location")
        mlflow.create_experiment(
            experiment_name,
            artifact_location=str(artifact_location) if artifact_location else None,
        )
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=config.get("run", {}).get("name"))
    mlflow.log_params(flatten_config(config))
    return mlflow, run


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[full_key] = value
    return flat


def gpu_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_peak_memory_allocated_mb": 0.0,
            "gpu_peak_memory_reserved_mb": 0.0,
        }
    bytes_per_mb = 1024.0 * 1024.0
    return {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / bytes_per_mb),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / bytes_per_mb),
        "gpu_peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / bytes_per_mb),
        "gpu_peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / bytes_per_mb),
    }


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def save_training_charts(history: list[dict[str, float]], metrics_dir: Path, run_name: str) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    if not history:
        return paths
    frame = pd.DataFrame(history)

    loss_path = metrics_dir / f"{run_name}_loss_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_name}: train/val loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)
    paths["loss_curve"] = loss_path

    f1_path = metrics_dir / f"{run_name}_f1_vs_epoch.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    if "val_macro_f1" in frame:
        ax.plot(frame["epoch"], frame["val_macro_f1"], label="val_macro_f1")
    if "val_micro_f1" in frame:
        ax.plot(frame["epoch"], frame["val_micro_f1"], label="val_micro_f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title(f"{run_name}: F1 vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f1_path, dpi=160)
    plt.close(fig)
    paths["f1_vs_epoch"] = f1_path
    return paths


def save_confusion_matrix_chart(metrics: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics.get("aggregate_confusion_matrix", [[0, 0], [0, 0]]), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Aggregate multilabel confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    val_auc: float,
    val_loss: float,
) -> Path:
    run_name = config.get("run", {}).get("name", "model")
    checkpoint_dir = Path(config.get("artifacts", {}).get("model_dir", "artifacts/models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "val_mean_auc": val_auc,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 42)))

    ensure_manifests(config, force=args.force_manifests)

    device = resolve_device(args.device or config.get("project", {}).get("device", "auto"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.get("training", {}).get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(str(config.get("training", {}).get("float32_matmul_precision", "high")))
    if device.type == "mps":
        print("MPS is enabled. If an unsupported op appears, rerun with --device cpu for debugging.")

    dataloaders = build_dataloaders(config)
    model = build_model(config)
    train_frame = dataloaders["train"].dataset.frame
    criterion, criterion_summary = build_criterion(config, train_frame, device)
    print(json.dumps({"criterion": criterion_summary}, indent=2))

    trainer = ExperimentTrainer(
        model=model,
        device=device,
        criterion=criterion,
        lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
        use_amp=bool(config.get("training", {}).get("mixed_precision", False)),
    )

    epochs = int(config.get("training", {}).get("epochs", 1))
    if config.get("training", {}).get("scheduler", "") == "cosine":
        trainer.setup_scheduler(num_training_steps=max(1, epochs * len(dataloaders["train"])))

    mlflow_run = maybe_start_mlflow(config, disabled=args.no_mlflow)
    mlflow = mlflow_run[0] if mlflow_run else None

    best_auc = float("-inf")
    best_checkpoint = None
    best_epoch = -1
    patience = int(config.get("training", {}).get("early_stopping_patience", 0))
    patience_counter = 0
    history: list[dict[str, float]] = []
    run_start_time = time.perf_counter()

    try:
        if mlflow is not None:
            mlflow.log_dict(criterion_summary, "criterion_summary.json")

        progress_interval = int(config.get("training", {}).get("progress_interval", 0))
        for epoch in range(1, epochs + 1):
            reset_gpu_peak_memory(device)
            epoch_start_time = time.perf_counter()
            train_loss = trainer.train_epoch(
                dataloaders["train"],
                epoch=epoch,
                total_epochs=epochs,
                progress_interval=progress_interval,
            )
            train_time_sec = time.perf_counter() - epoch_start_time
            val_start_time = time.perf_counter()
            val_loss, val_preds, val_labels = trainer.val_epoch(dataloaders["val"])
            val_time_sec = time.perf_counter() - val_start_time
            epoch_time_sec = time.perf_counter() - epoch_start_time
            auc_results = compute_mean_auc(val_labels, val_preds)
            val_metrics = compute_multilabel_metrics(
                val_labels,
                val_preds,
                threshold=float(config.get("evaluation", {}).get("threshold", 0.5)),
            )
            val_mean_auc = auc_results["mean"]

            history_row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mean_auc": float(val_mean_auc),
                "val_mean_average_precision": float(val_metrics["mean_pr_auc"] or 0.0),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_micro_f1": float(val_metrics["micro_f1"]),
                "val_macro_precision": float(val_metrics["macro_precision"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "train_time_sec": float(train_time_sec),
                "val_time_sec": float(val_time_sec),
                "epoch_time_sec": float(epoch_time_sec),
                **trainer.average_gpu_stats(),
                **gpu_memory_metrics(device),
            }
            history.append(history_row)

            print(
                f"epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_mean_auc={val_mean_auc:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"epoch_time_sec={epoch_time_sec:.1f} "
                f"gpu_peak_mb={history_row['gpu_peak_memory_allocated_mb']:.1f}"
            )

            if mlflow is not None:
                mlflow.log_metrics(history_row, step=epoch)
                mlflow.log_metrics(
                    {f"val_auc.{label}": value for label, value in auc_results.items()},
                    step=epoch,
                )

            improved = bool(np.isfinite(val_mean_auc) and val_mean_auc > best_auc)
            if improved:
                best_auc = val_mean_auc
                best_epoch = epoch
                patience_counter = 0
                best_checkpoint = save_checkpoint(
                    model,
                    trainer.optimizer,
                    config,
                    epoch,
                    val_mean_auc,
                    val_loss,
                )
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best_epoch={best_epoch} best_val_mean_auc={best_auc:.4f}"
                )
                break

        print_auc_table(auc_results)

        artifacts_dir = Path(config.get("artifacts", {}).get("metrics_dir", "artifacts/metrics"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        run_name = config.get("run", {}).get("name", "model")
        history_path = artifacts_dir / f"{run_name}_history.csv"
        auc_path = artifacts_dir / f"{run_name}_val_auc.json"
        criterion_path = artifacts_dir / f"{run_name}_criterion.json"
        pd.DataFrame(history).to_csv(history_path, index=False)
        auc_path.write_text(json.dumps(auc_results, indent=2))
        criterion_path.write_text(json.dumps(criterion_summary, indent=2))
        chart_paths = save_training_charts(history, artifacts_dir, run_name)

        test_metrics_path = None
        test_per_class_path = None
        test_confusion_path = None
        if best_checkpoint is not None and "test" in dataloaders:
            test_start_time = time.perf_counter()
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
            test_metrics = evaluate_model(
                model,
                dataloaders["test"],
                criterion,
                device,
                threshold=threshold,
                use_amp=bool(config.get("training", {}).get("mixed_precision", False)),
            )
            test_metrics.update(
                {
                    "split": "test",
                    "checkpoint_path": str(best_checkpoint),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_val_mean_auc": checkpoint.get("val_mean_auc"),
                    "test_time_sec": float(time.perf_counter() - test_start_time),
                    "total_run_time_sec": float(time.perf_counter() - run_start_time),
                    **{f"final_{key}": value for key, value in gpu_memory_metrics(device).items()},
                }
            )
            test_metrics_path = artifacts_dir / f"{run_name}_test_metrics.json"
            test_per_class_path = artifacts_dir / f"{run_name}_test_per_class_metrics.csv"
            test_confusion_path = artifacts_dir / f"{run_name}_test_confusion_matrix.png"
            save_metrics_json(test_metrics, test_metrics_path)
            save_per_class_csv(test_metrics, test_per_class_path)
            save_confusion_matrix_chart(test_metrics, test_confusion_path)
            print(
                f"Test metrics: loss={test_metrics['loss']:.4f} "
                f"mean_auroc={test_metrics['mean_auroc']:.4f} "
                f"mean_pr_auc={test_metrics['mean_pr_auc']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f}"
            )

        if mlflow is not None:
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(auc_path))
            mlflow.log_artifact(str(criterion_path))
            for chart_path in chart_paths.values():
                mlflow.log_artifact(str(chart_path))
            if best_checkpoint is not None:
                mlflow.log_artifact(str(best_checkpoint))
            if test_metrics_path is not None:
                mlflow.log_artifact(str(test_metrics_path))
            if test_per_class_path is not None:
                mlflow.log_artifact(str(test_per_class_path))
            if test_confusion_path is not None:
                mlflow.log_artifact(str(test_confusion_path))

        print(f"Best val mean AUC: {best_auc:.4f}")
        if best_checkpoint is not None:
            print(f"Best checkpoint: {best_checkpoint}")
        print(f"History: {history_path}")
        print(f"Validation AUC JSON: {auc_path}")
        if test_metrics_path is not None:
            print(f"Test metrics JSON: {test_metrics_path}")
            print(f"Test per-class CSV: {test_per_class_path}")
            print(f"Test confusion matrix PNG: {test_confusion_path}")
    finally:
        if mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
