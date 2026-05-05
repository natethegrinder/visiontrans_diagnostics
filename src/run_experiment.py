from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import load_config
from data import build_dataloaders, build_nih_manifests
from evaluate import compute_mean_auc, evaluate_model, print_auc_table, save_metrics_json, save_per_class_csv
from models import build_model
from train import ExperimentTrainer, build_criterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test a configured model.")
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


def _sample_manifest(source_path: Path, destination_path: Path, max_rows: int, seed: int) -> None:
    frame = pd.read_csv(source_path)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_index()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination_path, index=False)


def maybe_prepare_smoke_manifests(config: dict[str, Any]) -> dict[str, Any]:
    smoke_config = config.get("smoke_test", {})
    if not smoke_config.get("enabled", False):
        return config

    data_config = config["data"]
    seed = int(config.get("project", {}).get("seed", 42))
    run_name = config.get("run", {}).get("name", "smoke_test")
    smoke_dir = Path(data_config.get("manifest_dir", "../data/manifests")) / "smoke"

    train_path = smoke_dir / f"{run_name}_train.csv"
    val_path = smoke_dir / f"{run_name}_val.csv"
    test_path = smoke_dir / f"{run_name}_test.csv"

    _sample_manifest(
        Path(data_config["train_manifest"]),
        train_path,
        int(smoke_config.get("max_train_samples", 512)),
        seed,
    )
    _sample_manifest(
        Path(data_config["val_manifest"]),
        val_path,
        int(smoke_config.get("max_val_samples", 128)),
        seed,
    )
    if Path(data_config["test_manifest"]).exists():
        _sample_manifest(
            Path(data_config["test_manifest"]),
            test_path,
            int(smoke_config.get("max_test_samples", 128)),
            seed,
        )

    config = dict(config)
    config["data"] = dict(data_config)
    config["data"]["train_manifest"] = str(train_path)
    config["data"]["val_manifest"] = str(val_path)
    config["data"]["test_manifest"] = str(test_path)
    return config


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
    config = maybe_prepare_smoke_manifests(config)

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

    try:
        if mlflow is not None:
            mlflow.log_dict(criterion_summary, "criterion_summary.json")

        progress_interval = int(config.get("training", {}).get("progress_interval", 0))
        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(
                dataloaders["train"],
                epoch=epoch,
                total_epochs=epochs,
                progress_interval=progress_interval,
            )
            val_loss, val_preds, val_labels = trainer.val_epoch(dataloaders["val"])
            auc_results = compute_mean_auc(val_labels, val_preds)
            val_mean_auc = auc_results["mean"]

            history_row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mean_auc": float(val_mean_auc),
            }
            history.append(history_row)

            print(
                f"epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_mean_auc={val_mean_auc:.4f}"
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

        test_metrics_path = None
        test_per_class_path = None
        if best_checkpoint is not None and "test" in dataloaders:
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
                }
            )
            test_metrics_path = artifacts_dir / f"{run_name}_test_metrics.json"
            test_per_class_path = artifacts_dir / f"{run_name}_test_per_class_metrics.csv"
            save_metrics_json(test_metrics, test_metrics_path)
            save_per_class_csv(test_metrics, test_per_class_path)
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
            if best_checkpoint is not None:
                mlflow.log_artifact(str(best_checkpoint))
            if test_metrics_path is not None:
                mlflow.log_artifact(str(test_metrics_path))
            if test_per_class_path is not None:
                mlflow.log_artifact(str(test_per_class_path))

        print(f"Best val mean AUC: {best_auc:.4f}")
        if best_checkpoint is not None:
            print(f"Best checkpoint: {best_checkpoint}")
        print(f"History: {history_path}")
        print(f"Validation AUC JSON: {auc_path}")
        if test_metrics_path is not None:
            print(f"Test metrics JSON: {test_metrics_path}")
            print(f"Test per-class CSV: {test_per_class_path}")
    finally:
        if mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
