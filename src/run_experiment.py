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
from evaluate import compute_mean_auc, print_auc_table
from models import build_model
from train import DEFAULT_POS_WEIGHT, ViTTrainer


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
    if not force and all(path.exists() for path in manifest_paths):
        return

    build_nih_manifests(
        raw_dir=data_config["raw_dir"],
        annotations_dir=data_config["annotations_dir"],
        manifest_dir=data_config["manifest_dir"],
        val_fraction=float(data_config.get("val_fraction", 0.1)),
        seed=int(config.get("project", {}).get("seed", 42)),
    )


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
    smoke_dir = Path(data_config.get("manifest_dir", "data/manifests")) / "smoke"

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


def save_checkpoint(model: torch.nn.Module, config: dict[str, Any], epoch: int, val_auc: float) -> Path:
    run_name = config.get("run", {}).get("name", "model")
    checkpoint_dir = Path(config.get("artifacts", {}).get("model_dir", "artifacts/models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "val_mean_auc": val_auc,
            "model_state_dict": model.state_dict(),
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
    if device.type == "mps":
        print("MPS is enabled. If an unsupported op appears, rerun with --device cpu for debugging.")

    dataloaders = build_dataloaders(config)
    model = build_model(config)
    trainer = ViTTrainer(
        model=model,
        device=device,
        pos_weight=DEFAULT_POS_WEIGHT,
        lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )

    epochs = int(config.get("training", {}).get("epochs", 1))
    if config.get("training", {}).get("scheduler", "") == "cosine":
        trainer.setup_scheduler(num_training_steps=max(1, epochs * len(dataloaders["train"])))

    mlflow_run = maybe_start_mlflow(config, disabled=args.no_mlflow)
    mlflow = mlflow_run[0] if mlflow_run else None

    best_auc = float("-inf")
    best_checkpoint = None
    history: list[dict[str, float]] = []

    try:
        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(dataloaders["train"])
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

            if val_mean_auc > best_auc:
                best_auc = val_mean_auc
                best_checkpoint = save_checkpoint(model, config, epoch, val_mean_auc)

        print_auc_table(auc_results)

        artifacts_dir = Path(config.get("artifacts", {}).get("metrics_dir", "artifacts/metrics"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        run_name = config.get("run", {}).get("name", "model")
        history_path = artifacts_dir / f"{run_name}_history.csv"
        auc_path = artifacts_dir / f"{run_name}_val_auc.json"
        pd.DataFrame(history).to_csv(history_path, index=False)
        auc_path.write_text(json.dumps(auc_results, indent=2))

        if mlflow is not None:
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(auc_path))
            if best_checkpoint is not None:
                mlflow.log_artifact(str(best_checkpoint))

        print(f"Best val mean AUC: {best_auc:.4f}")
        if best_checkpoint is not None:
            print(f"Best checkpoint: {best_checkpoint}")
        print(f"History: {history_path}")
        print(f"Validation AUC JSON: {auc_path}")
    finally:
        if mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
