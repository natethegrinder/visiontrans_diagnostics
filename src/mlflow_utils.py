from __future__ import annotations

from typing import Mapping, Sequence

import mlflow
import os

def configure_mlflow(config: dict) -> None:
    mlflow_config = config.get("mlflow", {})

    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or mlflow_config.get("tracking_uri")
    )
    experiment_name = mlflow_config.get("experiment_name")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}", flush=True)
    if experiment_name:
        print(f"[MLflow] experiment_name={experiment_name}", flush=True)


def log_params_flat(params: Mapping[str, object]) -> None:
    sanitized = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            sanitized[key] = ",".join(str(item) for item in value)
        else:
            sanitized[key] = value
    mlflow.log_params(sanitized)


def log_epoch_metrics(metrics: Mapping[str, float], split: str, epoch: int) -> None:
    prefixed = {f"{split}_{key}": value for key, value in metrics.items()}
    mlflow.log_metrics(prefixed, step=epoch)


def log_label_statistics(pos_weight_stats: Mapping[str, object]) -> None:
    label_names: Sequence[str] = pos_weight_stats["label_names"]
    for label_name in label_names:
        mlflow.log_metric(f"positive_count_{label_name}", pos_weight_stats["positive_counts"][label_name])
        mlflow.log_metric(f"negative_count_{label_name}", pos_weight_stats["negative_counts"][label_name])
        mlflow.log_metric(f"prevalence_{label_name}", pos_weight_stats["prevalence"][label_name])
        mlflow.log_metric(f"pos_weight_{label_name}", pos_weight_stats["pos_weight"][label_name])
