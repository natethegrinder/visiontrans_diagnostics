from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow


def _stringify_value(value: Any) -> str | int | float | bool:
	if isinstance(value, (str, int, float, bool)):
		return value
	return str(value)


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
	flat_config: dict[str, str | int | float | bool] = {}
	for key, value in config.items():
		flat_key = f"{prefix}.{key}" if prefix else str(key)
		if isinstance(value, Mapping):
			flat_config.update(flatten_config(value, prefix=flat_key))
		else:
			flat_config[flat_key] = _stringify_value(value)
	return flat_config


def configure_mlflow(config: Mapping[str, Any]) -> dict[str, Any]:
	mlflow_config = dict(config.get("mlflow", {}))
	tracking_uri = str(mlflow_config.get("tracking_uri", "mlruns"))
	experiment_name = str(mlflow_config.get("experiment_name", "default"))

	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
	return mlflow_config


def log_config_params(config: Mapping[str, Any]) -> None:
	mlflow.log_params(flatten_config(config))


def log_metrics(metrics: Mapping[str, Any], step: int | None = None) -> None:
	numeric_metrics = {
		key: float(value)
		for key, value in metrics.items()
		if isinstance(value, (int, float)) and value is not None
	}
	if numeric_metrics:
		mlflow.log_metrics(numeric_metrics, step=step)


def log_dict_artifact(name: str, payload: Mapping[str, Any]) -> None:
	mlflow.log_dict(dict(payload), f"{name}.json")


def log_checkpoint_artifact(checkpoint_path: Path, enabled: bool) -> None:
	if enabled and checkpoint_path.exists():
		mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")


def resolve_nonconflicting_path(path: Path) -> Path:
	if not path.exists():
		return path

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_name = f"{path.stem}_{timestamp}"
	candidate = path.with_name(f"{base_name}{path.suffix}")
	index = 1
	while candidate.exists():
		candidate = path.with_name(f"{base_name}_{index}{path.suffix}")
		index += 1
	return candidate


def resolve_nonconflicting_directory(path: Path) -> Path:
	if not path.exists():
		return path

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_name = f"{path.name}_{timestamp}"
	candidate = path.with_name(base_name)
	index = 1
	while candidate.exists():
		candidate = path.with_name(f"{base_name}_{index}")
		index += 1
	return candidate
