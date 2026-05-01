from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from typing import cast

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

try:
	from .data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
	from .mlflow_utils import configure_mlflow, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
	from .models.resnet import build_resnet_model
except ImportError:
	from data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
	from mlflow_utils import configure_mlflow, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
	from models.resnet import build_resnet_model


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	merged = copy.deepcopy(base)
	for key, value in override.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = deep_merge_dicts(merged[key], value)
		else:
			merged[key] = copy.deepcopy(value)
	return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
	config_path = Path(config_path).resolve()
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle) or {}

	extends = config.pop("extends", None)
	if extends is None:
		return config

	parent_config_path = (config_path.parent / extends).resolve()
	parent_config = load_config(parent_config_path)
	return deep_merge_dicts(parent_config, config)


def resolve_device(config: dict[str, Any], requested_device: str | None = None) -> torch.device:
	if requested_device:
		return torch.device(requested_device)

	device_name = str(config.get("project", {}).get("device", "auto")).lower()
	if device_name == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_name)


def resolve_threshold(config: dict[str, Any], threshold_override: float | None) -> float:
	if threshold_override is not None:
		return float(threshold_override)
	return float(config.get("evaluation", {}).get("threshold", 0.5))


def resolve_checkpoint_path(config: dict[str, Any], checkpoint_override: str | None = None) -> Path:
	if checkpoint_override:
		return Path(checkpoint_override)

	run_name = str(config.get("run", {}).get("name", "resnet_best"))
	return Path("outputs") / "checkpoints" / f"{run_name}.pt"


def load_checkpoint_into_model(checkpoint_path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"])
		return checkpoint
	if isinstance(checkpoint, dict):
		model.load_state_dict(checkpoint)
		return {}

	raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def _compute_label_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
	return float((predictions == targets).mean())


def _compute_exact_match_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
	return float(np.all(predictions == targets, axis=1).mean())


def _compute_macro_auroc(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> tuple[float | None, dict[str, float | None]]:
	per_class: dict[str, float | None] = {}
	valid_scores: list[float] = []

	for index, label in enumerate(label_names):
		target_column = targets[:, index]
		probability_column = probabilities[:, index]
		if np.unique(target_column).size < 2:
			per_class[label] = None
			continue

		score = float(roc_auc_score(target_column, probability_column))
		per_class[label] = score
		valid_scores.append(score)

	if not valid_scores:
		return None, per_class
	return float(np.mean(valid_scores)), per_class


def _compute_macro_pr_auc(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> tuple[float | None, dict[str, float | None]]:
	per_class: dict[str, float | None] = {}
	valid_scores: list[float] = []

	for index, label in enumerate(label_names):
		target_column = targets[:, index]
		probability_column = probabilities[:, index]
		if np.unique(target_column).size < 2:
			per_class[label] = None
			continue

		score = float(average_precision_score(target_column, probability_column))
		per_class[label] = score
		valid_scores.append(score)

	if not valid_scores:
		return None, per_class
	return float(np.mean(valid_scores)), per_class


def _compute_per_class_f1(targets: np.ndarray, predictions: np.ndarray, label_names: list[str]) -> dict[str, float]:
	per_class_scores: dict[str, float] = {}
	for index, label in enumerate(label_names):
		per_class_scores[label] = float(
			f1_score(targets[:, index], predictions[:, index], average="binary", zero_division=0)
		)
	return per_class_scores


def _compute_per_class_precision(targets: np.ndarray, predictions: np.ndarray, label_names: list[str]) -> dict[str, float]:
	per_class_scores: dict[str, float] = {}
	for index, label in enumerate(label_names):
		per_class_scores[label] = float(
			precision_score(targets[:, index], predictions[:, index], average="binary", zero_division=0)
		)
	return per_class_scores


def _compute_per_class_recall(targets: np.ndarray, predictions: np.ndarray, label_names: list[str]) -> dict[str, float]:
	per_class_scores: dict[str, float] = {}
	for index, label in enumerate(label_names):
		per_class_scores[label] = float(
			recall_score(targets[:, index], predictions[:, index], average="binary", zero_division=0)
		)
	return per_class_scores


def _compute_label_prevalence(values: np.ndarray, label_names: list[str]) -> dict[str, float]:
	return {
		label: float(values[:, index].mean())
		for index, label in enumerate(label_names)
	}


def compute_pos_weight_from_frame(
	frame: Any,
	label_names: list[str] | tuple[str, ...],
	min_positive_count: float = 1.0,
	max_pos_weight: float | None = None,
) -> torch.Tensor:
	total_samples = float(len(frame))
	if total_samples <= 0:
		raise ValueError("Training manifest is empty; cannot compute class imbalance weights.")

	weights: list[float] = []
	for label in label_names:
		positive_count = float(frame[label].sum())
		adjusted_positive_count = max(positive_count, min_positive_count)
		negative_count = max(total_samples - positive_count, 0.0)
		pos_weight = negative_count / adjusted_positive_count
		if max_pos_weight is not None:
			pos_weight = min(pos_weight, max_pos_weight)
		weights.append(pos_weight)

	return torch.tensor(weights, dtype=torch.float32)


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


def build_evaluation_criterion(
	config: dict[str, Any],
	train_frame: Any,
	device: torch.device,
	label_names: list[str] | tuple[str, ...] = NIH_CHEST_XRAY_LABELS,
) -> nn.Module:
	training_config = config.get("training", {})
	loss_name = str(training_config.get("loss", "bce_with_logits")).lower()
	imbalance_config = training_config.get("imbalance", {})
	enabled = bool(imbalance_config.get("enabled", False))
	strategy = str(imbalance_config.get("strategy", "none")).lower()

	pos_weight: torch.Tensor | None = None
	if enabled and strategy not in {"none", "", "null"}:
		if strategy != "pos_weight":
			raise ValueError(f"Unsupported imbalance strategy '{strategy}'")
		min_positive_count = float(imbalance_config.get("min_positive_count", 1.0))
		max_pos_weight = imbalance_config.get("max_pos_weight")
		max_pos_weight_value = float(max_pos_weight) if max_pos_weight is not None else None
		pos_weight = compute_pos_weight_from_frame(
			train_frame,
			label_names,
			min_positive_count=min_positive_count,
			max_pos_weight=max_pos_weight_value,
		).to(device)

	if loss_name in {"bce", "bce_with_logits", "bcewithlogitsloss"}:
		return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	if loss_name in {"focal", "focal_loss", "focalloss"}:
		focal_config = training_config.get("focal_loss", {})
		gamma = float(focal_config.get("gamma", 2.0))
		alpha = focal_config.get("alpha")
		alpha_value = float(alpha) if alpha is not None else None
		return BinaryFocalLoss(gamma=gamma, alpha=alpha_value, pos_weight=pos_weight)

	raise ValueError(f"Unsupported loss '{loss_name}'")


def evaluate_model(
	model: nn.Module,
	dataloader: torch.utils.data.DataLoader,
	criterion: nn.Module,
	device: torch.device,
	threshold: float,
	label_names: list[str] | tuple[str, ...] = NIH_CHEST_XRAY_LABELS,
) -> dict[str, Any]:
	model.eval()
	total_loss = 0.0
	total_batches = 0
	all_probabilities: list[np.ndarray] = []
	all_targets: list[np.ndarray] = []

	with torch.no_grad():
		for images, targets in dataloader:
			images = images.to(device)
			targets = targets.to(device).float()

			logits = model(images)
			loss = criterion(logits, targets)
			probabilities = torch.sigmoid(logits)

			total_loss += float(loss.item())
			total_batches += 1
			all_probabilities.append(probabilities.cpu().numpy())
			all_targets.append(targets.cpu().numpy())

	if not all_probabilities:
		raise ValueError("Dataloader yielded no batches during evaluation")

	probabilities_np = np.concatenate(all_probabilities, axis=0)
	targets_np = np.concatenate(all_targets, axis=0)
	predictions_np = (probabilities_np >= threshold).astype(np.float32)

	average_loss = total_loss / max(total_batches, 1)
	label_accuracy = _compute_label_accuracy(predictions_np, targets_np)
	exact_match_accuracy = _compute_exact_match_accuracy(predictions_np, targets_np)
	auroc, per_class_auroc = _compute_macro_auroc(targets_np, probabilities_np, list(label_names))
	pr_auc, per_class_pr_auc = _compute_macro_pr_auc(targets_np, probabilities_np, list(label_names))
	macro_f1 = float(f1_score(targets_np, predictions_np, average="macro", zero_division=0))
	micro_f1 = float(f1_score(targets_np, predictions_np, average="micro", zero_division=0))
	samples_f1 = float(f1_score(targets_np, predictions_np, average="samples", zero_division=0))
	per_class_f1 = _compute_per_class_f1(targets_np, predictions_np, list(label_names))
	macro_precision = float(precision_score(targets_np, predictions_np, average="macro", zero_division=0))
	micro_precision = float(precision_score(targets_np, predictions_np, average="micro", zero_division=0))
	macro_recall = float(recall_score(targets_np, predictions_np, average="macro", zero_division=0))
	micro_recall = float(recall_score(targets_np, predictions_np, average="micro", zero_division=0))
	per_class_precision = _compute_per_class_precision(targets_np, predictions_np, list(label_names))
	per_class_recall = _compute_per_class_recall(targets_np, predictions_np, list(label_names))
	target_prevalence = _compute_label_prevalence(targets_np, list(label_names))
	predicted_prevalence = _compute_label_prevalence(predictions_np, list(label_names))

	confusion_matrices = multilabel_confusion_matrix(targets_np, predictions_np)
	confusion_matrix = {
		label: confusion_matrices[index].astype(int).tolist()
		for index, label in enumerate(label_names)
	}

	return {
		"loss": average_loss,
		"accuracy": label_accuracy,
		"label_accuracy": label_accuracy,
		"exact_match_accuracy": exact_match_accuracy,
		"auroc": auroc,
		"pr_auc": pr_auc,
		"f1_score": macro_f1,
		"macro_f1": macro_f1,
		"micro_f1": micro_f1,
		"samples_f1": samples_f1,
		"precision": macro_precision,
		"macro_precision": macro_precision,
		"micro_precision": micro_precision,
		"recall": macro_recall,
		"macro_recall": macro_recall,
		"micro_recall": micro_recall,
		"confusion_matrix": confusion_matrix,
		"per_class_auroc": per_class_auroc,
		"per_class_pr_auc": per_class_pr_auc,
		"per_class_f1": per_class_f1,
		"per_class_precision": per_class_precision,
		"per_class_recall": per_class_recall,
		"target_prevalence": target_prevalence,
		"predicted_prevalence": predicted_prevalence,
		"threshold": threshold,
		"num_examples": int(targets_np.shape[0]),
	}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate a CNN checkpoint on the NIH Chest X-ray dataset.")
	parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Path to the YAML config file.")
	parser.add_argument("--checkpoint", default=None, help="Path to the trained checkpoint. If omitted, evaluate all CNN experiment configs.")
	parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
	parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
	parser.add_argument("--threshold", type=float, default=None, help="Probability threshold for positive labels.")
	parser.add_argument("--output", default=None, help="Optional path for saving evaluation results as JSON.")
	parser.add_argument("--loss", choices=["bce", "focal"], default=None, help="Optional loss override for selecting and evaluating runs.")
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


def resolve_output_path(config: dict[str, Any], checkpoint_path: Path, split: str, output_override: str | None) -> Path:
	if output_override:
		return resolve_nonconflicting_path(Path(output_override))

	run_name = str(config.get("run", {}).get("name", checkpoint_path.stem))
	return resolve_nonconflicting_path(Path("outputs") / "evaluations" / f"{run_name}_{split}.json")


def save_metrics(output_path: Path, metrics: dict[str, Any]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_mlflow_metric_payload(split: str, metrics: dict[str, Any]) -> dict[str, float]:
	metric_names = (
		"loss",
		"accuracy",
		"label_accuracy",
		"exact_match_accuracy",
		"auroc",
		"pr_auc",
		"f1_score",
		"macro_f1",
		"micro_f1",
		"samples_f1",
		"precision",
		"macro_precision",
		"micro_precision",
		"recall",
		"macro_recall",
		"micro_recall",
		"threshold",
		"num_examples",
	)
	payload: dict[str, float] = {}
	for name in metric_names:
		value = metrics.get(name)
		if isinstance(value, (int, float)) and value is not None:
			payload[f"{split}_{name}"] = float(value)
	return payload


def log_evaluation_run(
	config: dict[str, Any],
	checkpoint_path: Path,
	device: torch.device,
	split: str,
	threshold: float,
	output_path: Path,
	metrics: dict[str, Any],
	checkpoint: dict[str, Any],
) -> None:
	mlflow_config = configure_mlflow(config)
	evaluation_run_name = f"{checkpoint_path.stem}_{split}_evaluation"
	with mlflow.start_run(run_name=evaluation_run_name):
		log_config_params(config)
		mlflow.log_param("evaluation_split", split)
		mlflow.log_param("evaluation_device", str(device))
		mlflow.log_param("evaluation_checkpoint_path", str(checkpoint_path))
		mlflow.log_param("evaluation_output_path", str(output_path))
		mlflow.log_param("evaluation_threshold", threshold)
		if checkpoint:
			if checkpoint.get("epoch") is not None:
				mlflow.log_param("checkpoint_epoch", checkpoint["epoch"])
			if checkpoint.get("best_val_loss") is not None:
				mlflow.log_param("checkpoint_best_val_loss", checkpoint["best_val_loss"])
		log_metrics(build_mlflow_metric_payload(split, metrics))
		log_dict_artifact(f"{split}_metrics", metrics)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(output_path), artifact_path="evaluations")


def run_single_evaluation(
	config_path: Path,
	checkpoint_path: Path,
	split: str,
	device_override: str | None,
	threshold_override: float | None,
	loss_override: str | None,
	focal_gamma_override: float | None,
	focal_alpha_override: float | None,
	output_override: str | None = None,
) -> tuple[Path, dict[str, Any]]:
	config = load_config(config_path)
	apply_loss_overrides(config, loss_override, focal_gamma_override, focal_alpha_override)
	device = resolve_device(config, device_override)
	threshold = resolve_threshold(config, threshold_override)

	data_module = build_nih_data_module(config)
	dataloaders = cast(dict[str, DataLoader], data_module["dataloaders"])
	if split not in dataloaders:
		raise ValueError(f"Split '{split}' is not available. Available splits: {sorted(dataloaders)}")

	train_frame = cast(Any, dataloaders["train"].dataset).frame
	model = build_resnet_model(config).to(device)
	checkpoint = load_checkpoint_into_model(checkpoint_path, model, device)
	criterion = build_evaluation_criterion(config, train_frame, device)
	metrics = evaluate_model(model, dataloaders[split], criterion, device, threshold)
	metrics["split"] = split
	metrics["config"] = str(config_path)
	metrics["checkpoint_path"] = str(checkpoint_path)
	if checkpoint:
		metrics["checkpoint_epoch"] = checkpoint.get("epoch")
		metrics["best_val_loss"] = checkpoint.get("best_val_loss")

	output_path = resolve_output_path(config, checkpoint_path, split, output_override)
	save_metrics(output_path, metrics)
	log_evaluation_run(config, checkpoint_path, device, split, threshold, output_path, metrics, checkpoint)
	return output_path, metrics


def discover_experiment_configs(configs_dir: Path) -> list[Path]:
	return sorted(
		path for path in configs_dir.glob("cnn*.yaml")
		if path.name not in {"cnn_template.yaml"}
	)


def run_all_experiments(
	split: str,
	device_override: str | None,
	threshold_override: float | None,
	loss_override: str | None,
	focal_gamma_override: float | None,
	focal_alpha_override: float | None,
) -> Path:
	configs_dir = Path("configs")
	results: list[dict[str, Any]] = []
	skipped: list[dict[str, str]] = []

	for config_path in discover_experiment_configs(configs_dir):
		config = load_config(config_path)
		apply_loss_overrides(config, loss_override, focal_gamma_override, focal_alpha_override)
		checkpoint_path = resolve_checkpoint_path(config)
		if not checkpoint_path.exists():
			skipped.append(
				{
					"config": str(config_path),
					"checkpoint_path": str(checkpoint_path),
					"reason": "checkpoint_not_found",
				}
			)
			continue

		output_path, metrics = run_single_evaluation(
			config_path=config_path,
			checkpoint_path=checkpoint_path,
			split=split,
			device_override=device_override,
			threshold_override=threshold_override,
			loss_override=loss_override,
			focal_gamma_override=focal_gamma_override,
			focal_alpha_override=focal_alpha_override,
		)
		results.append(
			{
				"config": str(config_path),
				"checkpoint_path": str(checkpoint_path),
				"output_path": str(output_path),
				"run_name": str(config.get("run", {}).get("name", checkpoint_path.stem)),
				"metrics": metrics,
			}
		)

	summary = {
		"split": split,
		"num_completed": len(results),
		"num_skipped": len(skipped),
		"results": results,
		"skipped": skipped,
	}

	output_path = Path("outputs") / "evaluations" / f"all_experiments_{split}.json"
	save_metrics(output_path, summary)

	if results or skipped:
		base_config = load_config(Path("configs") / "base.yaml")
		mlflow_config = configure_mlflow(base_config)
		with mlflow.start_run(run_name=f"all_experiments_{split}_evaluation"):
			mlflow.log_param("evaluation_mode", "all_experiments")
			mlflow.log_param("evaluation_split", split)
			mlflow.log_param("num_completed", len(results))
			mlflow.log_param("num_skipped", len(skipped))
			log_dict_artifact(f"all_experiments_{split}", summary)
			if bool(mlflow_config.get("log_artifacts", True)):
				mlflow.log_artifact(str(output_path), artifact_path="evaluations")

	return output_path


def main() -> None:
	args = build_arg_parser().parse_args()
	if args.checkpoint is None:
		output_path = run_all_experiments(
			args.split,
			args.device,
			args.threshold,
			args.loss,
			args.focal_gamma,
			args.focal_alpha,
		)
		print(f"Saved all experiment results to {output_path}")
		return

	output_path, _ = run_single_evaluation(
		config_path=Path(args.config),
		checkpoint_path=Path(args.checkpoint),
		split=args.split,
		device_override=args.device,
		threshold_override=args.threshold,
		loss_override=args.loss,
		focal_gamma_override=args.focal_gamma,
		focal_alpha_override=args.focal_alpha,
		output_override=args.output,
	)
	print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
	main()
