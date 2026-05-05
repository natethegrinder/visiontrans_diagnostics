from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

try:
	from tqdm.auto import tqdm
except ImportError:
	tqdm = None

try:
	from .data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
	from .evaluate import (
		build_evaluation_criterion,
		evaluate_model,
		load_checkpoint_into_model,
		load_config,
		resolve_checkpoint_path,
		resolve_device,
		resolve_threshold,
	)
	from .mlflow_utils import configure_mlflow, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
	from .models.resnet import build_resnet_model
except ImportError:
	from data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
	from evaluate import (
		build_evaluation_criterion,
		evaluate_model,
		load_checkpoint_into_model,
		load_config,
		resolve_checkpoint_path,
		resolve_device,
		resolve_threshold,
	)
	from mlflow_utils import configure_mlflow, log_config_params, log_dict_artifact, log_metrics, resolve_nonconflicting_path
	from models.resnet import build_resnet_model


T = TypeVar("T")


def _len_or_none(iterable: object) -> int | None:
	try:
		return len(iterable)  # type: ignore[arg-type]
	except TypeError:
		return None


def with_progress(
	iterable: Iterable[T],
	*,
	desc: str,
	total: int | None = None,
	disable: bool = False,
) -> Iterable[T]:
	if disable or tqdm is None:
		return iterable
	return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Tune per-label probability thresholds for NIH multilabel classification.")
	parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Path to the YAML config file.")
	parser.add_argument("--checkpoint", default=None, help="Path to the trained checkpoint. Defaults to the run checkpoint from config.")
	parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
	parser.add_argument("--loss", choices=["bce", "focal"], default=None, help="Optional loss override for loading the matching run config.")
	parser.add_argument("--tune-split", default="val", choices=["train", "val", "test"], help="Split used to search thresholds.")
	parser.add_argument("--report-split", default="test", choices=["train", "val", "test"], help="Optional split used to report tuned-threshold metrics.")
	parser.add_argument("--min-threshold", type=float, default=0.05, help="Lower bound of the threshold search range.")
	parser.add_argument("--max-threshold", type=float, default=0.95, help="Upper bound of the threshold search range.")
	parser.add_argument("--step", type=float, default=0.05, help="Step size of the threshold search grid.")
	parser.add_argument("--metric", choices=["f1"], default="f1", help="Metric used to select each label threshold.")
	parser.add_argument("--output", default=None, help="Optional output path for the threshold tuning JSON.")
	parser.add_argument("--no-progress", action="store_true", help="Disable progress bar display.")
	return parser


def _strip_loss_suffix(run_name: str) -> str:
	for suffix in ("_bce", "_focal"):
		if run_name.endswith(suffix):
			return run_name[: -len(suffix)]
	return run_name


def apply_loss_overrides(config: dict[str, Any], loss_override: str | None) -> None:
	if loss_override is None:
		return

	training_config = config.setdefault("training", {})
	training_config["loss"] = loss_override
	run_config = config.setdefault("run", {})
	base_run_name = _strip_loss_suffix(str(run_config.get("name", "resnet_best")))
	run_config["name"] = f"{base_run_name}_{loss_override}"


def collect_predictions(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	progress_desc: str,
	show_progress: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
	model.eval()
	total_loss = 0.0
	total_batches = 0
	all_probabilities: list[np.ndarray] = []
	all_targets: list[np.ndarray] = []
	batch_iterator = with_progress(
		dataloader,
		desc=progress_desc,
		total=_len_or_none(dataloader),
		disable=not show_progress,
	)

	with torch.no_grad():
		for images, targets in batch_iterator:
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
		raise ValueError("Dataloader yielded no batches during threshold tuning")

	return (
		total_loss / max(total_batches, 1),
		np.concatenate(all_probabilities, axis=0),
		np.concatenate(all_targets, axis=0),
	)


def build_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> np.ndarray:
	if not 0.0 < min_threshold <= 1.0:
		raise ValueError("min-threshold must be in (0, 1]")
	if not 0.0 < max_threshold <= 1.0:
		raise ValueError("max-threshold must be in (0, 1]")
	if min_threshold > max_threshold:
		raise ValueError("min-threshold must be <= max-threshold")
	if step <= 0.0:
		raise ValueError("step must be > 0")

	return np.round(np.arange(min_threshold, max_threshold + step * 0.5, step), 6)


def select_thresholds(
	probabilities: np.ndarray,
	targets: np.ndarray,
	label_names: list[str] | tuple[str, ...],
	threshold_grid: np.ndarray,
	progress_desc: str,
	show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
	selected_thresholds: list[float] = []
	per_label_summary: dict[str, dict[str, float]] = {}
	label_iterator = with_progress(
		enumerate(label_names),
		desc=progress_desc,
		total=len(label_names),
		disable=not show_progress,
	)

	for index, label in label_iterator:
		label_probabilities = probabilities[:, index]
		label_targets = targets[:, index]

		best_threshold = float(threshold_grid[0])
		best_f1 = -1.0
		best_precision = 0.0
		best_recall = 0.0

		for threshold in threshold_grid:
			predictions = (label_probabilities >= threshold).astype(np.float32)
			f1 = float(f1_score(label_targets, predictions, zero_division=0))
			precision = float(precision_score(label_targets, predictions, zero_division=0))
			recall = float(recall_score(label_targets, predictions, zero_division=0))

			if f1 > best_f1 or (f1 == best_f1 and abs(precision - recall) < abs(best_precision - best_recall)):
				best_threshold = float(threshold)
				best_f1 = f1
				best_precision = precision
				best_recall = recall

		selected_thresholds.append(best_threshold)
		per_label_summary[label] = {
			"threshold": best_threshold,
			"f1": best_f1,
			"precision": best_precision,
			"recall": best_recall,
			"target_prevalence": float(label_targets.mean()),
			"predicted_prevalence": float((label_probabilities >= best_threshold).mean()),
		}

	return np.array(selected_thresholds, dtype=np.float32), per_label_summary


def evaluate_with_thresholds(
	probabilities: np.ndarray,
	targets: np.ndarray,
	thresholds: np.ndarray,
	label_names: list[str] | tuple[str, ...],
	loss: float,
	threshold_source: str,
) -> dict[str, Any]:
	predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.float32)
	label_accuracy = float((predictions == targets).mean())
	exact_match_accuracy = float(np.all(predictions == targets, axis=1).mean())
	macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
	micro_f1 = float(f1_score(targets, predictions, average="micro", zero_division=0))
	samples_f1 = float(f1_score(targets, predictions, average="samples", zero_division=0))
	macro_precision = float(precision_score(targets, predictions, average="macro", zero_division=0))
	micro_precision = float(precision_score(targets, predictions, average="micro", zero_division=0))
	macro_recall = float(recall_score(targets, predictions, average="macro", zero_division=0))
	micro_recall = float(recall_score(targets, predictions, average="micro", zero_division=0))

	per_class_f1 = {
		label: float(f1_score(targets[:, index], predictions[:, index], zero_division=0))
		for index, label in enumerate(label_names)
	}
	per_class_precision = {
		label: float(precision_score(targets[:, index], predictions[:, index], zero_division=0))
		for index, label in enumerate(label_names)
	}
	per_class_recall = {
		label: float(recall_score(targets[:, index], predictions[:, index], zero_division=0))
		for index, label in enumerate(label_names)
	}
	target_prevalence = {
		label: float(targets[:, index].mean())
		for index, label in enumerate(label_names)
	}
	predicted_prevalence = {
		label: float(predictions[:, index].mean())
		for index, label in enumerate(label_names)
	}

	return {
		"loss": loss,
		"accuracy": label_accuracy,
		"label_accuracy": label_accuracy,
		"exact_match_accuracy": exact_match_accuracy,
		"macro_f1": macro_f1,
		"micro_f1": micro_f1,
		"samples_f1": samples_f1,
		"precision": macro_precision,
		"macro_precision": macro_precision,
		"micro_precision": micro_precision,
		"recall": macro_recall,
		"macro_recall": macro_recall,
		"micro_recall": micro_recall,
		"per_class_f1": per_class_f1,
		"per_class_precision": per_class_precision,
		"per_class_recall": per_class_recall,
		"target_prevalence": target_prevalence,
		"predicted_prevalence": predicted_prevalence,
		"threshold_source": threshold_source,
		"thresholds": {label: float(thresholds[index]) for index, label in enumerate(label_names)},
		"num_examples": int(targets.shape[0]),
	}


def resolve_output_path(config: dict[str, Any], checkpoint_path: Path, tune_split: str, report_split: str, output_override: str | None) -> Path:
	if output_override is not None:
		return resolve_nonconflicting_path(Path(output_override))

	run_name = str(config.get("run", {}).get("name", checkpoint_path.stem))
	default_path = Path("outputs") / "thresholds" / f"{run_name}_{tune_split}_to_{report_split}_thresholds.json"
	return resolve_nonconflicting_path(default_path)


def main() -> None:
	args = build_arg_parser().parse_args()
	config = load_config(args.config)
	apply_loss_overrides(config, args.loss)
	device = resolve_device(config, args.device)
	checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
	default_threshold = resolve_threshold(config, None)
	threshold_grid = build_threshold_grid(args.min_threshold, args.max_threshold, args.step)

	data_module = build_nih_data_module(config)
	dataloaders = cast(dict[str, DataLoader], data_module["dataloaders"])
	train_frame = cast(Any, dataloaders["train"].dataset).frame

	model = build_resnet_model(config).to(device)
	checkpoint = load_checkpoint_into_model(checkpoint_path, model, device)
	criterion = build_evaluation_criterion(config, train_frame, device)
	show_progress = not args.no_progress

	tune_dataloader = cast(DataLoader, dataloaders[args.tune_split])
	tune_loss, tune_probabilities, tune_targets = collect_predictions(
		model,
		tune_dataloader,
		criterion,
		device,
		progress_desc=f"Collecting {args.tune_split} predictions",
		show_progress=show_progress,
	)
	tuned_thresholds, per_label_summary = select_thresholds(
		tune_probabilities,
		tune_targets,
		NIH_CHEST_XRAY_LABELS,
		threshold_grid,
		progress_desc=f"Searching {args.tune_split} thresholds",
		show_progress=show_progress,
	)

	default_tune_metrics = evaluate_model(model, tune_dataloader, criterion, device, default_threshold)
	tuned_tune_metrics = evaluate_with_thresholds(
		probabilities=tune_probabilities,
		targets=tune_targets,
		thresholds=tuned_thresholds,
		label_names=NIH_CHEST_XRAY_LABELS,
		loss=tune_loss,
		threshold_source=args.tune_split,
	)

	report_payload: dict[str, Any] | None = None
	if args.report_split in dataloaders:
		report_dataloader = cast(DataLoader, dataloaders[args.report_split])
		report_loss, report_probabilities, report_targets = collect_predictions(
			model,
			report_dataloader,
			criterion,
			device,
			progress_desc=f"Collecting {args.report_split} predictions",
			show_progress=show_progress,
		)
		default_report_metrics = evaluate_model(model, report_dataloader, criterion, device, default_threshold)
		tuned_report_metrics = evaluate_with_thresholds(
			probabilities=report_probabilities,
			targets=report_targets,
			thresholds=tuned_thresholds,
			label_names=NIH_CHEST_XRAY_LABELS,
			loss=report_loss,
			threshold_source=args.tune_split,
		)
		report_payload = {
			"split": args.report_split,
			"default_threshold": default_threshold,
			"default_metrics": default_report_metrics,
			"tuned_metrics": tuned_report_metrics,
		}

	summary = {
		"config": args.config,
		"checkpoint_path": str(checkpoint_path),
		"checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
		"best_val_loss": checkpoint.get("best_val_loss") if checkpoint else None,
		"device": str(device),
		"loss_name": str(config.get("training", {}).get("loss", "bce_with_logits")),
		"tune_split": args.tune_split,
		"report_split": args.report_split,
		"selection_metric": args.metric,
		"search_space": {
			"min_threshold": args.min_threshold,
			"max_threshold": args.max_threshold,
			"step": args.step,
			"num_candidates": int(threshold_grid.size),
		},
		"tuned_thresholds": per_label_summary,
		"tune_metrics": {
			"default_threshold": default_threshold,
			"default_metrics": default_tune_metrics,
			"tuned_metrics": tuned_tune_metrics,
		},
		"report_metrics": report_payload,
	}

	output_path = resolve_output_path(config, checkpoint_path, args.tune_split, args.report_split, args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	mlflow_config = configure_mlflow(config)
	with mlflow.start_run(run_name=f"{checkpoint_path.stem}_{args.tune_split}_threshold_tuning"):
		log_config_params(config)
		mlflow.log_param("threshold_tuning_checkpoint", str(checkpoint_path))
		mlflow.log_param("threshold_tuning_split", args.tune_split)
		mlflow.log_param("threshold_report_split", args.report_split)
		mlflow.log_param("threshold_search_min", args.min_threshold)
		mlflow.log_param("threshold_search_max", args.max_threshold)
		mlflow.log_param("threshold_search_step", args.step)
		mlflow.log_param("threshold_output_path", str(output_path))
		log_metrics(
			{
				"tune_default_macro_f1": summary["tune_metrics"]["default_metrics"]["macro_f1"],
				"tune_tuned_macro_f1": summary["tune_metrics"]["tuned_metrics"]["macro_f1"],
				"tune_default_micro_f1": summary["tune_metrics"]["default_metrics"]["micro_f1"],
				"tune_tuned_micro_f1": summary["tune_metrics"]["tuned_metrics"]["micro_f1"],
			}
		)
		if report_payload is not None:
			log_metrics(
				{
					"report_default_macro_f1": report_payload["default_metrics"]["macro_f1"],
					"report_tuned_macro_f1": report_payload["tuned_metrics"]["macro_f1"],
					"report_default_micro_f1": report_payload["default_metrics"]["micro_f1"],
					"report_tuned_micro_f1": report_payload["tuned_metrics"]["micro_f1"],
				}
			)
		log_dict_artifact("threshold_tuning_summary", summary)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(output_path), artifact_path="thresholds")

	print(f"Saved threshold tuning summary to {output_path}")


if __name__ == "__main__":
	main()