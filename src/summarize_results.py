from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import numpy as np
from matplotlib import pyplot as plt
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

matplotlib.use("Agg")

try:
	from .evaluate import apply_loss_overrides, load_config
	from .mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_path
except ImportError:
	from evaluate import apply_loss_overrides, load_config
	from mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Summarize MLflow training metrics and evaluation outputs into report JSON and plots."
	)
	parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Path to the YAML config file.")
	parser.add_argument("--all-losses", action="store_true", help="Generate report summaries for both BCE and focal runs, plus comparison outputs.")
	parser.add_argument("--train-run-id", default=None, help="Optional MLflow training run ID.")
	parser.add_argument("--train-run-name", default=None, help="Optional MLflow training run name.")
	parser.add_argument("--eval-json", default=None, help="Optional evaluation JSON path.")
	parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluation split to summarize.")
	parser.add_argument("--output-dir", default="outputs/reports", help="Directory for summary JSON and plots.")
	parser.add_argument("--loss", choices=["bce", "focal"], default=None, help="Optional loss override used to resolve the run name.")
	parser.add_argument("--focal-gamma", type=float, default=None, help="Optional focal loss gamma override.")
	parser.add_argument("--focal-alpha", type=float, default=None, help="Optional focal loss alpha override.")
	return parser


def _format_timestamp(timestamp_ms: int | None) -> str | None:
	if timestamp_ms is None:
		return None
	return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _metric_history(client: MlflowClient, run_id: str, metric_name: str) -> list[dict[str, float | int]]:
	history = client.get_metric_history(run_id, metric_name)
	return [
		{"step": int(metric.step), "value": float(metric.value)}
		for metric in sorted(history, key=lambda item: (item.step, item.timestamp, item.value))
	]


def _history_values(history: list[dict[str, float | int]]) -> list[float]:
	return [float(item["value"]) for item in history]


def _latest_metric_value(history: list[dict[str, float | int]]) -> float | None:
	if not history:
		return None
	return float(history[-1]["value"])


def _max_metric_value(history: list[dict[str, float | int]]) -> float | None:
	if not history:
		return None
	return max(_history_values(history))


def _coerce_float(value: Any) -> float | None:
	if value is None:
		return None
	return float(value)


def _load_config_with_loss(
	config_path: str,
	loss: str | None,
	focal_gamma: float | None,
	focal_alpha: float | None,
) -> dict[str, Any]:
	config = load_config(config_path)
	apply_loss_overrides(config, loss, focal_gamma, focal_alpha)
	return config


def resolve_training_run(client: MlflowClient, experiment_id: str, run_id: str | None, run_name: str) -> Run:
	if run_id is not None:
		return client.get_run(run_id)

	runs = client.search_runs(
		[experiment_id],
		filter_string=f"attributes.run_name = '{run_name}'",
		order_by=["attributes.start_time DESC"],
		max_results=20,
	)
	for run in runs:
		if run.data.metrics:
			return run
	if runs:
		return runs[0]
	raise FileNotFoundError(f"Could not find an MLflow training run named '{run_name}'")


def resolve_evaluation_json(run_name: str, split: str, override: str | None) -> Path:
	if override is not None:
		path = Path(override)
		if not path.exists():
			raise FileNotFoundError(f"Evaluation JSON not found: {path}")
		return path

	output_dir = Path("outputs") / "evaluations"
	exact_path = output_dir / f"{run_name}_{split}.json"
	if exact_path.exists():
		return exact_path

	pattern = f"{run_name}_{split}*.json"
	candidates = sorted(output_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
	if candidates:
		return candidates[0]
	raise FileNotFoundError(f"Could not locate evaluation JSON matching '{pattern}' under {output_dir}")


def load_evaluation_summary(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def collect_metric_histories(client: MlflowClient, run_id: str) -> dict[str, list[dict[str, float | int]]]:
	metric_names = [
		"train_loss",
		"val_loss",
		"val_f1_score",
		"train_epoch_time_sec",
		"val_epoch_time_sec",
		"epoch_time_sec",
		"gpu_peak_memory_allocated_mb",
		"gpu_peak_memory_reserved_mb",
	]
	return {
		metric_name: _metric_history(client, run_id, metric_name)
		for metric_name in metric_names
	}


def build_epoch_rows(metric_histories: dict[str, list[dict[str, float | int]]]) -> list[dict[str, float | int]]:
	steps = sorted({int(item["step"]) for history in metric_histories.values() for item in history})
	rows: list[dict[str, float | int]] = []
	for step in steps:
		row: dict[str, float | int] = {"epoch": step}
		for metric_name, history in metric_histories.items():
			for item in history:
				if int(item["step"]) == step:
					row[metric_name] = float(item["value"])
					break
		rows.append(row)
	return rows


def build_summary_payload(
	training_run: Run,
	evaluation_summary: dict[str, Any],
	evaluation_json_path: Path,
	metric_histories: dict[str, list[dict[str, float | int]]],
	f1_plot_path: Path,
	confusion_plot_path: Path,
) -> dict[str, Any]:
	epoch_rows = build_epoch_rows(metric_histories)
	train_loss_history = metric_histories["train_loss"]
	val_loss_history = metric_histories["val_loss"]
	val_f1_history = metric_histories["val_f1_score"]
	train_time_history = metric_histories["train_epoch_time_sec"]
	val_time_history = metric_histories["val_epoch_time_sec"]
	epoch_time_history = metric_histories["epoch_time_sec"]
	gpu_peak_allocated_history = metric_histories["gpu_peak_memory_allocated_mb"]
	gpu_peak_reserved_history = metric_histories["gpu_peak_memory_reserved_mb"]

	aggregate_confusion = np.zeros((2, 2), dtype=np.int64)
	for matrix in evaluation_summary["confusion_matrix"].values():
		aggregate_confusion += np.asarray(matrix, dtype=np.int64)

	return {
		"training_run": {
			"run_id": training_run.info.run_id,
			"run_name": training_run.info.run_name,
			"status": training_run.info.status,
			"artifact_uri": training_run.info.artifact_uri,
			"start_time_utc": _format_timestamp(training_run.info.start_time),
			"end_time_utc": _format_timestamp(training_run.info.end_time),
			"device": training_run.data.params.get("device"),
			"checkpoint_path": training_run.data.params.get("checkpoint_path"),
			"loss_name": training_run.data.params.get("training.loss"),
			"num_epochs_run": int(training_run.data.metrics.get("num_epochs_run", len(epoch_rows))),
			"best_epoch": int(training_run.data.metrics.get("best_epoch", -1)),
			"best_val_loss": float(training_run.data.metrics.get("best_val_loss", np.nan)),
		},
		"loss_summary": {
			"train_loss_start": _latest_metric_value(train_loss_history[:1]),
			"train_loss_end": _latest_metric_value(train_loss_history),
			"train_loss_min": min(_history_values(train_loss_history)) if train_loss_history else None,
			"val_loss_start": _latest_metric_value(val_loss_history[:1]),
			"val_loss_end": _latest_metric_value(val_loss_history),
			"val_loss_min": min(_history_values(val_loss_history)) if val_loss_history else None,
			"best_epoch": int(training_run.data.metrics.get("best_epoch", -1)),
		},
		"training_time_summary": {
			"total_run_time_sec": float(training_run.data.metrics.get("total_run_time_sec", 0.0)),
			"mean_epoch_time_sec": float(np.mean(_history_values(epoch_time_history))) if epoch_time_history else None,
			"mean_train_epoch_time_sec": float(np.mean(_history_values(train_time_history))) if train_time_history else None,
			"mean_val_epoch_time_sec": float(np.mean(_history_values(val_time_history))) if val_time_history else None,
		},
		"gpu_memory_summary": {
			"peak_allocated_mb": _max_metric_value(gpu_peak_allocated_history),
			"peak_reserved_mb": _max_metric_value(gpu_peak_reserved_history),
			"last_peak_allocated_mb": _latest_metric_value(gpu_peak_allocated_history),
			"last_peak_reserved_mb": _latest_metric_value(gpu_peak_reserved_history),
		},
		"evaluation_summary": {
			"source_json": str(evaluation_json_path.resolve()),
			"split": evaluation_summary.get("split"),
			"loss": evaluation_summary.get("loss"),
			"auroc": evaluation_summary.get("auroc"),
			"pr_auc": evaluation_summary.get("pr_auc"),
			"macro_f1": evaluation_summary.get("macro_f1"),
			"micro_f1": evaluation_summary.get("micro_f1"),
			"macro_precision": evaluation_summary.get("macro_precision"),
			"macro_recall": evaluation_summary.get("macro_recall"),
			"exact_match_accuracy": evaluation_summary.get("exact_match_accuracy"),
			"num_examples": evaluation_summary.get("num_examples"),
		},
		"confusion_matrix_summary": {
			"aggregate_multilabel_confusion_matrix": aggregate_confusion.astype(int).tolist(),
			"per_label_confusion_matrix": evaluation_summary["confusion_matrix"],
		},
		"curves": {
			"epoch_rows": epoch_rows,
			"train_loss": train_loss_history,
			"val_loss": val_loss_history,
			"val_f1_score": val_f1_history,
		},
		"artifacts": {
			"f1_vs_epoch_plot": str(f1_plot_path.resolve()),
			"confusion_matrix_plot": str(confusion_plot_path.resolve()),
		},
	}


def plot_f1_history(history: list[dict[str, float | int]], output_path: Path, run_name: str) -> None:
	if not history:
		raise ValueError("MLflow run does not contain any val_f1_score history.")

	epochs = [int(item["step"]) for item in history]
	values = [float(item["value"]) for item in history]

	plt.figure(figsize=(8, 5))
	plt.plot(epochs, values, marker="o", linewidth=2.0, color="#1f77b4")
	plt.title(f"Validation F1 vs Epoch\n{run_name}")
	plt.xlabel("Epoch")
	plt.ylabel("Validation F1")
	plt.xticks(epochs)
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_confusion_matrix(confusion_by_label: dict[str, list[list[int]]], output_path: Path, run_name: str, split: str) -> None:
	aggregate_confusion = np.zeros((2, 2), dtype=np.int64)
	for matrix in confusion_by_label.values():
		aggregate_confusion += np.asarray(matrix, dtype=np.int64)

	fig, ax = plt.subplots(figsize=(6, 5))
	image = ax.imshow(aggregate_confusion, cmap="Blues")
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
	ax.set_yticks([0, 1], labels=["True 0", "True 1"])
	ax.set_title(f"Aggregate Multilabel Confusion Matrix\n{run_name} ({split})")
	for row in range(2):
		for col in range(2):
			ax.text(col, row, f"{int(aggregate_confusion[row, col])}", ha="center", va="center", color="#111111")
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close(fig)


def plot_metric_comparison(comparison_payload: dict[str, Any], output_path: Path) -> None:
	metric_names = list(comparison_payload["overall"].keys())
	bce_values = [float(comparison_payload["overall"][name]["bce"]) for name in metric_names]
	focal_values = [float(comparison_payload["overall"][name]["focal"]) for name in metric_names]
	positions = np.arange(len(metric_names))
	width = 0.35

	plt.figure(figsize=(10, 5))
	plt.bar(positions - width / 2, bce_values, width=width, label="BCE", color="#1f77b4")
	plt.bar(positions + width / 2, focal_values, width=width, label="Focal", color="#ff7f0e")
	plt.xticks(positions, metric_names, rotation=20)
	plt.ylabel("Score")
	plt.title(f"BCE vs Focal Overall Metrics ({comparison_payload['split']})")
	plt.legend()
	plt.grid(True, axis="y", linestyle="--", alpha=0.3)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_f1_comparison(
	bce_history: list[dict[str, float | int]],
	focal_history: list[dict[str, float | int]],
	output_path: Path,
	split: str,
) -> None:
	if not bce_history or not focal_history:
		raise ValueError("Both BCE and focal runs must have val_f1_score history for comparison plotting.")

	bce_epochs = [int(item["step"]) for item in bce_history]
	bce_values = [float(item["value"]) for item in bce_history]
	focal_epochs = [int(item["step"]) for item in focal_history]
	focal_values = [float(item["value"]) for item in focal_history]

	plt.figure(figsize=(8, 5))
	plt.plot(bce_epochs, bce_values, marker="o", linewidth=2.0, color="#1f77b4", label="BCE")
	plt.plot(focal_epochs, focal_values, marker="s", linewidth=2.0, color="#ff7f0e", label="Focal")
	plt.xlabel("Epoch")
	plt.ylabel("Validation F1")
	plt.title(f"BCE vs Focal Validation F1 ({split})")
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def log_single_report_to_mlflow(
	training_run: Run,
	evaluation_json_path: Path,
	json_path: Path,
	f1_plot_path: Path,
	confusion_plot_path: Path,
	summary: dict[str, Any],
	mlflow_config: dict[str, Any],
) -> None:
	with mlflow.start_run(run_name=f"{training_run.info.run_name}_{summary['evaluation_summary']['split']}_report_summary"):
		mlflow.log_param("source_training_run_id", training_run.info.run_id)
		mlflow.log_param("source_training_run_name", training_run.info.run_name)
		mlflow.log_param("source_evaluation_json", str(evaluation_json_path))
		mlflow.log_param("report_output_json", str(json_path))
		log_metrics(
			{
				"report_auroc": summary["evaluation_summary"]["auroc"],
				"report_macro_f1": summary["evaluation_summary"]["macro_f1"],
				"report_micro_f1": summary["evaluation_summary"]["micro_f1"],
				"report_total_run_time_sec": summary["training_time_summary"]["total_run_time_sec"],
				"report_peak_allocated_mb": summary["gpu_memory_summary"]["peak_allocated_mb"],
				"report_best_val_loss": summary["training_run"]["best_val_loss"],
			}
		)
		log_dict_artifact(f"{training_run.info.run_name}_{summary['evaluation_summary']['split']}_report_summary", summary)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(json_path), artifact_path="reports")
			mlflow.log_artifact(str(f1_plot_path), artifact_path="reports")
			mlflow.log_artifact(str(confusion_plot_path), artifact_path="reports")


def generate_single_report(
	config_path: str,
	output_dir: Path,
	split: str,
	loss: str | None,
	focal_gamma: float | None,
	focal_alpha: float | None,
	train_run_id: str | None,
	train_run_name: str | None,
	eval_json: str | None,
) -> dict[str, Any]:
	config = _load_config_with_loss(config_path, loss, focal_gamma, focal_alpha)
	mlflow_config = configure_mlflow(config)
	experiment_name = str(mlflow_config.get("experiment_name", "default"))
	experiment = mlflow.get_experiment_by_name(experiment_name)
	if experiment is None:
		raise FileNotFoundError(f"Could not find MLflow experiment '{experiment_name}'")

	resolved_run_name = train_run_name or str(config.get("run", {}).get("name", "cnn_resnet18_nih_baseline"))
	client = MlflowClient()
	training_run = resolve_training_run(client, experiment.experiment_id, train_run_id, resolved_run_name)
	evaluation_json_path = resolve_evaluation_json(resolved_run_name, split, eval_json)
	evaluation_summary = load_evaluation_summary(evaluation_json_path)
	metric_histories = collect_metric_histories(client, training_run.info.run_id)

	f1_plot_path = resolve_nonconflicting_path(output_dir / f"{resolved_run_name}_{split}_f1_vs_epoch.png")
	confusion_plot_path = resolve_nonconflicting_path(output_dir / f"{resolved_run_name}_{split}_confusion_matrix.png")
	plot_f1_history(metric_histories["val_f1_score"], f1_plot_path, resolved_run_name)
	plot_confusion_matrix(evaluation_summary["confusion_matrix"], confusion_plot_path, resolved_run_name, split)

	summary = build_summary_payload(
		training_run=training_run,
		evaluation_summary=evaluation_summary,
		evaluation_json_path=evaluation_json_path,
		metric_histories=metric_histories,
		f1_plot_path=f1_plot_path,
		confusion_plot_path=confusion_plot_path,
	)

	json_path = resolve_nonconflicting_path(output_dir / f"{resolved_run_name}_{split}_report_summary.json")
	json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	log_single_report_to_mlflow(
		training_run=training_run,
		evaluation_json_path=evaluation_json_path,
		json_path=json_path,
		f1_plot_path=f1_plot_path,
		confusion_plot_path=confusion_plot_path,
		summary=summary,
		mlflow_config=mlflow_config,
	)

	return {
		"loss": loss or str(summary["training_run"].get("loss_name") or resolved_run_name).lower(),
		"training_run": training_run,
		"run_name": resolved_run_name,
		"summary": summary,
		"json_path": json_path,
		"f1_plot_path": f1_plot_path,
		"confusion_plot_path": confusion_plot_path,
		"evaluation_json_path": evaluation_json_path,
		"metric_histories": metric_histories,
	}


def build_comparison_payload(reports_by_loss: dict[str, dict[str, Any]], split: str) -> dict[str, Any]:
	bce_report = reports_by_loss["bce"]
	focal_report = reports_by_loss["focal"]
	bce_summary = bce_report["summary"]
	focal_summary = focal_report["summary"]

	overall_metric_names = [
		"auroc",
		"pr_auc",
		"macro_f1",
		"micro_f1",
		"macro_precision",
		"macro_recall",
		"exact_match_accuracy",
	]
	overall: dict[str, dict[str, float]] = {}
	for metric_name in overall_metric_names:
		bce_value = float(bce_summary["evaluation_summary"][metric_name])
		focal_value = float(focal_summary["evaluation_summary"][metric_name])
		overall[metric_name] = {
			"bce": bce_value,
			"focal": focal_value,
			"delta_focal_minus_bce": focal_value - bce_value,
		}

	training_metrics = {
		"best_val_loss": {
			"bce": float(bce_summary["training_run"]["best_val_loss"]),
			"focal": float(focal_summary["training_run"]["best_val_loss"]),
		},
		"total_run_time_sec": {
			"bce": float(bce_summary["training_time_summary"]["total_run_time_sec"]),
			"focal": float(focal_summary["training_time_summary"]["total_run_time_sec"]),
		},
		"peak_allocated_mb": {
			"bce": float(bce_summary["gpu_memory_summary"]["peak_allocated_mb"]),
			"focal": float(focal_summary["gpu_memory_summary"]["peak_allocated_mb"]),
		},
	}
	for metric_name, values in training_metrics.items():
		values["delta_focal_minus_bce"] = float(values["focal"] - values["bce"])

	return {
		"split": split,
		"runs": {
			"bce": {
				"run_id": bce_summary["training_run"]["run_id"],
				"run_name": bce_summary["training_run"]["run_name"],
				"report_json": str(Path(bce_report["json_path"]).resolve()),
			},
			"focal": {
				"run_id": focal_summary["training_run"]["run_id"],
				"run_name": focal_summary["training_run"]["run_name"],
				"report_json": str(Path(focal_report["json_path"]).resolve()),
			},
		},
		"overall": overall,
		"training": training_metrics,
	}


def log_comparison_to_mlflow(
	comparison_payload: dict[str, Any],
	json_path: Path,
	metrics_plot_path: Path,
	f1_plot_path: Path,
	mlflow_config: dict[str, Any],
) -> None:
	with mlflow.start_run(run_name=f"bce_vs_focal_{comparison_payload['split']}_report_summary"):
		mlflow.log_param("bce_report_json", comparison_payload["runs"]["bce"]["report_json"])
		mlflow.log_param("focal_report_json", comparison_payload["runs"]["focal"]["report_json"])
		mlflow.log_param("comparison_output_json", str(json_path))
		log_metrics(
			{
				f"report_compare_{metric_name}_delta": values["delta_focal_minus_bce"]
				for metric_name, values in comparison_payload["overall"].items()
			}
		)
		log_dict_artifact(f"bce_vs_focal_{comparison_payload['split']}_report_summary", comparison_payload)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(json_path), artifact_path="reports")
			mlflow.log_artifact(str(metrics_plot_path), artifact_path="reports")
			mlflow.log_artifact(str(f1_plot_path), artifact_path="reports")


def generate_all_loss_reports(args: argparse.Namespace) -> list[str]:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	reports_by_loss: dict[str, dict[str, Any]] = {}
	for loss_name in ("bce", "focal"):
		reports_by_loss[loss_name] = generate_single_report(
			config_path=args.config,
			output_dir=output_dir,
			split=args.split,
			loss=loss_name,
			focal_gamma=args.focal_gamma,
			focal_alpha=args.focal_alpha,
			train_run_id=None,
			train_run_name=None,
			eval_json=None,
		)

	base_config = _load_config_with_loss(args.config, "bce", args.focal_gamma, args.focal_alpha)
	mlflow_config = configure_mlflow(base_config)
	comparison_payload = build_comparison_payload(reports_by_loss, args.split)
	metrics_plot_path = resolve_nonconflicting_path(output_dir / f"bce_vs_focal_{args.split}_report_metrics.png")
	f1_plot_path = resolve_nonconflicting_path(output_dir / f"bce_vs_focal_{args.split}_f1_vs_epoch.png")
	plot_metric_comparison(comparison_payload, metrics_plot_path)
	plot_f1_comparison(
		reports_by_loss["bce"]["metric_histories"]["val_f1_score"],
		reports_by_loss["focal"]["metric_histories"]["val_f1_score"],
		f1_plot_path,
		args.split,
	)

	comparison_payload = deepcopy(comparison_payload)
	comparison_payload["artifacts"] = {
		"metrics_comparison_plot": str(metrics_plot_path.resolve()),
		"f1_comparison_plot": str(f1_plot_path.resolve()),
	}
	json_path = resolve_nonconflicting_path(output_dir / f"bce_vs_focal_{args.split}_report_summary.json")
	json_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
	log_comparison_to_mlflow(comparison_payload, json_path, metrics_plot_path, f1_plot_path, mlflow_config)

	return [
		f"Saved BCE report JSON to {reports_by_loss['bce']['json_path']}",
		f"Saved focal report JSON to {reports_by_loss['focal']['json_path']}",
		f"Saved comparison JSON to {json_path}",
		f"Saved overall metric comparison plot to {metrics_plot_path}",
		f"Saved F1 comparison plot to {f1_plot_path}",
	]


def main() -> None:
	args = build_arg_parser().parse_args()
	if args.all_losses:
		for message in generate_all_loss_reports(args):
			print(message)
		return

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	report = generate_single_report(
		config_path=args.config,
		output_dir=output_dir,
		split=args.split,
		loss=args.loss,
		focal_gamma=args.focal_gamma,
		focal_alpha=args.focal_alpha,
		train_run_id=args.train_run_id,
		train_run_name=args.train_run_name,
		eval_json=args.eval_json,
	)

	print(f"Saved report JSON to {report['json_path']}")
	print(f"Saved F1 plot to {report['f1_plot_path']}")
	print(f"Saved confusion matrix plot to {report['confusion_plot_path']}")


if __name__ == "__main__":
	main()