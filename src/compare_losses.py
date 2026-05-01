from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow

try:
	from .data import NIH_CHEST_XRAY_LABELS
	from .evaluate import load_config
	from .mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_path
except ImportError:
	from data import NIH_CHEST_XRAY_LABELS
	from evaluate import load_config
	from mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Compare BCE and focal-loss evaluation results for NIH multilabel classification.")
	parser.add_argument("--bce", default=None, help="Path to the BCE evaluation JSON file.")
	parser.add_argument("--focal", default=None, help="Path to the focal-loss evaluation JSON file.")
	parser.add_argument("--output-dir", default="outputs/comparisons", help="Directory for comparison outputs.")
	parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Config file used for MLflow settings.")
	return parser


def resolve_default_result(loss_name: str) -> Path:
	output_dir = Path("outputs") / "evaluations"
	default_candidates = {
		"bce": [
			output_dir / "cnn_resnet18_nih_baseline_bce_test.json",
			output_dir / "cnn_resnet18_nih_baseline_test.json",
		],
		"focal": [output_dir / "cnn_resnet18_nih_baseline_focal_test.json"],
	}
	for candidate in default_candidates[loss_name]:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Could not locate a default {loss_name} evaluation result under {output_dir}")


def load_metrics(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def build_overall_comparison(bce_metrics: dict[str, Any], focal_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
	metric_names = ["auroc", "pr_auc", "macro_f1", "micro_f1", "precision", "recall"]
	comparison: dict[str, dict[str, float]] = {}
	for metric_name in metric_names:
		bce_value = float(bce_metrics[metric_name])
		focal_value = float(focal_metrics[metric_name])
		comparison[metric_name] = {
			"bce": bce_value,
			"focal": focal_value,
			"delta_focal_minus_bce": focal_value - bce_value,
		}
	return comparison


def build_per_label_comparison(bce_metrics: dict[str, Any], focal_metrics: dict[str, Any]) -> list[dict[str, float | str]]:
	rows: list[dict[str, float | str]] = []
	for label in NIH_CHEST_XRAY_LABELS:
		bce_auroc = float(bce_metrics["per_class_auroc"][label])
		focal_auroc = float(focal_metrics["per_class_auroc"][label])
		bce_f1 = float(bce_metrics["per_class_f1"][label])
		focal_f1 = float(focal_metrics["per_class_f1"][label])
		rows.append(
			{
				"label": label,
				"bce_auroc": bce_auroc,
				"focal_auroc": focal_auroc,
				"delta_auroc": focal_auroc - bce_auroc,
				"bce_f1": bce_f1,
				"focal_f1": focal_f1,
				"delta_f1": focal_f1 - bce_f1,
			}
		)
	return rows


def build_markdown_report(summary: dict[str, Any]) -> str:
	lines = [
		"# BCE vs Focal Comparison",
		"",
		"## Overall Metrics",
		"",
		"| Metric | BCE | Focal | Delta (Focal - BCE) |",
		"| --- | ---: | ---: | ---: |",
	]
	for metric_name, values in summary["overall"].items():
		lines.append(
			f"| {metric_name} | {values['bce']:.6f} | {values['focal']:.6f} | {values['delta_focal_minus_bce']:.6f} |"
		)

	lines.extend(
		[
			"",
			"## Per-label AUROC and F1",
			"",
			"| Label | BCE AUROC | Focal AUROC | Delta AUROC | BCE F1 | Focal F1 | Delta F1 |",
			"| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
		]
	)
	for row in summary["per_label"]:
		lines.append(
			"| {label} | {bce_auroc:.6f} | {focal_auroc:.6f} | {delta_auroc:.6f} | {bce_f1:.6f} | {focal_f1:.6f} | {delta_f1:.6f} |".format(**row)
		)
	return "\n".join(lines) + "\n"


def main() -> None:
	args = build_arg_parser().parse_args()
	bce_path = Path(args.bce) if args.bce is not None else resolve_default_result("bce")
	focal_path = Path(args.focal) if args.focal is not None else resolve_default_result("focal")
	bce_metrics = load_metrics(bce_path)
	focal_metrics = load_metrics(focal_path)

	summary = {
		"bce_path": str(bce_path.resolve()),
		"focal_path": str(focal_path.resolve()),
		"overall": build_overall_comparison(bce_metrics, focal_metrics),
		"per_label": build_per_label_comparison(bce_metrics, focal_metrics),
	}

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	json_path = resolve_nonconflicting_path(output_dir / "bce_vs_focal_comparison.json")
	markdown_path = resolve_nonconflicting_path(output_dir / "bce_vs_focal_comparison.md")
	json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	markdown_path.write_text(build_markdown_report(summary), encoding="utf-8")

	config = load_config(args.config)
	mlflow_config = configure_mlflow(config)
	with mlflow.start_run(run_name="bce_vs_focal_comparison"):
		mlflow.log_param("bce_result_path", str(bce_path))
		mlflow.log_param("focal_result_path", str(focal_path))
		log_metrics(
			{
				f"compare_{metric_name}_delta": values["delta_focal_minus_bce"]
				for metric_name, values in summary["overall"].items()
			}
		)
		log_dict_artifact("bce_vs_focal_comparison", summary)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(json_path), artifact_path="comparisons")
			mlflow.log_artifact(str(markdown_path), artifact_path="comparisons")

	print(f"Saved comparison JSON to {json_path}")
	print(f"Saved comparison Markdown to {markdown_path}")


if __name__ == "__main__":
	main()