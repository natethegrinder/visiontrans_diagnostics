"""Compare scratch ViT training strategies without changing the baseline architecture."""

from __future__ import annotations

import argparse

import mlflow

from common import (
    REPO_ROOT,
    best_history_entry,
    build_per_label_rows,
    copy_config_with_updates,
    ensure_output_dir,
    history_to_rows,
    save_json,
    save_rows_csv,
)
from mlflow_utils import configure_mlflow
from train import load_config, train_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the ViT baseline config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs. Defaults to outputs/experiments/experiment_4_vit_training_strategy_metrics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_config = load_config(REPO_ROOT / args.config)
    output_dir = ensure_output_dir("experiment_4_vit_training_strategy_metrics", args.output_dir)

    strategies = [
        ("default", {}),
        ("cosine_no_warmup", {"training": {"scheduler": "cosine", "warmup_epochs": 0}}),
        ("warmup_cosine_grad_clip_1_0", {"training": {"scheduler": "warmup_cosine", "gradient_clip_norm": 1.0}}),
    ]

    configure_mlflow(base_config)
    summary_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []
    report_index: dict[str, object] = {}

    parent_run_name = base_config.get("run", {}).get("name", "vit_baseline") + "_experiment_4"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_param("experiment_script", "experiment_4_vit_training_strategy_metrics")

        for strategy_name, updates in strategies:
            strategy_config = copy_config_with_updates(base_config, updates)
            strategy_checkpoint_dir = output_dir / "checkpoints" / strategy_name
            strategy_config = copy_config_with_updates(
                strategy_config,
                {
                    "run": {"name": f"{base_config.get('run', {}).get('name', 'vit_baseline')}_{strategy_name}"},
                    "training": {"checkpoint_dir": str(strategy_checkpoint_dir)},
                },
            )
            report = train_model(strategy_config, run_name=strategy_config["run"]["name"], nested_run=True)
            report_index[strategy_name] = report

            best_entry = best_history_entry(report, criterion="best_auc")
            total_training_time = sum(entry["train_metrics"]["epoch_time_sec"] for entry in report["history"])
            peak_memory = max(entry["val_metrics"]["peak_gpu_memory_mb"] for entry in report["history"])
            val_metrics = best_entry["val_metrics"]

            summary_row = {
                "strategy": strategy_name,
                "run_id": report["run_id"],
                "scheduler": strategy_config.get("training", {}).get("scheduler"),
                "warmup_epochs": strategy_config.get("training", {}).get("warmup_epochs", 0),
                "gradient_clip_norm": strategy_config.get("training", {}).get("gradient_clip_norm"),
                "best_auc_epoch": report["best_auc_epoch"],
                "best_macro_f1_epoch": report["best_macro_f1_epoch"],
                "train_loss": best_entry["train_metrics"]["loss"],
                "val_loss": val_metrics["loss"],
                "mean_auc": val_metrics["mean_auc"],
                "mean_average_precision": val_metrics["mean_average_precision"],
                "macro_f1": val_metrics["macro_f1"],
                "micro_f1": val_metrics["micro_f1"],
                "macro_precision": val_metrics["macro_precision"],
                "macro_recall": val_metrics["macro_recall"],
                "exact_match_accuracy": val_metrics["exact_match_accuracy"],
                "mean_binary_accuracy": val_metrics["mean_binary_accuracy"],
                "training_time_sec": total_training_time,
                "peak_gpu_memory_mb": peak_memory,
            }
            summary_rows.append(summary_row)
            per_label_rows.extend(
                build_per_label_rows(val_metrics, report["label_names"], extra={"strategy": strategy_name})
            )

            save_rows_csv(output_dir / f"{strategy_name}_epoch_history.csv", history_to_rows(report["history"]))

    save_rows_csv(output_dir / "strategy_comparison.csv", summary_rows)
    save_rows_csv(output_dir / "strategy_per_label_metrics.csv", per_label_rows)
    save_json(output_dir / "strategy_comparison.json", summary_rows)

    print(f"Saved Experiment 4 outputs to {output_dir}")


if __name__ == "__main__":
    main()
