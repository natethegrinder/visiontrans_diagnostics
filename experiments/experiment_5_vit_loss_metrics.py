"""Compare class-imbalance-aware loss functions for the scratch ViT baseline."""

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


RARE_LABELS = ("Hernia", "Pneumonia", "Fibrosis", "Cardiomegaly")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the ViT baseline config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs. Defaults to outputs/experiments/experiment_5_vit_loss_metrics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_config = load_config(REPO_ROOT / args.config)
    output_dir = ensure_output_dir("experiment_5_vit_loss_metrics", args.output_dir)

    loss_variants = [
        ("bce_with_pos_weight", {"training": {"loss": "bce_with_logits", "use_pos_weight": True}}),
        (
            "focal",
            {
                "training": {
                    "loss": "focal",
                    "use_pos_weight": False,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.25,
                }
            },
        ),
        (
            "focal_with_pos_weight",
            {
                "training": {
                    "loss": "focal_with_pos_weight",
                    "use_pos_weight": True,
                    "focal_gamma": 2.0,
                    "focal_alpha": 0.25,
                }
            },
        ),
    ]

    configure_mlflow(base_config)
    summary_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []
    best_metrics_by_variant: dict[str, dict[str, float]] = {}

    parent_run_name = base_config.get("run", {}).get("name", "vit_baseline") + "_experiment_5"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_param("experiment_script", "experiment_5_vit_loss_metrics")

        for variant_name, updates in loss_variants:
            variant_config = copy_config_with_updates(base_config, updates)
            variant_checkpoint_dir = output_dir / "checkpoints" / variant_name
            variant_config = copy_config_with_updates(
                variant_config,
                {
                    "run": {"name": f"{base_config.get('run', {}).get('name', 'vit_baseline')}_{variant_name}"},
                    "training": {"checkpoint_dir": str(variant_checkpoint_dir)},
                },
            )
            report = train_model(variant_config, run_name=variant_config["run"]["name"], nested_run=True)
            best_entry = best_history_entry(report, criterion="best_auc")
            val_metrics = best_entry["val_metrics"]
            best_metrics_by_variant[variant_name] = dict(val_metrics)

            summary_row = {
                "loss_variant": variant_name,
                "run_id": report["run_id"],
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
            }
            for label_name in RARE_LABELS:
                summary_row[f"auc_{label_name}"] = val_metrics.get(f"auc_{label_name}")
                summary_row[f"average_precision_{label_name}"] = val_metrics.get(f"average_precision_{label_name}")
                summary_row[f"f1_{label_name}"] = val_metrics.get(f"f1_{label_name}")
                summary_row[f"precision_{label_name}"] = val_metrics.get(f"precision_{label_name}")
                summary_row[f"recall_{label_name}"] = val_metrics.get(f"recall_{label_name}")
            summary_rows.append(summary_row)
            per_label_rows.extend(
                build_per_label_rows(val_metrics, report["label_names"], extra={"loss_variant": variant_name})
            )

            save_rows_csv(output_dir / f"{variant_name}_epoch_history.csv", history_to_rows(report["history"]))

    baseline_metrics = best_metrics_by_variant["bce_with_pos_weight"]
    for row in per_label_rows:
        label_name = row["label"]
        baseline_auc = baseline_metrics.get(f"auc_{label_name}")
        current_auc = row["auc"]
        row["auc_delta_vs_bce_pos_weight"] = (
            None if baseline_auc is None or current_auc is None else current_auc - baseline_auc
        )

    save_rows_csv(output_dir / "loss_comparison.csv", summary_rows)
    save_rows_csv(output_dir / "loss_per_label_metrics.csv", per_label_rows)
    save_json(output_dir / "loss_comparison.json", summary_rows)

    print(f"Saved Experiment 5 outputs to {output_dir}")


if __name__ == "__main__":
    main()
