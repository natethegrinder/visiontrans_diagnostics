"""Train the scratch ViT baseline and export core multilabel metrics."""

from __future__ import annotations

import argparse

from common import (
    REPO_ROOT,
    best_history_entry,
    build_per_label_rows,
    ensure_output_dir,
    history_to_rows,
    save_json,
    save_rows_csv,
)
from train import load_config, train_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the ViT baseline config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs. Defaults to outputs/experiments/experiment_1_vit_baseline_metrics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(REPO_ROOT / args.config)
    output_dir = ensure_output_dir("experiment_1_vit_baseline_metrics", args.output_dir)

    report = train_model(
        config,
        run_name=config.get("run", {}).get("name", "vit_baseline") + "_experiment_1",
    )

    best_entry = best_history_entry(report, criterion="best_auc")
    final_entry = report["history"][-1]
    label_names = report["label_names"]

    summary = {
        "run_id": report["run_id"],
        "device": report["device"],
        "best_auc_epoch": report["best_auc_epoch"],
        "best_macro_f1_epoch": report["best_macro_f1_epoch"],
        "best_auc_checkpoint": report["best_auc_checkpoint"],
        "best_macro_f1_checkpoint": report["best_macro_f1_checkpoint"],
        "best_val_metrics": best_entry["val_metrics"],
        "final_train_metrics": final_entry["train_metrics"],
        "final_val_metrics": final_entry["val_metrics"],
    }

    save_json(output_dir / "summary.json", summary)
    save_rows_csv(output_dir / "epoch_history.csv", history_to_rows(report["history"]))
    save_rows_csv(
        output_dir / "best_val_per_label_metrics.csv",
        build_per_label_rows(best_entry["val_metrics"], label_names),
    )

    print(f"Saved Experiment 1 outputs to {output_dir}")


if __name__ == "__main__":
    main()
