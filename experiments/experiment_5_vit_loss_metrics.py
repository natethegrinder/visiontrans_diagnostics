"""Pilot-first scratch ViT loss-function comparison.

Use the pilot variant or a tiny epoch override first. Full focal-loss grids can
take a long time, so screen only a few variants before any longer run.
"""

from __future__ import annotations

import argparse

from common import best_history_entry, build_per_label_rows, history_to_rows
from train import load_config, train_model
from utils import (
    REPO_ROOT,
    apply_overrides,
    apply_runtime_overrides,
    build_variant_row,
    ensure_output_dir,
    flatten_nested_metrics,
    load_experiment_config,
    prepare_variants,
    print_dry_run_plan,
    write_resolved_config,
    write_summary_csv,
    write_summary_json,
)


RARE_LABELS = ("Hernia", "Pneumonia", "Fibrosis", "Cardiomegaly")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the base ViT config.")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/experiment_5_vit_loss_metrics.yaml",
        help="Path to the experiment-specific YAML grid.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional experiment output directory override.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned variants without training.")
    parser.add_argument("--only", default=None, help="Run only the named variant from the experiment YAML.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of variants to run.")
    parser.add_argument("--epochs-override", type=int, default=None, help="Override training.epochs for cheap pilots.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for pilot train batches.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for pilot val batches.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_config = load_config(REPO_ROOT / args.config)
    experiment_config = load_experiment_config(REPO_ROOT / args.experiment_config)
    output_dir = ensure_output_dir(experiment_config["default_output_dir"], args.output_dir)

    runtime_defaults = experiment_config.get("runtime_defaults", {})
    selected_only = args.only
    if not args.dry_run and selected_only is None and args.max_runs is None:
        selected_only = experiment_config.get("default_pilot_variant")
    variants = prepare_variants(
        experiment_config,
        only=selected_only,
        max_runs=args.max_runs if args.max_runs is not None else runtime_defaults.get("max_runs"),
    )
    runtime_overrides = {
        "epochs_override": args.epochs_override if args.epochs_override is not None else runtime_defaults.get("epochs_override"),
        "max_train_batches": (
            args.max_train_batches if args.max_train_batches is not None else runtime_defaults.get("max_train_batches")
        ),
        "max_val_batches": args.max_val_batches if args.max_val_batches is not None else runtime_defaults.get("max_val_batches"),
        "num_workers_override": runtime_defaults.get("num_workers_override"),
    }

    if args.dry_run:
        print_dry_run_plan(experiment_config, variants, runtime_overrides)
        return

    summary_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []
    best_metrics_by_variant: dict[str, dict[str, float]] = {}

    for variant in variants:
        variant_output_dir = output_dir / variant["name"]
        variant_output_dir.mkdir(parents=True, exist_ok=True)

        resolved_config = apply_overrides(base_config, variant.get("overrides", {}))
        resolved_config = apply_runtime_overrides(
            resolved_config,
            epochs_override=runtime_overrides["epochs_override"],
            max_train_batches=runtime_overrides["max_train_batches"],
            max_val_batches=runtime_overrides["max_val_batches"],
            num_workers_override=runtime_overrides["num_workers_override"],
        )
        resolved_config.setdefault("run", {})
        resolved_config["run"]["name"] = f"{experiment_config['experiment_name']}__{variant['name']}"
        resolved_config.setdefault("training", {})
        resolved_config["training"]["checkpoint_dir"] = str(variant_output_dir / "checkpoints")

        resolved_config_path = variant_output_dir / "resolved_config.yaml"
        write_resolved_config(resolved_config_path, resolved_config)

        report = train_model(resolved_config, run_name=resolved_config["run"]["name"], nested_run=False)
        best_entry = best_history_entry(report, criterion="best_auc")
        best_val_metrics = dict(best_entry["val_metrics"])
        best_train_metrics = dict(best_entry["train_metrics"])
        best_metrics_by_variant[variant["name"]] = dict(best_val_metrics)
        peak_memory = max(entry["val_metrics"]["peak_gpu_memory_mb"] for entry in report["history"])

        summary_row = build_variant_row(
            experiment_name=experiment_config["experiment_name"],
            variant=variant,
            report=report,
            resolved_config_path=resolved_config_path,
            extra={
                "loss_function_name": resolved_config.get("training", {}).get("loss"),
                "use_pos_weight": resolved_config.get("training", {}).get("use_pos_weight"),
                "pos_weight_clamp": resolved_config.get("data", {}).get("pos_weight_clamp"),
                "focal_gamma": resolved_config.get("training", {}).get("focal_gamma"),
                "focal_alpha": resolved_config.get("training", {}).get("focal_alpha"),
                **flatten_nested_metrics("train", best_train_metrics),
                **flatten_nested_metrics("val", best_val_metrics),
                "epoch_time_sec": best_val_metrics.get("epoch_time_sec"),
                "peak_gpu_memory_mb": peak_memory,
            },
        )
        for label_name in RARE_LABELS:
            summary_row[f"val_auc_{label_name}"] = best_val_metrics.get(f"auc_{label_name}")
            summary_row[f"val_average_precision_{label_name}"] = best_val_metrics.get(
                f"average_precision_{label_name}"
            )
            summary_row[f"val_f1_{label_name}"] = best_val_metrics.get(f"f1_{label_name}")
            summary_row[f"val_precision_{label_name}"] = best_val_metrics.get(f"precision_{label_name}")
            summary_row[f"val_recall_{label_name}"] = best_val_metrics.get(f"recall_{label_name}")
        summary_rows.append(summary_row)

        per_label_rows.extend(
            build_per_label_rows(
                best_val_metrics,
                report["label_names"],
                extra={
                    "experiment_name": experiment_config["experiment_name"],
                    "variant_name": variant["name"],
                    "resolved_config_path": str(resolved_config_path),
                },
            )
        )

        write_summary_csv(variant_output_dir / "epoch_history.csv", history_to_rows(report["history"]))
        write_summary_json(
            variant_output_dir / "summary.json",
            {
                "experiment_name": experiment_config["experiment_name"],
                "variant_name": variant["name"],
                "variant_intent": variant.get("intent"),
                "overrides": variant.get("overrides", {}),
                "resolved_config_path": str(resolved_config_path),
                "report": report,
            },
        )

    baseline_variant = "bce_pos_weight_clamp50"
    if baseline_variant not in best_metrics_by_variant and "bce_pos_weight_clamp50_pilot" in best_metrics_by_variant:
        baseline_variant = "bce_pos_weight_clamp50_pilot"
    baseline_metrics = best_metrics_by_variant.get(baseline_variant)

    for row in per_label_rows:
        label_name = row["label"]
        if baseline_metrics is None:
            row["auc_delta_vs_bce_pos_weight"] = None
            row["average_precision_delta_vs_bce_pos_weight"] = None
            continue
        baseline_auc = baseline_metrics.get(f"auc_{label_name}")
        baseline_ap = baseline_metrics.get(f"average_precision_{label_name}")
        current_auc = row["auc"]
        current_ap = row["average_precision"]
        row["auc_delta_vs_bce_pos_weight"] = None if baseline_auc is None or current_auc is None else current_auc - baseline_auc
        row["average_precision_delta_vs_bce_pos_weight"] = (
            None if baseline_ap is None or current_ap is None else current_ap - baseline_ap
        )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_json(output_dir / "summary.json", summary_rows)
    write_summary_csv(output_dir / "per_label_metrics.csv", per_label_rows)
    print(f"Saved Experiment 5 outputs to {output_dir}")


if __name__ == "__main__":
    main()
