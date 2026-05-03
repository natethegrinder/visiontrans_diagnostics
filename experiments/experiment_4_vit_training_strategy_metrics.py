"""Pilot-first scratch ViT training-strategy comparison.

Start with the pilot variant or a tiny `--epochs-override` run. Full strategy
grids can be time-consuming on local hardware.
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the base ViT config.")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/experiment_4_vit_training_strategy_metrics.yaml",
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
    }

    if args.dry_run:
        print_dry_run_plan(experiment_config, variants, runtime_overrides)
        return

    summary_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []

    for variant in variants:
        variant_output_dir = output_dir / variant["name"]
        variant_output_dir.mkdir(parents=True, exist_ok=True)

        resolved_config = apply_overrides(base_config, variant.get("overrides", {}))
        resolved_config = apply_runtime_overrides(
            resolved_config,
            epochs_override=runtime_overrides["epochs_override"],
            max_train_batches=runtime_overrides["max_train_batches"],
            max_val_batches=runtime_overrides["max_val_batches"],
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
        peak_memory = max(entry["val_metrics"]["peak_gpu_memory_mb"] for entry in report["history"])

        summary_rows.append(
            build_variant_row(
                experiment_name=experiment_config["experiment_name"],
                variant=variant,
                report=report,
                resolved_config_path=resolved_config_path,
                extra={
                    "scheduler": resolved_config.get("training", {}).get("scheduler"),
                    "warmup_epochs": resolved_config.get("training", {}).get("warmup_epochs"),
                    "learning_rate": resolved_config.get("training", {}).get("learning_rate"),
                    "weight_decay": resolved_config.get("training", {}).get("weight_decay"),
                    "gradient_clip_norm": resolved_config.get("training", {}).get("gradient_clip_norm"),
                    **flatten_nested_metrics("train", best_train_metrics),
                    **flatten_nested_metrics("val", best_val_metrics),
                    "epoch_time_sec": best_val_metrics.get("epoch_time_sec"),
                    "peak_gpu_memory_mb": peak_memory,
                },
            )
        )
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

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_json(output_dir / "summary.json", summary_rows)
    write_summary_csv(output_dir / "per_label_metrics.csv", per_label_rows)
    print(f"Saved Experiment 4 outputs to {output_dir}")


if __name__ == "__main__":
    main()
