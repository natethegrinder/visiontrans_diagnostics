"""Final fixed-parameter ViT baseline experiment.

This script is intentionally separate from the earlier screening experiments.
Use `--dry-run` or a short smoke test before launching the full 40-epoch run,
because the complete fixed-baseline training job can still be time-consuming.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlflow.tracking import MlflowClient

from common import best_history_entry, build_confusion_rows, build_per_label_rows, history_to_rows
from train import load_config, train_model
from utils import (
    REPO_ROOT,
    apply_overrides,
    apply_runtime_overrides,
    ensure_output_dir,
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
        default="configs/experiments/experiment_final_vit_baseline_fixed.yaml",
        help="Path to the fixed-baseline experiment YAML.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved fixed run plan without training.")
    parser.add_argument("--only", default=None, help="Optional variant override; normally keep the single fixed variant.")
    parser.add_argument("--epochs-override", type=int, default=None, help="Override training.epochs for smoke tests.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batch cap for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional val batch cap for smoke tests.")
    return parser


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# def _build_attention_handoff_command(output_dir: Path) -> str:
#     checkpoint_path = _relative_to_repo(output_dir / "checkpoints" / "vit_best_auc.pt")
#     attention_output_dir = _relative_to_repo(output_dir / "attention")
#     return "\n".join(
#         [
#             "python experiments/experiment_3_vit_attention_metrics.py \\",
#             "  --config configs/vit_baseline.yaml \\",
#             "  --experiment-config configs/experiments/experiment_3_vit_attention_metrics.yaml \\",
#             f"  --checkpoint {checkpoint_path} \\",
#             "  --only last_layer_mean_top20_pilot \\",
#             "  --max-val-batches 100 \\",
#             "  --max-attention-batches 5 \\",
#             "  --max-heatmaps 20 \\",
#             f"  --output-dir {attention_output_dir}",
#         ]
#     )


def _log_artifacts(run_id: str | None, artifact_paths: list[Path]) -> None:
    if not run_id:
        return
    client = MlflowClient()
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            client.log_artifact(run_id, str(artifact_path))


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"[Experiment] Loading base config: {REPO_ROOT / args.config}", flush=True)
    base_config = load_config(REPO_ROOT / args.config)
    print(f"[Experiment] Loading experiment config: {REPO_ROOT / args.experiment_config}", flush=True)
    experiment_config = load_experiment_config(REPO_ROOT / args.experiment_config)
    output_dir = ensure_output_dir(experiment_config["default_output_dir"], args.output_dir)
    print(f"[Experiment] Output directory: {output_dir}", flush=True)

    runtime_defaults = experiment_config.get("runtime_defaults", {})
    selected_only = args.only or experiment_config.get("default_pilot_variant")
    variants = prepare_variants(experiment_config, only=selected_only, max_runs=1)
    runtime_overrides = {
        "epochs_override": args.epochs_override if args.epochs_override is not None else runtime_defaults.get("epochs_override"),
        "max_train_batches": (
            args.max_train_batches if args.max_train_batches is not None else runtime_defaults.get("max_train_batches")
        ),
        "max_val_batches": args.max_val_batches if args.max_val_batches is not None else runtime_defaults.get("max_val_batches"),
        "num_workers_override": runtime_defaults.get("num_workers_override"),
    }
    print(f"[Experiment] Selected variants: {[variant['name'] for variant in variants]}", flush=True)
    print(f"[Experiment] Runtime overrides: {runtime_overrides}", flush=True)

    if args.dry_run:
        print_dry_run_plan(experiment_config, variants, runtime_overrides)
        return

    if len(variants) != 1:
        raise ValueError("The final fixed-baseline experiment expects exactly one selected variant.")

    variant = variants[0]
    print(f"[Experiment] Starting fixed baseline variant: {variant['name']}", flush=True)
    print(f"[Experiment] Intent: {variant.get('intent')}", flush=True)

    resolved_config = apply_overrides(base_config, variant.get("overrides", {}))
    resolved_config = apply_runtime_overrides(
        resolved_config,
        epochs_override=runtime_overrides["epochs_override"],
        max_train_batches=runtime_overrides["max_train_batches"],
        max_val_batches=runtime_overrides["max_val_batches"],
        num_workers_override=runtime_overrides["num_workers_override"],
    )
    resolved_config.setdefault("run", {})
    resolved_config["run"]["name"] = experiment_config["experiment_name"]
    resolved_config.setdefault("training", {})
    resolved_config["training"]["checkpoint_dir"] = str(output_dir / "checkpoints")

    resolved_config_path = output_dir / "resolved_config.yaml"
    write_resolved_config(resolved_config_path, resolved_config)
    print(f"[Experiment] Wrote resolved config: {resolved_config_path}", flush=True)

    report = train_model(resolved_config, run_name=resolved_config["run"]["name"], nested_run=False)
    print(f"[Experiment] Finished run: {variant['name']}", flush=True)
    print(f"[Experiment] Run ID: {report.get('run_id')}", flush=True)
    print(
        f"[Experiment] Best mean AUC: {report.get('best_auc')} at epoch {report.get('best_auc_epoch')}",
        flush=True,
    )
    print(
        f"[Experiment] Best macro F1: {report.get('best_macro_f1')} at epoch {report.get('best_macro_f1_epoch')}",
        flush=True,
    )

    best_entry = best_history_entry(report, criterion="best_auc")
    final_entry = report["history"][-1]
    best_val_metrics = dict(best_entry["val_metrics"])
    final_train_metrics = dict(final_entry["train_metrics"])
    final_val_metrics = dict(final_entry["val_metrics"])
    test_metrics = dict(report["test_metrics"]) if report.get("test_metrics") is not None else {}
    label_names = list(report["label_names"])

    per_label_rows = build_per_label_rows(
        best_val_metrics,
        label_names,
        extra={"split": "val_best_auc"},
    )
    confusion_rows = build_confusion_rows(
        best_val_metrics,
        label_names,
        extra={"split": "val_best_auc"},
    )

    if test_metrics:
        test_per_label_rows = build_per_label_rows(
            test_metrics,
            label_names,
            extra={"split": "test"},
        )
        test_confusion_rows = build_confusion_rows(
            test_metrics,
            label_names,
            extra={"split": "test"},
        )
    else:
        test_per_label_rows = build_per_label_rows(
            {},
            label_names,
            extra={
                "split": "test",
                "evaluation_skipped": report.get("test_evaluation_skipped", True),
                "reason": report.get("test_evaluation_reason"),
            },
        )
        test_confusion_rows = build_confusion_rows(
            {},
            label_names,
            extra={
                "split": "test",
                "evaluation_skipped": report.get("test_evaluation_skipped", True),
                "reason": report.get("test_evaluation_reason"),
            },
        )

    summary_row = {
        "run_id": report.get("run_id"),
        "variant_name": variant["name"],
        "variant_intent": variant.get("intent"),
        "resolved_config_path": str(resolved_config_path),
        "best_auc": report.get("best_auc"),
        "best_auc_epoch": report.get("best_auc_epoch"),
        "best_macro_f1": report.get("best_macro_f1"),
        "best_macro_f1_epoch": report.get("best_macro_f1_epoch"),
        "final_train_loss": final_train_metrics.get("loss"),
        "final_val_loss": final_val_metrics.get("loss"),
        "final_val_mean_auc": final_val_metrics.get("mean_auc"),
        "final_val_mean_average_precision": final_val_metrics.get("mean_average_precision"),
        "final_val_macro_f1": final_val_metrics.get("macro_f1"),
        "final_val_micro_f1": final_val_metrics.get("micro_f1"),
        "test_mean_auc": test_metrics.get("mean_auc"),
        "test_mean_average_precision": test_metrics.get("mean_average_precision"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_micro_f1": test_metrics.get("micro_f1"),
        "total_runtime_sec": report.get("total_runtime_sec"),
        "stopped_early": report.get("stopped_early"),
        "stopped_epoch": report.get("stopped_epoch"),
        "best_auc_checkpoint": report.get("best_auc_checkpoint"),
        "best_macro_f1_checkpoint": report.get("best_macro_f1_checkpoint"),
        "final_checkpoint": report.get("final_checkpoint"),
        "val_total_true_positive": best_val_metrics.get("total_true_positive"),
        "val_total_false_positive": best_val_metrics.get("total_false_positive"),
        "val_total_true_negative": best_val_metrics.get("total_true_negative"),
        "val_total_false_negative": best_val_metrics.get("total_false_negative"),
        "test_total_true_positive": test_metrics.get("total_true_positive"),
        "test_total_false_positive": test_metrics.get("total_false_positive"),
        "test_total_true_negative": test_metrics.get("total_true_negative"),
        "test_total_false_negative": test_metrics.get("total_false_negative"),
        "test_evaluation_skipped": report.get("test_evaluation_skipped"),
        "test_evaluation_reason": report.get("test_evaluation_reason"),
    }

    # attention_command = _build_attention_handoff_command(output_dir)
    attention_command_path = output_dir / "attention_handoff_command.txt"
    attention_command_path.write_text(attention_command + "\n")

    summary_json = {
        "experiment_name": experiment_config["experiment_name"],
        "variant_name": variant["name"],
        "variant_intent": variant.get("intent"),
        "overrides": variant.get("overrides", {}),
        "resolved_config_path": str(resolved_config_path),
        "summary": summary_row,
        "report": report,
        "attention_handoff_command": attention_command,
    }
    test_summary_json = {
        "test_evaluation_skipped": report.get("test_evaluation_skipped"),
        "test_evaluation_reason": report.get("test_evaluation_reason"),
        "test_metrics": test_metrics if test_metrics else None,
        "best_auc_checkpoint": report.get("best_auc_checkpoint"),
    }

    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    epoch_history_path = output_dir / "epoch_history.csv"
    per_label_path = output_dir / "per_label_metrics.csv"
    confusion_path = output_dir / "confusion_metrics.csv"
    test_summary_path = output_dir / "test_summary.json"
    test_per_label_path = output_dir / "test_per_label_metrics.csv"
    test_confusion_path = output_dir / "test_confusion_metrics.csv"

    write_summary_csv(summary_csv_path, [summary_row])
    write_summary_json(summary_json_path, summary_json)
    write_summary_csv(epoch_history_path, history_to_rows(report["history"]))
    write_summary_csv(per_label_path, per_label_rows)
    write_summary_csv(confusion_path, confusion_rows)
    write_summary_json(test_summary_path, test_summary_json)
    write_summary_csv(test_per_label_path, test_per_label_rows)
    write_summary_csv(test_confusion_path, test_confusion_rows)

    _log_artifacts(
        report.get("run_id"),
        [
            resolved_config_path,
            summary_csv_path,
            summary_json_path,
            epoch_history_path,
            per_label_path,
            confusion_path,
            test_summary_path,
            test_per_label_path,
            test_confusion_path,
            attention_command_path,
        ],
    )

    # print("[Experiment] Recommended attention follow-up command:", flush=True)
    # print(attention_command, flush=True)
    print(f"[Experiment] Saved final fixed-baseline outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
