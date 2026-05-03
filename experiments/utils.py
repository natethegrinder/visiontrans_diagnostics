"""Utilities for config-driven ViT baseline experiments.

These helpers are designed for pilot-first workflows. Use `--dry-run` before
launching any real training job because even small ViT grids can be slow on a
local workstation.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text()) or {}
    return dict(config)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    for dotted_key, value in overrides.items():
        set_by_dotted_key(resolved, dotted_key, value)
    return resolved


def validate_variants(variants: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for variant in variants:
        name = str(variant["name"])
        if name in seen:
            raise ValueError(f"Duplicate variant name '{name}' in experiment config.")
        seen.add(name)


def prepare_variants(
    experiment_config: Mapping[str, Any],
    only: str | None = None,
    max_runs: int | None = None,
) -> list[dict[str, Any]]:
    variants = [dict(variant) for variant in experiment_config.get("variants", [])]
    validate_variants(variants)

    if only is not None:
        variants = [variant for variant in variants if variant["name"] == only]
        if not variants:
            available = ", ".join(variant["name"] for variant in experiment_config.get("variants", []))
            raise ValueError(f"Variant '{only}' not found. Available variants: {available}")

    if max_runs is not None:
        variants = variants[: max(0, int(max_runs))]
    return variants


def apply_runtime_overrides(
    config: Mapping[str, Any],
    *,
    epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_attention_batches: int | None = None,
    num_workers_override: int | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    if epochs_override is not None:
        set_by_dotted_key(resolved, "training.epochs", int(epochs_override))
    if max_train_batches is not None:
        set_by_dotted_key(resolved, "runtime.max_train_batches", int(max_train_batches))
    if max_val_batches is not None:
        set_by_dotted_key(resolved, "runtime.max_val_batches", int(max_val_batches))
    if max_attention_batches is not None:
        set_by_dotted_key(resolved, "runtime.max_attention_batches", int(max_attention_batches))
        set_by_dotted_key(resolved, "interpretability.max_attention_batches", int(max_attention_batches))
    if num_workers_override is not None:
        set_by_dotted_key(resolved, "data.num_workers", int(num_workers_override))
    return resolved


def ensure_output_dir(default_output_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir is not None else REPO_ROOT / str(default_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def write_summary_json(path: str | Path, summary: object) -> None:
    write_json(path, summary)


def write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_resolved_config(path: str | Path, config: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False))


def flatten_nested_metrics(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        compound_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_nested_metrics(compound_key, value))
        else:
            flattened[compound_key] = value
    return flattened


def build_variant_row(
    *,
    experiment_name: str,
    variant: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    resolved_config_path: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "variant_name": variant["name"],
        "variant_intent": variant.get("intent"),
        "changed_parameters": json.dumps(variant.get("overrides", {}), sort_keys=True),
        "resolved_config_path": str(resolved_config_path),
    }
    if report is not None:
        row["run_id"] = report.get("run_id")
        row["best_auc"] = report.get("best_auc")
        row["best_auc_epoch"] = report.get("best_auc_epoch")
        row["best_macro_f1"] = report.get("best_macro_f1")
        row["best_macro_f1_epoch"] = report.get("best_macro_f1_epoch")
        row["best_auc_checkpoint"] = report.get("best_auc_checkpoint")
        row["best_macro_f1_checkpoint"] = report.get("best_macro_f1_checkpoint")
        row["total_runtime_sec"] = report.get("total_runtime_sec")
    if extra:
        row.update(extra)
    return row


def print_dry_run_plan(
    experiment_config: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    runtime_overrides: Mapping[str, Any] | None = None,
) -> None:
    print(f"Experiment: {experiment_config.get('experiment_name')}")
    print(f"Description: {experiment_config.get('description')}")
    if runtime_overrides:
        print(f"Runtime overrides: {dict(runtime_overrides)}")
    print("Planned variants:")
    for variant in variants:
        print(f"- {variant['name']}")
        if variant.get("intent"):
            print(f"  intent: {variant['intent']}")
        overrides = variant.get("overrides", {})
        if overrides:
            print(f"  overrides: {json.dumps(overrides, sort_keys=True)}")
