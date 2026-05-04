"""Shared helpers for ViT baseline experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUTS_ROOT = REPO_ROOT / "outputs" / "experiments"


def ensure_output_dir(name: str, output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir is not None else OUTPUTS_ROOT / name
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


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def save_rows_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)


def history_to_rows(history: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in history:
        row: dict[str, object] = {"epoch": entry["epoch"]}
        for split_name in ("train_metrics", "val_metrics", "val_tuned_metrics"):
            split_metrics = entry.get(split_name)
            if not split_metrics:
                continue
            prefix = split_name.replace("_metrics", "")
            for key, value in split_metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def best_history_entry(report: Mapping[str, object], criterion: str = "best_auc") -> Mapping[str, object]:
    history = list(report["history"])
    if not history:
        raise ValueError("Training report history is empty.")

    if criterion == "best_auc":
        epoch = report.get("best_auc_epoch")
    elif criterion == "best_macro_f1":
        epoch = report.get("best_macro_f1_epoch")
    else:
        raise ValueError(f"Unsupported criterion '{criterion}'.")

    if epoch is None:
        return history[-1]
    return history[int(epoch)]


def build_per_label_rows(
    metrics: Mapping[str, float],
    label_names: Sequence[str],
    prefix: str = "",
    extra: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_name in label_names:
        row = {
            "label": label_name,
            f"{prefix}auc": metrics.get(f"auc_{label_name}"),
            f"{prefix}average_precision": metrics.get(f"average_precision_{label_name}"),
            f"{prefix}f1": metrics.get(f"f1_{label_name}"),
            f"{prefix}precision": metrics.get(f"precision_{label_name}"),
            f"{prefix}recall": metrics.get(f"recall_{label_name}"),
            f"{prefix}binary_accuracy": metrics.get(f"binary_accuracy_{label_name}"),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def build_confusion_rows(
    metrics: Mapping[str, float],
    label_names: Sequence[str],
    prefix: str = "",
    extra: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_name in label_names:
        row = {
            "label": label_name,
            f"{prefix}true_positive": metrics.get(f"true_positive_{label_name}"),
            f"{prefix}false_positive": metrics.get(f"false_positive_{label_name}"),
            f"{prefix}true_negative": metrics.get(f"true_negative_{label_name}"),
            f"{prefix}false_negative": metrics.get(f"false_negative_{label_name}"),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def copy_config_with_updates(config: Mapping[str, object], updates: Mapping[str, object]) -> dict:
    from train import merge_dicts

    return merge_dicts(dict(config), dict(updates))


def flatten_thresholds(thresholds_by_label: Mapping[str, float] | None) -> dict[str, float]:
    if not thresholds_by_label:
        return {}
    return {f"threshold_{label}": float(value) for label, value in thresholds_by_label.items()}
