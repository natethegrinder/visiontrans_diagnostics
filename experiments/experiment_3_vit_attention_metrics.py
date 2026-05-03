"""Evaluate ViT attention artifacts and optional bounding-box overlap metrics."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import REPO_ROOT, build_per_label_rows, ensure_output_dir, save_json, save_rows_csv
from data import build_nih_data_module
from evaluate import evaluate_epoch
from interpretability import build_cls_attention_heatmap
from losses import build_loss_function
from mlflow_utils import configure_mlflow, log_label_statistics, log_params_flat
from models import build_model
from train import build_run_params, load_config, merge_dicts, resolve_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vit_baseline.yaml", help="Path to the ViT baseline config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained ViT checkpoint.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--max-heatmaps", type=int, default=16, help="Maximum number of heatmaps to save.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs. Defaults to outputs/experiments/experiment_3_vit_attention_metrics.",
    )
    return parser


def _safe_mlflow_log_metrics(metrics: dict[str, float]) -> None:
    filtered = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    }
    if filtered:
        mlflow.log_metrics(filtered)


def _resolve_bbox_path(annotations_dir: Path) -> Path | None:
    for candidate in ("BBox_List_2017.csv", "BBox_list_2017.csv"):
        path = annotations_dir / candidate
        if path.exists():
            return path
    return None


def _load_bbox_frame(annotations_dir: Path) -> pd.DataFrame | None:
    bbox_path = _resolve_bbox_path(annotations_dir)
    if bbox_path is None:
        return None

    frame = pd.read_csv(bbox_path)
    rename_map = {}
    for column in frame.columns:
        normalized = column.strip().lower()
        if normalized == "image index":
            rename_map[column] = "image_name"
        elif normalized == "finding label":
            rename_map[column] = "label"
        elif normalized == "bbox x":
            rename_map[column] = "x"
        elif normalized == "bbox y":
            rename_map[column] = "y"
        elif normalized == "bbox w":
            rename_map[column] = "w"
        elif normalized == "bbox h":
            rename_map[column] = "h"
    frame = frame.rename(columns=rename_map)
    required = {"image_name", "x", "y", "w", "h"}
    if not required.issubset(frame.columns):
        return None
    return frame


def _denormalize_image(
    image_tensor: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    for channel_index, (channel_mean, channel_std) in enumerate(zip(mean, std)):
        image[channel_index] = image[channel_index] * channel_std + channel_mean
    image = image.clamp(0.0, 1.0).numpy()
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    return (image * 255.0).astype(np.uint8)


def _save_heatmap_overlay(
    output_path: Path,
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> None:
    base_image = _denormalize_image(image_tensor, mean=mean, std=std).astype(np.float32) / 255.0
    heatmap_array = heatmap.squeeze(0).detach().cpu().numpy()
    heatmap_rgb = np.zeros_like(base_image)
    heatmap_rgb[..., 0] = heatmap_array
    overlay = (0.65 * base_image + 0.35 * heatmap_rgb).clip(0.0, 1.0)
    Image.fromarray((overlay * 255.0).astype(np.uint8)).save(output_path)


def _compute_bbox_localization_metrics(
    heatmap: torch.Tensor,
    bbox_rows: pd.DataFrame,
    image_size: int,
    original_width: float,
    original_height: float,
    activation_quantile: float = 0.85,
) -> dict[str, float]:
    heatmap_array = heatmap.squeeze(0).detach().cpu().numpy()
    threshold_value = float(np.quantile(heatmap_array, activation_quantile))
    attention_mask = heatmap_array >= threshold_value

    bbox_mask = np.zeros((image_size, image_size), dtype=bool)
    scale_x = image_size / max(original_width, 1.0)
    scale_y = image_size / max(original_height, 1.0)
    for _, bbox in bbox_rows.iterrows():
        x0 = max(0, int(np.floor(float(bbox["x"]) * scale_x)))
        y0 = max(0, int(np.floor(float(bbox["y"]) * scale_y)))
        x1 = min(image_size, int(np.ceil((float(bbox["x"]) + float(bbox["w"])) * scale_x)))
        y1 = min(image_size, int(np.ceil((float(bbox["y"]) + float(bbox["h"])) * scale_y)))
        if x1 > x0 and y1 > y0:
            bbox_mask[y0:y1, x0:x1] = True

    intersection = np.logical_and(attention_mask, bbox_mask).sum()
    union = np.logical_or(attention_mask, bbox_mask).sum()
    bbox_area = bbox_mask.sum()
    return {
        "attention_bbox_iou": float(intersection / union) if union > 0 else float("nan"),
        "attention_bbox_overlap": float(intersection / bbox_area) if bbox_area > 0 else float("nan"),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = ensure_output_dir("experiment_3_vit_attention_metrics", args.output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(REPO_ROOT / args.config)
    checkpoint = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    config = merge_dicts(checkpoint_config, base_config)

    device = resolve_device(config)
    data_module = build_nih_data_module(config)
    split = args.split if args.split in data_module["dataloaders"] else "val"
    data_loader = data_module["dataloaders"][split]
    label_names = list(data_module["labels"])
    pos_weight_stats = data_module["pos_weight_stats"]
    use_pos_weight = bool(config.get("training", {}).get("use_pos_weight", True))
    pos_weight_tensor = pos_weight_stats["pos_weight_tensor"].to(device) if use_pos_weight else None
    loss_fn = build_loss_function(config, pos_weight=pos_weight_tensor)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    metrics = evaluate_epoch(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device=device,
        label_names=label_names,
        threshold=float(config.get("training", {}).get("threshold", 0.5)),
    )
    inference_time_sec = time.perf_counter() - inference_start
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )

    data_config = config.get("data", {})
    mean = tuple(float(value) for value in data_config.get("normalize_mean", [0.5]))
    std = tuple(float(value) for value in data_config.get("normalize_std", [0.25]))
    image_size = int(data_config.get("image_size", 224))
    patch_size = int(config.get("model", {}).get("patch_size", 16))
    bbox_frame = _load_bbox_frame(Path(data_config["annotations_dir"]))

    heatmap_rows: list[dict[str, object]] = []
    localization_rows: list[dict[str, object]] = []
    saved_heatmaps = 0
    sample_offset = 0
    dataset_frame = data_loader.dataset.frame.reset_index(drop=True)

    model.eval()
    with torch.inference_mode():
        for images, _ in data_loader:
            if saved_heatmaps >= args.max_heatmaps:
                break

            batch_size = images.size(0)
            batch_frame = dataset_frame.iloc[sample_offset : sample_offset + batch_size].reset_index(drop=True)
            sample_offset += batch_size

            logits, attn_maps = model(images.to(device), return_attention=True)
            attn_maps_cpu = [attention.detach().cpu() for attention in attn_maps]
            _, heatmaps = build_cls_attention_heatmap(
                attn_maps_cpu,
                image_size=image_size,
                patch_size=patch_size,
            )

            for sample_index in range(batch_size):
                if saved_heatmaps >= args.max_heatmaps:
                    break

                row = batch_frame.iloc[sample_index]
                image_name = str(row["image_name"])
                output_path = heatmap_dir / f"{saved_heatmaps:03d}_{image_name.replace('/', '_')}.png"
                _save_heatmap_overlay(
                    output_path,
                    image_tensor=images[sample_index],
                    heatmap=heatmaps[sample_index],
                    mean=mean,
                    std=std,
                )
                heatmap_rows.append(
                    {
                        "image_name": image_name,
                        "artifact_path": str(output_path),
                        "view_position": row.get("view_position"),
                    }
                )

                if bbox_frame is not None:
                    sample_boxes = bbox_frame[bbox_frame["image_name"].astype(str) == image_name]
                    if not sample_boxes.empty:
                        localization_metrics = _compute_bbox_localization_metrics(
                            heatmap=heatmaps[sample_index],
                            bbox_rows=sample_boxes,
                            image_size=image_size,
                            original_width=float(row.get("original_width", 1024) or 1024),
                            original_height=float(row.get("original_height", 1024) or 1024),
                        )
                        localization_rows.append(
                            {
                                "image_name": image_name,
                                "num_boxes": int(len(sample_boxes)),
                                **localization_metrics,
                            }
                        )
                saved_heatmaps += 1

    localization_summary = {}
    if localization_rows:
        localization_summary = {
            "mean_attention_bbox_iou": float(pd.DataFrame(localization_rows)["attention_bbox_iou"].mean()),
            "mean_attention_bbox_overlap": float(pd.DataFrame(localization_rows)["attention_bbox_overlap"].mean()),
            "localization_sample_count": len(localization_rows),
        }

    summary = {
        "checkpoint": str(REPO_ROOT / args.checkpoint),
        "split": split,
        "classification_metrics": metrics,
        "inference_time_sec": inference_time_sec,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "num_heatmaps_generated": saved_heatmaps,
        **localization_summary,
    }

    save_json(output_dir / "summary.json", summary)
    save_rows_csv(output_dir / "per_label_metrics.csv", build_per_label_rows(metrics, label_names))
    save_rows_csv(output_dir / "heatmaps.csv", heatmap_rows)
    if localization_rows:
        save_rows_csv(output_dir / "localization_metrics.csv", localization_rows)

    configure_mlflow(config)
    run_name = config.get("run", {}).get("name", "vit_baseline") + "_experiment_3"
    with mlflow.start_run(run_name=run_name):
        log_params_flat(build_run_params(config, pos_weight_stats))
        log_label_statistics(pos_weight_stats)
        mlflow.log_param("experiment_script", "experiment_3_vit_attention_metrics")
        mlflow.log_param("checkpoint", str(REPO_ROOT / args.checkpoint))
        mlflow.log_param("split", split)
        mlflow.log_param("max_heatmaps", args.max_heatmaps)
        _safe_mlflow_log_metrics({f"{split}_{key}": value for key, value in metrics.items()})
        _safe_mlflow_log_metrics(
            {
                "num_heatmaps_generated": float(saved_heatmaps),
                "inference_time_sec": inference_time_sec,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                **{key: float(value) for key, value in localization_summary.items()},
            }
        )
        mlflow.log_artifacts(str(heatmap_dir), artifact_path="attention_heatmaps")
        if localization_rows:
            mlflow.log_artifact(str(output_dir / "localization_metrics.csv"))
        mlflow.log_artifact(str(output_dir / "summary.json"))
        mlflow.log_artifact(str(output_dir / "per_label_metrics.csv"))

    print(f"Saved Experiment 3 outputs to {output_dir}")


if __name__ == "__main__":
    main()
