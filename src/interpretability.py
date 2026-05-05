from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from data import NIH_CHEST_XRAY_LABELS, normalize_nih_label


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()
        output.register_hook(self._backward_hook)

    def _backward_hook(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()

    def generate(self, image_tensor: torch.Tensor, target_index: int) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        target_score = logits[:, target_index].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = normalize_heatmap(cam_np)
        probability = float(torch.sigmoid(logits[0, target_index]).item())
        return cam_np, probability


def resolve_cnn_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    if hasattr(model, "layer4"):
        last_block = model.layer4[-1]
        if hasattr(last_block, "conv3"):
            return last_block.conv3
        if hasattr(last_block, "conv2"):
            return last_block.conv2
    raise ValueError("Could not resolve a default CNN target layer for Grad-CAM.")


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    return (heatmap - min_value) / (max_value - min_value + 1e-8)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    color_heatmap = plt.get_cmap(cmap)(normalize_heatmap(heatmap))[..., :3]
    overlay = (1.0 - alpha) * gray_rgb + alpha * color_heatmap
    return np.clip(overlay, 0.0, 1.0)


def load_bbox_annotations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={
            "Image Index": "image_name",
            "Finding Label": "label",
            "Bbox [x": "x",
            "y": "y",
            "w": "w",
            "h]": "h",
        }
    )
    frame["label"] = frame["label"].astype(str).map(normalize_nih_label)
    return frame.loc[:, ["image_name", "label", "x", "y", "w", "h"]].copy()


def build_ground_truth_mask(boxes: pd.DataFrame, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, row in boxes.iterrows():
        x0 = max(0, min(width, int(round(float(row["x"])))))
        y0 = max(0, min(height, int(round(float(row["y"])))))
        x1 = max(0, min(width, int(round(float(row["x"]) + float(row["w"])))))
        y1 = max(0, min(height, int(round(float(row["y"]) + float(row["h"])))))
        mask[y0:y1, x0:x1] = 1
    return mask


def compute_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def save_gradcam_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str,
    gt_mask: np.ndarray | None = None,
) -> None:
    panel_count = 4 if gt_mask is not None else 3
    figure, axes = plt.subplots(1, panel_count, figsize=(4 * panel_count, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay_heatmap(image, heatmap))
    axes[2].set_title("Overlay")
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("BBox mask")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def select_gradcam_candidates(
    manifest: pd.DataFrame,
    labels: list[str] | tuple[str, ...] = NIH_CHEST_XRAY_LABELS,
    max_per_label: int = 2,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label in labels:
        if label not in manifest.columns:
            continue
        positives = manifest[manifest[label] == 1].head(max_per_label)
        for _, row in positives.iterrows():
            candidates.append(
                {
                    "image_name": row["image_name"],
                    "image_path": row["image_path"],
                    "label": label,
                }
            )
    return candidates
