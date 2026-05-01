from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
	from .data import NIH_CHEST_XRAY_LABELS, build_image_transform, normalize_nih_label
	from .evaluate import load_config, resolve_device, resolve_threshold
	from .mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_directory, resolve_nonconflicting_path
	from .models.resnet import build_resnet_model
	from .resnet_predict import load_checkpoint_into_model
except ImportError:
	from data import NIH_CHEST_XRAY_LABELS, build_image_transform, normalize_nih_label
	from evaluate import load_config, resolve_device, resolve_threshold
	from mlflow_utils import configure_mlflow, log_dict_artifact, log_metrics, resolve_nonconflicting_directory, resolve_nonconflicting_path
	from models.resnet import build_resnet_model
	from resnet_predict import load_checkpoint_into_model


class GradCAM:
	def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
		self.model = model
		self.target_layer = target_layer
		self.activations: torch.Tensor | None = None
		self.gradients: torch.Tensor | None = None
		self.target_layer.register_forward_hook(self._forward_hook)

	def _forward_hook(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
		self.activations = output.detach()
		output.register_hook(self._backward_hook)

	def _backward_hook(self, gradient: torch.Tensor) -> None:
		self.gradients = gradient.detach()

	def generate(self, image_tensor: torch.Tensor, target_index: int) -> tuple[np.ndarray, float]:
		self.model.zero_grad(set_to_none=True)
		logits = self.model(image_tensor)
		target_score = logits[:, target_index].sum()
		target_score.backward()
		if self.activations is None or self.gradients is None:
			raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

		weights = self.gradients.mean(dim=(2, 3), keepdim=True)
		cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
		cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
		cam_np = cam.squeeze().detach().cpu().numpy()
		cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
		probability = float(torch.sigmoid(logits[0, target_index]).item())
		return cam_np, probability


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Compute CNN Grad-CAM IoU against NIH bounding boxes.")
	parser.add_argument("--config", default="configs/interpretability.yaml", help="Path to the interpretability config file.")
	parser.add_argument("--checkpoint", required=True, help="Path to the trained CNN checkpoint.")
	parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
	parser.add_argument("--threshold", type=float, default=None, help="Probability threshold for predicted labels.")
	parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of image-label pairs to evaluate.")
	return parser


def resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
	last_block = model.layer4[-1]
	if hasattr(last_block, "conv3"):
		return last_block.conv3
	return last_block.conv2


def load_bbox_annotations(path: Path) -> pd.DataFrame:
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


def save_overlay(image: Image.Image, heatmap: np.ndarray, gt_mask: np.ndarray, output_path: Path) -> None:
	figure, axes = plt.subplots(1, 3, figsize=(12, 4))
	axes[0].imshow(image, cmap="gray")
	axes[0].set_title("Image")
	axes[1].imshow(image, cmap="gray")
	axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
	axes[1].set_title("Grad-CAM")
	axes[2].imshow(gt_mask, cmap="gray")
	axes[2].set_title("BBox Mask")
	for axis in axes:
		axis.axis("off")
	figure.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


def main() -> None:
	args = build_arg_parser().parse_args()
	config = load_config(args.config)
	device = resolve_device(config, args.device)
	threshold = resolve_threshold(config, args.threshold)

	interpretability_config = config.get("interpretability", {})
	bbox_path = Path(interpretability_config.get("bbox_annotations", "data/annotations/BBox_List_2017.csv"))
	heatmap_threshold = float(interpretability_config.get("heatmap_threshold", 0.8))
	save_examples = bool(interpretability_config.get("save_examples", True))
	num_examples = int(interpretability_config.get("num_examples", 100))
	limit = args.limit if args.limit is not None else num_examples

	bbox_frame = load_bbox_annotations(bbox_path)
	test_manifest = Path(config.get("data", {}).get("test_manifest", "data/manifests/test.csv"))
	test_frame = pd.read_csv(test_manifest)
	merged = bbox_frame.merge(test_frame[["image_name", "image_path"]], on="image_name", how="inner")
	pairs = merged.groupby(["image_name", "label", "image_path"], as_index=False).agg(list)
	if limit > 0:
		pairs = pairs.head(limit)

	model = build_resnet_model(config).to(device)
	checkpoint = load_checkpoint_into_model(Path(args.checkpoint), model, device)
	model.eval()
	target_layer = resolve_target_layer(model)
	grad_cam = GradCAM(model, target_layer)

	data_config = config.get("data", {})
	image_size = int(data_config.get("image_size", 224))
	num_channels = int(data_config.get("num_channels", 1))
	transform = build_image_transform(image_size=image_size, num_channels=num_channels)

	results: list[dict[str, Any]] = []
	examples_dir = resolve_nonconflicting_directory(Path("outputs") / "interpretability" / Path(args.checkpoint).stem)
	for index, row in pairs.iterrows():
		image_path = Path(row["image_path"])
		label = str(row["label"])
		label_index = NIH_CHEST_XRAY_LABELS.index(label)
		image = Image.open(image_path)
		image = image.convert("L" if num_channels == 1 else "RGB")
		original_width, original_height = image.size
		image_tensor = transform(image).unsqueeze(0).to(device)
		cam, probability = grad_cam.generate(image_tensor, label_index)
		cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((original_width, original_height))) / 255.0
		pred_mask = (cam_resized >= heatmap_threshold).astype(np.uint8)

		boxes = pd.DataFrame({"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]})
		gt_mask = build_ground_truth_mask(boxes, original_width, original_height)
		iou = compute_iou(pred_mask, gt_mask)

		record = {
			"image_name": str(row["image_name"]),
			"image_path": str(image_path),
			"label": label,
			"probability": probability,
			"predicted_positive": probability >= threshold,
			"iou": iou,
		}
		results.append(record)

		if save_examples:
			overlay_path = examples_dir / f"{index:04d}_{Path(row['image_name']).stem}_{label}.png"
			save_overlay(image, cam_resized, gt_mask, overlay_path)

	aggregate_by_label: dict[str, dict[str, float | int]] = {}
	for label in NIH_CHEST_XRAY_LABELS:
		label_records = [record for record in results if record["label"] == label]
		if not label_records:
			continue
		aggregate_by_label[label] = {
			"num_examples": len(label_records),
			"mean_iou": float(np.mean([float(record["iou"]) for record in label_records])),
			"positive_rate": float(np.mean([float(record["predicted_positive"]) for record in label_records])),
		}

	summary = {
		"checkpoint": str(Path(args.checkpoint).resolve()),
		"checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
		"bbox_annotations": str(bbox_path.resolve()),
		"heatmap_threshold": heatmap_threshold,
		"num_examples": len(results),
		"mean_iou": float(np.mean([float(record["iou"]) for record in results])) if results else 0.0,
		"per_label": aggregate_by_label,
		"records": results,
	}

	output_dir = Path("outputs") / "interpretability"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = resolve_nonconflicting_path(output_dir / f"{Path(args.checkpoint).stem}_grad_cam_iou.json")
	output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	mlflow_config = configure_mlflow(config)
	with mlflow.start_run(run_name=f"{Path(args.checkpoint).stem}_grad_cam_iou"):
		mlflow.log_param("interpretability_checkpoint", str(args.checkpoint))
		mlflow.log_param("heatmap_threshold", heatmap_threshold)
		mlflow.log_param("num_examples", len(results))
		log_metrics(
			{
				"grad_cam_mean_iou": summary["mean_iou"],
				**{f"grad_cam_{label}_mean_iou": values["mean_iou"] for label, values in aggregate_by_label.items()},
			}
		)
		log_dict_artifact("grad_cam_iou_summary", summary)
		if bool(mlflow_config.get("log_artifacts", True)):
			mlflow.log_artifact(str(output_path), artifact_path="interpretability")
			if save_examples and examples_dir.exists():
				mlflow.log_artifacts(str(examples_dir), artifact_path="interpretability_examples")

	print(f"Saved interpretability summary to {output_path}")


if __name__ == "__main__":
	main()