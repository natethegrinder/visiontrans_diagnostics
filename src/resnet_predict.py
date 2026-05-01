from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

try:
    from .data import NIH_CHEST_XRAY_LABELS, build_image_transform
    from .models.resnet import build_resnet_model
except ImportError:
    from data import NIH_CHEST_XRAY_LABELS, build_image_transform
    from models.resnet import build_resnet_model


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    extends = config.pop("extends", None)
    if extends is None:
        return config

    parent_config_path = (config_path.parent / extends).resolve()
    parent_config = load_config(parent_config_path)
    return deep_merge_dicts(parent_config, config)


def resolve_device(config: dict[str, Any], requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)

    device_name = str(config.get("project", {}).get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_threshold(config: dict[str, Any], threshold_override: float | None) -> float:
    if threshold_override is not None:
        return float(threshold_override)
    return float(config.get("evaluation", {}).get("threshold", 0.5))


def load_checkpoint_into_model(checkpoint_path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return {}

    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    config: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    data_config = config.get("data", {})
    image_size = int(data_config.get("image_size", 224))
    num_channels = int(data_config.get("num_channels", 1))
    transform = build_image_transform(image_size=image_size, num_channels=num_channels)

    image = Image.open(image_path)
    image = image.convert("L" if num_channels == 1 else "RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu()

    disease_probabilities = {
        label: float(probabilities[index].item())
        for index, label in enumerate(NIH_CHEST_XRAY_LABELS)
    }

    predicted_labels = [
        label
        for label, probability in disease_probabilities.items()
        if probability >= threshold
    ]

    return {
        "image_path": str(image_path.resolve()),
        "threshold": threshold,
        "predicted_labels": predicted_labels,
        "disease_probabilities": disease_probabilities,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ResNet inference on a single NIH Chest X-ray image.")
    parser.add_argument("--image", required=True, help="Path to the image file to predict.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained ResNet checkpoint.")
    parser.add_argument("--config", default="configs/cnn_baseline.yaml", help="Path to the YAML config file.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold for positive labels.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    device = resolve_device(config, args.device)
    threshold = resolve_threshold(config, args.threshold)

    model = build_resnet_model(config).to(device)
    checkpoint = load_checkpoint_into_model(Path(args.checkpoint), model, device)
    prediction = predict_image(
        model=model,
        image_path=Path(args.image),
        config=config,
        device=device,
        threshold=threshold,
    )

    if checkpoint:
        prediction["checkpoint_epoch"] = checkpoint.get("epoch")
        prediction["best_val_loss"] = checkpoint.get("best_val_loss")

    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()