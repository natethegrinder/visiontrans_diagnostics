from __future__ import annotations

from torch import nn

from .resnet import build_resnet_model
from .vit import VisionTransformerClassifier, build_vit_model


def build_model(config: dict) -> nn.Module:
    model_config = config.get("model", {})
    family = model_config.get("family")

    if family == "vit":
        return build_vit_model(config)
    if family == "cnn":
        return build_resnet_model(config)

    raise ValueError(f"Unsupported model family '{family}'. Expected 'vit' or 'cnn'.")


__all__ = [
    "VisionTransformerClassifier",
    "build_model",
    "build_resnet_model",
    "build_vit_model",
]
