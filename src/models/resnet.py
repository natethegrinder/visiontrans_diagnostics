from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models import resnet18, resnet34, resnet50


RESNET_BUILDERS: dict[str, tuple[Callable[..., nn.Module], object]] = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


def _replace_resnet_input_layer(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    if in_channels == 3:
        return

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new_conv.weight.copy_(repeated / max(in_channels, 1))

    model.conv1 = new_conv


def build_resnet_model(config: dict) -> nn.Module:
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    architecture = model_config.get("architecture", "resnet18")
    if architecture not in RESNET_BUILDERS:
        supported = ", ".join(sorted(RESNET_BUILDERS))
        raise ValueError(f"Unsupported ResNet architecture '{architecture}'. Supported options: {supported}")

    builder, default_weights = RESNET_BUILDERS[architecture]
    pretrained = bool(model_config.get("pretrained", False))
    model = builder(weights=default_weights if pretrained else None)

    in_channels = int(data_config.get("num_channels", 1))
    num_classes = int(data_config.get("num_classes", 14))
    dropout = float(model_config.get("dropout", 0.0))

    _replace_resnet_input_layer(model, in_channels=in_channels, pretrained=pretrained)

    classifier = nn.Linear(model.fc.in_features, num_classes)
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout), classifier)
    else:
        model.fc = classifier

    return model
