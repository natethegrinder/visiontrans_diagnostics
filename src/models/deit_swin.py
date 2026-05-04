from __future__ import annotations

import timm
from torch import nn

_TIMM_NAMES: dict[str, str] = {
    "deit_b": "deit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}

def build_deit_swin_model(config: dict) -> nn.Module:
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    architecture = model_config.get("architecture", "deit_b")
    if architecture not in _TIMM_NAMES:
        supported = ", ".join(sorted(_TIMM_NAMES))
        raise ValueError(
            f"Unsupported timm_vit architecture '{architecture}'. Supported: {supported}"
        )

    pretrained = bool(model_config.get("pretrained", True))
    num_classes = int(data_config.get("num_classes", 14))

    # load the pre-trained model weights, using timm
    return timm.create_model(_TIMM_NAMES[architecture], pretrained=pretrained, num_classes=num_classes)