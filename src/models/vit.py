from __future__ import annotations

import torch
from torch import nn

from .components import ViTInputEmbedding


VIT_PRESETS = {
    "vit_tiny": {"hidden_dim": 192, "num_heads": 3, "num_layers": 12},
    "vit_small": {"hidden_dim": 384, "num_heads": 6, "num_layers": 12},
    "vit_base": {"hidden_dim": 768, "num_heads": 12, "num_layers": 12},
}


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_embedding = ViTInputEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim or hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.input_embedding(images)
        encoded_tokens = self.encoder(tokens)
        cls_token = self.norm(encoded_tokens[:, 0])
        return self.head(cls_token)


def build_vit_model(config: dict) -> VisionTransformerClassifier:
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    architecture = model_config.get("architecture", "vit_tiny")
    preset = VIT_PRESETS.get(architecture, {})

    image_size = int(data_config.get("image_size", 224))
    num_channels = int(data_config.get("num_channels", 1))
    num_classes = int(data_config.get("num_classes", 14))
    patch_size = int(model_config.get("patch_size", 16))
    hidden_dim = int(model_config.get("hidden_dim", preset.get("hidden_dim", 192)))
    num_heads = int(model_config.get("num_heads", preset.get("num_heads", 3)))
    num_layers = int(model_config.get("num_layers", preset.get("num_layers", 12)))
    mlp_dim = model_config.get("mlp_dim")
    dropout = float(model_config.get("dropout", 0.1))

    return VisionTransformerClassifier(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=num_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=int(mlp_dim) if mlp_dim is not None else None,
        dropout=dropout,
    )
