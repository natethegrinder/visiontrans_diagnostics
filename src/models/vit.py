from __future__ import annotations

import torch
from torch import nn

from .components import ViTInputEmbedding


VIT_PRESETS = {
    "vit_tiny": {"hidden_dim": 192, "num_heads": 6, "num_layers": 6},
    "vit_small": {"hidden_dim": 384, "num_heads": 6, "num_layers": 12},
    "vit_base": {"hidden_dim": 768, "num_heads": 12, "num_layers": 12},
}


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # tokens: (B, seq_len, hidden_dim)
        normalized_tokens = self.norm1(tokens)
        attention_output, attention_weights = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tokens = tokens + self.attention_dropout(attention_output)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens, attention_weights if return_attention else None


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            tokens, layer_attention = layer(tokens, return_attention=return_attention)
            if return_attention and layer_attention is not None:
                attention_maps.append(layer_attention)
        return tokens, attention_maps if return_attention else None


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
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not norm_first:
            raise ValueError(
                "The custom ViT encoder currently supports pre-norm blocks only. "
                "Set norm_first=True."
            )

        # Shape: (B, C, H, W) -> (B, num_patches + 1, hidden_dim)
        self.input_embedding = ViTInputEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder = TransformerEncoderStack(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=mlp_dim or hidden_dim * 4,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # images: (B, C, H, W)
        tokens = self.input_embedding(images)
        # encoded_tokens: (B, num_patches + 1, hidden_dim)
        encoded_tokens, attention_maps = self.encoder(tokens, return_attention=return_attention)
        # cls_token: (B, hidden_dim)
        cls_token = self.norm(encoded_tokens[:, 0])
        # logits: (B, num_classes)
        logits = self.head(cls_token)
        if return_attention:
            return logits, attention_maps or []
        return logits


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
    num_heads = int(model_config.get("num_heads", preset.get("num_heads", 6)))
    num_layers = int(model_config.get("num_layers", preset.get("num_layers", 6)))
    mlp_dim = model_config.get("dim_feedforward", model_config.get("mlp_dim", 768))
    dropout = float(model_config.get("dropout", 0.1))
    norm_first = bool(model_config.get("norm_first", True))

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
        norm_first=norm_first,
    )
