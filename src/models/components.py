from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for ViT patch extraction")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images)
        patches = patches.flatten(2).transpose(1, 2)
        return patches


class ViTInputEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.projection.weight)
        if self.patch_embed.projection.bias is not None:
            nn.init.zeros_(self.patch_embed.projection.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(images)
        batch_size = patch_tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.positional_embedding
        return self.dropout(tokens)
