from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_cls_attention(
    attn_maps: list[torch.Tensor],
    layer_index: int = -1,
) -> torch.Tensor:
    if not attn_maps:
        raise ValueError("attn_maps is empty. Call the model with return_attention=True first.")

    attention = attn_maps[layer_index]
    if attention.ndim != 4:
        raise ValueError(
            f"Expected attention map with shape (B, num_heads, seq_len, seq_len), got {tuple(attention.shape)}"
        )

    # attention[:, :, 0, 1:] selects CLS-to-patch attention and removes the CLS->CLS entry.
    return attention[:, :, 0, 1:]


def cls_attention_to_patch_grid(
    cls_attention: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 16,
    head_reduction: str = "mean",
) -> torch.Tensor:
    if cls_attention.ndim != 3:
        raise ValueError(
            f"Expected cls_attention with shape (B, num_heads, num_patches), got {tuple(cls_attention.shape)}"
        )
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")

    grid_size = image_size // patch_size
    expected_num_patches = grid_size * grid_size
    if cls_attention.size(-1) != expected_num_patches:
        raise ValueError(
            f"Expected {expected_num_patches} patches for image_size={image_size}, patch_size={patch_size}, "
            f"got {cls_attention.size(-1)}"
        )

    if head_reduction == "mean":
        reduced_attention = cls_attention.mean(dim=1)
    elif head_reduction == "max":
        reduced_attention = cls_attention.max(dim=1).values
    else:
        raise ValueError(f"Unsupported head_reduction '{head_reduction}'. Expected 'mean' or 'max'.")

    # reduced_attention: (B, num_patches) -> patch_grid: (B, grid_size, grid_size)
    return reduced_attention.view(cls_attention.size(0), grid_size, grid_size)


def upsample_patch_attention(
    patch_grid: torch.Tensor,
    output_size: int | tuple[int, int] = 224,
    normalize: bool = True,
) -> torch.Tensor:
    if patch_grid.ndim != 3:
        raise ValueError(
            f"Expected patch_grid with shape (B, grid_h, grid_w), got {tuple(patch_grid.shape)}"
        )

    heatmap = F.interpolate(
        patch_grid.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    if not normalize:
        return heatmap

    heatmap_min = heatmap.amin(dim=(2, 3), keepdim=True)
    heatmap_max = heatmap.amax(dim=(2, 3), keepdim=True)
    denom = (heatmap_max - heatmap_min).clamp_min(1e-8)
    return (heatmap - heatmap_min) / denom


def build_cls_attention_heatmap(
    attn_maps: list[torch.Tensor],
    image_size: int = 224,
    patch_size: int = 16,
    layer_index: int = -1,
    head_reduction: str = "mean",
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    cls_attention = extract_cls_attention(attn_maps, layer_index=layer_index)
    patch_grid = cls_attention_to_patch_grid(
        cls_attention,
        image_size=image_size,
        patch_size=patch_size,
        head_reduction=head_reduction,
    )
    heatmap = upsample_patch_attention(
        patch_grid,
        output_size=(image_size, image_size),
        normalize=normalize,
    )
    return patch_grid, heatmap
