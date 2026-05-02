from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SigmoidFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss_function(config: dict, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    training_config = config.get("training", {})
    loss_name = training_config.get("loss", "bce_with_logits")
    gamma = float(training_config.get("focal_gamma", 2.0))
    alpha = float(training_config.get("focal_alpha", 0.25))

    if loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "focal":
        return SigmoidFocalLoss(gamma=gamma, alpha=alpha, pos_weight=None)
    if loss_name == "focal_with_pos_weight":
        return SigmoidFocalLoss(gamma=gamma, alpha=alpha, pos_weight=pos_weight)

    raise ValueError(
        f"Unsupported loss '{loss_name}'. Expected one of: "
        "'bce_with_logits', 'focal', 'focal_with_pos_weight'."
    )
