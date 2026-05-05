from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from data import NIH_CHEST_XRAY_LABELS


def compute_pos_weight_from_frame(
    frame: Any,
    min_positive_count: float = 1.0,
    max_pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    total_samples = float(len(frame))
    if total_samples <= 0:
        raise ValueError("Training manifest is empty; cannot compute class imbalance weights.")

    weights: list[float] = []
    positive_counts: dict[str, float] = {}
    pos_weight_by_label: dict[str, float] = {}
    for label in NIH_CHEST_XRAY_LABELS:
        positive_count = float(frame[label].sum())
        adjusted_positive_count = max(positive_count, min_positive_count)
        negative_count = max(total_samples - positive_count, 0.0)
        pos_weight = negative_count / adjusted_positive_count
        if max_pos_weight is not None:
            pos_weight = min(pos_weight, max_pos_weight)
        weights.append(pos_weight)
        positive_counts[label] = positive_count
        pos_weight_by_label[label] = pos_weight

    return torch.tensor(weights, dtype=torch.float32), positive_counts, pos_weight_by_label


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_factor

        loss = focal_weight * bce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_criterion(
    config: dict[str, Any],
    train_frame: Any,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    training_config = config.get("training", {})
    loss_name = str(training_config.get("loss", "bce_with_logits")).lower()
    imbalance_config = training_config.get("imbalance", {})

    pos_weight: torch.Tensor | None = None
    summary: dict[str, Any] = {
        "loss_name": loss_name,
        "imbalance_strategy": "none",
    }

    if bool(imbalance_config.get("enabled", False)):
        strategy = str(imbalance_config.get("strategy", "pos_weight")).lower()
        if strategy != "pos_weight":
            raise ValueError(f"Unsupported imbalance strategy: {strategy}")
        pos_weight, positive_counts, pos_weight_by_label = compute_pos_weight_from_frame(
            train_frame,
            min_positive_count=float(imbalance_config.get("min_positive_count", 1.0)),
            max_pos_weight=(
                float(imbalance_config["max_pos_weight"])
                if imbalance_config.get("max_pos_weight") is not None
                else None
            ),
        )
        pos_weight = pos_weight.to(device)
        summary.update(
            {
                "imbalance_strategy": "pos_weight",
                "positive_count": positive_counts,
                "pos_weight": pos_weight_by_label,
            }
        )

    if loss_name in {"bce", "bce_with_logits", "bcewithlogitsloss"}:
        summary["loss_name"] = "bce_with_logits"
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), summary

    if loss_name in {"focal", "focal_loss", "focalloss"}:
        focal_config = training_config.get("focal_loss", {})
        gamma = float(focal_config.get("gamma", 2.0))
        alpha = focal_config.get("alpha")
        alpha_value = float(alpha) if alpha is not None else None
        summary.update({"loss_name": "focal", "focal_gamma": gamma, "focal_alpha": alpha_value})
        return BinaryFocalLoss(gamma=gamma, alpha=alpha_value, pos_weight=pos_weight), summary

    raise ValueError(f"Unsupported loss: {loss_name}")


class ExperimentTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def setup_scheduler(self, num_training_steps: int, warmup_ratio: float = 0.05) -> None:
        warmup_steps = max(1, int(num_training_steps * warmup_ratio))
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_steps = max(1, num_training_steps - warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    def freeze_backbone(self) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = 'head' in name
        print(f"  Backbone frozen — trainable params: {self._count_trainable():,}")

    def unfreeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"  Backbone unfrozen — trainable params: {self._count_trainable():,}")

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int | None = None,
        total_epochs: int | None = None,
        progress_interval: int = 0,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = len(loader)
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self.criterion(self.model(images), labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            if progress_interval > 0 and (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % progress_interval == 0
            ):
                epoch_text = (
                    f"epoch={epoch}/{total_epochs} "
                    if epoch is not None and total_epochs is not None
                    else ""
                )
                pct = 100.0 * batch_index / max(total_batches, 1)
                avg_loss = total_loss / batch_index
                print(
                    f"{epoch_text}batch={batch_index}/{total_batches} "
                    f"({pct:.1f}%) train_loss_avg={avg_loss:.4f}",
                    flush=True,
                )
        return total_loss / len(loader)

    def val_epoch(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns (avg_loss, preds, labels). preds are post-sigmoid probabilities."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.float().to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                total_loss += self.criterion(logits, labels).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return total_loss / len(loader), preds, labels


# Backward-compatible alias for existing imports.
ViTTrainer = ExperimentTrainer
