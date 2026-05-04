import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# pos_weight = neg/pos per class, clipped at 50 to prevent training instability.
# Order matches NIH_CHEST_XRAY_LABELS in data.py:
#   Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass,
#   Nodule, Pneumonia, Pneumothorax, Consolidation, Edema,
#   Emphysema, Fibrosis, Pleural_Thickening, Hernia
DEFAULT_POS_WEIGHT = torch.tensor([
     8.7, 39.4,  7.4,  4.6, 18.4,
    16.7, 50.0, 20.1, 23.0, 47.7,
    43.6, 50.0, 32.1, 50.0,
])

def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

class ViTTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        pos_weight: torch.Tensor = DEFAULT_POS_WEIGHT,
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
        loss_fn: str = "focal",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        self.loss_fn = loss_fn
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None # MPS doesn't support

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_fn == "focal":
            return _sigmoid_focal_loss(logits, labels)
        return self.criterion(logits, labels)

    def setup_scheduler(self, num_training_steps: int, warmup_ratio: float = 0.05) -> None:
        warmup_steps = max(1, int(num_training_steps * warmup_ratio))
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=num_training_steps - warmup_steps)
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

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()
            # loss = self.criterion(self.model(images), labels)
            # loss.backward()
            # self.optimizer.step()
            if self.scaler is not None: # AMP
                with torch.amp.autocast("cuda"):
                    loss = self._loss(self.model(images), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._loss(self.model(images), labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def val_epoch(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns (avg_loss, preds, labels). preds are post-sigmoid probabilities."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                # logits = self.model(images)
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images)
                else:
                    logits = self.model(images)

                # total_loss += self.criterion(logits, labels).item()
                total_loss += self._loss(logits, labels).item()

                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return total_loss / len(loader), preds, labels
