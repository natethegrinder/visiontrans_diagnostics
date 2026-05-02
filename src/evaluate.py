from __future__ import annotations

from typing import Sequence

import torch

from metrics import compute_multilabel_metrics


def evaluate_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label_names: Sequence[str],
    threshold: float = 0.5,
    collect_attention: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[list[torch.Tensor]]]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    epoch_logits: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []
    epoch_attention: list[list[torch.Tensor]] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if collect_attention:
                logits, attn_maps = model(images, return_attention=True)
                epoch_attention.append([attention.detach().cpu() for attention in attn_maps])
            else:
                logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            epoch_logits.append(logits.detach().cpu())
            epoch_labels.append(labels.detach().cpu())

    stacked_logits = torch.cat(epoch_logits, dim=0)
    stacked_labels = torch.cat(epoch_labels, dim=0)
    metrics = compute_multilabel_metrics(stacked_logits, stacked_labels, label_names, threshold=threshold)
    metrics["loss"] = running_loss / max(sample_count, 1)
    if collect_attention:
        return metrics, epoch_attention
    return metrics
