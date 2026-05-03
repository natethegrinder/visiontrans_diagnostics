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
    threshold: float | Sequence[float] | torch.Tensor = 0.5,
    collect_attention: bool = False,
    return_outputs: bool = False,
    max_attention_batches: int | None = None,
    max_batches: int | None = None,
    progress_log_interval: int | None = 50,
) -> (
    dict[str, float]
    | tuple[dict[str, float], list[list[torch.Tensor]]]
    | tuple[dict[str, float], torch.Tensor, torch.Tensor]
    | tuple[dict[str, float], list[list[torch.Tensor]], torch.Tensor, torch.Tensor]
):
    model.eval()
    running_loss = 0.0
    sample_count = 0
    epoch_logits: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []
    epoch_attention: list[list[torch.Tensor]] = []
    total_batches = len(data_loader)
    effective_total_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if batch_index == 0:
                print("[Eval] First val batch loaded", flush=True)
            if progress_log_interval and (
                batch_index == 0
                or (batch_index + 1) % progress_log_interval == 0
                or (batch_index + 1) == effective_total_batches
            ):
                print(f"[Eval] Batch {batch_index + 1}/{effective_total_batches}", flush=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if collect_attention:
                logits, attn_maps = model(images, return_attention=True)
                if max_attention_batches is None or batch_index < max_attention_batches:
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
    if collect_attention and return_outputs:
        return metrics, epoch_attention, stacked_logits, stacked_labels
    if collect_attention:
        return metrics, epoch_attention
    if return_outputs:
        return metrics, stacked_logits, stacked_labels
    return metrics
