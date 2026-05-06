from __future__ import annotations

<<<<<<< Updated upstream
from typing import List

=======
from pathlib import Path
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
>>>>>>> Stashed changes
import numpy as np
import torch
import torch.nn as nn
<<<<<<< Updated upstream
import cv2
=======
from PIL import Image
>>>>>>> Stashed changes


class AttentionRollout:

    def __init__(self, model: nn.Module):
        self.model = model
        self._attentions: List[torch.Tensor] = []
        self._hooks = []

        # disable fused kernels, we need all attention weights for rollout
        for block in self.model.blocks:
            if hasattr(block.attn, 'fused_attn'):
                block.attn.fused_attn = False

        # register hooks
        for block in self.model.blocks:
            hook = block.attn.attn_drop.register_forward_hook(
                lambda m, inp, out: self._attentions.append(out.detach())
            )
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self._attentions = []
        self.model.eval()

        with torch.no_grad():
            _ = self.model(x)

        rollout = None
        eye = None

        for attn in self._attentions:
            attn_mean = attn[0].mean(dim=0) # (N+1, N+1), merge multi-head
            if eye is None:
                eye = torch.eye(attn_mean.size(0), device=attn_mean.device)
            attn_res = 0.5 * attn_mean + 0.5 * eye # add residual
            attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True) # normalize
            rollout = attn_res if rollout is None else attn_res @ rollout

        patch_tokens = rollout[0, 1:]
        mask = patch_tokens.reshape(14, 14).cpu().numpy() # 224 / 16
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return cv2.resize(mask, (224, 224))


<<<<<<< Updated upstream
class GradCAM:

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        skip_first_token: bool = False,
        cnn_mode: bool = False,
    ):
        self.model = model
        self.skip_first_token = skip_first_token
        self.cnn_mode = cnn_mode
        self._activation = None
        self._gradient = None

        def fwd_hook(m, inp, out):
            self._activation = out.detach().clone()

        def bwd_hook(m, grad_input, grad_output):
            self._gradient = grad_output[0].detach().clone()

        self._fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()

        act = self._activation
        grad = self._gradient

        if self.cnn_mode:
            weights = grad.mean(dim=(2, 3))
            cam = (weights[0, :, None, None] * act[0]).sum(dim=0)
        else:
            if act.dim() == 4:
                B, d1, d2, d3 = act.shape
                act = act.reshape(B, d1 * d2, d3)
                grad = grad.reshape(B, d1 * d2, d3)

            # Practical workaround for CLS-token classifiers:
            # patch-token gradients can be weak because the classifier reads CLS only.
            # Use CLS gradient as channel weights, then apply them to patch activations.
            if self.skip_first_token:
                weights = grad[0, 0, :]
                cam = (weights * act[0, 1:, :]).sum(dim=-1)
            else:
                weights = grad[0].mean(dim=0)
                cam = (weights * act[0]).sum(dim=-1)
            n = cam.shape[0]
            h = int(n ** 0.5)
            cam = cam.reshape(h, h)

        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cv2.resize(cam_np, (224, 224))
=======
class AttentionRollout:

    def __init__(self, model: nn.Module):
        self.model = model
        self._attentions: List[torch.Tensor] = []
        self._hooks = []

        # disable fused kernels, we need all attention weights for rollout
        for block in self.model.blocks:
            if hasattr(block.attn, 'fused_attn'):
                block.attn.fused_attn = False

        # register hooks
        for block in self.model.blocks:
            hook = block.attn.attn_drop.register_forward_hook(
                lambda m, inp, out: self._attentions.append(out.detach())
            )
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self._attentions = []
        self.model.eval()

        with torch.no_grad():
            _ = self.model(x)

        rollout = None
        eye = None

        for attn in self._attentions:
            attn_mean = attn[0].mean(dim=0)  # (N+1, N+1), merge multi-head
            if eye is None:
                eye = torch.eye(attn_mean.size(0), device=attn_mean.device)
            attn_res = 0.5 * attn_mean + 0.5 * eye  # add residual
            attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)  # normalize
            rollout = attn_res if rollout is None else attn_res @ rollout

        patch_tokens = rollout[0, 1:]
        mask = patch_tokens.reshape(14, 14).cpu().numpy()  # 224 / 16
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return cv2.resize(mask, (224, 224))


class GradCAM:

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        skip_first_token: bool = False,
        cnn_mode: bool = False,
    ):
        self.model = model
        self.skip_first_token = skip_first_token
        self.cnn_mode = cnn_mode
        self._activation = None
        self._gradient = None
        self._last_logits = None

        def fwd_hook(m, inp, out):
            self._activation = out.detach().clone()

        def bwd_hook(m, grad_input, grad_output):
            self._gradient = grad_output[0].detach().clone()

        self._fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        self._last_logits = out.detach()
        out[0, class_idx].backward()

        act = self._activation
        grad = self._gradient

        if self.cnn_mode:
            weights = grad.mean(dim=(2, 3))
            cam = (weights[0, :, None, None] * act[0]).sum(dim=0)
        else:
            if act.dim() == 4:
                B, d1, d2, d3 = act.shape
                act = act.reshape(B, d1 * d2, d3)
                grad = grad.reshape(B, d1 * d2, d3)

            if self.skip_first_token:
                weights = grad[0, 0, :]
                cam = (weights * act[0, 1:, :]).sum(dim=-1)
            else:
                weights = grad[0].mean(dim=0)
                cam = (weights * act[0]).sum(dim=-1)
            n = cam.shape[0]
            h = int(n ** 0.5)
            cam = cam.reshape(h, h)

        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cv2.resize(cam_np, (224, 224))

    def generate(self, image_tensor: torch.Tensor, target_index: int) -> tuple[np.ndarray, float]:
        cam_np = self(image_tensor, target_index)
        probability = float(torch.sigmoid(self._last_logits[0, target_index]).item())
        return cam_np, probability
>>>>>>> Stashed changes


# Use the last stage output as the target layer.
# This corresponds to the final spatial feature map before global pooling and classification head, similar to the last conv layer in CNN Grad-CAM.
class SwinGradCAM:
    def __init__(self, model: nn.Module):
        self._impl = GradCAM(model, model.layers[-1])

    def remove_hooks(self) -> None:
        self._impl.remove_hooks()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        return self._impl(x, class_idx)


def overlay_heatmap(
<<<<<<< Updated upstream
    image: np.ndarray,
=======
    image: Image.Image | np.ndarray,
>>>>>>> Stashed changes
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
<<<<<<< Updated upstream
    img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colormap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, colormap, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# Reference: https://arxiv.org/pdf/1704.03296.pdf
def pointing_game(
=======
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    else:
        gray = np.asarray(image, dtype=np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    color_heatmap = plt.get_cmap(cmap)(normalize_heatmap(heatmap))[..., :3]
    overlay = (1.0 - alpha) * gray_rgb + alpha * color_heatmap
    return np.clip(overlay, 0.0, 1.0)


def pointing_game(
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    border: int = 0,
) -> int:
    h = heatmap.copy()
    if border > 0:
        h[:border, :] = 0
        h[-border:, :] = 0
        h[:, :border] = 0
        h[:, -border:] = 0
    peak_y, peak_x = np.unravel_index(h.argmax(), h.shape)
    return int(gt_mask[peak_y, peak_x] > 0)


def trim_border(heatmap: np.ndarray, border: int) -> np.ndarray:
    h = heatmap.copy()
    h[:border, :] = 0
    h[-border:, :] = 0
    h[:, :border] = 0
    h[:, -border:] = 0
    return h / (h.max() + 1e-8)


def load_bbox_annotations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={
            "Image Index": "image_name",
            "Finding Label": "label",
            "Bbox [x": "x",
            "y": "y",
            "w": "w",
            "h]": "h",
        }
    )
    frame["label"] = frame["label"].astype(str).map(normalize_nih_label)
    return frame.loc[:, ["image_name", "label", "x", "y", "w", "h"]].copy()


def build_ground_truth_mask(boxes: pd.DataFrame, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, row in boxes.iterrows():
        x0 = max(0, min(width, int(round(float(row["x"])))))
        y0 = max(0, min(height, int(round(float(row["y"])))))
        x1 = max(0, min(width, int(round(float(row["x"]) + float(row["w"])))))
        y1 = max(0, min(height, int(round(float(row["y"]) + float(row["h"])))))
        mask[y0:y1, x0:x1] = 1
    return mask


def compute_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def save_gradcam_figure(
    image: Image.Image,
>>>>>>> Stashed changes
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    border: int = 0,
) -> int:
    h = heatmap.copy()
    if border > 0:
        h[:border, :] = 0
        h[-border:, :] = 0
        h[:, :border] = 0
        h[:, -border:] = 0
    peak_y, peak_x = np.unravel_index(h.argmax(), h.shape)
    return int(gt_mask[peak_y, peak_x] > 0)


# try to cut border since most of the peak are on the border and not meaningful
# reference: garbage token register, Darcet et al. 2023
def trim_border(heatmap: np.ndarray, border: int) -> np.ndarray:
    """Zero out the outermost `border` pixels and renormalize interior to [0,1]."""
    h = heatmap.copy()
    h[:border, :] = 0
    h[-border:, :] = 0
    h[:, :border] = 0
    h[:, -border:] = 0
    return h / (h.max() + 1e-8)
