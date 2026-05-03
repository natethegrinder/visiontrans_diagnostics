import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from losses import SigmoidFocalLoss, build_loss_function


class LossesTests(unittest.TestCase):
    def test_build_loss_function_returns_expected_types(self) -> None:
        bce_loss = build_loss_function({"training": {"loss": "bce_with_logits"}})
        focal_loss = build_loss_function({"training": {"loss": "focal"}})
        focal_weighted_loss = build_loss_function(
            {"training": {"loss": "focal_with_pos_weight"}},
            pos_weight=torch.tensor([2.0, 1.0]),
        )

        self.assertIsInstance(bce_loss, nn.BCEWithLogitsLoss)
        self.assertIsInstance(focal_loss, SigmoidFocalLoss)
        self.assertIsInstance(focal_weighted_loss, SigmoidFocalLoss)

    def test_bce_loss_returns_scalar(self) -> None:
        loss_fn = build_loss_function({"training": {"loss": "bce_with_logits"}})
        logits = torch.tensor([[0.2, -1.0], [1.0, -0.5]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = loss_fn(logits, labels)

        self.assertEqual(loss.ndim, 0)

    def test_focal_loss_reduction_mean_returns_scalar(self) -> None:
        loss_fn = SigmoidFocalLoss(reduction="mean")
        logits = torch.tensor([[0.2, -1.0], [1.0, -0.5]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = loss_fn(logits, labels)

        self.assertEqual(loss.ndim, 0)

    def test_focal_loss_reduction_none_matches_logits_shape(self) -> None:
        loss_fn = SigmoidFocalLoss(reduction="none")
        logits = torch.tensor([[0.2, -1.0], [1.0, -0.5]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = loss_fn(logits, labels)

        self.assertEqual(tuple(loss.shape), tuple(logits.shape))

    def test_invalid_loss_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_loss_function({"training": {"loss": "not_a_real_loss"}})

    def test_pos_weight_changes_bce_value(self) -> None:
        logits = torch.tensor([[0.1], [0.1], [0.1], [0.1]], dtype=torch.float32)
        labels = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)

        unweighted = build_loss_function({"training": {"loss": "bce_with_logits"}})
        weighted = build_loss_function(
            {"training": {"loss": "bce_with_logits"}},
            pos_weight=torch.tensor([5.0], dtype=torch.float32),
        )

        unweighted_loss = float(unweighted(logits, labels).item())
        weighted_loss = float(weighted(logits, labels).item())

        self.assertNotEqual(unweighted_loss, weighted_loss)


if __name__ == "__main__":
    unittest.main()
