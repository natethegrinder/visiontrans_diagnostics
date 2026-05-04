import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import compute_multilabel_metrics, tune_multilabel_thresholds


def _probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities, eps=1e-6)


class MetricsTests(unittest.TestCase):
    def test_compute_multilabel_metrics_exposes_new_and_existing_keys(self) -> None:
        label_names = ["A", "B", "C"]
        probabilities = torch.tensor(
            [
                [0.9, 0.2, 0.8],
                [0.2, 0.8, 0.7],
                [0.7, 0.4, 0.3],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        metrics = compute_multilabel_metrics(
            logits=_probabilities_to_logits(probabilities),
            labels=labels,
            label_names=label_names,
            threshold=0.5,
        )

        expected_keys = {
            "mean_auc",
            "mean_average_precision",
            "macro_f1",
            "micro_f1",
            "exact_match_accuracy",
            "mean_binary_accuracy",
            "total_true_positive",
            "total_false_positive",
            "average_precision_A",
            "average_precision_B",
            "average_precision_C",
            "micro_precision",
            "micro_recall",
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()))

    def test_thresholding_supports_scalar_and_per_label_vector(self) -> None:
        label_names = ["A", "B"]
        probabilities = torch.tensor(
            [
                [0.8, 0.4],
                [0.4, 0.8],
                [0.6, 0.6],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logits = _probabilities_to_logits(probabilities)

        scalar_metrics = compute_multilabel_metrics(logits, labels, label_names, threshold=0.5)
        vector_metrics = compute_multilabel_metrics(
            logits,
            labels,
            label_names,
            threshold=torch.tensor([0.7, 0.3]),
        )

        self.assertNotEqual(scalar_metrics["macro_f1"], vector_metrics["macro_f1"])

    def test_single_class_labels_return_nan_auc_and_average_precision(self) -> None:
        label_names = ["A", "B"]
        probabilities = torch.tensor(
            [
                [0.8, 0.1],
                [0.7, 0.2],
                [0.9, 0.3],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        metrics = compute_multilabel_metrics(
            logits=_probabilities_to_logits(probabilities),
            labels=labels,
            label_names=label_names,
        )

        self.assertTrue(math.isnan(metrics["auc_A"]))
        self.assertTrue(math.isnan(metrics["average_precision_A"]))
        self.assertFalse(math.isnan(metrics["auc_B"]))
        self.assertFalse(math.isnan(metrics["average_precision_B"]))

    def test_exact_match_accuracy_matches_hand_computable_example(self) -> None:
        label_names = ["A", "B"]
        probabilities = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.9],
                [0.7, 0.4],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        metrics = compute_multilabel_metrics(
            logits=_probabilities_to_logits(probabilities),
            labels=labels,
            label_names=label_names,
            threshold=0.5,
        )

        self.assertAlmostEqual(metrics["exact_match_accuracy"], 2.0 / 3.0)

    def test_threshold_tuning_returns_one_threshold_per_label(self) -> None:
        label_names = ["A", "B", "C"]
        probabilities = torch.tensor(
            [
                [0.9, 0.1, 0.4],
                [0.2, 0.8, 0.8],
                [0.7, 0.6, 0.3],
                [0.3, 0.4, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        result = tune_multilabel_thresholds(
            logits=_probabilities_to_logits(probabilities),
            labels=labels,
            label_names=label_names,
            thresholds=[0.2, 0.5, 0.8],
        )

        self.assertEqual(set(result["thresholds_by_label"].keys()), set(label_names))
        self.assertEqual(tuple(result["threshold_tensor"].shape), (len(label_names),))
        self.assertEqual(set(result["best_f1_by_label"].keys()), set(label_names))

    def test_confusion_metrics_match_hand_computable_example(self) -> None:
        label_names = ["A", "B"]
        probabilities = torch.tensor(
            [
                [0.9, 0.8],
                [0.7, 0.3],
                [0.4, 0.9],
                [0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        metrics = compute_multilabel_metrics(
            logits=_probabilities_to_logits(probabilities),
            labels=labels,
            label_names=label_names,
            threshold=0.5,
        )

        self.assertEqual(metrics["true_positive_A"], 1)
        self.assertEqual(metrics["false_positive_A"], 1)
        self.assertEqual(metrics["true_negative_A"], 1)
        self.assertEqual(metrics["false_negative_A"], 1)
        self.assertEqual(metrics["true_positive_B"], 1)
        self.assertEqual(metrics["false_positive_B"], 1)
        self.assertEqual(metrics["true_negative_B"], 1)
        self.assertEqual(metrics["false_negative_B"], 1)
        self.assertEqual(metrics["total_true_positive"], 2)
        self.assertEqual(metrics["total_false_positive"], 2)
        self.assertEqual(metrics["total_true_negative"], 2)
        self.assertEqual(metrics["total_false_negative"], 2)


if __name__ == "__main__":
    unittest.main()
