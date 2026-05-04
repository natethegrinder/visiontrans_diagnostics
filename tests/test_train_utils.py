import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import evaluate_epoch
from train import (
    build_run_params,
    build_scheduler,
    evaluate_best_checkpoint_on_test_split,
    initialize_early_stopping,
    initialize_scheduler_learning_rate,
    step_scheduler,
    train_one_epoch,
    update_early_stopping_state,
)


class TrainUtilsTests(unittest.TestCase):
    def _toy_loader(self) -> DataLoader:
        inputs = torch.tensor(
            [
                [1.0, 0.0, 0.5, -0.2],
                [0.1, 0.8, -0.4, 0.3],
                [0.7, -0.6, 0.2, 0.9],
                [-0.3, 0.2, 0.4, -0.8],
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
        return DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

    def test_warmup_cosine_scheduler_builds_successfully(self) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(
            {"training": {"scheduler": "warmup_cosine", "warmup_epochs": 2, "epochs": 6}},
            optimizer,
        )

        self.assertIsNotNone(scheduler)

    def test_warmup_cosine_learning_rate_warms_up_then_decays(self) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = {"training": {"scheduler": "warmup_cosine", "warmup_epochs": 2, "epochs": 6}}
        scheduler = build_scheduler(config, optimizer)
        initialize_scheduler_learning_rate(config, optimizer, scheduler)

        lrs = [float(optimizer.param_groups[0]["lr"])]
        for epoch in range(6):
            optimizer.step()
            step_scheduler(config, scheduler, epoch)
            lrs.append(float(optimizer.param_groups[0]["lr"]))

        self.assertLess(lrs[0], lrs[1])
        self.assertLessEqual(lrs[1], lrs[2])
        self.assertLess(lrs[3], lrs[2])

    def test_step_warmup_cosine_learning_rate_warms_then_decays_per_step(self) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = {"training": {"scheduler": "step_warmup_cosine", "epochs": 2, "warmup_ratio": 0.25}}
        scheduler = build_scheduler(config, optimizer, steps_per_epoch=4)

        lrs = [float(optimizer.param_groups[0]["lr"])]
        for _ in range(8):
            optimizer.step()
            step_scheduler(config, scheduler, batch_level=True)
            lrs.append(float(optimizer.param_groups[0]["lr"]))

        self.assertGreater(lrs[1], lrs[0])
        self.assertGreaterEqual(max(lrs[:4]), lrs[1])
        self.assertLess(lrs[-1], max(lrs))

    def test_early_stopping_triggers_after_patience_without_min_delta_improvement(self) -> None:
        config = {
            "training": {
                "early_stopping_metric": "val_mean_auc",
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.001,
                "early_stopping_mode": "max",
            }
        }
        state = initialize_early_stopping(config)
        state = update_early_stopping_state(state, metric_value=0.7000, epoch=0)
        state = update_early_stopping_state(state, metric_value=0.7005, epoch=1)
        state = update_early_stopping_state(state, metric_value=0.7004, epoch=2)

        self.assertTrue(state["should_stop"])
        self.assertEqual(state["stopped_epoch"], 2)

    def test_test_evaluation_skips_gracefully_without_test_loader(self) -> None:
        config = {"training": {}}
        data_module = {"dataloaders": {"train": object(), "val": object()}}
        loss_fn = torch.nn.BCEWithLogitsLoss()
        result = evaluate_best_checkpoint_on_test_split(
            config=config,
            data_module=data_module,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
            label_names=["A"],
            checkpoint_path="unused.pt",
        )

        self.assertTrue(result["skipped"])
        self.assertIsNone(result["metrics"])

    def test_test_evaluation_uses_best_checkpoint_when_test_loader_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "vit_best_auc.pt"
            model = torch.nn.Linear(4, 2)
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            dataset = TensorDataset(torch.randn(2, 4), torch.zeros(2, 2))
            data_module = {"dataloaders": {"test": DataLoader(dataset, batch_size=1)}}
            loss_fn = torch.nn.BCEWithLogitsLoss()

            with mock.patch("train.build_model", return_value=torch.nn.Linear(4, 2)), mock.patch(
                "train.evaluate_epoch",
                return_value={"mean_auc": 0.75, "loss": 0.5},
            ) as evaluate_mock:
                result = evaluate_best_checkpoint_on_test_split(
                    config={"training": {"threshold": 0.5}},
                    data_module=data_module,
                    loss_fn=loss_fn,
                    device=torch.device("cpu"),
                    label_names=["A", "B"],
                    checkpoint_path=checkpoint_path,
                )

        self.assertFalse(result["skipped"])
        self.assertEqual(result["metrics"]["mean_auc"], 0.75)
        evaluate_mock.assert_called_once()

    def test_train_one_epoch_still_works_when_scaler_is_none(self) -> None:
        model = torch.nn.Linear(4, 2)
        loader = self._toy_loader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        metrics = train_one_epoch(
            config={"training": {"scheduler": "none"}},
            model=model,
            data_loader=loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=None,
            device=torch.device("cpu"),
            label_names=["A", "B"],
            scaler=None,
            progress_log_interval=None,
        )

        self.assertIn("loss", metrics)
        self.assertIn("mean_auc", metrics)

    def test_evaluate_epoch_still_works_when_use_amp_is_false(self) -> None:
        model = torch.nn.Linear(4, 2)
        loader = self._toy_loader()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        metrics = evaluate_epoch(
            model=model,
            data_loader=loader,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
            label_names=["A", "B"],
            use_amp=False,
            progress_log_interval=None,
        )

        self.assertIn("loss", metrics)
        self.assertIn("mean_auc", metrics)

    def test_use_amp_true_on_cpu_disables_amp_gracefully(self) -> None:
        model = torch.nn.Linear(4, 2)
        loader = self._toy_loader()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        metrics = evaluate_epoch(
            model=model,
            data_loader=loader,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
            label_names=["A", "B"],
            use_amp=True,
            progress_log_interval=None,
        )

        self.assertIn("loss", metrics)
        run_params = build_run_params(
            {"training": {"use_amp": True}, "data": {}, "model": {}, "project": {}},
            {"label_names": [], "positive_counts": {}, "negative_counts": {}, "prevalence": {}, "pos_weight": {}},
            amp_enabled=False,
        )
        self.assertTrue(run_params["use_amp"])
        self.assertFalse(run_params["amp_enabled"])


if __name__ == "__main__":
    unittest.main()
