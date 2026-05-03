import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import build_scheduler, initialize_scheduler_learning_rate, step_scheduler


class TrainUtilsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
