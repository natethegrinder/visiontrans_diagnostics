import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import NIH_CHEST_XRAY_LABELS, build_nih_data_module
from models import build_model


def _create_mock_nih_dataset(root: Path) -> dict:
    raw_dir = root / "raw"
    annotations_dir = root / "annotations"
    manifests_dir = root / "manifests"
    nested_images_dir = raw_dir / "images_001" / "images"
    nested_images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    metadata_rows = [
        {
            "Image Index": "image_1.png",
            "Finding Labels": "Atelectasis|Effusion",
            "Patient ID": 1,
            "View Position": "PA",
            "OriginalImageWidth": 1024,
            "OriginalImageHeight": 1024,
        },
        {
            "Image Index": "image_2.png",
            "Finding Labels": "No Finding",
            "Patient ID": 2,
            "View Position": "AP",
            "OriginalImageWidth": 1024,
            "OriginalImageHeight": 1024,
        },
        {
            "Image Index": "image_3.png",
            "Finding Labels": "Mass",
            "Patient ID": 3,
            "View Position": "PA",
            "OriginalImageWidth": 1024,
            "OriginalImageHeight": 1024,
        },
        {
            "Image Index": "image_4.png",
            "Finding Labels": "Nodule|Infiltration",
            "Patient ID": 4,
            "View Position": "PA",
            "OriginalImageWidth": 1024,
            "OriginalImageHeight": 1024,
        },
    ]
    pd.DataFrame(metadata_rows).to_csv(annotations_dir / "Data_Entry_2017.csv", index=False)
    (annotations_dir / "train_val_list.txt").write_text("image_1.png\nimage_2.png\nimage_4.png\n")
    (annotations_dir / "test_list.txt").write_text("image_3.png\n")

    for image_name in ["image_1.png", "image_2.png", "image_3.png", "image_4.png"]:
        image = Image.new("L", (1024, 1024), color=128)
        image.save(nested_images_dir / image_name)

    return {
        "project": {"seed": 42},
        "data": {
            "dataset": "nih_chest_xray",
            "raw_dir": str(raw_dir),
            "annotations_dir": str(annotations_dir),
            "manifest_dir": str(manifests_dir),
            "train_manifest": str(manifests_dir / "train.csv"),
            "val_manifest": str(manifests_dir / "val.csv"),
            "test_manifest": str(manifests_dir / "test.csv"),
            "image_size": 224,
            "num_classes": 14,
            "num_channels": 1,
            "batch_size": 2,
            "num_workers": 0,
            "val_fraction": 0.33,
        },
        "model": {
            "family": "vit",
            "architecture": "vit_tiny",
            "patch_size": 16,
            "hidden_dim": 192,
            "num_heads": 3,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "training": {"batch_size": 2},
    }


class ViTDataPipelineTests(unittest.TestCase):
    def test_data_module_creates_manifests_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _create_mock_nih_dataset(Path(tmp_dir))

            data_module = build_nih_data_module(config)
            dataloaders = data_module["dataloaders"]

            self.assertTrue(Path(config["data"]["train_manifest"]).exists())
            self.assertTrue(Path(config["data"]["val_manifest"]).exists())
            self.assertTrue(Path(config["data"]["test_manifest"]).exists())
            self.assertEqual(data_module["labels"], NIH_CHEST_XRAY_LABELS)

            images, labels = next(iter(dataloaders["train"]))
            self.assertEqual(tuple(images.shape[1:]), (1, 224, 224))
            self.assertEqual(tuple(labels.shape[1:]), (14,))

    def test_vit_model_exposes_expected_token_shape_before_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _create_mock_nih_dataset(Path(tmp_dir))
            data_module = build_nih_data_module(config)
            model = build_model(config)

            images, _ = next(iter(data_module["dataloaders"]["train"]))
            tokens = model.input_embedding(images)

            self.assertEqual(tuple(tokens.shape), (images.shape[0], 197, 192))

    def test_vit_model_forward_pass_returns_multilabel_logits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _create_mock_nih_dataset(Path(tmp_dir))
            data_module = build_nih_data_module(config)
            model = build_model(config)

            images, _ = next(iter(data_module["dataloaders"]["train"]))
            logits = model(images)

            self.assertEqual(tuple(logits.shape), (images.shape[0], 14))


if __name__ == "__main__":
    unittest.main()
