import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import (
    NIH_CHEST_XRAY_LABELS,
    NihChestXrayDataset,
    build_eval_transform,
    build_nih_manifests,
    build_train_transform,
    compute_pos_weight_from_manifest,
    sanity_check_nih_dataset,
)
from models.components import PatchEmbedding, ViTInputEmbedding


class DataPreparationTests(unittest.TestCase):
    def test_manifest_builder_and_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw" / "images"
            annotations_dir = root / "annotations"
            manifests_dir = root / "manifests"
            raw_dir.mkdir(parents=True)
            annotations_dir.mkdir(parents=True)

            metadata_rows = [
                {
                    "Image Index": "image_1.png",
                    "Finding Labels": "Atelectasis|Effusion",
                    "Patient ID": 1,
                },
                {
                    "Image Index": "image_2.png",
                    "Finding Labels": "No Finding",
                    "Patient ID": 2,
                },
                {
                    "Image Index": "image_3.png",
                    "Finding Labels": "Mass",
                    "Patient ID": 3,
                },
            ]
            pd.DataFrame(metadata_rows).to_csv(annotations_dir / "Data_Entry_2017.csv", index=False)
            (annotations_dir / "train_val_list.txt").write_text("image_1.png\nimage_2.png\n")
            (annotations_dir / "test_list.txt").write_text("image_3.png\n")

            for image_name in ["image_1.png", "image_2.png", "image_3.png"]:
                image = Image.new("L", (1024, 1024), color=128)
                image.save(raw_dir / image_name)

            manifest_paths = build_nih_manifests(
                raw_dir=root / "raw",
                annotations_dir=annotations_dir,
                manifest_dir=manifests_dir,
                val_fraction=0.5,
                seed=7,
            )

            self.assertTrue((manifests_dir / "train.csv").exists())
            self.assertTrue((manifests_dir / "val.csv").exists())
            self.assertTrue((manifests_dir / "test.csv").exists())

            train_df = pd.read_csv(manifest_paths["train"])
            self.assertEqual(len([label for label in NIH_CHEST_XRAY_LABELS if label in train_df.columns]), 14)

            dataset = NihChestXrayDataset(manifest_paths["train"], image_size=224, num_channels=1)
            image_tensor, label_tensor = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (1, 224, 224))
            self.assertEqual(tuple(label_tensor.shape), (14,))

            summary = sanity_check_nih_dataset(root / "raw", annotations_dir, metadata_filename="Data_Entry_2017.csv")
            self.assertEqual(summary["metadata_rows"], 3)
            self.assertEqual(summary["matched_images"], 3)
            self.assertEqual(summary["missing_images"], 0)

            pos_weight_stats = compute_pos_weight_from_manifest(manifest_paths["train"], pos_weight_clamp=50)
            train_manifest_df = pd.read_csv(manifest_paths["train"])
            self.assertEqual(tuple(pos_weight_stats["pos_weight_tensor"].shape), (14,))
            self.assertEqual(
                pos_weight_stats["positive_counts"]["Atelectasis"],
                int(train_manifest_df["Atelectasis"].sum()),
            )
            self.assertEqual(
                pos_weight_stats["negative_counts"]["Atelectasis"],
                len(train_manifest_df) - int(train_manifest_df["Atelectasis"].sum()),
            )

    def test_vit_input_embedding_shape(self) -> None:
        module = ViTInputEmbedding(
            image_size=224,
            patch_size=16,
            in_channels=1,
            embed_dim=192,
            dropout=0.1,
        )
        batch = torch.randn(2, 1, 224, 224)
        tokens = module(batch)
        self.assertEqual(tuple(tokens.shape), (2, 197, 192))

    def test_patch_embedding_linear_projection_shape(self) -> None:
        module = PatchEmbedding(
            image_size=224,
            patch_size=16,
            in_channels=1,
            embed_dim=192,
        )
        batch = torch.randn(2, 1, 224, 224)
        patch_tokens = module(batch)

        self.assertEqual(module.num_patches, 196)
        self.assertEqual(tuple(patch_tokens.shape), (2, 196, 192))
        self.assertEqual(tuple(module.projection.weight.shape), (192, 1, 16, 16))

    def test_vit_input_embedding_adds_cls_token_and_position_embeddings(self) -> None:
        module = ViTInputEmbedding(
            image_size=224,
            patch_size=16,
            in_channels=1,
            embed_dim=192,
            dropout=0.0,
        )
        batch = torch.randn(2, 1, 224, 224)

        patch_tokens = module.patch_embed(batch)
        output_tokens = module(batch)
        expected_cls = module.cls_token.expand(batch.size(0), -1, -1) + module.positional_embedding[:, :1, :]
        expected_patch_tokens = patch_tokens + module.positional_embedding[:, 1:, :]

        self.assertEqual(tuple(output_tokens.shape), (2, 197, 192))
        self.assertTrue(torch.allclose(output_tokens[:, :1, :], expected_cls, atol=1e-6))
        self.assertTrue(torch.allclose(output_tokens[:, 1:, :], expected_patch_tokens, atol=1e-6))

    def test_train_and_eval_transforms_split_augmentation_correctly(self) -> None:
        train_transform = build_train_transform(
            image_size=224,
            num_channels=1,
            augmentation_config={
                "enabled": True,
                "allow_horizontal_flip": True,
                "horizontal_flip_prob": 0.5,
                "rotation_degrees": 10,
                "color_jitter_brightness": 0.2,
                "color_jitter_contrast": 0.2,
            },
        )
        eval_transform = build_eval_transform(image_size=224, num_channels=1)

        train_transform_names = [type(step).__name__ for step in train_transform.transforms]
        eval_transform_names = [type(step).__name__ for step in eval_transform.transforms]

        self.assertEqual(
            train_transform_names,
            ["Resize", "RandomHorizontalFlip", "RandomRotation", "ColorJitter", "ToTensor", "Normalize"],
        )
        self.assertEqual(
            eval_transform_names,
            ["Resize", "ToTensor", "Normalize"],
        )

    def test_horizontal_flip_requires_explicit_allow_flag(self) -> None:
        train_transform = build_train_transform(
            image_size=224,
            num_channels=1,
            augmentation_config={
                "enabled": True,
                "allow_horizontal_flip": False,
                "horizontal_flip_prob": 0.5,
                "rotation_degrees": 10,
            },
        )
        train_transform_names = [type(step).__name__ for step in train_transform.transforms]
        self.assertEqual(
            train_transform_names,
            ["Resize", "RandomRotation", "ToTensor", "Normalize"],
        )


if __name__ == "__main__":
    unittest.main()
