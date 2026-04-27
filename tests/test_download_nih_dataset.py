import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.download_nih_dataset import (
    download_and_prepare_nih_dataset,
    export_nih_annotation_files,
    inspect_nih_download_layout,
    locate_nih_dataset_root,
    resolve_kaggle_dataset_reference,
)


class DownloadNihDatasetTests(unittest.TestCase):
    def test_resolve_kaggle_dataset_reference_from_url(self) -> None:
        handle, version = resolve_kaggle_dataset_reference(
            "https://www.kaggle.com/datasets/nih-chest-xrays/data/download?datasetVersionNumber=3"
        )
        self.assertEqual(handle, "nih-chest-xrays/data")
        self.assertEqual(version, 3)

    def test_locate_and_prepare_downloaded_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            download_dir = root / "downloads"
            dataset_root = download_dir / "nih-chest-xrays"
            images_dir = dataset_root / "images_001" / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "00000001.png").write_bytes(b"fake-png")
            (dataset_root / "Data_Entry_2017.csv").write_text("Image Index,Finding Labels\n00000001.png,No Finding\n")
            (dataset_root / "BBox_List_2017.csv").write_text("Image Index,Finding Label,Bbox x,Bbox y,Bbox w,Bbox h\n")
            (dataset_root / "train_val_list.txt").write_text("00000001.png\n")
            (dataset_root / "test_list.txt").write_text("")

            located_root = locate_nih_dataset_root(download_dir)
            self.assertEqual(located_root, dataset_root.resolve())

            layout = inspect_nih_download_layout(located_root)
            self.assertEqual(len(layout["image_archives"]), 1)
            self.assertTrue(layout["image_directories"][0].endswith("images_001/images"))

            annotations_dir = root / "annotations"
            exported = export_nih_annotation_files(located_root, annotations_dir)
            exported_names = sorted(path.name for path in exported)
            self.assertIn("Data_Entry_2017.csv", exported_names)
            self.assertIn("BBox_List_2017.csv", exported_names)
            self.assertIn("BBox_list_2017.csv", exported_names)
            self.assertIn("train_val_list.txt", exported_names)
            self.assertIn("test_list.txt", exported_names)

    def test_download_and_prepare_uses_kaggle_download_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "kaggle-cache" / "nih-chest-xrays"
            images_dir = dataset_root / "images_001" / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "00000001.png").write_bytes(b"fake-png")
            (dataset_root / "Data_Entry_2017.csv").write_text("Image Index,Finding Labels\n00000001.png,No Finding\n")
            (dataset_root / "train_val_list.txt").write_text("00000001.png\n")
            (dataset_root / "test_list.txt").write_text("")

            class FakeKaggleHub:
                @staticmethod
                def dataset_download(dataset, output_dir, force_download=False):
                    return str(dataset_root)

            with patch("src.download_nih_dataset._import_kagglehub", return_value=FakeKaggleHub):
                summary = download_and_prepare_nih_dataset(
                    dataset="https://www.kaggle.com/datasets/nih-chest-xrays/data/download?datasetVersionNumber=3",
                    output_dir=root / "raw",
                    annotations_dir=root / "annotations",
                )

            self.assertEqual(summary["dataset_handle"], "nih-chest-xrays/data")
            self.assertEqual(summary["dataset_version"], 3)
            self.assertEqual(Path(summary["dataset_root"]), dataset_root.resolve())
            self.assertTrue((root / "annotations" / "Data_Entry_2017.csv").exists())
            self.assertEqual(len(summary["image_archives"]), 1)
            self.assertEqual(len(summary["image_directories"]), 1)


if __name__ == "__main__":
    unittest.main()
