from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


NIH_CHEST_XRAY_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NIH_NO_FINDING_LABEL = "No Finding"
NIH_LABEL_ALIASES = {
    "No findings": NIH_NO_FINDING_LABEL,
    "No Findings": NIH_NO_FINDING_LABEL,
    "Pleural_thickening": "Pleural_Thickening",
}
NIH_METADATA_FILENAMES: tuple[str, ...] = (
    "Data_Entry_2017.csv",
    "Data_entry_2017.csv",
)
NIH_TRAIN_SPLIT_FILENAMES: tuple[str, ...] = ("train_val_list.txt",)
NIH_TEST_SPLIT_FILENAMES: tuple[str, ...] = ("test_list.txt",)


def normalize_nih_label(label: str) -> str:
    normalized = label.strip()
    return NIH_LABEL_ALIASES.get(normalized, normalized)


def parse_nih_labels(label_string: str) -> list[str]:
    if pd.isna(label_string):
        return []

    parsed = [normalize_nih_label(label) for label in str(label_string).split("|") if label.strip()]
    return [label for label in parsed if label != NIH_NO_FINDING_LABEL]


def encode_labels(label_string: str) -> dict[str, int]:
    active_labels = set(parse_nih_labels(label_string))
    return {label: int(label in active_labels) for label in NIH_CHEST_XRAY_LABELS}


def _build_image_index(raw_dir: Path) -> dict[str, Path]:
    supported_suffixes = {".png", ".jpg", ".jpeg"}
    image_paths = [
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    return {path.name: path.resolve() for path in image_paths}


def _read_split_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _resolve_existing_file(directory: Path, candidate_names: Sequence[str]) -> Path:
    for candidate_name in candidate_names:
        candidate_path = directory / candidate_name
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find any of {list(candidate_names)} under {directory}"
    )


def _resolve_optional_file(directory: Path, candidate_names: Sequence[str]) -> Optional[Path]:
    for candidate_name in candidate_names:
        candidate_path = directory / candidate_name
        if candidate_path.exists():
            return candidate_path

    return None


def _resolve_column_name(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized_columns = {column.strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def sanity_check_nih_dataset(
    raw_dir: str | Path,
    annotations_dir: str | Path,
    metadata_filename: Optional[str] = None,
) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    annotations_dir = Path(annotations_dir)
    metadata_path = (
        annotations_dir / metadata_filename
        if metadata_filename is not None
        else _resolve_existing_file(annotations_dir, NIH_METADATA_FILENAMES)
    )

    metadata = pd.read_csv(metadata_path)
    image_index_column = _resolve_column_name(metadata, ("Image Index",))
    finding_labels_column = _resolve_column_name(metadata, ("Finding Labels",))
    patient_id_column = _resolve_column_name(metadata, ("Patient ID",))
    width_column = _resolve_column_name(metadata, ("OriginalImageWidth",))
    height_column = _resolve_column_name(metadata, ("OriginalImageHeight",))
    view_position_column = _resolve_column_name(metadata, ("View Position",))

    if image_index_column is None or finding_labels_column is None:
        raise ValueError("NIH metadata must include 'Image Index' and 'Finding Labels' columns")

    image_index = _build_image_index(raw_dir)
    image_names = metadata[image_index_column].astype(str)
    matched_mask = image_names.isin(image_index.keys())
    parsed_labels = metadata[finding_labels_column].fillna("").apply(parse_nih_labels)
    all_labels = sorted({label for labels in parsed_labels for label in labels})
    unknown_labels = sorted(set(all_labels).difference(NIH_CHEST_XRAY_LABELS))

    summary: dict[str, object] = {
        "metadata_path": str(metadata_path),
        "metadata_rows": int(len(metadata)),
        "matched_images": int(matched_mask.sum()),
        "missing_images": int((~matched_mask).sum()),
        "num_raw_images_indexed": int(len(image_index)),
        "num_unique_patients": int(metadata[patient_id_column].nunique()) if patient_id_column else None,
        "num_disease_labels": len(NIH_CHEST_XRAY_LABELS),
        "observed_labels": all_labels,
        "unknown_labels": unknown_labels,
        "no_finding_count": int(
            metadata[finding_labels_column]
            .fillna("")
            .astype(str)
            .apply(lambda value: NIH_NO_FINDING_LABEL in [normalize_nih_label(label) for label in value.split("|")])
            .sum()
        ),
    }

    if width_column is not None and height_column is not None:
        summary["image_dimensions_in_metadata"] = (
            metadata[[width_column, height_column]]
            .drop_duplicates()
            .rename(columns={width_column: "width", height_column: "height"})
            .to_dict(orient="records")
        )

    if view_position_column is not None:
        summary["view_positions"] = (
            metadata[view_position_column].fillna("UNKNOWN").value_counts().sort_index().to_dict()
        )

    return summary


def build_nih_manifests(
    raw_dir: str | Path,
    annotations_dir: str | Path,
    manifest_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    metadata_filename: Optional[str] = None,
    train_list_filename: Optional[str] = None,
    test_list_filename: Optional[str] = None,
) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    annotations_dir = Path(annotations_dir)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        annotations_dir / metadata_filename
        if metadata_filename is not None
        else _resolve_existing_file(annotations_dir, NIH_METADATA_FILENAMES)
    )

    metadata = pd.read_csv(metadata_path)
    image_index_column = _resolve_column_name(metadata, ("Image Index",))
    finding_labels_column = _resolve_column_name(metadata, ("Finding Labels",))
    patient_id_column = _resolve_column_name(metadata, ("Patient ID",))
    follow_up_column = _resolve_column_name(metadata, ("Follow-up #",))
    patient_age_column = _resolve_column_name(metadata, ("Patient Age",))
    patient_gender_column = _resolve_column_name(metadata, ("Patient Gender",))
    view_position_column = _resolve_column_name(metadata, ("View Position",))
    original_width_column = _resolve_column_name(metadata, ("OriginalImageWidth",))
    original_height_column = _resolve_column_name(metadata, ("OriginalImageHeight",))

    if image_index_column is None or finding_labels_column is None:
        raise ValueError("Metadata CSV must include 'Image Index' and 'Finding Labels' columns")

    image_index = _build_image_index(raw_dir)
    if not image_index:
        raise FileNotFoundError(f"No image files were found under {raw_dir}")

    metadata = metadata.copy()
    metadata["image_name"] = metadata[image_index_column].astype(str)
    metadata["labels"] = metadata[finding_labels_column].fillna("").astype(str)
    metadata["patient_id"] = (
        metadata[patient_id_column] if patient_id_column is not None else pd.Series([-1] * len(metadata))
    )
    metadata["image_path"] = metadata["image_name"].map(lambda image_name: str(image_index.get(image_name, "")))
    metadata = metadata[metadata["image_path"] != ""].reset_index(drop=True)

    if metadata.empty:
        raise ValueError("None of the NIH metadata entries could be matched to files in data/raw")

    encoded_labels = metadata["labels"].apply(encode_labels).apply(pd.Series)
    metadata = pd.concat([metadata, encoded_labels], axis=1)

    train_list_path = (
        annotations_dir / train_list_filename
        if train_list_filename is not None
        else _resolve_optional_file(annotations_dir, NIH_TRAIN_SPLIT_FILENAMES)
    )
    test_list_path = (
        annotations_dir / test_list_filename
        if test_list_filename is not None
        else _resolve_optional_file(annotations_dir, NIH_TEST_SPLIT_FILENAMES)
    )
    has_official_train_list = train_list_path is not None and train_list_path.exists()
    has_official_test_list = test_list_path is not None and test_list_path.exists()

    if has_official_train_list:
        train_val_names = _read_split_list(train_list_path)
    else:
        train_val_names = set(metadata["image_name"].tolist())

    if has_official_test_list:
        test_names = _read_split_list(test_list_path)
    else:
        test_names = set()

    metadata["is_test"] = metadata["image_name"].isin(test_names)
    train_val_df = metadata[metadata["image_name"].isin(train_val_names) & ~metadata["is_test"]].copy()
    test_df = metadata[metadata["is_test"]].copy()

    if train_val_df.empty:
        train_val_df = metadata[~metadata["is_test"]].copy()

    if train_val_df.empty:
        raise ValueError("No train/validation examples were available after split resolution")

    patient_ids = train_val_df["patient_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    if len(patient_ids) <= 1:
        val_patients = set()
    else:
        val_count = max(1, int(round(len(patient_ids) * val_fraction)))
        val_count = min(val_count, len(patient_ids) - 1)
        val_patients = set(patient_ids[:val_count])

    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].copy()
    train_df = train_val_df[~train_val_df["patient_id"].isin(val_patients)].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Reduce val_fraction or verify patient IDs in metadata.")

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    metadata_columns_to_keep = []
    optional_columns = [
        ("follow_up", follow_up_column),
        ("patient_age", patient_age_column),
        ("patient_gender", patient_gender_column),
        ("view_position", view_position_column),
        ("original_width", original_width_column),
        ("original_height", original_height_column),
    ]
    for export_name, column_name in optional_columns:
        if column_name is not None:
            metadata[export_name] = metadata[column_name]
            metadata_columns_to_keep.append(export_name)

    exported_columns = [
        "image_name",
        "image_path",
        "patient_id",
        "labels",
        *metadata_columns_to_keep,
        *NIH_CHEST_XRAY_LABELS,
    ]

    written_paths: dict[str, Path] = {}
    for split_name, frame in split_frames.items():
        if frame.empty and split_name == "test":
            continue

        export_frame = frame.loc[:, exported_columns].copy()
        export_frame.insert(0, "split", split_name)
        export_path = manifest_dir / f"{split_name}.csv"
        export_frame.to_csv(export_path, index=False)
        written_paths[split_name] = export_path

    return written_paths


def default_normalization(num_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if num_channels == 1:
        return (0.5,), (0.25,)
    if num_channels == 3:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    raise ValueError(f"Unsupported number of channels: {num_channels}")


def build_image_transform(
    image_size: int,
    num_channels: int = 1,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> transforms.Compose:
    if mean is None or std is None:
        mean, std = default_normalization(num_channels)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class NihChestXrayDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        num_channels: int = 1,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path)
        self.image_size = image_size
        self.num_channels = num_channels
        self.transform = build_image_transform(
            image_size=image_size,
            num_channels=num_channels,
            mean=mean,
            std=std,
        )
        self.label_columns = list(NIH_CHEST_XRAY_LABELS)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"])
        image = image.convert("L" if self.num_channels == 1 else "RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(row[self.label_columns].to_numpy(dtype="float32"))
        return image_tensor, label_tensor


def build_dataloaders(
    config: dict,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> dict[str, DataLoader]:
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    image_size = int(data_config.get("image_size", 224))
    num_channels = int(data_config.get("num_channels", 1))
    batch_size = int(training_config.get("batch_size", data_config.get("batch_size", 32)))
    num_workers = int(data_config.get("num_workers", 4))

    datasets = {
        "train": NihChestXrayDataset(
            manifest_path=data_config["train_manifest"],
            image_size=image_size,
            num_channels=num_channels,
            mean=mean,
            std=std,
        ),
        "val": NihChestXrayDataset(
            manifest_path=data_config["val_manifest"],
            image_size=image_size,
            num_channels=num_channels,
            mean=mean,
            std=std,
        ),
    }

    if "test_manifest" in data_config and Path(data_config["test_manifest"]).exists():
        datasets["test"] = NihChestXrayDataset(
            manifest_path=data_config["test_manifest"],
            image_size=image_size,
            num_channels=num_channels,
            mean=mean,
            std=std,
        )

    dataloaders = {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split_name, dataset in datasets.items()
    }

    return dataloaders


def build_nih_data_module(config: dict) -> dict[str, object]:
    data_config = config.get("data", {})
    manifest_dir = Path(data_config["manifest_dir"])
    annotations_dir = Path(data_config.get("annotations_dir", manifest_dir.parent / "annotations"))

    train_manifest = Path(data_config["train_manifest"])
    val_manifest = Path(data_config["val_manifest"])
    test_manifest = Path(data_config["test_manifest"])
    if not train_manifest.exists() or not val_manifest.exists():
        build_nih_manifests(
            raw_dir=data_config["raw_dir"],
            annotations_dir=annotations_dir,
            manifest_dir=manifest_dir,
            val_fraction=float(data_config.get("val_fraction", 0.1)),
            seed=int(config.get("project", {}).get("seed", 42)),
        )

    dataloaders = build_dataloaders(config)
    return {"labels": NIH_CHEST_XRAY_LABELS, "dataloaders": dataloaders, "test_manifest": test_manifest}
