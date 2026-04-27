from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import parse_qs, urlparse


DEFAULT_KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/nih-chest-xrays/data/download?datasetVersionNumber=3"
)
DEFAULT_DATASET_HANDLE = "nih-chest-xrays/data"
IMAGE_ARCHIVE_PREFIX = "images_"


@dataclass(frozen=True)
class ExportRule:
    candidates: tuple[str, ...]
    destination: str
    required: bool = True
    aliases: tuple[str, ...] = ()


ANNOTATION_EXPORT_RULES: tuple[ExportRule, ...] = (
    ExportRule(
        candidates=("Data_Entry_2017.csv", "Data_entry_2017.csv"),
        destination="Data_Entry_2017.csv",
    ),
    ExportRule(
        candidates=("BBox_List_2017.csv", "BBox_list_2017.csv"),
        destination="BBox_List_2017.csv",
        required=False,
        aliases=("BBox_list_2017.csv",),
    ),
    ExportRule(candidates=("train_val_list.txt",), destination="train_val_list.txt"),
    ExportRule(candidates=("test_list.txt",), destination="test_list.txt"),
)


@dataclass
class NihDownloadSummary:
    dataset_handle: str
    dataset_version: Optional[int]
    kaggle_reference: str
    download_path: str
    dataset_root: str
    raw_dir: str
    annotations_dir: str
    image_archives: list[str] = field(default_factory=list)
    image_directories: list[str] = field(default_factory=list)
    exported_annotation_files: list[str] = field(default_factory=list)


def resolve_kaggle_dataset_reference(dataset: str, version: Optional[int] = None) -> tuple[str, Optional[int]]:
    dataset = dataset.strip()
    if dataset.startswith("http://") or dataset.startswith("https://"):
        parsed_url = urlparse(dataset)
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if len(path_parts) < 3 or path_parts[0] != "datasets":
            raise ValueError(f"Unsupported Kaggle dataset URL: {dataset}")

        handle = f"{path_parts[1]}/{path_parts[2]}"
        if version is None:
            query_version = parse_qs(parsed_url.query).get("datasetVersionNumber")
            if query_version:
                version = int(query_version[0])
        return handle, version

    handle = dataset.replace("https://www.kaggle.com/datasets/", "").strip("/")
    if not handle or "/" not in handle:
        raise ValueError(
            "Dataset must be a Kaggle dataset URL or a handle like 'nih-chest-xrays/data'"
        )
    return handle, version


def build_kaggle_download_reference(dataset_handle: str, version: Optional[int]) -> str:
    if version is None:
        return dataset_handle
    return f"{dataset_handle}/versions/{version}"


def _import_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the NIH dataset. Install it with 'pip install kagglehub'."
        ) from exc
    return kagglehub


def _find_existing_file(directory: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        candidate_path = directory / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _looks_like_dataset_root(candidate: Path) -> bool:
    has_metadata = _find_existing_file(candidate, ("Data_Entry_2017.csv", "Data_entry_2017.csv")) is not None
    has_image_archives = any(path.is_dir() and path.name.startswith(IMAGE_ARCHIVE_PREFIX) for path in candidate.iterdir())
    return has_metadata and has_image_archives


def locate_nih_dataset_root(download_path: str | Path) -> Path:
    download_path = Path(download_path).resolve()
    if not download_path.exists():
        raise FileNotFoundError(f"Downloaded path does not exist: {download_path}")

    if download_path.is_file():
        raise ValueError(f"Expected a directory from Kaggle download, received file: {download_path}")

    if _looks_like_dataset_root(download_path):
        return download_path

    candidates: list[Path] = []
    for metadata_name in ("Data_Entry_2017.csv", "Data_entry_2017.csv"):
        for metadata_path in download_path.rglob(metadata_name):
            parent = metadata_path.parent
            if _looks_like_dataset_root(parent):
                candidates.append(parent)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate the NIH dataset root under {download_path}. "
            "Expected a directory containing Data_Entry_2017.csv and images_00x folders."
        )

    return sorted(set(candidates), key=lambda path: len(path.parts))[0]


def inspect_nih_download_layout(dataset_root: str | Path) -> dict[str, list[str]]:
    dataset_root = Path(dataset_root)
    image_archives = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith(IMAGE_ARCHIVE_PREFIX)
    )
    if not image_archives:
        raise ValueError(f"No '{IMAGE_ARCHIVE_PREFIX}###' archive folders were found under {dataset_root}")

    image_directories: list[str] = []
    for archive_dir in image_archives:
        nested_images_dir = archive_dir / "images"
        image_directories.append(str(nested_images_dir if nested_images_dir.is_dir() else archive_dir))

    return {
        "image_archives": [str(path) for path in image_archives],
        "image_directories": image_directories,
    }


def export_nih_annotation_files(
    dataset_root: str | Path,
    annotations_dir: str | Path,
    overwrite: bool = True,
) -> list[Path]:
    dataset_root = Path(dataset_root)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    exported_files: list[Path] = []
    for rule in ANNOTATION_EXPORT_RULES:
        source_path = _find_existing_file(dataset_root, rule.candidates)
        if source_path is None:
            if rule.required:
                raise FileNotFoundError(
                    f"Missing required NIH file in downloaded dataset root: one of {list(rule.candidates)}"
                )
            continue

        destination_path = annotations_dir / rule.destination
        if overwrite or not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        exported_files.append(destination_path)

        for alias_name in rule.aliases:
            alias_path = annotations_dir / alias_name
            if overwrite or not alias_path.exists():
                shutil.copy2(source_path, alias_path)
            exported_files.append(alias_path)

    return exported_files


def download_and_prepare_nih_dataset(
    dataset: str = DEFAULT_KAGGLE_DATASET_URL,
    version: Optional[int] = None,
    output_dir: str | Path = "data/raw",
    annotations_dir: str | Path = "data/annotations",
    force_download: bool = False,
) -> dict[str, object]:
    dataset_handle, resolved_version = resolve_kaggle_dataset_reference(dataset, version)
    kaggle_reference = build_kaggle_download_reference(dataset_handle, resolved_version)

    kagglehub = _import_kagglehub()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    download_path = Path(
        kagglehub.dataset_download(
            kaggle_reference,
            output_dir=str(output_dir),
            force_download=force_download,
        )
    ).resolve()
    dataset_root = locate_nih_dataset_root(download_path)
    layout_summary = inspect_nih_download_layout(dataset_root)
    exported_files = export_nih_annotation_files(dataset_root, annotations_dir)

    summary = NihDownloadSummary(
        dataset_handle=dataset_handle,
        dataset_version=resolved_version,
        kaggle_reference=kaggle_reference,
        download_path=str(download_path),
        dataset_root=str(dataset_root),
        raw_dir=str(output_dir.resolve()),
        annotations_dir=str(annotations_dir.resolve()),
        image_archives=layout_summary["image_archives"],
        image_directories=layout_summary["image_directories"],
        exported_annotation_files=[str(path) for path in exported_files],
    )
    return summary.__dict__


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and stage the NIH Chest X-ray dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET_URL,
        help="Kaggle dataset URL or handle. Default is the NIH Chest X-ray dataset URL.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional Kaggle dataset version. If omitted and a full URL is provided, the version is parsed from it.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where the Kaggle dataset will be downloaded.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="data/annotations",
        help="Directory where Data_Entry_2017.csv, split lists, and bbox CSV will be copied.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if Kaggle has it cached locally.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = download_and_prepare_nih_dataset(
        dataset=args.dataset,
        version=args.version,
        output_dir=args.output_dir,
        annotations_dir=args.annotations_dir,
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
