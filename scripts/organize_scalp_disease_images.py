"""Organize scalp disease images into label-based folders.

This script can:
- move or copy images into datasets/hair_only/scalp_diseases/<label>/
- read labels from a CSV file
- derive labels from filenames like alopecia-areata_001.jpg
- derive labels from parent folder names when the source already has class folders

Examples:
    python scripts/organize_scalp_disease_images.py \
        --source datasets/hair_only/incoming_images \
        --mode filename

    python scripts/organize_scalp_disease_images.py \
        --source datasets/hair_only/incoming_images \
        --mode csv \
        --csv datasets/hair_only/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Iterator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_label(label: str) -> str:
    """Convert a label into a safe folder name."""

    cleaned = label.strip().lower().replace("_", "-").replace(" ", "-")
    cleaned = re.sub(r"[^a-z0-9-]+", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        raise ValueError(f"Could not normalize label from {label!r}")
    return cleaned


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def unique_destination_path(destination: Path) -> Path:
    """Append a numeric suffix if the destination already exists."""

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    index = 1
    while True:
        candidate = destination.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def find_source_file(source_root: Path, filename: str, dest_root: Path | None = None) -> Path | None:
    """Resolve a CSV filename against the source tree."""

    direct = source_root / filename
    if direct.exists() and direct.is_file():
        return direct

    matches = []
    for candidate in source_root.rglob(filename):
        if not candidate.is_file():
            continue
        if dest_root is not None and is_within(candidate, dest_root):
            continue
        matches.append(candidate)

    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] Multiple matches for {filename!r}; using {matches[0]}")
    return matches[0]


def iter_csv_items(csv_path: Path, filename_column: str, label_column: str) -> Iterator[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")

        missing = [name for name in (filename_column, label_column) if name not in reader.fieldnames]
        if missing:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"CSV file {csv_path} is missing columns: {', '.join(missing)}. Available columns: {available}"
            )

        for row in reader:
            filename = (row.get(filename_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not filename or not label:
                continue
            yield filename, label


def iter_filename_items(source_root: Path, separator: str, recursive: bool) -> Iterator[tuple[Path, str]]:
    if recursive:
        candidates = source_root.rglob("*")
    else:
        candidates = source_root.iterdir()

    for path in candidates:
        if not is_image_file(path):
            continue
        if separator not in path.stem:
            continue
        label = path.stem.split(separator, 1)[0]
        yield path, label


def iter_parent_items(source_root: Path) -> Iterator[tuple[Path, str]]:
    for path in source_root.rglob("*"):
        if not is_image_file(path):
            continue
        yield path, path.parent.name


def transfer_image(source: Path, destination: Path, action: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_destination_path(destination)

    if action == "copy":
        shutil.copy2(source, destination)
    elif action == "move":
        shutil.move(str(source), str(destination))
    else:
        raise ValueError(f"Unsupported action: {action}")

    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize scalp disease images into datasets/hair_only/scalp_diseases/<label>/ folders."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/hair_only/incoming_images"),
        help="Source directory containing the images to organize.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("datasets/hair_only/scalp_diseases"),
        help="Destination root where label folders will be created.",
    )
    parser.add_argument(
        "--mode",
        choices=("csv", "filename", "parent"),
        default="filename",
        help="How to get the label for each image.",
    )
    parser.add_argument(
        "--action",
        choices=("move", "copy"),
        default="move",
        help="Move files into the destination or copy them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without changing files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV file used when --mode csv is selected.",
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        help="CSV column containing the image filename.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column containing the disease label.",
    )
    parser.add_argument(
        "--separator",
        default="_",
        help="Separator used to extract the label from filenames in filename mode.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan the source folder in filename mode.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_root = args.source
    dest_root = args.dest

    if not source_root.exists():
        parser.error(
            f"Source directory not found: {source_root}. "
            "Create it, or pass --source to the folder that contains your images."
        )

    dest_root.mkdir(parents=True, exist_ok=True)

    operations: list[tuple[Path, str]] = []

    if args.mode == "csv":
        if args.csv is None:
            parser.error("--csv is required when --mode csv is selected")

        for filename, label in iter_csv_items(args.csv, args.filename_column, args.label_column):
            source_file = find_source_file(source_root, filename, dest_root=dest_root)
            if source_file is None:
                print(f"[WARN] Missing file skipped: {filename}")
                continue
            operations.append((source_file, normalize_label(label)))

    elif args.mode == "filename":
        for source_file, raw_label in iter_filename_items(source_root, args.separator, args.recursive):
            operations.append((source_file, normalize_label(raw_label)))

    elif args.mode == "parent":
        for source_file, raw_label in iter_parent_items(source_root):
            if is_within(source_file, dest_root):
                continue
            operations.append((source_file, normalize_label(raw_label)))

    else:
        parser.error(f"Unsupported mode: {args.mode}")

    if not operations:
        print("No images matched the selected mode.")
        return 0

    counts_by_label: dict[str, int] = {}
    for source_file, label in operations:
        destination_dir = dest_root / label
        destination_path = destination_dir / source_file.name
        counts_by_label[label] = counts_by_label.get(label, 0) + 1

        if args.dry_run:
            print(f"[DRY-RUN] {source_file} -> {destination_path}")
            continue

        final_path = transfer_image(source_file, destination_path, args.action)
        print(f"{source_file} -> {final_path}")

    print("\nSummary:")
    print(f"  Mode: {args.mode}")
    print(f"  Action: {args.action}")
    print(f"  Destination root: {dest_root}")
    print(f"  Images processed: {len(operations)}")
    print(f"  Labels created: {len(counts_by_label)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())