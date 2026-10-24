"""Portable manifests and integrity checks for every experiment stage."""

import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(path: Path, verify_files: bool = True) -> list[dict]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Dataset manifest is empty.")
    required = {
        "image_path",
        "mask_path",
        "label",
        "split",
        "group_id",
        "sha256",
        "instrument_class",
    }
    groups, hashes = {}, {}
    for row in rows:
        if not required <= row.keys():
            raise ValueError("Dataset manifest is missing required columns.")
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1) or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid manifest label or split.")
        if row["split"] == "train" and row["label"]:
            raise ValueError("Training must contain normal images only.")
        for mapping, key in ((groups, row["group_id"]), (hashes, row["sha256"])):
            previous = mapping.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Cross-split duplicate or source-group leakage.")
        row["image_path"] = (path.parent / row["image_path"]).resolve()
        row["mask_path"] = (path.parent / row["mask_path"]).resolve() if row["mask_path"] else None
        if verify_files and file_sha256(row["image_path"]) != row["sha256"]:
            raise ValueError(f"Image changed after preparation: {row['image_path']}")
        if verify_files and row["mask_path"]:
            if file_sha256(row["mask_path"]) != row.get("mask_sha256"):
                raise ValueError(f"Mask changed after preparation: {row['mask_path']}")
    if len({r["instrument_class"] for r in rows}) != 1:
        raise ValueError("One instrument class is required per model.")
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
