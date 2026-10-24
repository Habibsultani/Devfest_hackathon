"""Index Purdue folders, validate masks and assign entire source groups to splits."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re

import imagehash
import numpy as np
from PIL import Image, ImageOps

from src.dataset import file_sha256, load_manifest, write_json
from src.preprocessing import validate_image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def source_id(stem: str) -> str:
    """Keep numbered synthetic variants of the same IMG capture together."""
    match = re.search(r"IMG_\d+", stem, flags=re.IGNORECASE)
    return match.group(0).upper() if match else stem.lower()


def group_images(records: list[dict], max_distance: int = 4) -> list[dict]:
    if not 0 <= max_distance <= 16:
        raise ValueError("pHash distance must be between 0 and 16.")
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    pairs = []
    for i, first in enumerate(records):
        for j in range(i):
            second = records[j]
            distance = (int(first["phash"], 16) ^ int(second["phash"], 16)).bit_count()
            same_source = first["source_id"] == second["source_id"]
            identical = first["pixel_sha256"] == second["pixel_sha256"]
            if same_source or identical or distance <= max_distance:
                parents[find(i)] = find(j)
                pairs.append({"first": first["source_path"], "second": second["source_path"],
                              "phash_distance": distance, "same_source_id": same_source,
                              "identical_pixels": identical})
    members = {}
    for i, row in enumerate(records):
        members.setdefault(find(i), []).append(row)
    for group in members.values():
        group_id = hashlib.sha256("\n".join(sorted(r["source_path"] for r in group)).encode()).hexdigest()[:16]
        for row in group:
            row["group_id"] = group_id
    return pairs


def assign_splits(records: list[dict], seed: int) -> dict:
    # Original test data is never promoted into training or calibration.
    test_groups = {r["group_id"] for r in records if r["original_split"] == "test"}
    candidates = sorted({r["group_id"] for r in records} - test_groups)
    if len(candidates) < 3:
        raise ValueError("Fewer than three independent training-source groups remain. Choose another class; do not weaken leakage checks.")
    random.Random(seed).shuffle(candidates)
    n_validation = max(1, round(len(candidates) * 0.2))
    n_test = max(1, round(len(candidates) * 0.2))
    validation = set(candidates[:n_validation])
    test_groups.update(candidates[n_validation:n_validation + n_test])
    for row in records:
        row["split"] = "test" if row["group_id"] in test_groups else "validation" if row["group_id"] in validation else "train"
        if row["split"] == "train" and row["label"]:
            raise ValueError("An abnormal image was assigned to training.")
    return {split: dict(Counter("defect" if r["label"] else "good" for r in records if r["split"] == split))
            for split in ("train", "validation", "test")}


def prepare_dataset(raw: Path, output: Path, instrument: str, seed: int = 42, phash_distance: int = 4) -> dict:
    raw, output = raw.resolve(), output.resolve()
    root = raw / instrument
    records = []
    used_masks = set()
    for original_split in ("train", "test"):
        for image_path in sorted((root / original_split).rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Unsupported dataset file: {image_path}")
            defect_type = image_path.parent.name
            label = int(defect_type != "good")
            if original_split == "train" and label:
                raise ValueError("Purdue training folder must contain normal images only.")
            mask = None
            if label:
                candidates = [p for p in (root / "ground_truth" / defect_type).glob("*")
                              if p.stem in (image_path.stem, image_path.stem + "_mask") and p.suffix.lower() in EXTENSIONS]
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one defect mask for {image_path}")
                mask = candidates[0]
                used_masks.add(mask)
            with Image.open(image_path) as image:
                original_size = image.size
                rgb = validate_image(image)
                phash = str(imagehash.phash(rgb))
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if mask:
                with Image.open(mask) as mask_image:
                    if mask_image.size != original_size:
                        raise ValueError(f"Mask/image dimensions differ: {mask}")
                    values = np.unique(np.asarray(mask_image.convert("L")))
                    if not np.any(values > 127) or 0 not in values:
                        raise ValueError(f"Expected a nonempty foreground/background defect mask: {mask}")
            records.append({"source_path": image_path.relative_to(raw).as_posix(),
                            "image_path": os.path.relpath(image_path, output).replace(os.sep, "/"),
                            "instrument_class": instrument, "label": label, "defect_type": defect_type,
                            "mask_path": os.path.relpath(mask, output).replace(os.sep, "/") if mask else "",
                            "original_split": original_split, "source_id": source_id(image_path.stem),
                            "sha256": file_sha256(image_path), "mask_sha256": file_sha256(mask) if mask else "",
                            "pixel_sha256": pixels, "phash": phash, "width": rgb.width, "height": rgb.height})
    all_masks = {p for p in (root / "ground_truth").rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if all_masks != used_masks:
        raise ValueError(f"Unpaired masks: {sorted(str(p) for p in all_masks - used_masks)}")
    if not records or not any(r["label"] for r in records):
        raise ValueError("Expected readable normal images, defects and masks.")
    pairs = group_images(records, phash_distance)
    counts = assign_splits(records, seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    load_manifest(manifest)
    source_path = raw / f"{instrument}_source.json"
    provenance = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    summary = {"instrument_class": instrument, "seed": seed, "phash_max_distance": phash_distance,
               "counts": counts, "total_images": len(records), "total_masks": len(used_masks),
               "independent_groups": len({r["group_id"] for r in records}),
               "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
               "mask_binarization": "Convert to grayscale and use >127; source masks include soft edges.",
               "manifest_sha256": file_sha256(manifest),
               "dataset_revision": provenance["revision"] if provenance else "unverified local data",
               "policy": "Original test groups stay test. Train-source groups split 60/20/20. Validation is normal-only.",
               "limitations": "pHash and capture IDs are heuristics, not proof of independence between physical instruments."}
    write_json(output / "summary.json", summary)
    write_json(output / "duplicate_audit.json", {"linked_pairs": pairs, "cross_split_linked_pairs": 0})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/instrument1"))
    parser.add_argument("--instrument", default="instrument1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.raw, args.output, args.instrument, args.seed, args.phash_distance), indent=2))


if __name__ == "__main__":
    main()
