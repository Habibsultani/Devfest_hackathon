import copy
import csv

import pytest

from scripts.prepare_dataset import assign_splits, group_images, source_id
from src.dataset import load_manifest


def record(path, phash, original_split="train", label=0):
    return {"source_path": path, "source_id": source_id(path), "phash": phash,
            "pixel_sha256": path, "original_split": original_split, "label": label}


def test_source_variants_and_transitive_near_duplicates_stay_together():
    rows = [record("0000_IMG_0989", "0000000000000000", "test", 1),
            record("0001_IMG_0989", "ffffffffffffffff", "test", 1),
            record("normal1", "fffffffffffffffe"),
            record("normal2", "fffffffffffffffc")]
    group_images(rows, max_distance=1)
    assert len({r["group_id"] for r in rows}) == 1


def test_original_test_groups_are_never_reassigned():
    rows = [{"group_id": f"group{i}", "original_split": "train", "label": 0} for i in range(8)]
    rows.extend([{"group_id": "test", "original_split": "test", "label": 1},
                 {"group_id": "test", "original_split": "train", "label": 0}])
    repeated = copy.deepcopy(rows)
    assign_splits(rows, 42)
    assign_splits(repeated, 42)
    assert rows == repeated
    assert {r["split"] for r in rows if r["group_id"] == "test"} == {"test"}
    assert {r["split"] for r in rows} == {"train", "validation", "test"}


def test_insufficient_independent_groups_fail():
    with pytest.raises(ValueError, match="independent"):
        assign_splits([{"group_id": "a", "original_split": "train", "label": 0}], 42)


def test_manifest_rejects_cross_split_leakage(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = [{"image_path": "a.png", "mask_path": "", "label": 0,
             "split": split, "group_id": "same", "sha256": "same", "instrument_class": "instrument1"}
            for split in ("train", "test")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path, verify_files=False)
