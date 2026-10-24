"""Freeze equal per-instrument cohorts with balanced validation/test labels."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random

from scripts.prepare_dataset import prepare_dataset
from src.dataset import file_sha256, load_manifest, write_json


def choose_balanced_rows(rows: list[dict], seed: int = 42, train_normal: int = 12,
                         validation_per_label: int = 4, test_per_label: int = 4) -> tuple[list[dict], dict]:
    """Allocate source groups first, then sample within their exclusive fold.

    Sampling never looks at a model score. Unselected relatives are excluded,
    never reassigned to another fold. The explicit seed fixes the first feasible
    allocation, rather than choosing an allocation based on model performance.
    """
    if min(train_normal, validation_per_label, test_per_label) < 1:
        raise ValueError("All balance quotas must be positive.")
    normals = sum(int(r["label"]) == 0 for r in rows)
    if normals < train_normal + validation_per_label + test_per_label:
        raise ValueError("Not enough normal images for the requested balance quotas.")
    if len(rows) - normals < validation_per_label + test_per_label:
        raise ValueError("Not enough defect images for the requested balance quotas.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    anomaly_groups = sorted(g for g, records in groups.items() if any(int(r["label"]) for r in records))
    pure_groups = sorted(set(groups) - set(anomaly_groups))
    if len(anomaly_groups) < 2:
        raise ValueError("Need at least two defect source groups to separate validation and test without leakage.")
    rng = random.Random(seed)

    def sample_groups(group_ids, label, count):
        pools = []
        for group in group_ids:
            pool = [r for r in groups[group] if int(r["label"]) == label]
            rng.shuffle(pool)
            if pool:
                pools.append(pool)
        rng.shuffle(pools)
        chosen = []
        # Round-robin gives each source group representation before extra variants.
        while len(chosen) < count and pools:
            remaining = []
            for pool in pools:
                if len(chosen) < count:
                    chosen.append(pool.pop())
                if pool:
                    remaining.append(pool)
            pools = remaining
        return chosen

    for attempt in range(10000):
        shuffled = anomaly_groups[:]
        rng.shuffle(shuffled)
        boundary = rng.randint(1, len(shuffled) - 1)
        owners = {g: "validation" for g in shuffled[:boundary]}
        owners.update({g: "test" for g in shuffled[boundary:]})
        selected = []
        feasible = True
        for split, quota in (("validation", validation_per_label), ("test", test_per_label)):
            owned = [g for g, owner in owners.items() if owner == split]
            defects = sample_groups(owned, 1, quota)
            if len(defects) != quota:
                feasible = False
                break
            normals = sample_groups(owned, 0, quota)
            available = [g for g in pure_groups if g not in owners]
            rng.shuffle(available)
            # Small groups limit unused relatives while retaining group separation.
            available.sort(key=lambda g: len(groups[g]))
            while len(normals) < quota and available:
                group = available.pop(0)
                owners[group] = split
                normals.extend(sample_groups([group], 0, quota - len(normals)))
            if len(normals) != quota:
                feasible = False
                break
            selected.extend(dict(row, split=split) for row in normals + defects)
        if not feasible:
            continue
        training_groups = [g for g in pure_groups if g not in owners
                           and all(r["original_split"] == "train" for r in groups[g])]
        training = sample_groups(training_groups, 0, train_normal)
        if len(training) != train_normal:
            continue
        selected.extend(dict(row, split="train") for row in training)
        selected.sort(key=lambda r: (r["split"], r["source_path"]))
        chosen_paths = {r["source_path"] for r in selected}
        return selected, {"seed": seed, "first_feasible_allocation_attempt": attempt + 1,
                          "train_normal": train_normal, "validation_per_label": validation_per_label,
                          "test_per_label": test_per_label,
                          "excluded_sources": [r["source_path"] for r in rows if r["source_path"] not in chosen_paths],
                          "selection_rule": "Source groups assigned first; sample without scores. Equal per-instrument quotas. No oversampling or duplicated images."}
    raise ValueError("Cannot satisfy balance quotas without group leakage. Reduce quotas or choose a class with more independent source captures.")


def prepare_balanced(raw: Path, output: Path, instrument: str, seed: int = 42,
                     train_normal: int = 12, validation_per_label: int = 4, test_per_label: int = 4) -> dict:
    index = output / "_index"
    original_summary = prepare_dataset(raw, index, instrument, seed)
    with (index / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    selected, audit = choose_balanced_rows(all_rows, seed, train_normal, validation_per_label, test_per_label)
    for row in selected:
        for field in ("image_path", "mask_path"):
            if row[field]:
                absolute = (index / row[field]).resolve()
                row[field] = os.path.relpath(absolute, output.resolve()).replace(os.sep, "/")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    verified = load_manifest(manifest)
    counts = {split: dict(Counter("defect" if r["label"] else "good" for r in verified if r["split"] == split))
              for split in ("train", "validation", "test")}
    summary = dict(original_summary, counts=counts, total_images=len(selected),
                   source_total_images=len(all_rows), excluded_images=len(all_rows) - len(selected),
                   total_masks=sum(r["label"] for r in verified), manifest_sha256=file_sha256(manifest),
                   policy="Balanced v2: equal normal training count across instruments; equal normal/defect validation and test counts; source groups disjoint.",
                   evaluation_status="Fresh classes selected by source counts before training; test labels reserved for final evaluation.",
                   independent_groups=len({r["group_id"] for r in selected}),
                   group_counts={split: len({r["group_id"] for r in selected if r["split"] == split}) for split in counts},
                   defect_group_counts={split: len({r["group_id"] for r in selected if r["split"] == split and int(r["label"])}) for split in ("validation", "test")},
                   balance_limitation="Image counts are balanced. Defect variants share few source captures, so this is not balanced independent clinical evidence.")
    write_json(output / "summary.json", summary)
    write_json(output / "selection_audit.json", audit)
    with (index / "duplicate_audit.json").open(encoding="utf-8") as stream:
        duplicates = json.load(stream)
    write_json(output / "duplicate_audit.json", duplicates)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-normal", type=int, default=12)
    parser.add_argument("--validation-per-label", type=int, default=4)
    parser.add_argument("--test-per-label", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare_balanced(args.raw, args.output, args.instrument, args.seed, args.train_normal,
                                      args.validation_per_label, args.test_per_label), indent=2))


if __name__ == "__main__":
    main()
