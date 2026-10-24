"""Paired brightness/JPEG stress tests with a fixed model and threshold."""

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance

from scripts.evaluate_model import verify_experiment
from src.dataset import write_json
from src.evaluation import image_metrics
from src.inference import DEFAULT_CONFIG, get_inspector
from src.preprocessing import validate_image


def jpeg_compress(image: Image.Image) -> Image.Image:
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=60)
    stream.seek(0)
    with Image.open(stream) as restored:
        return restored.convert("RGB")


def test_robustness(manifest: Path, config: Path, output: Path) -> dict:
    inspector = get_inspector(config)
    rows = [r for r in verify_experiment(manifest, inspector) if r["split"] == "test"]
    conditions = {
        "baseline": lambda image: image,
        "brightness_0.8": lambda image: ImageEnhance.Brightness(image).enhance(0.8),
        "brightness_1.2": lambda image: ImageEnhance.Brightness(image).enhance(1.2),
        "jpeg_quality_60": jpeg_compress,
    }
    records, summary = [], {}
    baseline = {}
    for condition, transform in conditions.items():
        labels, scores, changes = [], [], []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                rgb = validate_image(image)
            result = inspector.inspect(transform(rgb))
            score = result["anomaly_score"]
            if condition == "baseline":
                baseline[row["source_path"]] = result
            original = baseline[row["source_path"]]
            labels.append(row["label"])
            scores.append(score)
            changes.append(result["requires_review"] != original["requires_review"])
            records.append(
                {
                    "source_path": row["source_path"],
                    "condition": condition,
                    "label": row["label"],
                    "anomaly_score": score,
                    "score_delta": score - original["anomaly_score"],
                    "requires_review": result["requires_review"],
                    "decision_changed": changes[-1],
                }
            )
        summary[condition] = image_metrics(labels, scores, inspector.threshold)
        summary[condition]["changed_decisions"] = sum(changes)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "robustness_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "artifact_sha256": inspector.config["artifact_sha256"],
        "threshold": inspector.threshold,
        "manifest_sha256": inspector.config["manifest_sha256"],
        "conditions": summary,
        "method": (
            "Paired whole-image transformations, fixed model and threshold. No "
            "retraining or model selection from test results."
        ),
        "limitations": (
            "Does not isolate background changes or test an instrument-region masking improvement."
        ),
    }
    write_json(output / "robustness.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/instrument12/manifest.csv")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("reports/instrument12"))
    args = parser.parse_args()
    print(json.dumps(test_robustness(args.manifest, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
