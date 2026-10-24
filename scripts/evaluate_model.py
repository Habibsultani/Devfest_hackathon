"""Evaluate the locked checkpoint on the manifest's untouched test fold."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

from src.dataset import file_sha256, load_manifest, write_json
from src.evaluation import image_metrics
from src.inference import Inspector, DEFAULT_CONFIG, get_inspector
from src.preprocessing import validate_image


def verify_experiment(manifest: Path, inspector: Inspector) -> list[dict]:
    if file_sha256(manifest) != inspector.config["manifest_sha256"]:
        raise ValueError("Manifest differs from the one used to train and calibrate this artifact.")
    rows = load_manifest(manifest)
    if rows[0]["instrument_class"] != inspector.config["instrument_class"]:
        raise ValueError("Manifest and model instrument classes differ.")
    return rows


def load_mask(row: dict) -> Image.Image:
    with Image.open(row["image_path"]) as image:
        orientation = image.getexif().get(274, 1)
        size = validate_image(image).size
    if not row["label"]:
        return Image.new("L", size, 0)
    if row["mask_path"] is None:
        raise ValueError("Localization evaluation requires every defect mask.")
    with Image.open(row["mask_path"]) as image:
        mask = image.convert("L")
    exif = mask.getexif()
    exif[274] = orientation
    mask.info["exif"] = exif.tobytes()
    mask = ImageOps.exif_transpose(mask)
    if mask.size != size:
        raise ValueError("Ground truth does not align with the EXIF-oriented image.")
    return mask


def evaluate_model(manifest: Path, config_path: Path, output: Path) -> dict:
    inspector = get_inspector(config_path)
    all_rows = verify_experiment(manifest, inspector)
    rows = [r for r in all_rows if r["split"] == "test"]
    if not rows:
        raise ValueError("No held-out test images in manifest.")
    # Warm up on a validation image; never record artifact-load time as inference.
    warmup = next(r for r in all_rows if r["split"] == "validation")
    with Image.open(warmup["image_path"]) as image:
        inspector.inspect(image)
    results, pixel_labels, pixel_scores = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "evaluated_model_config.json")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    examples = set()
    # Fixed display scale based only on validation maps, never on test labels.
    with Image.open(warmup["image_path"]) as image:
        display_max = float(np.max(inspector.inspect(image)["anomaly_map"]))
    for row in rows:
        with Image.open(row["image_path"]) as image:
            rgb = validate_image(image)
        start = perf_counter()
        result = inspector.inspect(rgb)
        latency = perf_counter() - start
        anomaly_map = result["anomaly_map"]
        mask = load_mask(row)
        # Pooled pixel AUROC at a declared max-side resolution; omit letterbox padding.
        scale = 256 / max(rgb.size)
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        labels = np.asarray(mask.resize(size, Image.Resampling.NEAREST)) > 127
        scores = np.asarray(Image.fromarray(anomaly_map).resize(size, Image.Resampling.BILINEAR))
        pixel_labels.append(labels.ravel())
        pixel_scores.append(scores.ravel())
        results.append({"source_path": row["source_path"], "group_id": row["group_id"],
                        "label": row["label"], "defect_type": row["defect_type"],
                        "anomaly_score": result["anomaly_score"], "threshold": result["threshold"],
                        "requires_review": result["requires_review"], "latency_seconds": latency,
                        "model_version": result["model_version"]})
        # Local research previews only; ignored by Git due to dataset restrictions.
        if row["label"] not in examples:
            examples.add(row["label"])
            display_rgb = rgb.resize(size, Image.Resampling.BILINEAR)
            fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
            axes[0].imshow(display_rgb)
            axes[0].set_title("Original")
            axes[1].imshow(display_rgb)
            heat = axes[1].imshow(scores, cmap="inferno", alpha=0.55, vmin=0, vmax=display_max)
            axes[1].set_title("Suspicious regions")
            axes[2].imshow(labels, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Ground-truth defect mask")
            for ax in axes:
                ax.axis("off")
            fig.colorbar(heat, ax=axes[1], shrink=0.6, label="Raw anomaly score")
            fig.savefig(figures / f"example_overlay_{row['label']}.png", dpi=130)
            plt.close(fig)
    labels = [r["label"] for r in results]
    scores = [r["anomaly_score"] for r in results]
    metrics = image_metrics(labels, scores, inspector.threshold)
    pixels_y, pixels_s = np.concatenate(pixel_labels), np.concatenate(pixel_scores)
    latencies = [r["latency_seconds"] for r in results]
    metrics.update({"threshold": inspector.threshold, "model_version": inspector.config["model_version"],
                    "artifact_sha256": inspector.config["artifact_sha256"],
                    "manifest_sha256": inspector.config["manifest_sha256"],
                    "config_sha256": file_sha256(config_path),
                    "test_groups": len({r["group_id"] for r in rows}),
                    "defect_type_counts": dict(Counter(r["defect_type"] for r in rows)),
                    "pixel_auroc": float(roc_auc_score(pixels_y, pixels_s)) if len(np.unique(pixels_y)) == 2 else None,
                    "localization_method": "Pooled pixel AUROC; masks nearest-neighbor, maps bilinear, max side 256, padding excluded; normal masks zero.",
                    "localization_evaluated_pixels": len(pixels_y),
                    "latency_median_seconds": float(np.median(latencies)),
                    "latency_mean_seconds": float(np.mean(latencies)),
                    "inference_device": getattr(inspector, "device", "cpu"),
                    "latency_scope": "Preprocessing + forward + original-size map alignment, after warmup; excludes file decoding and model loading. For SAM backend the segmentation cache is warm.",
                    "limitations": "Small single-class split; related test variants are not independent observations. Simulated defects; no clinical claims."})
    write_json(output / "metrics.json", metrics)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    predictions = np.array(scores) >= inspector.threshold
    ConfusionMatrixDisplay.from_predictions(labels, predictions, labels=[0, 1], display_labels=["Normal", "Defect"], colorbar=False)
    plt.title("Held-out screening results")
    plt.savefig(figures / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    if len(set(labels)) == 2:
        RocCurveDisplay.from_predictions(labels, scores)
        plt.title("Held-out image-level ROC")
        plt.savefig(figures / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
    # Commit numerical provenance, not source images or masks.
    shutil.copyfile(manifest, output / "dataset_manifest.csv")
    for source, target in (("summary.json", "dataset_summary.json"), ("duplicate_audit.json", "duplicate_audit.json")):
        if (manifest.parent / source).exists():
            shutil.copyfile(manifest.parent / source, output / target)
    if (manifest.parent / "selection_audit.json").exists():
        shutil.copyfile(manifest.parent / "selection_audit.json", output / "selection_audit.json")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/instrument12/manifest.csv"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("reports/instrument12"))
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.manifest, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
