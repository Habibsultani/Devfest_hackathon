"""Build a genuine normal-only PatchCore bank and calibrate on validation normals."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from src.dataset import file_sha256, load_manifest, write_json
from src.inference import build_patchcore, Inspector, DEFAULT_CONFIG
from src.evaluation import image_metrics
from src.preprocessing import preprocess_image


def calibrate_threshold(scores: list[float], quantile: float = 0.95) -> float:
    if not scores or not np.isfinite(scores).all() or not 0 < quantile <= 1:
        raise ValueError("Threshold calibration requires finite validation scores and quantile in (0, 1].")
    # Strictly above the empirical quantile, with >= used at inference.
    return float(np.nextafter(float(np.quantile(scores, quantile, method="higher")), np.inf))


def balanced_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict]:
    if set(labels) != {0, 1} or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Balanced calibration requires both labels and finite scores.")
    unique = np.unique(scores)
    candidates = [float(value) for value in unique] + [float(np.nextafter(unique[-1], np.inf))]
    choices = []
    for threshold in candidates:
        metrics = image_metrics(labels, scores, threshold)
        balanced_accuracy = (metrics["recall"] + 1 - metrics["false_positive_rate"]) / 2
        # Tie rule fixed before test evaluation: fewer false alarms, then smaller threshold.
        choices.append(((round(balanced_accuracy, 12), -metrics["false_positive_rate"], -threshold), threshold, metrics))
    _, threshold, metrics = max(choices, key=lambda candidate: candidate[0])
    return threshold, dict(metrics, balanced_accuracy=(metrics["recall"] + 1 - metrics["false_positive_rate"]) / 2)


def train_model(manifest: Path, config_path: Path, input_size: int = 256,
                coreset_ratio: float = 0.05, seed: int = 42, quantile: float = 0.95,
                calibration: str = "balanced-accuracy", region_mask: bool = False) -> dict:
    if not 0 < coreset_ratio <= 1:
        raise ValueError("Coreset ratio must be in (0, 1].")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rows = load_manifest(manifest)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not training or not validation or any(r["label"] for r in training):
        raise ValueError("Training must be nonempty and normal-only; validation must be nonempty.")
    if calibration not in ("balanced-accuracy", "normal-quantile"):
        raise ValueError("Unsupported calibration method.")
    model = build_patchcore(pretrained=True)
    model.train()
    # The extractor remains frozen and in eval mode. No network fine-tuning.
    model.feature_extractor.eval()
    with torch.no_grad():
        for row in training:
            with Image.open(row["image_path"]) as image:
                tensor, _ = preprocess_image(image, input_size, region_mask)
            model(tensor.unsqueeze(0))
    n_embeddings = sum(t.shape[0] for t in model.embedding_store)
    if int(n_embeddings * coreset_ratio) < 1:
        raise ValueError("Coreset selection would produce an empty memory bank.")
    model.subsample_embedding(sampling_ratio=coreset_ratio)
    model.eval()
    validation_results = []
    with torch.inference_mode():
        for row in validation:
            with Image.open(row["image_path"]) as image:
                tensor, _ = preprocess_image(image, input_size, region_mask)
            score = float(model(tensor.unsqueeze(0)).pred_score.item())
            validation_results.append({"source_path": row["source_path"], "group_id": row["group_id"],
                                       "label": row["label"], "score": score})
    if calibration == "balanced-accuracy":
        threshold, validation_metrics = balanced_threshold([r["label"] for r in validation_results],
                                                          [r["score"] for r in validation_results])
    else:
        if any(r["label"] for r in validation_results):
            raise ValueError("Normal-quantile calibration requires normal-only validation.")
        threshold = calibrate_threshold([r["score"] for r in validation_results], quantile)
        validation_metrics = None
    config_path = config_path.resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    instrument = rows[0]["instrument_class"]
    model_version = f"medinspect-{instrument}-balanced-v2" if calibration == "balanced-accuracy" else f"medinspect-{instrument}-v1"
    if region_mask:
        model_version = f"medinspect-{instrument}-balanced-v3-region"
    artifact = config_path.parent / f"{model_version}.pt"
    config = {
        "schema_version": 1, "status": "trained", "artifact_format": "torch-state-dict",
        "artifact_path": artifact.name, "model_version": model_version,
        "instrument_class": instrument,
        "model": {"backbone": "resnet18", "layers": ["layer2", "layer3"], "num_neighbors": 9},
        "preprocessing": {"input_size": input_size, "resize": "letterbox-bilinear", "padding": "white",
                          "normalization": "imagenet", "exif_transpose": True, "region_mask": region_mask},
        "threshold": threshold, "threshold_method": calibration,
        "threshold_selection": "Maximize validation balanced accuracy; ties prefer lower false-positive rate, then smaller threshold." if calibration == "balanced-accuracy" else "Normal validation quantile, higher interpolation, nextafter upward.",
        "threshold_quantile": quantile if calibration == "normal-quantile" else None,
        "seed": seed, "coreset_sampling_ratio": coreset_ratio,
        "training_images": len(training), "validation_images": len(validation),
        "training_patch_embeddings": n_embeddings, "memory_bank_shape": list(model.memory_bank.shape),
        "manifest_sha256": file_sha256(manifest), "cpu_threads": 4,
        "dependencies": {name: version(name) for name in ("torch", "torchvision", "anomalib", "timm", "Pillow", "numpy")},
        "score_interpretation": "Raw PatchCore distance, not defect probability. Larger means more anomalous.",
        "scope": "Research prototype for one declared class. No safety or sterility certification.",
    }
    metadata = {name: config[name] for name in ("model", "preprocessing", "model_version", "instrument_class", "threshold", "manifest_sha256")}
    temporary = artifact.with_suffix(".part")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(artifact)
    config["artifact_sha256"] = file_sha256(artifact)
    write_json(config_path, config)
    # Verify the actual saved artifact reproduces in-memory outputs.
    inspector = Inspector(config_path)
    probe = validation[0]
    with Image.open(probe["image_path"]) as image:
        first = inspector.inspect(image)
        tensor, _ = preprocess_image(image, input_size, region_mask)
    with torch.inference_mode():
        reference = float(model(tensor.unsqueeze(0)).pred_score.item())
    if not np.isclose(first["anomaly_score"], reference, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Saved artifact reload does not reproduce the original model score.")
    write_json(config_path.parent / "training_record.json", {
        "manifest_sha256": config["manifest_sha256"], "artifact_sha256": config["artifact_sha256"],
        "train_sources": [r["source_path"] for r in training], "validation_results": validation_results,
        "threshold": threshold, "validation_metrics": validation_metrics, "reload_score_verified": True,
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/instrument12/manifest.csv"))
    parser.add_argument("--config", type=Path, default=Path("models/instrument12/model_config.json"))
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--coreset-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--calibration", choices=["balanced-accuracy", "normal-quantile"], default="balanced-accuracy")
    parser.add_argument("--region-mask", action="store_true", help="Experimental independent blue-background suppression.")
    args = parser.parse_args()
    print(json.dumps(train_model(args.manifest, args.config, args.input_size, args.coreset_ratio,
                                 args.seed, args.threshold_quantile, args.calibration, args.region_mask), indent=2))


if __name__ == "__main__":
    main()
