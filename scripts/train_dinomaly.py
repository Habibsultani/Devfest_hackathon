"""Adapt official Dinomaly components on balanced normal instrument patches."""

import argparse
import json
import random
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from src.dataset import file_sha256, load_manifest, write_json
from src.dinomaly_inference import predict_tiled
from src.dinomaly_model import SurgicalDinomaly
from src.evaluation import image_metrics
from src.preprocessing import validate_image
from src.segmentation import get_segmenter
from src.tiling import extract_tiles, tile_tensor

CLASSES = ("instrument12", "instrument19")


def f1_threshold(labels, scores):
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("F1 calibration requires both labels and finite validation scores.")
    candidates = list(np.unique(scores)) + [float(np.nextafter(max(scores), np.inf))]
    choices = []
    for threshold in candidates:
        metrics = image_metrics(labels, scores, float(threshold))
        choices.append(
            (
                (metrics["f1"] or 0, -metrics["false_positive_rate"], -threshold),
                float(threshold),
                metrics,
            )
        )
    _, threshold, metrics = max(choices, key=lambda item: item[0])
    return threshold, metrics


def validate(model, folds, device):
    model.eval()
    results = {}
    for instrument in CLASSES:
        predictions = []
        for row in folds[instrument]:
            with Image.open(row["image_path"]) as image:
                score, _ = predict_tiled(model, image, device)
            predictions.append(
                {"source_path": row["source_path"], "label": row["label"], "score": score}
            )
        threshold, metrics = f1_threshold(
            [r["label"] for r in predictions], [r["score"] for r in predictions]
        )
        results[instrument] = {
            "threshold": threshold,
            "metrics": metrics,
            "predictions": predictions,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=200)
    args = parser.parse_args()
    if args.batch_size % 2 or args.steps < 1 or args.validate_every < 1:
        parser.error("Use a positive step count and an even batch size for equal class sampling.")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path("models/dinomaly")
    output.mkdir(parents=True, exist_ok=True)
    sampler = random.Random(seed)
    training, validation, manifest_hashes = {}, {}, {}
    for instrument in CLASSES:
        manifest = Path("data/processed") / instrument / "manifest.csv"
        rows = load_manifest(manifest)
        training[instrument] = [r for r in rows if r["split"] == "train"]
        validation[instrument] = [r for r in rows if r["split"] == "validation"]
        manifest_hashes[instrument] = file_sha256(manifest)
        if len(training[instrument]) != 12 or any(r["label"] for r in training[instrument]):
            raise ValueError("Expected twelve normal training images per instrument.")
        if sorted(r["label"] for r in validation[instrument]) != [0] * 4 + [1] * 4:
            raise ValueError(
                "Validation must have four normal and four defect images per instrument."
            )
    protocol = {
        "model_version": "medinspect-dinomaly-sam-lora-v4",
        "steps": args.steps,
        "validate_every": args.validate_every,
        "batch_size": args.batch_size,
        "seed": seed,
        "manifest_sha256": manifest_hashes,
        "device": device,
        "encoder": "DINOv2-reg small/14",
        "decoder_depth": 8,
        "lora_rank": 8,
        "masking": (
            "Official MobileSAM with 16 deterministic RGB-derived positive prompts; no GT masks."
        ),
        "patching": (
            "Native 256x256 non-overlapping tiles, resize to 252 for DINO "
            "stride14, black background."
        ),
        "scoring": "Mean of largest 0.1% of stitched instrument-region anomaly values.",
        "calibration": (
            "Per-instrument validation F1; ties lower false-positive rate then smaller threshold."
        ),
        "checkpoint_selection": (
            "Validation macro F1, then lower macro FPR, then earlier checkpoint."
        ),
        "training_precision": "CUDA BF16 with FP32 positive-kernel sums; FP32 inference.",
        "evaluation_status": (
            "Development comparison; this test cohort was previously inspected for PatchCore."
        ),
    }
    write_json(Path("reports/dinomaly_protocol.json"), protocol)
    segmenter = get_segmenter(device)
    tile_pools = {}
    for instrument in CLASSES:
        tile_pools[instrument] = []
        for index, row in enumerate(training[instrument]):
            with Image.open(row["image_path"]) as image:
                rgb = validate_image(image)
            mask = segmenter.segment(rgb)
            tiles = [tile for _, tile in extract_tiles(rgb, mask)]
            tile_pools[instrument].append(tiles)
            print(f"{instrument} normal {index + 1}/12: {len(tiles)} native patches", flush=True)
    source = json.loads(Path("models/dino_source.json").read_text())
    if file_sha256(Path("models") / source["checkpoint"]) != source["sha256"]:
        raise ValueError("Pretrained DINOv2 source checksum mismatch.")
    model = SurgicalDinomaly(pretrained=True).to(device)
    lora = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    reconstruction = list(model.bottleneck.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": reconstruction, "lr": 2e-3}, {"params": lora, "lr": 1e-5}],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history = []
    best_key = (-1.0, -1.0)
    best = None
    start = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        model.encoder.eval()
        batch = []
        for instrument in CLASSES:
            for _ in range(args.batch_size // 2):
                tile = sampler.choice(sampler.choice(tile_pools[instrument]))
                batch.append(tile_tensor(tile))
        tensor = torch.stack(batch).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(tensor, global_step=step)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite Dinomaly loss; no checkpoint is promoted.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(reconstruction + lora, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 25 == 0 or step == 1:
            print(
                (
                    f"step {step}/{args.steps} loss={loss.item():.6f} "
                    f"elapsed={perf_counter() - start:.1f}s"
                ),
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            evaluation = validate(model, validation, device)
            macro_f1 = sum(r["metrics"]["f1"] or 0 for r in evaluation.values()) / 2
            macro_fpr = sum(r["metrics"]["false_positive_rate"] for r in evaluation.values()) / 2
            key = (macro_f1, -macro_fpr)
            history.append({"step": step, "loss": float(loss.item()), "validation": evaluation})
            write_json(output / "training_history.json", history)
            print(
                f"validation step={step} macro_f1={macro_f1:.4f} macro_fpr={macro_fpr:.4f}",
                flush=True,
            )
            if key > best_key:
                best_key, best = key, evaluation
                metadata = {
                    "model_version": protocol["model_version"],
                    "manifest_sha256": manifest_hashes,
                    "thresholds": {c: r["threshold"] for c, r in best.items()},
                    "sam_sha256": segmenter.config["sha256"],
                    "selected_step": step,
                    "segmentation_version": "rgb-automatic-prompts-v3",
                    "lora_adapter_norm": float(
                        sum(
                            p.detach().float().norm().item()
                            for n, p in model.named_parameters()
                            if n.startswith("encoder.") and n.endswith(".b")
                        )
                    ),
                    "architecture": "SurgicalDinomaly-small-reg14-lora8-decoder8",
                }
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "metadata": metadata,
                    },
                    output / "candidate.part",
                )
                (output / "candidate.part").replace(output / "medinspect-dinomaly-sam-lora-v4.pt")
    artifact = output / "medinspect-dinomaly-sam-lora-v4.pt"
    checksum = file_sha256(artifact)
    metadata = torch.load(artifact, map_location="cpu", weights_only=True)["metadata"]
    dependencies = {
        p: version(p)
        for p in ("torch", "torchvision", "anomalib", "timm", "Pillow", "numpy", "mobile_sam")
    }
    pretrained = Path("models/pretrained/dinov2_vits14_reg4_pretrain.pth")
    write_json(
        output / "training_record.json",
        {
            "protocol": protocol,
            "selected_step": metadata["selected_step"],
            "selected_validation": best,
            "artifact_sha256": checksum,
            "pretrained_dino_sha256": file_sha256(pretrained),
            "train_sources": {c: [r["source_path"] for r in training[c]] for c in CLASSES},
        },
    )
    for instrument in CLASSES:
        config = {
            "schema_version": 1,
            "status": "trained",
            "backend": "dinomaly-sam-tiles",
            "artifact_format": "torch-state-dict",
            "artifact_path": "../dinomaly/" + artifact.name,
            "artifact_sha256": checksum,
            "model_version": protocol["model_version"],
            "instrument_class": instrument,
            "threshold": best[instrument]["threshold"],
            "manifest_sha256": manifest_hashes[instrument],
            "sam_sha256": segmenter.config["sha256"],
            "segmentation_version": "rgb-automatic-prompts-v3",
            "dependencies": dependencies,
            "training_images": 24,
            "validation_images": 16,
            "threshold_method": "per-class balanced-validation-F1",
            "selected_step": metadata["selected_step"],
            "score_interpretation": (
                "Raw cosine reconstruction error aggregated over SAM foreground, not a probability."
            ),
            "evaluation_status": protocol["evaluation_status"],
        }
        write_json(Path("models") / instrument / "dinomaly_config.json", config)
    print(f"Saved reloadable unified Dinomaly artifact: {artifact}, sha256={checksum}", flush=True)


if __name__ == "__main__":
    main()
