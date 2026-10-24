"""High-resolution SAM + tiled Dinomaly implementation of the stable API."""

import json
import math
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

from src.dataset import file_sha256
from src.preprocessing import validate_image
from src.segmentation import get_segmenter
from src.tiling import extract_tiles, instrument_score, stitch_tile_maps, tile_tensor


@lru_cache(maxsize=2)
def load_dinomaly(artifact_path: str, expected_sha256: str, device: str):
    from src.dinomaly_model import SurgicalDinomaly

    path = Path(artifact_path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Dinomaly artifact checksum mismatch.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SurgicalDinomaly(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["metadata"], Lock()


def predict_tiled(
    model, image: Image.Image, device: str, batch_size: int = 4
) -> tuple[float, np.ndarray]:
    rgb = validate_image(image)
    mask = get_segmenter(device).segment(rgb)
    tiles = extract_tiles(rgb, mask)
    maps = []
    with torch.inference_mode():
        for index in range(0, len(tiles), batch_size):
            tensor = torch.stack(
                [tile_tensor(tile) for _, tile in tiles[index : index + batch_size]]
            ).to(device)
            # FP32 inference keeps deployment scores consistent between CPU and
            # CUDA; mixed precision is used only for optimization during training.
            output = model(tensor)
            maps.extend(output.anomaly_map[:, 0].float().cpu().numpy())
    anomaly_map = stitch_tile_maps(tiles, maps, mask.shape, mask)
    return instrument_score(anomaly_map, mask), anomaly_map


class DinomalyInspector:
    def __init__(self, config_path: Path):
        self.config = json.loads(Path(config_path).read_text())
        config = self.config
        if config.get("backend") != "dinomaly-sam-tiles" or config.get("status") != "trained":
            raise ValueError("Unsupported or untrained Dinomaly configuration.")
        for package in (
            "torch",
            "torchvision",
            "anomalib",
            "timm",
            "Pillow",
            "numpy",
            "mobile_sam",
        ):
            # CPU and CUDA wheels of the same Torch release share the state format.
            if version(package).split("+")[0] != config["dependencies"][package].split("+")[0]:
                raise ValueError(f"{package} version differs from the saved Dinomaly environment.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(4)
        self.model, metadata, self.lock = load_dinomaly(
            str((Path(config_path).parent / config["artifact_path"]).resolve()),
            config["artifact_sha256"],
            self.device,
        )
        for key in ("model_version", "sam_sha256", "segmentation_version"):
            if metadata[key] != config[key]:
                raise ValueError(f"Dinomaly artifact/config mismatch: {key}")
        if metadata["manifest_sha256"][config["instrument_class"]] != config["manifest_sha256"]:
            raise ValueError("Dinomaly artifact/manifest mismatch.")
        sam = get_segmenter(self.device)
        if sam.config["sha256"] != config["sam_sha256"]:
            raise ValueError("Configured SAM weights differ from training.")
        self.threshold = float(config["threshold"])
        if not math.isfinite(self.threshold):
            raise ValueError("Nonfinite Dinomaly threshold.")
        if self.threshold != metadata["thresholds"][config["instrument_class"]]:
            raise ValueError("Dinomaly threshold differs from saved artifact metadata.")

    def inspect(self, image: Image.Image, *, instrument_class: str | None = None) -> dict:
        if instrument_class is not None and instrument_class != self.config["instrument_class"]:
            raise ValueError(
                f"This model configuration supports only {self.config['instrument_class']}."
            )
        with self.lock:
            score, anomaly_map = predict_tiled(self.model, image, self.device)
        return {
            "anomaly_score": float(score),
            "anomaly_map": anomaly_map,
            "requires_review": bool(score >= self.threshold),
            "threshold": self.threshold,
            "model_version": self.config["model_version"],
            "instrument_class": self.config["instrument_class"],
        }
