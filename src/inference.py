"""Stable PIL/NumPy API for Person 2. No fabricated inference fallbacks."""

from functools import lru_cache
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image
import torch

from src.dataset import file_sha256
from src.preprocessing import align_anomaly_map, preprocess_image

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_CONFIG = MODELS_DIR / "instrument12" / "dinomaly_config.json"
REGISTRY = MODELS_DIR / "model_registry.json"


class ModelLoadError(RuntimeError):
    """A missing, incompatible or damaged saved model requires operator action."""


def build_patchcore(pretrained: bool):
    # Keep Anomalib internal objects entirely behind this adapter.
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    return PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"],
                          pre_trained=pretrained, num_neighbors=9)


class Inspector:
    def __init__(self, config_path: Path):
        config_path = Path(config_path).resolve()
        try:
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
            config = self.config
            if not isinstance(config, dict):
                raise ValueError("Model configuration must be a JSON object.")
            if config.get("status") != "trained":
                raise ValueError("No trained model is configured. Run scripts.train_model first.")
            if config.get("schema_version") != 1 or config.get("artifact_format") != "torch-state-dict":
                raise ValueError("Unsupported model configuration or artifact format.")
            expected = {"backbone": "resnet18", "layers": ["layer2", "layer3"], "num_neighbors": 9}
            if config["model"] != expected:
                raise ValueError("Unsupported PatchCore architecture.")
            expected_preprocessing = {"input_size": config["preprocessing"]["input_size"],
                                          "resize": "letterbox-bilinear", "padding": "white",
                                          "normalization": "imagenet", "exif_transpose": True}
            if "region_mask" in config["preprocessing"]:
                if not isinstance(config["preprocessing"]["region_mask"], bool):
                    raise ValueError("Invalid instrument-region preprocessing setting.")
                expected_preprocessing["region_mask"] = config["preprocessing"]["region_mask"]
            if config["preprocessing"] != expected_preprocessing:
                raise ValueError("Unsupported preprocessing configuration.")
            size = config["preprocessing"]["input_size"]
            if not isinstance(size, int) or size < 32 or size % 8:
                raise ValueError("Invalid saved input size.")
            if not isinstance(config["model_version"], str) or not config["model_version"]:
                raise ValueError("Missing model version.")
            self.threshold = float(config["threshold"])
            if not math.isfinite(self.threshold) or self.threshold < 0:
                raise ValueError("Model threshold must be finite and nonnegative.")
            for name in ("torch", "torchvision", "anomalib", "timm", "Pillow", "numpy"):
                if version(name).split("+")[0] != config["dependencies"][name].split("+")[0]:
                    raise ValueError(f"{name} version differs from training. Install requirements-lock.txt or rebuild the artifact.")
            artifact = config_path.parent / config["artifact_path"]
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing model artifact: {artifact}. Train it or retrieve the verified team artifact.")
            if file_sha256(artifact) != config["artifact_sha256"]:
                raise ValueError("Model artifact checksum mismatch. Retrieve an unmodified artifact.")
            torch.set_num_threads(config.get("cpu_threads", 4))
            checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
            for name in ("model", "preprocessing", "model_version", "instrument_class", "threshold", "manifest_sha256"):
                if checkpoint["metadata"][name] != config[name]:
                    raise ValueError(f"Configuration does not match saved artifact: {name}")
            self.model = build_patchcore(pretrained=False)
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model.eval()
            if self.model.memory_bank.numel() == 0:
                raise ValueError("Saved PatchCore memory bank is empty.")
            self._lock = Lock()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            raise ModelLoadError(f"Could not load MedInspect model: {error}") from error

    def inspect(self, image: Image.Image, *, instrument_class: str | None = None) -> dict:
        if instrument_class is not None and instrument_class != self.config["instrument_class"]:
            raise ValueError(f"Unsupported instrument class. This model supports only {self.config['instrument_class']}.")
        tensor, geometry = preprocess_image(image, self.config["preprocessing"]["input_size"],
                                             self.config["preprocessing"].get("region_mask", False))
        try:
            with self._lock, torch.inference_mode():
                prediction = self.model(tensor.unsqueeze(0))
            score = float(prediction.pred_score.reshape(-1)[0].item())
            raw_map = prediction.anomaly_map[0, 0]
            if tuple(raw_map.shape) != tuple(tensor.shape[-2:]) or not math.isfinite(score):
                raise ValueError("Nonfinite score or unexpected anomaly-map dimensions.")
            anomaly_map = align_anomaly_map(raw_map, geometry)
        except (RuntimeError, ValueError, AttributeError, IndexError) as error:
            raise RuntimeError(f"PatchCore inference failed: {error}") from error
        return {"anomaly_score": score, "anomaly_map": anomaly_map,
                "requires_review": bool(score >= self.threshold), "model_version": self.config["model_version"],
                "threshold": self.threshold, "instrument_class": self.config["instrument_class"]}


@lru_cache(maxsize=2)
def _cached_inspector(config_path: str, config_text: str, artifact_stamp: tuple) -> Inspector:
    if json.loads(config_text).get("backend") == "dinomaly-sam-tiles":
        from src.dinomaly_inference import DinomalyInspector
        try:
            return DinomalyInspector(Path(config_path))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            raise ModelLoadError(f"Could not load tiled Dinomaly model: {error}") from error
    return Inspector(Path(config_path))


def supported_classes() -> list[str]:
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        if not isinstance(registry, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in registry.items()):
            raise ValueError("Invalid model registry.")
        return sorted(registry)
    except (OSError, ValueError) as error:
        raise ModelLoadError(f"Could not read supported instrument classes: {error}") from error


def get_inspector(config_path: str | Path | None = None, instrument_class: str | None = None) -> Inspector:
    """Cache model loading; reload if config contents or artifact stat change."""
    selected = config_path or os.environ.get("MEDINSPECT_MODEL_CONFIG")
    if selected is None and instrument_class is not None:
        if instrument_class not in supported_classes():
            raise ValueError(f"Unsupported instrument class: {instrument_class}. Choose from {supported_classes()}.")
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        selected = REGISTRY.parent / registry[instrument_class]
    path = Path(selected or DEFAULT_CONFIG).resolve()
    try:
        contents = path.read_text(encoding="utf-8")
        config = json.loads(contents)
        if not isinstance(config, dict):
            raise ValueError("Model configuration must be a JSON object.")
        artifact = path.parent / config.get("artifact_path", "missing.pt")
        stat = artifact.stat() if artifact.is_file() else None
        stamp = (stat.st_mtime_ns, stat.st_size) if stat else ()
    except (OSError, ValueError, TypeError) as error:
        raise ModelLoadError(f"Model configuration is missing or invalid at {path}: {error}") from error
    return _cached_inspector(str(path), contents, stamp)


def inspect_image(image: Image.Image, *, config_path: str | Path | None = None,
                  instrument_class: str | None = None) -> dict:
    """Return raw score, original-size float32 map, review flag and model version.

    Supply the declared instrument class if available. The model cannot identify
    instrument classes or reject unrelated photos automatically. Maps follow the
    EXIF-oriented image; use ImageOps.exif_transpose for the matching UI display.
    """
    return get_inspector(config_path, instrument_class).inspect(image, instrument_class=instrument_class)
