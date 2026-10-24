"""Validate the boundary with Person One without implementing their inference."""

import importlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.artifacts import ensure_all_artifacts
from src.model_config import ModelConfig, ModelError

# Multiple cached gateway revisions may share Person One's process-wide model.
_INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class InspectionResult:
    anomaly_score: float
    anomaly_map: np.ndarray
    requires_review: bool
    model_version: str


def validate_result(
    raw: object, image_size: tuple[int, int], config: ModelConfig
) -> InspectionResult:
    if not isinstance(raw, Mapping):
        raise ModelError("Inference must return a dictionary matching the agreed API contract.")
    try:
        score_value = raw["anomaly_score"]
        if isinstance(score_value, (bool, np.bool_)) or not isinstance(
            score_value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("anomaly_score must be a finite scalar number.")
        score = float(score_value)
        if not np.isfinite(score):
            raise ValueError("anomaly_score must be finite.")
        anomaly_map = raw["anomaly_map"]
        width, height = image_size
        if not isinstance(anomaly_map, np.ndarray) or anomaly_map.shape != (height, width):
            raise ValueError("anomaly_map must be a NumPy H x W array aligned to the RGB input.")
        if anomaly_map.dtype.kind not in "uif" or not np.isfinite(anomaly_map).all():
            raise ValueError("anomaly_map must contain only finite real numbers.")
        review = raw["requires_review"]
        if not isinstance(review, (bool, np.bool_)):
            raise ValueError("requires_review must be a boolean.")
        if bool(review) != (score >= config.threshold):
            raise ValueError("requires_review disagrees with the configured validation threshold.")
        if raw["model_version"] != config.model_version:
            raise ValueError("Returned model_version does not match the loaded configuration.")
        if "instrument_class" in raw and raw["instrument_class"] != config.instrument_class:
            raise ValueError("Returned instrument_class does not match the selected class.")
        snapshot = np.array(anomaly_map, dtype=np.float32, copy=True)
        if not np.isfinite(snapshot).all():
            raise ValueError("anomaly_map exceeds the supported float32 display range.")
        snapshot.setflags(write=False)
        return InspectionResult(score, snapshot, bool(review), config.model_version)
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise ModelError(f"Invalid inference result: {exc}") from exc


class ModelGateway:
    """A shared, serialized gateway to Person One's cached inspect_image function."""

    def __init__(self, config: ModelConfig, inspect: Callable):
        self.config = config
        self._inspect = inspect
        self._lock = _INFERENCE_LOCK

    def inspect(self, image: Image.Image, instrument_class: str) -> InspectionResult:
        if instrument_class != self.config.instrument_class:
            raise ModelError("This instrument class is unsupported by the selected model.")
        if image.mode != "RGB":
            raise ModelError("The inference interface requires a PIL RGB image.")
        try:
            # Person One alone applies model resizing, tensor conversion and normalization.
            with self._lock:
                if self.config.backend != "legacy":
                    raw = self._inspect(
                        image.copy(),
                        instrument_class=instrument_class,
                        config_path=self.config.config_path,
                    )
                else:
                    raw = self._inspect(image.copy())
                return validate_result(raw, image.size, self.config)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(
                "Inference failed. No inspection was produced. Check the model, "
                "preprocessing, and required dependencies."
            ) from exc


def load_gateway(config: ModelConfig) -> ModelGateway:
    ensure_all_artifacts(config)
    try:
        module = importlib.import_module("src.inference")
        inspect = getattr(module, "inspect_image", None)
        if not callable(inspect):
            raise ModelError("Person One's src.inference must expose inspect_image(image).")
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(
            "Person One's inference module or a required model dependency is unavailable. "
            "Supply src/inference.py and install the verified inference dependencies."
        ) from exc
    return ModelGateway(config, inspect)
