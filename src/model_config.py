"""Application-side metadata contract; never chooses a model threshold."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelError(RuntimeError):
    """A model dependency or inference result is unavailable or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    model_version: str
    instrument_class: str
    threshold: float
    artifact_path: Path
    artifact_sha256: str
    artifact_url: str | None
    heatmap_min: float
    heatmap_max: float
    config_sha256: str


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty string of at most 200 characters.")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters.")
    return value.strip()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    """Read Person One's metadata. Relative artifact paths are project-relative."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "models/model_config.json"
    try:
        raw = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise ModelError(
            "Model configuration is missing. Person One must supply models/model_config.json "
            "and the trained model before analysis is available."
        ) from exc
    except OSError as exc:
        raise ModelError("The model configuration cannot be read.") from exc
    try:
        data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("Use a configuration object with schema_version 1.")
        artifact = data["artifact"]
        artifact_path = Path(_text(artifact["path"], "artifact.path"))
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        checksum = _text(artifact["sha256"], "artifact.sha256").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("artifact.sha256 must be the actual 64-character SHA-256 digest.")
        url = artifact.get("url")
        if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
            raise ValueError("artifact.url must be an HTTPS URL or null.")
        scale = data["heatmap_scale"]
        low = _number(scale["min"], "heatmap_scale.min")
        high = _number(scale["max"], "heatmap_scale.max")
        if high <= low:
            raise ValueError("heatmap_scale.max must exceed heatmap_scale.min.")
        return ModelConfig(
            model_version=_text(data["model_version"], "model_version"),
            instrument_class=_text(data["instrument_class"], "instrument_class"),
            threshold=_number(data["threshold"], "threshold"),
            artifact_path=artifact_path.resolve(),
            artifact_sha256=checksum,
            artifact_url=url,
            heatmap_min=low,
            heatmap_max=high,
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ModelError(f"Invalid model configuration: {exc}") from exc
