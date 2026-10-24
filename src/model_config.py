"""Shared application metadata adapter for versioned model configurations."""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelError(RuntimeError):
    """A model dependency or inference result is unavailable or invalid."""


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    sha256: str
    url: str | None = None
    api_url: str | None = None


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
    config_path: Path | None = None
    backend: str = "legacy"
    artifact_api_url: str | None = None
    auxiliary_artifacts: tuple[ArtifactSpec, ...] = ()

    @property
    def artifacts(self) -> tuple[ArtifactSpec, ...]:
        return (
            ArtifactSpec(
                self.artifact_path, self.artifact_sha256, self.artifact_url, self.artifact_api_url
            ),
            *self.auxiliary_artifacts,
        )


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


def _checksum(value: object) -> str:
    value = _text(value, "artifact.sha256").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("artifact.sha256 must be an actual 64-character SHA-256 digest.")
    return value


def _url(value: object, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.startswith("https://")):
        raise ValueError(f"{name} must be an HTTPS URL or null.")
    return value


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    """Read canonical AI metadata (config-relative), or legacy UI metadata (root-relative)."""
    if path is None:
        path = os.environ.get("MEDINSPECT_MODEL_CONFIG") or (
            PROJECT_ROOT / "models/instrument12/dinomaly_config.json"
        )
    config_path = Path(path).resolve()
    try:
        raw = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise ModelError(f"Model configuration is missing: {config_path}") from exc
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
        canonical = "artifact_path" in data
        if canonical and data.get("status") != "trained":
            raise ValueError("The selected model is not trained.")
        artifact = (
            {
                "path": data["artifact_path"],
                "sha256": data["artifact_sha256"],
                "url": data.get("artifact_download_url"),
            }
            if canonical
            else data["artifact"]
        )
        artifact_path = Path(_text(artifact["path"], "artifact.path"))
        if not artifact_path.is_absolute():
            artifact_path = (config_path.parent if canonical else PROJECT_ROOT) / artifact_path
        scale = data["heatmap_scale"]
        low, high = (
            _number(scale["min"], "heatmap_scale.min"),
            _number(scale["max"], "heatmap_scale.max"),
        )
        if high <= low:
            raise ValueError("heatmap_scale.max must exceed heatmap_scale.min.")
        backend = data.get("backend", "patchcore" if canonical else "legacy")
        if backend not in {"dinomaly-sam-tiles", "patchcore", "legacy"}:
            raise ValueError("Unsupported model backend.")
        auxiliary = ()
        if backend == "dinomaly-sam-tiles":
            sam = json.loads((PROJECT_ROOT / "models/sam_config.json").read_text(encoding="utf-8"))
            if sam["sha256"] != data["sam_sha256"]:
                raise ValueError("SAM metadata differs from the trained model.")
            auxiliary = (
                ArtifactSpec(
                    (PROJECT_ROOT / "models" / sam["checkpoint"]).resolve(),
                    _checksum(sam["sha256"]),
                    _url(sam["source_url"], "SAM source_url"),
                ),
            )
        return ModelConfig(
            model_version=_text(data["model_version"], "model_version"),
            instrument_class=_text(data["instrument_class"], "instrument_class"),
            threshold=_number(data["threshold"], "threshold"),
            artifact_path=artifact_path.resolve(),
            artifact_sha256=_checksum(artifact["sha256"]),
            artifact_url=_url(artifact.get("url"), "artifact.url"),
            heatmap_min=low,
            heatmap_max=high,
            config_sha256=hashlib.sha256(raw).hexdigest(),
            config_path=config_path,
            backend=backend,
            artifact_api_url=_url(data.get("artifact_api_url"), "artifact_api_url"),
            auxiliary_artifacts=auxiliary,
        )
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
        raise ModelError(f"Invalid model configuration: {exc}") from exc


def load_model_configs() -> dict[str, ModelConfig]:
    """Honor an explicit deployment override, otherwise expose the versioned registry."""
    override = os.environ.get("MEDINSPECT_MODEL_CONFIG")
    if override:
        config = load_model_config(override)
        return {config.instrument_class: config}
    registry_path = PROJECT_ROOT / "models/model_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict) or not registry:
            raise ValueError("The model registry must be a nonempty object.")
        configs = {}
        for name, relative in registry.items():
            config = load_model_config(registry_path.parent / _text(relative, "registry path"))
            if config.instrument_class != name:
                raise ValueError("Registry class and configured class differ.")
            configs[name] = config
        return configs
    except (OSError, ValueError, TypeError) as exc:
        raise ModelError(f"Model registry is unavailable or invalid: {exc}") from exc
