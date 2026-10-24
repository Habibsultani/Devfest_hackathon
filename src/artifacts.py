"""Retrieve trusted model artifacts reproducibly before Person One loads them."""

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from src.model_config import ModelConfig, ModelError

MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_artifact(config: ModelConfig) -> Path:
    """Verify an existing artifact or atomically download the configured HTTPS URL.

    The checked-in configuration is trusted deployment input, never an upload.
    No checkpoint is deserialized by this module.
    """
    target = config.artifact_path
    temporary = None
    try:
        if target.is_file():
            if sha256_file(target) != config.artifact_sha256:
                raise ModelError(
                    "Model checksum mismatch. Replace the artifact with the trusted file."
                )
            return target
        if not config.artifact_url:
            raise ModelError(
                "The trained model artifact is missing. Supply the configured checkpoint "
                "or an HTTPS artifact URL with its SHA-256 checksum."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            config.artifact_url, headers={"User-Agent": "MedInspect-AI/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise ModelError("The model download redirected to an insecure URL.")
            digest = hashlib.sha256()
            total = 0
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False
            ) as handle:
                temporary = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARTIFACT_BYTES:
                        raise ModelError("The artifact exceeds the supported 1 GB download limit.")
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != config.artifact_sha256:
                raise ModelError("Downloaded model checksum mismatch. The file was discarded.")
        os.replace(temporary, target)
        temporary = None
        return target
    except ModelError:
        raise
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise ModelError("The model artifact could not be retrieved or verified.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
