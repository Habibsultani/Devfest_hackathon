"""Retrieve trusted model artifacts reproducibly before Person One loads them."""

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from src.artifact_io import SafeAssetRedirect, github_token
from src.model_config import ArtifactSpec, ModelConfig, ModelError

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
    return ensure_spec(config.artifacts[0])


def ensure_all_artifacts(config: ModelConfig) -> Path:
    for artifact in config.artifacts:
        ensure_spec(artifact)
    return config.artifact_path


def ensure_spec(spec: ArtifactSpec) -> Path:
    target = spec.path
    temporary = None
    try:
        if target.is_file():
            if sha256_file(target) != spec.sha256:
                raise ModelError(
                    "Model checksum mismatch. Replace the artifact with the trusted file."
                )
            return target
        url = spec.api_url or spec.url
        if not url:
            raise ModelError(
                "The trained model artifact is missing. Supply the configured checkpoint "
                "or an HTTPS artifact URL with its SHA-256 checksum."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "MedInspect-AI/1.0"}
        if spec.api_url:
            if urlparse(url).hostname != "api.github.com":
                raise ModelError("Authenticated model retrieval must use the GitHub API.")
            headers.update(
                Authorization="Bearer " + github_token(), Accept="application/octet-stream"
            )
        request = urllib.request.Request(url, headers=headers)
        open_request = (
            urllib.request.build_opener(SafeAssetRedirect()).open
            if spec.api_url
            else urllib.request.urlopen
        )
        with open_request(request, timeout=120) as response:
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
            if digest.hexdigest() != spec.sha256:
                raise ModelError("Downloaded model checksum mismatch. The file was discarded.")
        os.replace(temporary, target)
        temporary = None
        return target
    except ModelError:
        raise
    except RuntimeError as exc:
        raise ModelError(str(exc)) from exc
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise ModelError("The model artifact could not be retrieved or verified.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
