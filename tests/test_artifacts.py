import io

import pytest

from src.artifacts import ensure_artifact
from src.model_config import ModelError


class DownloadResponse(io.BytesIO):
    def geturl(self):
        return "https://example.invalid/fixture.ckpt"


def test_local_artifact_verification_and_tampering(config):
    assert ensure_artifact(config) == config.artifact_path
    config.artifact_path.write_bytes(b"tampered")
    with pytest.raises(ModelError, match="checksum mismatch"):
        ensure_artifact(config)


def test_missing_artifact_has_no_substitute(config):
    config.artifact_path.unlink()
    with pytest.raises(ModelError, match="artifact is missing"):
        ensure_artifact(config)


def test_download_is_checked_before_atomic_install(downloadable_config, config, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: DownloadResponse(config.artifact_path.read_bytes()),
    )
    target = ensure_artifact(downloadable_config)
    assert target.read_bytes() == config.artifact_path.read_bytes()
    assert not list(target.parent.glob("*.part"))


def test_failed_checksum_discards_download(downloadable_config, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: DownloadResponse(b"wrong checkpoint")
    )
    with pytest.raises(ModelError, match="checksum mismatch"):
        ensure_artifact(downloadable_config)
    assert not downloadable_config.artifact_path.exists()
    assert not list(downloadable_config.artifact_path.parent.glob("*.part"))
