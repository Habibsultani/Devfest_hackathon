import io
import urllib.error
from dataclasses import replace

import pytest

from src.artifacts import ensure_all_artifacts, ensure_artifact, ensure_spec
from src.model_config import ArtifactSpec, ModelError


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


def test_auxiliary_artifact_is_required_and_checksum_verified(config, tmp_path):
    auxiliary = ArtifactSpec(tmp_path / "mobile_sam.pt", config.artifact_sha256)
    combined = replace(config, auxiliary_artifacts=(auxiliary,))
    with pytest.raises(ModelError, match="artifact is missing"):
        ensure_all_artifacts(combined)
    auxiliary.path.write_bytes(b"tampered SAM")
    with pytest.raises(ModelError, match="checksum mismatch"):
        ensure_all_artifacts(combined)
    auxiliary.path.write_bytes(config.artifact_path.read_bytes())
    assert ensure_all_artifacts(combined) == config.artifact_path


@pytest.mark.parametrize(
    ("code", "headers", "message"),
    [
        (401, {}, "credential"),
        (403, {}, "draft release"),
        (404, {}, "draft release"),
        (403, {"X-RateLimit-Remaining": "0"}, "rate limit"),
        (429, {}, "rate limit"),
        (503, {}, "download service"),
    ],
)
def test_http_failure_is_actionable_without_exposing_credentials(
    tmp_path, monkeypatch, code, headers, message
):
    spec = ArtifactSpec(
        tmp_path / "model.pt", "a" * 64, api_url="https://api.github.com/repos/team/assets/1"
    )

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            spec.api_url + "?secret=signed-value",
            code,
            "private response details",
            headers,
            io.BytesIO(b"private response body"),
        )

    monkeypatch.setattr("src.artifacts.github_token", lambda: "private-token")
    monkeypatch.setattr("urllib.request.OpenerDirector.open", fail)
    with pytest.raises(ModelError) as caught:
        ensure_spec(spec)
    text = str(caught.value)
    assert f"model.pt (HTTP {code})" in text
    assert message in text
    assert all(
        secret not in text for secret in ("private-token", "signed-value", "private response")
    )
    assert not spec.path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_public_auxiliary_host_denial_does_not_blame_github_token(tmp_path, monkeypatch):
    spec = ArtifactSpec(tmp_path / "mobile_sam.pt", "a" * 64, "https://weights.example/model.pt")

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(spec.url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(ModelError, match=r"mobile_sam.pt \(HTTP 403\)") as caught:
        ensure_spec(spec)
    assert "GITHUB_TOKEN" not in str(caught.value)
