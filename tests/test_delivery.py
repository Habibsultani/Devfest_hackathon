import hashlib
import json
import zipfile

import pytest

from scripts.doctor import collect_checks
from scripts.package_source import build_archive


def test_source_archive_excludes_secrets_data_and_models_and_verifies_hashes(tmp_path):
    sources = {
        "app.py": b"# application\n",
        ".streamlit/config.toml": b"[theme]\n",
        "scripts/start.sh": b"#!/bin/bash\n",
        ".env": b"private",
        ".streamlit/secrets.toml": b"private",
        "data/raw/image.png": b"private",
        "models/checkpoint.ckpt": b"private",
        ".git/config": b"private",
        "docs/~$open-presentation.pptx": b"temporary Office metadata",
        ".venv-gpu/Lib/model.py": b"environment",
        ".venv-review/Lib/module.py": b"environment",
        "inspection-private.csv": b"private",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    output = tmp_path / "dist/source.zip"
    result = build_archive(tmp_path, output, list(sources), revision="fixture", dirty=True)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert names == {
            "medinspect-ai/app.py",
            "medinspect-ai/.streamlit/config.toml",
            "medinspect-ai/scripts/start.sh",
            "medinspect-ai/SOURCE_MANIFEST.json",
        }
        manifest = json.loads(archive.read("medinspect-ai/SOURCE_MANIFEST.json"))
        for name, record in manifest["files"].items():
            content = archive.read("medinspect-ai/" + name)
            assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert archive.getinfo("medinspect-ai/scripts/start.sh").external_attr >> 16 == 0o100755
    assert result["file_count"] == 3
    assert output.with_suffix(".zip.sha256").read_text().startswith(result["sha256"])


def test_source_archive_is_reproducible_and_rejects_overwrites(tmp_path):
    (tmp_path / "app.py").write_text("content", encoding="utf-8")
    args = dict(root=tmp_path, names=["app.py"], revision="fixture", dirty=False)
    first = build_archive(output=tmp_path / "a.zip", **args)
    second = build_archive(output=tmp_path / "b.zip", **args)
    assert first["sha256"] == second["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        build_archive(output=tmp_path / "a.zip", **args)


@pytest.mark.parametrize("size", [0.00001, float("nan"), -1])
def test_packaging_size_limits_fail_without_leaving_a_zip(tmp_path, size):
    (tmp_path / "app.py").write_text("content", encoding="utf-8")
    output = tmp_path / "dist/source.zip"
    with pytest.raises(ValueError):
        build_archive(
            tmp_path, output, ["app.py"], revision="fixture", dirty=False, max_size_mb=size
        )
    assert not output.exists()
    assert not list(output.parent.glob("*.part"))


def test_source_archive_rejects_symlink_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("outside", encoding="utf-8")
    try:
        (project / "link.txt").symlink_to(private)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks.")
    with pytest.raises(ValueError, match="outside the project"):
        build_archive(
            project, project / "dist/source.zip", ["link.txt"], revision="fixture", dirty=False
        )


def test_doctor_keeps_application_and_model_readiness_separate(tmp_path, monkeypatch):
    for name in (
        "requirements.txt",
        "app.py",
        "assets/styles.css",
        ".streamlit/config.toml",
        "src/ui.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    checks = collect_checks(root=tmp_path, config_path=tmp_path / "missing-model.json")
    assert all(check.status == "ok" for check in checks if check.scope == "application")
    assert all(check.status == "pending" for check in checks if check.scope == "model")
    (tmp_path / "assets/styles.css").unlink()
    checks = collect_checks(root=tmp_path, config_path=tmp_path / "missing-model.json")
    assert any(check.name == "assets/styles.css" and check.status == "error" for check in checks)
