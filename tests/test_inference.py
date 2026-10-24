import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from src.dataset import load_manifest
from src.inference import DEFAULT_CONFIG, Inspector, ModelLoadError, get_inspector, inspect_image, supported_classes


def test_missing_model_has_explicit_error(tmp_path):
    with pytest.raises(ModelLoadError, match="missing or invalid"):
        get_inspector(tmp_path / "missing.json")
    path = tmp_path / "config.json"
    path.write_text('{"status":"untrained"}', encoding="utf-8")
    with pytest.raises(ModelLoadError, match="No trained model"):
        get_inspector(path)


@pytest.mark.parametrize("contents", ["[]", "null", "broken json"])
def test_invalid_configuration_rejected(tmp_path, contents):
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ModelLoadError):
        get_inspector(path)


def test_unsupported_declared_class_is_rejected():
    with pytest.raises(ValueError, match="Unsupported instrument class"):
        inspect_image(Image.new("RGB", (32, 32)), instrument_class="instrument20")
    assert supported_classes() == ["instrument12", "instrument19"]


@pytest.fixture
def trained_model():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    artifact = DEFAULT_CONFIG.parent / config["artifact_path"]
    manifest = Path("data/processed/instrument12/manifest.csv")
    if config.get("status") != "trained" or not artifact.is_file() or not manifest.is_file():
        pytest.skip("Local trained artifact and dataset required; run the pipeline first.")
    return get_inspector(DEFAULT_CONFIG), load_manifest(manifest)


@pytest.mark.integration
def test_real_unseen_images_produce_dynamic_scores_and_aligned_maps(trained_model):
    inspector, rows = trained_model
    normal = next(r for r in rows if r["split"] == "test" and not r["label"])
    defect = next(r for r in rows if r["split"] == "test" and r["label"])
    predictions = []
    for row in (normal, defect):
        with Image.open(row["image_path"]) as image:
            result = inspector.inspect(image)
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["requires_review"], bool)
        assert result["anomaly_map"].shape == (int(row["height"]), int(row["width"]))
        assert result["anomaly_map"].dtype == np.float32
        assert np.isfinite(result["anomaly_map"]).all()
        assert np.ptp(result["anomaly_map"]) > 0
        predictions.append(result)
    assert predictions[0]["anomaly_score"] != predictions[1]["anomaly_score"]


@pytest.mark.integration
def test_reload_in_a_fresh_process_matches_saved_model(trained_model, tmp_path):
    inspector, rows = trained_model
    probe = next(r for r in rows if r["split"] == "test")
    with Image.open(probe["image_path"]) as image:
        expected = inspector.inspect(image)
    map_path = tmp_path / "reloaded.npy"
    output = subprocess.check_output([sys.executable, "-m", "scripts.inspect_image", str(probe["image_path"]), "--map-output", str(map_path)], text=True)
    actual = json.loads(output)
    assert actual["anomaly_score"] == pytest.approx(expected["anomaly_score"], rel=1e-6)
    assert actual["anomaly_map_shape"] == list(expected["anomaly_map"].shape)
    np.testing.assert_allclose(np.load(map_path), expected["anomaly_map"], rtol=1e-6, atol=1e-6)


@pytest.mark.integration
def test_modified_artifact_and_wrong_class_rejected(trained_model, tmp_path):
    inspector, _ = trained_model
    with pytest.raises(ValueError, match="supports only"):
        inspector.inspect(Image.new("RGB", (32, 32)), instrument_class="instrument2")
    config = dict(inspector.config)
    config["artifact_path"] = "corrupt.pt"
    (tmp_path / "corrupt.pt").write_bytes(b"invalid checkpoint")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        get_inspector(path)


@pytest.mark.integration
@pytest.mark.parametrize("instrument", ["instrument12", "instrument19"])
def test_declared_class_routes_to_its_own_saved_model(trained_model, instrument):
    rows = load_manifest(Path("data/processed") / instrument / "manifest.csv")
    probe = next(r for r in rows if r["split"] == "test")
    with Image.open(probe["image_path"]) as image:
        result = inspect_image(image, instrument_class=instrument)
    assert result["instrument_class"] == instrument
    assert result["model_version"] == "medinspect-dinomaly-sam-lora-v4"
    assert np.isfinite(result["anomaly_score"])
