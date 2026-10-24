import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.calibrate_display import calibrate_display


def test_display_calibration_reads_validation_images_only(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"frozen manifest")
    config_path = tmp_path / "model.json"
    config = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "instrument_class": "instrument19",
        "threshold": 0.7,
        "artifact_sha256": "unchanged",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    validation = tmp_path / "validation.png"
    Image.new("RGB", (10, 10)).save(validation)
    rows = [
        {
            "split": split,
            "instrument_class": "instrument19",
            "image_path": validation if split == "validation" else tmp_path / "must-not-read.png",
        }
        for split in ("train", "validation", "test")
    ]
    calls = []

    def inspect(image, *, instrument_class):
        calls.append(instrument_class)
        return {"anomaly_map": np.full((10, 10), 0.75, dtype=np.float32)}

    monkeypatch.setattr("scripts.calibrate_display.load_manifest", lambda _: rows)
    monkeypatch.setattr(
        "scripts.calibrate_display.get_inspector", lambda _: SimpleNamespace(inspect=inspect)
    )
    scale = calibrate_display(config_path, manifest)
    assert calls == ["instrument19"]
    assert scale["max"] == 0.75
    saved = json.loads(config_path.read_text())
    assert saved.pop("heatmap_scale") == scale
    assert saved == config
    manifest.write_bytes(b"changed manifest")
    with pytest.raises(ValueError, match="manifest differs"):
        calibrate_display(config_path, manifest)
