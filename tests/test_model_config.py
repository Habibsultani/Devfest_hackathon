import json

import pytest

from src.model_config import PROJECT_ROOT, ModelError, load_model_config, load_model_configs


@pytest.fixture
def config_dict(config):
    return {
        "schema_version": 1,
        "model_version": config.model_version,
        "instrument_class": config.instrument_class,
        "threshold": config.threshold,
        "artifact": {"path": str(config.artifact_path), "sha256": config.artifact_sha256},
        "heatmap_scale": {"min": config.heatmap_min, "max": config.heatmap_max},
    }


def test_missing_config_is_explicit(tmp_path):
    with pytest.raises(ModelError, match="configuration is missing"):
        load_model_config(tmp_path / "missing.json")


def test_valid_config_and_stable_fingerprint(tmp_path, config_dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    config = load_model_config(path)
    assert config.threshold == 2.0
    assert config.config_sha256 == load_model_config(path).config_sha256


@pytest.mark.parametrize("threshold", [None, True, "2.0", float("nan")])
def test_config_rejects_placeholder_or_invalid_threshold(tmp_path, config_dict, threshold):
    config_dict["threshold"] = threshold
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    with pytest.raises(ModelError, match="threshold"):
        load_model_config(path)


def test_example_cannot_be_mistaken_for_a_working_model():
    with pytest.raises(ModelError):
        load_model_config("models/model_config.example.json")


def test_boolean_schema_version_is_not_an_integer_version(tmp_path, config_dict):
    config_dict["schema_version"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    with pytest.raises(ModelError, match="schema_version"):
        load_model_config(path)


def test_canonical_registry_resolves_both_classes_and_all_artifacts(monkeypatch):
    monkeypatch.delenv("MEDINSPECT_MODEL_CONFIG", raising=False)
    configs = load_model_configs()
    assert set(configs) == {"instrument12", "instrument19"}
    assert configs["instrument12"].threshold != configs["instrument19"].threshold
    for name, config in configs.items():
        assert config.instrument_class == name
        assert config.backend == "dinomaly-sam-tiles"
        assert config.config_path == PROJECT_ROOT / "models" / name / "dinomaly_config.json"
        assert config.artifact_path.parent == PROJECT_ROOT / "models/dinomaly"
        assert len(config.artifacts) == 2
        assert config.artifacts[1].path == PROJECT_ROOT / "models/mobile_sam.pt"
        assert config.heatmap_max > config.threshold


def test_override_is_single_class_and_never_silently_falls_back(monkeypatch):
    path = PROJECT_ROOT / "models/instrument19/dinomaly_config.json"
    monkeypatch.setenv("MEDINSPECT_MODEL_CONFIG", str(path))
    assert set(load_model_configs()) == {"instrument19"}
    monkeypatch.setenv("MEDINSPECT_MODEL_CONFIG", "does-not-exist.json")
    with pytest.raises(ModelError, match="configuration is missing"):
        load_model_configs()
