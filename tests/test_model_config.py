import json

import pytest

from src.model_config import ModelError, load_model_config


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
