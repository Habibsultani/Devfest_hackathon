import sys
import types

import numpy as np
import pytest
from PIL import Image

from src.model_config import ModelError
from src.model_gateway import ModelGateway, load_gateway, validate_result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("anomaly_score", float("nan")),
        ("anomaly_score", float("inf")),
        ("anomaly_score", True),
        ("anomaly_score", "2.5"),
        ("anomaly_map", np.zeros((3, 2))),
        ("anomaly_map", np.zeros((1, 2, 3))),
        ("anomaly_map", [[1, 2, 3], [1, 2, 3]]),
        ("anomaly_map", np.full((2, 3), np.nan)),
        ("anomaly_map", np.full((2, 3), "bad")),
        ("requires_review", "False"),
        ("requires_review", False),
        ("model_version", "different-model"),
    ],
)
def test_invalid_model_outputs_fail_closed(config, raw_result, field, value):
    raw_result[field] = value
    with pytest.raises(ModelError, match="Invalid inference result"):
        validate_result(raw_result, (3, 2), config)


def test_threshold_equality_and_map_snapshot(config, raw_result):
    raw_result["anomaly_score"] = np.float32(config.threshold)
    result = validate_result(raw_result, (3, 2), config)
    assert result.requires_review is True
    raw_result["anomaly_map"][:] = 99
    assert result.anomaly_map[0, 0] == 0
    assert not result.anomaly_map.flags.writeable


def test_missing_field(config, raw_result):
    del raw_result["model_version"]
    with pytest.raises(ModelError):
        validate_result(raw_result, (3, 2), config)


def test_calls_p1_once_without_resizing_or_normalizing(config, raw_result):
    calls = []
    original = Image.new("RGB", (3, 2), (100, 120, 140))

    def inspect(image):
        calls.append((image.size, image.mode, image.getpixel((0, 0))))
        image.putpixel((0, 0), (0, 0, 0))
        return raw_result

    gateway = ModelGateway(config, inspect)
    gateway.inspect(original, config.instrument_class)
    assert calls == [((3, 2), "RGB", (100, 120, 140))]
    assert original.getpixel((0, 0)) == (100, 120, 140)


def test_unsupported_class_never_calls_inference(config):
    def should_not_run(image):
        pytest.fail("Unsupported instrument reached the model")

    with pytest.raises(ModelError, match="unsupported"):
        ModelGateway(config, should_not_run).inspect(Image.new("RGB", (3, 2)), "other")


def test_model_exception_never_becomes_a_prediction(config):
    def broken(image):
        raise RuntimeError("checkpoint loading failed")

    with pytest.raises(ModelError, match="No inspection was produced"):
        ModelGateway(config, broken).inspect(Image.new("RGB", (3, 2)), config.instrument_class)


def test_loads_only_p1_api_after_artifact_verification(config, raw_result, monkeypatch):
    module = types.ModuleType("src.inference")
    module.inspect_image = lambda image: raw_result
    monkeypatch.setitem(sys.modules, "src.inference", module)
    result = load_gateway(config).inspect(Image.new("RGB", (3, 2)), config.instrument_class)
    assert result.anomaly_score == 2.5


def test_missing_inference_has_explicit_error(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.inference", None)
    with pytest.raises(ModelError, match="inference module"):
        load_gateway(config)
