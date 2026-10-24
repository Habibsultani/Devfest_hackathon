import hashlib
from dataclasses import replace

import numpy as np
import pytest

from src.model_config import ModelConfig
from src.model_gateway import validate_result


@pytest.fixture
def config(tmp_path):
    artifact = tmp_path / "fixture.ckpt"
    # This is an artifact-verification fixture, never a trained model.
    artifact.write_bytes(b"test fixture - not a model")
    return ModelConfig(
        model_version="fixture-v1",
        instrument_class="fixture-class",
        threshold=2.0,
        artifact_path=artifact,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        artifact_url=None,
        heatmap_min=0.0,
        heatmap_max=4.0,
        config_sha256="a" * 64,
    )


@pytest.fixture
def raw_result(config):
    return {
        "anomaly_score": 2.5,
        "anomaly_map": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.float32),
        "requires_review": True,
        "model_version": config.model_version,
    }


@pytest.fixture
def result(config, raw_result):
    return validate_result(raw_result, (3, 2), config)


@pytest.fixture
def downloadable_config(config, tmp_path):
    return replace(
        config,
        artifact_path=tmp_path / "download" / "fixture.ckpt",
        artifact_url="https://example.invalid/trusted-fixture.ckpt",
    )
