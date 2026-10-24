"""UI integration fixtures only. These tests do not measure real PatchCore accuracy."""

import io

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.model_config import ModelError
from src.model_gateway import ModelGateway


class UploadedFixture(io.BytesIO):
    name = "fixture.png"


def uploaded_fixture(color="white"):
    output = UploadedFixture()
    Image.new("RGB", (3, 2), color).save(output, format="PNG")
    return output


def by_label(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture(autouse=True)
def clear_resource_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def app_harness(config, raw_result, monkeypatch):
    upload = {"file": uploaded_fixture()}
    calls = []

    def inspect(image):
        calls.append(image)
        return raw_result

    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: upload["file"])
    monkeypatch.setattr(
        "src.model_config.load_model_configs", lambda: {config.instrument_class: config}
    )
    monkeypatch.setattr(
        "src.model_gateway.load_gateway", lambda config: ModelGateway(config, inspect)
    )
    app = AppTest.from_file("app.py", default_timeout=10).run()
    by_label(app.text_input, "Batch ID").set_value("BATCH-001")
    by_label(app.text_input, "Instrument ID").set_value("INST-014")
    by_label(app.selectbox, "Instrument class").select(config.instrument_class)
    app.run()
    assert not app.exception
    return app, upload, calls


def test_missing_model_launches_without_crashing(monkeypatch):
    def missing(*args):
        raise ModelError("Model configuration is missing.")

    monkeypatch.setattr("src.model_config.load_model_configs", missing)
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert by_label(app.button, "Analyze instrument").disabled
    assert any("configuration is missing" in warning.value for warning in app.warning)
    assert len(app.session_state.records) == 0


def test_analyze_review_export_and_no_duplicate_or_repeated_inference(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert by_label(app.metric, "Anomaly score").value == "2.5"
    by_label(app.selectbox, "Operator decision").select("Accept")
    by_label(app.text_area, "Operator note").set_value("Reviewed by operator")
    by_label(app.button, "Save inspection").click().run()
    assert not app.exception
    records = app.session_state.records
    assert len(records) == 1
    assert records[0].batch_id == "BATCH-001"
    assert records[0].requires_review is True
    assert records[0].operator_decision == "Accept"
    assert by_label(app.button, "Save inspection").disabled
    assert len(app.get("download_button")) == 1
    app.run()
    assert len(app.session_state.records) == 1
    assert len(calls) == 1


def test_new_image_or_changed_metadata_cannot_reuse_stale_result(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    upload["file"] = uploaded_fixture("red")
    app.run()
    assert "active_inspection" not in app.session_state
    assert not any(button.label == "Save inspection" for button in app.button)
    by_label(app.button, "Analyze instrument").click().run()
    assert len(calls) == 2
    by_label(app.text_input, "Instrument ID").set_value("INST-015").run()
    assert "active_inspection" not in app.session_state


def test_failure_clears_previous_result_and_never_saves(app_harness, monkeypatch):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    assert "active_inspection" in app.session_state

    def unavailable(*args):
        raise ModelError("Trained checkpoint unavailable.")

    st.cache_resource.clear()
    monkeypatch.setattr("src.model_gateway.load_gateway", unavailable)
    by_label(app.button, "Analyze instrument").click().run()
    assert not app.exception
    assert "active_inspection" not in app.session_state
    assert len(app.session_state.records) == 0
    assert any("checkpoint unavailable" in error.value for error in app.error)


def test_review_requires_explicit_decision(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    by_label(app.button, "Save inspection").click().run()
    assert len(app.session_state.records) == 0
    assert any("Choose an operator decision" in error.value for error in app.error)


def test_unknown_class_clears_result_and_disables_analysis(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    by_label(app.selectbox, "Instrument class").select("Other / unknown").run()
    assert not app.exception
    assert "active_inspection" not in app.session_state
    assert by_label(app.button, "Analyze instrument").disabled
    assert len(calls) == 1


def test_next_inspection_preserves_saved_records_and_batch(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    by_label(app.selectbox, "Operator decision").select("Accept")
    by_label(app.button, "Save inspection").click().run()
    by_label(app.button, "Start new inspection").click().run()
    assert not app.exception
    assert len(app.session_state.records) == 1
    assert by_label(app.text_input, "Batch ID").value == "BATCH-001"
    assert by_label(app.text_input, "Instrument ID").value == ""
    assert app.session_state.upload_revision == 1
    assert "active_inspection" not in app.session_state
    assert len(calls) == 1


def test_report_filter_has_an_empty_state_without_losing_csv(app_harness):
    app, upload, calls = app_harness
    by_label(app.button, "Analyze instrument").click().run()
    by_label(app.selectbox, "Operator decision").select("Accept")
    by_label(app.button, "Save inspection").click().run()
    by_label(app.text_input, "Search inspections").set_value("not-present").run()
    assert not app.exception
    assert len(app.session_state.records) == 1
    assert len(app.get("download_button")) == 1
