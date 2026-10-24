"""MedInspect AI — Person Two's Streamlit inspection and review application."""

import hashlib
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

import streamlit as st

from src.model_config import ModelConfig, ModelError, load_model_config
from src.model_gateway import ModelGateway, load_gateway
from src.reporting import DECISIONS, create_record, records_to_csv, validate_identifier
from src.uploads import UploadError, decode_upload
from src.visualization import color_legend, create_overlay

LOGGER = logging.getLogger(__name__)
NOTICE = (
    "Research prototype for visible-defect screening. Human review is required. "
    "Results do not certify instrument quality, medical-device safety, or sterility."
)


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_gateway(config: ModelConfig, artifact_revision: tuple) -> ModelGateway:
    """Verify and share the backend; Person One caches the model inside inspect_image."""
    return load_gateway(config)


def artifact_revision(config: ModelConfig) -> tuple:
    try:
        stat = config.artifact_path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return ("not-downloaded",)
    except OSError as exc:
        raise ModelError("The configured artifact cannot be accessed.") from exc


def render_result(snapshot: dict) -> None:
    result = snapshot["result"]
    config = snapshot["config"]
    st.subheader("Inspection result")
    if result.requires_review:
        st.warning("AI screening: flagged for human review")
    else:
        st.info("AI screening: below the review threshold. An operator decision is still required.")
    score, threshold, latency = st.columns(3)
    score.metric("Anomaly score", f"{result.anomaly_score:.6g}")
    threshold.metric("Review threshold", f"{config.threshold:.6g}")
    latency.metric("Inference time", f"{snapshot['inference_ms'] / 1000:.2f} s")
    st.caption(
        f"Model: {result.model_version} · Class: {config.instrument_class} · "
        "Scores are model-specific values, not defect probabilities."
    )
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    heatmap, overlay = create_overlay(
        snapshot["image"],
        result.anomaly_map,
        scale_min=config.heatmap_min,
        scale_max=config.heatmap_max,
        opacity=opacity,
    )
    original_column, overlay_column = st.columns(2)
    original_column.image(snapshot["image"], caption="Original image", width="stretch")
    overlay_column.image(overlay, caption="Suspicious regions · anomaly overlay", width="stretch")
    st.image(color_legend(), width="stretch")
    st.caption(
        f"Lower anomaly ({config.heatmap_min:g}) → higher anomaly ({config.heatmap_max:g}). "
        "The color scale is fixed for this model; out-of-range colors are clipped. "
        "Colors do not identify a defect type."
    )
    with st.expander("View anomaly map and inspection details"):
        st.image(heatmap, caption="Anomaly map on the same fixed color scale", width="stretch")
        st.write(
            {
                "Batch ID": snapshot["batch_id"],
                "Instrument ID": snapshot["instrument_id"],
                "Image": snapshot["filename"],
                "Analyzed at (UTC)": snapshot["analyzed_at_utc"],
                "Image SHA-256": snapshot["image_sha256"],
                "Artifact SHA-256": config.artifact_sha256,
            }
        )

    saved = any(
        record.inspection_id == snapshot["inspection_id"] for record in st.session_state.records
    )
    st.subheader("Operator review")
    st.caption("Record your assessment separately from the AI screening result.")
    with st.form(f"review_{snapshot['inspection_id']}"):
        decision = st.selectbox(
            "Operator decision", DECISIONS, index=None, placeholder="Choose a decision"
        )
        note = st.text_area(
            "Operator note", max_chars=2000, help="Required for an Inconclusive decision."
        )
        save = st.form_submit_button("Save inspection", disabled=saved, type="primary")
    if save and not saved:
        try:
            record = create_record(
                inspection_id=snapshot["inspection_id"],
                analyzed_at_utc=snapshot["analyzed_at_utc"],
                batch_id=snapshot["batch_id"],
                instrument_id=snapshot["instrument_id"],
                source_filename=snapshot["filename"],
                image_sha256=snapshot["image_sha256"],
                result=result,
                config=config,
                inference_ms=snapshot["inference_ms"],
                decision=decision,
                note=note,
            )
            st.session_state.records.append(record)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if saved:
        st.success("Inspection saved in this session. Download it from Session report.")


def render_inspection(config: ModelConfig | None, config_error: str | None) -> None:
    st.subheader("New inspection")
    st.caption("1 · Identify the instrument    2 · Analyze an image    3 · Record your review")
    left, right = st.columns(2)
    batch_id = left.text_input("Batch ID", placeholder="BATCH-001", max_chars=80)
    instrument_id = right.text_input("Instrument ID", placeholder="INST-014", max_chars=80)
    classes = ["Select instrument class"]
    if config:
        classes.append(config.instrument_class)
    classes.append("Other / unknown")
    instrument_class = st.selectbox("Instrument class", classes)
    if instrument_class == "Other / unknown":
        st.warning("This model cannot inspect an unsupported or unknown instrument class.")
    st.caption("Confirm the class yourself; the app does not recognize the instrument type.")
    uploaded = st.file_uploader("Upload instrument image", type=["jpg", "jpeg", "png"])
    st.caption("JPEG or PNG · Up to 10 MB and 12 megapixels · One still image")
    image = None
    image_sha256 = None
    if uploaded is not None:
        content = uploaded.getvalue()
        try:
            image = decode_upload(content, uploaded.name)
            image_sha256 = hashlib.sha256(content).hexdigest()
        except UploadError as exc:
            st.error(str(exc))

    context = (
        image_sha256,
        batch_id.strip(),
        instrument_id.strip(),
        instrument_class,
        config.config_sha256 if config else None,
    )
    if st.session_state.get("inspection_context") != context:
        st.session_state.pop("active_inspection", None)
        st.session_state.inspection_context = context

    metadata_error = None
    try:
        validate_identifier(batch_id, "Batch ID")
        validate_identifier(instrument_id, "Instrument ID")
    except ValueError as exc:
        metadata_error = str(exc)

    supported = config is not None and instrument_class == config.instrument_class
    if config_error:
        st.warning(config_error)
    if image is not None and "active_inspection" not in st.session_state:
        st.image(image, caption="Image ready for inspection", width="stretch")
    if image is not None and metadata_error:
        st.info(metadata_error)
    ready = image is not None and config is not None and supported and not metadata_error
    if st.button("Analyze instrument", type="primary", disabled=not ready):
        # Drop the old result before any attempted analysis, including failed retries.
        st.session_state.pop("active_inspection", None)
        try:
            with st.spinner("Running instrument analysis…"):
                gateway = cached_gateway(config, artifact_revision(config))
                started = time.perf_counter()
                result = gateway.inspect(image, instrument_class)
                elapsed = (time.perf_counter() - started) * 1000
                snapshot = {
                    "inspection_id": str(uuid.uuid4()),
                    "analyzed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "image": image.copy(),
                    "filename": uploaded.name,
                    "image_sha256": image_sha256,
                    "batch_id": batch_id.strip(),
                    "instrument_id": instrument_id.strip(),
                    "result": result,
                    "config": config,
                    "inference_ms": elapsed,
                }
                # Rendering errors must also prevent recording an unusable inspection.
                create_overlay(
                    image,
                    result.anomaly_map,
                    scale_min=config.heatmap_min,
                    scale_max=config.heatmap_max,
                )
                st.session_state.active_inspection = snapshot
        except ModelError as exc:
            LOGGER.exception("Instrument analysis failed")
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Unexpected analysis or visualization failure")
            st.error("Analysis could not be completed. No result or inspection was saved.")
    if "active_inspection" in st.session_state:
        render_result(st.session_state.active_inspection)


def render_report() -> None:
    st.subheader("Session report")
    st.caption(
        "Records belong to this browser session. Download the CSV before leaving or restarting."
    )
    records = st.session_state.records
    if not records:
        st.info("No inspections saved yet. Analyze an image and record an operator decision.")
        return
    total, flagged, inconclusive = st.columns(3)
    total.metric("Saved inspections", len(records))
    flagged.metric("AI review flags", sum(record.requires_review for record in records))
    inconclusive.metric(
        "Operator inconclusive",
        sum(record.operator_decision == "Inconclusive" for record in records),
    )
    st.dataframe([asdict(record) for record in records], hide_index=True, width="stretch")
    st.download_button(
        "Download inspection CSV",
        data=records_to_csv(records),
        file_name=f"inspection-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(
        "Timestamps are UTC. CSV text beginning with formula characters is escaped for Excel."
    )


def main() -> None:
    st.set_page_config(page_title="MedInspect AI", page_icon="🔬", layout="wide")
    st.session_state.setdefault("records", [])
    config, config_error = None, None
    try:
        config = load_model_config(os.environ.get("MEDINSPECT_MODEL_CONFIG"))
    except ModelError as exc:
        config_error = str(exc)
    with st.sidebar:
        st.markdown("### MedInspect AI")
        st.caption("VISIBLE-DEFECT SCREENING")
        st.divider()
        if config:
            st.write("**Selected model**")
            st.write(config.model_version)
            st.caption(f"Supported class: {config.instrument_class}")
            st.caption("Artifact and inference are verified when analysis starts.")
        else:
            st.warning("Awaiting trained model")
        st.divider()
        st.metric("Session inspections", len(st.session_state.records))
        st.caption("Human review stays separate from the AI screening flag.")
    st.title("MedInspect AI")
    st.caption("AI-assisted surgical instrument inspection")
    st.info(NOTICE)
    inspect_tab, report_tab, about_tab = st.tabs(["Inspect instrument", "Session report", "About"])
    with inspect_tab:
        render_inspection(config, config_error)
    with report_tab:
        render_report()
    with about_tab:
        st.subheader("A screening aid for human reviewers")
        st.write(
            "A trained PatchCore model compares the image with learned normal appearances. "
            "The score and anomaly map highlight appearances that may need attention. "
            "Background, lighting, viewpoint, or an unsupported instrument can affect results."
        )
        st.write(
            "The operator selects Accept, Flag for review, or Inconclusive. "
            "An Accept decision is an operator record, not a safety certification. "
            "The app does not classify regions as cracks or corrosion."
        )
        st.write(
            "Uploaded images and inspection records are processed in server memory for the "
            "current session. There is no database or persistent multi-user history. "
            "Use only images you are authorized to upload; avoid patient information."
        )


if __name__ == "__main__":
    main()
