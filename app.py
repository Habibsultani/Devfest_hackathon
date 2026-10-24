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
from src.reporting import (
    DECISIONS,
    create_record,
    filter_records,
    records_to_csv,
    validate_identifier,
)
from src.ui import (
    apply_design_system,
    empty_state,
    icon,
    panel_heading,
    render_footer,
    render_header,
    render_progress,
    render_session_overview,
    render_sidebar,
)
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


def start_new_inspection() -> None:
    """Reset the working image and ID, keeping the batch, class, and saved records."""
    st.session_state.pop("active_inspection", None)
    st.session_state.pop("inspection_context", None)
    st.session_state.pop("instrument_id", None)
    st.session_state.upload_revision = st.session_state.get("upload_revision", 0) + 1


def render_result(snapshot: dict) -> None:
    result = snapshot["result"]
    config = snapshot["config"]
    with st.container(key="results_panel"):
        panel_heading(
            "03", "Analysis result", "Model observations for this image.", "center_focus_strong"
        )
        if result.requires_review:
            st.warning("AI screening: flagged for human review", icon=":material/flag:")
        else:
            st.info(
                "AI screening: below the review threshold. An operator decision is still required.",
                icon=":material/visibility:",
            )
        score, threshold, latency = st.columns(3)
        score.metric("Anomaly score", f"{result.anomaly_score:.6g}")
        threshold.metric("Review threshold", f"{config.threshold:.6g}")
        latency.metric("Inference time", f"{snapshot['inference_ms'] / 1000:.2f} s")
        st.caption(
            f"Model: {result.model_version} · Class: {config.instrument_class}. "
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
        original_column, overlay_column = st.columns(2, gap="medium")
        with original_column:
            st.html(f'<div class="mi-image-label">{icon("image")}Original image</div>')
            st.image(snapshot["image"], width="stretch")
        with overlay_column:
            st.html(f'<div class="mi-image-label">{icon("layers")}Suspicious regions</div>')
            st.image(overlay, width="stretch")
        st.image(color_legend(), width=360)
        st.caption(
            f"Lower anomaly ({config.heatmap_min:g}) → higher anomaly ({config.heatmap_max:g}). "
            "Fixed display scale; out-of-range colors are clipped. "
            "Colors do not identify a defect type."
        )
        with st.expander("Anomaly map and inspection details", icon=":material/info:"):
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
    with st.container(key="review_panel"):
        panel_heading(
            "04",
            "Operator review",
            "Your assessment stays separate from the AI flag.",
            "rate_review",
        )
        with st.form(f"review_{snapshot['inspection_id']}"):
            decision = st.selectbox(
                "Operator decision", DECISIONS, index=None, placeholder="Choose a decision"
            )
            note = st.text_area(
                "Operator note",
                placeholder="Describe what you observed and why you chose this decision…",
                max_chars=2000,
                help="Required for an Inconclusive decision.",
            )
            save = st.form_submit_button(
                "Save inspection", disabled=saved, type="primary", icon=":material/check:"
            )
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
        st.button(
            "Start new inspection",
            icon=":material/add:",
            on_click=start_new_inspection,
        )


def render_inspection(config: ModelConfig | None) -> None:
    surface = st.container(key="inspection_surface")
    with surface:
        progress = st.empty()
        details_column, image_column = st.columns([1, 1.3], gap="medium")
    with details_column, st.container(key="instrument_panel"):
        panel_heading(
            "01", "Instrument details", "Keep every inspection linked to its instrument.", "sell"
        )
        left, right = st.columns(2)
        batch_id = left.text_input(
            "Batch ID", placeholder="e.g. BATCH-001", max_chars=80, key="batch_id"
        )
        instrument_id = right.text_input(
            "Instrument ID", placeholder="e.g. INST-014", max_chars=80, key="instrument_id"
        )
        classes = ["Select instrument class"]
        if config:
            classes.append(config.instrument_class)
        classes.append("Other / unknown")
        instrument_class = st.selectbox("Instrument class", classes, key="instrument_class")
        st.html(
            f'<div class="mi-field-note">{icon("info")}<span>All fields are required. '
            "Confirm the instrument class before analysis.</span></div>"
        )
        if instrument_class == "Other / unknown":
            st.warning("This model cannot inspect an unsupported or unknown instrument class.")
        details_feedback = st.empty()

    with image_column, st.container(key="upload_panel"):
        panel_heading(
            "02", "Instrument image", "Upload a clear image of the full instrument.", "image"
        )
        uploaded = st.file_uploader(
            "Upload instrument image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"instrument_upload_{st.session_state.get('upload_revision', 0)}",
        )
        st.html(
            '<div class="mi-format-list">'
            f"<span>{icon('image')}JPEG or PNG</span>"
            f"<span>{icon('hard_drive')}Up to 10 MB</span>"
            f"<span>{icon('aspect_ratio')}Max. 12 megapixels</span></div>"
        )
        st.caption("Use one still image. Good lighting and a consistent background help.")
        preview = st.empty()
        upload_feedback = st.empty()
    image = None
    image_sha256 = None
    if uploaded is not None:
        content = uploaded.getvalue()
        try:
            image = decode_upload(content, uploaded.name)
            image_sha256 = hashlib.sha256(content).hexdigest()
        except UploadError as exc:
            upload_feedback.error(str(exc))

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
    if image is not None and "active_inspection" not in st.session_state:
        preview.image(image, caption="Image ready for inspection", width="stretch")

    metadata_error = None
    try:
        validate_identifier(batch_id, "Batch ID")
        validate_identifier(instrument_id, "Instrument ID")
    except ValueError as exc:
        metadata_error = str(exc)

    supported = config is not None and instrument_class == config.instrument_class
    if image is not None and metadata_error:
        details_feedback.info(metadata_error)
    ready = image is not None and config is not None and supported and not metadata_error
    with surface, st.container(key="analysis_action"):
        hint_column, action_column = st.columns([1.9, 1], vertical_alignment="center")
        if config is None:
            hint_title, hint_text = (
                "Analysis is waiting for a model",
                "You can prepare your instrument details and image in the meantime.",
            )
        elif image is None:
            hint_title, hint_text = (
                "Add an image to continue",
                "Upload a supported JPEG or PNG image.",
            )
        elif metadata_error or not supported:
            hint_title, hint_text = (
                "Complete the instrument details",
                "Add both IDs and confirm a supported instrument class.",
            )
        else:
            hint_title, hint_text = (
                "Ready to analyze",
                "Review the model observations before recording your decision.",
            )
        hint_column.html(
            f'<div class="mi-analysis-hint">{icon("center_focus_strong" if ready else "info")}'
            f"<div><strong>{hint_title}</strong><p>{hint_text}</p></div></div>"
        )
        analyze = action_column.button(
            "Analyze instrument",
            type="primary",
            disabled=not ready,
            icon=":material/center_focus_strong:",
            width="stretch",
        )
    if analyze:
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
    else:
        with st.container(key="results_panel"):
            empty_state(
                "Analysis & review",
                "The anomaly score, visual map, and operator review will appear here "
                "after a successful analysis.",
                "center_focus_strong",
                compact=True,
            )
    active = st.session_state.get("active_inspection")
    saved = active is not None and any(
        record.inspection_id == active["inspection_id"] for record in st.session_state.records
    )
    with progress.container():
        render_progress(
            details_ready=not metadata_error and supported,
            image_ready=image is not None,
            analyzed=active is not None,
            saved=saved,
        )


def render_report() -> None:
    st.subheader("Session report")
    st.caption(
        "Records belong to this browser session. Download the CSV before leaving or restarting."
    )
    records = st.session_state.records
    if not records:
        with st.container(key="report_panel"):
            empty_state(
                "No inspections saved yet.",
                "Analyze an image and record an operator decision. "
                "Your saved inspections and CSV export will appear here.",
                "assignment",
            )
        return
    search_column, decision_column = st.columns([2, 1], vertical_alignment="bottom")
    query = search_column.text_input(
        "Search inspections",
        placeholder="Search IDs, class, model, or notes…",
        key="report_search",
    )
    decision = decision_column.selectbox(
        "Filter by decision", ["All decisions", *DECISIONS], key="report_decision"
    )
    visible = filter_records(records, query=query, decision=decision)
    st.caption(
        f"Showing {len(visible)} of {len(records)} saved inspections. CSV includes all records."
    )
    if not visible:
        with st.container(key="report_panel"):
            empty_state(
                "No matching inspections", "Try a different search or decision filter.", "search"
            )
    st.dataframe(
        [asdict(record) for record in reversed(visible)],
        hide_index=True,
        width="stretch",
        column_order=[
            "timestamp_utc",
            "batch_id",
            "instrument_id",
            "anomaly_score",
            "requires_review",
            "operator_decision",
            "model_version",
        ],
        column_config={
            "timestamp_utc": st.column_config.TextColumn("Saved at (UTC)"),
            "batch_id": "Batch ID",
            "instrument_id": "Instrument ID",
            "anomaly_score": st.column_config.NumberColumn("AI score", format="%.4f"),
            "requires_review": st.column_config.CheckboxColumn("AI review flag"),
            "operator_decision": "Operator decision",
            "model_version": "Model version",
        },
    )
    st.download_button(
        "Download inspection CSV",
        data=records_to_csv(records),
        file_name=f"inspection-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv",
        mime="text/csv",
        type="primary",
        icon=":material/download:",
        help="Exports every saved inspection in this session, regardless of the current filters.",
    )
    st.caption(
        "Timestamps are UTC. CSV text beginning with formula characters is escaped for Excel."
    )


def main() -> None:
    st.set_page_config(
        page_title="MedInspect AI", page_icon=":material/center_focus_strong:", layout="wide"
    )
    apply_design_system()
    st.session_state.setdefault("records", [])
    config, config_error = None, None
    try:
        config = load_model_config(os.environ.get("MEDINSPECT_MODEL_CONFIG"))
    except ModelError as exc:
        config_error = str(exc)
    render_sidebar(config, config_error)
    render_header()
    overview = st.empty()
    inspect_tab, report_tab, about_tab = st.tabs(["Inspect instrument", "Session report", "About"])
    with inspect_tab:
        render_inspection(config)
    with report_tab:
        render_report()
    with about_tab:
        render_about()
    # Populate this reserved position after input changes have invalidated stale analyses.
    with overview.container():
        render_session_overview(st.session_state.records, st.session_state.get("active_inspection"))
    render_footer(NOTICE)


def render_about() -> None:
    st.subheader("Designed for human assessment")
    st.caption(
        "Understand what the screening result can tell you, and where your judgment matters."
    )
    workflow_column, limits_column = st.columns(2, gap="medium")
    with workflow_column, st.container(key="about_workflow"):
        panel_heading(
            "01", "How screening works", "Image evidence, followed by your decision.", "science"
        )
        st.html(
            '<div class="mi-about-item">'
            f"{icon('compare')}<div><strong>Compare with learned normal appearances</strong>"
            "<p>PatchCore produces a model-specific score and anomaly map. "
            "The score is not a defect probability.</p></div></div>"
            '<div class="mi-about-item">'
            f"{icon('visibility')}<div><strong>Review suspicious regions</strong>"
            "<p>The fixed-scale overlay draws attention to unusual appearances. "
            "It does not classify regions as cracks or corrosion.</p></div></div>"
            '<div class="mi-about-item">'
            f"{icon('rate_review')}<div><strong>Record an independent decision</strong>"
            "<p>Choose Accept, Flag for review, or Inconclusive. "
            "An Accept decision is an operator record, not a safety certification.</p></div></div>"
        )
    with limits_column, st.container(key="about_limits"):
        panel_heading("02", "Know the limits", "Use the result in its proper context.", "shield")
        st.html(
            '<div class="mi-about-item">'
            f"{icon('photo_camera')}<div><strong>Image conditions matter</strong>"
            "<p>Lighting, background, viewpoint, and unsupported instruments can "
            "cause false alarms or missed defects.</p></div></div>"
            '<div class="mi-about-item">'
            f"{icon('history')}<div><strong>History belongs to this session</strong>"
            "<p>Images and records are processed in server memory. There is no database "
            "or persistent multi-user history. Download your CSV before leaving.</p></div></div>"
            '<div class="mi-about-item">'
            f"{icon('lock')}<div><strong>Upload appropriate images</strong>"
            "<p>Use only images you are authorized to upload. Avoid patient information. "
            "This is a research prototype, with no safety or sterility certification."
            "</p></div></div>"
        )


if __name__ == "__main__":
    main()
