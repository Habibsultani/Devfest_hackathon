"""Shared presentation components and the MedInspect interface design system."""

from html import escape
from pathlib import Path

import streamlit as st

from src.model_config import ModelConfig

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def icon(name: str, *, size: str = "") -> str:
    """Use the same rounded Material Symbols typeface as native widget icons."""
    return f'<span class="mi-icon {escape(size)}" aria-hidden="true">{escape(name)}</span>'


def apply_design_system() -> None:
    # Reuse the pinned Streamlit font assets so the UI needs no external font service.
    media = Path(st.__file__).parent / "static" / "static" / "media"
    sans = next(media.glob("SourceSansVF-Upright*.woff2")).name
    symbols = next(media.glob("MaterialSymbols-Rounded*.woff2")).name
    fonts = f"""
    @font-face {{font-family: 'MedInspect Sans';
        src: url('static/media/{sans}') format('woff2');
        font-style: normal; font-weight: 100 900; font-display: swap;}}
    @font-face {{font-family: 'MedInspect Icons';
        src: url('static/media/{symbols}') format('woff2');
        font-style: normal; font-weight: 100 700; font-display: block;}}
    """
    st.html(f"<style>{fonts}\n{(ASSETS / 'styles.css').read_text(encoding='utf-8')}</style>")


def panel_heading(number: str, title: str, description: str, symbol: str) -> None:
    st.html(
        '<div class="mi-panel-heading">'
        f'<div class="mi-step-number">{escape(number)}</div>'
        f'<div class="mi-panel-copy"><h3>{escape(title)}</h3>'
        f"<p>{escape(description)}</p></div>{icon(symbol)}</div>"
    )


def empty_state(title: str, description: str, symbol: str, *, compact: bool = False) -> None:
    modifier = " mi-empty--compact" if compact else ""
    st.html(
        f'<div class="mi-empty{modifier}"><div class="mi-empty-icon">{icon(symbol)}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(description)}</p></div>"
    )


def render_sidebar(config: ModelConfig | None, config_error: str | None) -> None:
    with st.sidebar:
        st.html(
            '<div class="mi-brand">'
            f'<div class="mi-brand-mark">{icon("center_focus_strong")}</div>'
            '<div><div class="mi-brand-name">MedInspect<span> AI</span></div>'
            '<div class="mi-brand-subtitle">QUALITY CONTROL</div></div></div>'
            '<div class="mi-sidebar-label">YOUR WORKSPACE</div>'
            '<a class="mi-workspace-link" href="#inspection-workspace" target="_self">'
            f"{icon('dashboard')}<span>Inspection console</span>{icon('chevron_right')}</a>"
        )
        with st.container(key="model_status"):
            st.html(f'<div class="mi-sidebar-label">{icon("hub")} MODEL CONNECTION</div>')
            if config:
                st.html(
                    '<div class="mi-model-title">'
                    f"{icon('deployed_code')}<span>{escape(config.model_version)}</span></div>"
                    f'<p class="mi-sidebar-description">{escape(config.instrument_class)}</p>'
                    '<div class="mi-status mi-status--neutral">'
                    '<span class="mi-status-dot"></span>Verification on first analysis</div>'
                )
            else:
                st.html(
                    '<div class="mi-model-title">'
                    f"{icon('hourglass_top')}<span>Awaiting trained model</span></div>"
                    '<p class="mi-sidebar-description">Your workspace is ready. '
                    "Connect a trained model to enable analysis.</p>"
                    '<div class="mi-status mi-status--pending">'
                    '<span class="mi-status-dot"></span>Integration pending</div>'
                )
            if config_error:
                with st.expander("Model setup details", icon=":material/settings:"):
                    st.warning(config_error)
        st.html(
            '<div class="mi-sidebar-label">INSPECTION STEPS</div>'
            '<ol class="mi-sidebar-steps">'
            "<li><span>01</span><div>Identify the instrument"
            "<small>Add batch and instrument IDs</small></div></li>"
            "<li><span>02</span><div>Upload an image"
            "<small>Use a clear, consistent view</small></div></li>"
            "<li><span>03</span><div>Review the analysis"
            "<small>Check the model score and map</small></div></li>"
            "<li><span>04</span><div>Record your decision"
            "<small>Save and export your assessment</small></div></li></ol>"
        )
        with st.expander("Image capture tips", icon=":material/lightbulb:"):
            st.write(
                "Use a sharp, well-lit image. Keep the entire instrument visible, "
                "use the model's expected background and viewpoint, and avoid glare."
            )
        st.html(
            '<div class="mi-sidebar-bottom">'
            f"{icon('shield')}<div><strong>Human review, always</strong>"
            "<p>A screening aid for your assessment.<br>Research edition</p></div></div>"
        )


def render_header() -> None:
    st.html(
        '<div class="mi-topline"><div class="mi-breadcrumb">Workspace '
        f"{icon('chevron_right')}<span>Instrument inspection</span></div>"
        f'<span class="mi-research-tag">{icon("science")}Research prototype</span></div>'
    )
    st.title("Inspection workspace")
    st.markdown(
        '<p class="mi-page-description">Inspect an instrument image, review suspicious regions, '
        "and record your assessment.</p>",
        unsafe_allow_html=True,
    )


def render_session_overview(records: list, active: dict | None) -> None:
    pending = int(
        active is not None
        and not any(record.inspection_id == active["inspection_id"] for record in records)
    )
    items = (
        ("Saved inspections", len(records), "fact_check", "In this browser session"),
        (
            "AI review flags",
            sum(record.requires_review for record in records),
            "flag",
            "Across saved records",
        ),
        ("Awaiting your review", pending, "rate_review", "Analyzed, not yet saved"),
    )
    content = []
    for label, value, symbol, hint in items:
        content.append(
            f'<div class="mi-overview-item"><div class="mi-overview-icon">{icon(symbol)}</div>'
            f'<div><div class="mi-overview-label">{escape(label)}</div>'
            f'<div class="mi-overview-value">{value}</div>'
            f'<div class="mi-overview-hint">{escape(hint)}</div></div></div>'
        )
    st.html(
        '<section class="mi-overview" aria-label="Current session overview">'
        + "".join(content)
        + "</section>"
    )


def render_footer(notice: str) -> None:
    st.html(
        f'<footer class="mi-footer">{icon("verified_user")}<p>{escape(notice)}</p>'
        '<span class="mi-footer-session">Session storage only</span></footer>'
    )
