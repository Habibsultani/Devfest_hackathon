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
            '<div class="mi-sidebar-context">'
            f"{icon('fact_check')}<div><strong>Inspection workspace</strong>"
            "<span>Visible-defect screening</span></div></div>"
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
                    '<p class="mi-sidebar-description">Image preparation is available. '
                    "AI analysis needs a trained model.</p>"
                    '<div class="mi-status mi-status--pending">'
                    '<span class="mi-status-dot"></span>Integration pending</div>'
                )
            if config_error:
                with st.expander("Model setup details", icon=":material/settings:"):
                    st.warning(config_error)
        with st.expander("How to inspect", icon=":material/help:"):
            st.markdown(
                "1. Add the batch, instrument ID, and supported class.\n"
                "2. Upload a clear instrument image.\n"
                "3. Review the model score and map.\n"
                "4. Record your decision and export the CSV."
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
        '<div class="mi-topline"><div class="mi-breadcrumb">QUALITY CONTROL '
        f"{icon('chevron_right')}<span>Research workspace</span></div>"
        f'<span class="mi-research-tag">{icon("science")}Research prototype</span></div>'
    )
    st.title("Inspection workspace")
    st.markdown(
        '<p class="mi-page-description">Prepare an image, review the evidence, '
        "and keep a record of your decision.</p>",
        unsafe_allow_html=True,
    )


def render_session_overview(records: list, active: dict | None) -> None:
    pending = int(
        active is not None
        and not any(record.inspection_id == active["inspection_id"] for record in records)
    )
    items = (
        ("Saved inspections", len(records), "fact_check"),
        (
            "AI review flags",
            sum(record.requires_review for record in records),
            "flag",
        ),
        ("Awaiting review", pending, "rate_review"),
    )
    content = []
    for label, value, symbol in items:
        content.append(
            f'<div class="mi-overview-item"><div class="mi-overview-icon">{icon(symbol)}</div>'
            f'<span class="mi-overview-label">{escape(label)}</span>'
            f'<span class="mi-overview-value">{value}</span></div>'
        )
    st.html(
        '<section class="mi-overview" aria-label="Current session overview">'
        + "".join(content)
        + "</section>"
    )


def render_progress(*, details_ready: bool, image_ready: bool, analyzed: bool, saved: bool) -> None:
    """Reflect completed work rather than implying an unavailable model has run."""
    steps = (
        ("Instrument details", details_ready),
        ("Image upload", image_ready),
        ("Analysis", analyzed),
        ("Operator review", saved),
    )
    first_pending = next((index for index, (_, done) in enumerate(steps) if not done), None)
    items = []
    for index, (label, done) in enumerate(steps):
        status = "complete" if done else "current" if index == first_pending else "upcoming"
        current = ' aria-current="step"' if status == "current" else ""
        marker = icon("check") if done else str(index + 1).zfill(2)
        accessible = "Completed" if done else "Current" if status == "current" else "Upcoming"
        items.append(
            f'<li class="mi-progress-step mi-progress--{status}"{current}>'
            f'<span class="mi-progress-marker">{marker}</span>'
            f'<span>{escape(label)}<span class="mi-sr-only">: {accessible}</span></span></li>'
        )
    st.html('<ol class="mi-progress" aria-label="Inspection progress">' + "".join(items) + "</ol>")


def render_footer(notice: str) -> None:
    st.html(
        f'<footer class="mi-footer">{icon("verified_user")}<p>{escape(notice)}</p>'
        '<span class="mi-footer-session">Session storage only</span></footer>'
    )
