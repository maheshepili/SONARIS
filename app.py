"""SONARIS Streamlit interface for batch side-scan sonar analysis."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.inference.batch_processor import process_batch


APP_TITLE = "SONARIS"
NO_SIGNIFICANT_FINDING = "No Significant Finding"


def _clear_previous_session() -> None:
    """Remove only the temporary directory owned by this browser session."""
    previous_directory = st.session_state.get("analysis_directory")
    if previous_directory:
        shutil.rmtree(previous_directory, ignore_errors=True)
    for key in ("analysis_directory", "batch_result", "display_names", "reviews", "review_notice"):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith(("review_decision_", "review_notes_")):
            st.session_state.pop(key)


def _safe_upload_name(name: str, used_names: set[str], used_stems: set[str]) -> str:
    """Make each uploaded file name safe and unique for batch output artifacts."""
    original = Path(name).name
    path = Path(original)
    for index in range(10_000):
        candidate = original if index == 0 else f"upload_{index}_{original}"
        candidate_path = Path(candidate)
        if candidate not in used_names and candidate_path.stem not in used_stems:
            used_names.add(candidate)
            used_stems.add(candidate_path.stem)
            return candidate
    raise RuntimeError(f"Could not create a unique upload name for {name}")


def _save_uploads(uploaded_files: Iterable[Any], session_directory: Path) -> dict[str, str]:
    """Save uploads in a per-analysis temporary directory and retain display names."""
    input_directory = session_directory / "uploads"
    input_directory.mkdir(parents=True)
    display_names: dict[str, str] = {}
    used_names: set[str] = set()
    used_stems: set[str] = set()
    for uploaded_file in uploaded_files:
        saved_name = _safe_upload_name(uploaded_file.name, used_names, used_stems)
        (input_directory / saved_name).write_bytes(uploaded_file.getbuffer())
        display_names[saved_name] = Path(uploaded_file.name).name
    return display_names


def _pipeline_confidence(record: dict[str, Any]) -> float | None:
    detections = record.get("known_object_detection", {}).get("detections", [])
    confidences = [
        detection.get("confidence")
        for detection in detections
        if detection.get("class") == "Pipeline" and detection.get("confidence") is not None
    ]
    return max(confidences, default=None)


def _summary_metrics(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    findings = sum(record.get("finding") != NO_SIGNIFICANT_FINDING for record in records)
    anomalies = sum(record.get("anomaly_result", {}).get("detected", False) for record in records)
    verification = sum(record.get("requires_expert_verification", False) for record in records)
    return findings, anomalies, verification


def _artifact_path(record: dict[str, Any], suffix: str) -> Path:
    return Path(st.session_state["analysis_directory"]) / "results" / f"{Path(record['filename']).stem}_{suffix}.png"


def _finding_badge(finding: str) -> str:
    classes = {
        "Known Pipeline": "finding-pipeline",
        "Known Pipeline + Potential Anomaly": "finding-combined",
        "Potential Anomaly": "finding-anomaly",
        NO_SIGNIFICANT_FINDING: "finding-clear",
    }
    return f'<span class="badge {classes.get(finding, "finding-clear")}">{escape(finding)}</span>'


def _priority_badge(priority: str) -> str:
    return f'<span class="badge priority-{escape(priority.lower())}">{escape(priority)}</span>'


def _verification_badge(required: bool) -> str:
    label = "Expert verification required" if required else "Not required"
    style = "verification-required" if required else "verification-clear"
    return f'<span class="verification {style}">{label}</span>'


def _review_status(record: dict[str, Any]) -> str:
    """Return the human-review state without changing AI-generated evidence."""
    if not record["requires_expert_verification"]:
        return "Not Required"
    review = st.session_state.get("reviews", {}).get(record["filename"])
    return review["status"] if review else "Pending Review"


def _review_badge(status: str) -> str:
    styles = {
        "Verified": "review-verified",
        "Rejected": "review-rejected",
        "Pending Review": "review-pending",
        "Not Required": "review-not-required",
    }
    return f'<span class="review-status {styles[status]}">{status}</span>'


def _render_summary(records: list[dict[str, Any]]) -> None:
    findings, anomalies, verification = _summary_metrics(records)
    st.markdown('<p class="section-eyebrow">Mission overview</p>', unsafe_allow_html=True)
    st.markdown("<h2>Analysis summary</h2>", unsafe_allow_html=True)
    cards = (
        ("Images Processed", len(records), "IMAGES"),
        ("Findings", findings, "ASSESSMENTS"),
        ("Anomalies", anomalies, "RECONSTRUCTION"),
        ("Expert Verification", verification, "REVIEW QUEUE"),
    )
    for column, (label, value, detail) in zip(st.columns(4), cards):
        with column:
            st.markdown(
                f'<div class="metric-card"><span>{label}</span><strong>{value:02d}</strong><small>{detail}</small></div>',
                unsafe_allow_html=True,
            )


def _render_results_table(records: list[dict[str, Any]], display_names: dict[str, str]) -> None:
    rows = []
    for record in records:
        filename = display_names.get(record["filename"], record["filename"])
        rows.append(
            "<tr>"
            f'<td class="filename">{escape(filename)}</td>'
            f"<td>{_finding_badge(record['finding'])}</td>"
            f"<td>{_priority_badge(record['priority'])}</td>"
            f"<td>{_review_badge(_review_status(record))}</td>"
            "</tr>"
        )
    st.markdown(
        '<div class="results-table-wrap"><table class="results-table">'
        "<thead><tr><th>Image</th><th>Finding</th><th>Priority</th><th>Review status</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_results(records: list[dict[str, Any]], display_names: dict[str, str]) -> None:
    st.markdown('<p class="section-eyebrow">Operational findings</p>', unsafe_allow_html=True)
    st.markdown("<h2>Results</h2>", unsafe_allow_html=True)
    _render_results_table(records, display_names)

    selected_filename = st.selectbox(
        "Select result for analyst review",
        options=[record["filename"] for record in records],
        format_func=lambda filename: display_names.get(filename, filename),
    )
    selected = next(record for record in records if record["filename"] == selected_filename)
    anomaly = selected.get("anomaly_result", {})
    candidate_regions = selected.get("candidate_regions", anomaly.get("candidate_regions", []))
    confidence = _pipeline_confidence(selected)

    selected_display_name = display_names.get(selected_filename, selected_filename)
    st.markdown(
        '<div class="analyst-header">'
        '<div><span class="section-eyebrow">Analyst review panel</span>'
        f"<h3>{escape(selected_display_name)}</h3></div>"
        f'<div class="analyst-badges"><span class="ai-label">AI-generated assessment</span>{_finding_badge(selected["finding"])}'
        f'{_priority_badge(selected["priority"])}{_verification_badge(selected["requires_expert_verification"])}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    confidence_column, anomaly_column, regions_column, verification_column = st.columns(4)
    anomaly_score = anomaly.get("max_reconstruction_error")
    evidence_cards = (
        ("Pipeline confidence", "—" if confidence is None else f"{confidence:.1%}"),
        ("Anomaly score", "—" if anomaly_score is None else f"{anomaly_score:.5f}"),
        ("Candidate regions", str(len(candidate_regions))),
        ("Verification", "Required" if selected["requires_expert_verification"] else "Not required"),
    )
    for column, (label, value) in zip(
        (confidence_column, anomaly_column, regions_column, verification_column), evidence_cards
    ):
        with column:
            st.markdown(
                f'<div class="evidence-card"><span>{label}</span><strong>{value}</strong></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="reason-panel"><span>Assessment rationale</span>'
        f"<p>{escape(selected.get('reason', 'No explanation was returned by the inference pipeline.'))}</p></div>",
        unsafe_allow_html=True,
    )
    if candidate_regions:
        with st.expander("Candidate region details"):
            st.json(candidate_regions)

    _render_analyst_decision(selected)

    detection_image = _artifact_path(selected, "detections")
    heatmap_image = _artifact_path(selected, "anomaly_heatmap")
    image_columns = st.columns(2)
    with image_columns[0]:
        st.markdown('<p class="visual-title">Pipeline detections</p>', unsafe_allow_html=True)
        if detection_image.is_file():
            st.image(str(detection_image), use_container_width=True)
        else:
            st.info("No detection image was generated for this result.")
    with image_columns[1]:
        st.markdown('<p class="visual-title">Anomaly heatmap</p>', unsafe_allow_html=True)
        if heatmap_image.is_file():
            st.image(str(heatmap_image), use_container_width=True)
        else:
            st.info("No anomaly heatmap was generated for this result.")


def _render_analyst_decision(selected: dict[str, Any]) -> None:
    """Render human confirmation controls separately from immutable AI evidence."""
    status = _review_status(selected)
    st.markdown(
        '<div class="decision-header"><div><span class="section-eyebrow">Analyst decision</span>'
        '<h3>Human verification</h3></div>'
        f'<div>{_review_badge(status)}</div></div>',
        unsafe_allow_html=True,
    )
    if status == "Not Required":
        st.markdown(
            '<div class="decision-panel"><p>This result does not require expert verification. '
            'The AI assessment and evidence remain available above.</p></div>',
            unsafe_allow_html=True,
        )
        return

    filename = selected["filename"]
    existing_review = st.session_state.get("reviews", {}).get(filename, {})
    decision_key = f"review_decision_{filename}"
    notes_key = f"review_notes_{filename}"
    if decision_key not in st.session_state:
        st.session_state[decision_key] = (
            "Reject Finding" if existing_review.get("status") == "Rejected" else "Confirm Finding"
        )
    if notes_key not in st.session_state:
        st.session_state[notes_key] = existing_review.get("notes", "")

    with st.container(border=True):
        decision = st.radio(
            "Decision",
            ("Confirm Finding", "Reject Finding"),
            horizontal=True,
            key=decision_key,
        )
        notes = st.text_area(
            "Analyst notes",
            placeholder="Record evidence reviewed, operational context, or the reason for the decision.",
            key=notes_key,
        )
        if st.button("Save Review", type="primary", key=f"save_review_{filename}"):
            st.session_state.setdefault("reviews", {})[filename] = {
                "status": "Verified" if decision == "Confirm Finding" else "Rejected",
                "notes": notes,
            }
            st.session_state["review_notice"] = "Analyst review saved. The result table has been updated."
            st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """<style>
        :root {
          --bg: #07121d;
          --panel: #0d1d2a;
          --panel-soft: #112635;
          --line: #244356;
          --text: #eaf4f7;
          --muted: #9fb7c4;
          --cyan: #39c5dc;
          --cyan-soft: #102f40;
          --green: #67d5a6;
          --amber: #f4c66b;
          --red: #ed8796;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
          background: var(--bg); color: var(--text);
        }
        [data-testid="stHeader"] {background: transparent;}
        .block-container {max-width: 1320px; padding: 1.4rem 2.2rem 3rem;}
        [data-testid="stSidebar"] {background: #081923; border-right: 1px solid var(--line);}
        [data-testid="stSidebar"] > div:first-child {padding: 1.5rem 1rem;}
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {color: var(--text);}
        h2 {font-size: 1.35rem; letter-spacing: -0.01em; margin: 0.05rem 0 1rem;}
        h3 {font-size: 1.25rem; margin: 0.25rem 0 0;}
        .topbar {
          display: flex; justify-content: space-between; align-items: center;
          padding: 0.35rem 0 1.45rem; border-bottom: 1px solid var(--line); margin-bottom: 1.8rem;
        }
        .brand {font-size: 1.75rem; font-weight: 750; letter-spacing: 0.17em; color: var(--text);}
        .subtitle {color: var(--muted); font-size: 0.91rem; margin-top: 0.25rem;}
        .system-ready {
          color: var(--green); border: 1px solid #27644f; background: #0e2d29;
          padding: 0.42rem 0.7rem; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
          letter-spacing: 0.08em;
        }
        .status-dot {font-size: 0.7rem; padding-right: 0.35rem;}
        .sidebar-brand {font-size: 1.25rem; font-weight: 750; letter-spacing: 0.15em; color: var(--text); margin-bottom: 0.25rem;}
        .sidebar-copy {color: var(--muted); font-size: 0.78rem; line-height: 1.4; margin-bottom: 2rem;}
        .sidebar-label, .section-eyebrow {
          color: var(--cyan); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em;
          text-transform: uppercase;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {gap: 0.35rem;}
        [data-testid="stSidebar"] label {
          background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 0.35rem 0.45rem;
        }
        [data-testid="stSidebar"] label:has(input:checked) {background: var(--cyan-soft); border-color: #23778b;}
        [data-testid="stSidebar"] label p {font-size: 0.9rem !important; color: var(--text) !important;}
        .metric-card, .evidence-card {
          min-height: 128px; background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; padding: 1rem 1.05rem; box-shadow: 0 9px 20px rgba(0, 0, 0, 0.15);
        }
        .metric-card span, .evidence-card span, .reason-panel span {
          display: block; color: var(--muted); font-size: 0.72rem; font-weight: 700;
          letter-spacing: 0.08em; text-transform: uppercase;
        }
        .metric-card strong {display: block; color: var(--text); font-size: 2.25rem; line-height: 1.15; margin: 0.45rem 0 0.15rem;}
        .metric-card small {color: var(--cyan); font-size: 0.65rem; font-weight: 700; letter-spacing: 0.11em;}
        .evidence-card {min-height: 92px; padding: 0.85rem 0.95rem;}
        .evidence-card strong {display: block; color: var(--text); font-size: 1.1rem; margin-top: 0.45rem; overflow-wrap: anywhere;}
        .results-table-wrap {overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; margin: 0.6rem 0 1.3rem;}
        .results-table {width: 100%; border-collapse: collapse; min-width: 760px;}
        .results-table th {color: var(--muted); background: #0a1823; font-size: 0.69rem; letter-spacing: 0.1em; text-align: left; text-transform: uppercase;}
        .results-table th, .results-table td {padding: 0.85rem 1rem; border-bottom: 1px solid #1a3646;}
        .results-table tr:last-child td {border-bottom: 0;}
        .results-table td {color: var(--text); font-size: 0.88rem;}
        .results-table .filename {font-family: "SFMono-Regular", Consolas, monospace; color: #cce9ef;}
        .badge, .verification {
          display: inline-block; border-radius: 999px; padding: 0.28rem 0.55rem;
          font-size: 0.7rem; font-weight: 700; letter-spacing: 0.025em; white-space: nowrap;
        }
        .finding-pipeline {background: #123a49; border: 1px solid #237b91; color: #8be9f7;}
        .finding-combined {background: #3b2f19; border: 1px solid #9a7829; color: #ffd985;}
        .finding-anomaly {background: #3d2029; border: 1px solid #98505e; color: #ffadb9;}
        .finding-clear {background: #17312e; border: 1px solid #377566; color: #9ee9cf;}
        .priority-low {background: #17312e; border: 1px solid #377566; color: #9ee9cf;}
        .priority-medium {background: #3b2f19; border: 1px solid #9a7829; color: #ffd985;}
        .priority-high {background: #3d2029; border: 1px solid #98505e; color: #ffadb9;}
        .verification-required {background: #3d2029; border: 1px solid #98505e; color: #ffadb9;}
        .verification-clear {background: #15302d; border: 1px solid #377566; color: #9ee9cf;}
        .review-status {
          display: inline-block; border-radius: 999px; padding: 0.28rem 0.55rem;
          font-size: 0.7rem; font-weight: 700; letter-spacing: 0.025em; white-space: nowrap;
        }
        .review-verified {background: #15302d; border: 1px solid #377566; color: #9ee9cf;}
        .review-rejected {background: #3d2029; border: 1px solid #98505e; color: #ffadb9;}
        .review-pending {background: #3b2f19; border: 1px solid #9a7829; color: #ffd985;}
        .review-not-required {background: #162936; border: 1px solid #3d5b69; color: #b9ced6;}
        .analyst-header {
          display: flex; align-items: center; justify-content: space-between; gap: 1rem;
          margin: 1.8rem 0 0.8rem; padding: 1rem 1.1rem; background: var(--panel);
          border: 1px solid var(--line); border-left: 3px solid var(--cyan); border-radius: 9px;
        }
        .analyst-badges {display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.45rem;}
        .ai-label {align-self: center; color: var(--muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;}
        .reason-panel {background: #0b1a26; border: 1px solid var(--line); border-radius: 8px; padding: 0.9rem 1rem; margin: 0.9rem 0 1rem;}
        .reason-panel p {color: #d3e2e7; margin: 0.4rem 0 0; line-height: 1.55;}
        .decision-header {
          display: flex; justify-content: space-between; align-items: center; gap: 1rem;
          margin: 1.5rem 0 0.65rem; padding-top: 1.1rem; border-top: 1px solid var(--line);
        }
        .decision-panel {background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 0.85rem 1rem;}
        .decision-panel p {color: var(--muted); margin: 0;}
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--panel); border-color: var(--line); border-radius: 9px;
        }
        [data-testid="stTextArea"] textarea {
          background: #091722; color: var(--text); border-color: var(--line);
        }
        [data-testid="stRadio"] label p {color: var(--text) !important;}
        .visual-title {color: var(--text); font-size: 0.88rem; font-weight: 700; margin: 0.7rem 0 0.45rem;}
        [data-testid="stFileUploader"] {background: var(--panel); border: 1px dashed #34748a; border-radius: 10px; padding: 1rem;}
        [data-testid="stFileUploader"] section {background: transparent;}
        [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {color: var(--muted) !important;}
        .stButton > button {
          background: #0c7892; color: #f4feff; border: 1px solid #3ac3dc; border-radius: 7px;
          font-weight: 750; letter-spacing: 0.06em; text-transform: uppercase; min-height: 2.7rem;
        }
        .stButton > button:hover {background: #1090aa; border-color: #85eafa; color: #ffffff;}
        .stButton > button:disabled {background: #1a3541; border-color: #31515d; color: #7f98a3;}
        [data-testid="stSelectbox"] > div > div, [data-testid="stSelectbox"] input {
          background: var(--panel); color: var(--text); border-color: var(--line);
        }
        [data-testid="stExpander"] {background: var(--panel); border: 1px solid var(--line); border-radius: 8px;}
        [data-testid="stExpander"] summary, [data-testid="stExpander"] p {color: var(--text) !important;}
        [data-testid="stImage"] img {border: 1px solid var(--line); border-radius: 8px;}
        [data-testid="stAlert"] {border-radius: 8px;}
        hr {border-color: var(--line) !important;}
        @media (max-width: 820px) {
          .block-container {padding: 1rem;}
          .topbar, .analyst-header {align-items: flex-start; flex-direction: column;}
          .analyst-badges {justify-content: flex-start;}
        }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">SONARIS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-copy">Underwater intelligence and sonar operations workspace.</div>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-label">Workspace</p>', unsafe_allow_html=True)
        st.radio("Navigation", ("Analysis", "Review", "Reports"), label_visibility="collapsed")
        st.divider()
        st.markdown('<p class="sidebar-label">System</p>', unsafe_allow_html=True)
        st.caption("Inference services use the configured project models and thresholds.")

    st.markdown(
        '<div class="topbar"><div><div class="brand">SONARIS</div>'
        '<div class="subtitle">Underwater Intelligence &amp; Side-Scan Sonar Analysis</div></div>'
        '<div class="system-ready"><span class="status-dot">●</span>SYSTEM READY</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-eyebrow">SSS data ingestion</p>', unsafe_allow_html=True)
    st.markdown("<h2>Upload sonar imagery</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop side-scan sonar imagery here or browse files",
        type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Analyze Images", type="primary", disabled=not uploaded_files, use_container_width=True):
        _clear_previous_session()
        session_directory = Path(tempfile.mkdtemp(prefix="sonaris_streamlit_"))
        try:
            with st.spinner("Running the SONARIS detection and anomaly pipeline..."):
                display_names = _save_uploads(uploaded_files, session_directory)
                batch_result = process_batch(
                    session_directory / "uploads", session_directory / "results"
                )
            st.session_state["analysis_directory"] = str(session_directory)
            st.session_state["batch_result"] = batch_result
            st.session_state["display_names"] = display_names
            st.session_state["reviews"] = {}
            st.success(f"Analysis complete: {batch_result['summary']['total_images']} image(s) processed.")
        except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError, OSError) as error:
            shutil.rmtree(session_directory, ignore_errors=True)
            st.error(f"Analysis could not be completed: {error}")

    batch_result = st.session_state.get("batch_result")
    if batch_result:
        st.divider()
        if review_notice := st.session_state.pop("review_notice", None):
            st.success(review_notice)
        records = batch_result.get("images", [])
        _render_summary(records)
        if records:
            st.divider()
            _render_results(records, st.session_state.get("display_names", {}))
        else:
            st.info("No supported images were available to analyze.")


if __name__ == "__main__":
    main()
