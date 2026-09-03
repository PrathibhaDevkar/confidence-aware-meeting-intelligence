"""Confidence-Aware Meeting Intelligence - Streamlit demo.

Mirrors the reference project's shape (University_Docs/.../COVID_Detection/
app.py, see docs/architecture.md): single-file UI + separate business-logic
module (pipeline_runner.py), @st.cache_resource for one-time loads, sidebar
controls, st.tabs, two-column layouts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

from app.pipeline_runner import (
    list_live_mode_meetings,
    list_sample_meetings,
    load_processed_transcript,
    load_sample,
    run_live,
    score_all_action_items,
)
from pipeline.llm_extractor import DEFAULT_MODEL

st.set_page_config(
    page_title="Confidence-Aware Meeting Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
TIER_ORDER = {"High": 3, "Medium": 2, "Low": 1}


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama() -> bool:
    try:
        return requests.get("http://localhost:11434", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data(show_spinner=False)
def cached_load_sample(meeting_id: str):
    return load_sample(meeting_id)


@st.cache_data(show_spinner="Running extraction pipeline live (Ollama)... this can take a minute.")
def cached_run_live(meeting_id: str, model: str):
    transcript = load_processed_transcript(meeting_id)
    summary, candidates = run_live(transcript, model=model)
    return transcript, summary, candidates


st.title("Confidence-Aware Meeting Intelligence")
st.caption(
    "Meeting summarization + action-item extraction that tells you *how sure* it is - "
    "not just what it thinks."
)

ollama_up = check_ollama()

with st.sidebar:
    st.header("Settings")

    mode_options = ["Sample meetings (offline)"]
    if ollama_up:
        mode_options.append("Live extraction")
    mode = st.radio("Mode", mode_options)
    if not ollama_up:
        st.caption(
            "Ollama not detected at localhost:11434 - live mode needs `ollama serve` "
            "running locally. Showing bundled sample meetings only."
        )

    if mode == "Sample meetings (offline)":
        meeting_id = st.selectbox("Meeting", list_sample_meetings())
    else:
        meeting_id = st.selectbox("Meeting", list_live_mode_meetings())
        st.caption("Scoped to the 20 gold-evaluation meetings.")

    confidence_filter = st.select_slider(
        "Minimum confidence to show", options=["Low", "Medium", "High"], value="Low"
    )

if mode == "Sample meetings (offline)":
    transcript, summary, candidates = cached_load_sample(meeting_id)
else:
    transcript, summary, candidates = cached_run_live(meeting_id, DEFAULT_MODEL)

with st.spinner("Scoring confidence..."):
    confidences = score_all_action_items(summary, candidates, transcript)

tab_summary, tab_actions, tab_compare, tab_about = st.tabs(
    ["Summary", "Action Items", "Model Comparison", "About"]
)

with tab_summary:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transcript")
        transcript_text = "\n".join(f"{u.speaker or 'UNKNOWN'}: {u.text}" for u in transcript.utterances)
        st.text_area("transcript", transcript_text, height=450, label_visibility="collapsed")
    with col2:
        st.subheader("Summary")
        st.info(summary.abstractive_summary)
        st.subheader("Key Decisions")
        if summary.key_decisions:
            for d in summary.key_decisions:
                st.markdown(f"- {d}")
        else:
            st.caption("None extracted.")

with tab_actions:
    st.subheader(f"Action Items ({len(summary.action_items)} extracted)")
    shown = 0
    for item, conf in zip(summary.action_items, confidences):
        if TIER_ORDER[conf.tier] < TIER_ORDER[confidence_filter]:
            continue
        shown += 1
        with st.expander(f"{TIER_BADGE[conf.tier]}  ({conf.score:.0%})  —  {item.task}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Owner:** {item.owner or '_unclear_'}")
            c2.markdown(f"**Deadline:** {item.deadline or '_none stated_'}")
            st.markdown(f"**Evidence:** “{item.evidence_span}”")
            st.caption(f"Why this score: {conf.rationale}")

    if shown == 0:
        st.warning(f"No action items meet the '{confidence_filter}' confidence threshold.")
    elif shown < len(summary.action_items):
        st.caption(f"{len(summary.action_items) - shown} lower-confidence item(s) hidden by the filter above.")

with tab_compare:
    st.subheader("LLM vs. classifier comparison")
    c1, c2 = st.columns(2)
    c1.metric("LLM arm: action items", len(summary.action_items))
    c2.metric("Classifier arm: candidates flagged", len(candidates))

    st.markdown("---")
    st.markdown(
        "**Does the confidence layer actually work?** From the held-out evaluation "
        "(`reports/evaluation_summary.md`) - never seen during calibration training:"
    )
    st.markdown(
        "| Confidence tier | Precision |\n"
        "|---|---|\n"
        "| Top third | **58%** |\n"
        "| Middle third | 32% |\n"
        "| Bottom third | 16% |\n"
        "| No filtering (baseline) | 35% |"
    )
    st.caption("Top-third confidence nearly doubles precision over no filtering at all - on genuinely unseen data.")

with tab_about:
    about_path = ROOT / "docs" / "architecture.md"
    if about_path.exists():
        st.markdown(about_path.read_text())
