"""Point-and-click gold-set labeling tool - no CSV editing required.

Run with: streamlit run scripts/labeling_app.py
Saves to data/eval_gold/labeling_sheet.csv after every row, so progress is
never lost - close the browser any time and resume later, it picks up at
the first unlabeled row.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from pipeline.schema import Transcript

CSV_PATH = ROOT / "data" / "eval_gold" / "labeling_sheet.csv"
PROCESSED_ROOT = ROOT / "data" / "processed"

st.set_page_config(page_title="Gold Set Labeling", layout="centered")


@st.cache_data
def _load_full_transcript_text(meeting_id: str) -> str:
    dataset = meeting_id.split("_", 1)[0]
    path = PROCESSED_ROOT / dataset / f"{meeting_id}.json"
    if not path.exists():
        return "(transcript file not found)"
    transcript = Transcript.model_validate_json(path.read_text())
    lines = []
    for u in transcript.utterances:
        speaker = u.speaker or "UNKNOWN"
        lines.append(f"{speaker}: {u.text}")
    return "\n".join(lines)


def _load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, dtype=str).fillna("")


if "df" not in st.session_state:
    st.session_state.df = _load_data()
    unlabeled = st.session_state.df.index[st.session_state.df["correct_task"] == ""]
    st.session_state.idx = int(unlabeled[0]) if len(unlabeled) else len(st.session_state.df)

df = st.session_state.df
total = len(df)
idx = st.session_state.idx

st.title("Gold Set Labeling")
labeled_count = int((df["correct_task"] != "").sum())
st.progress(labeled_count / total if total else 0)
st.caption(f"{labeled_count} / {total} labeled")

nav_prev, _, nav_next = st.columns([1, 3, 1])
if nav_prev.button("← Previous", disabled=idx <= 0):
    st.session_state.idx = max(0, idx - 1)
    st.rerun()

if idx >= total:
    st.success("All rows labeled! Let Claude know you're done.")
    st.stop()

row = df.iloc[idx]

st.markdown(f"**Meeting:** `{row['meeting_id']}`  |  **Arm:** `{row['arm']}`  |  **Row {idx + 1} of {total}**")

st.markdown("### Claimed action item")
st.info(row["text_shown"])

col1, col2 = st.columns(2)
col1.markdown(f"**Owner claimed:** {row['owner'] or '*(none)*'}")
col2.markdown(f"**Deadline claimed:** {row['deadline'] or '*(none)*'}")

st.markdown("### Supporting evidence from transcript")
st.warning(row["evidence_span"] or "*(empty - not grounded)*")

with st.expander("Show full meeting transcript for context"):
    st.text(_load_full_transcript_text(row["meeting_id"]))

st.markdown("---")
st.markdown("**1. Is this a genuine action item, described accurately?**")
correct_task = st.radio(
    "correct_task", ["Yes", "No"], index=None, key=f"task_{idx}",
    label_visibility="collapsed", horizontal=True,
)

correct_owner = correct_deadline = None
if correct_task == "No":
    st.caption("Owner/deadline will be auto-marked No - there's no real task for them to be correct about.")
elif correct_task == "Yes":
    st.markdown("**2. Is the claimed owner correct?** (blank owner is correct if the transcript genuinely doesn't say)")
    correct_owner = st.radio(
        "correct_owner", ["Yes", "No"], index=None, key=f"owner_{idx}",
        label_visibility="collapsed", horizontal=True,
    )
    st.markdown("**3. Is the claimed deadline correct?** (blank deadline is correct if none was stated)")
    correct_deadline = st.radio(
        "correct_deadline", ["Yes", "No"], index=None, key=f"deadline_{idx}",
        label_visibility="collapsed", horizontal=True,
    )

notes = st.text_input("Notes (optional)", key=f"notes_{idx}", value=row["notes"])

can_save = correct_task == "No" or (correct_task == "Yes" and correct_owner is not None and correct_deadline is not None)

if st.button("Save & Next →", disabled=not can_save, type="primary", use_container_width=True):
    st.session_state.df.at[idx, "correct_task"] = "y" if correct_task == "Yes" else "n"
    st.session_state.df.at[idx, "correct_owner"] = "n" if correct_task == "No" else ("y" if correct_owner == "Yes" else "n")
    st.session_state.df.at[idx, "correct_deadline"] = "n" if correct_task == "No" else ("y" if correct_deadline == "Yes" else "n")
    st.session_state.df.at[idx, "notes"] = notes
    st.session_state.df.to_csv(CSV_PATH, index=False)
    st.session_state.idx += 1
    st.rerun()
