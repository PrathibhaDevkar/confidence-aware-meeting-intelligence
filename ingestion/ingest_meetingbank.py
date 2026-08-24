"""Normalizes MeetingBank's flat transcript blobs into the canonical Transcript schema.

MeetingBank ships one raw text blob per segment, with no speaker labels or
timestamps (unlike QMSum). We split each blob into sentence-level pseudo-turns
with speaker=None so downstream stages have addressable spans to reference,
while being explicit that speaker-attribution signals will be weaker for this
dataset than for QMSum.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import Transcript, Utterance

RAW_ROOT = Path("data/raw/meetingbank")
OUT_ROOT = Path("data/processed/meetingbank")
SPLIT_FILES = {"train": "train.jsonl", "val": "validation.jsonl", "test": "test.jsonl"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sanitize_id(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", uid).strip("_")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _build_transcript(row: dict, split: str) -> Transcript:
    meeting_id = f"meetingbank_{_sanitize_id(row['uid'])}"
    utterances = [
        Utterance(speaker=None, text=sentence, turn_index=i)
        for i, sentence in enumerate(_split_sentences(row.get("transcript", "")))
    ]
    return Transcript(
        meeting_id=meeting_id,
        source_dataset="meetingbank",
        split=split,
        participants=[],
        utterances=utterances,
        reference_summary=row.get("summary"),
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for split, filename in SPLIT_FILES.items():
        path = RAW_ROOT / filename
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                transcript = _build_transcript(row, split)
                out_path = OUT_ROOT / f"{transcript.meeting_id}.json"
                out_path.write_text(transcript.model_dump_json(indent=2))
                count += 1
    print(f"Wrote {count} transcripts to {OUT_ROOT}")


if __name__ == "__main__":
    main()
