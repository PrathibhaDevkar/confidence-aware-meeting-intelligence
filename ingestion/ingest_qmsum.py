"""Normalizes QMSum's per-meeting JSON files into the canonical Transcript schema.

QMSum organizes meetings by domain: data/Academic (ICSI), data/Product (AMI),
data/Committee (parliamentary), and data/ALL (all three combined). We ingest
from data/ALL so every meeting is captured exactly once, and infer the domain
from which domain-specific folder also contains the same meeting_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import Transcript, Utterance

RAW_ROOT = Path("data/raw/qmsum/QMSum-main/data")
ALL_ROOT = RAW_ROOT / "ALL"
OUT_ROOT = Path("data/processed/qmsum")
SPLITS = ["train", "val", "test"]
DOMAIN_FOLDERS = {"Academic": "icsi", "Product": "ami", "Committee": "committee"}


def _domain_for_meeting(meeting_id: str) -> str:
    for folder, domain in DOMAIN_FOLDERS.items():
        for split in SPLITS:
            if (RAW_ROOT / folder / split / f"{meeting_id}.json").exists():
                return domain
    return "unknown"


def _build_transcript(path: Path, split: str) -> Transcript:
    raw = json.loads(path.read_text())
    meeting_id = path.stem

    utterances = []
    speakers: set[str] = set()
    for i, turn in enumerate(raw.get("meeting_transcripts", [])):
        speaker = turn.get("speaker")
        if speaker:
            speakers.add(speaker)
        utterances.append(
            Utterance(speaker=speaker, text=turn.get("content", ""), turn_index=i)
        )

    reference_summary = None
    general_queries = raw.get("general_query_list") or []
    if general_queries:
        reference_summary = general_queries[0].get("answer")

    domain = _domain_for_meeting(meeting_id)
    return Transcript(
        meeting_id=f"qmsum_{domain}_{meeting_id}",
        source_dataset="qmsum",
        split=split,
        participants=sorted(speakers),
        utterances=utterances,
        reference_summary=reference_summary,
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for split in SPLITS:
        split_dir = ALL_ROOT / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.json")):
            transcript = _build_transcript(path, split)
            out_path = OUT_ROOT / f"{transcript.meeting_id}.json"
            out_path.write_text(transcript.model_dump_json(indent=2))
            count += 1
    print(f"Wrote {count} transcripts to {OUT_ROOT}")


if __name__ == "__main__":
    main()
