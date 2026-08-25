"""Exports a CSV combining both comparison arms' outputs across the gold
meetings, ready for manual labeling in Excel/Numbers/Google Sheets. See
docs/labeling_guidelines.md for how to fill in the three judgment columns.

Usage: python3 eval/export_labeling_sheet.py
(run after eval/run_batch_extraction.py has produced data/outputs/{llm,classifier}/)

QMSum meetings turned out much longer than expected, so the raw combined
pool is ~844 rows - well beyond the ~150-250 item labeling budget. This
does a stratified-by-meeting random subsample down to TARGET_TOTAL, with a
floor per meeting so every meeting still gets some representation.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import CandidateSentence, SummaryOutput

GOLD_IDS_PATH = Path("data/eval_gold/selected_meetings.txt")
LLM_DIR = Path("data/outputs/llm")
CLF_DIR = Path("data/outputs/classifier")
OUT_PATH = Path("data/eval_gold/labeling_sheet.csv")

FIELDS = [
    "row_id", "meeting_id", "arm", "item_index", "text_shown", "owner", "deadline",
    "evidence_span", "correct_task", "correct_owner", "correct_deadline", "notes",
]

TARGET_TOTAL = 225
MIN_PER_MEETING = 3
SEED = 42


def _stratified_sample(rows: list[dict], target_total: int, min_per_meeting: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_meeting[row["meeting_id"]].append(row)

    if target_total >= len(rows):
        return rows

    quotas = {
        mid: max(min_per_meeting, round(target_total * len(pool) / len(rows)))
        for mid, pool in by_meeting.items()
    }
    quotas = {mid: min(q, len(by_meeting[mid])) for mid, q in quotas.items()}

    sampled = []
    for mid, pool in by_meeting.items():
        sampled.extend(rng.sample(pool, quotas[mid]))

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    meeting_ids = [m.strip() for m in GOLD_IDS_PATH.read_text().splitlines() if m.strip()]
    rows = []
    row_id = 0

    for meeting_id in meeting_ids:
        llm_path = LLM_DIR / f"{meeting_id}.json"
        if llm_path.exists():
            summary = SummaryOutput.model_validate_json(llm_path.read_text())
            for i, item in enumerate(summary.action_items):
                row_id += 1
                rows.append(
                    {
                        "row_id": row_id,
                        "meeting_id": meeting_id,
                        "arm": "llm",
                        "item_index": i,
                        "text_shown": item.task,
                        "owner": item.owner or "",
                        "deadline": item.deadline or "",
                        "evidence_span": item.evidence_span,
                        "correct_task": "",
                        "correct_owner": "",
                        "correct_deadline": "",
                        "notes": "",
                    }
                )
        else:
            print(f"warning: no LLM output for {meeting_id}, skipping that arm")

        clf_path = CLF_DIR / f"{meeting_id}.json"
        if clf_path.exists():
            candidates = [
                CandidateSentence.model_validate(c) for c in json.loads(clf_path.read_text())
            ]
            for i, cand in enumerate(candidates):
                row_id += 1
                rows.append(
                    {
                        "row_id": row_id,
                        "meeting_id": meeting_id,
                        "arm": "classifier",
                        "item_index": i,
                        "text_shown": cand.text,
                        "owner": cand.owner or "",
                        "deadline": cand.deadline or "",
                        "evidence_span": cand.text,
                        "correct_task": "",
                        "correct_owner": "",
                        "correct_deadline": "",
                        "notes": "",
                    }
                )
        else:
            print(f"warning: no classifier output for {meeting_id}, skipping that arm")

    print(f"Raw pool: {len(rows)} rows across {len(meeting_ids)} meetings")
    sampled = _stratified_sample(rows, TARGET_TOTAL, MIN_PER_MEETING, SEED)
    for i, row in enumerate(sampled, 1):
        row["row_id"] = i

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Subsampled to {len(sampled)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
