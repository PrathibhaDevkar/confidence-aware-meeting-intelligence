"""Selects the held-out gold evaluation meetings, stratified across QMSum's
three domains (icsi/ami/committee) and MeetingBank, all from the `test` split
so the gold set stays uncontaminated by later prompt-tuning decisions.

Fixed seed -> reproducible selection. Re-running overwrites the same file.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import Transcript

PROCESSED_ROOT = Path("data/processed")
OUT_PATH = Path("data/eval_gold/selected_meetings.txt")
SEED = 42
PER_STRATUM = 5  # 3 QMSum domains + MeetingBank = 4 strata -> ~20 meetings


def _load_test_split(dataset_dir: str) -> list[Transcript]:
    transcripts = []
    for path in sorted((PROCESSED_ROOT / dataset_dir).glob("*.json")):
        t = Transcript.model_validate_json(path.read_text())
        if t.split == "test":
            transcripts.append(t)
    return transcripts


def main() -> None:
    rng = random.Random(SEED)
    qmsum = _load_test_split("qmsum")
    meetingbank = _load_test_split("meetingbank")

    strata = {
        "qmsum_icsi": [t for t in qmsum if t.meeting_id.startswith("qmsum_icsi_")],
        "qmsum_ami": [t for t in qmsum if t.meeting_id.startswith("qmsum_ami_")],
        "qmsum_committee": [t for t in qmsum if t.meeting_id.startswith("qmsum_committee_")],
        "meetingbank": meetingbank,
    }

    selected: list[str] = []
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for stratum, candidates in strata.items():
            k = min(PER_STRATUM, len(candidates))
            picked = rng.sample(candidates, k) if candidates else []
            print(f"{stratum}: {len(candidates)} candidates in test split, selected {k}")
            for t in picked:
                f.write(f"{t.meeting_id}\n")
                selected.append(t.meeting_id)

    print(f"\nSelected {len(selected)} gold meetings -> {OUT_PATH}")


if __name__ == "__main__":
    main()
