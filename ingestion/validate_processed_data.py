"""Sanity-checks every processed transcript against the canonical schema and
reports basic corpus statistics (utterance counts, speaker counts, transcript
length distribution) per source dataset.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import Transcript

PROCESSED_ROOT = Path("data/processed")


def _summarize(label: str, transcripts: list[Transcript]) -> None:
    if not transcripts:
        print(f"{label}: no transcripts found")
        return
    utterance_counts = [len(t.utterances) for t in transcripts]
    char_counts = [sum(len(u.text) for u in t.utterances) for t in transcripts]
    speaker_counts = [len(t.participants) for t in transcripts]
    with_summary = sum(1 for t in transcripts if t.reference_summary)

    print(f"\n{label}: {len(transcripts)} transcripts")
    print(f"  with reference_summary: {with_summary}/{len(transcripts)}")
    print(
        "  utterances/meeting: min={} median={} max={}".format(
            min(utterance_counts), statistics.median(utterance_counts), max(utterance_counts)
        )
    )
    print(
        "  chars/meeting: min={} median={} max={}".format(
            min(char_counts), statistics.median(char_counts), max(char_counts)
        )
    )
    print(
        "  speakers/meeting: min={} median={} max={}".format(
            min(speaker_counts), statistics.median(speaker_counts), max(speaker_counts)
        )
    )


def main() -> None:
    errors = []
    by_dataset: dict[str, list[Transcript]] = {}

    for path in sorted(PROCESSED_ROOT.glob("*/*.json")):
        try:
            transcript = Transcript.model_validate_json(path.read_text())
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"{path}: {exc}")
            continue
        by_dataset.setdefault(transcript.source_dataset, []).append(transcript)

    for dataset, transcripts in sorted(by_dataset.items()):
        _summarize(dataset, transcripts)

    print(f"\nTotal valid transcripts: {sum(len(v) for v in by_dataset.values())}")
    if errors:
        print(f"\n{len(errors)} file(s) FAILED validation:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("All processed files passed schema validation.")


if __name__ == "__main__":
    main()
