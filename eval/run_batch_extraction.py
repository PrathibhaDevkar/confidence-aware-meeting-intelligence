#!/usr/bin/env python3
"""Runs both comparison arms (local-LLM extraction + classifier prediction)
across a list of meeting_ids, writing results to data/outputs/{llm,classifier}/.

No Batches API here (that was a Claude-API-specific optimization no longer
applicable now that extraction runs against a local Ollama model) - this is
a plain sequential loop, which is fine at gold-set scale (20 meetings).

Usage:
    python3 eval/run_batch_extraction.py --meeting-ids data/eval_gold/selected_meetings.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier.predict import predict_candidates
from pipeline.llm_extractor import DEFAULT_MODEL, extract
from pipeline.schema import Transcript

PROCESSED_ROOT = Path("data/processed")


def _find_transcript_path(meeting_id: str) -> Path:
    dataset = meeting_id.split("_", 1)[0]  # "qmsum" | "meetingbank"
    path = PROCESSED_ROOT / dataset / f"{meeting_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No processed transcript for {meeting_id} at {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-ids", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    meeting_ids = [m.strip() for m in args.meeting_ids.read_text().splitlines() if m.strip()]
    llm_out_dir = Path("data/outputs/llm")
    clf_out_dir = Path("data/outputs/classifier")
    llm_out_dir.mkdir(parents=True, exist_ok=True)
    clf_out_dir.mkdir(parents=True, exist_ok=True)

    for i, meeting_id in enumerate(meeting_ids, 1):
        print(f"[{i}/{len(meeting_ids)}] {meeting_id}")
        transcript = Transcript.model_validate_json(_find_transcript_path(meeting_id).read_text())

        llm_path = llm_out_dir / f"{meeting_id}.json"
        if args.skip_existing and llm_path.exists():
            print("  llm: skipped (exists)")
        else:
            start = time.time()
            summary = extract(transcript, model=args.model)
            llm_path.write_text(summary.model_dump_json(indent=2))
            print(f"  llm: {len(summary.action_items)} action items ({time.time() - start:.1f}s)")

        clf_path = clf_out_dir / f"{meeting_id}.json"
        if args.skip_existing and clf_path.exists():
            print("  classifier: skipped (exists)")
        else:
            candidates = predict_candidates(transcript)
            clf_path.write_text(
                "[" + ",\n".join(c.model_dump_json() for c in candidates) + "]"
            )
            print(f"  classifier: {len(candidates)} candidates")

    print(f"\nDone: {len(meeting_ids)} meetings processed.")


if __name__ == "__main__":
    main()
