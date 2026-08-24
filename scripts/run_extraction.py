#!/usr/bin/env python3
"""CLI: extract a summary + action items from one processed transcript.

Usage:
    python3 scripts/run_extraction.py --input data/processed/qmsum/<id>.json \
        [--model llama3.2:3b] [--out data/outputs/llm/<id>.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.llm_extractor import DEFAULT_MODEL, extract
from pipeline.schema import Transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    transcript = Transcript.model_validate_json(args.input.read_text())
    print(f"Extracting {transcript.meeting_id} ({len(transcript.utterances)} utterances) with {args.model}...")

    result = extract(transcript, model=args.model)

    out_path = args.out or Path("data/outputs/llm") / f"{transcript.meeting_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2))

    print(f"Summary: {result.abstractive_summary[:200]}")
    print(f"Key decisions: {len(result.key_decisions)}")
    print(f"Action items: {len(result.action_items)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
