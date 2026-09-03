"""Business logic bridging the UI to the extraction/classifier/confidence
pipeline - this project's equivalent of the COVID_Detection reference
project's models.py (see docs/architecture.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classifier.predict import predict_candidates
from confidence.composite_score import score_action_item
from pipeline.llm_extractor import DEFAULT_MODEL, extract
from pipeline.schema import CandidateSentence, ConfidenceResult, SummaryOutput, Transcript

SAMPLE_DIR = ROOT / "app" / "sample_meetings"
PROCESSED_ROOT = ROOT / "data" / "processed"
_CONFIDENCE_TO_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def list_sample_meetings() -> list[str]:
    return sorted(p.stem.removesuffix("_transcript") for p in SAMPLE_DIR.glob("*_transcript.json"))


def load_sample(meeting_id: str) -> tuple[Transcript, SummaryOutput, list[CandidateSentence]]:
    transcript = Transcript.model_validate_json((SAMPLE_DIR / f"{meeting_id}_transcript.json").read_text())
    summary = SummaryOutput.model_validate_json((SAMPLE_DIR / f"{meeting_id}_llm.json").read_text())
    candidates = [
        CandidateSentence.model_validate(c)
        for c in json.loads((SAMPLE_DIR / f"{meeting_id}_classifier.json").read_text())
    ]
    return transcript, summary, candidates


def list_live_mode_meetings() -> list[str]:
    """Scoped to the 20 gold meetings, not all 7,124 processed transcripts -
    a full listing would be an unusable dropdown, and these are the ones
    already known to run cleanly through both arms."""
    gold_path = ROOT / "data" / "eval_gold" / "selected_meetings.txt"
    return sorted(m.strip() for m in gold_path.read_text().splitlines() if m.strip())


def load_processed_transcript(meeting_id: str) -> Transcript:
    dataset = meeting_id.split("_", 1)[0]
    return Transcript.model_validate_json((PROCESSED_ROOT / dataset / f"{meeting_id}.json").read_text())


def run_live(transcript: Transcript, model: str = DEFAULT_MODEL) -> tuple[SummaryOutput, list[CandidateSentence]]:
    summary = extract(transcript, model=model)
    candidates = predict_candidates(transcript)
    return summary, candidates


def score_all_action_items(
    summary: SummaryOutput, candidates: list[CandidateSentence], transcript: Transcript
) -> list[ConfidenceResult]:
    candidate_texts = [c.text for c in candidates]
    results = []
    for item in summary.action_items:
        model_confidence = _CONFIDENCE_TO_SCORE[item.llm_confidence]
        results.append(score_action_item(item.evidence_span, transcript, model_confidence, candidate_texts))
    return results
