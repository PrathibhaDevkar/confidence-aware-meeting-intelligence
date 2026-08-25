"""Verifies that a model's claimed evidence_span actually appears in the
source transcript. An empty or ungrounded evidence_span is treated as a hard
signal (score 0.0) regardless of the model's own self-reported confidence -
see docs/architecture.md for why this matters (evidence_span was found
empty in early Phase 2 testing before the prompt was fixed).
"""
from __future__ import annotations

from rapidfuzz import fuzz

from pipeline.schema import Transcript

FUZZY_THRESHOLD = 0.85


def is_grounded(
    evidence_span: str, transcript: Transcript, fuzzy_threshold: float = FUZZY_THRESHOLD
) -> tuple[bool, float]:
    """Returns (is_grounded, match_score in [0, 1])."""
    span = (evidence_span or "").strip()
    if not span:
        return False, 0.0

    full_text = " ".join(u.text for u in transcript.utterances)
    if span in full_text:
        return True, 1.0

    score = fuzz.partial_ratio(span, full_text) / 100.0
    return score >= fuzzy_threshold, score
