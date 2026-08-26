"""Verifies that a model's claimed evidence_span actually appears in the
source transcript. An empty or ungrounded evidence_span is treated as a hard
signal (score 0.0) regardless of the model's own self-reported confidence -
see docs/architecture.md for why this matters (evidence_span was found
empty in early Phase 2 testing before the prompt was fixed).
"""
from __future__ import annotations

from typing import Optional

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


def find_evidence_speaker(evidence_span: str, transcript: Transcript) -> Optional[str]:
    """Finds which utterance an evidence_span most likely came from and
    returns its speaker. Needed because classify_owner_mention's
    "explicit_self" category requires knowing who is speaking - matching
    against the whole concatenated transcript (is_grounded) loses that."""
    span = (evidence_span or "").strip()
    if not span:
        return None

    best_speaker, best_score = None, 0.0
    for utterance in transcript.utterances:
        if not utterance.text:
            continue
        score = fuzz.partial_ratio(span, utterance.text) / 100.0
        if score > best_score:
            best_score, best_speaker = score, utterance.speaker

    return best_speaker if best_score >= FUZZY_THRESHOLD else None
