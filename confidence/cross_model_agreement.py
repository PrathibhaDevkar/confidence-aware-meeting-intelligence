"""Aligns each LLM-extracted action item against the classifier's flagged
candidate sentences. Agreement between the two independently-built arms
raises confidence; disagreement is surfaced as a signal rather than hidden -
this is what ties the two required comparison arms into one system instead
of two disconnected deliverables.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from pipeline.schema import ActionItem, CandidateSentence

AGREEMENT_THRESHOLD = 0.7


def best_match_score(text: str, candidate_texts: list[str]) -> float:
    """Best fuzzy match ratio, in [0, 1], between `text` and any string in
    `candidate_texts`. Direction-agnostic - used for both "does a classifier
    candidate support this LLM item" and "does an LLM item support this
    classifier candidate" (confidence/calibration_model.py needs both)."""
    best = 0.0
    for other in candidate_texts:
        score = fuzz.partial_ratio(text, other) / 100.0
        best = max(best, score)
    return best


def compute_agreement(
    llm_items: list[ActionItem], classifier_candidates: list[CandidateSentence]
) -> list[float]:
    """Returns one agreement score in [0, 1] per llm_items entry (same order),
    = the best fuzzy match against any classifier candidate's text."""
    candidate_texts = [c.text for c in classifier_candidates]
    return [best_match_score(item.evidence_span, candidate_texts) for item in llm_items]
