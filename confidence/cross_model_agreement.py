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


def compute_agreement(
    llm_items: list[ActionItem], classifier_candidates: list[CandidateSentence]
) -> list[float]:
    """Returns one agreement score in [0, 1] per llm_items entry (same order),
    = the best fuzzy match against any classifier candidate's text."""
    scores = []
    for item in llm_items:
        best = 0.0
        for candidate in classifier_candidates:
            score = fuzz.partial_ratio(item.evidence_span, candidate.text) / 100.0
            best = max(best, score)
        scores.append(best)
    return scores
