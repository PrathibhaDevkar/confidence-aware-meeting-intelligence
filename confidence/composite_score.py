"""Live confidence scoring for a single action item - the one function the
Streamlit demo calls at render time. Loads the trained calibration model
once (models/calibration_lr.joblib) and scores new, unlabeled action items
using the exact same feature computation calibration_model.py trained on,
so there's no train/serve skew between the two.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from confidence.cross_model_agreement import best_match_score
from confidence.grounding_check import find_evidence_speaker, is_grounded
from confidence.owner_attribution import classify_owner_mention
from pipeline.schema import ConfidenceResult, Transcript

MODEL_PATH = Path("models/calibration_lr.joblib")

OWNER_CATEGORIES = [
    "explicit_self", "explicit_named", "explicit_third_person_named",
    "inferred_pronoun", "inferred_unassigned",
]

HIGH_THRESHOLD = 0.66
MEDIUM_THRESHOLD = 0.33

_model_cache: Optional[dict] = None


def _load_model() -> dict:
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def compute_signals(evidence_span: str, transcript: Transcript, comparison_texts: list[str]):
    """The four raw signals, before feature encoding - shared by training
    (calibration_model.py) and live scoring (score_action_item below)."""
    speaker = find_evidence_speaker(evidence_span, transcript)
    category = classify_owner_mention(evidence_span, speaker, transcript.participants)
    grounded, grounding_score = is_grounded(evidence_span, transcript)
    agreement = best_match_score(evidence_span, comparison_texts) if comparison_texts else 0.0
    return category, grounded, grounding_score, agreement


def build_feature_vector(
    category: str, grounded: bool, grounding_score: float, agreement: float,
    model_confidence: float, evidence_span: str,
) -> list[float]:
    """The canonical feature vector for the calibration model - shared by
    training (calibration_model.py, via these same signal functions) and
    live scoring (score_action_item below)."""
    one_hot = [1.0 if category == c else 0.0 for c in OWNER_CATEGORIES]
    return one_hot + [
        float(grounded), grounding_score, agreement, model_confidence, float(len(evidence_span or "")),
    ]


def _tier_for(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def score_action_item(
    evidence_span: str,
    transcript: Transcript,
    model_confidence: float,
    comparison_texts: list[str],
) -> ConfidenceResult:
    """
    evidence_span: the quote to score (an ActionItem's evidence_span, or a
        CandidateSentence's text).
    model_confidence: unified 0-1 confidence from whichever arm produced this
        item (mapped llm_confidence for the LLM arm, or is_action_item_prob
        directly for the classifier arm).
    comparison_texts: the OTHER arm's texts for this meeting, for cross-model
        agreement (classifier candidates' .text when scoring an LLM item, or
        LLM items' .evidence_span when scoring a classifier candidate).
    """
    bundle = _load_model()
    category, grounded, grounding_score, agreement = compute_signals(
        evidence_span, transcript, comparison_texts
    )

    features = np.array(
        [build_feature_vector(category, grounded, grounding_score, agreement, model_confidence, evidence_span)]
    )
    score = float(bundle["model"].predict_proba(features)[0, 1])

    rationale = (
        f"{'grounded in transcript' if grounded else f'NOT grounded (best match {grounding_score:.0%})'}; "
        f"owner: {category.replace('_', ' ')}; "
        f"{'other model agrees' if agreement >= 0.7 else 'no cross-model corroboration'} ({agreement:.0%})"
    )

    return ConfidenceResult(score=score, tier=_tier_for(score), rationale=rationale)
