"""The key validating result: does confidence tiering measurably raise
precision? Precision only, not recall - the gold set labels the correctness
of EXTRACTED candidates, it doesn't exhaustively identify every true action
item per meeting, so recall against a complete ground-truth list isn't
computable from this labeling methodology.

Tiers here are QUANTILE-based (top/middle/bottom third by predicted
probability), not composite_score.py's fixed 0.66/0.33 thresholds. Checked
the actual score distribution first: it's compressed into roughly 0.06-0.70
with most mass around 0.5-0.6, so the fixed absolute thresholds (meant for
the live demo, where "High confidence" should mean something in absolute
terms to a user) leave "High" and "Low" as tiny, noisy groups here (n=3-8).
Tertiles guarantee balanced groups for a statistically legible "does rank
correlate with precision" comparison - the two threshold schemes serve
different purposes and are deliberately kept separate.

Reports two numbers, clearly separated:
- HELD-OUT (honest): only the 25% test split calibration_model.py never
  trained on. Small sample (~57 items, ~19/tier), but not circular.
- FULL SET (reference only): all 228 items, including ones the calibration
  model was trained on - expect optimistic bias here, shown for a larger-n
  sanity check only, never as the headline number.

Usage: python3 eval/action_item_prf.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from confidence.composite_score import build_feature_vector, compute_signals
from pipeline.schema import CandidateSentence, SummaryOutput, Transcript

LABELS_PATH = Path("data/eval_gold/labeling_sheet.csv")
LLM_DIR = Path("data/outputs/llm")
CLF_DIR = Path("data/outputs/classifier")
PROCESSED_ROOT = Path("data/processed")
MODEL_PATH = Path("models/calibration_lr.joblib")
REPORT_OUT = Path("reports/evaluation_summary.md")

_CONFIDENCE_TO_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _load_transcript(meeting_id: str) -> Transcript:
    dataset = meeting_id.split("_", 1)[0]
    return Transcript.model_validate_json((PROCESSED_ROOT / dataset / f"{meeting_id}.json").read_text())


def _load_meeting_outputs(meeting_id: str):
    llm_items = []
    llm_path = LLM_DIR / f"{meeting_id}.json"
    if llm_path.exists():
        llm_items = SummaryOutput.model_validate_json(llm_path.read_text()).action_items

    clf_candidates = []
    clf_path = CLF_DIR / f"{meeting_id}.json"
    if clf_path.exists():
        clf_candidates = [CandidateSentence.model_validate(c) for c in json.loads(clf_path.read_text())]

    return llm_items, clf_candidates


def _tertile_labels(probs: np.ndarray) -> list[str]:
    """Top/middle/bottom third BY RANK within this specific set of scores -
    see module docstring for why this is quantile-based rather than using
    composite_score.py's fixed thresholds."""
    lower, upper = np.percentile(probs, [33.33, 66.67])
    labels = []
    for p in probs:
        if p >= upper:
            labels.append("Top third")
        elif p >= lower:
            labels.append("Middle third")
        else:
            labels.append("Bottom third")
    return labels


def _precision_by_tier(tiers: list[str], correct: list[int]) -> dict[str, tuple[float, int]]:
    by_tier: dict[str, list[int]] = defaultdict(list)
    for tier, is_correct in zip(tiers, correct):
        by_tier[tier].append(is_correct)

    result = {}
    for tier in ["Top third", "Middle third", "Bottom third"]:
        if tier in by_tier:
            n = len(by_tier[tier])
            result[tier] = (sum(by_tier[tier]) / n, n)
    return result


def main() -> None:
    rows = list(csv.DictReader(LABELS_PATH.open()))
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    transcripts_cache: dict[str, Transcript] = {}
    outputs_cache: dict[str, tuple] = {}

    X, y, tiers, arms = [], [], [], []
    for row in rows:
        mid = row["meeting_id"]
        if mid not in transcripts_cache:
            transcripts_cache[mid] = _load_transcript(mid)
            outputs_cache[mid] = _load_meeting_outputs(mid)
        transcript = transcripts_cache[mid]
        llm_items, clf_candidates = outputs_cache[mid]

        evidence = row["evidence_span"]
        if row["arm"] == "llm":
            item = llm_items[int(row["item_index"])]
            comparison_texts = [c.text for c in clf_candidates]
            model_confidence = _CONFIDENCE_TO_SCORE[item.llm_confidence]
        else:
            candidate = clf_candidates[int(row["item_index"])]
            comparison_texts = [i.evidence_span for i in llm_items]
            model_confidence = candidate.is_action_item_prob

        category, grounded, grounding_score, agreement = compute_signals(evidence, transcript, comparison_texts)
        features = build_feature_vector(category, grounded, grounding_score, agreement, model_confidence, evidence)
        X.append(features)

        fully_correct = int(
            row["correct_task"] == "y" and row["correct_owner"] == "y" and row["correct_deadline"] == "y"
        )
        y.append(fully_correct)
        arms.append(row["arm"])

    X = np.array(X)
    y = np.array(y)
    probs = model.predict_proba(X)[:, 1]

    # Reproduce calibration_model.py's exact split (same X, y, random_state, stratify)
    # to identify which rows were held out from training - avoids evaluating the
    # model on data it already saw.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=y)

    # Tertile boundaries computed separately within each set (held-out vs full) -
    # "top third" means top third of THAT set's own score distribution.
    full_tiers = _tertile_labels(probs)
    held_out_probs = probs[test_idx]
    held_out_tiers = _tertile_labels(held_out_probs)

    full_precision = _precision_by_tier(full_tiers, list(y))
    held_out_precision = _precision_by_tier(held_out_tiers, list(y[test_idx]))
    baseline_precision = y.mean()
    held_out_baseline = y[test_idx].mean()

    lines = ["# Evaluation Summary\n"]
    lines.append("## Action-item precision by confidence tier\n")
    lines.append("Precision only (see module docstring for why recall isn't computable from this labeling setup).\n")
    lines.append(f"**No-confidence-filter baseline** (held-out): {held_out_baseline:.0%}\n")
    lines.append("\n### Held-out (honest - never seen during training)\n")
    lines.append("| Tier | Precision | n |")
    lines.append("|---|---|---|")
    for tier, (prec, n) in held_out_precision.items():
        lines.append(f"| {tier} | {prec:.0%} | {n} |")

    lines.append(f"\n### Full set (reference only, includes training data - baseline {baseline_precision:.0%})\n")
    lines.append("| Tier | Precision | n |")
    lines.append("|---|---|---|")
    for tier, (prec, n) in full_precision.items():
        lines.append(f"| {tier} | {prec:.0%} | {n} |")

    report = "\n".join(lines)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report + "\n")
    print(report)
    print(f"\nSaved {REPORT_OUT}")


if __name__ == "__main__":
    main()
