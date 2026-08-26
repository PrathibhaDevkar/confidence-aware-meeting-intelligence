"""Trains the confidence-calibration model: logistic regression on the four
independent signals (owner-attribution category, grounding, cross-model
agreement, the originating model's own confidence) -> empirical correctness,
using the hand-labeled gold set (data/eval_gold/labeling_sheet.csv) as
ground truth. "Empirically correct" here means task AND owner AND deadline
are all marked correct - the strict bar a user-facing confidence score
should actually mean.

Usage: python3 confidence/calibration_model.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from confidence.composite_score import OWNER_CATEGORIES, build_feature_vector, compute_signals
from pipeline.schema import CandidateSentence, SummaryOutput, Transcript

LABELS_PATH = Path("data/eval_gold/labeling_sheet.csv")
LLM_DIR = Path("data/outputs/llm")
CLF_DIR = Path("data/outputs/classifier")
PROCESSED_ROOT = Path("data/processed")
MODEL_OUT = Path("models/calibration_lr.joblib")
REPORT_OUT = Path("reports/calibration_reliability_diagram.png")

FEATURE_NAMES = OWNER_CATEGORIES + [
    "grounded", "grounding_score", "agreement", "model_confidence", "evidence_length",
]
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


def build_features(row: dict, transcript: Transcript, llm_items: list, clf_candidates: list) -> list[float]:
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
    return build_feature_vector(category, grounded, grounding_score, agreement, model_confidence, evidence)


def main() -> None:
    rows = list(csv.DictReader(LABELS_PATH.open()))

    transcripts_cache: dict[str, Transcript] = {}
    outputs_cache: dict[str, tuple] = {}

    X, y = [], []
    for row in rows:
        mid = row["meeting_id"]
        if mid not in transcripts_cache:
            transcripts_cache[mid] = _load_transcript(mid)
            outputs_cache[mid] = _load_meeting_outputs(mid)

        features = build_features(row, transcripts_cache[mid], *outputs_cache[mid])
        X.append(features)

        fully_correct = row["correct_task"] == "y" and row["correct_owner"] == "y" and row["correct_deadline"] == "y"
        y.append(int(fully_correct))

    X = np.array(X)
    y = np.array(y)
    print(f"{len(y)} labeled examples, {y.sum()} fully-correct ({y.mean() * 100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    print("\nHeld-out test set performance:")
    print(classification_report(y_test, model.predict(X_test), target_names=["incorrect", "correct"]))

    print("Feature coefficients:")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {name:24s} {coef:+.3f}")

    # Reliability diagram over the FULL set: does predicted confidence track observed accuracy?
    probs = model.predict_proba(X)[:, 1]
    bucket_edges = np.linspace(0, 1, 6)
    bucket_idx = np.clip(np.digitize(probs, bucket_edges) - 1, 0, len(bucket_edges) - 2)

    bucket_acc, bucket_conf, bucket_n = [], [], []
    for b in range(len(bucket_edges) - 1):
        mask = bucket_idx == b
        if mask.sum() > 0:
            bucket_acc.append(y[mask].mean())
            bucket_conf.append(probs[mask].mean())
            bucket_n.append(int(mask.sum()))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.scatter(bucket_conf, bucket_acc, s=[n * 15 for n in bucket_n], alpha=0.7, label="observed (size = n)")
    for c, a, n in zip(bucket_conf, bucket_acc, bucket_n):
        plt.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Calibration reliability diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORT_OUT, dpi=120, bbox_inches="tight")
    print(f"\nSaved {REPORT_OUT}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "owner_categories": OWNER_CATEGORIES}, MODEL_OUT)
    print(f"Saved {MODEL_OUT}")


if __name__ == "__main__":
    main()
