"""Builds a weakly-labeled (sentence, is_action_item) training set from QMSum
+ MeetingBank utterances. Neither source dataset ships gold action-item-
sentence labels, so this uses two independent, dataset-agnostic signals:

1. Modal/commitment language ("will", "need to", "let's", ...)
2. Token overlap between the utterance and the meeting's reference_summary
   (a proxy for "this sentence is salient enough that the summary reflects it")

Both signals are weak on their own; requiring both approximates "actionable-
sounding AND salient." This is intentionally noisy distant supervision, not
gold labels - the Phase 4 hand-labeled gold set (held out here entirely) is
the real test set, never used for training.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import Transcript

PROCESSED_ROOT = Path("data/processed")
GOLD_IDS_PATH = Path("data/eval_gold/selected_meetings.txt")
OUT_PATH = Path("data/outputs/classifier/training_data.jsonl")

NEGATIVE_TO_POSITIVE_RATIO = 3
OVERLAP_THRESHOLD = 0.25
MIN_TEXT_LEN = 8
SEED = 42

MODAL_RE = re.compile(
    r"\b(will|should|need(?:s)? to|going to|gonna|have to|has to|must|"
    r"let'?s|can you|could you|shall)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "should", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "this", "that",
    "these", "those", "not", "no", "so", "just", "yeah", "okay", "um", "uh",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _overlap_score(utterance_text: str, summary_text: str | None) -> float:
    u_words = _content_words(utterance_text)
    if not u_words:
        return 0.0
    s_words = _content_words(summary_text or "")
    return len(u_words & s_words) / len(u_words)


def main() -> None:
    gold_ids = set(GOLD_IDS_PATH.read_text().splitlines())
    positives, negatives = [], []

    for path in sorted(PROCESSED_ROOT.glob("*/*.json")):
        transcript = Transcript.model_validate_json(path.read_text())
        if transcript.meeting_id in gold_ids:
            continue  # held out for Phase 4 gold evaluation, never trained on

        for utterance in transcript.utterances:
            text = utterance.text.strip()
            if len(text) < MIN_TEXT_LEN:
                continue

            has_modal = bool(MODAL_RE.search(text))
            overlap = _overlap_score(text, transcript.reference_summary)
            label = 1 if (has_modal and overlap >= OVERLAP_THRESHOLD) else 0

            row = {"text": text, "label": label, "meeting_id": transcript.meeting_id}
            (positives if label else negatives).append(row)

    rng = random.Random(SEED)
    rng.shuffle(negatives)
    negatives = negatives[: len(positives) * NEGATIVE_TO_POSITIVE_RATIO]

    rows = positives + negatives
    rng.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"positives={len(positives)} negatives_kept={len(negatives)} total={len(rows)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
