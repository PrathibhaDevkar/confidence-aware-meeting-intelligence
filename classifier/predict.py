"""Predicts action-item-candidate sentences for a transcript using the
fine-tuned classifier, then fills owner/deadline via a lightweight rule-based
pass (not a second trained model - see docs/architecture.md for why the
classifier arm is scoped to binary sentence classification only).

Usage: python3 classifier/predict.py --input data/processed/qmsum/<id>.json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import CandidateSentence, Transcript

MODEL_PATH = Path("models/action_item_classifier")

_SELF_COMMIT_RE = re.compile(r"\bI(?:'ll|\s+will|\s+can|\s+shall|'m going to)\b", re.IGNORECASE)
_DEADLINE_RE = re.compile(
    r"\b(today|tomorrow|tonight|this week|next week|end of (?:day|week|month)"
    r"|by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|\w+day)"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_model_cache: dict[str, tuple] = {}


def _load(model_path: Path):
    key = str(model_path)
    if key not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        _model_cache[key] = (tokenizer, model)
    return _model_cache[key]


def _guess_owner(text: str, speaker: Optional[str], participants: list[str]) -> Optional[str]:
    if speaker and _SELF_COMMIT_RE.search(text):
        return speaker
    for name in participants:
        if name and name != speaker and re.search(rf"\b{re.escape(name)}\b", text):
            return name
    return None


def _guess_deadline(text: str) -> Optional[str]:
    match = _DEADLINE_RE.search(text)
    return match.group(0) if match else None


def predict_candidates(
    transcript: Transcript, model_path: Path = MODEL_PATH, threshold: float = 0.5
) -> list[CandidateSentence]:
    tokenizer, model = _load(model_path)
    candidates = []

    for utterance in transcript.utterances:
        text = utterance.text.strip()
        if len(text) < 8:
            continue

        inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

        if prob < threshold:
            continue

        candidates.append(
            CandidateSentence(
                utterance_index=utterance.turn_index,
                text=text,
                is_action_item_prob=prob,
                owner=_guess_owner(text, utterance.speaker, transcript.participants),
                deadline=_guess_deadline(text),
            )
        )

    return candidates


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    transcript = Transcript.model_validate_json(args.input.read_text())
    candidates = predict_candidates(transcript, args.model, args.threshold)

    print(f"{len(candidates)} candidate action items (threshold={args.threshold}):")
    for c in candidates:
        print(f"  [{c.is_action_item_prob:.2f}] owner={c.owner!r} deadline={c.deadline!r} :: {c.text[:80]}")


if __name__ == "__main__":
    main()
