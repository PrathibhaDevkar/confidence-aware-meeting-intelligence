"""Transparent, rule-based *suggestions* to speed up gold-set labeling -
not a model judgment, not the final label. Every suggestion comes with a
plain-text reason so it can be sanity-checked at a glance, and the labeling
app requires an explicit click either way (no pre-selected radio) so a
suggestion is a starting point for review, not a silent default.

This is deliberately NOT an LLM call: using the same local model (or a
similar one) to pre-judge its own extraction arm's output would reintroduce
the circularity problem the human gold-labeling step exists to avoid. Plain
pattern-matching keeps the suggestion fully inspectable and independent of
either comparison arm's own reasoning.
"""
from __future__ import annotations

import re
from typing import Optional

from confidence.grounding_check import find_evidence_speaker, is_grounded
from confidence.owner_attribution import classify_owner_mention
from pipeline.schema import Transcript

_MODAL_RE = re.compile(
    r"\b(will|should|need(?:s)? to|going to|gonna|have to|has to|must|"
    r"let'?s|can you|could you|shall)\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(today|tomorrow|tonight|this week|next week|end of (?:day|week|month)"
    r"|by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|\w+day)"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_STRONG_OWNER_CATEGORIES = {"explicit_self", "explicit_named", "explicit_third_person_named"}


def suggest_task(text_shown: str, evidence_span: str, transcript: Optional[Transcript]) -> tuple[str, str]:
    if transcript is not None:
        grounded, score = is_grounded(evidence_span, transcript)
        if not grounded:
            return "No", f"evidence not found in transcript (best match {score:.0%})"

    text = (evidence_span or text_shown or "").strip()
    if len(text) < 15:
        return "No", "too short to be a real commitment"
    if "?" in text:
        return "No", "contains a question, not a clean commitment"
    if _MODAL_RE.search(text):
        return "Yes", "contains commitment language (will/need to/let's/...)"
    return "No", "no commitment language found"


def suggest_owner(owner: str, evidence_span: str, transcript: Optional[Transcript]) -> tuple[str, str]:
    participants = transcript.participants if transcript else []
    speaker = find_evidence_speaker(evidence_span, transcript) if transcript else None
    category = classify_owner_mention(evidence_span, speaker, participants)

    if not owner:
        if category in ("inferred_unassigned", "inferred_pronoun"):
            return "Yes", "blank owner matches - transcript doesn't clearly name one either"
        return "No", f"owner left blank but text looks {category}"

    if category in _STRONG_OWNER_CATEGORIES:
        return "Yes", f"owner named, text independently looks {category}"
    return "No", f"owner claimed but text only looks {category}"


def suggest_deadline(deadline: str, evidence_span: str) -> tuple[str, str]:
    text = evidence_span or ""
    found = _DEADLINE_RE.search(text)

    if not deadline:
        if not found:
            return "Yes", "blank deadline matches - no time language found"
        return "No", f"deadline left blank but text mentions '{found.group(0)}'"

    if found and found.group(0).lower() in deadline.lower():
        return "Yes", f"claimed deadline matches text ('{found.group(0)}')"
    return "No", "claimed deadline not found in the evidence text"
