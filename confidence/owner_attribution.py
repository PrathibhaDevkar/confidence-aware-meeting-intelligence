"""Independently re-derives the owner-attribution category from the raw
evidence_span text, using rules rather than trusting the LLM's self-report.
Agreement/disagreement between this and the LLM's own `owner_attribution`
field is itself a confidence signal (composite_score.py).
"""
from __future__ import annotations

import re
from typing import Optional

from pipeline.schema import OwnerAttribution

_SELF_COMMIT_RE = re.compile(r"\bI(?:'ll|\s+will|\s+can|\s+shall|'m going to)\b", re.IGNORECASE)
_DIRECT_ADDRESS_RE = re.compile(
    r"\byou(?:'ll|\s+will|\s+should|\s+need to)\b|\bcan you\b|\bcould you\b", re.IGNORECASE
)
_PRONOUN_RE = re.compile(r"\b(he|she|they|him|her|them)\b", re.IGNORECASE)


def classify_owner_mention(
    evidence_span: str, speaker: Optional[str], participants: list[str]
) -> OwnerAttribution:
    text = evidence_span or ""

    if speaker and _SELF_COMMIT_RE.search(text):
        return "explicit_self"

    if _DIRECT_ADDRESS_RE.search(text):
        return "explicit_named"

    for name in participants:
        if name and name != speaker and re.search(rf"\b{re.escape(name)}\b", text):
            return "explicit_third_person_named"

    if _PRONOUN_RE.search(text):
        return "inferred_pronoun"

    return "inferred_unassigned"
