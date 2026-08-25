"""Canonical data contract shared by every ingestion, pipeline, and eval stage."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Utterance(BaseModel):
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: str
    turn_index: int


class Transcript(BaseModel):
    meeting_id: str
    source_dataset: str  # "qmsum" | "meetingbank"
    split: Optional[str] = None  # "train" | "val"/"validation" | "test"
    participants: list[str]
    utterances: list[Utterance]
    reference_summary: Optional[str] = None


OwnerAttribution = Literal[
    "explicit_self",  # first-person commitment ("I'll send it")
    "explicit_named",  # directly addressed, resolvable to a participant
    "explicit_third_person_named",  # named but not directly addressed
    "inferred_pronoun",  # pronoun with no clear antecedent nearby
    "inferred_unassigned",  # no owner language at all
]

Priority = Literal["high", "medium", "low"]

Basis = Literal["explicit_statement", "contextual_inference", "speaker_role_default"]


class ActionItem(BaseModel):
    task: str
    # evidence_span comes right after task, before owner/deadline/etc., so the
    # model grounds itself in an actual quote before reasoning about the more
    # judgment-heavy fields - field order matches JSON generation order for
    # this local model.
    evidence_span: str  # verbatim quote from the transcript this was extracted from
    owner: Optional[str] = None
    owner_attribution: OwnerAttribution
    deadline: Optional[str] = None
    priority: Priority
    llm_confidence: Priority
    basis: Basis


class SummaryOutput(BaseModel):
    abstractive_summary: str
    key_decisions: list[str]
    action_items: list[ActionItem]


class CandidateSentence(BaseModel):
    """One classifier-flagged action-item candidate (classifier/predict.py).

    owner/deadline come from a lightweight rule-based post-process, not the
    classifier itself - see docs/architecture.md for why the classifier arm
    is scoped to binary sentence classification only.
    """

    utterance_index: int
    text: str
    is_action_item_prob: float
    owner: Optional[str] = None
    deadline: Optional[str] = None
