"""Canonical data contract shared by every ingestion, pipeline, and eval stage."""
from __future__ import annotations

from typing import Optional

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
