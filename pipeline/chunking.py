"""Splits long transcripts into utterance-aligned chunks that fit the local
model's context window. Chunk boundaries always fall between utterances,
never mid-utterance, so evidence_span quotes stay verbatim-matchable.
"""
from __future__ import annotations

from pipeline.schema import Transcript, Utterance

DEFAULT_MAX_CHARS = 6000  # ~1500-2000 tokens, leaves headroom in an 8192-token context


def chunk_transcript(
    transcript: Transcript, max_chars: int = DEFAULT_MAX_CHARS
) -> list[list[Utterance]]:
    chunks: list[list[Utterance]] = []
    current: list[Utterance] = []
    current_chars = 0

    for utterance in transcript.utterances:
        utterance_len = len(utterance.text)
        if current and current_chars + utterance_len > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(utterance)
        current_chars += utterance_len

    if current:
        chunks.append(current)

    return chunks or [[]]


def render_chunk(utterances: list[Utterance]) -> str:
    lines = []
    for u in utterances:
        speaker = u.speaker or "UNKNOWN"
        lines.append(f"{speaker}: {u.text}")
    return "\n".join(lines)
