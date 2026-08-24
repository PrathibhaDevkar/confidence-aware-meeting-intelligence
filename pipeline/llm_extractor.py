"""Prompted extraction via a locally-run Ollama model (default: llama3.2:3b).

No Anthropic API dependency by design (see docs/architecture.md) - this is the
"prompted LLM extraction" arm of the required LLM-vs-classifier comparison,
running fully offline and at zero marginal cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from pipeline.chunking import chunk_transcript, render_chunk
from pipeline.schema import ActionItem, SummaryOutput, Transcript

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
NUM_CTX = 8192  # tokens; balances capability against the 8GB-RAM budget
MAX_RETRIES = 2

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "extraction_system_prompt.md").read_text()
USER_TEMPLATE = (PROMPTS_DIR / "extraction_user_template.md").read_text()

_SCHEMA = SummaryOutput.model_json_schema()


def _call_ollama(prompt: str, model: str, response_format: dict) -> dict:
    last_error: Exception | None = None
    for _ in range(MAX_RETRIES):
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "format": response_format,
                "stream": False,
                "options": {"num_ctx": NUM_CTX},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json()["response"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"Model did not return valid JSON after {MAX_RETRIES} attempts: {last_error}")


def _extract_chunk(text: str, source_dataset: str, meeting_id: str, model: str) -> SummaryOutput:
    prompt = USER_TEMPLATE.format(
        source_dataset=source_dataset, meeting_id=meeting_id, transcript_text=text
    )
    data = _call_ollama(prompt, model, _SCHEMA)
    return SummaryOutput.model_validate(data)


_SUMMARY_OF_SUMMARIES_SCHEMA = {
    "type": "object",
    "properties": {"abstractive_summary": {"type": "string"}},
    "required": ["abstractive_summary"],
}


def _reduce_summaries(partial_summaries: list[str], meeting_id: str, model: str) -> str:
    joined = "\n\n".join(f"Excerpt {i + 1} summary: {s}" for i, s in enumerate(partial_summaries))
    prompt = (
        f"These are summaries of consecutive excerpts from one meeting ({meeting_id}), "
        f"in order:\n\n{joined}\n\nWrite one overall abstractive summary of the whole meeting."
    )
    data = _call_ollama(prompt, model, _SUMMARY_OF_SUMMARIES_SCHEMA)
    return data["abstractive_summary"]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract(transcript: Transcript, model: str = DEFAULT_MODEL) -> SummaryOutput:
    chunks = chunk_transcript(transcript)

    if len(chunks) == 1:
        return _extract_chunk(
            render_chunk(chunks[0]), transcript.source_dataset, transcript.meeting_id, model
        )

    partials: list[SummaryOutput] = []
    for chunk in chunks:
        if not chunk:
            continue
        partials.append(
            _extract_chunk(render_chunk(chunk), transcript.source_dataset, transcript.meeting_id, model)
        )

    all_action_items: list[ActionItem] = [item for p in partials for item in p.action_items]
    all_decisions = _dedupe_preserve_order([d for p in partials for d in p.key_decisions])
    overall_summary = _reduce_summaries(
        [p.abstractive_summary for p in partials], transcript.meeting_id, model
    )

    return SummaryOutput(
        abstractive_summary=overall_summary,
        key_decisions=all_decisions,
        action_items=all_action_items,
    )
