"""LLM-as-judge scoring for generated summaries: rates coherence, consistency,
fluency, relevance (1-5 each) against the source transcript.

Honest limitation: this uses the SAME local model (llama3.2:3b) that
generated the summaries, since there's no separate/larger model available
(no paid API - see docs/architecture.md). A model judging its own outputs
can be biased toward rating them favorably. Treat these scores as a weak
secondary signal, not an independent quality gate - the real validating
result is eval/action_item_prf.py's precision-by-confidence-tier, which is
checked against genuine human labels instead.

For long transcripts, only the first ~6000 chars are shown to the judge
(same budget as one extraction chunk) - consistency-checking is partial for
very long MeetingBank meetings as a result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from pipeline.schema import Transcript

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
NUM_CTX = 8192
TRANSCRIPT_CHAR_BUDGET = 6000

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
JUDGE_TEMPLATE = (PROMPTS_DIR / "judge_rubric_prompt.md").read_text()

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
        "coherence_reason": {"type": "string"},
        "consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "consistency_reason": {"type": "string"},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency_reason": {"type": "string"},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance_reason": {"type": "string"},
    },
    "required": [
        "coherence", "coherence_reason", "consistency", "consistency_reason",
        "fluency", "fluency_reason", "relevance", "relevance_reason",
    ],
}


def judge_summary(transcript: Transcript, summary_text: str, model: str = DEFAULT_MODEL) -> dict:
    transcript_text = " ".join(u.text for u in transcript.utterances)[:TRANSCRIPT_CHAR_BUDGET]

    prompt = JUDGE_TEMPLATE.format(
        source_dataset=transcript.source_dataset,
        meeting_id=transcript.meeting_id,
        transcript_text=transcript_text,
        summary_text=summary_text,
    )

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "format": _JUDGE_SCHEMA,
            "stream": False,
            "options": {"num_ctx": NUM_CTX},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])
