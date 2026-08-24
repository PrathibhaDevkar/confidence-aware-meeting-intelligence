# Architecture & Scope

## Problem

Commercial AI meeting summarizers (Otter.ai, Fireflies, Fathom, Gong, etc.) present every summary and extracted action item with equal, undifferentiated confidence. Research shows this is exactly backwards: most quality complaints trace back to upstream transcription/speaker-diarization errors that compound into wrong action-item ownership, not to the summarization step itself. Commercial action-item accuracy sits at only 62-87%.

## Approach

A pipeline that treats confidence as a first-class, computed output rather than an afterthought:

1. **Prompted LLM extraction** (Claude API, structured outputs) — summary + action items with per-field metadata (`owner_attribution`, `evidence_span`, `llm_confidence`, `basis`).
2. **Fine-tuned classifier baseline** (DistilBERT, binary action-item-sentence classification) — an independent second opinion, and a cost/latency/accuracy comparison point.
3. **Confidence layer** — combines owner-attribution category, evidence-grounding verification, cross-model agreement (between 1 and 2), and the LLM's self-reported confidence into one calibrated score (logistic regression, trained against a hand-labeled gold set).
4. **Hybrid evaluation** — ROUGE/BERTScore (cheap regression check only), LLM-as-judge (primary quality signal), and hand-labeled action-item precision/recall broken down by confidence tier (the key validating result: does "High confidence only" measurably raise precision?).

## Explicit scope boundaries (deliberate, not oversights)

- **No ASR / speaker diarization from raw audio.** This project starts from existing transcripts (QMSum, MeetingBank) and treats upstream transcription error as a given condition to model confidence around, not a problem this project solves directly.
- **Datasets**: QMSum + MeetingBank only for v1. AMI/ICSI direct ingestion and ELITR/AutoMin are out of scope for v1.
- **Classifier baseline**: binary sentence-level classification only. Owner/deadline for this arm come from a lightweight rule-based/spaCy post-process, not a second trained model — keeps the two comparison arms proportionate in effort.
- **Gold evaluation set**: ~15-20 meetings / ~150-250 hand-labeled action items, selected once and held out from all prompt-tuning decisions.

## Phase plan

1. **Setup & scoping** — repo skeleton, scope boundaries (this phase).
2. **Data acquisition & normalization** — QMSum + MeetingBank into a canonical `Transcript` schema; select the gold evaluation meetings.
3. **Prompted-LLM extraction pipeline** — Claude structured-output extraction (`pipeline/llm_extractor.py`).
4. **Fine-tuned classifier baseline** — DistilBERT action-item-sentence classifier (`classifier/`).
5. **Confidence-tracking & calibration layer** — owner attribution, grounding check, cross-model agreement, calibrated composite score (`confidence/`).
6. **Hybrid evaluation harness** — ROUGE/BERTScore + LLM-as-judge + hand-labeled precision/recall by confidence tier (`eval/`).
7. **Streamlit demo** — ties every stage together (`app/`).
8. **Polish & packaging** — README, docs, optional deployment.
