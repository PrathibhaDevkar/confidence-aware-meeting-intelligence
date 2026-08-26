# Architecture & Scope

## Problem

Commercial AI meeting summarizers (Otter.ai, Fireflies, Fathom, Gong, etc.) present every summary and extracted action item with equal, undifferentiated confidence. Research shows this is exactly backwards: most quality complaints trace back to upstream transcription/speaker-diarization errors that compound into wrong action-item ownership, not to the summarization step itself. Commercial action-item accuracy sits at only 62-87%.

## Approach

A pipeline that treats confidence as a first-class, computed output rather than an afterthought:

1. **Prompted LLM extraction** (local Ollama model, `llama3.2:3b`, JSON-schema-constrained structured output) — summary + action items with per-field metadata (`owner_attribution`, `evidence_span`, `llm_confidence`, `basis`). Runs fully offline at zero marginal cost — no paid API dependency. A small local model makes more mistakes than a frontier API model would; that tradeoff is deliberate (see below) and the eval harness (Phase 6) measures it honestly rather than hiding it.
2. **Fine-tuned classifier baseline** (DistilBERT, binary action-item-sentence classification) — an independent second opinion, and a cost/latency/accuracy comparison point.
3. **Confidence layer** — combines owner-attribution category, evidence-grounding verification, cross-model agreement (between 1 and 2), and the LLM's self-reported confidence into one calibrated score (logistic regression, trained against a hand-labeled gold set).
4. **Hybrid evaluation** — ROUGE/BERTScore (cheap regression check only), LLM-as-judge (primary quality signal), and hand-labeled action-item precision/recall broken down by confidence tier (the key validating result: does "High confidence only" measurably raise precision?).

## Explicit scope boundaries (deliberate, not oversights)

- **No ASR / speaker diarization from raw audio.** This project starts from existing transcripts (QMSum, MeetingBank) and treats upstream transcription error as a given condition to model confidence around, not a problem this project solves directly.
- **Datasets**: QMSum + MeetingBank only for v1. AMI/ICSI direct ingestion and ELITR/AutoMin are out of scope for v1.
- **Classifier baseline**: binary sentence-level classification only. Owner/deadline for this arm come from a lightweight rule-based/spaCy post-process, not a second trained model — keeps the two comparison arms proportionate in effort.
- **Gold evaluation set**: ~15-20 meetings / ~150-250 hand-labeled action items, selected once and held out from all prompt-tuning decisions.
- **No paid API**: runs entirely on a local Ollama model (`llama3.2:3b`, ~2GB) rather than Claude/GPT, by explicit choice (no API budget available). This is a real constraint, not a preference — see `requirements.txt` for how to add a paid-API arm back in later if that changes.

## Data-cleaning note

QMSum's source (AMI/ICSI) transcripts carry transcription-convention annotation tags (`{disfmarker}`, `{vocalsound}`, `{pause}`, `{gap}`, `{comment}`, `{nonvocalsound}`). These are stripped during ingestion (`ingestion/ingest_qmsum.py`) — empirically, leaving them in caused the extraction model to miss real action items entirely (0 action items found on a test meeting with the tags present vs. 3-4 correctly found after cleaning).

## Gold-labeling stability check (found and fixed a real pattern)

The gold set (228 rows) was labeled with rule-based *suggestions* shown alongside each row (`confidence/pseudo_label_heuristic.py`) to speed up review - not model judgments, plain pattern-matching, specifically to avoid an LLM pre-judging its own output. After first-pass labeling matched the suggestions on 96% of rows (218/228), a stability recheck on a random sample surfaced a real, specific gap: **rows where the LLM's `evidence_span` doesn't actually support its own claimed task were still getting marked correct.** E.g., task "Create a report summarizing the meeting's key points" backed only by evidence "Okay we have to fill in all this stuff." A targeted second pass on the 7 lowest task/evidence-similarity rows (of the ones marked correct) found 3 genuine errors, which were corrected.

This is a real, disclosed limitation of the gold set, not swept under the rug: single-annotator labeling assisted by imperfect suggestions has some anchoring risk, partially mitigated here by the stability check but not eliminated by it. Final distribution after correction: 142 `n` / 86 `y` on `correct_task` (out of 228).

## Phase plan

1. **Setup & scoping** — repo skeleton, scope boundaries (this phase).
2. **Data acquisition & normalization** — QMSum + MeetingBank into a canonical `Transcript` schema; select the gold evaluation meetings.
3. **Prompted-LLM extraction pipeline** — local Ollama structured-output extraction (`pipeline/llm_extractor.py`).
4. **Fine-tuned classifier baseline** — DistilBERT action-item-sentence classifier (`classifier/`).
5. **Confidence-tracking & calibration layer** — owner attribution, grounding check, cross-model agreement, calibrated composite score (`confidence/`).
6. **Hybrid evaluation harness** — ROUGE/BERTScore + LLM-as-judge + hand-labeled precision/recall by confidence tier (`eval/`).
7. **Streamlit demo** — ties every stage together (`app/`).
8. **Polish & packaging** — README, docs, optional deployment.
