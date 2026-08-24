# Confidence-Aware Meeting Intelligence

An LLM-based meeting/document summarizer and action-item extractor that explicitly tracks and surfaces *confidence* for every extracted field — instead of presenting every summary and action item with equal, false certainty like existing commercial tools (Otter.ai, Fireflies, Fathom, etc.).

## Why

Research into commercial AI meeting tools shows that most complaints about "wrong" summaries are actually complaints about upstream transcription/speaker-attribution errors compounding into misattributed action items — and none of these tools tell you *how sure* they are about any given extraction. Action-item accuracy in these tools sits at only 62-87%, with no confidence signal at all.

This project builds a pipeline that:

1. Extracts structured summaries + action items from meeting transcripts using a local LLM (Ollama, `llama3.2:3b` — prompted extraction, JSON-schema-constrained structured outputs, zero API cost)
2. Independently trains a fine-tuned classifier baseline for action-item detection, for a head-to-head cost/latency/accuracy comparison
3. Combines four independent signals (owner-attribution category, evidence grounding, cross-model agreement, LLM self-reported confidence) into one calibrated confidence score per action item
4. Evaluates the whole system with a hybrid metric stack (ROUGE/BERTScore as a cheap check, LLM-as-judge as the real quality gate, and hand-labeled precision/recall broken down by confidence tier)

See [`docs/architecture.md`](docs/architecture.md) for the full design and scope boundaries.

## Status

Work in progress — see the phase breakdown in `docs/architecture.md`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# LLM extraction runs against a local Ollama model - no API key needed
brew install ollama          # or see https://ollama.com/download
brew services start ollama
ollama pull llama3.2:3b      # ~2GB

# Download and normalize QMSum + MeetingBank
./scripts/download_datasets.sh
python3 ingestion/ingest_qmsum.py
python3 ingestion/ingest_meetingbank.py
```
