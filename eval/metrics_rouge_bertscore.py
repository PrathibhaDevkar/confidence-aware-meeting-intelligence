"""Cheap regression-check metrics: ROUGE-L and BERTScore against each
meeting's reference summary. Per the research this project was scoped
against, these correlate weakly with human judgment on hallucination/
quality (ROUGE-L ~0.13, BERTScore ~0.19 Spearman) - tracked here as a fast,
deterministic sanity signal only, never as the actual quality gate (that's
eval/llm_judge.py's job, itself secondary to action_item_prf.py's result).

Uses distilbert-base-uncased for BERTScore (already downloaded for the
classifier baseline) rather than the ~1.3GB default (roberta-large), to
avoid another large model download.
"""
from __future__ import annotations

from bert_score import score as bert_score
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(reference: str, hypothesis: str) -> float:
    return _scorer.score(reference, hypothesis)["rougeL"].fmeasure


def bertscore_batch(references: list[str], hypotheses: list[str]) -> list[float]:
    _, _, f1 = bert_score(hypotheses, references, model_type="distilbert-base-uncased", lang="en", verbose=False)
    return f1.tolist()
