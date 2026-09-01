# Evaluation Summary

## Action-item precision by confidence tier

Precision only (see module docstring for why recall isn't computable from this labeling setup).

**No-confidence-filter baseline** (held-out): 35%


### Held-out (honest - never seen during training)

| Tier | Precision | n |
|---|---|---|
| Top third | 58% | 19 |
| Middle third | 32% | 19 |
| Bottom third | 16% | 19 |

### Full set (reference only, includes training data - baseline 35%)

| Tier | Precision | n |
|---|---|---|
| Top third | 61% | 76 |
| Middle third | 29% | 76 |
| Bottom third | 16% | 76 |

## Summarization quality (secondary signals - see module docstrings)

n = 20 gold meetings (LLM arm)

| Metric | Average |
|---|---|
| ROUGE-L (cheap regression check only) | 0.199 |
| BERTScore F1 (distilbert, cheap regression check only) | 0.782 |
| LLM-judge coherence (1-5, same model self-judging - weak signal) | 2.95 |
| LLM-judge consistency (1-5, same model self-judging - weak signal) | 3.45 |
| LLM-judge fluency (1-5, same model self-judging - weak signal) | 3.35 |
| LLM-judge relevance (1-5, same model self-judging - weak signal) | 2.85 |
