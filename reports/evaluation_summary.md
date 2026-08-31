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
