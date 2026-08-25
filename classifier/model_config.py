"""Fine-tuning configs for the action-item-sentence classifier.

DistilBERT is the default (laptop/single-GPU friendly, ~66M params).
bert-base is available as an optional comparison run - see docs/architecture.md.
"""

DISTILBERT = {
    "name": "distilbert-base-uncased",
    "batch_size": 16,
    "epochs": 3,
    "lr": 5e-5,
    "max_length": 64,
}

BERT_BASE = {
    "name": "bert-base-uncased",
    "batch_size": 8,
    "epochs": 3,
    "lr": 3e-5,
    "max_length": 64,
}

DEFAULT = DISTILBERT
