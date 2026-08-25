"""Fine-tunes a binary action-item-sentence classifier on the weakly-labeled
training data from prepare_training_data.py.

Usage: python3 classifier/train.py [--config distilbert|bert_base]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from classifier.model_config import BERT_BASE, DISTILBERT

DATA_PATH = Path("data/outputs/classifier/training_data.jsonl")
MODEL_OUT = Path("models/action_item_classifier")
CONFIGS = {"distilbert": DISTILBERT, "bert_base": BERT_BASE}


class SentenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(config: dict) -> None:
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=config["max_length"], return_tensors="pt"
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding=True, max_length=config["max_length"], return_tensors="pt"
    )

    train_ds = SentenceDataset(train_enc, train_labels)
    val_ds = SentenceDataset(val_enc, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(config["name"], num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Held-out validation metrics:", metrics)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default="distilbert")
    args = parser.parse_args()
    main(CONFIGS[args.config])
