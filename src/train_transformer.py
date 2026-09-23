"""
Fine-tunes a DistilBERT classifier on the issue-triage dataset -- the
upgrade path from the baseline (TF-IDF-style hashing vectorizer + linear
SVM, see src/train.py). Motivation: the baseline is strong on the majority
classes (bug, feature) but weak on the minority classes (question,
documentation) because of class imbalance (see docs/upgrade_transformer.md
and docs/resume_pitch.md for the real baseline numbers this is trying to
beat: 81.8% accuracy overall, but only 0.20 recall on `question`).

This needs a GPU to be practical. Fine-tuning a transformer on CPU over
anything beyond a tiny sample is extremely slow -- this script checks
torch.cuda.is_available() and warns loudly (but still runs) if there's no
GPU, so you can do a quick correctness check on a small --sample-size
before committing a GPU machine to the full run.

Handles class imbalance with class-weighted cross-entropy loss (computed
from the actual training label distribution), rather than relying on the
model to notice the imbalance on its own the way the baseline did.

Usage:
    # quick sanity check on the bundled small sample dataset (CPU is fine for this)
    python -m src.train_transformer

    # full run on the real dataset (needs a GPU for this to finish in reasonable time)
    python -m src.train_transformer --train-csv data\\nlbse23-issue-classification-train.csv --test-csv data\\nlbse23-issue-classification-test.csv

    # a faster first pass on a subset before committing to the full dataset
    python -m src.train_transformer --train-csv <real train csv> --test-csv <real test csv> --sample-size 200000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data_pipeline import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, LABELS, load_dataset

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert_classifier"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class IssueDataset(Dataset):
    """Wraps tokenized issue text + integer labels for the Trainer API.

    Deliberately does NOT pad here. `tokenizer(texts, padding=True, ...)`
    called once over the *whole* dataset pads every example out to the
    length of the single longest example in the entire dataset (capped at
    max_length by truncation) -- on real GitHub issue text, where a lot of
    bodies are long enough to hit the max_length cap, that means almost
    every training example ends up padded/truncated to a full max_length
    sequence, even one-line issue titles. That's wasted compute on every
    single batch. Instead, this leaves sequences at their natural
    (truncated) length and pads per-*batch* via a DataCollatorWithPadding
    in train() below, so a batch of short issues stays short.
    """

    def __init__(self, texts: list[str], labels: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedLossTrainer(Trainer):
    """A Trainer that applies class weights to the loss, so the rare classes
    (question, documentation) aren't drowned out by the common ones
    (bug, feature) the way they were in the baseline model."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, target_names=LABELS, output_dict=True, zero_division=0)
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}


def train(
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    test_csv: str | Path = DEFAULT_TEST_CSV,
    sample_size: int | None = None,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 2e-5,
    fp16: bool = False,
) -> dict:
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        if fp16:
            print(
                "fp16 mixed precision requested. Note: this only reliably speeds things up on "
                "GPUs with Tensor Cores (RTX/Volta+). On older/consumer GPUs without them (e.g. "
                "GTX 16-series), fp16 can be a wash or even slightly slower due to cast/loss-"
                "scaling overhead -- if a run with --fp16 isn't faster than without it, leave it off.",
                flush=True,
            )
    else:
        print(
            "WARNING: no CUDA GPU detected. Fine-tuning DistilBERT on CPU is very slow for "
            "anything but a small dataset -- if this is meant to be a full training run, stop "
            "and check that your GPU drivers / CUDA-enabled torch install are set up correctly "
            "(torch.cuda.is_available() should be True). Continuing anyway, e.g. for a small "
            "--sample-size sanity check.",
            flush=True,
        )

    t0 = time.time()
    print(f"Loading data from {train_csv} and {test_csv} ...", flush=True)
    train_ds = load_dataset(train_csv)
    test_ds = load_dataset(test_csv)

    if sample_size and sample_size < len(train_ds.text):
        idx = train_ds.text.sample(n=sample_size, random_state=42).index
        train_ds.text, train_ds.label = train_ds.text.loc[idx], train_ds.label.loc[idx]
        print(f"Subsampled training set to {sample_size} rows.", flush=True)

    print(
        f"Loaded {len(train_ds.text)} train rows, {len(test_ds.text)} test rows "
        f"in {time.time() - t0:.1f}s.",
        flush=True,
    )

    print(f"Loading tokenizer/model ({MODEL_NAME})... this downloads ~270MB the first time.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    print("Tokenizing...", flush=True)
    train_dataset = IssueDataset(train_ds.text.tolist(), train_ds.label.tolist(), tokenizer, max_length)
    test_dataset = IssueDataset(test_ds.text.tolist(), test_ds.label.tolist(), tokenizer, max_length)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array(LABELS), y=train_ds.label.tolist()
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(
        f"Class weights (to counter imbalance): {dict(zip(LABELS, class_weights.round(2)))}",
        flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        # fp16 is opt-in (--fp16), not automatic just because a GPU exists --
        # see the note printed above about Tensor-Core-less GPUs.
        fp16=has_gpu and fp16,
        report_to=[],
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    print("Fine-tuning... this is the slow part, let it run.", flush=True)
    t1 = time.time()
    trainer.train()
    print(f"Fine-tuning done in {time.time() - t1:.1f}s.", flush=True)

    print("Evaluating on the test set...", flush=True)
    preds_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(preds_output.predictions, axis=-1)
    pred_labels = [ID2LABEL[i] for i in pred_ids]
    true_labels = test_ds.label.tolist()
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(classification_report(true_labels, pred_labels, zero_division=0))
    print(f"Saved model to {OUTPUT_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Train on a random subset of this many rows -- recommended for a first pass on the "
        "real dataset (e.g. 200000) before committing to a multi-hour full run.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable fp16 mixed precision. Off by default -- only reliably helps on GPUs with "
        "Tensor Cores (RTX/Volta+); on older GPUs (e.g. GTX 16-series) it can be a wash or "
        "slightly slower. Try it and compare if you're not sure which camp your GPU is in.",
    )
    args = parser.parse_args()
    train(
        args.train_csv,
        args.test_csv,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
