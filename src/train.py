"""
Trains the baseline issue-triage classifier: TF-IDF vectorizer + linear SVM
(via SGDClassifier with hinge loss, which supports predict_proba-style scores
through calibration), and saves both the vectorizer and model as a single
scikit-learn Pipeline artifact.

This is the "Day 3: baseline classifier" step. The transformer upgrade
(DistilBERT fine-tuning) is a separate, heavier follow-on step
(see docs/upgrade_transformer.md) that needs real-scale data and a GPU,
which is why it's tracked as a later milestone rather than part of the
baseline.

Usage:
    python -m src.train                     # trains on the bundled sample data
    python -m src.train --train-csv path/to/real/train.csv --test-csv path/to/real/test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.data_pipeline import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, load_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "baseline_classifier.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, random_state=42),
                    cv=3,
                ),
            ),
        ]
    )


def train(train_csv: str | Path = DEFAULT_TRAIN_CSV, test_csv: str | Path = DEFAULT_TEST_CSV) -> dict:
    train_ds = load_dataset(train_csv)
    test_ds = load_dataset(test_csv)

    pipeline = build_pipeline()
    pipeline.fit(train_ds.text, train_ds.label)

    preds = pipeline.predict(test_ds.text)
    report = classification_report(test_ds.label, preds, output_dict=True, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(classification_report(test_ds.label, preds, zero_division=0))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    args = parser.parse_args()
    train(args.train_csv, args.test_csv)


if __name__ == "__main__":
    main()
