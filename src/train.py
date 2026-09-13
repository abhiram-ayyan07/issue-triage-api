"""
Trains the baseline issue-triage classifier: a hashing-based text vectorizer
+ TF-IDF reweighting + linear SVM (via SGDClassifier with hinge loss, which
supports predict_proba-style scores through calibration), saved as a single
scikit-learn Pipeline artifact.

This is the "Day 3: baseline classifier" step. The transformer upgrade
(DistilBERT fine-tuning) is a separate, heavier follow-on step
(see docs/upgrade_transformer.md) that needs real-scale data and a GPU,
which is why it's tracked as a later milestone rather than part of the
baseline.

Why HashingVectorizer instead of TfidfVectorizer: TfidfVectorizer (and
CountVectorizer under the hood) builds an explicit in-memory dictionary of
every unique word/word-pair it sees, then sorts and trims it. On the small
bundled sample data that's instant. On the real ~1.2M-row NLBSE'23 dataset
with word-pairs (ngram_range=(1, 2)) enabled, that raw dictionary can balloon
into tens of millions of entries before it's trimmed down, and sorting that
many entries in a plain Python loop is what made training look "stuck" for
several minutes (it wasn't frozen, just slow). HashingVectorizer sidesteps
this: it maps tokens straight into a fixed-size array via a hash function,
with no dictionary-building or sorting step at all, so its runtime doesn't
blow up with corpus size the same way.

Usage:
    python -m src.train                     # trains on the bundled sample data
    python -m src.train --train-csv path/to/real/train.csv --test-csv path/to/real/test.csv
    python -m src.train --sample-size 100000  # quick run on a random subset, for a fast sanity check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
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
                "hashing",
                HashingVectorizer(
                    n_features=2**18,
                    ngram_range=(1, 2),
                    stop_words="english",
                    alternate_sign=False,
                    norm=None,
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            (
                "clf",
                CalibratedClassifierCV(
                    SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, random_state=42),
                    cv=3,
                ),
            ),
        ]
    )


def train(
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    test_csv: str | Path = DEFAULT_TEST_CSV,
    sample_size: int | None = None,
) -> dict:
    t0 = time.time()
    print(f"Loading data from {train_csv} and {test_csv} ...", flush=True)
    train_ds = load_dataset(train_csv)
    test_ds = load_dataset(test_csv)

    if sample_size and sample_size < len(train_ds.text):
        idx = train_ds.text.sample(n=sample_size, random_state=42).index
        train_ds.text, train_ds.label = train_ds.text.loc[idx], train_ds.label.loc[idx]
        print(f"Subsampled training set down to {sample_size} rows for a quick run.", flush=True)

    print(
        f"Loaded {len(train_ds.text)} training rows and {len(test_ds.text)} test rows "
        f"in {time.time() - t0:.1f}s.",
        flush=True,
    )

    pipeline = build_pipeline()

    print("Fitting pipeline (vectorizing + training the classifier)... this is the slow part, "
          "please wait for it to finish rather than interrupting.", flush=True)
    t1 = time.time()
    pipeline.fit(train_ds.text, train_ds.label)
    print(f"Fit done in {time.time() - t1:.1f}s.", flush=True)

    print("Evaluating on the test set...", flush=True)
    preds = pipeline.predict(test_ds.text)
    report = classification_report(test_ds.label, preds, output_dict=True, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print(classification_report(test_ds.label, preds, zero_division=0))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optionally train on a random subset of this many rows, for a fast sanity-check "
        "run before committing to a full multi-hour pass on the real dataset.",
    )
    args = parser.parse_args()
    train(args.train_csv, args.test_csv, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
