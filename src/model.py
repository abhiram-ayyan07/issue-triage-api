"""Loads the trained baseline classifier and exposes a simple predict() API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from src.train import MODEL_PATH


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_pipeline():
    if not Path(MODEL_PATH).exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict(title: str, body: str = "") -> dict:
    pipeline = _load_pipeline()
    text = f"{title} {body}".strip()
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = float(max(proba))
    scores = {cls: float(p) for cls, p in zip(classes, proba)}
    return {"label": label, "confidence": confidence, "scores": scores}
