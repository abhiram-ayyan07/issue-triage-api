"""
Loads the trained classifier and exposes a simple predict() API used by
src/api.py.

Auto-selects the best available model: if the DistilBERT upgrade has been
trained (models/distilbert_classifier/ exists, see src/train_transformer.py),
it's used; otherwise this falls back to the baseline scikit-learn pipeline
(models/baseline_classifier.joblib, see src/train.py). Either way, api.py
doesn't need to know or care which one is actually running -- both backends
return the same {label, confidence, scores} shape.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from src.train import MODEL_PATH

# Deliberately NOT imported from src.train_transformer: that module imports
# torch/transformers at the top level, which aren't installed by default
# (see requirements-transformer.txt) -- importing it here would force every
# consumer of src.model, including the baseline-only path, to have torch
# installed. Duplicating this one path constant keeps the baseline API
# usable with just requirements.txt.
TRANSFORMER_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert_classifier"
MAX_LENGTH = 256  # must match what src/train_transformer.py trained with


class ModelNotTrainedError(RuntimeError):
    pass


def _load_baseline_backend():
    if not Path(MODEL_PATH).exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    pipeline = joblib.load(MODEL_PATH)

    def predict_fn(text: str) -> dict:
        label = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        return {
            "label": label,
            "confidence": float(max(proba)),
            "scores": {cls: float(p) for cls, p in zip(classes, proba)},
        }

    return predict_fn


def _load_transformer_backend():
    # Imported lazily: torch/transformers are a heavy, optional dependency
    # (see requirements-transformer.txt) only needed if this model exists.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = model.config.id2label

    def predict_fn(text: str) -> dict:
        inputs = tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
        label = max(scores, key=scores.get)
        return {"label": label, "confidence": scores[label], "scores": scores}

    return predict_fn


@lru_cache(maxsize=1)
def _load_backend():
    transformer_config = Path(TRANSFORMER_DIR) / "config.json"
    if transformer_config.exists():
        return _load_transformer_backend()
    return _load_baseline_backend()


def predict(title: str, body: str = "") -> dict:
    backend = _load_backend()
    text = f"{title} {body}".strip()
    return backend(text)
