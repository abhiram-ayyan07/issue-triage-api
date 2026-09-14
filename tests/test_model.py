from pathlib import Path

from src import model as model_module
from src.model import predict


def test_predict_returns_valid_label():
    result = predict("App crashes on startup", "Stack trace attached, throws an exception")
    assert result["label"] in {"bug", "feature", "question", "documentation"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6


def test_predict_handles_empty_body():
    result = predict("Add dark mode support")
    assert result["label"] in {"bug", "feature", "question", "documentation"}


def test_backend_selection_falls_back_to_baseline_when_no_transformer(monkeypatch, tmp_path):
    """When the DistilBERT model hasn't been trained, _load_backend must pick the
    baseline scikit-learn model rather than trying (and failing) to import
    torch/transformers, which aren't installed in the default dev environment --
    they're an optional extra, see requirements-transformer.txt."""
    monkeypatch.setattr(model_module, "TRANSFORMER_DIR", tmp_path / "distilbert_classifier")
    model_module._load_backend.cache_clear()
    try:
        backend = model_module._load_backend()
        assert callable(backend)
    finally:
        model_module._load_backend.cache_clear()


def test_transformer_dir_constant_matches_train_transformer_output_dir():
    # src.train_transformer imports torch/transformers at module level, which
    # are an optional extra (requirements-transformer.txt) not installed in
    # the default CI environment -- skip there rather than failing.
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.train_transformer import OUTPUT_DIR

    assert Path(model_module.TRANSFORMER_DIR) == Path(OUTPUT_DIR)
