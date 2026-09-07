from src.model import predict


def test_predict_returns_valid_label():
    result = predict("App crashes on startup", "Stack trace attached, throws an exception")
    assert result["label"] in {"bug", "feature", "question", "documentation"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6


def test_predict_handles_empty_body():
    result = predict("Add dark mode support")
    assert result["label"] in {"bug", "feature", "question", "documentation"}
