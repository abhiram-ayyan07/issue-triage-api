from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint():
    resp = client.post(
        "/predict",
        json={"title": "Add dark mode support", "body": "It would be great to toggle a dark theme"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in {"bug", "feature", "question", "documentation"}
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_requires_title():
    resp = client.post("/predict", json={"body": "no title provided"})
    assert resp.status_code == 422


def test_recent_predictions_after_predict():
    client.post("/predict", json={"title": "How do I configure a custom API base URL?"})
    resp = client.get("/predictions/recent?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1
