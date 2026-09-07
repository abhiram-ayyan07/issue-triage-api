"""
FastAPI service for the GitHub Issue Triage classifier.

Endpoints:
    GET  /health           -> liveness check
    POST /predict          -> classify a single issue (title + optional body)
    GET  /predictions/recent -> last N predictions logged to SQL

Run locally:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.db import log_prediction, recent_predictions
from src.model import ModelNotTrainedError, predict

app = FastAPI(
    title="GitHub Issue Triage API",
    description="Classifies GitHub issues into bug / feature / question / documentation.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Issue title")
    body: str = Field("", description="Issue body (optional)")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_issue(req: PredictRequest) -> PredictResponse:
    try:
        result = predict(req.title, req.body)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    log_prediction(
        title=req.title,
        body=req.body,
        predicted_label=result["label"],
        confidence=result["confidence"],
    )
    return PredictResponse(**result)


@app.get("/predictions/recent")
def get_recent_predictions(limit: int = 20) -> list[dict]:
    return recent_predictions(limit=limit)
