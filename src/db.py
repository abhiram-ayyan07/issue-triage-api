"""
SQL logging for prediction requests. Uses SQLAlchemy against SQLite by
default (models/predictions.db) so the project runs with zero external
services; set DATABASE_URL to point at Postgres/MySQL/etc. in production
(e.g. on the AWS deploy, see docs/aws_deploy_runbook.md).
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_DB_URL = f"sqlite:///{MODEL_DIR / 'predictions.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(String, nullable=True)
    predicted_label: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DATABASE_URL, echo=False, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def log_prediction(title: str, body: str | None, predicted_label: str, confidence: float) -> None:
    engine = get_engine()
    with Session(engine) as session:
        session.add(
            PredictionLog(
                title=title,
                body=body,
                predicted_label=predicted_label,
                confidence=confidence,
            )
        )
        session.commit()


def recent_predictions(limit: int = 20) -> list[dict]:
    engine = get_engine()
    with Session(engine) as session:
        rows = (
            session.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "title": r.title,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
