# Resume bullets + project pitch

## Resume bullets (pick 2-3 depending on space)

- Built and deployed a GitHub issue-triage classifier API (Python, FastAPI,
  scikit-learn, Docker, AWS EC2) that labels incoming issues as bug,
  feature, question, or documentation from title/body text, trained on the
  1.2M-row NLBSE'23 issue-classification benchmark.
- Designed a full ML service pipeline — data ingestion, TF-IDF + calibrated
  linear SVM training, model serialization, and a versioned REST API — with
  SQL-backed request/prediction logging for observability.
- Containerized the service with Docker and wrote a reproducible AWS EC2
  deployment runbook; added a GitHub Actions CI pipeline running lint, unit
  tests, and a Docker build on every push.
- Wrote unit and integration tests (pytest) covering the data pipeline,
  model inference, and API endpoints, keeping the service deployable with
  a single command.

## 30-second pitch (for interviews / cover letters)

"I built an API that automatically triages GitHub issues — given a title
and body, it predicts whether it's a bug report, a feature request, a
question, or a documentation issue. It's a FastAPI service backed by a
scikit-learn text classifier, with SQL logging of every prediction so you
can audit what the model is doing in production. It's containerized with
Docker, has a CI pipeline that lints, tests, and builds the image on every
push, and I wrote up a runbook for deploying it to a single EC2 instance.
I used it as a way to get hands-on with the full lifecycle of shipping an
ML-backed service — not just training a model, but building the API layer,
logging, tests, containerization, and deployment around it — since that's
the kind of end-to-end ownership I want in a new-grad SWE/ML role."

## Notes on how the project was actually built (for your own reference)

- The real dataset (NLBSE'23 Issue Report Classification, ~1.2M labeled
  GitHub issues) is hosted on Azure Blob Storage, which isn't reachable
  from network-restricted build sandboxes. The pipeline, training script,
  and tests were built and validated against a small synthetic sample
  dataset with the *same schema and label set* (`data/generate_sample_data.py`)
  so the whole thing is provably wired together correctly.
- Before you demo or put this on your resume as "trained on 1.2M issues,"
  run `data/download_dataset.sh` from a machine with normal internet
  access, then retrain with `python -m src.train --train-csv <real train
  csv> --test-csv <real test csv>`. Swap in the real metrics from
  `models/baseline_metrics.json` for anything you state in an interview —
  the sample-data metrics are artificially perfect (the sample data reuses
  a small bank of template sentences per label) and are not representative
  of real-world performance.
- Docker was not build/run-tested in this environment (no Docker daemon
  available in this sandbox) — validate `docker build .` and
  `docker run` locally or in CI before relying on it for a live demo.
