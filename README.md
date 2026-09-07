# GitHub Issue Triage API

Classifies incoming GitHub issues as `bug`, `feature`, `question`, or
`documentation` from their title/body text. Built as a flagship resume
project to demonstrate end-to-end ML service ownership: data pipeline,
model training, a REST API, SQL logging, containerization, CI, and a cloud
deployment runbook.

Dataset: [NLBSE'23 Issue Report Classification benchmark](https://github.com/nlbse2023/issue-report-classification)
(~1.2M labeled GitHub issues; labels bug/feature/question/documentation).

## Status

| Piece | Status |
|---|---|
| Dataset pipeline | Done ([src/data_pipeline.py](src/data_pipeline.py)) |
| Baseline classifier (TF-IDF + linear SVM) | Done ([src/train.py](src/train.py)) |
| FastAPI service | Done ([src/api.py](src/api.py)) |
| SQL prediction logging | Done ([src/db.py](src/db.py)) |
| Docker | Done ([Dockerfile](Dockerfile)) — not build-tested in this sandbox (no Docker daemon here); verify locally/in CI |
| CI (GitHub Actions) | Done ([.github/workflows/ci.yml](.github/workflows/ci.yml)) |
| AWS deploy runbook | Done ([docs/aws_deploy_runbook.md](docs/aws_deploy_runbook.md)) |
| Resume bullets + pitch | Done ([docs/resume_pitch.md](docs/resume_pitch.md)) |
| Real-scale training on the full dataset | **Not done** — needs a machine with unrestricted network access, see below |
| DistilBERT transformer upgrade | **Not done** — see [docs/upgrade_transformer.md](docs/upgrade_transformer.md) |

**Important:** everything above was built and tested against a small
synthetic sample dataset (`data/generate_sample_data.py`) with the same
schema/labels as the real data, because the real dataset is hosted on a
host this build environment's network can't reach. The pipeline is proven
correct end-to-end, but **retrain on the real data** (see below) before
using the metrics or the model in an actual demo or interview — see
[docs/resume_pitch.md](docs/resume_pitch.md) for details on why the sample
metrics look artificially perfect.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# generates data/sample_issues_{train,test}.csv
python data/generate_sample_data.py

# trains models/baseline_classifier.joblib
python -m src.train

# run tests
pytest -v

# run the API
uvicorn src.api:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "Stack trace attached"}'
```

## Training on the real dataset

This sandbox's network can't reach the file host, so run this part on a
machine with normal internet access:

```bash
bash data/download_dataset.sh
python -m src.train \
  --train-csv data/nlbse23-issue-classification-train.csv \
  --test-csv data/nlbse23-issue-classification-test.csv
```

## Docker

```bash
docker build -t issue-triage-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models issue-triage-api
```

## Project layout

```
src/            data pipeline, training, model loading, FastAPI app, SQL logging
data/           dataset download script + synthetic sample-data generator
models/         trained model artifact + metrics (gitignored, generated locally)
tests/          pytest suite (pipeline, model, API)
docs/           AWS runbook, resume pitch, transformer upgrade plan
.github/workflows/ci.yml   lint + test + Docker build on every push
```
