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
| Baseline classifier (hashing vectorizer + linear SVM) | Done ([src/train.py](src/train.py)) |
| FastAPI service | Done ([src/api.py](src/api.py)) |
| SQL prediction logging | Done ([src/db.py](src/db.py)) |
| Docker | Done ([Dockerfile](Dockerfile)) — not build-tested in this sandbox (no Docker daemon here); verify locally/in CI |
| CI (GitHub Actions) | Done ([.github/workflows/ci.yml](.github/workflows/ci.yml)) |
| AWS deploy runbook | Done ([docs/aws_deploy_runbook.md](docs/aws_deploy_runbook.md)) |
| Resume bullets + pitch | Done ([docs/resume_pitch.md](docs/resume_pitch.md)) |
| Real-scale training on the full dataset | **Done** — 81.8% accuracy on the full 142,320-row test set, see [docs/resume_pitch.md](docs/resume_pitch.md) |
| DistilBERT transformer upgrade | **Not done** — see [docs/upgrade_transformer.md](docs/upgrade_transformer.md), motivated by the class-imbalance weakness found in the real-data results |

**Note:** the pipeline was first built and tested against a small synthetic
sample dataset (`data/generate_sample_data.py`) since the real dataset's
host wasn't reachable from the original build sandbox — it's still there
for quick local iteration. The model has since been retrained on the real
NLBSE'23 data; see [docs/resume_pitch.md](docs/resume_pitch.md) for the
real metrics and what they mean before quoting numbers in an interview.

## Quickstart

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# generates data\sample_issues_{train,test}.csv
python data\generate_sample_data.py

# trains models\baseline_classifier.joblib
python -m src.train

# run tests
pytest -v

# run the API
uvicorn src.api:app --reload --port 8000
```

If `Activate.ps1` is blocked by PowerShell's execution policy, either run
PowerShell as Administrator once and do
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or activate with
`.venv\Scripts\activate.bat` from `cmd.exe` instead.

Then, in a second PowerShell window (the first one is busy running the
server):

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"title\": \"App crashes on startup\", \"body\": \"Stack trace attached\"}'
```

(Use `curl.exe`, not plain `curl` — PowerShell aliases `curl` to
`Invoke-WebRequest`, which takes different flags. Easier alternative: open
`http://localhost:8000/docs` in your browser for an interactive form to
test `/predict` with, no curl needed at all.)

### macOS / Linux / WSL / Git Bash

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
machine with normal internet access.

**Windows** — use the Python downloader (no bash/curl/tar needed):

```powershell
python data\download_dataset.py
python -m src.train `
  --train-csv data\nlbse23-issue-classification-train.csv `
  --test-csv data\nlbse23-issue-classification-test.csv
```

**macOS / Linux / WSL** — either the same Python script, or the bash version:

```bash
python data/download_dataset.py
# or: bash data/download_dataset.sh

python -m src.train \
  --train-csv data/nlbse23-issue-classification-train.csv \
  --test-csv data/nlbse23-issue-classification-test.csv
```

Heads up: the train file alone is a sizeable download (1.2M rows) and will
take a while depending on your connection — let it run.

## Docker

Requires Docker Desktop installed and running on Windows.

**Windows (PowerShell):**

```powershell
docker build -t issue-triage-api .
docker run -p 8000:8000 -v ${PWD}\models:/app/models issue-triage-api
```

**macOS / Linux:**

```bash
docker build -t issue-triage-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models issue-triage-api
```

## Project layout

```
src/            data pipeline, training, model loading, FastAPI app, SQL logging
data/           dataset download scripts (Windows-friendly .py + bash .sh) + synthetic sample-data generator
models/         trained model artifact + metrics (gitignored, generated locally)
tests/          pytest suite (pipeline, model, API)
docs/           AWS runbook, resume pitch, transformer upgrade plan
.github/workflows/ci.yml   lint + test + Docker build on every push
```
