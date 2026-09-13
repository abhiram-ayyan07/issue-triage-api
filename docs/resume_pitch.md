# Resume bullets + project pitch

## Real results (trained on the full dataset, 2026-09-13)

Trained on the real NLBSE'23 dataset (~1.08M training rows after cleaning,
evaluated on the full 142,320-row held-out test set):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bug | 0.84 | 0.91 | 0.87 | 74,781 |
| feature | 0.79 | 0.84 | 0.82 | 52,797 |
| documentation | 0.74 | 0.39 | 0.51 | 6,252 |
| question | 0.73 | 0.20 | 0.31 | 8,490 |
| **Overall accuracy** | | | **81.8%** | 142,320 |

Read: the model is strong on the two majority classes (bug, feature — 90%
of the data) and weak on the two minority classes (documentation, question
— the other 10%), especially recall — it misses most actual questions and
a majority of actual documentation issues, because a baseline linear model
trained on imbalanced data defaults to guessing the common classes. This
is a normal, expected first-baseline result, and exactly the motivation
for the class-weighting/oversampling and DistilBERT upgrade tracked in
`docs/upgrade_transformer.md` — "identified and explained a class-imbalance
weakness, with a concrete plan to fix it" is a stronger interview answer
than a suspiciously perfect model.

## Resume bullets (pick 2-3 depending on space)

- Built and deployed a GitHub issue-triage classifier API (Python, FastAPI,
  scikit-learn, Docker, AWS EC2) that labels incoming issues as bug,
  feature, question, or documentation from title/body text, trained on the
  1.2M-row NLBSE'23 issue-classification benchmark, reaching 81.8% overall
  accuracy on a 142K-row held-out test set.
- Designed a full ML service pipeline — data ingestion, a hashing-vectorizer
  + calibrated linear SVM model trained at full dataset scale, model
  serialization, and a versioned REST API — with SQL-backed
  request/prediction logging for observability.
- Diagnosed a class-imbalance weakness in the baseline (74% precision but
  only 20% recall on the rarest class) from the confusion matrix, and
  scoped a concrete remediation plan (class weighting, then a fine-tuned
  DistilBERT upgrade) rather than stopping at a single headline accuracy
  number.
- Containerized the service with Docker and wrote a reproducible AWS EC2
  deployment runbook; added a GitHub Actions CI pipeline running lint, unit
  tests, and a Docker build on every push.
- Wrote unit and integration tests (pytest) covering the data pipeline,
  model inference, and API endpoints, keeping the service deployable with
  a single command.

## 30-second pitch (for interviews / cover letters)

"I built an API that automatically triages GitHub issues — given a title
and body, it predicts whether it's a bug report, a feature request, a
question, or a documentation issue. It's trained on the real NLBSE'23
benchmark, about 1.2 million labeled issues, and gets 82% accuracy overall
— though I noticed it's noticeably weaker on the rarer categories like
questions and documentation, since they only make up about 10% of the
data combined, which is a classic class-imbalance problem. That's actually
what I want to tackle next, either with class weighting or by upgrading to
a fine-tuned transformer model. It's a FastAPI service backed by the
classifier, with SQL logging of every prediction so you can audit what
the model is doing in production. It's containerized with Docker, has a
CI pipeline that lints, tests, and builds the image on every push, and I
wrote up a runbook for deploying it to a single EC2 instance. I used it as
a way to get hands-on with the full lifecycle of shipping an ML-backed
service — not just training a model, but building the API layer, logging,
tests, containerization, and deployment around it, plus actually
diagnosing where the model falls short instead of stopping at one
accuracy number — since that's the kind of end-to-end ownership I want in
a new-grad SWE/ML role."

## Notes on how the project was actually built (for your own reference)

- The pipeline and training script were first built and validated against
  a small synthetic sample dataset with the same schema/label set
  (`data/generate_sample_data.py`), since the real dataset's host wasn't
  reachable from the original build sandbox — that's why an earlier
  version of this doc had placeholder metrics.
- The model has since been retrained on the real NLBSE'23 dataset
  (~1.2M rows) on your own machine, and the "Real results" section above
  reflects that actual run — those are the numbers to use in interviews
  and on your resume, not the earlier sample-data ones.
- Docker was not build/run-tested in the original build sandbox (no Docker
  daemon available there) — validate `docker build .` and `docker run`
  locally or in CI before relying on it for a live demo, if you haven't
  already.
