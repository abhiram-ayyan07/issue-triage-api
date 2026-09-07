#!/usr/bin/env bash
# Downloads the real NLBSE'23 Issue Report Classification benchmark dataset.
#
# NOTE: this sandbox's outbound network is allowlisted and cannot reach the
# blob storage host these files live on, so this script can't be run from
# inside the dev container. Run it on your own machine (or an EC2 instance)
# where you have normal internet access.
#
# Dataset details (from https://github.com/nlbse2023/issue-report-classification):
#   - ~1.2M labeled issues (train) + 142,320 labeled issues (test)
#   - Columns: label, id, title, body, author_association
#   - Labels: bug (~52.5%), feature (~37%), question (~6%), documentation (~4.4%)

set -euo pipefail

OUT_DIR="$(dirname "$0")"
BASE_URL="https://tickettagger.blob.core.windows.net/datasets"

echo "Downloading train set..."
curl -L -o "$OUT_DIR/nlbse23-issue-classification-train.csv.tar.gz" \
  "$BASE_URL/nlbse23-issue-classification-train.csv.tar.gz"

echo "Downloading test set..."
curl -L -o "$OUT_DIR/nlbse23-issue-classification-test.csv.tar.gz" \
  "$BASE_URL/nlbse23-issue-classification-test.csv.tar.gz"

echo "Extracting..."
tar -xzf "$OUT_DIR/nlbse23-issue-classification-train.csv.tar.gz" -C "$OUT_DIR"
tar -xzf "$OUT_DIR/nlbse23-issue-classification-test.csv.tar.gz" -C "$OUT_DIR"

echo "Done. Point src/data_pipeline.py at the extracted CSVs (see --train-csv/--test-csv flags)."
