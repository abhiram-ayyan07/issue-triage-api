"""
Cross-platform downloader for the real NLBSE'23 Issue Report Classification
benchmark dataset (works on Windows, macOS, Linux — no bash/curl/tar needed).

NOTE: this sandbox's outbound network is allowlisted and cannot reach the
blob storage host these files live on, so this script can't be run from
inside the dev container/sandbox. Run it on your own machine where you have
normal internet access.

Dataset details (from https://github.com/nlbse2023/issue-report-classification):
    - ~1.2M labeled issues (train) + 142,320 labeled issues (test)
    - Columns: label, id, title, body, author_association
    - Labels: bug (~52.5%), feature (~37%), question (~6%), documentation (~4.4%)

Usage:
    python data/download_dataset.py
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://tickettagger.blob.core.windows.net/datasets"
OUT_DIR = Path(__file__).resolve().parent

FILES = [
    "nlbse23-issue-classification-train.csv.tar.gz",
    "nlbse23-issue-classification-test.csv.tar.gz",
]


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        dest = OUT_DIR / filename
        download(url, dest)

        print(f"Extracting {dest}...")
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(OUT_DIR)

    print("Done. Point src/train.py --train-csv/--test-csv at the extracted CSVs, e.g.:")
    print(
        "  python -m src.train "
        "--train-csv data/nlbse23-issue-classification-train.csv "
        "--test-csv data/nlbse23-issue-classification-test.csv"
    )


if __name__ == "__main__":
    main()
