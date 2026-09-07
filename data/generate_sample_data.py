"""
Generates a small, realistic *stand-in* dataset shaped exactly like the real
NLBSE'23 issue-report-classification benchmark (same columns, same label set,
similar class balance), so the rest of the pipeline can be built and tested
without network access to the real ~1.2M-row dataset.

Swap this out for the real data by running data/download_dataset.sh on a
machine with normal internet access, then pointing src/data_pipeline.py at
the extracted CSVs.

Columns: label, id, title, body, author_association
Labels:  bug (~52.5%), feature (~37%), question (~6%), documentation (~4.4%)
"""

import csv
import random
from pathlib import Path

random.seed(42)

BUG_TITLES = [
    "App crashes on startup after update",
    "NullPointerException when saving form",
    "Memory leak in background worker",
    "Incorrect total shown on checkout page",
    "Login fails with 500 error on Safari",
    "Race condition causes duplicate records",
    "Images fail to load on slow connections",
    "Dropdown menu freezes the UI",
    "Timezone offset is wrong for UTC-5 users",
    "Off-by-one error in pagination",
]
BUG_BODIES = [
    "Steps to reproduce: open the app, click submit, observe the stack trace in the console. Expected the form to save, instead it throws an exception.",
    "This started happening after upgrading to the latest release. Reverting to the previous version fixes it.",
    "I can consistently reproduce this on Chrome 128 and Firefox 130. Attached logs and a screenshot.",
    "Seems related to a recent change in the caching layer. The bug only shows up under concurrent requests.",
    "Environment: Ubuntu 22.04, Python 3.11. The traceback points to a null reference in the request handler.",
]

FEATURE_TITLES = [
    "Add dark mode support",
    "Support exporting reports as CSV",
    "Allow filtering the dashboard by date range",
    "Add keyboard shortcuts for common actions",
    "Support OAuth login with GitHub",
    "Add bulk delete for list items",
    "Allow custom themes via config file",
    "Add webhook support for status changes",
    "Support multi-language localization",
    "Add drag-and-drop reordering",
]
FEATURE_BODIES = [
    "It would be great if users could toggle a dark theme from the settings menu, similar to other tools we use.",
    "Right now there's no way to export data out of the app. Adding a CSV export button would unblock our reporting workflow.",
    "We'd like to filter results by a custom date range instead of only the preset options currently available.",
    "This would make power users much faster and matches a common request in the community forum.",
    "Happy to help scope this out or submit a PR if the maintainers are open to the idea.",
]

QUESTION_TITLES = [
    "How do I configure a custom API base URL?",
    "Is there a way to run this without Docker?",
    "What's the recommended way to handle auth tokens?",
    "Does this support Python 3.12?",
    "How can I contribute a translation?",
    "Is there a rate limit on the public API?",
]
QUESTION_BODIES = [
    "I looked through the docs but couldn't find guidance on this. Could someone point me in the right direction?",
    "Trying to set this up locally and hit a wall — is there a recommended setup guide beyond the README?",
    "Not sure if this is a bug or expected behavior, so asking here first before filing an issue.",
    "Would appreciate any pointers, happy to update the docs once I understand the right approach.",
]

DOC_TITLES = [
    "README is missing setup instructions for Windows",
    "Clarify the meaning of the `strict` config flag",
    "Add example for the batch endpoint",
    "Fix broken link in CONTRIBUTING.md",
    "Document environment variables required in production",
]
DOC_BODIES = [
    "The docs mention this flag but don't explain what it actually changes at runtime. A short example would help.",
    "New contributors keep hitting this because the setup section skips a required step. Proposing a doc update.",
    "Found a dead link while reading through the contributing guide, should point to the new discussions page.",
    "It'd help to have a concrete example of the request/response shape for this endpoint in the docs.",
]

AUTHOR_ASSOCIATIONS = ["NONE", "CONTRIBUTOR", "MEMBER", "OWNER", "COLLABORATOR"]

LABEL_BANK = {
    "bug": (BUG_TITLES, BUG_BODIES),
    "feature": (FEATURE_TITLES, FEATURE_BODIES),
    "question": (QUESTION_TITLES, QUESTION_BODIES),
    "documentation": (DOC_TITLES, DOC_BODIES),
}

# Matches the real dataset's approximate class balance.
LABEL_WEIGHTS = {"bug": 0.525, "feature": 0.37, "question": 0.06, "documentation": 0.044}


def _sample_row(row_id: int) -> dict:
    label = random.choices(list(LABEL_WEIGHTS), weights=list(LABEL_WEIGHTS.values()))[0]
    titles, bodies = LABEL_BANK[label]
    title = random.choice(titles)
    body = random.choice(bodies)
    return {
        "label": label,
        "id": row_id,
        "title": title,
        "body": body,
        "author_association": random.choice(AUTHOR_ASSOCIATIONS),
    }


def generate(n_train: int = 800, n_test: int = 200, out_dir: str = "data") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split, n in (("train", n_train), ("test", n_test)):
        rows = [_sample_row(i) for i in range(n)]
        file_path = out_path / f"sample_issues_{split}.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "id", "title", "body", "author_association"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {file_path}")


if __name__ == "__main__":
    generate()
