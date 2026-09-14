# Upgrade path: DistilBERT fine-tune

The baseline (`src/train.py`) is a hashing vectorizer + a calibrated linear
SVM — fast to train, no GPU needed, and it proved the whole system works
end to end. Trained on the real data, it got **81.8% overall accuracy**,
but was noticeably weaker on the rare classes: only 0.20 recall on
`question` and 0.39 recall on `documentation` (see `docs/resume_pitch.md`
for the full breakdown), because those two classes together make up only
about 10% of the data. `src/train_transformer.py` (implemented) fine-tunes
a DistilBERT classifier with class-weighted loss to specifically target
that weakness, rather than chasing overall accuracy alone.

## Status: implemented, not yet run on the real dataset

The training script, the class-weighted loss, and the API integration are
all done and pass an offline logic check (see below). What's still
pending is actually running it against the real ~1.2M-row dataset on a
GPU machine and recording the real results — do that next, then update
this doc and `docs/resume_pitch.md` with the actual before/after numbers.

## Why this needs its own requirements file and a GPU

`torch` + `transformers` + `accelerate` (in `requirements-transformer.txt`,
kept separate from `requirements.txt` since together they're a multi-GB
install and the baseline API doesn't need them) and a CUDA-capable GPU.
Fine-tuning DistilBERT on the full dataset on CPU alone is impractical —
use `--sample-size` for a fast sanity check on CPU, but the full run needs
a real GPU.

```powershell
pip install -r requirements-transformer.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If that prints `False` on a machine with an NVIDIA GPU, the installed
`torch` build doesn't have CUDA support — see
https://pytorch.org/get-started/locally/ for the right install command for
your CUDA version, and reinstall.

## Running it

```powershell
# quick sanity check first (a few minutes, even on CPU)
python -m src.train_transformer --train-csv data\nlbse23-issue-classification-train.csv --test-csv data\nlbse23-issue-classification-test.csv --sample-size 20000 --epochs 1

# then the real run, once the sanity check looks reasonable
python -m src.train_transformer --train-csv data\nlbse23-issue-classification-train.csv --test-csv data\nlbse23-issue-classification-test.csv
```

`--sample-size` (subsample the training set), `--epochs`, `--batch-size`,
`--max-length`, and `--lr` are all adjustable — see `python -m
src.train_transformer --help`. If you hit a CUDA out-of-memory error,
lower `--batch-size` first (e.g. 8 or 4).

## What it does differently from the baseline

- Tokenizes `title + " " + body` with the real `distilbert-base-uncased`
  WordPiece tokenizer, truncated to 256 tokens.
- Computes class weights from the actual training label distribution
  (`sklearn.utils.class_weight.compute_class_weight("balanced", ...)`) and
  applies them via a weighted cross-entropy loss (`WeightedLossTrainer` in
  `src/train_transformer.py`), so the model is explicitly penalized more
  for getting the rare classes wrong instead of being allowed to default
  to guessing the common ones.
- Tracks macro-F1 (the average F1 across all four classes equally) as the
  model-selection metric, not plain accuracy — accuracy alone would look
  great while still hiding poor performance on the rare classes, which is
  exactly the failure mode this upgrade is trying to fix.
- Saves to `models/distilbert_classifier/` in the standard Hugging Face
  format (`config.json`, `model.safetensors`, tokenizer files).

## How the API picks it up automatically

`src/model.py` checks whether `models/distilbert_classifier/config.json`
exists. If it does, the API serves predictions from the fine-tuned
transformer; otherwise it falls back to the baseline scikit-learn model.
`src/api.py` didn't need to change at all — both backends return the same
`{label, confidence, scores}` shape. This means: run
`train_transformer.py` once, and the next time you start `uvicorn
src.api:app`, it's automatically using the upgraded model — no code
changes, no flag to flip.

## Suggested resume framing once you have real numbers

"...iterated from a hashing-vectorizer + linear SVM baseline (81.8%
accuracy) to a fine-tuned DistilBERT model with class-weighted loss,
improving macro-F1 by X points and recall on the rarest class
(`question`) from 20% to Y%" is a much stronger bullet than either model
alone — it shows you can diagnose a specific weakness (not just "train a
model") and iterate toward fixing it. Only write the specific X/Y numbers
in once you've actually run this on the real held-out test set.
