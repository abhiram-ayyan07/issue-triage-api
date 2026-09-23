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

## Status: done — trained and evaluated on the real dataset

Trained on a 120,000-row subsample of the real training set (2 epochs, on
a laptop GTX 1650), evaluated on the full 142,320-row held-out test set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bug | 0.92 | 0.89 | 0.90 |
| feature | 0.87 | 0.87 | 0.87 |
| documentation | 0.62 | 0.68 | 0.65 |
| question | 0.52 | 0.66 | 0.58 |
| **Accuracy** | | | **85.7%** |
| **Macro F1** | | | **0.75** |

Versus the baseline: accuracy 81.8% -> 85.7%, macro-F1 0.629 -> 0.75, and
critically, recall on the two classes this upgrade targeted improved a lot:
`question` 0.20 -> 0.66, `documentation` 0.39 -> 0.68. See
`docs/resume_pitch.md` for the full before/after table and resume framing.

A 120K-row subsample (not the full ~1.08M rows) was used deliberately —
see "Choosing a training set size" below for why, and the real timing.

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
`--max-length`, `--lr`, and `--fp16` are all adjustable — see `python -m
src.train_transformer --help`. If you hit a CUDA out-of-memory error,
lower `--batch-size` first (e.g. 8 or 4).

## Choosing a training set size

The full dataset is ~1.08M rows. On a GTX 1650, once the padding/fp16
fixes below were in place, a 20K-row/1-epoch sanity check took about 1h47m
(down from 6.6 hours before the fixes) at ~3.0s/training-step. Scaling that
up linearly, the full dataset at the default 2 epochs would take roughly
4-5 days of continuous training — impractical for a laptop GPU on a normal
timeline. A bounded subsample is the practical middle ground:

```powershell
python -m src.train_transformer --train-csv data\nlbse23-issue-classification-train.csv --test-csv data\nlbse23-issue-classification-test.csv --sample-size 120000 --epochs 2
```

120,000 rows / 2 epochs took about 13-14 hours (an overnight run) and got
6x more training data than the 20K sanity check — see the results above.
Note that end-of-epoch evaluation always runs against the *full* 142,320-row
test set regardless of the training subsample size, so there's a fixed
~40-45 minutes of eval time per epoch no matter how small the training
subsample is.

**If your machine might restart or sleep mid-run** (Windows updates, a
laptop lid closing, etc.): `save_strategy="epoch"` means a checkpoint is
written to `models/distilbert_classifier/checkpoints/` after every
completed epoch, and the *final* model files are only written after all
epochs plus the final evaluation finish. So if a run gets interrupted:
check the checkpoints folder for how far it got (the folder name,
`checkpoint-<N>`, is the training step number) and check whether
`models/distilbert_classifier/config.json` and `metrics.json` exist and
are newer than the last checkpoint — if so, the run actually completed
before the interruption. If it was cut off mid-run with no completed final
model, the current script doesn't auto-resume from a checkpoint — it would
need `--resume-from-checkpoint` support added, or just be restarted from
scratch. Worth asking for that if you hit an actual mid-run interruption.

## If it's running much slower than expected

A 20k-row/1-epoch sanity check should take minutes on a real GPU, not
hours. If it's taking hours, check these in order before assuming your GPU
is just slow:

1. **Is the GPU actually being used, or just sitting idle?** In a second
   terminal, run `nvidia-smi -l 2` while training is in the "Fine-tuning..."
   phase and watch it refresh every 2s. Look at `Pwr` (should be well above
   the idle wattage, not sitting at a few watts) and the performance state
   (`P0`-`P2` = active, `P8` = idle/low-power). If it stays at `P8` and near
   0% GPU-Util the whole time, the GPU isn't actually being exercised —
   check:
   - **Windows power plan**: set it to "High performance" (or "Best
     Performance" on newer Windows), not "Balanced" or "Power saver".
   - **Laptop plugged in**: running on battery throttles GPU boost clocks
     on most laptops.
   - **NVIDIA Control Panel** → Manage 3D Settings → Power management mode
     → "Prefer maximum performance" (both globally and for `python.exe`
     specifically, if it's listed under Program Settings).
2. **`--fp16` is off by default.** Mixed precision (fp16) only reliably
   speeds things up on GPUs with Tensor Cores (RTX 20-series and newer,
   or Volta+ datacenter GPUs). On older/consumer GPUs without them — e.g.
   the GTX 16-series (1650/1660), which is Turing but specifically the
   variant *without* Tensor Cores — fp16 can be a wash or even slightly
   slower due to cast/loss-scaling overhead, so it's opt-in via `--fp16`
   rather than automatic. Try a short run with and without it and compare;
   don't assume it'll help.
3. **Padding**: batches are padded dynamically to the longest sequence in
   *each mini-batch* (`DataCollatorWithPadding`), not to a fixed length.
   If you're comparing timings against an older run of this script from
   before this was fixed, expect the old numbers to be noticeably slower
   for the same data — the old version padded every example in the entire
   dataset out to the length of the single longest example (up to
   `--max-length`), so on real GitHub issue text (where a lot of bodies
   hit the truncation cap) almost every batch was needlessly running at
   the full max length.

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

## Resume framing

See `docs/resume_pitch.md` for the full bullets and pitch using the real
numbers above — "iterated from a hashing-vectorizer + linear SVM baseline
(81.8% accuracy) to a fine-tuned DistilBERT model with class-weighted loss,
raising accuracy to 85.7% and more than tripling recall on the rarest
class (question: 20% -> 66%)" is a much stronger bullet than either model
alone, since it shows diagnosing a specific weakness and iterating toward
fixing it, not just training a model once.
