# Upgrade path: DistilBERT fine-tune

The baseline (`src/train.py`) is TF-IDF + a calibrated linear SVM — fast to
train, no GPU needed, good enough to prove the whole system works end to
end. The planned upgrade is a fine-tuned DistilBERT classifier, which should
meaningfully beat the baseline on the real (imbalanced, noisy) dataset,
especially on the minority classes (`question`, `documentation`).

This is intentionally scoped as a separate follow-on step because it needs:

1. **The real dataset at scale.** Fine-tuning on the ~800-row synthetic
   sample would just memorize the template sentences (which is exactly what
   the baseline already does — see the suspiciously perfect metrics in
   `models/baseline_metrics.json`). This step only makes sense once
   `data/download_dataset.sh` has been run somewhere with real network
   access.
2. **A GPU.** Fine-tuning DistilBERT on ~1.2M rows on CPU is impractical.
   Use a spot GPU instance (e.g. AWS `g4dn.xlarge`, or a free Colab/Kaggle
   GPU for a first pass) rather than the same EC2 box serving the API.

## Planned approach

- `transformers` + `datasets` (Hugging Face), `distilbert-base-uncased` as
  the base checkpoint.
- Tokenize `title + " " + body`, truncate to 256 tokens (issue titles/bodies
  are usually short; check the real token-length distribution first).
- Fine-tune with class weighting or a weighted loss to account for the
  ~52/37/6/4.4 class imbalance — plain accuracy will look great and hide
  poor recall on `question`/`documentation` otherwise. Track macro-F1, not
  just accuracy.
- Export with `torch.jit` or ONNX for faster CPU inference in the serving
  API, or keep it on a GPU-backed endpoint if latency allows.
- Swap `src/model.py`'s `predict()` to load the transformer artifact behind
  the same interface, so `src/api.py` doesn't need to change — the baseline
  and the upgraded model should be drop-in compatible from the API's point
  of view. Consider keeping both behind a feature flag so you can A/B or
  fall back to the fast baseline.

## Suggested resume framing once this lands

"...and iterated from a TF-IDF baseline to a fine-tuned DistilBERT model,
improving macro-F1 by X points, particularly on the minority
question/documentation classes" is a much stronger bullet than just
"trained a classifier" — but only write the specific number once you've
actually measured it on the real held-out test set.
