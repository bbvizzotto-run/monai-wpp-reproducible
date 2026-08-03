# Changelog

## 0.3.0 - Milestone 3

- Added reproducible seeding for Python, NumPy, PyTorch, and MONAI.
- Added a binary training engine with BCEWithLogitsLoss and AdamW.
- Added validation ROC-AUC model selection and early stopping.
- Added checkpoint and epoch-history export.
- Added paired original/WhatsApp prediction and metric export for one held-out fold.
- Added an optional PyTorch smoke test for the training engine.

## 0.2.0 - Milestone 2

- Added safe 2-D grayscale image loading.
- Added condition-specific and paired datasets.
- Added deterministic evaluation and training-only MONAI transforms.
- Added a configurable model factory.
- Added synthetic paired-data generation.
- Added similarity-based pair-candidate generation.
- Added tests for loading, pairing, and optional MONAI components.

## 0.1.0 - Initial scaffold

- Added manifest validation, grouped folds, metrics, tests, and project documentation.
