# Changelog

## 0.5.0 - Milestone 5

- Added ImageNet transfer learning for ResNet-18, ResNet-50, and ResNet-101 through torchvision.
- Added deterministic grayscale adaptation by averaging pretrained RGB kernels in the first convolution.
- Added a frozen ResNet-18 outer-fold pilot runner with conservative augmentation.
- Added environment and pilot-configuration metadata export.
- Documented the private Colab pilot workflow.

## 0.4.0 - Milestone 4

- Added validation for original-only tooth-level clinical case metadata.
- Allowed bilateral teeth from the same patient to share one panoramic source path.
- Rejected cross-patient source duplication and tooth/side inconsistencies.
- Added deterministic patient-grouped clinical fold preparation.
- Added fold summaries, patient indexes, and WhatsApp transmission-log templates.
- Added tests for the private clinical preparation workflow.

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
