# MONAI WhatsApp Radiograph Reproducibility Pipeline

Reproducible research pipeline for paired evaluation of original and WhatsApp-compressed panoramic radiographs in a binary third-molar–mandibular-canal proximity task.

## Project status

Milestone 3 provides:

- a privacy-preserving paired-image manifest;
- strict validation and patient-grouped stratified folds;
- safe 2-D grayscale image loading;
- condition-specific and paired datasets;
- MONAI preprocessing and training-only augmentation;
- a model factory for ResNet, DenseNet, and EfficientNet;
- synthetic non-clinical data for smoke tests;
- nested one-fold training with AdamW, BCEWithLogitsLoss, and early stopping;
- refitting on the complete outer-training partition;
- checkpoint, history, metrics, and paired prediction export;
- similarity-based candidate generation for unnamed image folders;
- automated tests.

Five-fold orchestration, paired confidence intervals, calibration analysis, and manuscript figures remain future milestones. No clinical images or direct identifiers belong in this repository.

## Installation

```bash
git clone https://github.com/bbvizzotto-run/monai-wpp-reproducible.git
cd monai-wpp-reproducible
python -m venv .venv
```

Activate the environment and install either the lightweight validation tools:

```bash
pip install -e ".[dev]"
```

or the complete research environment:

```bash
pip install -e ".[all]"
```

Dependency ranges are provisional. Exact versions will be frozen before definitive experiments.

## Data model

Create a local CSV based on `data/metadata.example.csv`:

```csv
case_id,patient_id,original_path,whatsapp_path,label,side
case_001,patient_001,data/private/original/image_001.jpg,data/private/whatsapp/image_a.jpg,1,right
```

`patient_id` is mandatory because folds must be grouped by patient. If left and right third molars are separate observations, use one row per side but keep the same `patient_id`.

## Generate a non-clinical smoke-test dataset

```bash
python scripts/generate_synthetic_data.py --output data/synthetic --n-cases 20
python scripts/validate_dataset.py --manifest data/synthetic/metadata.csv --root data/synthetic
```

Synthetic images are geometric test patterns and must never be interpreted as clinical examples.

## Create grouped folds

```bash
python scripts/create_folds.py \
  --manifest data/synthetic/metadata.csv \
  --root data/synthetic \
  --output outputs/folds.synthetic.csv \
  --n-splits 5 \
  --seed 42
```

Every retained fold must contain both classes, and all rows from the same patient must remain in one fold.

## Train one outer fold

The outer test fold is never used for early stopping or model selection. The remaining outer-training cases are divided into grouped inner folds. Early stopping selects an epoch count using original images from one inner validation fold. The model is then reinitialized, trained for that fixed epoch count on the complete outer-training partition, and evaluated on paired original and WhatsApp images from the untouched outer test fold.

```bash
python scripts/train_fold.py \
  --manifest outputs/folds.synthetic.csv \
  --root data/synthetic \
  --fold 0 \
  --output-dir outputs/training \
  --architecture resnet18 \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 0.0001 \
  --patience 3 \
  --inner-splits 4 \
  --inner-fold 0 \
  --device auto
```

For a quicker CPU-only smoke test, add:

```bash
--epochs 2 --image-size 96 96
```

The command creates:

```text
outputs/training/fold_0/selection/best_model.pt
outputs/training/fold_0/selection/best_model.json
outputs/training/fold_0/selection/history.csv
outputs/training/fold_0/best_model.pt
outputs/training/fold_0/refit_history.csv
outputs/training/fold_0/predictions.csv
outputs/training/fold_0/metrics.json
```

Results obtained from the synthetic dataset are only software smoke tests and have no clinical or scientific interpretation.

## Suggest pairs for unnamed folders

```bash
python scripts/suggest_pairs.py \
  --original-dir /protected/path/original \
  --whatsapp-dir /protected/path/whatsapp \
  --output outputs/pair_candidates.csv \
  --top-k 3
```

The similarity tool only ranks candidates. A qualified researcher must manually confirm every final original/WhatsApp pairing.

## Experiment design

The primary experiment is `train_original_evaluate_paired`:

1. the outer test fold is isolated before any model selection;
2. epoch selection uses only original images from grouped inner training and validation partitions;
3. the final model is refit on every case in the outer-training partition;
4. paired original and WhatsApp images from the untouched outer test fold are evaluated with the same model and threshold;
5. case-level out-of-fold probabilities will support paired statistical tests.

A later secondary experiment may train separate models for each condition, but its interpretation must remain distinct.

## Run tests

```bash
pytest
```

Tests requiring PyTorch or MONAI are skipped when optional training dependencies are absent.

## Research safeguards

- Never commit radiographs, direct identifiers, clinical metadata, or unapproved derived data.
- Split by `patient_id`, never by image, crop, tooth, or augmented sample.
- Apply random augmentation only to training partitions.
- Keep original and WhatsApp counterparts in the same fold.
- Never use the outer test fold for early stopping, threshold selection, or hyperparameter selection.
- Never use similarity scores as final pair labels without manual confirmation.
- Generate final tables and figures directly from saved out-of-fold predictions.

## License

MIT License. See `LICENSE`.
