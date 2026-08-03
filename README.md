# MONAI WhatsApp Radiograph Reproducibility Pipeline

Reproducible research pipeline for paired evaluation of original and WhatsApp-compressed panoramic radiographs in a binary third-molar–mandibular-canal proximity task.

## Project status

Milestone 2 provides:

- a privacy-preserving paired-image manifest;
- strict validation and patient-grouped stratified folds;
- safe 2-D grayscale image loading;
- condition-specific and paired datasets;
- MONAI preprocessing and training-only augmentation;
- a model factory for ResNet, DenseNet, and EfficientNet;
- synthetic non-clinical data for smoke tests;
- similarity-based candidate generation for unnamed image folders;
- basic diagnostic metrics and automated tests.

The definitive experiments, paired confidence intervals, model-training loop, out-of-fold prediction export, and manuscript figures remain future milestones. No clinical images or direct identifiers belong in this repository.

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
  --output outputs/folds.csv \
  --n-splits 5 \
  --seed 42
```

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

The first primary experiment is configured as `train_original_evaluate_paired`:

1. training and model selection use original images only within each training fold;
2. the held-out cases are evaluated in both original and WhatsApp conditions;
3. the same trained model, threshold, and held-out cases are used for both conditions;
4. case-level out-of-fold probabilities will support paired statistical tests.

A later secondary experiment may train separate models for each condition, but its interpretation must remain distinct.

## Run tests

```bash
pytest
```

Tests requiring MONAI are skipped when optional training dependencies are absent.

## Research safeguards

- Never commit radiographs, direct identifiers, clinical metadata, or unapproved derived data.
- Split by `patient_id`, never by image, crop, tooth, or augmented sample.
- Apply random augmentation only to training partitions.
- Keep original and WhatsApp counterparts in the same fold.
- Never use similarity scores as final pair labels without manual confirmation.
- Generate final tables and figures directly from saved out-of-fold predictions.

## License

MIT License. See `LICENSE`.
