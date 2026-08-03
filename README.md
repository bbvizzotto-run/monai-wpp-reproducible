# MONAI WhatsApp Radiograph Reproducibility Pipeline

Reproducible research pipeline for paired evaluation of original and WhatsApp-compressed panoramic radiographs in a binary third-molar–mandibular-canal proximity task.

## Project status

This repository is being rebuilt from the ground up. The current milestone provides:

- a privacy-preserving dataset manifest specification;
- strict manifest validation;
- patient-grouped, stratified cross-validation folds;
- basic binary-classification metrics;
- automated tests and continuous integration.

Training, paired statistical inference, figure generation, and the final manuscript tables will be added in subsequent milestones. No clinical images or patient identifiers belong in this repository.

## Data model

Create a local CSV manifest based on `data/metadata.example.csv`:

```csv
case_id,patient_id,original_path,whatsapp_path,label,side
case_001,patient_001,data/private/original/image_001.jpg,data/private/whatsapp/image_a.jpg,1,right
case_002,patient_002,data/private/original/image_002.jpg,data/private/whatsapp/image_b.jpg,0,left
```

Required columns:

- `case_id`: anonymous unique case identifier;
- `patient_id`: anonymous patient/group identifier used to prevent leakage;
- `original_path`: path to the original radiograph;
- `whatsapp_path`: path to the paired WhatsApp-compressed image;
- `label`: reference-standard class (`0` or `1`).

The optional `side` column can be used when left and right third molars are separate cases.

## Installation

```bash
git clone https://github.com/bbvizzotto-run/monai-wpp-reproducible.git
cd monai-wpp-reproducible
python -m venv .venv
```

Activate the environment and install the project:

```bash
pip install -e ".[dev]"
```

The dependency ranges are provisional during the reconstruction phase. Exact versions will be frozen before the definitive experiments.

## Validate a manifest

```bash
python scripts/validate_dataset.py \
  --manifest data/metadata.example.csv \
  --root . \
  --skip-file-check
```

Remove `--skip-file-check` when validating the real local dataset.

## Create grouped cross-validation folds

```bash
python scripts/create_folds.py \
  --manifest data/metadata.example.csv \
  --output outputs/folds.csv \
  --n-splits 2 \
  --seed 42 \
  --skip-file-check
```

For the final dataset, the planned default is five patient-grouped stratified folds, subject to verification of the number of independent patients and class distribution.

## Run tests

```bash
pytest
```

## Research safeguards

- Never commit radiographs, direct identifiers, clinical metadata, or unapproved derived data.
- Split data by `patient_id`, not by image file, crop, tooth, or augmented sample.
- Apply augmentation only within the training partition.
- Preserve paired original/WhatsApp observations in the same fold.
- Generate manuscript tables and figures directly from saved out-of-fold predictions.

## License

MIT License. See `LICENSE`.
