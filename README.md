# MONAI WhatsApp Radiograph Reproducibility Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21894253.svg)](https://doi.org/10.5281/zenodo.21894253)

Reproducible research pipeline for paired evaluation of original and WhatsApp-compressed panoramic radiographs in a binary third-molar–mandibular-canal proximity task.

## Project status

Milestone 6 provides:

- a privacy-preserving paired-image manifest;
- validation for original-only tooth-level clinical metadata;
- strict patient-grouped stratified folds;
- support for bilateral teeth sharing one panoramic source image;
- fold summaries, patient indexes, and WhatsApp transmission-log templates;
- safe 2-D grayscale image loading;
- condition-specific and paired datasets;
- MONAI preprocessing and training-only augmentation;
- ImageNet-pretrained ResNet transfer learning with grayscale kernel adaptation;
- grouped inner epoch selection without outer-test leakage;
- refitting on the complete outer-training partition;
- paired original/WhatsApp evaluation on untouched outer folds;
- independent training and inner-split seeds for stability experiments;
- a resumable Windows runner for additional model seeds;
- multi-seed out-of-fold consolidation;
- patient-clustered bootstrap confidence intervals;
- cross-seed probability stability and consensus analysis;
- manuscript-oriented tables, figures, metadata, and reports;
- synthetic non-clinical tests.

Clinical images, direct identifiers, private manifests, checkpoints, prediction tables, and generated clinical reports must remain outside the public repository. Future work includes manuscript integration, external validation, and any separately prespecified model comparisons.

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

For statistical analysis without the training stack:

```bash
pip install -e ".[stats]"
```

Exact software versions used in definitive experiments should be preserved in the exported run metadata and manuscript.

## Data model

Create a local paired CSV based on `data/metadata.example.csv`:

```csv
case_id,patient_id,original_path,whatsapp_path,label,side
case_001,patient_001,data/private/original/image_001.jpg,data/private/whatsapp/image_a.jpg,1,right
```

`patient_id` is mandatory because folds must be grouped by patient. If left and right third molars are separate observations, use one row per side but keep the same `patient_id`.

Before WhatsApp images exist, create a private original-only CSV based on `data/clinical_cases.example.csv`:

```csv
case_id,patient_id,original_path,label,tooth,side
case_001_left,patient_001,data/private/original/patient_001.jpg,1,38,left
case_001_right,patient_001,data/private/original/patient_001.jpg,0,48,right
```

Two bilateral cases from the same patient may share one `original_path`. The same source path must never be shared by different patients.

## Prepare original-only clinical metadata

The clinical preparation command validates the tooth-level case table, creates deterministic patient-grouped folds, summarizes each fold, and can emit a patient index plus one WhatsApp transmission-log row per complete panoramic radiograph.

```bash
python scripts/prepare_clinical_folds.py \
  --cases /protected/project/cases_original_only.csv \
  --root /protected/project \
  --output /protected/project/clinical_cases_folds.csv \
  --summary-output /protected/project/fold_summary.csv \
  --patient-index-output /protected/project/patient_index.csv \
  --whatsapp-log-output /protected/project/whatsapp_transmission_log.csv \
  --n-splits 5 \
  --seed 42
```

These outputs contain private clinical metadata and should remain outside the public repository. Fold assignment must be frozen before definitive model training.

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

## Run seed-stability training on Windows

Use a fixed `SplitSeed` to preserve the grouped inner validation partition while changing the training and augmentation seed:

```powershell
.\scripts\run_seed_stability.ps1 `
  -Manifest "C:\protected\metadata\manifest.csv" `
  -Root "C:\protected\dataset" `
  -OutputRoot "C:\MONAI\outputs\resnet18_seed_stability" `
  -Seeds 43,44 `
  -SplitSeed 42 `
  -BatchSize 4 `
  -Epochs 30 `
  -NumWorkers 0 `
  -Device cuda
```

Completed folds are skipped when the command is resumed. The script also creates a lightweight ZIP without model checkpoints.

## Analyze multiple seeds

The final analysis requires one complete result directory per seed. It validates identical cases, patients, labels, and outer folds before calculating any statistics.

```powershell
python .\scripts\analyze_seed_stability.py `
  --run "42=C:\MONAI\outputs\resnet18_5fold" `
  --run "43=C:\MONAI\outputs\resnet18_seed_stability\seed_43" `
  --run "44=C:\MONAI\outputs\resnet18_seed_stability\seed_44" `
  --output-dir "C:\MONAI\outputs\resnet18_final_analysis" `
  --expected-folds 5 `
  --threshold 0.5 `
  --bootstrap-repetitions 10000 `
  --bootstrap-seed 20260804
```

The analysis produces pooled out-of-fold metrics, paired differences, patient-clustered confidence intervals, cross-seed agreement summaries, model-averaged probabilities, PNG/SVG figures, and a manuscript-oriented report. See `docs/MULTI_SEED_ANALYSIS.md` for the complete output description and interpretation safeguards.

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
5. case-level out-of-fold probabilities support paired statistical analysis;
6. patient-clustered resampling preserves bilateral observations within the same bootstrap cluster;
7. additional training seeds assess stochastic stability without changing outer folds or the inner split.

Any later model, threshold, preprocessing, or training-condition comparison must be labeled as a separate prespecified or exploratory experiment.

## Run tests

```bash
pytest
```

Tests requiring PyTorch or MONAI are skipped when optional training dependencies are absent.

## Research safeguards

- Never commit radiographs, direct identifiers, clinical metadata, checkpoints, predictions, or generated private reports.
- Split by `patient_id`, never by image, crop, tooth, or augmented sample.
- Apply random augmentation only to training partitions.
- Keep original and WhatsApp counterparts in the same fold.
- Never use the outer test fold for early stopping, threshold selection, or hyperparameter selection.
- Keep the grouped inner split fixed when isolating the effect of model seeds.
- Never use similarity scores as final pair labels without manual confirmation.
- Generate final tables and figures directly from saved out-of-fold predictions.
- Do not claim equivalence or non-inferiority without a prespecified clinical margin.

## Citation

If you use this software, please cite the permanently archived version used in the study:

> Vizzotto, B. B. (2026). *Reproducible paired evaluation pipeline for original and WhatsApp-compressed panoramic radiographs* (Version 0.6.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21894253

Citation metadata are also available in `CITATION.cff`.

## License

MIT License. See `LICENSE`.
