# Multi-seed paired statistical analysis

This milestone consolidates the five out-of-fold prediction files from each training seed and reproduces the final paired analysis of original and WhatsApp-compressed ROIs.

## Required input structure

Each seed directory must contain the complete five-fold lightweight results:

```text
seed_root/
  fold_0/
    metrics.json
    predictions.csv
    selection/history.csv
  fold_1/
  fold_2/
  fold_3/
  fold_4/
```

The command verifies that all seeds contain the same case IDs, patient IDs, labels, and fixed outer-fold assignments. It rejects missing folds, duplicate cases, invalid probabilities, and inconsistent labels or partitions.

## Windows example

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

Install the statistical dependencies before running:

```powershell
python -m pip install -e ".[stats]"
```

## Analysis outputs

The output directory contains:

- `seed_level_summary.csv`: pooled out-of-fold metrics and paired changes for each seed;
- `fold_level_summary.csv`: selected epochs and fold-level performance;
- `pairwise_seed_probability_stability.csv`: cross-seed probability correlations and agreement;
- `seed_consensus_summary.csv`: unanimous decisions, Fleiss kappa, and model-averaged metrics;
- `case_level_model_averaged_predictions.csv`: aligned probabilities and their across-seed mean and standard deviation;
- `patient_cluster_bootstrap_cis.csv`: patient-clustered 95% confidence intervals;
- `exploratory_patient_and_seed_bootstrap_deltas.csv`: exploratory intervals that resample patients and the finite observed seed set;
- `analysis_metadata.json`: paths, seeds, thresholds, and bootstrap configuration;
- `analysis_report.md`: manuscript-oriented numerical summary;
- PNG and SVG figures for AUC, selected epochs, and compression effects.

## Statistical interpretation

The patient is the bootstrap sampling unit, so bilateral teeth remain together whenever a patient is resampled. The same patient resample is applied to every seed and both image conditions within each repetition, preserving the paired comparison.

The primary comparison is the mean paired difference across the prespecified seeds. The additional patient-and-seed bootstrap is exploratory because only a small finite set of training seeds is available; it is not a formal random-effects model.

A confidence interval centered near zero supports the statement that no consistent performance loss was detected after compression. It does not, by itself, establish equivalence or non-inferiority without a prespecified clinical margin.

## Privacy

Analysis outputs include case and patient identifiers inherited from private prediction files. Do not commit generated CSV files, reports, metadata, checkpoints, radiographs, or private manifests to the public repository. Only the generic analysis code and synthetic tests belong in GitHub.
