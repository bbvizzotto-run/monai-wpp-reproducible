#!/usr/bin/env python3
"""Analyze fixed-fold paired predictions across multiple ResNet training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from monai_wpp.evaluation.seed_stability import (
    align_seed_runs,
    load_seed_run,
    pairwise_seed_stability,
    parse_run_spec,
    patient_cluster_bootstrap,
    seed_consensus,
    summarize_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help=(
            "Complete five-fold result directory for one seed. Repeat for every seed, "
            "for example --run 42=outputs/resnet18_seed42."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_804)
    return parser.parse_args()


def _save_figures(seed_summary: pd.DataFrame, fold_summary: pd.DataFrame, output: Path) -> None:
    auc = seed_summary[["seed", "original_roc_auc", "whatsapp_roc_auc"]].set_index("seed")
    axis = auc.plot(kind="bar")
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Out-of-fold AUC")
    axis.set_ylim(0, 1)
    axis.set_title("Out-of-fold AUC by training seed and image condition")
    plt.tight_layout()
    plt.savefig(output / "figure_auc_by_seed.png", dpi=300)
    plt.savefig(output / "figure_auc_by_seed.svg")
    plt.close()

    epochs = fold_summary.pivot(index="fold", columns="seed", values="selected_epochs")
    axis = epochs.plot(kind="bar")
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Selected epochs")
    axis.set_title("Selected epoch count by outer fold and training seed")
    plt.tight_layout()
    plt.savefig(output / "figure_selected_epochs.png", dpi=300)
    plt.savefig(output / "figure_selected_epochs.svg")
    plt.close()

    effects = seed_summary[
        ["seed", "delta_roc_auc", "delta_accuracy", "delta_brier_score"]
    ].set_index("seed")
    axis = effects.plot(kind="bar")
    axis.axhline(0, linewidth=1)
    axis.set_xlabel("Training seed")
    axis.set_ylabel("WhatsApp minus original")
    axis.set_title("Estimated compression effect by training seed")
    plt.tight_layout()
    plt.savefig(output / "figure_compression_effect_by_seed.png", dpi=300)
    plt.savefig(output / "figure_compression_effect_by_seed.svg")
    plt.close()


def _write_report(
    output: Path,
    seed_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    bootstrap_cis: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    auc_original_mean = seed_summary["original_roc_auc"].mean()
    auc_original_sd = seed_summary["original_roc_auc"].std(ddof=1)
    auc_whatsapp_mean = seed_summary["whatsapp_roc_auc"].mean()
    auc_whatsapp_sd = seed_summary["whatsapp_roc_auc"].std(ddof=1)
    delta_auc_mean = seed_summary["delta_roc_auc"].mean()
    delta_auc_sd = seed_summary["delta_roc_auc"].std(ddof=1)
    accuracy_original_mean = seed_summary["original_accuracy"].mean()
    accuracy_whatsapp_mean = seed_summary["whatsapp_accuracy"].mean()

    delta_auc_ci = bootstrap_cis[
        (bootstrap_cis["scope"] == "seed_mean")
        & (bootstrap_cis["condition"] == "whatsapp_minus_original")
        & (bootstrap_cis["metric"] == "roc_auc")
    ].iloc[0]
    original_consensus = consensus[consensus["condition"] == "original"].iloc[0]
    whatsapp_consensus = consensus[consensus["condition"] == "whatsapp"].iloc[0]

    text = f"""# Multi-seed paired analysis

## Data integrity

- Training seeds: {', '.join(str(value) for value in seed_summary['seed'])}.
- Fixed outer folds per seed: {fold_summary['fold'].nunique()}.
- Out-of-fold cases per seed: {int(seed_summary['original_n'].iloc[0])}.
- The analysis verifies identical case, patient, label, and outer-fold assignments across seeds.

## Main results

- Original AUC: {auc_original_mean:.3f} ± {auc_original_sd:.3f} across seeds.
- WhatsApp AUC: {auc_whatsapp_mean:.3f} ± {auc_whatsapp_sd:.3f} across seeds.
- Paired AUC difference (WhatsApp minus original): {delta_auc_mean:+.3f} ± {delta_auc_sd:.3f}.
- Patient-clustered 95% bootstrap interval for the mean AUC difference: {delta_auc_ci['ci_low']:+.3f} to {delta_auc_ci['ci_high']:+.3f}.
- Original accuracy: {accuracy_original_mean:.3f} across seeds.
- WhatsApp accuracy: {accuracy_whatsapp_mean:.3f} across seeds.
- Unanimous binary decisions across all seeds: {int(original_consensus['unanimous_cases'])} original cases and {int(whatsapp_consensus['unanimous_cases'])} WhatsApp cases.
- Selected epoch counts ranged from {int(fold_summary['selected_epochs'].min())} to {int(fold_summary['selected_epochs'].max())}.

## Interpretation

Absolute predictions may vary with model initialization and stochastic augmentation. The paired compression effect should therefore be interpreted separately from absolute diagnostic performance. A small confidence interval centered near zero supports the conclusion that no consistent loss of discrimination was detected after WhatsApp compression, but it does not establish formal equivalence or non-inferiority unless a clinical margin was prespecified.

## Manuscript-ready wording

Across the prespecified training seeds, mean out-of-fold AUC was {auc_original_mean:.3f} ± {auc_original_sd:.3f} for original ROIs and {auc_whatsapp_mean:.3f} ± {auc_whatsapp_sd:.3f} for WhatsApp-compressed ROIs. The mean paired AUC difference was {delta_auc_mean:+.3f} ± {delta_auc_sd:.3f} (WhatsApp minus original). A patient-clustered bootstrap applied to the mean across seeds yielded a 95% confidence interval from {delta_auc_ci['ci_low']:+.3f} to {delta_auc_ci['ci_high']:+.3f}. Although absolute predictions varied across model seeds, the estimated effect of WhatsApp compression remained small.
"""
    (output / "analysis_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if args.expected_folds < 2:
        raise ValueError("expected-folds must be at least 2.")

    parsed_runs = [parse_run_spec(specification) for specification in args.run]
    if len({seed for seed, _ in parsed_runs}) != len(parsed_runs):
        raise ValueError("Each seed may be supplied only once.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    runs = [
        load_seed_run(seed, path, expected_folds=args.expected_folds)
        for seed, path in parsed_runs
    ]
    wide = align_seed_runs(runs)
    seeds = sorted(run.seed for run in runs)

    seed_summary = summarize_seeds(runs, threshold=args.threshold)
    fold_summary = pd.concat([run.fold_summary for run in runs], ignore_index=True)
    pairwise = pairwise_seed_stability(wide, seeds)
    consensus, case_level = seed_consensus(wide, seeds, threshold=args.threshold)
    bootstrap_cis, exploratory = patient_cluster_bootstrap(
        wide,
        seeds,
        repetitions=args.bootstrap_repetitions,
        random_seed=args.bootstrap_seed,
        threshold=args.threshold,
    )

    seed_summary.to_csv(output / "seed_level_summary.csv", index=False)
    fold_summary.to_csv(output / "fold_level_summary.csv", index=False)
    pairwise.to_csv(output / "pairwise_seed_probability_stability.csv", index=False)
    consensus.to_csv(output / "seed_consensus_summary.csv", index=False)
    case_level.to_csv(output / "case_level_model_averaged_predictions.csv", index=False)
    bootstrap_cis.to_csv(output / "patient_cluster_bootstrap_cis.csv", index=False)
    exploratory.to_csv(output / "exploratory_patient_and_seed_bootstrap_deltas.csv", index=False)

    metadata = {
        "runs": {str(run.seed): str(run.root) for run in runs},
        "seeds": seeds,
        "expected_folds": args.expected_folds,
        "threshold": args.threshold,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "case_count": int(len(wide)),
        "patient_count": int(wide["patient_id"].nunique()),
        "privacy_note": (
            "Outputs contain case and patient identifiers and must remain outside the public repository."
        ),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    _save_figures(seed_summary, fold_summary, output)
    _write_report(output, seed_summary, fold_summary, bootstrap_cis, consensus)

    print(f"Saved analysis to: {output}")
    print(seed_summary[["seed", "original_roc_auc", "whatsapp_roc_auc", "delta_roc_auc"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
