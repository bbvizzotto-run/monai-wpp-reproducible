param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [string]$OutputRoot = "C:\MONAI\outputs\resnet18_seed_stability",

    [int[]]$Seeds = @(43, 44),

    [int]$SplitSeed = 42,

    [int]$BatchSize = 4,

    [int]$Epochs = 30,

    [int]$NumWorkers = 0,

    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Manifest)) {
    throw "Manifest not found: $Manifest"
}
if (-not (Test-Path $Root)) {
    throw "Dataset root not found: $Root"
}

New-Item -ItemType Directory -Force $OutputRoot | Out-Null

foreach ($seed in $Seeds) {
    $seedOutput = Join-Path $OutputRoot "seed_$seed"
    New-Item -ItemType Directory -Force $seedOutput | Out-Null

    foreach ($fold in 0..4) {
        $metricsPath = Join-Path $seedOutput "fold_$fold\metrics.json"
        if (Test-Path $metricsPath) {
            Write-Host "Skipping completed seed=$seed fold=$fold"
            continue
        }

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "TRAINING SEED $seed - OUTER FOLD $fold"
        Write-Host "Fixed inner split seed: $SplitSeed"
        Write-Host "============================================================"

        & python .\scripts\train_pilot_fold.py `
            --manifest $Manifest `
            --root $Root `
            --fold $fold `
            --output-dir $seedOutput `
            --epochs $Epochs `
            --batch-size $BatchSize `
            --num-workers $NumWorkers `
            --device $Device `
            --seed $seed `
            --split-seed $SplitSeed

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for seed=$seed fold=$fold"
        }
    }
}

$lightRoot = "${OutputRoot}_light"
Remove-Item $lightRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $lightRoot | Out-Null

foreach ($seed in $Seeds) {
    $sourceSeed = Join-Path $OutputRoot "seed_$seed"
    $lightSeed = Join-Path $lightRoot "seed_$seed"
    New-Item -ItemType Directory -Force $lightSeed | Out-Null

    foreach ($fold in 0..4) {
        $sourceFold = Join-Path $sourceSeed "fold_$fold"
        $lightFold = Join-Path $lightSeed "fold_$fold"
        $lightSelection = Join-Path $lightFold "selection"
        New-Item -ItemType Directory -Force $lightSelection | Out-Null

        foreach ($file in @(
            "metrics.json",
            "predictions.csv",
            "refit_history.csv",
            "pilot_run_metadata.json",
            "provenance.json"
        )) {
            $source = Join-Path $sourceFold $file
            if (Test-Path $source) {
                Copy-Item $source $lightFold
            }
        }

        foreach ($file in @("history.csv", "best_model.json")) {
            $source = Join-Path (Join-Path $sourceFold "selection") $file
            if (Test-Path $source) {
                Copy-Item $source $lightSelection
            }
        }
    }
}

$archivePath = "${OutputRoot}_light.zip"
Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path "$lightRoot\*" -DestinationPath $archivePath -Force

Write-Host ""
Write-Host "All requested seed runs completed."
Write-Host "Full checkpoints: $OutputRoot"
Write-Host "Light package: $archivePath"
