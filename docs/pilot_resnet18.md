# ResNet-18 outer-fold pilot

The first clinical run is frozen to one outer fold and is intended to validate the complete training and export workflow before launching all five folds.

## Prespecified configuration

- architecture: ResNet-18;
- initialization: official torchvision ImageNet weights;
- grayscale adaptation: average the RGB kernels of the first convolution into one input channel;
- input size: 224 × 224;
- training condition: original ROI only;
- paired evaluation: original and WhatsApp ROI versions of the same untouched outer-test cases;
- rotation: at most 7 degrees;
- translation: at most 5% of the resized dimensions;
- horizontal flip: disabled;
- optimizer: AdamW;
- learning rate: 1e-4;
- weight decay: 1e-4;
- batch size: 8;
- maximum epochs: 30;
- early-stopping patience: 8;
- threshold: 0.5;
- grouped inner folds: 4;
- seed: 42.

## Command

```bash
python scripts/train_pilot_fold.py \
  --manifest /protected/monai_training_ready_v1/metadata/manifest.csv \
  --root /protected/monai_training_ready_v1 \
  --fold 0 \
  --output-dir outputs/pilot_resnet18 \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda
```

Run a two-epoch smoke test before the complete pilot. Clinical images and private manifests must remain outside the public repository.
