# Parameter-Matched Experiment

## Purpose

This folder contains the primary architecture comparison for radio-map prediction. U-Net, RadioUNet-style, FNO and U-FNO are matched to approximately six million trainable parameters.

The experiment asks:

> When model capacity, input information, dataset split and training protocol are controlled, which architecture provides the best radio-map prediction performance?

## Local project path

```text
C:\Users\user\Desktop\FYP\parameter
```

Shared dataset:

```text
C:\Users\user\Desktop\FYP\RadioMapSeer
```

## Parameter-matched models

```text
U-Net:       6,011,097 parameters, base width 28
RadioUNet:   6,136,582 parameters, base width 20
FNO:         5,930,731 parameters, width 43, 20x20 modes, depth 4
U-FNO:       6,040,613 parameters, width 42, 20x20 modes, depth 4
```

The largest pairwise difference remains within the predefined 5% tolerance.

## Shared training configuration

```yaml
seed: 42
output_dir: outputs

training:
  epochs: 50
  batch_size: 4
  lr: 0.001
  weight_decay: 0.0001
  patience: 15
  amp: false
  grad_clip: 1.0
  loss: l1
  gradient_weight: 0.0
  scheduler: cosine
  min_lr: 0.000001
```

FP32 is used because half-precision cuFFT failed for the padded 265x265 FNO/U-FNO feature maps.

## Required path changes

Edit:

```text
configs/base.yaml
```

Set:

```yaml
data:
  root: "C:/Users/user/Desktop/FYP/RadioMapSeer"
```

Check and replace old defaults in:

```text
scripts/evaluate.py
scripts/compare_predictions.py
```

Use:

```text
C:/Users/user/Desktop/FYP/RadioMapSeer
```

Find old paths:

```powershell
Get-ChildItem . -Recurse -File | Select-String -Pattern "C:/Users/user/Desktop/ufno|/data1/ieug23/wth023|RadioMapSeer"
```

## Installation

```powershell
cd C:\Users\user\Desktop\FYP\parameter
```

```powershell
conda env create -f environment.yml
```

```powershell
conda activate ufno
```

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

```powershell
python -m pytest -q
```

## Build manifests

```powershell
python scripts/build_manifest.py --root "C:/Users/user/Desktop/FYP/RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT2/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt2.csv"
```

```powershell
python scripts/build_manifest.py --root "C:/Users/user/Desktop/FYP/RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT4/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt4.csv"
```

Expected:

```text
IRT2: 56,080 paired samples
IRT4: 1,402 paired samples
```

Do not regenerate the city split when comparing trained models. Keep:

```text
data/splits/train_maps.txt
data/splits/val_maps.txt
data/splits/test_maps.txt
```

## Train locally

U-Net:

```powershell
python scripts/train.py --config configs/unet.yaml
```

RadioUNet-style:

```powershell
python scripts/train.py --config configs/radiounet.yaml
```

FNO:

```powershell
python scripts/train.py --config configs/fno.yaml
```

U-FNO:

```powershell
python scripts/train.py --config configs/ufno.yaml
```

Department-specific Slurm scripts are intentionally excluded. On another system, create a batch script that activates the environment, changes to this folder and invokes the same Python commands.

## Multiple seeds

Use:

```text
42, 123, 2026
```

For each new run, change only `seed:` in `configs/base.yaml`. Do not change the model width, split, optimizer, loss, scheduler, batch size or epoch budget.

Outputs are stored under:

```text
outputs/<model>/seed_<seed>/
```

## Evaluation

```powershell
python scripts/evaluate.py --checkpoint outputs/unet/seed_42/best.pt --split test --data-root "C:/Users/user/Desktop/FYP/RadioMapSeer" --num-workers 0
```

Repeat for RadioUNet, FNO and U-FNO, and for every seed.

The evaluation creates:

```text
test_metrics.json
test_per_sample.csv
```

## Same-sample visualization

```powershell
python scripts/compare_predictions.py --map-id 389 --tx-id 45 --data-root "C:/Users/user/Desktop/FYP/RadioMapSeer"
```

The figure should use shared colour limits and show:

- Building mask
- Transmitter heatmap
- Distance channel
- Ground truth
- Four predictions
- Four absolute-error maps
- Per-image RMSE, MAE and bias

## Final reporting

For every model, report individual seed values and:

```text
mean ± sample standard deviation
```

Primary metrics:

- Test RMSE in dB
- Test MAE in dB
- Normalized RMSE
- Relative L2 error

Efficiency metrics:

- Parameters
- Inference latency
- Throughput
- Peak GPU memory
- Epoch duration

This folder is the primary capacity-controlled experiment. Do not describe it as compute matched.
