# FLOP-Matched Experiment

## Purpose

This folder contains the compute-matched comparison for U-Net, RadioUNet-style, FNO and U-FNO. Models are selected near a target of 11 billion adjusted analytical floating-point operations per forward pass.

The experiment asks:

> When theoretical forward computation is approximately equal, which architecture gives the best radio-map prediction performance?

This experiment is separate from the approximately 6M parameter-matched comparison. Parameter counts are allowed to differ.

## Local project path

```text
C:\Users\user\Desktop\FYP\flops
```

Shared dataset:

```text
C:\Users\user\Desktop\FYP\RadioMapSeer
```

## FLOP convention

```text
Input: [1, 5, 256, 256]
Batch size for profiling: 1
Forward pass only
FP32
Target: 11,000,000,000 adjusted analytical FLOPs
Tolerance: ±10%
Accepted interval: 9,900,000,000 to 12,100,000,000 FLOPs
```

`fvcore` counts supported operations. For FNO and U-FNO, the FFT and inverse FFT are added using:

```text
depth × 10 × width × N × log2(N)
```

where:

```text
N = (256 + padding) × (256 + padding)
```

This is an adjusted analytical estimate, not an exact GPU instruction count.

## Selected FLOP-matched models

```text
U-Net:
  base width 28
  6,011,097 parameters
  10,817,601,536 adjusted FLOPs

RadioUNet-style:
  base width 20
  6,136,582 parameters
  11,124,408,320 adjusted FLOPs

FNO:
  width 122
  20x20 modes
  depth 4
  47,706,405 parameters
  10,989,014,340 adjusted FLOPs

U-FNO:
  width 31
  20x20 modes
  depth 4
  3,292,395 parameters
  11,237,529,432 adjusted FLOPs
```

The parameter imbalance is expected because this experiment matches computation rather than learnable capacity.

## Required path changes

Edit:

```text
configs/base_flops.yaml
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

Find stale paths:

```powershell
Get-ChildItem . -Recurse -File | Select-String -Pattern "C:/Users/user/Desktop/ufno|/data1/ieug23/wth023|RadioMapSeer"
```

Department-specific Slurm scripts are intentionally excluded from this repository. The experiment is still reproducible through the Python entry points.

## Installation

```powershell
cd C:\Users\user\Desktop\FYP\flops
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

## Build local manifests

```powershell
python scripts/build_manifest.py --root "C:/Users/user/Desktop/FYP/RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT2/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt2.csv"
```

Copy the original fixed split files from:

```text
C:\Users\user\Desktop\FYP\parameter\data\splits
```

to:

```text
C:\Users\user\Desktop\FYP\flops\data\splits
```

Do not generate a different split.

## Reproduce the FLOP search

```powershell
python scripts/search_and_write_flop_configs.py --target-gflops 11.0 --tolerance 0.10 --write-configs
```

This creates:

```text
flop_search_results.csv
configs/flops/unet.yaml
configs/flops/radiounet.yaml
configs/flops/fno.yaml
configs/flops/ufno.yaml
```

The selection rule is the smallest absolute difference from the target. Validation and test accuracy must not influence model selection.

## Measure selected configurations

```powershell
python scripts/profile_adjusted_flops.py --config configs/flops/unet.yaml
```

```powershell
python scripts/profile_adjusted_flops.py --config configs/flops/radiounet.yaml
```

```powershell
python scripts/profile_adjusted_flops.py --config configs/flops/fno.yaml
```

```powershell
python scripts/profile_adjusted_flops.py --config configs/flops/ufno.yaml
```

Every adjusted result must remain between 9.9 and 12.1 GFLOPs, equivalent to 9.9 and 12.1 billion FLOPs.

## Train locally

```powershell
python scripts/train.py --config configs/flops/unet.yaml
```

```powershell
python scripts/train.py --config configs/flops/radiounet.yaml
```

```powershell
python scripts/train.py --config configs/flops/fno.yaml
```

```powershell
python scripts/train.py --config configs/flops/ufno.yaml
```

Shared training protocol:

```text
50 epochs
Batch size 4
AdamW
Initial learning rate 1e-3
Weight decay 1e-4
Cosine annealing to 1e-6
L1 loss
Gradient weight 0
FP32
Seeds 42, 123 and 2026
```

If the width-122 FNO does not fit the available GPU at batch size 4, use batch size 2 for all four FLOP-matched models and document the change.

## Evaluation

```powershell
python scripts/evaluate.py --checkpoint outputs_flops/unet/seed_42/best.pt --split test --data-root "C:/Users/user/Desktop/FYP/RadioMapSeer" --num-workers 0
```

Repeat for every architecture and seed.

## Reporting

Report:

- Individual seed results
- Mean ± sample standard deviation
- Test RMSE in dB
- Test MAE in dB
- Normalized RMSE
- Relative L2 error
- Parameter count
- Adjusted analytical FLOPs
- A100 latency
- Throughput
- Peak GPU memory
- Epoch duration

Keep the unsupported-operation limitation explicit. Low-cost activation, addition, padding and pooling operations may remain omitted, while FFT work is included analytically.

This folder is compute matched, not parameter matched.
