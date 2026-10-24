# FYP: Fair Comparison of U-Net, RadioUNet, FNO and U-FNO for Radio-Map Prediction

## Project overview

This repository contains two controlled experiments for 2D outdoor radio-map prediction using the RadioMapSeer dataset:

1. **Parameter-matched comparison** in `parameter/`
   - U-Net, RadioUNet-style, FNO and U-FNO are matched to approximately 6 million trainable parameters.
   - This experiment asks which architecture performs better when learnable capacity is approximately equal.

2. **FLOP-matched comparison** in `flops/`
   - The same four architectures are matched to approximately 11 billion adjusted analytical floating-point operations per forward pass.
   - Parameter counts are allowed to differ.
   - This experiment asks which architecture performs better when theoretical forward computation is approximately equal.

The local dataset is stored once at:

```text
C:\Users\user\Desktop\FYP\RadioMapSeer
```

The dataset is intentionally excluded from Git because of its size and licensing/distribution requirements.

## Repository layout

```text
FYP/
├── README.md
├── .gitignore
├── RadioMapSeer/                 # Local dataset, not committed
├── parameter/                    # Approx. 6M parameter-matched experiment
│   ├── README.md
│   ├── configs/
│   ├── data/
│   ├── scripts/
│   ├── src/
│   └── tests/
└── flops/                        # Approx. 11B FLOP-matched experiment
    ├── README.md
    ├── configs/
    ├── data/
    ├── scripts/
    ├── src/
    └── tests/
```

Department-specific Slurm batch scripts are not included because the scripts contain cluster-specific partitions, paths, environment names and email settings. The Python training entry point remains available in each experiment folder.

## Dataset structure

The expected local dataset structure is:

```text
RadioMapSeer/
├── gain/
│   ├── IRT2/
│   └── IRT4/
└── png/
    ├── antennas/
    └── buildings_complete/
```

Expected counts:

```text
IRT2 targets:       56,080
IRT4 targets:        1,402
Antenna maps:       56,080
Building maps:         701
```

## Five-channel model input

All models receive the same five channels:

1. Binary building mask
2. Gaussian transmitter-location heatmap
3. Normalized distance from each pixel to the transmitter
4. Signed horizontal displacement from the transmitter
5. Signed vertical displacement from the transmitter

Only the building mask and transmitter map must be supplied as raw images. The remaining three channels are derived automatically from the transmitter position. Ground truth is used only for training and evaluation, never as model input.

## Shared dataset split

The split is city-map-disjoint:

```text
Training:   491 maps, 39,280 IRT2 samples
Validation: 105 maps,  8,400 IRT2 samples
Test:       105 maps,  8,400 IRT2 samples
```

Every transmitter sample associated with one city map stays in the same partition. This prevents the same building geometry from appearing in both training and testing.

## Environment

Each experiment can use the same Conda environment:

```powershell
conda env create -f parameter\environment.yml
conda activate ufno
```

If the environment already exists:

```powershell
conda activate ufno
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Rebuild local manifests

Run from the parameter folder:

```powershell
cd C:\Users\user\Desktop\FYP\parameter
```

```powershell
python scripts/build_manifest.py --root "C:/Users/user/Desktop/FYP/RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT2/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt2.csv"
```

```powershell
python scripts/build_manifest.py --root "C:/Users/user/Desktop/FYP/RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT4/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt4.csv"
```

The fixed split files can be copied from `parameter/data/splits/` to `flops/data/splits/`. Rebuild the FLOP manifests from inside `flops/` using the same commands and dataset root.


### Parameter-matched result

```text
Approximately equal parameters
Different FLOPs and runtimes
```

### FLOP-matched result

```text
Approximately equal adjusted analytical FLOPs
Different parameter counts
```

For both experiments, report:

- Test RMSE in dB
- Test MAE in dB
- Normalized RMSE
- Relative L2 error
- Individual seed values
- Mean and sample standard deviation across seeds 42, 123 and 2026
- Parameter count
- A100 inference latency
- Throughput
- Peak GPU memory
- Training time

## Important limitations

- The RadioUNet implementation should be described as **RadioUNet-style** unless it is verified as a layer-for-layer reproduction of the official architecture.
- The FLOP experiment uses an adjusted analytical estimate for FFT operations, not exact GPU instruction counts.
- Equal FLOPs do not imply equal latency.
- Equal parameters do not imply equal computation.
- The unsuccessful gradient-loss experiment is an ablation and is not part of the final primary comparison.
