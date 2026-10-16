# Fair Comparison of U-Net, RadioUNet, FNO, and U-FNO for 2D Outdoor Radio-Map Prediction

## 1. Project overview

This Final Year Project investigates whether a change in **model architecture**, particularly the use of Fourier Neural Operators, improves outdoor 2D radio-map prediction when model size, data access, preprocessing, and training conditions are controlled.

The four models are:

1. **U-Net**: a conventional convolutional encoder-decoder baseline.
2. **RadioUNet-style network**: a two-stage coarse-to-refinement convolutional baseline designed for radio-map prediction.
3. **FNO**: a Fourier Neural Operator using global spectral convolutions.
4. **U-FNO**: a hybrid model combining spectral processing with local U-Net-style processing.

The prediction task is:

```text
Urban building geometry + transmitter information
                         ↓
           256 × 256 outdoor path-gain map
```

The main research question is:

> When all models have approximately the same number of trainable parameters and receive the same input information, can FNO or U-FNO provide a better accuracy-efficiency trade-off than conventional convolutional models for outdoor 2D radio-map prediction?

This project uses the **RadioMapSeer IRT2 and IRT4 datasets**. The dataset itself is not included in the Git repository.

---

## 2. Current experiment status

The following preparation steps have been completed:

- 56,080 IRT2 samples were paired successfully.
- 1,402 IRT4 samples were paired successfully.
- All 701 building maps were detected.
- The city-disjoint split contains:
  - 491 training maps;
  - 105 validation maps;
  - 105 test maps.
- There is no city-map overlap between the three partitions.
- Building masks were verified as binary images with values 0 and 255.
- Transmitter maps were verified to contain one transmitter pixel.
- Inputs were verified to have shape `[5, 256, 256]`.
- Targets were verified to have shape `[1, 256, 256]`.
- All four models were adjusted to approximately six million trainable parameters.
- CUDA training, validation, checkpointing, mixed precision, and metric calculation were tested successfully with U-Net.

The remaining research work includes full multi-seed training, final test-set evaluation, regional error analysis, and IRT4/input-channel ablations.

---

## 3. Dataset directory

The expected dataset location on the current Windows machine is:

```text
C:\Users\user\Desktop\ufno\RadioMapSeer
```

The confirmed directory structure is:

```text
RadioMapSeer/
├── gain/
│   ├── IRT2/                  # 56,080 path-gain maps
│   └── IRT4/                  # 1,402 path-gain maps
└── png/
    ├── antennas/              # 56,080 transmitter maps
    └── buildings_complete/    # 701 building maps
```

Example files:

```text
png/buildings_complete/0.png
png/antennas/0_0.png
gain/IRT2/0_0.png
gain/IRT4/0_0.png
```

All images are 256 × 256 pixels.

> The RadioMapSeer directory must not be committed to Git. It contains a large number of image files and should remain outside version control.

---

## 4. Project structure

```text
ufno/
├── RadioMapSeer/                  # Local dataset, ignored by Git
├── configs/
│   ├── base.yaml
│   ├── unet.yaml
│   ├── radiounet.yaml
│   ├── fno.yaml
│   └── ufno.yaml
├── data/
│   ├── manifests/
│   │   ├── irt2.csv
│   │   └── irt4.csv
│   └── splits/
│       ├── train_maps.txt
│       ├── val_maps.txt
│       └── test_maps.txt
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_manifest.py
│   ├── create_splits.py
│   ├── profile_model.py
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   └── utils.py
├── tests/
│   └── test_smoke.py
├── environment.yml
├── requirements.txt
├── pytest.ini
├── smoke_test_all.ps1
├── train_all.ps1
└── README.md
```

---

## 5. Why five input channels are used

Every model receives exactly the same five input channels:

```text
Channel 0: binary building mask
Channel 1: Gaussian transmitter-position map
Channel 2: normalized distance from the transmitter
Channel 3: normalized horizontal displacement from the transmitter
Channel 4: normalized vertical displacement from the transmitter
```

The complete input tensor has shape:

```text
[5, 256, 256]
```

### 5.1 Channel 0: building mask

The building channel represents the urban geometry:

```text
0 = open area
1 = building
```

Buildings strongly affect outdoor propagation through blockage, reflection, diffraction, and radio-shadow formation. This channel provides the environmental geometry needed to predict the path-gain map.

The original PNG contains values 0 and 255. The loader divides the image by 255, producing values 0 and 1.

### 5.2 Channel 1: Gaussian transmitter map

The original antenna image contains one nonzero transmitter pixel. The data loader detects this position and creates a small Gaussian heatmap centred on the transmitter.

This is used instead of only a single-pixel impulse because repeated pooling or downsampling may weaken or remove a one-pixel feature. The Gaussian representation preserves transmitter-location information more reliably in convolutional encoder-decoder models.

The Gaussian includes location only. It does not contain target path-gain information.

### 5.3 Channel 2: normalized transmitter distance

For each pixel `(x, y)`, the distance channel is:

```text
D(x, y) = sqrt((x - xt)^2 + (y - yt)^2) / image_diagonal
```

where `(xt, yt)` is the transmitter position.

Distance is physically relevant because signal attenuation generally depends strongly on transmitter-receiver separation. Providing this field prevents every architecture from having to rediscover Euclidean distance from a small transmitter marker.

### 5.4 Channel 3: signed horizontal displacement

```text
dx(x, y) = (x - xt) / (width - 1)
```

Negative values indicate pixels to the left of the transmitter. Positive values indicate pixels to the right.

### 5.5 Channel 4: signed vertical displacement

```text
dy(x, y) = (y - yt) / (height - 1)
```

Negative values indicate pixels above the transmitter. Positive values indicate pixels below it.

Distance alone does not preserve direction. Two pixels can have the same distance while being located on opposite sides of the transmitter. The signed displacement channels provide this directional information.

### 5.6 Why all models receive these channels

FNO implementations often include explicit spatial coordinates, while CNN implementations may not. Giving coordinate-derived information only to FNO would make the comparison unfair.

Therefore:

- every model receives the same five channels;
- no model receives additional hidden coordinate channels;
- no model receives sparse ground-truth radio measurements;
- the target map is never included in the input.

A separate two-channel ablation can later use only the building mask and transmitter map. That ablation must also be applied identically to all four models.

---

## 6. Output and normalization

Each model predicts one channel:

```text
[1, 256, 256]
```

The PNG target is normalized during loading:

```text
normalized_target = target_png / 255
```

The network therefore predicts a continuous value in the normalized path-gain space.

For evaluation, predictions are clamped to `[0, 1]` and converted to dB using the configured mapping:

```text
dB = minimum_dB + normalized_value × (maximum_dB - minimum_dB)
```

The current configuration uses:

```text
minimum_dB = -127
a maximum_dB = -47
```

Before the final thesis submission, this conversion must be checked against the exact RadioMapSeer folder and threshold convention being used. The normalized metrics remain valid independently, but a wrong physical mapping would make the reported dB metrics incorrect.

---

## 7. City-disjoint dataset split

The samples are split by **city-map ID**, not by individual transmitter sample.

```text
Training:   491 city maps, 39,280 IRT2 samples
Validation: 105 city maps,  8,400 IRT2 samples
Test:       105 city maps,  8,400 IRT2 samples
```

Every transmitter associated with a city remains in the same partition.

This is essential because a random transmitter-level split could place the same building layout in training and testing. In that case, the model would be evaluated on transmitter positions from urban geometries already seen during training. That would overestimate generalization.

The split files are fixed:

```text
data/splits/train_maps.txt
data/splits/val_maps.txt
data/splits/test_maps.txt
```

These files must not be regenerated between models, runs, or seeds.

IRT4 experiments should reuse the same city split so that IRT2 and IRT4 evaluations contain aligned urban environments.

---

## 8. Fair-comparison design

The purpose of the experiment is to identify whether performance changes are caused by architecture rather than by an unequal model size, unequal data, or unequal training resources.

### 8.1 Parameter matching

The current configurations produce:

| Model | Trainable parameters | Difference from 6M |
|---|---:|---:|
| U-Net | 6,011,097 | +0.18% |
| RadioUNet-style | 6,136,582 | +2.28% |
| FNO | 5,930,731 | -1.15% |
| U-FNO | 6,040,613 | +0.68% |

The largest difference between the models is within the selected 5% tolerance.

This makes the main experiment **parameter matched**, but not necessarily compute matched. Different operations can have different costs even with the same number of weights.

### 8.2 Shared data and preprocessing

All models use:

- the same IRT2 manifest;
- the same city split;
- the same five input channels;
- the same target normalization;
- the same augmentation policy;
- the same batch size;
- the same validation and checkpoint-selection rules.

### 8.3 Shared training protocol

The primary common-protocol experiment uses:

```text
Batch size:                 4
Maximum epochs:             50
Optimizer:                  AdamW
Initial learning rate:      1e-3
Weight decay:               1e-4
Scheduler:                  cosine annealing
Minimum learning rate:      1e-6
Loss:                       L1
Gradient-loss weight:       0
Gradient clipping:          1.0
Automatic mixed precision:  enabled on CUDA
Early-stopping patience:    15 epochs
Checkpoint selection:       lowest validation loss
```

Using the same learning rate for all architectures answers the primary question:

> How do the architectures compare under the same training protocol?

A later learning-rate sensitivity study may test the same candidate values for every model, for example:

```text
1e-4, 3e-4, 1e-3, 3e-3
```

Any model-specific learning-rate selection must use the validation set only and the same tuning budget for every model.

### 8.4 Shared augmentation

The training dataset can apply:

- horizontal flipping;
- vertical flipping;
- rotation in multiples of 90 degrees.

The building map, transmitter map, and target are transformed together. The distance and displacement channels are recalculated from the transformed transmitter location.

Arbitrary-angle rotations, elastic transformations, colour changes, and transformations applied independently to channels are not used because they could alter the physical meaning or introduce interpolation artefacts.

### 8.5 Test-set separation

The test set must not be used to:

- select the model width;
- select the learning rate;
- determine early stopping;
- select a checkpoint;
- choose augmentation;
- choose the number of Fourier modes.

The best checkpoint is selected using validation loss. The test set is used only for final evaluation.

### 8.6 Multiple random seeds

The final comparison should use at least three seeds:

```text
42
123
2026
```

For every model, report:

```text
mean ± standard deviation
```

A single run is insufficient to determine whether a small difference is caused by architecture or random initialization.

### 8.7 Efficiency reporting

Equal parameter counts do not imply equal computational cost. The project records:

- trainable parameter count;
- measured inference latency;
- epoch duration;
- peak GPU memory;
- total training time;
- throughput;
- best-validation epoch.

The FLOP profiler does not count `torch.fft.rfft2` and `torch.fft.irfft2`. Therefore, FNO and U-FNO FLOP values are incomplete and must not be directly presented as complete computational costs.

The thesis should state this limitation and prioritize measured latency, memory, and training time when discussing efficiency.

### 8.8 RadioUNet naming limitation

The current RadioUNet implementation is a controlled two-stage, coarse-to-refinement RadioUNet-style model. It is not yet verified as a layer-for-layer reproduction of the original authors' implementation.

For accurate reporting, use one of these approaches:

1. Label it **RadioUNet-style two-stage refinement network**; or
2. Port and validate the exact official RadioUNet architecture before claiming an exact reproduction.

This point should be reviewed before the final thesis methodology is fixed.

---

## 9. Conda environment installation

### 9.1 Requirements

- Windows 10 or 11;
- Miniconda or Anaconda;
- NVIDIA GPU recommended;
- NVIDIA driver compatible with the selected PyTorch CUDA build;
- Git for version control.

### 9.2 Create the environment

Open PowerShell or Anaconda Prompt:

```powershell
cd "C:\Users\user\Desktop\ufno"
conda env create -f environment.yml
conda activate radio-map-fyp
```

If the supplied environment already exists:

```powershell
conda env remove -n radio-map-fyp
conda env create -f environment.yml
conda activate radio-map-fyp
```

The existing environment named `ufno` may also be used if all package checks and tests pass.

### 9.3 Verify PyTorch and CUDA

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected result:

```text
CUDA available: True
GPU: NVIDIA ...
```

### 9.4 Verify imports

```powershell
python -c "import torch,numpy,pandas,PIL,yaml,tqdm; print('Required imports passed')"
```

### 9.5 Run tests

```powershell
python -m pytest -q
```

Expected:

```text
1 passed
```

---

## 10. Dataset preparation

The following commands are included for reproducibility. They do not need to be rerun if the current manifests and split files already exist.

### 10.1 Inspect the dataset

```powershell
python scripts/inspect_dataset.py --root "C:\Users\user\Desktop\ufno\RadioMapSeer"
```

Expected counts:

```text
56,080 gain/IRT2 images
56,080 png/antennas images
1,402 gain/IRT4 images
701 png/buildings_complete images
```

### 10.2 Build the IRT2 manifest

```powershell
python scripts/build_manifest.py --root "C:\Users\user\Desktop\ufno\RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT2/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt2.csv"
```

Expected:

```text
paired=56080
```

### 10.3 Create the city-disjoint split

```powershell
python scripts/create_splits.py --manifest "data/manifests/irt2.csv" --out "data/splits" --seed 42
```

Expected:

```text
train 491
val 105
test 105
```

### 10.4 Build the IRT4 manifest

```powershell
python scripts/build_manifest.py --root "C:\Users\user\Desktop\ufno\RadioMapSeer" --building-glob "png/buildings_complete/*.png" --tx-glob "png/antennas/*.png" --target-glob "gain/IRT4/*.png" --building-regex "^(?P<map>\d+)$" --tx-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --target-regex "^(?P<map>\d+)_(?P<tx>\d+)$" --out "data/manifests/irt4.csv"
```

Expected:

```text
paired=1402
```

Do not create a new independent split for IRT4. Reuse the IRT2 city split.

---

## 11. Configuration files

The common settings are in:

```text
configs/base.yaml
```

The model-specific width settings are in:

```text
configs/unet.yaml
configs/radiounet.yaml
configs/fno.yaml
configs/ufno.yaml
```

The controlled widths are:

```text
U-Net base width:       28
RadioUNet base width:   20
FNO hidden width:       43
U-FNO hidden width:     42
```

For FNO and U-FNO:

```text
Fourier modes: 20 × 20
Fourier depth: 4 blocks
Padding: 9 pixels
```

---

## 12. Verify model sizes and preliminary latency

```powershell
python scripts/profile_model.py --config configs/unet.yaml
python scripts/profile_model.py --config configs/radiounet.yaml
python scripts/profile_model.py --config configs/fno.yaml
python scripts/profile_model.py --config configs/ufno.yaml
```

The confirmed measurements on the current machine were approximately:

| Model | Parameters | Preliminary latency |
|---|---:|---:|
| U-Net | 6,011,097 | 4.31 ms |
| RadioUNet-style | 6,136,582 | 6.33 ms |
| FNO | 5,930,731 | 9.56 ms |
| U-FNO | 6,040,613 | 18.07 ms |

Latency depends on the GPU, software versions, warm-up, and measurement settings. It must be measured again under the final documented hardware configuration before it is used in the thesis.

---

## 13. Short smoke test

Before full training, run:

```powershell
.\smoke_test_all.ps1
```

This processes two training batches and two validation batches for each model.

The terminal displays separate persistent progress bars:

```text
Epoch 01/01 Train
Epoch 01/01 Valid
```

Each progress bar reports running loss, running dB RMSE, current learning rate, elapsed time, and estimated remaining time.

After all smoke tests pass, delete the preliminary outputs:

```powershell
Remove-Item -Recurse -Force outputs
```

Smoke-test checkpoints must not be confused with final checkpoints because their cosine schedule is configured for one epoch.

---

## 14. Full training

### 14.1 Train models individually

```powershell
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/radiounet.yaml
python scripts/train.py --config configs/fno.yaml
python scripts/train.py --config configs/ufno.yaml
```

Train one model at a time to avoid competition for GPU memory.

### 14.2 Train all models sequentially

```powershell
.\train_all.ps1
```

This runs all four configurations one after another and stops if one model fails.

### 14.3 Resume interrupted training

```powershell
python scripts/train.py --config configs/ufno.yaml --resume
```

Change the configuration path for the relevant model.

Resume restores:

- model weights;
- optimizer state;
- scheduler state;
- mixed-precision gradient scaler;
- current epoch;
- best validation loss;
- early-stopping counter.

Do not use `--resume` for a new independent seed or experiment.

---

## 15. Training outputs

Each model creates:

```text
outputs/<model>/seed_<seed>/
├── best.pt
├── last.pt
├── history.csv
├── resolved_config.json
└── tb/
```

### `best.pt`

The checkpoint with the lowest validation loss. Use this for final testing.

### `last.pt`

The most recent checkpoint. Use this only for resuming interrupted training or diagnostic analysis.

### `history.csv`

Contains per-epoch values including:

- learning rate;
- training loss;
- validation loss;
- training RMSE;
- validation RMSE;
- validation MAE;
- validation relative L2 error;
- epoch duration;
- peak GPU memory.

### `resolved_config.json`

Stores the exact merged configuration used for the run.

---

## 16. TensorBoard

Open another terminal:

```powershell
conda activate radio-map-fyp
cd "C:\Users\user\Desktop\ufno"
tensorboard --logdir outputs
```

Open:

```text
http://localhost:6006
```

Tracked quantities include:

- training and validation loss;
- training and validation RMSE;
- learning rate;
- epoch duration;
- peak GPU memory.

---

## 17. Final evaluation

Use the best validation checkpoint for every model:

```powershell
python scripts/evaluate.py --checkpoint outputs/unet/seed_42/best.pt --split test
python scripts/evaluate.py --checkpoint outputs/radiounet/seed_42/best.pt --split test
python scripts/evaluate.py --checkpoint outputs/fno/seed_42/best.pt --split test
python scripts/evaluate.py --checkpoint outputs/ufno/seed_42/best.pt --split test
```

Each evaluation creates:

```text
test_metrics.json
test_per_sample.csv
```

Currently reported metrics include:

- RMSE in dB;
- MAE in dB;
- normalized RMSE;
- relative L2 error.

The per-sample CSV allows paired comparison because every model predicts the same city-transmitter samples.

---

## 18. Recommended final evaluation additions

For a complete architectural comparison, the following should still be implemented or reported:

1. Three or more random seeds.
2. Mean and standard deviation across seeds.
3. Per-city bootstrap confidence intervals.
4. Paired per-city statistical testing.
5. Boundary-region RMSE.
6. High-gradient-region RMSE.
7. Error by transmitter-distance band.
8. LoS versus NLoS error, if a deterministic LoS mask is implemented.
9. SSIM and PSNR as supplementary image metrics.
10. A two-channel input ablation.
11. IRT4 experiments.
12. Matched-size IRT2 versus IRT4 experiments.
13. Cross-simulator IRT2-to-IRT4 generalization.
14. Exact RadioUNet reproduction or explicit RadioUNet-style naming.

These additions are not required to verify that the current code runs, but they strengthen the scientific conclusion.

---

## 19. IRT4 comparison caution

IRT2 contains 56,080 samples, while IRT4 contains only 1,402 samples in the current dataset. A direct comparison of an IRT2-trained model against an IRT4-trained model changes both:

- propagation complexity; and
- training-set size.

Therefore, a fair IRT2/IRT4 study should include:

1. all-model training on IRT4 using the same city split;
2. a matched IRT2 subset containing the same city-transmitter pairs as IRT4;
3. optional IRT2 pretraining followed by controlled IRT4 fine-tuning;
4. cross-simulator testing on aligned samples.

---

## 20. Reproducibility checklist

Before accepting a result for the thesis, confirm:

- [ ] The same city split was used.
- [ ] The same input channels were used.
- [ ] The same target normalization was used.
- [ ] The same augmentation policy was used.
- [ ] Batch size was four for all models.
- [ ] Maximum epochs was 50 for all models.
- [ ] AdamW and the same weight decay were used.
- [ ] The same cosine schedule was used.
- [ ] L1 loss was used for the primary experiment.
- [ ] No test result influenced hyperparameter selection.
- [ ] `best.pt` was selected using validation loss.
- [ ] The random seed was recorded.
- [ ] GPU and software versions were recorded.
- [ ] Parameter count was recorded.
- [ ] Latency, epoch duration, and peak memory were recorded.
- [ ] FNO/U-FNO FLOPs were labelled incomplete.
- [ ] Results were repeated using multiple seeds.

---

## 21. Git repository policy

The following should be committed:

```text
configs/
scripts/
src/
tests/
data/splits/
environment.yml
requirements.txt
pytest.ini
README.md
smoke_test_all.ps1
train_all.ps1
```

The following should not be committed:

```text
RadioMapSeer/
outputs/
*.pt
*.pth
*.ckpt
TensorBoard event files
generated predictions
```

The manifests contain local relative paths and may be regenerated. They can remain ignored if desired.

---

## 22. Questions for supervisor review

The following design choices should be reviewed before the final experimental campaign:

1. Is a ±5% parameter tolerance acceptable for the primary comparison?
2. Is a shared learning rate of `1e-3` preferred, or should every model receive an equal-budget validation-based learning-rate search?
3. Should the current model be labelled RadioUNet-style, or must the official RadioUNet be reproduced exactly?
4. Is L1 loss sufficient for the primary comparison, with gradient loss treated as a separate ablation?
5. Are the five physically informed input channels acceptable, or should the two-channel building-plus-transmitter setting be the main benchmark?
6. Which physical dB mapping should be used for the exact RadioMapSeer target convention?
7. Is parameter matching sufficient for the primary test, with computation reported separately, or is a second compute-matched experiment required?
8. Which regional metrics are required to support claims about boundaries and shadow regions?
9. Should IRT4 be treated as a separate small-data benchmark, a transfer-learning benchmark, or both?

---

## 23. Summary

The current project provides a controlled, parameter-matched framework for comparing convolutional, spectral, and hybrid architectures for outdoor 2D path-gain prediction.

The primary fairness controls are:

- approximately six million parameters per model;
- identical five-channel input information;
- city-disjoint train, validation, and test partitions;
- identical preprocessing and augmentation;
- identical optimizer, learning rate, scheduler, loss, batch size, and epoch limit;
- validation-only checkpoint selection;
- separate reporting of computational cost.

The setup is ready for final smoke testing and full 50-epoch training. Multi-seed evaluation, regional error analysis, exact physical-unit verification, and IRT4 controls remain necessary before drawing the final scientific conclusion.
