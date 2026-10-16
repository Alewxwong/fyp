# Fair U-Net / RadioUNet / FNO / U-FNO Radio-Map Benchmark

A reproducible PyTorch project for 2D outdoor path-gain prediction on RadioMapSeer IRT2/IRT4. All models share data, channels, splits, optimizer, losses and evaluation.

## Important dataset note
RadioMapSeer releases have different folder names. This project does not guess silently. It builds a manifest from globs and regexes. Run `inspect_dataset.py`, then use `build_manifest.py` with patterns matching your downloaded copy.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate ufno
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For CPU-only systems, remove `pytorch-cuda=12.1` and the `nvidia` channel, or use `pip install -r requirements.txt` in a clean Python 3.11 environment.

## 2. Inspect your existing dataset

```bash
python scripts/inspect_dataset.py --root "D:/datasets/RadioMapSeer"
```

## 3. Build a manifest
Example only, change globs and regexes after inspection:

```bash
python scripts/build_manifest.py   --root "D:/datasets/RadioMapSeer"   --building-glob "**/buildings/*.png"   --tx-glob "**/antennas/*.png"   --target-glob "**/IRT2/*.png"   --building-regex "(?P<map>\d+)"   --tx-regex "(?P<map>\d+)[^0-9]+(?P<tx>\d+)"   --target-regex "(?P<map>\d+)[^0-9]+(?P<tx>\d+)"   --out data/manifests/irt2.csv
```

The CSV must contain `map_id,tx_id,building_path,tx_path,target_path`. Paths are relative to `data.root`.

## 4. City-disjoint split

```bash
python scripts/create_splits.py --manifest data/manifests/irt2.csv --out data/splits --seed 42
```

Never randomly split transmitter samples. Every Tx from a city remains in one split.

## 5. Configure
Edit `configs/base.yaml`:

- `data.root`: your RadioMapSeer directory
- `data.manifest`: IRT2 or IRT4 CSV
- `min_db`, `max_db`: verify against the exact release/threshold convention
- `input_mode`: `five` or `two`

Five channels are building mask, Gaussian Tx map, normalized Tx distance, signed dx, signed dy. Two channels are building mask and Tx map.

## 6. Smoke test

```bash
pytest -q
python scripts/train.py --config configs/unet.yaml --max-epochs 1
```

## 7. Train

```bash
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/radiounet.yaml
python scripts/train.py --config configs/fno.yaml
python scripts/train.py --config configs/ufno.yaml
```

Run seeds 42, 123, and 2026 by changing `seed` in copied configs. Widths supplied are starting points, not guaranteed parameter-matched. Run `profile_model.py` and adjust only width/base_width until parameter counts are within your stated tolerance.

## 8. Profile fairness

```bash
python scripts/profile_model.py --config configs/unet.yaml
python scripts/profile_model.py --config configs/radiounet.yaml
python scripts/profile_model.py --config configs/fno.yaml
python scripts/profile_model.py --config configs/ufno.yaml
```

Record parameters, FLOPs, latency, GPU model, CUDA version and batch size.

## 9. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/unet/seed_42/best.pt
```

Outputs include aggregate JSON and per-sample CSV so confidence intervals and paired per-city tests can be added without treating correlated pixels as independent samples.

## Scientific cautions

- This `RadioUNet` is a clean W-shaped coarse-to-refinement implementation, not a claim of bit-for-bit reproduction of every layer in the authors' notebook. For a paper reproduction experiment, port the official architecture unchanged and retain this implementation as a controlled radio-specific baseline.
- Do not compare unmatched IRT2 and IRT4 sizes as though only ray complexity changed.
- Do not select hyperparameters on the test set.
- Keep coordinate-derived input channels identical across architectures.
- First compare plain L1; use gradient loss only as a separate ablation.

## Matched parameter budget in the supplied configs
Using PyTorch `numel()` convention, the supplied widths target approximately six million trainable parameters:
- U-Net base width 28: approximately 6.011M
- RadioUNet base width 20: approximately 6.137M
- FNO width 43: approximately 5.931M
- U-FNO width 42: approximately 6.041M
Verify with `scripts/profile_model.py` in the actual Conda environment before experiments.
