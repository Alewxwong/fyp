import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import RadioMapDataset
from src.models import build_model


MODEL_PATHS = {
    "U-Net": "outputs/unet/seed_42/best.pt",
    "RadioUNet": "outputs/radiounet/seed_42/best.pt",
    "FNO": "outputs/fno/seed_42/best.pt",
    "U-FNO": "outputs/ufno/seed_42/best.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare all four trained models on exactly the same RadioMapSeer sample."
    )
    parser.add_argument("--map-id", default=None, help="City map ID, for example 650")
    parser.add_argument("--tx-id", default=None, help="Transmitter ID, for example 0")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--data-root",
        default="C:/Users/user/Desktop/FYP/RadioMapSeer",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default uses map and transmitter IDs.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window.",
    )
    return parser.parse_args()


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def choose_sample(dataset, map_id, tx_id):
    frame = dataset.df.copy()
    frame["map_id"] = frame["map_id"].astype(str)
    frame["tx_id"] = frame["tx_id"].astype(str)

    if map_id is None and tx_id is None:
        index = len(frame) // 2
    else:
        matches = frame.index[
            (frame["map_id"] == str(map_id))
            & (frame["tx_id"] == str(tx_id))
        ].tolist()
        if not matches:
            available = frame[["map_id", "tx_id"]].head(10).to_dict("records")
            raise ValueError(
                f"Sample map_id={map_id}, tx_id={tx_id} is not in the selected split. "
                f"First available samples: {available}"
            )
        index = matches[0]

    return index, dataset[index]


def image_metrics(prediction_db, target_db):
    error = prediction_db - target_db
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def main():
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    missing = [path for path in MODEL_PATHS.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    first_checkpoint = torch.load(
        next(iter(MODEL_PATHS.values())), map_location="cpu", weights_only=False
    )
    config = first_checkpoint["config"]
    data_config = config["data"]
    data_config["root"] = args.data_root
    data_config["num_workers"] = 0
    data_config["pin_memory"] = False

    dataset = RadioMapDataset(
        data_config["manifest"],
        data_config["root"],
        args.split,
        data_config["split_dir"],
        data_config["input_mode"],
        data_config["tx_sigma"],
        False,
    )
    _, sample = choose_sample(dataset, args.map_id, args.tx_id)

    x = sample["x"].unsqueeze(0).to(device)
    target_norm = sample["y"].squeeze(0).numpy()
    min_db = float(data_config["min_db"])
    max_db = float(data_config["max_db"])
    scale_db = max_db - min_db
    target_db = target_norm * scale_db + min_db

    predictions = {}
    metrics = {}
    with torch.inference_mode():
        for name, checkpoint_path in MODEL_PATHS.items():
            _, model = load_checkpoint(checkpoint_path, device)
            prediction_norm = model(x).clamp(0.0, 1.0).squeeze().cpu().numpy()
            prediction_db = prediction_norm * scale_db + min_db
            predictions[name] = prediction_db
            metrics[name] = image_metrics(prediction_db, target_db)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    building = sample["x"][0].numpy()
    tx_map = sample["x"][1].numpy()
    distance = sample["x"][2].numpy()

    errors = {name: np.abs(pred - target_db) for name, pred in predictions.items()}
    all_errors = np.concatenate([value.ravel() for value in errors.values()])
    error_max = max(1.0, float(np.percentile(all_errors, 99)))

    figure, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    figure.suptitle(
        f"Radio-map comparison | split={args.split} | map={sample['map_id']} | Tx={sample['tx_id']}",
        fontsize=16,
    )

    axes[0, 0].imshow(building, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input: building mask")
    axes[0, 1].imshow(tx_map, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("Input: Tx heatmap")
    axes[0, 2].imshow(distance, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title("Input: normalized Tx distance")
    gt_image = axes[0, 3].imshow(target_db, cmap="turbo", vmin=min_db, vmax=max_db)
    axes[0, 3].set_title("Ground truth path gain (dB)")

    names = list(MODEL_PATHS)
    for column, name in enumerate(names):
        prediction_image = axes[1, column].imshow(
            predictions[name], cmap="turbo", vmin=min_db, vmax=max_db
        )
        result = metrics[name]
        axes[1, column].set_title(
            f"{name}\nRMSE={result['rmse']:.3f} dB | MAE={result['mae']:.3f} dB"
        )
        error_image = axes[2, column].imshow(
            errors[name], cmap="magma", vmin=0, vmax=error_max
        )
        axes[2, column].set_title(
            f"{name} absolute error\nBias={result['bias']:+.3f} dB"
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.colorbar(gt_image, ax=axes[0, 3], fraction=0.046, pad=0.04, label="Path gain (dB)")
    figure.colorbar(
        prediction_image,
        ax=list(axes[1, :]),
        fraction=0.018,
        pad=0.02,
        label="Path gain (dB)",
    )
    figure.colorbar(
        error_image,
        ax=list(axes[2, :]),
        fraction=0.018,
        pad=0.02,
        label="Absolute error (dB)",
    )

    output = Path(args.output) if args.output else Path(
        f"comparison_map_{sample['map_id']}_tx_{sample['tx_id']}.png"
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")

    print(f"Device: {device}")
    print(f"Sample: map={sample['map_id']}, tx={sample['tx_id']}, split={args.split}")
    for name in names:
        result = metrics[name]
        print(
            f"{name:12s} RMSE={result['rmse']:.4f} dB  "
            f"MAE={result['mae']:.4f} dB  Bias={result['bias']:+.4f} dB"
        )
    print(f"Saved figure: {output.resolve()}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
