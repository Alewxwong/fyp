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

MODELS = [
    ("U-Net", "unet"),
    ("RadioUNet-style", "radiounet"),
    ("FNO", "fno"),
    ("U-FNO", "ufno"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Compare four models on one identical RadioMapSeer sample.")
    p.add_argument("--output-root", required=True, help="outputs for parameter experiment or outputs_flops for FLOP experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-name", default="best.pt")
    p.add_argument("--map-id", default=None)
    p.add_argument("--tx-id", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", default="C:/Users/user/Desktop/FYP/RadioMapSeer")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def checkpoint_paths(output_root, seed, checkpoint_name):
    return {
        display: Path(output_root) / folder / f"seed_{seed}" / checkpoint_name
        for display, folder in MODELS
    }


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        matches = frame.index[(frame["map_id"] == str(map_id)) & (frame["tx_id"] == str(tx_id))].tolist()
        if not matches:
            examples = frame[["map_id", "tx_id"]].head(15).to_dict("records")
            raise ValueError(f"map_id={map_id}, tx_id={tx_id} is not in split={dataset.split}. Examples: {examples}")
        index = matches[0]
    return dataset[index]


def metrics(pred_db, target_db, min_db, max_db):
    error = pred_db - target_db
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    nrmse = rmse / float(max_db - min_db)
    relative_l2 = float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(target_db.ravel()), 1e-12))
    return {"rmse_db": rmse, "mae_db": mae, "bias_db": bias, "nrmse": nrmse, "relative_l2": relative_l2}


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    paths = checkpoint_paths(args.output_root, args.seed, args.checkpoint_name)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    first = torch.load(next(iter(paths.values())), map_location="cpu", weights_only=False)
    cfg = first["config"]
    d = cfg["data"]
    d["root"] = args.data_root
    d["num_workers"] = 0
    d["pin_memory"] = False
    dataset = RadioMapDataset(d["manifest"], d["root"], args.split, d["split_dir"], d["input_mode"], d["tx_sigma"], False)
    sample = choose_sample(dataset, args.map_id, args.tx_id)

    x = sample["x"].unsqueeze(0).to(device)
    target_norm = sample["y"].squeeze(0).numpy()
    min_db, max_db = float(d["min_db"]), float(d["max_db"])
    target_db = target_norm * (max_db - min_db) + min_db

    predictions, results = {}, {}
    with torch.inference_mode():
        for name, path in paths.items():
            _, model = load_model(path, device)
            pred_norm = model(x).clamp(0, 1).squeeze().cpu().numpy()
            pred_db = pred_norm * (max_db - min_db) + min_db
            predictions[name] = pred_db
            results[name] = metrics(pred_db, target_db, min_db, max_db)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    errors = {name: np.abs(pred - target_db) for name, pred in predictions.items()}
    error_limit = max(1.0, float(np.percentile(np.concatenate([x.ravel() for x in errors.values()]), 99)))

    fig, axes = plt.subplots(3, 4, figsize=(19, 13), constrained_layout=True)
    fig.suptitle(f"{Path(args.output_root).name} | seed={args.seed} | split={args.split} | map={sample['map_id']} | Tx={sample['tx_id']}", fontsize=16)

    axes[0, 0].imshow(sample["x"][0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input: building mask")
    axes[0, 1].imshow(sample["x"][1].numpy(), cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("Input: Tx heatmap")
    axes[0, 2].imshow(sample["x"][2].numpy(), cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title("Input: normalized distance")
    gt_im = axes[0, 3].imshow(target_db, cmap="turbo", vmin=min_db, vmax=max_db)
    axes[0, 3].set_title("Ground truth (dB)")

    pred_im = err_im = None
    for col, (name, _) in enumerate(MODELS):
        pred_im = axes[1, col].imshow(predictions[name], cmap="turbo", vmin=min_db, vmax=max_db)
        m = results[name]
        axes[1, col].set_title(f"{name}\nRMSE={m['rmse_db']:.3f} dB | MAE={m['mae_db']:.3f} dB\nnRMSE={m['nrmse']:.4f} | RelL2={m['relative_l2']:.4f}")
        err_im = axes[2, col].imshow(errors[name], cmap="magma", vmin=0, vmax=error_limit)
        axes[2, col].set_title(f"{name} absolute error\nBias={m['bias_db']:+.3f} dB")

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(gt_im, ax=axes[0, 3], fraction=0.046, pad=0.04, label="Path gain (dB)")
    fig.colorbar(pred_im, ax=list(axes[1, :]), fraction=0.018, pad=0.02, label="Path gain (dB)")
    fig.colorbar(err_im, ax=list(axes[2, :]), fraction=0.018, pad=0.02, label="Absolute error (dB)")

    output = Path(args.output) if args.output else Path(f"comparison_{Path(args.output_root).name}_seed{args.seed}_map{sample['map_id']}_tx{sample['tx_id']}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")

    print(f"Device: {device}")
    print(f"Sample: map={sample['map_id']} tx={sample['tx_id']} split={args.split}")
    for name, _ in MODELS:
        m = results[name]
        print(f"{name:18s} RMSE={m['rmse_db']:.4f} dB  MAE={m['mae_db']:.4f} dB  Bias={m['bias_db']:+.4f} dB  nRMSE={m['nrmse']:.6f}  RelL2={m['relative_l2']:.6f}")
    print(f"Saved: {output.resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
