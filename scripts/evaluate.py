import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import RadioMapDataset
from src.metrics import batch_sums, finalize
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--data-root",
        default="C:/Users/user/Desktop/ufno/RadioMapSeer",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    data_config = config["data"]

    # Override server settings for local Windows evaluation.
    data_config["root"] = args.data_root
    data_config["num_workers"] = args.num_workers
    data_config["pin_memory"] = device.type == "cuda"

    dataset = RadioMapDataset(
        data_config["manifest"],
        data_config["root"],
        args.split,
        data_config["split_dir"],
        data_config["input_mode"],
        data_config["tx_sigma"],
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=data_config["pin_memory"],
    )

    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    totals = {
        "se": 0.0,
        "ae": 0.0,
        "n": 0,
        "nse": 0.0,
        "den": 0.0,
    }
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {args.split}", dynamic_ncols=True):
            prediction = model(batch["x"].to(device)).clamp(0, 1).cpu().numpy()
            target = batch["y"].numpy()

            for index in range(len(prediction)):
                sums = batch_sums(
                    prediction[index:index + 1],
                    target[index:index + 1],
                    data_config["min_db"],
                    data_config["max_db"],
                )
                metrics = finalize(sums)
                rows.append(
                    {
                        "map_id": batch["map_id"][index],
                        "tx_id": batch["tx_id"][index],
                        **metrics,
                    }
                )
                for key in totals:
                    totals[key] += sums[key]

    results = finalize(totals)
    output_dir = Path(args.checkpoint).parent
    (output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        output_dir / f"{args.split}_per_sample.csv",
        index=False,
    )

    print(json.dumps(results, indent=2))
    print(f"Saved: {output_dir / f'{args.split}_metrics.json'}")
    print(f"Saved: {output_dir / f'{args.split}_per_sample.csv'}")


if __name__ == "__main__":
    main()
