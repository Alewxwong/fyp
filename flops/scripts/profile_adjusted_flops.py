import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.models import build_model
from src.utils import load_config
from scripts.flop_utils import measure_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    cfg["model"]["in_channels"] = 2 if cfg["data"]["input_mode"] == "two" else 5
    model = build_model(cfg["model"]).eval()
    x = torch.randn(1, cfg["model"]["in_channels"], a.height, a.width)
    result = measure_model(model, cfg, x)
    result["model"] = cfg["model"]["name"]
    result["config"] = a.config
    result["supported_gflops"] = result["supported_flops"] / 1e9
    result["fft_gflops"] = result["fft_estimate"] / 1e9
    result["adjusted_gflops"] = result["adjusted_flops"] / 1e9
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Model: {result['model']}")
        print(f"Parameters: {result['parameters']:,}")
        print(f"fvcore supported GFLOPs: {result['supported_gflops']:.6f}")
        print(f"Analytical FFT GFLOPs: {result['fft_gflops']:.6f}")
        print(f"Adjusted analytical GFLOPs: {result['adjusted_gflops']:.6f}")
        print(f"Unsupported operations: {result['unsupported_ops']}")


if __name__ == "__main__":
    main()
