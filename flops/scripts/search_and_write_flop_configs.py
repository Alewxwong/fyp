import argparse
import csv
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.models import build_model
from src.utils import load_config
from scripts.flop_utils import measure_model


def candidates(name):
    common = {"in_channels": 5, "out_channels": 1}
    if name == "unet":
        for b in range(20, 37):
            yield {**common, "name": name, "base_width": b}
    elif name == "radiounet":
        for b in range(14, 27):
            yield {**common, "name": name, "base_width": b}
    elif name == "fno":
        for w in range(40, 137, 2):
            yield {**common, "name": name, "width": w, "modes1": 20, "modes2": 20, "depth": 4, "padding": 9}
    elif name == "ufno":
        for w in range(20, 45):
            yield {**common, "name": name, "width": w, "modes1": 20, "modes2": 20, "depth": 4, "padding": 9}


def write_config(path, model_config):
    lines = ["base: ../base_flops.yaml", "", "model:", f"  name: {model_config['name']}"]
    if "base_width" in model_config:
        lines.append(f"  base_width: {model_config['base_width']}")
    else:
        for key in ["width", "modes1", "modes2", "depth", "padding"]:
            lines.append(f"  {key}: {model_config[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-gflops", type=float, default=11.0)
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--output-csv", default="flop_search_results.csv")
    p.add_argument("--write-configs", action="store_true")
    a = p.parse_args()
    base = load_config("configs/base_flops.yaml")
    x = torch.randn(1, 5, 256, 256)
    rows = []
    selections = {}
    for name in ["unet", "radiounet", "fno", "ufno"]:
        best = None
        print(f"Searching {name} ...", flush=True)
        for mc in candidates(name):
            cfg = copy.deepcopy(base)
            cfg["model"].update(mc)
            model = build_model(cfg["model"]).eval()
            result = measure_model(model, cfg, x)
            g = result["adjusted_flops"] / 1e9
            row = {
                "model": name,
                "base_width": mc.get("base_width", ""),
                "width": mc.get("width", ""),
                "modes1": mc.get("modes1", ""),
                "parameters": result["parameters"],
                "supported_gflops": result["supported_flops"] / 1e9,
                "fft_gflops": result["fft_estimate"] / 1e9,
                "adjusted_gflops": g,
                "difference_from_target": abs(g - a.target_gflops),
                "within_tolerance": abs(g - a.target_gflops) <= a.target_gflops * a.tolerance,
            }
            rows.append(row)
            key = (row["difference_from_target"], row["parameters"])
            if best is None or key < best[0]:
                best = (key, mc, row)
            del model
        selections[name] = best
        print(f"Selected {name}: {best[2]}")
    with open(a.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    if a.write_configs:
        out = Path("configs/flops"); out.mkdir(parents=True, exist_ok=True)
        for name, (_, mc, row) in selections.items():
            if not row["within_tolerance"]:
                raise RuntimeError(f"No {name} candidate is within tolerance. Best: {row}")
            write_config(out / f"{name}.yaml", mc)
    print(f"Saved search table: {a.output_csv}")
    if a.write_configs:
        print("Wrote final configs to configs/flops/")


if __name__ == "__main__":
    main()
