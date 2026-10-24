import argparse
import csv
import json
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import RadioMapDataset
from src.losses import make_loss
from src.metrics import batch_sums, finalize
from src.models import build_model
from src.utils import atomic_save, load_config, seed_everything


def make_loader(cfg, split):
    d = cfg["data"]
    dataset = RadioMapDataset(
        d["manifest"], d["root"], split, d["split_dir"],
        d["input_mode"], d["tx_sigma"],
        d["augment"] and split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=d["num_workers"],
        pin_memory=d["pin_memory"],
        persistent_workers=d["num_workers"] > 0,
        drop_last=False,
    )


def run_epoch(model, loader, device, loss_fn, optimizer, scaler, cfg,
              training, epoch, epochs, max_batches=None):
    model.train(training)
    phase = "Train" if training else "Valid"
    total_loss = 0.0
    processed = 0
    sums = {"se": 0.0, "ae": 0.0, "n": 0, "nse": 0.0, "den": 0.0}
    available = len(loader)
    shown_total = min(available, max_batches) if max_batches is not None else available

    batches = enumerate(loader)
    if max_batches is not None:
        batches = islice(batches, max_batches)
    progress = tqdm(
        batches,
        total=shown_total,
        desc=f"Epoch {epoch:02d}/{epochs:02d} {phase}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_idx, batch in progress:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
            ):
                prediction = model(x)
                loss = loss_fn(prediction, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {phase.lower()} loss at epoch {epoch}, "
                    f"batch {batch_idx}: {loss.item()}"
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()

            batch_size = x.size(0)
            processed += batch_size
            total_loss += loss.item() * batch_size
            metric_sums = batch_sums(
                prediction.detach().float().cpu().numpy(),
                y.detach().cpu().numpy(),
                cfg["data"]["min_db"],
                cfg["data"]["max_db"],
            )
            for key in sums:
                sums[key] += metric_sums[key]

            running = finalize(sums)
            progress.set_postfix(
                loss=f"{total_loss / max(processed, 1):.5f}",
                rmse=f"{running['rmse_db']:.3f}dB",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

    if processed == 0:
        raise RuntimeError(f"No samples were processed in the {phase.lower()} loader")
    return total_loss / processed, finalize(sums), processed


def append_history(path, row):
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    cfg["model"]["in_channels"] = 2 if cfg["data"]["input_mode"] == "two" else 5
    model = build_model(cfg["model"]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    epochs = args.max_epochs or cfg["training"]["epochs"]

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    if cfg["training"].get("scheduler", "cosine") != "cosine":
        raise ValueError("This final pipeline currently supports scheduler: cosine")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["training"].get("min_lr", 1e-6)
    )
    loss_fn = make_loss(
        cfg["training"]["loss"], cfg["training"]["gradient_weight"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg["training"]["amp"] and device.type == "cuda"
    )

    run_dir = (
        Path(cfg["output_dir"])
        / cfg["model"]["name"]
        / f"seed_{cfg['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    start_epoch = 1
    best_loss = float("inf")
    bad_epochs = 0
    if args.resume:
        checkpoint_path = run_dir / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Train/validation samples: {len(train_loader.dataset):,}/{len(val_loader.dataset):,}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Epochs: {epochs}")
    print(f"Initial LR: {cfg['training']['lr']:.2e}; scheduler: cosine")

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_loss, train_metrics, train_n = run_epoch(
                model, train_loader, device, loss_fn, optimizer, scaler, cfg,
                True, epoch, epochs, args.max_train_batches,
            )
            val_loss, val_metrics, val_n = run_epoch(
                model, val_loader, device, loss_fn, optimizer, scaler, cfg,
                False, epoch, epochs, args.max_val_batches,
            )
            epoch_seconds = time.perf_counter() - epoch_start
            peak_memory_gb = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda" else 0.0
            )
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse_db": train_metrics["rmse_db"],
                "val_rmse_db": val_metrics["rmse_db"],
                "val_mae_db": val_metrics["mae_db"],
                "val_relative_l2": val_metrics["relative_l2"],
                "train_samples": train_n,
                "val_samples": val_n,
                "epoch_seconds": epoch_seconds,
                "peak_gpu_memory_gb": peak_memory_gb,
            }
            append_history(run_dir / "history.csv", row)

            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars(
                "rmse_db",
                {"train": train_metrics["rmse_db"], "val": val_metrics["rmse_db"]},
                epoch,
            )
            writer.add_scalar("optimization/lr", current_lr, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)
            writer.add_scalar("memory/peak_gpu_gb", peak_memory_gb, epoch)

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "val": val_metrics,
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "parameter_count": parameter_count,
            }
            atomic_save(state, run_dir / "last.pt")
            if improved:
                atomic_save(state, run_dir / "best.pt")

            print(
                f"Epoch {epoch:02d}/{epochs:02d} complete | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"val RMSE={val_metrics['rmse_db']:.3f} dB | "
                f"time={epoch_seconds:.1f}s | peak GPU={peak_memory_gb:.2f} GB | "
                f"best={'yes' if improved else 'no'}"
            )

            if bad_epochs >= cfg["training"]["patience"]:
                print(
                    f"Early stopping: no validation-loss improvement for "
                    f"{bad_epochs} epochs."
                )
                break
    finally:
        writer.close()

    print(f"Training finished. Results: {run_dir}")
    print(f"Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
