import argparse
import json
import math
import random
import signal
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageDraw

from snake_jepa.snake_data import build_snake_loaders
from snake_jepa.snake_world_model import SnakePatchWorldModel, SnakePatchWorldModelConfig
from snake_jepa.train_snake_jepa import detect_device, make_model_config, move_batch, reconstruction_loss


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent",
    "levels": ["level_1", "level_2", "level_3", "random_levels"],
    "output_dir": "runs/snake_jepa_autoencoder",
    "device": "auto",
    "run_name": "snake-autoencoder",
    "seed": 11,
    "image_size": 320,
    "patch_size": 16,
    "history_size": 4,
    "batch_size": 8,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_fraction": 0.1,
    "stride": 2,
    "max_clips_per_level": 0,
    "max_windows_per_clip": 0,
    "encoder_dim": 192,
    "encoder_depth": 4,
    "encoder_heads": 4,
    "encoder_mlp_ratio": 4.0,
    "latent_dim": 96,
    "dynamics_dim": 192,
    "dynamics_depth": 3,
    "dynamics_heads": 4,
    "dynamics_mlp_ratio": 2.0,
    "decoder_dim": 192,
    "decoder_depth": 4,
    "decoder_heads": 4,
    "decoder_mlp_ratio": 4.0,
    "dropout": 0.0,
    "recon_foreground_weight": 10.0,
    "recon_foreground_threshold": 0.08,
    "recon_batch_saliency_weight": 0.0,
    "recon_batch_saliency_threshold": 0.05,
    "recon_chroma_weight": 0.0,
    "recon_chroma_threshold": 0.25,
    "recon_chroma_value_threshold": 0.2,
    "grad_clip_norm": 1.0,
    "preview_every": 1,
    "checkpoint_every": 5,
    "max_train_batches": 0,
    "max_val_batches": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train only the snake frame encoder/decoder")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-clips-per-level", type=int, default=None)
    parser.add_argument("--max-windows-per-clip", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--recon-foreground-weight", type=float, default=None)
    parser.add_argument("--recon-batch-saliency-weight", type=float, default=None)
    parser.add_argument("--recon-batch-saliency-threshold", type=float, default=None)
    parser.add_argument("--recon-chroma-weight", type=float, default=None)
    parser.add_argument("--recon-chroma-threshold", type=float, default=None)
    parser.add_argument("--recon-chroma-value-threshold", type=float, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    for key in (
        "dataset_root",
        "levels",
        "run_name",
        "device",
        "image_size",
        "patch_size",
        "epochs",
        "batch_size",
        "max_clips_per_level",
        "max_windows_per_clip",
        "max_train_batches",
        "max_val_batches",
        "recon_foreground_weight",
        "recon_batch_saliency_weight",
        "recon_batch_saliency_threshold",
        "recon_chroma_weight",
        "recon_chroma_threshold",
        "recon_chroma_value_threshold",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_dir(config: dict) -> Path:
    return Path(config["output_dir"]) / str(config["run_name"])


def get_checkpoint_dir(config: dict) -> Path:
    return get_run_dir(config) / "checkpoints"


def get_latest_checkpoint_path(config: dict) -> Path:
    return get_checkpoint_dir(config) / "latest.pt"


def encode_decode(model: SnakePatchWorldModel, frames: torch.Tensor) -> torch.Tensor:
    latents = model.encode_frame_patches(frames)
    return model.decoder(latents)


def flatten_batch_frames(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    history = batch["history_frames"]
    batch_size, history_size = history.shape[:2]
    history = history.reshape(batch_size * history_size, *history.shape[2:])
    return torch.cat([history, batch["next_frame"]], dim=0)


def run_epoch(
    model: SnakePatchWorldModel,
    loader,
    device: torch.device,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mae": 0.0, "mse": 0.0, "max_abs": 0.0}
    count = 0
    max_batches = int(config["max_train_batches"] if training else config["max_val_batches"])

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        frames = flatten_batch_frames(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            recon = encode_decode(model, frames)
            loss = reconstruction_loss(recon, frames, config)
            loss.backward()
            clip_norm = float(config.get("grad_clip_norm", 0.0))
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                recon = encode_decode(model, frames)
                loss = reconstruction_loss(recon, frames, config)

        with torch.no_grad():
            err = (recon - frames).abs()
            batch_size = frames.size(0)
            totals["loss"] += float(loss.item()) * batch_size
            totals["mae"] += float(err.mean().item()) * batch_size
            totals["mse"] += float(err.square().mean().item()) * batch_size
            totals["max_abs"] = max(totals["max_abs"], float(err.max().item()))
            count += batch_size

        if max_batches > 0 and step >= max_batches:
            break

    result = {key: value / max(1, count) for key, value in totals.items() if key != "max_abs"}
    result["max_abs"] = totals["max_abs"]
    return result


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


@torch.no_grad()
def save_preview(model: SnakePatchWorldModel, loader, device: torch.device, output_dir: Path, epoch: int) -> None:
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    frames = flatten_batch_frames(batch)[:4]
    recon = encode_decode(model, frames)
    tile_w, tile_h = _to_pil(frames[0]).size
    canvas = Image.new("RGB", (4 * tile_w, 3 * tile_h + 24), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for row, label in enumerate(["target", "recon", "diff"]):
        draw.text((6, row * tile_h + 6), label, fill=(235, 235, 235))
    for idx, (target_frame, recon_frame) in enumerate(zip(frames, recon)):
        target = _to_pil(target_frame)
        prediction = _to_pil(recon_frame)
        diff = ImageChops.difference(target, prediction)
        canvas.paste(target, (idx * tile_w, 0))
        canvas.paste(prediction, (idx * tile_w, tile_h))
        canvas.paste(diff, (idx * tile_w, 2 * tile_h))
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")


def save_checkpoint(
    model: SnakePatchWorldModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    val_metrics: dict[str, float],
    best_val: float,
    interrupted: bool = False,
) -> None:
    checkpoint_dir = get_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_type": "snake_patch_autoencoder_v1",
        "config": config,
        "val_metrics": val_metrics,
        "best_val": best_val,
        "model_config": asdict(model.cfg),
        "interrupted": interrupted,
    }
    torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    if val_metrics["loss"] <= best_val:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def maybe_resume(
    model: SnakePatchWorldModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
) -> tuple[int, float]:
    latest_path = get_latest_checkpoint_path(config)
    if not latest_path.exists():
        return 1, math.inf
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val", math.inf))
    print(f"resuming from {latest_path} at epoch {start_epoch}")
    return start_epoch, best_val


def main() -> None:
    args = parse_args()
    config = load_config(args)
    set_seed(int(config["seed"]))

    device = detect_device(str(config["device"]))
    run_dir = get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_snake_loaders(
        config["dataset_root"],
        levels=list(config["levels"]) if config.get("levels") else None,
        history_size=int(config["history_size"]),
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        val_fraction=float(config["val_fraction"]),
        num_workers=int(config["num_workers"]),
        stride=int(config["stride"]),
        max_clips_per_level=int(config["max_clips_per_level"]),
        max_windows_per_clip=int(config["max_windows_per_clip"]),
    )

    model_config: SnakePatchWorldModelConfig = make_model_config(config)
    model = SnakePatchWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch, best_val = maybe_resume(model, optimizer, config, device)

    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = model.count_parameters()
    with (run_dir / "train_config.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2)

    metrics_path = run_dir / "metrics.jsonl"
    print(f"device: {device}")
    print(f"train frames per epoch: {len(train_loader.dataset) * (int(config['history_size']) + 1)}")
    print(f"val frames per epoch: {len(val_loader.dataset) * (int(config['history_size']) + 1)}")
    print(f"parameters: {model.count_parameters():,}")
    print(f"run dir: {run_dir}")

    stop_requested = {"value": False}

    def _request_stop(signum, _frame) -> None:
        if not stop_requested["value"]:
            stop_requested["value"] = True
            print(f"received signal {signum}, stopping after the current epoch")
        else:
            raise KeyboardInterrupt

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    last_epoch = start_epoch - 1
    last_val_metrics = {"loss": math.inf}
    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            train_metrics = run_epoch(model, train_loader, device, config, optimizer)
            val_metrics = run_epoch(model, val_loader, device, config)
            last_epoch = epoch
            last_val_metrics = val_metrics
            print(
                f"epoch {epoch:03d} | train {train_metrics['loss']:.4f} | val {val_metrics['loss']:.4f} | "
                f"mae {val_metrics['mae']:.4f} | max_abs {val_metrics['max_abs']:.4f}"
            )

            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")

            if epoch % int(config["preview_every"]) == 0:
                save_preview(model, val_loader, device, run_dir, epoch)

            previous_best = best_val
            best_val = min(best_val, val_metrics["loss"])
            if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= previous_best:
                save_checkpoint(model, optimizer, config_dump, epoch, val_metrics, best_val)

            if stop_requested["value"]:
                save_checkpoint(model, optimizer, config_dump, epoch, val_metrics, best_val, interrupted=True)
                print("graceful shutdown complete")
                break
    except KeyboardInterrupt:
        print("keyboard interrupt received, saving latest checkpoint")
        save_checkpoint(
            model,
            optimizer,
            config_dump,
            max(last_epoch, start_epoch - 1),
            last_val_metrics,
            best_val,
            interrupted=True,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
