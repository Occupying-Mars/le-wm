import argparse
import json
import math
import random
import signal
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import wandb

from debug_box_data import build_debug_box_loaders
from debug_box_world_model import TinyViTEncoder, LatentDecoder


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/world-env/datasets/debug_box",
    "output_dir": "runs/debug_box_autoencoder",
    "device": "auto",
    "run_name": "debug-box-autoencoder",
    "wandb_project": "debug-box-world-model",
    "wandb_entity": "krishnapg2315",
    "seed": 7,
    "image_size": 128,
    "patch_size": 16,
    "batch_size": 8,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_fraction": 0.1,
    "encoder_dim": 128,
    "encoder_depth": 4,
    "encoder_heads": 4,
    "encoder_mlp_ratio": 4.0,
    "decoder_dim": 128,
    "decoder_depth": 4,
    "decoder_heads": 4,
    "decoder_mlp_ratio": 4.0,
    "dropout": 0.0,
    "preview_every": 1,
    "checkpoint_every": 5,
    "max_train_batches": 0,
    "max_val_batches": 0,
}


class DebugBoxAutoencoder(torch.nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.encoder = TinyViTEncoder(
            image_size=int(config["image_size"]),
            patch_size=int(config["patch_size"]),
            dim=int(config["encoder_dim"]),
            depth=int(config["encoder_depth"]),
            heads=int(config["encoder_heads"]),
            mlp_ratio=float(config["encoder_mlp_ratio"]),
            latent_dim=int(config["encoder_dim"]),
            dropout=float(config["dropout"]),
        )
        self.decoder = LatentDecoder(
            latent_dim=int(config["encoder_dim"]),
            image_size=int(config["image_size"]),
            patch_size=int(config["patch_size"]),
            hidden_dim=int(config["decoder_dim"]),
            depth=int(config["decoder_depth"]),
            heads=int(config["decoder_heads"]),
            mlp_ratio=float(config["decoder_mlp_ratio"]),
            dropout=float(config["dropout"]),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_cls(frames)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_latent = self.encode(frames)
        recon = self.decoder(cls_latent)
        return {"cls_latent": cls_latent, "recon": recon}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a debug-box cls autoencoder diagnostic")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r") as handle:
                config.update(json.load(handle))
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_losses(batch: dict[str, torch.Tensor], recon: torch.Tensor) -> dict[str, torch.Tensor]:
    target = batch["next_frame"]
    recon_loss = F.l1_loss(recon, target)
    mse_loss = F.mse_loss(recon, target)
    return {"loss": recon_loss + mse_loss, "recon_loss": recon_loss, "mse_loss": mse_loss}


def _spatial_moments(frames: torch.Tensor) -> torch.Tensor:
    gray = frames.mean(dim=1)
    weights = (gray - gray.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-6)
    h, w = gray.shape[-2:]
    ys = torch.linspace(0.0, 1.0, h, device=frames.device, dtype=frames.dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=frames.device, dtype=frames.dtype).view(1, 1, w)
    norm = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mean_y = (weights * ys).sum(dim=(-2, -1), keepdim=True) / norm
    mean_x = (weights * xs).sum(dim=(-2, -1), keepdim=True) / norm
    return torch.cat([mean_x, mean_y], dim=-1).squeeze(-2)


@torch.no_grad()
def compute_diagnostics(model: DebugBoxAutoencoder, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    cls_latent = model.encode(batch["next_frame"])
    recon = model.decoder(cls_latent)
    zero_recon = model.decoder(torch.zeros_like(cls_latent))
    gt_center = _spatial_moments(batch["next_frame"])
    pred_center = _spatial_moments(recon)
    return {
        "cls_std": float(cls_latent.std(dim=0).mean().item()),
        "cls_norm": float(cls_latent.norm(dim=-1).mean().item()),
        "zero_latent_decode_delta": float(F.l1_loss(recon, zero_recon).item()),
        "center_error": float(F.l1_loss(pred_center, gt_center).item()),
    }


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def create_preview_image(target_next: torch.Tensor, pred_next: torch.Tensor) -> Image.Image:
    target_tile = _to_pil(target_next)
    pred_tile = _to_pil(pred_next)
    tile_w, tile_h = target_tile.size
    canvas = Image.new("RGB", (2 * tile_w, tile_h + 32), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "target", fill=(235, 235, 235))
    draw.text((tile_w + 8, 8), "recon", fill=(235, 235, 235))
    canvas.paste(target_tile, (0, 32))
    canvas.paste(pred_tile, (tile_w, 32))
    return canvas


@torch.no_grad()
def save_preview(model: DebugBoxAutoencoder, loader, device: torch.device, output_dir: Path, epoch: int) -> Image.Image:
    batch = next(iter(loader))
    batch = move_batch(batch, device)
    recon = model(batch["next_frame"])["recon"]
    canvas = create_preview_image(batch["next_frame"][0], recon[0])
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")
    return canvas


def train_epoch(model: DebugBoxAutoencoder, loader, optimizer: torch.optim.Optimizer, device: torch.device, config: dict) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon_loss": 0.0, "mse_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["next_frame"])
        losses = compute_losses(batch, output["recon"])
        losses["loss"].backward()
        optimizer.step()
        batch_size = batch["next_frame"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if int(config["max_train_batches"]) > 0 and step >= int(config["max_train_batches"]):
            break
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: DebugBoxAutoencoder, loader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon_loss": 0.0, "mse_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        output = model(batch["next_frame"])
        losses = compute_losses(batch, output["recon"])
        batch_size = batch["next_frame"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if int(config["max_val_batches"]) > 0 and step >= int(config["max_val_batches"]):
            break
    return {key: value / max(1, count) for key, value in totals.items()}


def get_run_dir(config: dict) -> Path:
    return Path(config["output_dir"]) / str(config["run_name"])


def get_checkpoint_dir(config: dict) -> Path:
    return get_run_dir(config) / "checkpoints"


def get_latest_checkpoint_path(config: dict) -> Path:
    return get_checkpoint_dir(config) / "latest.pt"


def save_checkpoint(model: DebugBoxAutoencoder, optimizer: torch.optim.Optimizer, config: dict, checkpoint_dir: Path, epoch: int, val_metrics: dict[str, float], best_val: float, interrupted: bool = False) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "val_metrics": val_metrics,
        "best_val": best_val,
        "interrupted": interrupted,
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, latest_path)
    if val_metrics["loss"] <= best_val:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def maybe_resume(model: DebugBoxAutoencoder, optimizer: torch.optim.Optimizer, config: dict, device: torch.device) -> tuple[int, float]:
    latest_path = get_latest_checkpoint_path(config)
    if not latest_path.exists():
        return 1, math.inf
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer_state = checkpoint.get("optimizer_state")
    optimizer_loaded = False
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_loaded = True
        except ValueError as exc:
            print(f"optimizer state mismatch at {latest_path}, skipping optimizer resume: {exc}")
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val", math.inf))
    if optimizer_loaded:
        print(f"resuming autoencoder and optimizer from {latest_path} at epoch {start_epoch}")
    else:
        print(f"resuming autoencoder weights only from {latest_path} at epoch {start_epoch}")
    return start_epoch, best_val


def init_wandb(config: dict) -> wandb.sdk.wandb_run.Run:
    run_dir = get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=str(config["wandb_project"]),
        entity=str(config["wandb_entity"]),
        name=str(config["run_name"]),
        id=str(config["run_name"]),
        resume="allow",
        dir=str(run_dir),
        config=config,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args)
    set_seed(int(config["seed"]))

    device = detect_device(str(config["device"]))
    run_dir = get_run_dir(config)
    checkpoint_dir = get_checkpoint_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_debug_box_loaders(
        config["dataset_root"],
        history_size=1,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        val_fraction=float(config["val_fraction"]),
        num_workers=int(config["num_workers"]),
    )

    model = DebugBoxAutoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    start_epoch, best_val = maybe_resume(model, optimizer, config, device)
    wandb_run = init_wandb(config)

    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (run_dir / "train_config.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2)

    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"autoencoder parameters: {config_dump['parameter_count']:,}")
    print(f"wandb run: {wandb_run.name}")

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
    last_val_metrics = {"loss": math.inf, "recon_loss": math.inf, "mse_loss": math.inf}
    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, device, config)
            val_metrics = evaluate(model, val_loader, device, config)
            diagnostics = compute_diagnostics(model, move_batch(next(iter(val_loader)), device))
            last_epoch = epoch
            last_val_metrics = val_metrics
            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"recon {val_metrics['recon_loss']:.4f} | "
                f"mse {val_metrics['mse_loss']:.4f}"
            )

            log_payload = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/recon_loss": train_metrics["recon_loss"],
                "train/mse_loss": train_metrics["mse_loss"],
                "val/loss": val_metrics["loss"],
                "val/recon_loss": val_metrics["recon_loss"],
                "val/mse_loss": val_metrics["mse_loss"],
                "best_val/loss": min(best_val, val_metrics["loss"]),
                "diag/cls_std": diagnostics["cls_std"],
                "diag/cls_norm": diagnostics["cls_norm"],
                "diag/zero_latent_decode_delta": diagnostics["zero_latent_decode_delta"],
                "diag/center_error": diagnostics["center_error"],
            }

            if epoch % int(config["preview_every"]) == 0:
                preview_image = save_preview(model, val_loader, device, run_dir, epoch)
                log_payload["val/example"] = wandb.Image(preview_image, caption=f"epoch {epoch}")

            wandb_run.log(log_payload, step=epoch)

            previous_best = best_val
            best_val = min(best_val, val_metrics["loss"])
            if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= previous_best:
                save_checkpoint(model, optimizer, config_dump, checkpoint_dir, epoch, val_metrics, best_val)

            if stop_requested["value"]:
                print("stop requested, saving latest checkpoint and ending run")
                save_checkpoint(model, optimizer, config_dump, checkpoint_dir, epoch, val_metrics, best_val, interrupted=True)
                break
    except KeyboardInterrupt:
        print("keyboard interrupt received, saving latest checkpoint")
        save_checkpoint(model, optimizer, config_dump, checkpoint_dir, max(last_epoch, start_epoch - 1), last_val_metrics, best_val, interrupted=True)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        wandb_run.finish()


if __name__ == "__main__":
    main()
