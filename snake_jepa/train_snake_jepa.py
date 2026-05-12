import argparse
import json
import math
import random
import signal
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from snake_jepa.snake_data import build_snake_loaders
from snake_jepa.snake_world_model import SnakePatchWorldModel, SnakePatchWorldModelConfig


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent",
    "levels": ["level_1", "level_2", "level_3", "random_levels"],
    "output_dir": "runs/snake_jepa",
    "device": "auto",
    "run_name": "snake-jepa",
    "seed": 7,
    "image_size": 128,
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
    "latent_loss_weight": 1.0,
    "pred_recon_loss_weight": 1.0,
    "target_recon_loss_weight": 0.5,
    "history_recon_loss_weight": 0.1,
    "recon_foreground_weight": 10.0,
    "recon_foreground_threshold": 0.08,
    "sigreg_weight": 0.03,
    "sigreg_knots": 17,
    "sigreg_num_proj": 512,
    "grad_clip_norm": 1.0,
    "preview_every": 1,
    "checkpoint_every": 5,
    "max_train_batches": 0,
    "max_val_batches": 0,
    "wandb_enabled": False,
    "wandb_project": "snake-jepa",
    "wandb_entity": "krishnapg2315",
}


class SIGReg(torch.nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 512) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a lewm-style jepa world model on snake frames")
    parser.add_argument("--config", type=str, default="config/train/snake_jepa.json")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-clips-per-level", type=int, default=None)
    parser.add_argument("--max-windows-per-clip", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--recon-foreground-weight", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(args.config)
    if config_path.exists():
        config.update(json.loads(config_path.read_text()))
    if args.dataset_root is not None:
        config["dataset_root"] = args.dataset_root
    if args.levels is not None:
        config["levels"] = args.levels
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_clips_per_level is not None:
        config["max_clips_per_level"] = args.max_clips_per_level
    if args.max_windows_per_clip is not None:
        config["max_windows_per_clip"] = args.max_windows_per_clip
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        config["max_val_batches"] = args.max_val_batches
    if args.sigreg_weight is not None:
        config["sigreg_weight"] = args.sigreg_weight
    if args.recon_foreground_weight is not None:
        config["recon_foreground_weight"] = args.recon_foreground_weight
    if args.wandb:
        config["wandb_enabled"] = True
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


def make_model_config(config: dict) -> SnakePatchWorldModelConfig:
    return SnakePatchWorldModelConfig(
        image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]),
        history_size=int(config["history_size"]),
        action_dim=4,
        encoder_dim=int(config["encoder_dim"]),
        encoder_depth=int(config["encoder_depth"]),
        encoder_heads=int(config["encoder_heads"]),
        encoder_mlp_ratio=float(config["encoder_mlp_ratio"]),
        latent_dim=int(config["latent_dim"]),
        dynamics_dim=int(config["dynamics_dim"]),
        dynamics_depth=int(config["dynamics_depth"]),
        dynamics_heads=int(config["dynamics_heads"]),
        dynamics_mlp_ratio=float(config["dynamics_mlp_ratio"]),
        decoder_dim=int(config["decoder_dim"]),
        decoder_depth=int(config["decoder_depth"]),
        decoder_heads=int(config["decoder_heads"]),
        decoder_mlp_ratio=float(config["decoder_mlp_ratio"]),
        dropout=float(config["dropout"]),
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
    threshold = float(config.get("recon_foreground_threshold", 0.08))
    foreground_weight = float(config.get("recon_foreground_weight", 0.0))
    mask = (target.amax(dim=-3, keepdim=True) > threshold).float()
    weight = 1.0 + foreground_weight * mask
    l1 = ((pred - target).abs() * weight).mean()
    mse = ((pred - target).square() * weight).mean()
    return l1 + mse


def compute_losses(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    latent_loss = F.mse_loss(output["pred_next_latent"], output["target_next_latent"].detach())
    pred_recon_loss = reconstruction_loss(output["pred_next_frame"], batch["next_frame"], config)
    target_recon_loss = reconstruction_loss(output["target_next_recon"], batch["next_frame"], config)
    history_recon_loss = reconstruction_loss(output["history_recon"], batch["history_frames"], config)
    sigreg_loss = output["sigreg_loss"]

    total = (
        float(config["latent_loss_weight"]) * latent_loss
        + float(config["pred_recon_loss_weight"]) * pred_recon_loss
        + float(config["target_recon_loss_weight"]) * target_recon_loss
        + float(config["history_recon_loss_weight"]) * history_recon_loss
        + float(config["sigreg_weight"]) * sigreg_loss
    )
    return {
        "loss": total,
        "latent_loss": latent_loss,
        "pred_recon_loss": pred_recon_loss,
        "target_recon_loss": target_recon_loss,
        "history_recon_loss": history_recon_loss,
        "sigreg_loss": sigreg_loss,
    }


def sigreg_history_input(history_latents: torch.Tensor) -> torch.Tensor:
    if history_latents.dim() == 3:
        return history_latents.transpose(0, 1)
    if history_latents.dim() == 4:
        batch_size, history_size, num_patches, latent_dim = history_latents.shape
        return history_latents.permute(1, 0, 2, 3).reshape(
            history_size,
            batch_size * num_patches,
            latent_dim,
        )
    raise ValueError(f"unsupported history_latents shape: {tuple(history_latents.shape)}")


def get_run_dir(config: dict) -> Path:
    return Path(config["output_dir"]) / str(config["run_name"])


def get_checkpoint_dir(config: dict) -> Path:
    return get_run_dir(config) / "checkpoints"


def get_latest_checkpoint_path(config: dict) -> Path:
    return get_checkpoint_dir(config) / "latest.pt"


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def create_preview_image(history: torch.Tensor, target_next: torch.Tensor, pred_next: torch.Tensor, target_recon: torch.Tensor) -> Image.Image:
    history_tiles = [_to_pil(frame) for frame in history]
    target_tiles = [_to_pil(target_next) for _ in history_tiles]
    pred_tiles = [_to_pil(pred_next) for _ in history_tiles]
    recon_tiles = [_to_pil(target_recon) for _ in history_tiles]

    tile_w, tile_h = history_tiles[0].size
    columns = len(history_tiles)
    rows = 4
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h + 28), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    labels = ["history", "target next", "pred next", "target recon"]
    for row, label in enumerate(labels):
        draw.text((6, row * tile_h + 6), label, fill=(235, 235, 235))

    for idx, tile in enumerate(history_tiles):
        canvas.paste(tile, (idx * tile_w, 0))
    for idx, tile in enumerate(target_tiles):
        canvas.paste(tile, (idx * tile_w, tile_h))
    for idx, tile in enumerate(pred_tiles):
        canvas.paste(tile, (idx * tile_w, 2 * tile_h))
    for idx, tile in enumerate(recon_tiles):
        canvas.paste(tile, (idx * tile_w, 3 * tile_h))
    return canvas


@torch.no_grad()
def save_preview(model: SnakePatchWorldModel, loader, device: torch.device, output_dir: Path, epoch: int) -> Image.Image:
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    output = model(batch["history_frames"], batch["actions_one_hot"], batch["next_frame"])
    canvas = create_preview_image(
        batch["history_frames"][0],
        batch["next_frame"][0],
        output["pred_next_frame"][0],
        output["target_next_recon"][0],
    )
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")
    return canvas


def run_epoch(
    model: SnakePatchWorldModel,
    sigreg: SIGReg,
    loader,
    device: torch.device,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "pred_recon_loss": 0.0,
        "target_recon_loss": 0.0,
        "history_recon_loss": 0.0,
        "sigreg_loss": 0.0,
    }
    count = 0
    max_batches = int(config["max_train_batches"] if training else config["max_val_batches"])

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["history_frames"], batch["actions_one_hot"], batch["next_frame"])
            output["sigreg_loss"] = sigreg(sigreg_history_input(output["history_latents"]))
            losses = compute_losses(output, batch, config)
            losses["loss"].backward()
            clip_norm = float(config.get("grad_clip_norm", 0.0))
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(batch["history_frames"], batch["actions_one_hot"], batch["next_frame"])
                output["sigreg_loss"] = sigreg(sigreg_history_input(output["history_latents"]))
                losses = compute_losses(output, batch, config)

        batch_size = batch["next_frame"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if max_batches > 0 and step >= max_batches:
            break

    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    model: SnakePatchWorldModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    checkpoint_dir: Path,
    epoch: int,
    val_metrics: dict[str, float],
    best_val: float,
    interrupted: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_type": "snake_patch_world_model_v1",
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
    optimizer_loaded = False
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_loaded = True
    except (KeyError, ValueError) as exc:
        print(f"optimizer state mismatch at {latest_path}, skipping optimizer resume: {exc}")
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val", math.inf))
    if optimizer_loaded:
        print(f"resuming model and optimizer from {latest_path} at epoch {start_epoch}")
    else:
        print(f"resuming model weights only from {latest_path} at epoch {start_epoch}")
    return start_epoch, best_val


def init_wandb(config: dict):
    if not bool(config.get("wandb_enabled", False)):
        return None
    import wandb

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

    model = SnakePatchWorldModel(make_model_config(config)).to(device)
    sigreg = SIGReg(
        knots=int(config["sigreg_knots"]),
        num_proj=int(config["sigreg_num_proj"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch, best_val = maybe_resume(model, optimizer, config, device)
    wandb_run = init_wandb(config)

    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = model.count_parameters()
    with (run_dir / "train_config.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2)

    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
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
            train_metrics = run_epoch(model, sigreg, train_loader, device, config, optimizer)
            val_metrics = run_epoch(model, sigreg, val_loader, device, config)
            last_epoch = epoch
            last_val_metrics = val_metrics
            print(
                f"epoch {epoch:03d} | train {train_metrics['loss']:.4f} | val {val_metrics['loss']:.4f} | "
                f"pred_recon {val_metrics['pred_recon_loss']:.4f} | latent {val_metrics['latent_loss']:.4f}"
            )

            log_payload = {
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "best_val/loss": min(best_val, val_metrics["loss"]),
            }

            if epoch % int(config["preview_every"]) == 0:
                preview_image = save_preview(model, val_loader, device, run_dir, epoch)
                if wandb_run is not None:
                    import wandb

                    log_payload["val/example"] = wandb.Image(preview_image, caption=f"epoch {epoch}")

            if wandb_run is not None:
                wandb_run.log(log_payload, step=epoch)

            previous_best = best_val
            best_val = min(best_val, val_metrics["loss"])
            if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= previous_best:
                save_checkpoint(model, optimizer, config_dump, checkpoint_dir, epoch, val_metrics, best_val)

            if stop_requested["value"]:
                save_checkpoint(model, optimizer, config_dump, checkpoint_dir, epoch, val_metrics, best_val, interrupted=True)
                print("graceful shutdown complete")
                break
    except KeyboardInterrupt:
        print("keyboard interrupt received, saving latest checkpoint")
        save_checkpoint(
            model,
            optimizer,
            config_dump,
            checkpoint_dir,
            max(last_epoch, start_epoch - 1),
            last_val_metrics,
            best_val,
            interrupted=True,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
