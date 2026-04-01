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
import wandb

from debug_box_data import build_debug_box_loaders
from debug_box_world_model import DebugBoxWorldModel, DebugBoxWorldModelConfig, LatentDecoder


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/world-env/datasets/debug_box",
    "wm_output_dir": "runs/debug_box_vit_mps",
    "wm_run_name": "first-box-lewm-1",
    "output_dir": "runs/debug_box_decoder",
    "device": "auto",
    "run_name": "debug-box-decoder",
    "wandb_project": "debug-box-world-model",
    "wandb_entity": "krishnapg2315",
    "seed": 7,
    "batch_size": 8,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_fraction": 0.1,
    "preview_every": 1,
    "checkpoint_every": 5,
    "max_train_batches": 0,
    "max_val_batches": 0,
}
DECODER_INPUT_MODE = "vit_preproj_cls_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train decoder on frozen debug-box world-model latents")
    parser.add_argument("--config", type=str, default="config/train/debug_box_decoder.json")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wm-run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r") as handle:
            config.update(json.load(handle))
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.wm_run_name is not None:
        config["wm_run_name"] = args.wm_run_name
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
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


def get_run_dir(config: dict) -> Path:
    return Path(config["output_dir"]) / str(config["run_name"])


def get_checkpoint_dir(config: dict) -> Path:
    return get_run_dir(config) / "checkpoints"


def get_latest_checkpoint_path(config: dict) -> Path:
    return get_checkpoint_dir(config) / "latest.pt"


def get_wm_checkpoint_path(config: dict) -> Path:
    return Path(config["wm_output_dir"]) / str(config["wm_run_name"]) / "checkpoints" / "best.pt"


def load_world_model(config: dict, device: torch.device) -> DebugBoxWorldModel:
    checkpoint_path = get_wm_checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"world-model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = DebugBoxWorldModelConfig(**checkpoint["model_config"])
    model = DebugBoxWorldModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_losses(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    recon_loss = F.l1_loss(output["pred_frame"], batch["next_frame"])
    mse_loss = F.mse_loss(output["pred_frame"], batch["next_frame"])
    loss = recon_loss + mse_loss
    return {"loss": loss, "recon_loss": recon_loss, "mse_loss": mse_loss}


def _spatial_moments(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = frames.mean(dim=1)
    weights = (gray - gray.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-6)
    h, w = gray.shape[-2:]
    ys = torch.linspace(0.0, 1.0, h, device=frames.device, dtype=frames.dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=frames.device, dtype=frames.dtype).view(1, 1, w)
    norm = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mean_y = (weights * ys).sum(dim=(-2, -1), keepdim=True) / norm
    mean_x = (weights * xs).sum(dim=(-2, -1), keepdim=True) / norm
    center = torch.cat([mean_x, mean_y], dim=-1).squeeze(-2)
    var_y = (weights * (ys - mean_y).square()).sum(dim=(-2, -1), keepdim=True) / norm
    var_x = (weights * (xs - mean_x).square()).sum(dim=(-2, -1), keepdim=True) / norm
    spread = torch.cat([var_x.sqrt(), var_y.sqrt()], dim=-1).squeeze(-2)
    return center, spread


@torch.no_grad()
def compute_decoder_diagnostics(
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    cls_latents = world_model.encode_next_cls(batch["next_frame"])
    pred_frame = decoder(cls_latents)
    zero_frame = decoder(torch.zeros_like(cls_latents))

    gt_center, gt_spread = _spatial_moments(batch["next_frame"])
    pred_center, pred_spread = _spatial_moments(pred_frame)

    return {
        "cls_std": float(cls_latents.std(dim=0).mean().item()),
        "cls_norm": float(cls_latents.norm(dim=-1).mean().item()),
        "zero_latent_decode_delta": float(F.l1_loss(pred_frame, zero_frame).item()),
        "center_error": float(F.l1_loss(pred_center, gt_center).item()),
        "spread_error": float(F.l1_loss(pred_spread, gt_spread).item()),
    }


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def create_preview_image(history: torch.Tensor, target_next: torch.Tensor, pred_next: torch.Tensor) -> Image.Image:
    tiles = [_to_pil(frame) for frame in history]
    target_tiles = [_to_pil(target_next) for _ in range(len(history))]
    pred_tiles = [_to_pil(pred_next) for _ in range(len(history))]

    tile_w, tile_h = tiles[0].size
    columns = len(history)
    rows = 3
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h + 32), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    labels = ["history", "target next", "decoded next"]
    for row, label in enumerate(labels):
        draw.text((8, row * tile_h + 8), label, fill=(235, 235, 235))

    for idx, tile in enumerate(tiles):
        canvas.paste(tile, (idx * tile_w, 0))
    for idx, tile in enumerate(target_tiles):
        canvas.paste(tile, (idx * tile_w, tile_h))
    for idx, tile in enumerate(pred_tiles):
        canvas.paste(tile, (idx * tile_w, 2 * tile_h))
    return canvas


@torch.no_grad()
def save_preview(
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    loader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> Image.Image:
    batch = next(iter(loader))
    batch = move_batch(batch, device)
    next_cls_latent = world_model.encode_next_cls(batch["next_frame"])
    pred_frame = decoder(next_cls_latent)
    canvas = create_preview_image(batch["history_frames"][0], batch["next_frame"][0], pred_frame[0])
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")
    return canvas


def train_epoch(
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    decoder.train()
    totals = {"loss": 0.0, "recon_loss": 0.0, "mse_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_cls_latent = world_model.encode_next_cls(batch["next_frame"])
        pred_frame = decoder(next_cls_latent)
        losses = compute_losses(batch, {"pred_frame": pred_frame})
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
def evaluate(
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    loader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    decoder.eval()
    totals = {"loss": 0.0, "recon_loss": 0.0, "mse_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        next_cls_latent = world_model.encode_next_cls(batch["next_frame"])
        pred_frame = decoder(next_cls_latent)
        losses = compute_losses(batch, {"pred_frame": pred_frame})
        batch_size = batch["next_frame"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if int(config["max_val_batches"]) > 0 and step >= int(config["max_val_batches"]):
            break
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def evaluate_diagnostics(
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    loader,
    device: torch.device,
) -> dict[str, float]:
    batch = next(iter(loader))
    batch = move_batch(batch, device)
    return compute_decoder_diagnostics(world_model, decoder, batch)


def save_checkpoint(
    decoder: LatentDecoder,
    optimizer: torch.optim.Optimizer,
    config: dict,
    checkpoint_dir: Path,
    epoch: int,
    val_metrics: dict[str, float],
    best_val: float,
    world_model_config: dict,
    interrupted: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "decoder_input_mode": DECODER_INPUT_MODE,
        "config": config,
        "val_metrics": val_metrics,
        "best_val": best_val,
        "world_model_config": world_model_config,
        "interrupted": interrupted,
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, latest_path)
    if val_metrics["loss"] <= best_val:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def maybe_resume(
    decoder: LatentDecoder,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
) -> tuple[int, float]:
    latest_path = get_latest_checkpoint_path(config)
    if not latest_path.exists():
        return 1, math.inf
    checkpoint = torch.load(latest_path, map_location=device)
    checkpoint_mode = checkpoint.get("decoder_input_mode", "cls_global_v0")
    if checkpoint_mode != DECODER_INPUT_MODE:
        print(
            f"decoder checkpoint at {latest_path} was trained with {checkpoint_mode}, "
            f"expected {DECODER_INPUT_MODE}; starting from scratch"
        )
        return 1, math.inf
    try:
        decoder.load_state_dict(checkpoint["decoder_state"])
    except RuntimeError as exc:
        print(
            f"decoder checkpoint at {latest_path} is incompatible with the current "
            f"patch-query decoder, starting from scratch: {exc}"
        )
        return 1, math.inf
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_loaded = True
    except ValueError as exc:
        optimizer_loaded = False
        print(f"optimizer state mismatch at {latest_path}, skipping optimizer resume: {exc}")
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val", math.inf))
    if optimizer_loaded:
        print(f"resuming decoder and optimizer from {latest_path} at epoch {start_epoch}")
    else:
        print(f"resuming decoder weights only from {latest_path} at epoch {start_epoch}")
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

    world_model = load_world_model(config, device)
    train_loader, val_loader = build_debug_box_loaders(
        config["dataset_root"],
        history_size=int(world_model.cfg.history_size),
        image_size=int(world_model.cfg.image_size),
        batch_size=int(config["batch_size"]),
        val_fraction=float(config["val_fraction"]),
        num_workers=int(config["num_workers"]),
    )

    decoder = LatentDecoder(
        latent_dim=int(world_model.cfg.encoder_dim),
        image_size=int(world_model.cfg.image_size),
        patch_size=int(world_model.cfg.patch_size),
        hidden_dim=int(world_model.cfg.decoder_dim),
        depth=int(world_model.cfg.decoder_depth),
        heads=int(world_model.cfg.decoder_heads),
        mlp_ratio=float(world_model.cfg.decoder_mlp_ratio),
        dropout=float(world_model.cfg.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch, best_val = maybe_resume(decoder, optimizer, config, device)
    wandb_run = init_wandb(config)

    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    with (run_dir / "train_config.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2)

    print(f"device: {device}")
    print(f"world model: {get_wm_checkpoint_path(config)}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"decoder parameters: {config_dump['parameter_count']:,}")
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
            train_metrics = train_epoch(world_model, decoder, train_loader, optimizer, device, config)
            val_metrics = evaluate(world_model, decoder, val_loader, device, config)
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
            }

            diagnostics = evaluate_diagnostics(world_model, decoder, val_loader, device)
            log_payload.update(
                {
                    "diag/cls_std": diagnostics["cls_std"],
                    "diag/cls_norm": diagnostics["cls_norm"],
                    "diag/zero_latent_decode_delta": diagnostics["zero_latent_decode_delta"],
                    "diag/center_error": diagnostics["center_error"],
                    "diag/spread_error": diagnostics["spread_error"],
                }
            )

            if epoch % int(config["preview_every"]) == 0:
                preview_image = save_preview(world_model, decoder, val_loader, device, run_dir, epoch)
                log_payload["val/example"] = wandb.Image(preview_image, caption=f"epoch {epoch}")

            wandb_run.log(log_payload, step=epoch)

            previous_best = best_val
            best_val = min(best_val, val_metrics["loss"])
            if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= previous_best:
                save_checkpoint(
                    decoder,
                    optimizer,
                    config_dump,
                    checkpoint_dir,
                    epoch,
                    val_metrics,
                    best_val,
                    asdict(world_model.cfg),
                )

            if stop_requested["value"]:
                save_checkpoint(
                    decoder,
                    optimizer,
                    config_dump,
                    checkpoint_dir,
                    epoch,
                    val_metrics,
                    best_val,
                    asdict(world_model.cfg),
                    interrupted=True,
                )
                print("graceful shutdown complete")
                break
    except KeyboardInterrupt:
        print("keyboard interrupt received, saving latest checkpoint")
        save_checkpoint(
            decoder,
            optimizer,
            config_dump,
            checkpoint_dir,
            max(last_epoch, start_epoch - 1),
            last_val_metrics,
            best_val,
            asdict(world_model.cfg),
            interrupted=True,
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        wandb_run.finish()


if __name__ == "__main__":
    main()
