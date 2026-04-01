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
from debug_box_world_model import DebugBoxWorldModel, DebugBoxWorldModelConfig


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/world-env/datasets/debug_box",
    "output_dir": "runs/debug_box_vit_mps",
    "device": "auto",
    "run_name": "debug-box-vit",
    "wandb_project": "debug-box-world-model",
    "wandb_entity": "krishnapg2315",
    "seed": 7,
    "image_size": 128,
    "patch_size": 16,
    "history_size": 4,
    "batch_size": 4,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_fraction": 0.1,
    "encoder_dim": 128,
    "encoder_depth": 4,
    "encoder_heads": 4,
    "encoder_mlp_ratio": 4.0,
    "latent_dim": 64,
    "dynamics_dim": 128,
    "dynamics_depth": 2,
    "dynamics_heads": 4,
    "dynamics_mlp_ratio": 2.0,
    "dropout": 0.0,
    "sigreg_weight": 0.09,
    "sigreg_knots": 17,
    "sigreg_num_proj": 1024,
    "preview_every": 1,
    "checkpoint_every": 5,
    "max_train_batches": 0,
    "max_val_batches": 0,
}


class SIGReg(torch.nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
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
    parser = argparse.ArgumentParser(description="train a small vit world model on debug-box")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train/debug_box_vit_mps.json",
        help="json config path",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r") as handle:
            config.update(json.load(handle))

    if args.dataset_root is not None:
        config["dataset_root"] = args.dataset_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
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


def make_model_config(config: dict) -> DebugBoxWorldModelConfig:
    return DebugBoxWorldModelConfig(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        history_size=config["history_size"],
        encoder_dim=config["encoder_dim"],
        encoder_depth=config["encoder_depth"],
        encoder_heads=config["encoder_heads"],
        encoder_mlp_ratio=config["encoder_mlp_ratio"],
        latent_dim=config["latent_dim"],
        dynamics_dim=config["dynamics_dim"],
        dynamics_depth=config["dynamics_depth"],
        dynamics_heads=config["dynamics_heads"],
        dynamics_mlp_ratio=config["dynamics_mlp_ratio"],
        dropout=config["dropout"],
    )


def compute_losses(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    latent_loss = F.mse_loss(
        output["pred_next_latent"],
        output["target_next_latent"].detach(),
    )
    sigreg_loss = output["sigreg_loss"]
    total = latent_loss + float(config["sigreg_weight"]) * sigreg_loss
    return {
        "loss": total,
        "latent_loss": latent_loss,
        "sigreg_loss": sigreg_loss,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_run_dir(config: dict) -> Path:
    return Path(config["output_dir"]) / str(config["run_name"])


def get_checkpoint_dir(config: dict) -> Path:
    return get_run_dir(config) / "checkpoints"


def get_latest_checkpoint_path(config: dict) -> Path:
    return get_checkpoint_dir(config) / "latest.pt"


@torch.no_grad()
def evaluate(model: DebugBoxWorldModel, loader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "latent_loss": 0.0, "sigreg_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        output = model.predict_next(batch["history_frames"], batch["actions_one_hot"])
        output["target_next_latent"] = model.encode_next(batch["next_frame"])
        output["sigreg_loss"] = evaluate.sigreg(output["history_latents"].transpose(0, 1))
        losses = compute_losses(batch, output, config)
        batch_size = batch["history_frames"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if int(config["max_val_batches"]) > 0 and step >= int(config["max_val_batches"]):
            break
    return {key: value / max(1, count) for key, value in totals.items()}


def _to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def create_preview_image(
    history: torch.Tensor,
    target_next: torch.Tensor,
) -> Image.Image:
    tiles = [_to_pil(frame) for frame in history]
    tiles += [_to_pil(target_next) for _ in range(len(history))]

    tile_w, tile_h = tiles[0].size
    columns = len(history)
    rows = 2
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h + 32), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    labels = ["history", "target next"]
    for row, label in enumerate(labels):
        draw.text((8, row * tile_h + 8), label, fill=(235, 235, 235))

    for idx, tile in enumerate(tiles[:columns]):
        canvas.paste(tile, (idx * tile_w, 0))
    for idx, tile in enumerate(tiles[columns:]):
        canvas.paste(tile, (idx * tile_w, tile_h))
    return canvas


@torch.no_grad()
def save_preview(
    model: DebugBoxWorldModel,
    loader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> Image.Image:
    model.eval()
    batch = next(iter(loader))
    batch = move_batch(batch, device)
    history = batch["history_frames"][0]
    target_next = batch["next_frame"][0]
    canvas = create_preview_image(history, target_next)

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")
    return canvas


def train_epoch(
    model: DebugBoxWorldModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "latent_loss": 0.0, "sigreg_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model.predict_next(batch["history_frames"], batch["actions_one_hot"])
        output["target_next_latent"] = model.encode_next(batch["next_frame"])
        output["sigreg_loss"] = train_epoch.sigreg(output["history_latents"].transpose(0, 1))
        losses = compute_losses(batch, output, config)
        losses["loss"].backward()
        optimizer.step()

        batch_size = batch["history_frames"].size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size
        if int(config["max_train_batches"]) > 0 and step >= int(config["max_train_batches"]):
            break
    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    model: DebugBoxWorldModel,
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
        "config": config,
        "val_metrics": val_metrics,
        "best_val": best_val,
        "model_config": asdict(model.cfg),
        "interrupted": interrupted,
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, latest_path)
    if val_metrics["loss"] <= best_val:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def maybe_resume(
    model: DebugBoxWorldModel,
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
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_loaded = True
        except ValueError as exc:
            print(f"optimizer state mismatch at {latest_path}, skipping optimizer resume: {exc}")
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val", math.inf))
    if optimizer_loaded:
        print(f"resuming model and optimizer from {latest_path} at epoch {start_epoch}")
    else:
        print(f"resuming model weights only from {latest_path} at epoch {start_epoch}")
    return start_epoch, best_val


def init_wandb(config: dict, model: DebugBoxWorldModel) -> wandb.sdk.wandb_run.Run:
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
        history_size=int(config["history_size"]),
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        val_fraction=float(config["val_fraction"]),
        num_workers=int(config["num_workers"]),
    )

    model_cfg = make_model_config(config)
    model = DebugBoxWorldModel(model_cfg).to(device)
    model.freeze_decoder_modules()
    sigreg = SIGReg(
        knots=int(config["sigreg_knots"]),
        num_proj=int(config["sigreg_num_proj"]),
    ).to(device)
    train_epoch.sigreg = sigreg
    evaluate.sigreg = sigreg
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch, best_val = maybe_resume(model, optimizer, config, device)
    wandb_run = init_wandb(config, model)

    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = model.count_parameters()
    with (run_dir / "train_config.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2)

    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"parameters: {model.count_parameters():,}")
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
    last_val_metrics = {"loss": math.inf, "latent_loss": math.inf, "sigreg_loss": math.inf}
    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, device, config)
            val_metrics = evaluate(model, val_loader, device, config)
            last_epoch = epoch
            last_val_metrics = val_metrics
            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"latent {val_metrics['latent_loss']:.4f} | "
                f"sigreg {val_metrics['sigreg_loss']:.4f}"
            )

            log_payload = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/latent_loss": train_metrics["latent_loss"],
                "train/sigreg_loss": train_metrics["sigreg_loss"],
                "val/loss": val_metrics["loss"],
                "val/latent_loss": val_metrics["latent_loss"],
                "val/sigreg_loss": val_metrics["sigreg_loss"],
                "best_val/loss": min(best_val, val_metrics["loss"]),
            }

            if epoch % int(config["preview_every"]) == 0:
                preview_image = save_preview(model, val_loader, device, run_dir, epoch)
                log_payload["val/example"] = wandb.Image(preview_image, caption=f"epoch {epoch}")

            wandb_run.log(log_payload, step=epoch)

            previous_best = best_val
            best_val = min(best_val, val_metrics["loss"])
            if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= previous_best:
                save_checkpoint(
                    model,
                    optimizer,
                    config_dump,
                    checkpoint_dir,
                    epoch,
                    val_metrics,
                    best_val,
                )

            if stop_requested["value"]:
                save_checkpoint(
                    model,
                    optimizer,
                    config_dump,
                    checkpoint_dir,
                    epoch,
                    val_metrics,
                    best_val,
                    interrupted=True,
                )
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
        wandb_run.finish()


if __name__ == "__main__":
    main()
