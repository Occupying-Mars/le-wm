import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from snake_jepa.snake_board import FOOD, HEAD, NUM_BOARD_CLASSES_WITH_HEAD, SNAKE, render_board
from snake_jepa.snake_board_model import SnakeBoardDynamics, SnakeBoardDynamicsConfig
from snake_jepa.snake_data import build_snake_board_loaders


DEFAULT_CONFIG = {
    "dataset_root": "/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent",
    "levels": ["level_1", "level_2", "level_3", "random_levels"],
    "output_dir": "runs/snake_board_dynamics",
    "run_name": "snake-board-dynamics",
    "device": "auto",
    "seed": 7,
    "history_size": 4,
    "rollout_steps": 1,
    "rollout_feedback": "soft",
    "batch_size": 32,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_fraction": 0.1,
    "stride": 1,
    "max_clips_per_level": 0,
    "max_windows_per_clip": 0,
    "hidden_dim": 128,
    "depth": 8,
    "dropout": 0.0,
    "split_head": False,
    "board_cache_dir": "",
    "class_weights": [0.1, 2.0, 10.0, 15.0],
    "grad_clip_norm": 1.0,
    "max_train_batches": 0,
    "max_val_batches": 0,
    "preview_every": 5,
    "checkpoint_every": 5,
    "wandb_enabled": False,
    "wandb_project": "snake-jepa",
    "wandb_entity": "krishnapg2315",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train board-space snake dynamics from png-derived boards")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--rollout-feedback", choices=["soft", "hard"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-clips-per-level", type=int, default=None)
    parser.add_argument("--max-windows-per-clip", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--preview-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--split-head", action="store_true")
    parser.add_argument("--board-cache-dir", type=str, default=None)
    parser.add_argument("--class-weights", type=float, nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(Path(args.config).read_text()))
    for key in (
        "dataset_root",
        "levels",
        "output_dir",
        "run_name",
        "device",
        "epochs",
        "batch_size",
        "history_size",
        "rollout_steps",
        "rollout_feedback",
        "lr",
        "max_clips_per_level",
        "max_windows_per_clip",
        "max_train_batches",
        "max_val_batches",
        "preview_every",
        "checkpoint_every",
        "hidden_dim",
        "depth",
        "board_cache_dir",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.class_weights is not None:
        config["class_weights"] = list(args.class_weights)
    if args.split_head:
        config["split_head"] = True
    if args.wandb:
        config["wandb_enabled"] = True
    if bool(config.get("split_head", False)):
        config["num_classes"] = NUM_BOARD_CLASSES_WITH_HEAD
        if len(config["class_weights"]) == 4:
            config["class_weights"] = [*config["class_weights"], config["class_weights"][2]]
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def board_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    pixel_loss = F.cross_entropy(logits, target.long(), reduction="none")
    pixel_weight = class_weights.gather(0, target.long().clamp(0, class_weights.numel() - 1).reshape(-1))
    pixel_weight = pixel_weight.reshape_as(pixel_loss)
    return (pixel_loss * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def rollout_loss_and_logits(
    model: SnakeBoardDynamics,
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    feedback: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = batch["history_boards"]
    target_boards = batch["target_boards"]
    target_actions = batch["target_actions"]
    action_history = batch["actions"]
    losses = []
    first_logits = None
    for step in range(target_boards.size(1)):
        action_window = action_history.clone()
        action_window[:, -1] = target_actions[:, step]
        action_one_hot = F.one_hot(action_window, num_classes=4).float()
        logits = model(history, action_one_hot)
        if first_logits is None:
            first_logits = logits
        losses.append(board_loss(logits, target_boards[:, step], class_weights))
        if feedback == "hard":
            pred_board = logits.argmax(dim=1)
            if history.dim() == 5:
                history = history.argmax(dim=2)
            history = torch.cat([history[:, 1:], pred_board[:, None]], dim=1)
        else:
            pred_probs = logits.softmax(dim=1)
            if history.dim() == 4:
                history = F.one_hot(history.long(), num_classes=model.cfg.num_classes).permute(0, 1, 4, 2, 3).float()
            history = torch.cat([history[:, 1:], pred_probs[:, None]], dim=1)
        action_history = torch.cat([action_history[:, 1:], target_actions[:, step : step + 1]], dim=1)
    if first_logits is None:
        raise ValueError("rollout_steps must be at least 1")
    return torch.stack(losses).mean(), first_logits


@torch.no_grad()
def board_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, tuple[float, int]]:
    pred = logits.argmax(dim=1)
    correct = pred.eq(target.long())
    metrics = {"board_acc": (float(correct.float().mean().item()), target.numel())}
    for name, mask in {
        "nonempty_acc": target.ne(0),
        "snake_acc": target.eq(SNAKE) | target.eq(HEAD),
        "food_acc": target.eq(FOOD),
    }.items():
        denom = int(mask.sum().item())
        metrics[name] = (float(correct[mask].float().mean().item()), denom) if denom else (0.0, 0)
    return metrics


def run_epoch(model, loader, device, config, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    class_weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)
    totals = {"loss": 0.0}
    metric_totals = {"board_acc": 0.0, "nonempty_acc": 0.0, "snake_acc": 0.0, "food_acc": 0.0}
    metric_counts = {key: 0 for key in metric_totals}
    count = 0
    max_batches = int(config["max_train_batches"] if training else config["max_val_batches"])
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            loss, logits = rollout_loss_and_logits(model, batch, class_weights, str(config.get("rollout_feedback", "soft")))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(config["grad_clip_norm"])
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        batch_size = batch["next_board"].size(0)
        totals["loss"] += float(loss.item()) * batch_size
        for key, (value, weight) in board_metrics(logits, batch["next_board"]).items():
            metric_totals[key] += value * weight
            metric_counts[key] += weight
        count += batch_size
        if max_batches > 0 and step >= max_batches:
            break
    result = {key: value / max(1, count) for key, value in totals.items()}
    result.update({key: metric_totals[key] / max(1, metric_counts[key]) for key in metric_totals})
    return result


def save_preview(model, loader, device, run_dir: Path, epoch: int, image_size: int = 320) -> Image.Image:
    model.eval()
    batch = move_batch(next(iter(loader)), device)
    with torch.no_grad():
        pred = model(batch["history_boards"], batch["actions_one_hot"]).argmax(dim=1)[0]
    history = [render_board(board, image_size) for board in batch["history_boards"][0]]
    target = render_board(batch["next_board"][0], image_size)
    pred_image = render_board(pred, image_size)
    tiles = history + [target, pred_image]
    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (len(tiles) * tile_w, tile_h + 28), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    labels = [f"h{i}" for i in range(len(history))] + ["target", "pred"]
    for idx, (label, tile) in enumerate(zip(labels, tiles)):
        x = idx * tile_w
        draw.text((x + 6, 6), label, fill=(235, 235, 235))
        canvas.paste(tile, (x, 28))
    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / f"epoch_{epoch:03d}.png")
    return canvas


def save_checkpoint(model, optimizer, config, run_dir: Path, epoch: int, val_metrics: dict, best_val: float) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_type": "snake_board_dynamics_v1",
        "config": config,
        "model_config": asdict(model.cfg),
        "val_metrics": val_metrics,
        "best_val": best_val,
    }
    torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    if val_metrics["loss"] <= best_val:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def init_wandb(config: dict, run_dir: Path):
    if not bool(config["wandb_enabled"]):
        return None
    import wandb

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
    run_dir = Path(config["output_dir"]) / str(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_snake_board_loaders(
        config["dataset_root"],
        levels=list(config["levels"]) if config.get("levels") else None,
        history_size=int(config["history_size"]),
        rollout_steps=int(config["rollout_steps"]),
        batch_size=int(config["batch_size"]),
        val_fraction=float(config["val_fraction"]),
        num_workers=int(config["num_workers"]),
        stride=int(config["stride"]),
        max_clips_per_level=int(config["max_clips_per_level"]),
        max_windows_per_clip=int(config["max_windows_per_clip"]),
        split_head=bool(config.get("split_head", False)),
        board_cache_dir=str(config.get("board_cache_dir", "")),
    )
    model = SnakeBoardDynamics(
        SnakeBoardDynamicsConfig(
            history_size=int(config["history_size"]),
            num_classes=int(config.get("num_classes", 4)),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            dropout=float(config["dropout"]),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    wandb_run = init_wandb(config, run_dir)
    config_dump = dict(config)
    config_dump["device_resolved"] = str(device)
    config_dump["parameter_count"] = model.count_parameters()
    (run_dir / "train_config.json").write_text(json.dumps(config_dump, indent=2))
    print(f"device: {device}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"parameters: {model.count_parameters():,}")
    print(f"run dir: {run_dir}")
    best_val = math.inf
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, config, optimizer)
        val_metrics = run_epoch(model, val_loader, device, config)
        best_val = min(best_val, val_metrics["loss"])
        print(
            f"epoch {epoch:03d} | train {train_metrics['loss']:.4f} | val {val_metrics['loss']:.4f} | "
            f"board_acc {val_metrics['board_acc']:.3f} | nonempty {val_metrics['nonempty_acc']:.3f} | "
            f"snake {val_metrics['snake_acc']:.3f} | food {val_metrics['food_acc']:.3f}",
            flush=True,
        )
        log_payload = {"epoch": epoch, "best_val/loss": best_val}
        log_payload.update({f"train/{key}": value for key, value in train_metrics.items()})
        log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})
        if epoch % int(config["preview_every"]) == 0:
            preview = save_preview(model, val_loader, device, run_dir, epoch)
            if wandb_run is not None:
                import wandb

                log_payload["val/board_preview"] = wandb.Image(preview, caption=f"epoch {epoch}")
        if wandb_run is not None:
            wandb_run.log(log_payload, step=epoch)
        if epoch % int(config["checkpoint_every"]) == 0 or val_metrics["loss"] <= best_val:
            save_checkpoint(model, optimizer, config_dump, run_dir, epoch, val_metrics, best_val)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
