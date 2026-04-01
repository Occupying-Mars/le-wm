import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from debug_box_data import build_debug_box_loaders
from debug_box_world_model import DebugBoxWorldModel, DebugBoxWorldModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="probe debug-box latents for x,y state information")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/Users/krishna/Public/ml-experiments/snake-we/world-env/datasets/debug_box",
    )
    parser.add_argument("--wm-output-dir", type=str, default="runs/debug_box_vit_mps")
    parser.add_argument("--wm-run-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_checkpoint_path(root: str | Path, run_name: str) -> Path:
    run_dir = Path(root) / run_name / "checkpoints"
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    if best_path.exists():
        return best_path
    if latest_path.exists():
        return latest_path
    raise FileNotFoundError(f"no checkpoint found in {run_dir}")


def load_world_model(args: argparse.Namespace, device: torch.device) -> DebugBoxWorldModel:
    checkpoint = torch.load(get_checkpoint_path(args.wm_output_dir, args.wm_run_name), map_location=device)
    config = DebugBoxWorldModelConfig(**checkpoint["model_config"])
    model = DebugBoxWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def collect_features(model: DebugBoxWorldModel, loader, device: torch.device) -> dict[str, torch.Tensor]:
    cls_features = []
    projected_features = []
    targets = []
    for batch in loader:
        next_frame = batch["next_frame"].to(device)
        next_state = batch["next_state"].to(device)
        cls_features.append(model.encode_next_cls(next_frame))
        projected_features.append(model.encode_next(next_frame))
        targets.append(next_state)
    return {
        "preproj_cls": torch.cat(cls_features, dim=0),
        "projected_latent": torch.cat(projected_features, dim=0),
        "target": torch.cat(targets, dim=0),
    }


def fit_linear_probe(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(features.size(0), 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    solution = torch.linalg.lstsq(design, target).solution
    return solution


def apply_linear_probe(features: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(features.size(0), 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    return design @ probe


def summarize_probe(name: str, train_feats: torch.Tensor, train_target: torch.Tensor, val_feats: torch.Tensor, val_target: torch.Tensor) -> None:
    probe = fit_linear_probe(train_feats, train_target)
    train_pred = apply_linear_probe(train_feats, probe)
    val_pred = apply_linear_probe(val_feats, probe)
    train_mse = F.mse_loss(train_pred, train_target).item()
    val_mse = F.mse_loss(val_pred, val_target).item()
    train_mae = F.l1_loss(train_pred, train_target).item()
    val_mae = F.l1_loss(val_pred, val_target).item()
    print(
        f"{name}: "
        f"train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
        f"train_mae={train_mae:.6f} val_mae={val_mae:.6f}"
    )


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"device: {device}")

    model = load_world_model(args, device)
    train_loader, val_loader = build_debug_box_loaders(
        args.dataset_root,
        history_size=int(model.cfg.history_size),
        image_size=int(model.cfg.image_size),
        batch_size=int(args.batch_size),
        val_fraction=float(args.val_fraction),
        num_workers=int(args.num_workers),
    )

    train_data = collect_features(model, train_loader, device)
    val_data = collect_features(model, val_loader, device)

    print(f"train samples: {train_data['target'].size(0)}")
    print(f"val samples: {val_data['target'].size(0)}")
    summarize_probe(
        "preproj_cls",
        train_data["preproj_cls"],
        train_data["target"],
        val_data["preproj_cls"],
        val_data["target"],
    )
    summarize_probe(
        "projected_latent",
        train_data["projected_latent"],
        train_data["target"],
        val_data["projected_latent"],
        val_data["target"],
    )


if __name__ == "__main__":
    main()
