from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_jepa.infer_snake_board_dynamics import checkpoint_path, detect_device
from snake_jepa.snake_board import FOOD, SNAKE, extract_board
from snake_jepa.snake_board_rollout import initialize_snake_body, legalize_snake_transition
from snake_jepa.snake_board_model import SnakeBoardDynamics, SnakeBoardDynamicsConfig
from snake_jepa.snake_data import discover_snake_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate autoregressive snake board dynamics rollouts")
    parser.add_argument("--dataset-root", type=str, default="/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent")
    parser.add_argument("--output-dir", type=str, default="runs/snake_board_dynamics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--levels", nargs="*", default=["level_1"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--legalize-snake", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> SnakeBoardDynamics:
    path = checkpoint_path(args.output_dir, args.run_name, args.checkpoint)
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type != "snake_board_dynamics_v1":
        raise RuntimeError(f"unsupported checkpoint model_type={model_type!r}")
    model = SnakeBoardDynamics(SnakeBoardDynamicsConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def update_metric(metrics: dict[str, list[float]], name: str, correct: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    if mask is None:
        denom = correct.numel()
        value = float(correct.float().mean().item())
    else:
        denom = int(mask.sum().item())
        value = float(correct[mask].float().mean().item()) if denom else 0.0
    metrics[name][0] += value * denom
    metrics[name][1] += denom


@torch.no_grad()
def eval_clip(
    model: SnakeBoardDynamics,
    clip,
    device: torch.device,
    steps: int,
    *,
    legalize_snake: bool = False,
) -> dict[str, list[float]]:
    history_size = int(model.cfg.history_size)
    boards = [extract_board(frame.path) for frame in clip.frames]
    pred_history = boards[:history_size]
    action_history = [frame.action for frame in clip.frames[1 : history_size + 1]]
    snake_body = initialize_snake_body(pred_history)
    metrics = {
        "board_acc": [0.0, 0],
        "nonempty_acc": [0.0, 0],
        "snake_acc": [0.0, 0],
        "food_acc": [0.0, 0],
        "exact_board": [0.0, 0],
    }
    max_steps = min(int(steps), len(boards) - history_size)
    for offset in range(max_steps):
        target_index = history_size + offset
        action = clip.frames[target_index].action
        history = torch.stack(pred_history[-history_size:], dim=0).unsqueeze(0).to(device)
        action_window = list(action_history[-history_size:])
        action_window[-1] = action
        action_tensor = torch.tensor(action_window, dtype=torch.long, device=device).unsqueeze(0)
        action_one_hot = F.one_hot(action_tensor, num_classes=4).float()
        pred = model(history, action_one_hot)[0].argmax(dim=0).cpu()
        if legalize_snake:
            pred, snake_body, action = legalize_snake_transition(pred_history[-1], pred, snake_body, action)
        target = boards[target_index]
        correct = pred.eq(target)
        update_metric(metrics, "board_acc", correct)
        update_metric(metrics, "nonempty_acc", correct, target.ne(0))
        update_metric(metrics, "snake_acc", correct, target.eq(SNAKE))
        update_metric(metrics, "food_acc", correct, target.eq(FOOD))
        metrics["exact_board"][0] += float(bool(correct.all()))
        metrics["exact_board"][1] += 1
        pred_history.append(pred)
        action_history[-1] = action
        action_history.append(action)
    return metrics


def merge_metrics(total: dict[str, list[float]], current: dict[str, list[float]]) -> None:
    for key, value in current.items():
        total[key][0] += value[0]
        total[key][1] += value[1]


def finalize(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {key: value / max(1, count) for key, (value, count) in metrics.items()}


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    model = load_model(args, device)
    clips = discover_snake_clips(args.dataset_root, levels=list(args.levels), max_clips_per_level=int(args.max_clips))
    total = {
        "board_acc": [0.0, 0],
        "nonempty_acc": [0.0, 0],
        "snake_acc": [0.0, 0],
        "food_acc": [0.0, 0],
        "exact_board": [0.0, 0],
    }
    for index, clip in enumerate(clips, start=1):
        merge_metrics(
            total,
            eval_clip(model, clip, device, int(args.steps), legalize_snake=bool(args.legalize_snake)),
        )
        progress_every = int(args.progress_every)
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(clips)):
            partial = finalize(total)
            print(
                f"progress {index}/{len(clips)} | exact {partial['exact_board']:.3f} | "
                f"snake {partial['snake_acc']:.3f} | food {partial['food_acc']:.3f}",
                flush=True,
            )
    result = finalize(total)
    result["clips"] = len(clips)
    result["steps"] = int(args.steps)
    result["legalize_snake"] = bool(args.legalize_snake)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
