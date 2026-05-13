from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from snake_jepa.eval_snake_board_rollout import load_model
from snake_jepa.infer_snake_board_dynamics import detect_device
from snake_jepa.snake_board import extract_board, render_board
from snake_jepa.snake_board_rollout import initialize_snake_body, legalize_snake_transition
from snake_jepa.snake_data import discover_snake_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="save target-vs-model snake board rollout gifs")
    parser.add_argument("--dataset-root", type=str, default="/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent")
    parser.add_argument("--output-dir", type=str, default="runs/snake_board_dynamics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--levels", nargs="*", default=["level_1"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--gif-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--legalize-snake", action="store_true")
    parser.add_argument("--gif-dir", type=str, default="")
    return parser.parse_args()


def labeled_pair(target: Image.Image, pred: Image.Image, label: str) -> Image.Image:
    pad = 24
    canvas = Image.new("RGB", (target.width * 2, target.height + pad), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"target | model {label}", fill=(240, 240, 240))
    canvas.paste(target, (0, pad))
    canvas.paste(pred, (target.width, pad))
    return canvas


@torch.no_grad()
def save_clip_gif(model, clip, device: torch.device, output: Path, *, steps: int, image_size: int, legalize_snake: bool) -> None:
    history_size = int(model.cfg.history_size)
    boards = [extract_board(frame.path) for frame in clip.frames]
    pred_history = boards[:history_size]
    action_history = [frame.action for frame in clip.frames[1 : history_size + 1]]
    snake_body = initialize_snake_body(pred_history)
    frames: list[Image.Image] = []
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
        label = f"step {offset + 1:02d} action {action}"
        frames.append(labeled_pair(render_board(target, image_size), render_board(pred, image_size), label))
        pred_history.append(pred)
        action_history[-1] = action
        action_history.append(action)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=180, loop=0)


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    model = load_model(args, device)
    clips = discover_snake_clips(args.dataset_root, levels=list(args.levels), max_clips_per_level=int(args.max_clips))
    gif_dir = Path(args.gif_dir) if args.gif_dir else Path(args.output_dir) / args.run_name / "rollouts" / "gifs"
    for index, clip in enumerate(clips[: int(args.gif_count)]):
        suffix = "legalized" if args.legalize_snake else "raw"
        output = gif_dir / f"{index:03d}_{clip.level_kind}_{clip.clip_id}_{suffix}.gif"
        save_clip_gif(
            model,
            clip,
            device,
            output,
            steps=int(args.steps),
            image_size=int(args.image_size),
            legalize_snake=bool(args.legalize_snake),
        )
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
