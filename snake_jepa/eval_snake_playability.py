from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_jepa.eval_snake_board_rollout import load_model
from snake_jepa.infer_snake_board_dynamics import detect_device
from snake_jepa.snake_board import FOOD, HEAD, OBSTACLE, SNAKE, extract_board
from snake_jepa.snake_board_rollout import initialize_snake_body, legalize_snake_transition, terminal_transition
from snake_jepa.snake_data import discover_snake_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="verify playable snake model rollouts under sampled actions")
    parser.add_argument("--dataset-root", type=str, default="/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent")
    parser.add_argument("--output-dir", type=str, default="runs/snake_board_dynamics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--levels", nargs="*", default=["level_1", "level_2", "level_3", "random_levels"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def board_errors(board: torch.Tensor, body: list[tuple[int, int]], *, split_head: bool, expected_obstacles: torch.Tensor) -> list[str]:
    errors: list[str] = []
    snake_mask = board.eq(SNAKE) | board.eq(HEAD)
    food_mask = board.eq(FOOD)
    obstacle_mask = board.eq(OBSTACLE)
    body_set = set(body)
    board_snake = set(map(tuple, snake_mask.nonzero().tolist()))

    if int(food_mask.sum().item()) != 1:
        errors.append("food_count")
    if body_set != board_snake:
        errors.append("snake_body_mismatch")
    if food_mask.logical_and(snake_mask).any().item():
        errors.append("food_on_snake")
    if food_mask.logical_and(obstacle_mask).any().item():
        errors.append("food_on_obstacle")
    if snake_mask.logical_and(obstacle_mask).any().item():
        errors.append("snake_on_obstacle")
    if not obstacle_mask.eq(expected_obstacles).all().item():
        errors.append("obstacles_changed")
    if split_head and int(board.eq(HEAD).sum().item()) != 1:
        errors.append("head_count")
    return errors


@torch.no_grad()
def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    device = detect_device(args.device)
    model = load_model(args, device)
    history_size = int(model.cfg.history_size)
    split_head = int(model.cfg.num_classes) > 4
    clips = discover_snake_clips(args.dataset_root, levels=list(args.levels), max_clips_per_level=int(args.max_clips))

    totals = Counter()
    examples: list[dict[str, object]] = []
    for clip in clips:
        if len(clip.frames) <= history_size:
            continue
        history_boards = [extract_board(frame.path, split_head=split_head) for frame in clip.frames[:history_size]]
        expected_obstacles = history_boards[-1].eq(OBSTACLE)
        action_history = [frame.action for frame in clip.frames[1 : history_size + 1]]
        snake_body = initialize_snake_body(history_boards)
        totals["clips"] += 1
        for step in range(int(args.steps)):
            requested_action = rng.randrange(4)
            game_over, effective_action, reason = terminal_transition(history_boards[-1], snake_body, requested_action)
            if game_over:
                totals["terminal_events"] += 1
                totals[f"terminal_{reason}"] += 1
                break

            history = torch.stack(history_boards[-history_size:], dim=0).unsqueeze(0).to(device)
            action_window = list(action_history[-history_size:])
            action_window[-1] = effective_action
            action_tensor = torch.tensor(action_window, dtype=torch.long, device=device).unsqueeze(0)
            action_one_hot = F.one_hot(action_tensor, num_classes=4).float()
            logits = model(history, action_one_hot)[0].cpu()
            pred = logits.argmax(dim=0)
            pred, snake_body, effective_action = legalize_snake_transition(
                history_boards[-1],
                pred,
                snake_body,
                effective_action,
                food_scores=logits[FOOD],
            )
            errors = board_errors(pred, list(snake_body), split_head=split_head, expected_obstacles=expected_obstacles)
            totals["steps"] += 1
            if errors:
                totals["bad_steps"] += 1
                for error in errors:
                    totals[f"error_{error}"] += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "level": clip.level_kind,
                            "clip": clip.clip_id,
                            "step": step + 1,
                            "errors": errors,
                        }
                    )
                break
            history_boards.append(pred)
            action_history[-1] = effective_action
            action_history.append(effective_action)

    result = dict(totals)
    result["ok"] = int(totals["bad_steps"]) == 0
    result["examples"] = examples
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
