from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button
from PIL import Image

from snake_jepa.snake_board import extract_board, render_board
from snake_jepa.snake_board_rollout import initialize_snake_body, legalize_snake_transition, terminal_transition
from snake_jepa.snake_board_model import SnakeBoardDynamics, SnakeBoardDynamicsConfig
from snake_jepa.snake_data import SnakeClip, discover_snake_clips


ACTION_NAMES = ["up", "right", "down", "left"]
KEY_TO_ACTION = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
    "w": 0,
    "d": 1,
    "s": 2,
    "a": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="play or roll out a trained snake board dynamics model")
    parser.add_argument("--dataset-root", type=str, default="/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent")
    parser.add_argument("--output-dir", type=str, default="runs/snake_board_dynamics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--levels", nargs="*", default=["level_1", "level_2", "level_3", "random_levels"])
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--save-gif", type=str, default="")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--legalize-snake", action="store_true")
    return parser.parse_args()


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_path(output_dir: str | Path, run_name: str, checkpoint: str) -> Path:
    run_dir = Path(output_dir) / run_name / "checkpoints"
    if checkpoint in {"best", "latest"}:
        return run_dir / f"{checkpoint}.pt"
    path = Path(checkpoint)
    if path.exists():
        return path
    return run_dir / checkpoint


def load_model(args: argparse.Namespace, device: torch.device) -> SnakeBoardDynamics:
    path = checkpoint_path(args.output_dir, args.run_name, args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type != "snake_board_dynamics_v1":
        raise RuntimeError(f"unsupported checkpoint model_type={model_type!r}")
    model = SnakeBoardDynamics(SnakeBoardDynamicsConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def board_to_array(board: torch.Tensor, image_size: int):
    image = render_board(board, image_size)
    return image


class SnakeBoardUI:
    def __init__(
        self,
        clips: list[SnakeClip],
        model: SnakeBoardDynamics,
        device: torch.device,
        sample_index: int,
        image_size: int,
        legalize_snake: bool,
    ) -> None:
        self.clips = clips
        self.model = model
        self.device = device
        self.image_size = image_size
        self.legalize_snake = bool(legalize_snake)
        self.sample_index = sample_index if sample_index >= 0 else random.randint(0, len(clips) - 1)
        self.selected_action = 1
        self.step_count = 0
        self.history_boards: list[torch.Tensor] = []
        self.action_history: list[int] = []
        self.snake_body = initialize_snake_body([])
        self.game_over = False
        self.game_over_reason = ""
        self.seed_context()

        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(bottom=0.24)
        self._add_buttons()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def seed_context(self) -> None:
        clip = self.clips[self.sample_index % len(self.clips)]
        history_size = int(self.model.cfg.history_size)
        split_head = int(self.model.cfg.num_classes) > 4
        if len(clip.frames) <= history_size:
            raise RuntimeError(f"clip {clip.clip_id} is too short for history_size={history_size}")
        self.history_boards = [extract_board(frame.path, split_head=split_head) for frame in clip.frames[:history_size]]
        self.action_history = [frame.action for frame in clip.frames[1 : history_size + 1]]
        self.snake_body = initialize_snake_body(self.history_boards)
        self.game_over = False
        self.game_over_reason = ""
        self.step_count = 0

    def _add_buttons(self) -> None:
        buttons = [
            ("up", [0.28, 0.06, 0.08, 0.07], lambda _e: self.step(0)),
            ("right", [0.37, 0.06, 0.08, 0.07], lambda _e: self.step(1)),
            ("down", [0.46, 0.06, 0.08, 0.07], lambda _e: self.step(2)),
            ("left", [0.55, 0.06, 0.08, 0.07], lambda _e: self.step(3)),
            ("reset", [0.66, 0.06, 0.10, 0.07], self.reset),
            ("new seed", [0.78, 0.06, 0.12, 0.07], self.new_seed),
        ]
        self.button_refs = []
        for label, rect, callback in buttons:
            axis = plt.axes(rect)
            button = Button(axis, label)
            button.on_clicked(callback)
            self.button_refs.append(button)

    @torch.no_grad()
    def predict_next(self, action: int) -> torch.Tensor:
        history = torch.stack(self.history_boards[-int(self.model.cfg.history_size):], dim=0).unsqueeze(0).to(self.device)
        actions = list(self.action_history[-int(self.model.cfg.history_size):])
        actions[-1] = int(action)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0)
        action_one_hot = F.one_hot(action_tensor, num_classes=4).float()
        return self.model(history, action_one_hot)[0].argmax(dim=0).cpu()

    def step(self, action: int) -> None:
        if self.game_over:
            return
        self.selected_action = int(action)
        if self.legalize_snake:
            game_over, effective, reason = terminal_transition(
                self.history_boards[-1],
                self.snake_body,
                self.selected_action,
            )
            self.selected_action = effective
            if game_over:
                self.game_over = True
                self.game_over_reason = reason
                self.render()
                return
        pred = self.predict_next(self.selected_action)
        if self.legalize_snake:
            pred, self.snake_body, self.selected_action = legalize_snake_transition(
                self.history_boards[-1],
                pred,
                self.snake_body,
                self.selected_action,
            )
        self.action_history[-1] = self.selected_action
        self.history_boards.append(pred)
        self.action_history.append(self.selected_action)
        self.step_count += 1
        self.render()

    def reset(self, _event=None) -> None:
        self.seed_context()
        self.render()

    def new_seed(self, _event=None) -> None:
        self.sample_index = random.randint(0, len(self.clips) - 1)
        self.seed_context()
        self.render()

    def on_key(self, event) -> None:
        if event.key in KEY_TO_ACTION:
            self.step(KEY_TO_ACTION[event.key])
        elif event.key == "r":
            self.reset()
        elif event.key == "n":
            self.new_seed()

    def render(self) -> None:
        history_images = [render_board(board, self.image_size) for board in self.history_boards[-int(self.model.cfg.history_size):]]
        history_strip = Image.new("RGB", (self.image_size * len(history_images), self.image_size))
        for idx, image in enumerate(history_images):
            history_strip.paste(image, (idx * self.image_size, 0))
        current = render_board(self.history_boards[-1], self.image_size)
        self.axes[0].clear()
        self.axes[0].imshow(history_strip)
        self.axes[0].set_title("model board history")
        self.axes[0].axis("off")
        self.axes[1].clear()
        self.axes[1].imshow(current)
        self.axes[1].set_title("current model board")
        self.axes[1].axis("off")
        clip = self.clips[self.sample_index % len(self.clips)]
        state = f"GAME OVER: {self.game_over_reason}" if self.game_over else f"last action {ACTION_NAMES[self.selected_action]}"
        self.fig.suptitle(
            f"seed {clip.level_kind}/{clip.clip_id} | model steps {self.step_count} | "
            f"{state}",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()


@torch.no_grad()
def save_teacher_forced_gif(
    model: SnakeBoardDynamics,
    clip: SnakeClip,
    device: torch.device,
    path: str | Path,
    *,
    image_size: int,
    steps: int,
    legalize_snake: bool,
) -> None:
    history_size = int(model.cfg.history_size)
    split_head = int(model.cfg.num_classes) > 4
    boards = [extract_board(frame.path, split_head=split_head) for frame in clip.frames[:history_size]]
    actions = [frame.action for frame in clip.frames[1 : history_size + 1]]
    snake_body = initialize_snake_body(boards)
    frames = [render_board(board, image_size) for board in boards]
    max_steps = min(int(steps), len(clip.frames) - history_size - 1)
    for offset in range(max_steps):
        action = clip.frames[history_size + offset].action
        history = torch.stack(boards[-history_size:], dim=0).unsqueeze(0).to(device)
        action_window = list(actions[-history_size:])
        action_window[-1] = action
        action_tensor = torch.tensor(action_window, dtype=torch.long, device=device).unsqueeze(0)
        action_one_hot = F.one_hot(action_tensor, num_classes=4).float()
        pred = model(history, action_one_hot)[0].argmax(dim=0).cpu()
        if legalize_snake:
            pred, snake_body, action = legalize_snake_transition(boards[-1], pred, snake_body, action)
        boards.append(pred)
        actions[-1] = action
        actions.append(action)
        frames.append(render_board(pred, image_size))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=160, loop=0)


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"[snake-board-dyn] device: {device}")
    model = load_model(args, device)
    clips = discover_snake_clips(args.dataset_root, levels=list(args.levels), max_clips_per_level=0)
    print(f"[snake-board-dyn] clips: {len(clips)}")
    sample_index = args.sample_index if args.sample_index >= 0 else random.randint(0, len(clips) - 1)
    if args.save_gif:
        save_teacher_forced_gif(
            model,
            clips[sample_index % len(clips)],
            device,
            args.save_gif,
            image_size=int(args.image_size),
            steps=int(args.steps),
            legalize_snake=bool(args.legalize_snake),
        )
        print(f"[snake-board-dyn] saved {args.save_gif}")
        return
    print("[snake-board-dyn] controls: arrow keys or wasd, r reset, n new seed")
    SnakeBoardUI(clips, model, device, sample_index, int(args.image_size), bool(args.legalize_snake))
    plt.show()


if __name__ == "__main__":
    main()
