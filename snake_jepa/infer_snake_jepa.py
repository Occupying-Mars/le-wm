from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button
from PIL import Image

from snake_jepa.snake_board import render_board
from snake_jepa.snake_data import SnakeClip, discover_snake_clips
from snake_jepa.snake_world_model import SnakePatchWorldModel, SnakePatchWorldModelConfig


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
    parser = argparse.ArgumentParser(description="play a trained snake jepa world model")
    parser.add_argument("--dataset-root", type=str, default="/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent")
    parser.add_argument("--output-dir", type=str, default="runs/snake_jepa")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--levels", nargs="*", default=["level_1", "level_2", "level_3", "random_levels"])
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--decode-mode", choices=["board", "pixel"], default="board")
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


def load_model(args: argparse.Namespace, device: torch.device) -> SnakePatchWorldModel:
    path = checkpoint_path(args.output_dir, args.run_name, args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type not in {None, "snake_patch_world_model_v1"}:
        raise RuntimeError(f"unsupported snake checkpoint model_type={model_type!r}")
    config = SnakePatchWorldModelConfig(**checkpoint["model_config"])
    model = SnakePatchWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_frame(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
        frame = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
        frame = frame.view(image_size, image_size, 3).permute(2, 0, 1)
    return frame.float().div(255.0)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    frame = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    frame = frame.view(image.height, image.width, 3).permute(2, 0, 1)
    return frame.float().div(255.0)


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    return image.mul(255).byte().permute(1, 2, 0).numpy()


class SnakeWorldUI:
    def __init__(
        self,
        clips: list[SnakeClip],
        model: SnakePatchWorldModel,
        device: torch.device,
        sample_index: int,
        decode_mode: str,
    ) -> None:
        self.clips = clips
        self.model = model
        self.device = device
        self.decode_mode = decode_mode
        self.sample_index = sample_index if sample_index >= 0 else random.randint(0, len(clips) - 1)
        self.selected_action = 1
        self.step_count = 0

        self.history_frames: list[torch.Tensor] = []
        self.action_history: list[int] = []
        self.seed_context()

        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(bottom=0.24)
        self._add_buttons()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def seed_context(self) -> None:
        clip = self.clips[self.sample_index % len(self.clips)]
        history_size = int(self.model.cfg.history_size)
        image_size = int(self.model.cfg.image_size)
        if len(clip.frames) <= history_size:
            raise RuntimeError(f"clip {clip.clip_id} is too short for history_size={history_size}")
        self.history_frames = [
            load_frame(frame.path, image_size) for frame in clip.frames[:history_size]
        ]
        self.action_history = [frame.action for frame in clip.frames[:history_size]]
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
        actions = list(self.action_history[-int(self.model.cfg.history_size):])
        actions[-1] = action
        history = torch.stack(self.history_frames[-int(self.model.cfg.history_size):], dim=0).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0)
        action_one_hot = F.one_hot(action_tensor, num_classes=4).float()
        output = self.model.predict_next(history, action_one_hot)
        if self.decode_mode == "pixel":
            return self.model.decoder(output["pred_next_latent"])[0].cpu()
        board_logits = self.model.board_decoder(output["pred_next_latent"])
        board = board_logits[0].argmax(dim=0)
        return pil_to_tensor(render_board(board, int(self.model.cfg.image_size)))

    def step(self, action: int) -> None:
        self.selected_action = int(action)
        pred = self.predict_next(self.selected_action)
        self.history_frames.append(pred)
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
        history_strip = torch.cat(self.history_frames[-int(self.model.cfg.history_size):], dim=2)
        current = self.history_frames[-1]

        self.axes[0].clear()
        self.axes[0].imshow(tensor_to_np(history_strip))
        self.axes[0].set_title("model history")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(tensor_to_np(current))
        self.axes[1].set_title("current model frame")
        self.axes[1].axis("off")

        clip = self.clips[self.sample_index % len(self.clips)]
        self.fig.suptitle(
            f"seed {clip.level_kind}/{clip.clip_id} | model steps {self.step_count} | "
            f"last action {ACTION_NAMES[self.selected_action]} | decode {self.decode_mode}",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"[snake-jepa] device: {device}")
    model = load_model(args, device)
    clips = discover_snake_clips(args.dataset_root, levels=list(args.levels), max_clips_per_level=0)
    print(f"[snake-jepa] clips: {len(clips)}")
    print("[snake-jepa] controls: arrow keys or wasd, r reset, n new seed")
    SnakeWorldUI(clips, model, device, args.sample_index, args.decode_mode)
    plt.show()


if __name__ == "__main__":
    main()
