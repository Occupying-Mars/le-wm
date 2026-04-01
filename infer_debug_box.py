from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from PIL import Image

from debug_box_data import DebugBoxSample, discover_debug_box_clips
from debug_box_world_model import DebugBoxWorldModel, DebugBoxWorldModelConfig, LatentDecoder


ACTION_NAMES = ["up", "right", "down", "left"]
DECODER_INPUT_MODE = "vit_preproj_cls_v1"


@dataclass(frozen=True)
class WindowRef:
    clip_index: int
    start: int


@dataclass(frozen=True)
class FrameRef:
    clip_index: int
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="interactive inference for debug-box lewm")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/Users/krishna/Public/ml-experiments/snake-we/world-env/datasets/debug_box",
    )
    parser.add_argument("--wm-output-dir", type=str, default="runs/debug_box_vit_mps")
    parser.add_argument("--wm-run-name", type=str, required=True)
    parser.add_argument("--decoder-output-dir", type=str, default="runs/debug_box_decoder")
    parser.add_argument("--decoder-run-name", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frame(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        frame = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
        frame = frame.view(image_size, image_size, 3).permute(2, 0, 1)
    return frame.float().div(255.0)


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    image = image.clamp(0.0, 1.0)
    return image.mul(255).byte().permute(1, 2, 0).cpu().numpy()


def build_windows(clips: list[DebugBoxSample], history_size: int) -> list[WindowRef]:
    windows: list[WindowRef] = []
    for clip_index, clip in enumerate(clips):
        max_start = len(clip.frame_paths) - (history_size + 1)
        for start in range(max_start + 1):
            windows.append(WindowRef(clip_index=clip_index, start=start))
    if not windows:
        raise RuntimeError("no valid windows found for inference")
    return windows


def build_frame_refs(clips: list[DebugBoxSample]) -> list[FrameRef]:
    frame_refs: list[FrameRef] = []
    for clip_index, clip in enumerate(clips):
        for frame_index in range(len(clip.frame_paths)):
            frame_refs.append(FrameRef(clip_index=clip_index, frame_index=frame_index))
    if not frame_refs:
        raise RuntimeError("no frames found for decoder-only inference")
    return frame_refs


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


def load_decoder(args: argparse.Namespace, world_model: DebugBoxWorldModel, device: torch.device) -> LatentDecoder:
    checkpoint_path = get_checkpoint_path(args.decoder_output_dir, args.decoder_run_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_mode = checkpoint.get("decoder_input_mode", "cls_global_v0")
    if checkpoint_mode != DECODER_INPUT_MODE:
        raise RuntimeError(
            f"decoder checkpoint expects {checkpoint_mode}, but decoder-only reconstruction now uses "
            f"{DECODER_INPUT_MODE}. retrain the decoder with the updated token path."
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
    try:
        decoder.load_state_dict(checkpoint["decoder_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"decoder checkpoint is incompatible with the current patch-query decoder: {checkpoint_path}. "
            "retrain the debug-box decoder or load a checkpoint produced by the new architecture."
        ) from exc
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


class InferenceUI:
    def __init__(
        self,
        clips: list[DebugBoxSample],
        windows: list[WindowRef],
        world_model: DebugBoxWorldModel,
        decoder: LatentDecoder,
        device: torch.device,
        sample_index: int,
    ) -> None:
        self.clips = clips
        self.windows = windows
        self.world_model = world_model
        self.decoder = decoder
        self.device = device
        self.window_index = sample_index % len(windows)
        self.selected_action = 1

        self.fig, self.axes = plt.subplots(1, 3, figsize=(13, 4))
        plt.subplots_adjust(bottom=0.22)
        self._add_buttons()
        self.render()

    def _add_buttons(self) -> None:
        buttons = [
            ("prev", [0.08, 0.05, 0.10, 0.06], self.prev_sample),
            ("next", [0.19, 0.05, 0.10, 0.06], self.next_sample),
            ("up", [0.40, 0.05, 0.08, 0.06], lambda _e: self.set_action(0)),
            ("right", [0.49, 0.05, 0.08, 0.06], lambda _e: self.set_action(1)),
            ("down", [0.58, 0.05, 0.08, 0.06], lambda _e: self.set_action(2)),
            ("left", [0.67, 0.05, 0.08, 0.06], lambda _e: self.set_action(3)),
            ("gt action", [0.78, 0.05, 0.12, 0.06], self.use_gt_action),
        ]
        self.button_refs = []
        for label, rect, callback in buttons:
            axis = plt.axes(rect)
            button = Button(axis, label)
            button.on_clicked(callback)
            self.button_refs.append(button)

    def _get_window(self) -> tuple[DebugBoxSample, WindowRef]:
        window = self.windows[self.window_index]
        return self.clips[window.clip_index], window

    def _build_context(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        clip, window = self._get_window()
        history_size = int(self.world_model.cfg.history_size)
        image_size = int(self.world_model.cfg.image_size)
        start = window.start
        hist_end = start + history_size

        history_frames = torch.stack(
            [load_frame(path, image_size) for path in clip.frame_paths[start:hist_end]],
            dim=0,
        )
        gt_next = load_frame(clip.frame_paths[hist_end], image_size)

        past_actions = clip.actions[start : hist_end - 1]
        gt_action = clip.actions[hist_end - 1]
        actions = torch.tensor(past_actions + [self.selected_action], dtype=torch.long)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=4).float()
        return history_frames, actions_one_hot, gt_next, gt_action

    @torch.no_grad()
    def predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        history_frames, actions_one_hot, gt_next, gt_action = self._build_context()
        history_batch = history_frames.unsqueeze(0).to(self.device)
        action_batch = actions_one_hot.unsqueeze(0).to(self.device)
        output = self.world_model.predict_next(history_batch, action_batch)
        pred_frame = self.world_model.decoder(output["pred_next_latent"])[0].cpu()

        history_strip = torch.cat(list(history_frames), dim=2)
        return (
            tensor_to_np(history_strip),
            tensor_to_np(pred_frame),
            tensor_to_np(gt_next),
            gt_action,
        )

    def render(self) -> None:
        history_img, pred_img, gt_img, gt_action = self.predict()
        clip, window = self._get_window()

        self.axes[0].clear()
        self.axes[0].imshow(history_img)
        self.axes[0].set_title("context")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(pred_img)
        self.axes[1].set_title(f"predicted next ({ACTION_NAMES[self.selected_action]})")
        self.axes[1].axis("off")

        self.axes[2].clear()
        self.axes[2].imshow(gt_img)
        self.axes[2].set_title(f"ground truth next ({ACTION_NAMES[gt_action]})")
        self.axes[2].axis("off")

        self.fig.suptitle(
            f"sample {self.window_index} | clip {window.clip_index} | start {window.start} | "
            f"selected={ACTION_NAMES[self.selected_action]} | gt={ACTION_NAMES[gt_action]}",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def prev_sample(self, _event) -> None:
        self.window_index = (self.window_index - 1) % len(self.windows)
        self.render()

    def next_sample(self, _event) -> None:
        self.window_index = (self.window_index + 1) % len(self.windows)
        self.render()

    def set_action(self, action: int) -> None:
        self.selected_action = action
        self.render()

    def use_gt_action(self, _event) -> None:
        _, _, _, gt_action = self._build_context()
        self.selected_action = gt_action
        self.render()


@torch.no_grad()
def show_decoder_only_grid(
    clips: list[DebugBoxSample],
    world_model: DebugBoxWorldModel,
    decoder: LatentDecoder,
    device: torch.device,
    sample_index: int,
    num_samples: int,
) -> None:
    frame_refs = build_frame_refs(clips)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    start_index = sample_index if sample_index >= 0 else random.randint(0, len(frame_refs) - 1)
    selected = [frame_refs[(start_index + offset) % len(frame_refs)] for offset in range(num_samples)]

    image_size = int(world_model.cfg.image_size)
    gt_frames = torch.stack(
        [load_frame(clips[ref.clip_index].frame_paths[ref.frame_index], image_size) for ref in selected],
        dim=0,
    )
    cls_latents = world_model.encode_next_cls(gt_frames.to(device))
    decoded = decoder(cls_latents).cpu()

    fig, axes = plt.subplots(2, num_samples, figsize=(2.8 * num_samples, 5.2), squeeze=False)
    for column, ref in enumerate(selected):
        axes[0, column].imshow(tensor_to_np(gt_frames[column]))
        axes[0, column].set_title(f"gt {column + 1}\nc{ref.clip_index} f{ref.frame_index}")
        axes[0, column].axis("off")

        axes[1, column].imshow(tensor_to_np(decoded[column]))
        axes[1, column].set_title(f"decoded {column + 1}")
        axes[1, column].axis("off")

    fig.suptitle("decoder-only: gt frame -> last-layer cls -> decoder reconstruction", fontsize=12)
    plt.tight_layout()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"[infer-debug-box] device: {device}")

    world_model = load_world_model(args, device)
    decoder = load_decoder(args, world_model, device)
    clips = discover_debug_box_clips(args.dataset_root)
    if args.decoder_only:
        print(f"[infer-debug-box] loaded {len(clips)} clips")
        print(f"[infer-debug-box] decoder-only mode with {args.num_samples} samples")
        show_decoder_only_grid(
            clips,
            world_model,
            decoder,
            device,
            args.sample_index,
            args.num_samples,
        )
        plt.show()
        return

    windows = build_windows(clips, int(world_model.cfg.history_size))
    sample_index = args.sample_index if args.sample_index >= 0 else random.randint(0, len(windows) - 1)

    print(f"[infer-debug-box] loaded {len(clips)} clips and {len(windows)} windows")
    print(f"[infer-debug-box] starting sample index: {sample_index}")
    print(f"[infer-debug-box] controls: prev/next sample, action buttons, gt action")

    InferenceUI(clips, windows, world_model, decoder, device, sample_index)
    plt.show()


if __name__ == "__main__":
    main()
