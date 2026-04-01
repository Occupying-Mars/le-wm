import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DebugBoxSample:
    frame_paths: list[Path]
    actions: list[int]
    states: list[tuple[float, float]]


def _load_clip(clip_dir: Path) -> DebugBoxSample | None:
    meta_path = clip_dir / "clip_metadata.json"
    frames_dir = clip_dir / "frames"
    if not meta_path.exists() or not frames_dir.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

    frames = meta.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        return None

    frame_paths: list[Path] = []
    actions: list[int] = []
    states: list[tuple[float, float]] = []
    max_pos = max(1.0, float(meta.get("frame_size", 256) - meta.get("box_size", 72)))

    for frame in frames:
        image_name = frame.get("image")
        action = int(frame.get("action", 0))
        box_x = float(frame.get("box_x", 0.0)) / max_pos
        box_y = float(frame.get("box_y", 0.0)) / max_pos
        image_path = frames_dir / str(image_name)
        if not image_path.exists():
            return None
        frame_paths.append(image_path)
        actions.append(action)
        states.append((box_x, box_y))

    return DebugBoxSample(frame_paths=frame_paths, actions=actions, states=states)


def discover_debug_box_clips(dataset_root: str | Path) -> list[DebugBoxSample]:
    root = Path(dataset_root)
    clips: list[DebugBoxSample] = []
    for clip_dir in sorted((root / "level_1").glob("clip_*")):
        sample = _load_clip(clip_dir)
        if sample is not None:
            clips.append(sample)
    if not clips:
        raise FileNotFoundError(f"no valid debug-box clips found under {root}")
    return clips


class DebugBoxDataset(Dataset):
    def __init__(
        self,
        clips: list[DebugBoxSample],
        *,
        history_size: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.history_size = history_size
        self.image_size = image_size
        self.samples: list[tuple[DebugBoxSample, int]] = []

        for clip in clips:
            max_start = len(clip.frame_paths) - (history_size + 1)
            for start in range(max_start + 1):
                self.samples.append((clip, start))

        if not self.samples:
            raise ValueError("debug-box dataset is empty after windowing")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            frame = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
            frame = frame.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        return frame.float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip, start = self.samples[index]
        hist_end = start + self.history_size

        history_frames = [
            self._load_frame(path) for path in clip.frame_paths[start:hist_end]
        ]
        next_frame = self._load_frame(clip.frame_paths[hist_end])

        actions = torch.tensor(clip.actions[start:hist_end], dtype=torch.long)
        actions_one_hot = F.one_hot(actions, num_classes=4).float()

        history_states = torch.tensor(clip.states[start:hist_end], dtype=torch.float32)
        next_state = torch.tensor(clip.states[hist_end], dtype=torch.float32)

        return {
            "history_frames": torch.stack(history_frames, dim=0),
            "next_frame": next_frame,
            "actions": actions,
            "actions_one_hot": actions_one_hot,
            "history_states": history_states,
            "next_state": next_state,
        }


def build_debug_box_loaders(
    dataset_root: str | Path,
    *,
    history_size: int,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    clips = discover_debug_box_clips(dataset_root)
    split_idx = max(1, int(round(len(clips) * (1.0 - val_fraction))))
    split_idx = min(split_idx, len(clips) - 1) if len(clips) > 1 else len(clips)

    train_clips = clips[:split_idx]
    val_clips = clips[split_idx:] if split_idx < len(clips) else clips[:1]

    train_dataset = DebugBoxDataset(
        train_clips, history_size=history_size, image_size=image_size
    )
    val_dataset = DebugBoxDataset(
        val_clips, history_size=history_size, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
