import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from snake_jepa.snake_board import extract_board


@dataclass(frozen=True)
class SnakeFrame:
    path: Path
    action: int


@dataclass(frozen=True)
class SnakeClip:
    clip_id: str
    level_kind: str
    frames: list[SnakeFrame]


def _load_clip(clip_dir: Path, level_kind: str) -> SnakeClip | None:
    meta_path = clip_dir / "clip_metadata.json"
    frames_dir = clip_dir / "frames"
    if not meta_path.exists() or not frames_dir.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

    meta_frames = meta.get("frames", [])
    if not isinstance(meta_frames, list) or len(meta_frames) < 2:
        return None

    frames: list[SnakeFrame] = []
    for frame in meta_frames:
        if not isinstance(frame, dict):
            return None
        image_name = frame.get("image")
        if not isinstance(image_name, str):
            return None
        image_path = frames_dir / image_name
        if image_path.suffix.lower() != ".png" or not image_path.exists():
            return None
        action = int(frame.get("action", 0))
        if action < 0 or action > 3:
            action = 0
        frames.append(SnakeFrame(path=image_path, action=action))

    return SnakeClip(
        clip_id=str(meta.get("clip_id", clip_dir.name)),
        level_kind=level_kind,
        frames=frames,
    )


def discover_snake_clips(
    dataset_root: str | Path,
    *,
    levels: list[str] | None = None,
    max_clips_per_level: int = 0,
) -> list[SnakeClip]:
    root = Path(dataset_root)
    if levels is None:
        levels = ["level_1", "level_2", "level_3", "random_levels"]

    clips: list[SnakeClip] = []
    for level in levels:
        level_dir = root / level
        if not level_dir.exists():
            continue
        prefix = "random_clip_*" if level == "random_levels" else "clip_*"
        added = 0
        for clip_dir in sorted(level_dir.glob(prefix)):
            sample = _load_clip(clip_dir, level)
            if sample is None:
                continue
            clips.append(sample)
            added += 1
            if max_clips_per_level > 0 and added >= max_clips_per_level:
                break

    if not clips:
        raise FileNotFoundError(f"no valid snake clips found under {root}")
    return clips


class SnakeFrameDataset(Dataset):
    def __init__(
        self,
        clips: list[SnakeClip],
        *,
        history_size: int,
        image_size: int,
        stride: int = 1,
        max_windows_per_clip: int = 0,
        include_boards: bool = False,
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        self.image_size = int(image_size)
        self.include_boards = bool(include_boards)
        self.samples: list[tuple[SnakeClip, int]] = []

        for clip in clips:
            max_start = len(clip.frames) - (self.history_size + 1)
            if max_start < 0:
                continue
            added = 0
            for start in range(0, max_start + 1, max(1, int(stride))):
                self.samples.append((clip, start))
                added += 1
                if max_windows_per_clip > 0 and added >= max_windows_per_clip:
                    break

        if not self.samples:
            raise ValueError("snake dataset is empty after windowing")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            frame = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
            frame = frame.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        return frame.float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip, start = self.samples[index]
        hist_end = start + self.history_size

        history_frames = [
            self._load_frame(frame.path) for frame in clip.frames[start:hist_end]
        ]
        next_frame = self._load_frame(clip.frames[hist_end].path)

        actions = torch.tensor(
            [frame.action for frame in clip.frames[start + 1 : hist_end + 1]],
            dtype=torch.long,
        )
        sample = {
            "history_frames": torch.stack(history_frames, dim=0),
            "next_frame": next_frame,
            "actions": actions,
            "actions_one_hot": F.one_hot(actions, num_classes=4).float(),
        }
        if self.include_boards:
            history_boards = [
                extract_board(frame.path) for frame in clip.frames[start:hist_end]
            ]
            sample["history_boards"] = torch.stack(history_boards, dim=0)
            sample["next_board"] = extract_board(clip.frames[hist_end].path)
        return sample


class SnakeBoardDataset(Dataset):
    def __init__(
        self,
        clips: list[SnakeClip],
        *,
        history_size: int,
        rollout_steps: int = 1,
        stride: int = 1,
        max_windows_per_clip: int = 0,
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        self.rollout_steps = max(1, int(rollout_steps))
        self.samples: list[tuple[SnakeClip, int]] = []
        self._board_cache: dict[Path, torch.Tensor] = {}

        for clip in clips:
            max_start = len(clip.frames) - (self.history_size + self.rollout_steps)
            if max_start < 0:
                continue
            added = 0
            for start in range(0, max_start + 1, max(1, int(stride))):
                self.samples.append((clip, start))
                added += 1
                if max_windows_per_clip > 0 and added >= max_windows_per_clip:
                    break

        if not self.samples:
            raise ValueError("snake board dataset is empty after windowing")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_board(self, path: Path) -> torch.Tensor:
        board = self._board_cache.get(path)
        if board is None:
            board = extract_board(path)
            self._board_cache[path] = board
        return board

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip, start = self.samples[index]
        hist_end = start + self.history_size
        rollout_end = hist_end + self.rollout_steps
        history_boards = [
            self._load_board(frame.path) for frame in clip.frames[start:hist_end]
        ]
        actions = torch.tensor(
            [frame.action for frame in clip.frames[start + 1 : hist_end + 1]],
            dtype=torch.long,
        )
        target_boards = [
            self._load_board(frame.path) for frame in clip.frames[hist_end:rollout_end]
        ]
        target_actions = torch.tensor(
            [frame.action for frame in clip.frames[hist_end:rollout_end]],
            dtype=torch.long,
        )
        return {
            "history_boards": torch.stack(history_boards, dim=0),
            "next_board": target_boards[0],
            "target_boards": torch.stack(target_boards, dim=0),
            "actions": actions,
            "actions_one_hot": F.one_hot(actions, num_classes=4).float(),
            "target_actions": target_actions,
        }


def build_snake_loaders(
    dataset_root: str | Path,
    *,
    levels: list[str] | None,
    history_size: int,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    stride: int = 1,
    max_clips_per_level: int = 0,
    max_windows_per_clip: int = 0,
    include_boards: bool = False,
) -> tuple[DataLoader, DataLoader]:
    clips = discover_snake_clips(
        dataset_root,
        levels=levels,
        max_clips_per_level=max_clips_per_level,
    )
    split_idx = max(1, int(round(len(clips) * (1.0 - val_fraction))))
    split_idx = min(split_idx, len(clips) - 1) if len(clips) > 1 else len(clips)

    train_clips = clips[:split_idx]
    val_clips = clips[split_idx:] if split_idx < len(clips) else clips[:1]

    train_dataset = SnakeFrameDataset(
        train_clips,
        history_size=history_size,
        image_size=image_size,
        stride=stride,
        max_windows_per_clip=max_windows_per_clip,
        include_boards=include_boards,
    )
    val_dataset = SnakeFrameDataset(
        val_clips,
        history_size=history_size,
        image_size=image_size,
        stride=stride,
        max_windows_per_clip=max_windows_per_clip,
        include_boards=include_boards,
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


def build_snake_board_loaders(
    dataset_root: str | Path,
    *,
    levels: list[str] | None,
    history_size: int,
    rollout_steps: int = 1,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    stride: int = 1,
    max_clips_per_level: int = 0,
    max_windows_per_clip: int = 0,
) -> tuple[DataLoader, DataLoader]:
    clips = discover_snake_clips(
        dataset_root,
        levels=levels,
        max_clips_per_level=max_clips_per_level,
    )
    split_idx = max(1, int(round(len(clips) * (1.0 - val_fraction))))
    split_idx = min(split_idx, len(clips) - 1) if len(clips) > 1 else len(clips)

    train_clips = clips[:split_idx]
    val_clips = clips[split_idx:] if split_idx < len(clips) else clips[:1]

    train_dataset = SnakeBoardDataset(
        train_clips,
        history_size=history_size,
        rollout_steps=rollout_steps,
        stride=stride,
        max_windows_per_clip=max_windows_per_clip,
    )
    val_dataset = SnakeBoardDataset(
        val_clips,
        history_size=history_size,
        rollout_steps=rollout_steps,
        stride=stride,
        max_windows_per_clip=max_windows_per_clip,
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
