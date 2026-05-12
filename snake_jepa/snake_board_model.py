from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from snake_jepa.snake_board import NUM_BOARD_CLASSES


@dataclass
class SnakeBoardDynamicsConfig:
    history_size: int = 4
    action_dim: int = 4
    num_classes: int = NUM_BOARD_CLASSES
    hidden_dim: int = 128
    depth: int = 8
    dropout: float = 0.0


class BoardResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class SnakeBoardDynamics(nn.Module):
    def __init__(self, cfg: SnakeBoardDynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.history_size * cfg.num_classes + cfg.action_dim
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[BoardResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.out_proj = nn.Conv2d(cfg.hidden_dim, cfg.num_classes, kernel_size=1)

    def forward(self, history_boards: torch.Tensor, actions_one_hot: torch.Tensor) -> torch.Tensor:
        if history_boards.dim() not in {3, 4, 5}:
            raise ValueError(f"expected board history rank 3, 4, or 5, got {tuple(history_boards.shape)}")
        if history_boards.dim() == 3:
            history_boards = history_boards.unsqueeze(0)
        batch_size, history_size = history_boards.shape[:2]
        if history_size != self.cfg.history_size:
            raise ValueError(f"expected history size {self.cfg.history_size}, got {history_size}")
        if history_boards.dim() == 5:
            if history_boards.size(2) != self.cfg.num_classes:
                raise ValueError(f"expected {self.cfg.num_classes} class channels, got {history_boards.size(2)}")
            board_one_hot = history_boards.float()
            height, width = history_boards.shape[-2:]
        else:
            height, width = history_boards.shape[-2:]
            board_one_hot = F.one_hot(history_boards.long(), num_classes=self.cfg.num_classes)
            board_one_hot = board_one_hot.permute(0, 1, 4, 2, 3).float()
        board_channels = board_one_hot.reshape(batch_size, history_size * self.cfg.num_classes, height, width)
        next_action = actions_one_hot[:, -1]
        action_channels = next_action[:, :, None, None].expand(batch_size, self.cfg.action_dim, height, width)
        return self.out_proj(self.blocks(self.in_proj(torch.cat([board_channels, action_channels], dim=1))))

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
