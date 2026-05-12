from dataclasses import dataclass

import torch
from torch import nn

from debug_box_world_model import LatentDynamics, TinyViTEncoder
from snake_jepa.snake_board import NUM_BOARD_CLASSES


@dataclass
class SnakePatchWorldModelConfig:
    image_size: int = 128
    patch_size: int = 16
    history_size: int = 4
    action_dim: int = 4
    encoder_dim: int = 192
    encoder_depth: int = 4
    encoder_heads: int = 4
    encoder_mlp_ratio: float = 4.0
    latent_dim: int = 96
    dynamics_dim: int = 192
    dynamics_depth: int = 3
    dynamics_heads: int = 4
    dynamics_mlp_ratio: float = 2.0
    decoder_dim: int = 192
    decoder_depth: int = 4
    decoder_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    dropout: float = 0.0


class SnakePatchWorldModel(nn.Module):
    def __init__(self, cfg: SnakePatchWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TinyViTEncoder(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            dim=cfg.encoder_dim,
            depth=cfg.encoder_depth,
            heads=cfg.encoder_heads,
            mlp_ratio=cfg.encoder_mlp_ratio,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
        self.dynamics = LatentDynamics(
            history_size=cfg.history_size,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            dim=cfg.dynamics_dim,
            depth=cfg.dynamics_depth,
            heads=cfg.dynamics_heads,
            mlp_ratio=cfg.dynamics_mlp_ratio,
            dropout=cfg.dropout,
        )
        self.decoder = OrderedPatchDecoder(
            latent_dim=cfg.latent_dim,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.decoder_dim,
            dropout=cfg.dropout,
        )
        self.board_decoder = PatchBoardDecoder(
            latent_dim=cfg.latent_dim,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.decoder_dim,
            num_classes=NUM_BOARD_CLASSES,
            dropout=cfg.dropout,
        )

    @property
    def num_patches(self) -> int:
        grid = self.cfg.image_size // self.cfg.patch_size
        return grid * grid

    def encode_frame_patches(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_patches(frames)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        batch_size, history_size = history_frames.shape[:2]
        flat_frames = history_frames.reshape(batch_size * history_size, *history_frames.shape[2:])
        flat_latents = self.encode_frame_patches(flat_frames)
        return flat_latents.reshape(batch_size, history_size, flat_latents.size(1), flat_latents.size(2))

    def encode_next(self, next_frame: torch.Tensor) -> torch.Tensor:
        return self.encode_frame_patches(next_frame)

    def predict_patch_latents(
        self,
        history_latents: torch.Tensor,
        actions_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_size, num_patches, latent_dim = history_latents.shape
        patch_history = history_latents.permute(0, 2, 1, 3).reshape(
            batch_size * num_patches,
            history_size,
            latent_dim,
        )
        patch_actions = actions_one_hot.unsqueeze(1).expand(
            batch_size,
            num_patches,
            history_size,
            actions_one_hot.size(-1),
        ).reshape(batch_size * num_patches, history_size, actions_one_hot.size(-1))
        pred = self.dynamics(patch_history, patch_actions)
        return pred.reshape(batch_size, num_patches, latent_dim)

    def predict_next(self, history_frames: torch.Tensor, actions_one_hot: torch.Tensor) -> dict[str, torch.Tensor]:
        history_latents = self.encode_history(history_frames)
        pred_next_latent = self.predict_patch_latents(history_latents, actions_one_hot)
        return {
            "history_latents": history_latents,
            "pred_next_latent": pred_next_latent,
        }

    def decode_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, history_size, num_patches, latent_dim = latents.shape
        flat = latents.reshape(batch_size * history_size, num_patches, latent_dim)
        decoded = self.decoder(flat)
        return decoded.reshape(batch_size, history_size, *decoded.shape[1:])

    def decode_board_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, history_size, num_patches, latent_dim = latents.shape
        flat = latents.reshape(batch_size * history_size, num_patches, latent_dim)
        decoded = self.board_decoder(flat)
        return decoded.reshape(batch_size, history_size, *decoded.shape[1:])

    def forward(
        self,
        history_frames: torch.Tensor,
        actions_one_hot: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_latents = self.encode_history(history_frames)
        target_next_latent = self.encode_next(next_frame)
        pred_next_latent = self.predict_patch_latents(history_latents, actions_one_hot)

        history_recon = self.decode_sequence(history_latents)
        history_board_logits = self.decode_board_sequence(history_latents)
        pred_next_frame = self.decoder(pred_next_latent)
        target_next_recon = self.decoder(target_next_latent)
        pred_next_board_logits = self.board_decoder(pred_next_latent)
        target_next_board_logits = self.board_decoder(target_next_latent)

        return {
            "history_latents": history_latents,
            "target_next_latent": target_next_latent,
            "pred_next_latent": pred_next_latent,
            "history_recon": history_recon,
            "history_board_logits": history_board_logits,
            "pred_next_frame": pred_next_frame,
            "target_next_recon": target_next_recon,
            "pred_next_board_logits": pred_next_board_logits,
            "target_next_board_logits": target_next_board_logits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OrderedPatchDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = self.patch_size * self.patch_size * 3
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.patch_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"expected patch latents with rank 3, got {tuple(latents.shape)}")
        batch_size, num_patches, _ = latents.shape
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        patches = self.net(latents).sigmoid()
        patches = patches.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            3,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            3,
            self.image_size,
            self.image_size,
        )


class PatchBoardDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"expected patch latents with rank 3, got {tuple(latents.shape)}")
        batch_size, num_patches, _ = latents.shape
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        logits = self.net(latents)
        return logits.reshape(batch_size, self.grid_size, self.grid_size, self.num_classes).permute(0, 3, 1, 2)
