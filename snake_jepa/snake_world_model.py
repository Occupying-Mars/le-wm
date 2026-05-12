from dataclasses import dataclass

import torch
from torch import nn

from debug_box_world_model import TinyViTEncoder, TransformerBlock
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
    pixel_dynamics: bool = False
    pixel_dynamics_hidden: int = 64
    pixel_dynamics_depth: int = 6


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
        self.dynamics = SpatialLatentDynamics(
            history_size=cfg.history_size,
            num_patches=self.num_patches,
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
            depth=cfg.decoder_depth,
            heads=cfg.decoder_heads,
            mlp_ratio=cfg.decoder_mlp_ratio,
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
        self.pixel_dynamics = (
            PixelDynamics(
                history_size=cfg.history_size,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.pixel_dynamics_hidden,
                depth=cfg.pixel_dynamics_depth,
            )
            if cfg.pixel_dynamics
            else None
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
        next_actions_one_hot = actions_one_hot[:, -1]
        return self.dynamics(history_latents, next_actions_one_hot)

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
        if self.pixel_dynamics is not None:
            pred_next_frame = self.pixel_dynamics(history_frames, actions_one_hot[:, -1])
        else:
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


class SpatialLatentDynamics(nn.Module):
    def __init__(
        self,
        *,
        history_size: int,
        num_patches: int,
        latent_dim: int,
        action_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_size = int(history_size)
        self.num_patches = int(num_patches)
        self.time_embed = nn.Parameter(torch.zeros(1, self.history_size, 1, dim))
        self.patch_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, dim))
        self.in_proj = nn.Linear(latent_dim + action_dim, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed, std=0.02)

    def forward(self, history_latents: torch.Tensor, next_actions_one_hot: torch.Tensor) -> torch.Tensor:
        if history_latents.dim() != 4:
            raise ValueError(f"expected history latents with rank 4, got {tuple(history_latents.shape)}")
        batch_size, history_size, num_patches, _ = history_latents.shape
        if next_actions_one_hot.dim() != 2:
            raise ValueError(f"expected next actions with rank 2, got {tuple(next_actions_one_hot.shape)}")
        if history_size != self.history_size:
            raise ValueError(f"expected history size {self.history_size}, got {history_size}")
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        actions = next_actions_one_hot[:, None, None, :].expand(batch_size, history_size, num_patches, -1)
        x = torch.cat([history_latents, actions], dim=-1)
        x = self.in_proj(x) + self.time_embed + self.patch_embed
        x = x.reshape(batch_size, history_size * num_patches, -1)
        for block in self.blocks:
            x = block(x, causal=False)
        x = x.reshape(batch_size, history_size, num_patches, -1)
        return self.out_proj(self.norm(x[:, -1]))


class PixelResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PixelDynamics(nn.Module):
    def __init__(
        self,
        *,
        history_size: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        in_channels = int(history_size) * 3 + int(action_dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[PixelResidualBlock(hidden_dim) for _ in range(int(depth))])
        self.out_proj = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)

    def forward(self, history_frames: torch.Tensor, next_actions_one_hot: torch.Tensor) -> torch.Tensor:
        if history_frames.dim() != 5:
            raise ValueError(f"expected history frames with rank 5, got {tuple(history_frames.shape)}")
        batch_size, history_size, channels, height, width = history_frames.shape
        if channels != 3:
            raise ValueError(f"expected RGB history frames, got {channels} channels")
        actions = next_actions_one_hot[:, :, None, None].expand(batch_size, next_actions_one_hot.size(1), height, width)
        x = torch.cat([history_frames.reshape(batch_size, history_size * channels, height, width), actions], dim=1)
        delta = self.out_proj(self.blocks(self.in_proj(x)))
        last_frame = history_frames[:, -1]
        return (last_frame + delta).clamp(0.0, 1.0)


class OrderedPatchDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
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
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.patch_dim)
        refine_dim = min(64, hidden_dim)
        self.pixel_refiner = nn.Sequential(
            nn.Conv2d(3, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, 3, kernel_size=3, padding=1),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.pixel_refiner[-1].weight)
        nn.init.zeros_(self.pixel_refiner[-1].bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"expected patch latents with rank 3, got {tuple(latents.shape)}")
        batch_size, num_patches, _ = latents.shape
        if num_patches != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patches, got {num_patches}")
        patches = self.in_proj(latents) + self.pos_embed
        for block in self.blocks:
            patches = block(patches, causal=False)
        patches = self.out_proj(self.norm(patches))
        patches = patches.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            3,
        )
        image = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            3,
            self.image_size,
            self.image_size,
        )
        image = image + self.pixel_refiner(image)
        return image.sigmoid()


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
