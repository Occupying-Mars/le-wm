from dataclasses import dataclass

import torch
from torch import nn


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, causal: bool) -> torch.Tensor:
        attn_mask = _causal_mask(x.size(1), x.device) if causal else None
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class TinyViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, latent_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(frames)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed[:, : x.size(1)])
        for block in self.blocks:
            x = block(x, causal=False)
        return self.norm(x)

    def encode_patches(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._encode_tokens(frames)
        return self.proj(x[:, 1:])

    def encode_cls(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._encode_tokens(frames)
        return x[:, 0]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encode_cls(frames))


class LatentDynamics(nn.Module):
    def __init__(
        self,
        *,
        history_size: int,
        latent_dim: int,
        action_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, history_size, dim))
        self.in_proj = nn.Linear(latent_dim + action_dim, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latents: torch.Tensor, actions_one_hot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latents, actions_one_hot], dim=-1)
        x = self.in_proj(x) + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x, causal=True)
        x = self.norm(x[:, -1])
        return self.out_proj(x)


class LatentDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        image_size: int,
        patch_size: int = 16,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size * 3
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_patch = nn.Linear(hidden_dim, self.patch_dim)

        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 2:
            batch_size = latents.size(0)
            context = self.latent_proj(latents).unsqueeze(1)
        elif latents.dim() == 3:
            batch_size = latents.size(0)
            if latents.size(1) != self.num_patches:
                raise ValueError(
                    f"expected {self.num_patches} patch latents, got {latents.size(1)}"
                )
            context = self.latent_proj(latents)
        else:
            raise ValueError(f"expected latents with rank 2 or 3, got shape {tuple(latents.shape)}")
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, context)

        patches = self.to_patch(self.norm(queries))
        patches = patches.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            3,
        )
        images = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, 3, self.image_size, self.image_size
        )
        return images.sigmoid()


@dataclass
class DebugBoxWorldModelConfig:
    image_size: int = 128
    patch_size: int = 16
    history_size: int = 4
    action_dim: int = 4
    encoder_dim: int = 128
    encoder_depth: int = 4
    encoder_heads: int = 4
    encoder_mlp_ratio: float = 4.0
    latent_dim: int = 64
    dynamics_dim: int = 128
    dynamics_depth: int = 2
    dynamics_heads: int = 4
    dynamics_mlp_ratio: float = 2.0
    decoder_dim: int = 128
    decoder_depth: int = 4
    decoder_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    dropout: float = 0.0


class DebugBoxWorldModel(nn.Module):
    def __init__(self, cfg: DebugBoxWorldModelConfig) -> None:
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
        self.decoder = LatentDecoder(
            latent_dim=cfg.latent_dim,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.decoder_dim,
            depth=cfg.decoder_depth,
            heads=cfg.decoder_heads,
            mlp_ratio=cfg.decoder_mlp_ratio,
            dropout=cfg.dropout,
        )
        self.state_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 2),
            nn.Sigmoid(),
        )

    def freeze_decoder_modules(self) -> None:
        for module in [self.decoder, self.state_head]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        batch_size, history_size = history_frames.shape[:2]
        flat_frames = history_frames.view(batch_size * history_size, *history_frames.shape[2:])
        flat_latents = self.encoder(flat_frames)
        return flat_latents.view(batch_size, history_size, -1)

    def encode_next(self, next_frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(next_frame)

    def encode_next_tokens(self, next_frame: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_patches(next_frame)

    def encode_next_cls(self, next_frame: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_cls(next_frame)

    def predict_next(self, history_frames: torch.Tensor, actions_one_hot: torch.Tensor) -> dict[str, torch.Tensor]:
        history_latents = self.encode_history(history_frames)
        pred_next_latent = self.dynamics(history_latents, actions_one_hot)
        return {
            "history_latents": history_latents,
            "pred_next_latent": pred_next_latent,
        }

    def decode_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, history_size, dim = latents.shape
        flat = latents.view(batch_size * history_size, dim)
        decoded = self.decoder(flat)
        return decoded.view(batch_size, history_size, *decoded.shape[1:])

    def forward(
        self,
        history_frames: torch.Tensor,
        actions_one_hot: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_latents = self.encode_history(history_frames)
        target_next_latent = self.encode_next(next_frame)
        pred_next_latent = self.dynamics(history_latents, actions_one_hot)

        history_recon = self.decode_sequence(history_latents)
        pred_next_frame = self.decoder(pred_next_latent)
        target_next_recon = self.decoder(target_next_latent)
        pred_state = self.state_head(pred_next_latent)

        return {
            "history_latents": history_latents,
            "target_next_latent": target_next_latent,
            "pred_next_latent": pred_next_latent,
            "history_recon": history_recon,
            "pred_next_frame": pred_next_frame,
            "target_next_recon": target_next_recon,
            "pred_state": pred_state,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
