# snake jepa training

the snake image dataset is here:

```bash
/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent
```

it has `level_1`, `level_2`, `level_3`, and `random_levels` clips with `frames/*.png` plus `clip_metadata.json`.

## smoke test

```bash
uv run python -m snake_jepa.train_snake_jepa \
  --run-name snake-jepa-smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --max-clips-per-level 1 \
  --max-windows-per-clip 2 \
  --max-train-batches 1 \
  --max-val-batches 1
```

this should create:

```text
runs/snake_jepa/snake-jepa-smoke/checkpoints/best.pt
runs/snake_jepa/snake-jepa-smoke/previews/epoch_001.png
```

## real training

```bash
uv run python -m snake_jepa.train_snake_jepa \
  --run-name snake-jepa-full \
  --device auto \
  --wandb
```

the default config trains a lewm-style patch-latent model with:

- `320x320` nearest-neighbor resized pixel-art frames
- vit frame encoder
- causal action-conditioned latent dynamics applied per spatial patch
- ordered patch image decoder
- picture-derived `20x20` board decoder for exact game-state decoding
- latent prediction loss
- predicted-frame reconstruction loss
- predicted-board cross-entropy loss
- target/history reconstruction losses
- foreground-weighted reconstruction so snake/food/walls do not get averaged away
- `SIGReg` latent regularization

checkpoints are saved to:

```text
runs/snake_jepa/<run-name>/checkpoints/
```

## decoder-first check

before spending time on dynamics, prove the snake image encoder/decoder can reconstruct real frames:

```bash
uv run python -m snake_jepa.train_snake_autoencoder \
  --run-name snake-ae-overfit \
  --device auto \
  --epochs 200 \
  --image-size 320 \
  --levels level_1 \
  --max-clips-per-level 1 \
  --max-windows-per-clip 1 \
  --batch-size 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

this writes only under:

```text
runs/snake_jepa_autoencoder/<run-name>/
```

the preview rows are target, reconstruction, and absolute pixel diff.

## play / rollout from a checkpoint

```bash
uv run python -m snake_jepa.infer_snake_jepa \
  --run-name snake-jepa-full \
  --checkpoint best \
  --decode-mode board
```

controls:

- arrow keys or `w/a/s/d`: choose the next action
- `r`: reset to the same seed clip
- `n`: seed from a different real dataset clip

the ui starts from real context frames, then rolls forward through learned latent dynamics. `--decode-mode board` uses the learned board decoder plus deterministic rendering; `--decode-mode pixel` uses the raw image decoder.
