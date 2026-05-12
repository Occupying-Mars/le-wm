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
  --device auto
```

the default config trains a lewm-style patch-latent model with:

- vit frame encoder
- causal action-conditioned latent dynamics applied per spatial patch
- ordered patch image decoder
- latent prediction loss
- predicted-frame reconstruction loss
- target/history reconstruction losses
- foreground-weighted reconstruction so snake/food/walls do not get averaged away
- `SIGReg` latent regularization

checkpoints are saved to:

```text
runs/snake_jepa/<run-name>/checkpoints/
```

## play / rollout from a checkpoint

```bash
uv run python -m snake_jepa.infer_snake_jepa \
  --run-name snake-jepa-full \
  --checkpoint best
```

controls:

- arrow keys or `w/a/s/d`: choose the next action
- `r`: reset to the same seed clip
- `n`: seed from a different real dataset clip

the ui starts from real context frames, then rolls forward entirely through the model decoder.
