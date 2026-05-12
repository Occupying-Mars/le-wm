# snake jepa approach and current results

## objective

train a lewm-style JEPA world model on the snake image dataset so a user can seed from real frames, choose actions, roll the learned model forward, and decode playable predicted frames.

## dataset

path:

```text
/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent
```

structure:

```text
level_1/clip_*/clip_metadata.json
level_1/clip_*/frames/frame_*.png
level_2/...
level_3/...
random_levels/random_clip_*/...
```

observed counts:

- 2502 metadata clips
- 327327 png frames
- four action ids: `0=up`, `1=right`, `2=down`, `3=left`

## implementation

new repo files:

- `snake_data.py`: snake dataset discovery and sliding-window dataloader.
- `snake_world_model.py`: patch-latent lewm-style world model.
- `train_snake_jepa.py`: training script for encoder + dynamics + decoder.
- `infer_snake_jepa.py`: interactive model rollout/play UI.
- `TRAIN_SNAKE_JEPA.md`: commands.

model design:

- `TinyViTEncoder.encode_patches(frame)` gives one latent per spatial patch.
- dynamics predicts the next latent for each patch from that patch's history plus the action history.
- `LatentDecoder` decodes the full set of predicted patch latents into the next RGB frame.
- reconstruction loss is foreground-weighted so sparse snake/food pixels are not dominated by black background.

## smoke verification

syntax:

```bash
uv run python -m py_compile snake_data.py train_snake_jepa.py infer_snake_jepa.py snake_world_model.py
```

completed smoke run:

```bash
uv run python train_snake_jepa.py \
  --run-name snake-jepa-smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --max-clips-per-level 1 \
  --max-windows-per-clip 2 \
  --max-train-batches 1 \
  --max-val-batches 1
```

produced:

```text
runs/snake_jepa/snake-jepa-smoke/checkpoints/best.pt
runs/snake_jepa/snake-jepa-smoke/previews/epoch_001.png
```

## training observations

global-latent experiments were not sufficient. the target reconstruction collapsed toward dark/blurry boards because the frame is mostly black. foreground-weighted loss improved the signal but a single cls/global latent still did not preserve exact board state.

the current committed direction is patch-latent reconstruction/dynamics, because it gives the decoder enough spatial information for precise board reconstruction while staying within the lewm encoder/dynamics/decoder pattern.

## known gap

exact food respawn after eating is not explicitly present in metadata as rng state or food coordinates. the model can learn food position from pixels, but perfect deterministic respawn for arbitrary player actions may be underdetermined from image/action history alone unless the environment rng is encoded in history or extra state is added.

