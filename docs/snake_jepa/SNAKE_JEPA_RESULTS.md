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

- `snake_jepa/snake_data.py`: snake dataset discovery and sliding-window dataloader.
- `snake_jepa/snake_board.py`: picture-derived board extraction and deterministic board renderer.
- `snake_jepa/snake_world_model.py`: patch-latent lewm-style world model.
- `snake_jepa/train_snake_jepa.py`: training script for encoder + dynamics + decoder.
- `snake_jepa/train_snake_autoencoder.py`: decoder-first reconstruction check.
- `snake_jepa/infer_snake_jepa.py`: interactive model rollout/play UI with board or pixel decode modes.
- `docs/snake_jepa/TRAIN_SNAKE_JEPA.md`: commands.

model design:

- `TinyViTEncoder.encode_patches(frame)` gives one latent per spatial patch.
- dynamics predicts the next latent for each patch from that patch's history plus the action history.
- `OrderedPatchDecoder` decodes each ordered patch latent directly back into its image patch.
- `PatchBoardDecoder` decodes predicted latents into a `20x20` board from labels extracted from the png pixels, but this is diagnostic-only now.
- reconstruction loss is foreground-weighted so sparse snake/food pixels are not dominated by black background.
- board cross-entropy can be enabled separately from pixel reconstruction; board losses default to `0`.
- board accuracy, non-empty accuracy, snake-cell accuracy, and food-cell accuracy are logged to wandb.

## smoke verification

syntax:

```bash
uv run python -m py_compile snake_jepa/snake_board.py snake_jepa/snake_data.py snake_jepa/snake_world_model.py snake_jepa/train_snake_jepa.py snake_jepa/train_snake_autoencoder.py
```

completed smoke run:

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

produced:

```text
runs/snake_jepa/snake-jepa-smoke/checkpoints/best.pt
runs/snake_jepa/snake-jepa-smoke/previews/epoch_001.png
```

## training observations

global-latent experiments were not sufficient. the target reconstruction collapsed toward dark/blurry boards because the frame is mostly black. foreground-weighted loss improved the signal but a single cls/global latent still did not preserve exact board state.

the current committed direction is patch-latent reconstruction/dynamics, because it gives the decoder enough spatial information for precise board reconstruction while staying within the lewm encoder/dynamics/decoder pattern.

action-alignment fix: dataset generation records each frame after `env.step(action)`, so `frame_i.action` is the action that produced `frame_i`. next-frame training must use `clip.frames[start + 1 : hist_end + 1]` for the action sequence, not `clip.frames[start:hist_end]`.

latest observation: a true one-level/one-clip/one-window overfit with ordered patch decoding and `sigreg_weight=0` no longer collapses to pure black/noise, but it still is not exact enough for the win condition. it learns strong grid/border structure and sparse object colors, but not a faithful playable frame. the next blocker is high-fidelity decoding, not dataset discovery or metadata usage.

decoder-first check:

- `snake-ae-overfit-one-window`: at `128x128`, autoencoder mae reached `0.0099`, but max pixel error stayed high and preview still had object/grid errors.
- `snake-ae-overfit-one-window-320`: sharper target frames, but learned pixel reconstruction still spent capacity on fixed grid lines and was not exact.

board-decoder check:

- `snake-jepa-board-overfit-one-window`: one-window JEPA overfit at `320x320` drove `pred_board_loss` from `0.8781` to `0.0011` by epoch 120.
- pixel `pred_recon_loss` was still `0.2253`, so the current likely path is board decoding + deterministic rendering rather than relying only on raw pixel reconstruction.
- inference now defaults back to `--decode-mode pixel`; board decoding is diagnostic-only.
- label fix: the first board extractor over-counted cyan grid lines as snake cells. corrected extraction now classifies from the cell center area, with full-cell magenta detection for food. the pre-fix board run should be treated as invalid for final quality.

wandb:

- project: `https://wandb.ai/krishnapg2315/snake-jepa`
- tracked smoke run: `https://wandb.ai/krishnapg2315/snake-jepa/runs/snake-jepa-board-wandb-smoke`

## known gap

exact food respawn after eating is not explicitly present in metadata as rng state or food coordinates. the model can learn food position from pixels, but perfect deterministic respawn for arbitrary player actions may be underdetermined from image/action history alone unless the environment rng is encoded in history or extra state is added.
