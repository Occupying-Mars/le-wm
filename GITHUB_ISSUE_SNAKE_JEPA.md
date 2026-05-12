# title

train snake jepa world model to playable rollout quality

# body

## objective

train a lewm-style JEPA model on the snake image dataset so a user can seed from real frames, choose actions, roll the model forward, and decode playable predicted frames. target behavior: snake motion, walls, food, eating, and food respawn are visually correct enough to play on the learned model.

## committed implementation

commits:

- `1752c9a docs: add codebase overview`
- `45f9b22 feat: add snake jepa training pipeline`

new files:

- `snake_data.py`: discovers `/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent`, loads clip metadata and png frames, produces history/next/action batches.
- `snake_world_model.py`: lewm-style patch-latent world model.
- `train_snake_jepa.py`: trains encoder + patch dynamics + decoder.
- `infer_snake_jepa.py`: interactive playable rollout UI.
- `config/train/snake_jepa.json`: default training config.
- `TRAIN_SNAKE_JEPA.md`: run commands.
- `SNAKE_JEPA_RESULTS.md`: current results and caveats.

## dataset

path:

```text
/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent
```

observed:

- 2502 clips
- 327327 png frames
- levels: `level_1`, `level_2`, `level_3`, `random_levels`
- actions: `0=up`, `1=right`, `2=down`, `3=left`

## approach

the first global-latent approach was not enough: because the board is mostly black, reconstruction loss could improve while predicted/decoded frames collapsed to dark blurry boards.

current direction:

- use `TinyViTEncoder.encode_patches` instead of only cls/global latent
- predict next latent independently per spatial patch with action-conditioned causal `LatentDynamics`
- decode the full predicted patch-latent grid with `LatentDecoder`
- foreground-weight reconstruction loss so snake/food/walls matter more than black background

this keeps the implementation inside the current lewm-style encoder/dynamics/decoder code path.

## verified

syntax:

```bash
uv run python -m py_compile snake_data.py snake_world_model.py train_snake_jepa.py infer_snake_jepa.py
```

patch-model smoke:

```bash
uv run python train_snake_jepa.py \
  --run-name snake-jepa-patch-smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --max-clips-per-level 1 \
  --max-windows-per-clip 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

output:

```text
device: cpu
train samples: 3
val samples: 1
parameters: 4,849,344
run dir: runs/snake_jepa/snake-jepa-patch-smoke
epoch 001 | train 4.2607 | val 4.0884 | pred_recon 1.1235 | latent 0.8942
```

load/decode smoke produced:

```text
pred_latent_shape (1, 64, 96)
img_shape (1, 3, 128, 128)
```

## next steps

1. run a longer patch-latent subset job and inspect previews.
2. if reconstruction is still blurry, pretrain the patch autoencoder before dynamics training.
3. only then scale to all clips.
4. evaluate whether food respawn is learnable from image/action history alone.

## caveat

metadata does not expose rng state or explicit food coordinates. food location is visible in pixels, but exact future food respawn after eating may be underdetermined from image/action history alone unless the model learns the environment's hidden spawn process or extra state is added.

## blocked live issue creation

`gh auth status` reports the active github token for `Occupying-Mars` is invalid. after re-auth, create the issue with:

```bash
gh issue create \
  --title "train snake jepa world model to playable rollout quality" \
  --body-file GITHUB_ISSUE_SNAKE_JEPA.md
```

