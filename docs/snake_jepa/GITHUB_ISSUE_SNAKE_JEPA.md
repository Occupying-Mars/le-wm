# title

train snake jepa world model to playable rollout quality

# body

## objective

train a lewm-style JEPA model on the snake image dataset so a user can seed from real frames, choose actions, roll the model forward, and decode playable predicted frames. target behavior: snake motion, walls, food, eating, and food respawn are visually correct enough to play on the learned model.

## committed implementation

commits:

- `1752c9a docs: add codebase overview`
- `45f9b22 feat: add snake jepa training pipeline`
- `b690baf docs: add snake jepa issue draft`
- `58f721c fix: use ordered patch decoder for snake jepa`
- `d8380d6 chore: organize snake jepa files`

new files:

- `snake_jepa/snake_data.py`: discovers `/Users/krishna/Public/ml-experiments/snake-we/datasets/snake_agent`, loads clip metadata and png frames, produces history/next/action batches.
- `snake_jepa/snake_board.py`: extracts picture-derived board labels from png cells and renders predicted boards.
- `snake_jepa/snake_world_model.py`: lewm-style patch-latent world model.
- `snake_jepa/train_snake_jepa.py`: trains encoder + patch dynamics + decoder.
- `snake_jepa/train_snake_autoencoder.py`: decoder-first autoencoder check.
- `snake_jepa/infer_snake_jepa.py`: interactive playable rollout UI.
- `config/train/snake_jepa.json`: default training config.
- `docs/snake_jepa/TRAIN_SNAKE_JEPA.md`: run commands.
- `docs/snake_jepa/SNAKE_JEPA_RESULTS.md`: current results and caveats.

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
- decode the full predicted patch-latent grid with `OrderedPatchDecoder`
- decode the same predicted patch-latent grid into a `20x20` board with `PatchBoardDecoder`
- foreground-weight reconstruction loss so snake/food/walls matter more than black background
- track wandb metrics/previews under `krishnapg2315/snake-jepa`

this keeps the implementation inside the current lewm-style encoder/dynamics/decoder code path.

## verified

syntax:

```bash
uv run python -m py_compile snake_jepa/snake_board.py snake_jepa/snake_data.py snake_jepa/snake_world_model.py snake_jepa/train_snake_jepa.py snake_jepa/train_snake_autoencoder.py
```

patch-model smoke:

```bash
uv run python -m snake_jepa.train_snake_jepa \
  --run-name snake-jepa-patch-smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --max-clips-per-level 1 \
  --max-windows-per-clip 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

latest board-smoke output:

```text
device: cpu
train samples: 1
val samples: 1
parameters: 3,177,028
run dir: runs/snake_jepa/snake-jepa-board-smoke-v2
epoch 001 | train 3.9468 | val 3.6924 | pred_recon 0.8432 | pred_board 0.8781 | latent 0.8025
```

wandb smoke:

```text
https://wandb.ai/krishnapg2315/snake-jepa/runs/snake-jepa-board-wandb-smoke
```

one-window overfit:

- pixel autoencoder still was not exact enough at `128x128` or `320x320`.
- board decoder reached `pred_board_loss=0.0011` by epoch 120 on `snake-jepa-board-overfit-one-window`.

## next steps

1. route interactive inference through predicted board rendering instead of raw pixel decoder.
2. train a wandb-tracked subset across multiple clips and inspect predicted-board previews.
3. add board accuracy metrics, especially snake/food cells, not just cross-entropy.
4. evaluate whether food respawn is learnable from image/action history alone.

## caveat

metadata does not expose rng state or explicit food coordinates. food location is visible in pixels, but exact future food respawn after eating may be underdetermined from image/action history alone unless the model learns the environment's hidden spawn process or extra state is added.

## blocked live issue creation

repo issue creation was attempted, but `Occupying-Mars/le-wm` has issues disabled. if issues are enabled later, create it with:

```bash
gh issue create \
  --title "train snake jepa world model to playable rollout quality" \
  --body-file docs/snake_jepa/GITHUB_ISSUE_SNAKE_JEPA.md
```
