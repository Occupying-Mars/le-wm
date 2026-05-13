# snake board dynamics status

## current best run

run:

```bash
uv run python -m snake_jepa.train_snake_board_dynamics \
  --run-name snake-board-dyn-hard-rollout10-level1-500clips-effective-actions \
  --device auto --epochs 10 --batch-size 64 --rollout-steps 10 \
  --rollout-feedback hard --levels level_1 --max-clips-per-level 500 \
  --max-windows-per-clip 20 --max-train-batches 120 --max-val-batches 40 \
  --preview-every 5 --checkpoint-every 5 --lr 0.00025 \
  --class-weights 1.0 2.0 8.0 12.0 --wandb
```

wandb:

```text
https://wandb.ai/krishnapg2315/snake-jepa/runs/snake-board-dyn-hard-rollout10-level1-500clips-effective-actions
```

## fixes made

- dataset actions now use the effective action when the environment reports `ignored_action=true`.
- rollout eval prints progress instead of going silent.
- training exposes `--history-size`.
- inference/eval can use `--legalize-snake` to decode legal snake body motion.
- legalized rollouts now keep a legal food cell after eating, even when the model fails to emit one.
- `snake_jepa.save_snake_board_rollout_gifs` saves target-vs-model rollout GIFs.

## results

raw 500-clip, 20-step rollout:

```json
{
  "board_acc": 0.9970745027780533,
  "exact_board": 0.5124,
  "food_acc": 0.7776,
  "nonempty_acc": 0.718636445169951,
  "snake_acc": 0.6736810007344189
}
```

corrected-action retrain, raw 100-clip, 20-step rollout:

```json
{
  "board_acc": 0.9974100024700164,
  "exact_board": 0.5415,
  "food_acc": 0.798,
  "nonempty_acc": 0.7240128096674142,
  "snake_acc": 0.6689013041153522
}
```

corrected-action retrain, legalized 500-clip, 20-step rollout:

```json
{
  "board_acc": 0.9993362506330014,
  "exact_board": 0.7733,
  "food_acc": 0.7753,
  "nonempty_acc": 0.8906817837587833,
  "snake_acc": 0.9786520291661737
}
```

## diagnosis

the raw model mainly fails by accumulating snake-body errors. the first inspected failure was a stale tail cell: predicted snake where the target was empty. a constrained decoder fixes most of that, pushing snake accuracy to about `0.979` over 500 clips.

remaining exact-board failures are dominated by food. on a 100-clip diagnostic run after legalization, food states broke down as:

```text
food_exact: 1612
pred_no_food: 381
food_different: 7
```

the legal food fallback fixes no-food playable states, but it cannot reproduce the dataset's hidden/random recorded spawn location from board pixels alone.

## useful commands

500-clip legalized audit:

```bash
uv run python -m snake_jepa.eval_snake_board_rollout \
  --run-name snake-board-dyn-hard-rollout10-level1-500clips-effective-actions \
  --checkpoint best --levels level_1 --max-clips 500 --steps 20 \
  --progress-every 25 --legalize-snake \
  --json-out runs/snake_board_dynamics/snake-board-dyn-hard-rollout10-level1-500clips-effective-actions/rollouts/eval_level1_500clips_20step_legalized.json
```

comparison GIFs:

```bash
uv run python -m snake_jepa.save_snake_board_rollout_gifs \
  --run-name snake-board-dyn-hard-rollout10-level1-500clips-effective-actions \
  --checkpoint best --levels level_1 --max-clips 20 --gif-count 4 \
  --steps 20 --legalize-snake
```

play the model with legal decoding:

```bash
uv run python -m snake_jepa.infer_snake_board_dynamics \
  --run-name snake-board-dyn-hard-rollout10-level1-500clips-effective-actions \
  --checkpoint best --legalize-snake
```

## not done

this is not goal-complete. the model plus legal decoder is playable enough to keep snake motion coherent and keep food present, but it is not perfect against recorded validation clips because exact food spawn location is not fully determined by the visible board history.
