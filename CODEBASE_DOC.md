# codebase doc

this repo contains two related tracks:

- `lewm` proper: the paper-style JEPA world model trained on `stable_worldmodel` hdf5 datasets and evaluated with model-predictive control.
- `debug_box`: local diagnostic experiments for a simple moving-box dataset, with custom vit encoders, latent dynamics, decoders, probing, and matplotlib inference tools.

## setup notes

repo instructions say to activate the venv and use `uv` for actions:

```bash
source .venv/bin/activate
uv run python train_debug_box_vit.py
```

the checked-in docs/configs have a version mismatch:

- `README.md` says `uv venv --python=3.10` and installs `stable-worldmodel[train,env]`.
- `pyproject.toml` says `requires-python = ">=3.12"` and only lists a small dependency subset.
- several scripts also need packages not listed in `pyproject.toml`, including `hydra`, `lightning`, `stable_pretraining`, `stable_worldmodel`, `torchvision`, `sklearn`, `einops`, and `matplotlib`.

## file map

### main lewm path

- `jepa.py`: wraps encoder, action encoder, autoregressive predictor, optional projection heads, rollout, and cost computation.
- `module.py`: transformer building blocks, `SIGReg`, action `Embedder`, projection `MLP`, and `ARPredictor`.
- `train.py`: hydra training entrypoint for lewm on `stable_worldmodel` hdf5 datasets.
- `eval.py`: hydra evaluation/planning entrypoint using `stable_worldmodel.World`, `AutoCostModel`, and either CEM or gradient solvers.
- `utils.py`: image preprocessing, column normalization, and object-checkpoint callback.
- `config/train/lewm.yaml`: base training config.
- `config/train/data/*.yaml`: dataset configs for pusht, tworoom, dmc/reacher, and ogbench cube.
- `config/eval/*.yaml`: evaluation configs for pusht, tworoom, cube, and reacher.
- `config/eval/solver/*.yaml`: CEM and Adam/gradient planner configs.

### debug-box path

- `debug_box_data.py`: discovers clips under `dataset_root/level_1/clip_*`, loads `clip_metadata.json`, normalizes box `(x, y)`, builds sliding history windows.
- `debug_box_world_model.py`: custom tiny vit encoder, causal latent dynamics transformer, patch-query decoder, and `DebugBoxWorldModel`.
- `train_debug_box_vit.py`: trains the debug-box encoder+dynamics with latent MSE plus `SIGReg`.
- `train_debug_box_autoencoder.py`: trains a cls-latent autoencoder for debug-box frames.
- `train_debug_box_decoder.py`: trains a separate decoder on frozen world-model pre-projection cls latents.
- `train_debug_box_frozen_ae_predictor.py`: trains dynamics over frozen autoencoder latents.
- `probe_debug_box_latents.py`: fits simple linear probes from latents to normalized box position.
- `infer_debug_box.py`: interactive world-model / decoder visualization.
- `infer_debug_box_autoencoder.py`: interactive or grid autoencoder reconstruction visualization.
- `infer_debug_box_frozen_ae_predictor.py`: interactive frozen-autoencoder predictor visualization.

`main.py` is only a uv-generated hello-world stub.

## lewm training flow

`train.py` does the following:

1. loads an hdf5 dataset via `stable_worldmodel.data.HDF5Dataset`.
2. preprocesses pixels to imagenet-normalized tensors and normalizes non-pixel columns.
3. creates a huggingface-style vit backbone through `stable_pretraining.backbone.utils.vit_hf`.
4. builds:
   - vit encoder
   - action `Embedder`
   - `ARPredictor`
   - encoder projection `MLP`
   - predictor projection `MLP`
   - `JEPA`
5. wraps the model in `stable_pretraining.Module`.
6. trains with:
   - next-embedding prediction loss
   - gaussian latent regularization via `SIGReg`
7. saves object checkpoints through `ModelObjectCallBack`.

core loss in `lejepa_forward`:

```python
loss = pred_loss + cfg.loss.sigreg.weight * sigreg_loss
```

training command:

```bash
uv run python train.py data=pusht
```

## lewm evaluation flow

`eval.py`:

1. creates a `stable_worldmodel.World`.
2. builds image transforms for current and goal pixels.
3. computes standard scalers for cached non-pixel columns.
4. loads a policy:
   - `random`, or
   - `stable_worldmodel.policy.AutoCostModel(cfg.policy)`
5. instantiates a planner from `config/eval/solver`.
6. evaluates from sampled dataset start states and goal offsets.
7. appends config and metrics to a result text file.

example:

```bash
uv run python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

the policy path is relative to `$STABLEWM_HOME` and should omit `_object.ckpt`.

## debug-box data format

expected dataset root:

```text
debug_box/
  level_1/
    clip_*/
      clip_metadata.json
      frames/
        ...
```

`clip_metadata.json` must include a `frames` list. each frame entry is expected to include:

- `image`
- `action`
- `box_x`
- `box_y`

the dataset returns:

- `history_frames`: `(history_size, 3, image_size, image_size)`
- `next_frame`: `(3, image_size, image_size)`
- `actions`: integer action ids
- `actions_one_hot`: one-hot over 4 actions
- `history_states`: normalized `(x, y)`
- `next_state`: normalized `(x, y)`

clip splitting is deterministic by sorted clip order, not randomized.

## debug-box model

`DebugBoxWorldModel` combines:

- `TinyViTEncoder`: patch conv, cls token, learned position embedding, transformer blocks, cls projection.
- `LatentDynamics`: causal transformer over `[latent, action_one_hot]` history.
- `LatentDecoder`: query-token cross-attention decoder from latent to image patches.
- `state_head`: latent to normalized `(x, y)`, currently not used by the main debug-box training losses.

default config:

- image size: `128`
- patch size: `16`
- history size: `4`
- action dim: `4`
- encoder dim: `128`
- latent dim: `64`
- dynamics dim: `128`

## debug-box training modes

### end-to-end-ish vit predictor

entrypoint:

```bash
uv run python train_debug_box_vit.py --config config/train/debug_box_vit_mps.json
```

important behavior:

- trains `encoder` and `dynamics`.
- calls `model.freeze_decoder_modules()`, so `decoder` and `state_head` are frozen.
- loss is latent prediction MSE plus `SIGReg`.
- target latent is detached.
- previews currently show history plus repeated target frames, not decoded predictions.

### cls autoencoder

entrypoint:

```bash
uv run python train_debug_box_autoencoder.py --run-name box-autoenc-cls-1
```

behavior:

- trains `TinyViTEncoder.encode_cls(frame)` plus `LatentDecoder`.
- reconstructs `next_frame`.
- loss is `l1 + mse`.
- logs diagnostics for latent spread, zero-latent decode delta, and center error.

### separate decoder for world-model cls latents

entrypoint:

```bash
uv run python train_debug_box_decoder.py --config config/train/debug_box_decoder.json
```

behavior:

- loads frozen world model from `wm_output_dir/wm_run_name/checkpoints/best.pt`.
- encodes ground-truth next frames using `world_model.encode_next_cls`.
- trains a decoder from pre-projection cls latents to frames.
- checkpoint marks `decoder_input_mode = "vit_preproj_cls_v1"`.

sharp edge: this decoder expects `encoder_dim` cls latents, while `DebugBoxWorldModel.dynamics` predicts `latent_dim` projected latents. it is suitable for decoder-only cls reconstruction unless an adapter/predictor is added.

### frozen-autoencoder latent predictor

entrypoint:

```bash
uv run python train_debug_box_frozen_ae_predictor.py --ae-run-name box-autoenc-cls-1
```

behavior:

- loads frozen autoencoder.
- trains `LatentDynamics` over autoencoder cls latents.
- decodes predicted latent through the frozen autoencoder decoder.
- default total loss is latent MSE only because `decoded_loss_weight = 0.0`.

## debug-box inference tools

world-model ui:

```bash
uv run python infer_debug_box.py \
  --wm-run-name first-box-lewm-1 \
  --decoder-run-name box-decoder-cls-fix-collapse-1
```

decoder-only grid:

```bash
uv run python infer_debug_box.py \
  --wm-run-name first-box-lewm-1 \
  --decoder-run-name box-decoder-cls-fix-collapse-1 \
  --decoder-only
```

autoencoder ui:

```bash
uv run python infer_debug_box_autoencoder.py --ae-run-name box-autoenc-cls-1
```

frozen-autoencoder predictor ui:

```bash
uv run python infer_debug_box_frozen_ae_predictor.py \
  --ae-run-name box-autoenc-cls-1 \
  --predictor-run-name frozen-ae-predictor-1
```

notable issue in `infer_debug_box.py`: the interactive world-model path loads the external decoder, but `InferenceUI.predict()` decodes predicted latents with `self.world_model.decoder`, not `self.decoder`. since the world-model decoder is frozen/untrained in `train_debug_box_vit.py`, this likely makes full prediction visuals misleading. decoder-only mode does use the external decoder.

## existing local runs

run configs already present:

- `runs/debug_box_vit_mps/first-box-lewm-1/train_config.json`
  - device resolved to `mps`
  - params: `1,191,296`
  - run name: `first-box-lewm-1`
- `runs/debug_box_autoencoder/box-autoenc-cls-1/train_config.json`
  - device resolved to `mps`
  - params: `1,834,880`
- `runs/debug_box_decoder/box-decoder-cls-fix-collapse-1/train_config.json`
  - world model: `first-box-lewm-1`
  - params: `909,952`
- `runs/debug_box_frozen_ae_predictor/frozen-ae-predictor-1/train_config.json`
  - autoencoder: `box-autoenc-cls-1`
  - predictor params: `299,264`

## checkpoint conventions

debug-box scripts save under:

```text
runs/<experiment_family>/<run_name>/checkpoints/
  epoch_###.pt
  latest.pt
  best.pt
```

training scripts generally resume from `latest.pt` and prefer `best.pt` for loading dependency models.

lewm proper uses `$STABLEWM_HOME` / `stable_worldmodel` cache paths and object checkpoints named like:

```text
<name>_object.ckpt
<name>_weights.ckpt
```

## gotchas

- `infer_debug_box.py` full prediction ui probably ignores the trained external decoder.
- `train_debug_box_vit.py` previews do not visualize model predictions.
- debug-box full model contains a decoder and state head, but current vit training freezes both.
- `state_head` is not supervised in the debug-box scripts read here.
- `pyproject.toml` does not fully describe the dependencies used by the repo.
- README says python 3.10, while `pyproject.toml` says python >=3.12.
- several config defaults contain machine-local absolute paths under `/Users/krishna/Public/ml-experiments/...`.

