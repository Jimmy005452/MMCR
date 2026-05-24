# RL MMCR SAC Actor-Critic

This package implements a continuous value-based actor-critic baseline using SAC.
It reuses the maintained PPO-GAE environment and model-merging code, but replaces
PPO with off-policy SAC over continuous coefficient vectors.

The actor is a positive softplus-normal policy, so each task coefficient is sampled
independently and constrained only to be non-negative:

```text
coefficients >= 0
sum(coefficients) is not constrained
```

There is no binary gate and no per-layer sum-to-one normalization in this method.

Notes:

- `--random-steps` counts environment steps, not episodes. In layer-wise mode,
  one episode contains one step per merged layer.
- `--episode-reward-only` uses only the terminal objective as reward. If you
  want interval rewards, omit that flag and set `--reward-eval-interval`.
- The default SAC entropy temperature is intentionally small (`--alpha 0.02`)
  because the positive coefficient space is larger than the old simplex space.
- The default `--coefficient-mode positive` leaves coefficients unnormalized
  during merging. The SAC CLI no longer exposes simplex softmax weights.
- The default `--coefficient-init 0.3` initializes the deterministic actor
  near task-arithmetic-style scales instead of starting from 1.0 for every task.

## Layer-wise SAC

```bash
python -m rl_methods.rl_mmcr_SAC_Actor_Critic.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --source-baseline-json results/source_baselines_8datasets_test.json \
  --output-dir rl_mmcr_SAC_Actor_Critic_runs/layer_interval4_seed2026 \
  --merge-granularity layer \
  --episodes 300 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --reward-eval-interval 4 \
  --batch-size 128 \
  --random-steps 200 \
  --updates-per-step 1 \
  --lr 3e-4 \
  --critic-lr 3e-4 \
  --log-every 20 \
  --seed 2026 \
  --gpu 0 \
  --amp
```

## Global SAC

This uses one coefficient vector shared by all layers. It is a cheaper sanity
check for whether the reward can guide coefficient search at all.

```bash
python -m rl_methods.rl_mmcr_SAC_Actor_Critic.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_mmcr_SAC_Actor_Critic_runs/global_seed2027 \
  --merge-granularity global \
  --episodes 500 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --batch-size 128 \
  --random-steps 100 \
  --updates-per-step 4 \
  --lr 3e-4 \
  --critic-lr 3e-4 \
  --log-every 25 \
  --seed 2027 \
  --gpu 1 \
  --amp
```

Outputs are saved under `--output-dir`:

- `encoder.pt`
- `results.json`
- `training_curves.png`
- `reward_curves.png`

To avoid recomputing source-model retention denominators at startup, first run
`python -m rl_methods.source_baselines ...` from the parent README and pass the
result with `--source-baseline-json`.

## State Mode

Use `--state-mode full_coefficients` when testing whether SAC can learn a stable Q function from the current partial merge. Compared with the default `minimal` state, this exposes the full layer-by-task coefficient table plus a mask indicating which layers have already been filled.

For 8 datasets and 27 merge layers:

```text
minimal state: 25 dims
full_coefficients state: 260 dims
```

Recommended SAC diagnostic setting:

```bash
--state-mode full_coefficients \
--activation-reward-coef 0.0
```

## Conservative Actor Updates

SAC can overestimate Q and let the actor exploit unreliable critic regions. These options make the actor update more conservative while still training the critics from replay:

```bash
--freeze-actor-during-random-steps \
--actor-update-delay 4 \
--action-anchor-coef 0.05 \
--cql-coef 0.1
```

- `--freeze-actor-during-random-steps`: random warmup fills replay and updates critics, but the actor is not changed until `--random-steps` is exhausted.
- `--actor-update-delay`: updates the actor once every N critic updates.
- `--action-anchor-coef`: penalizes actor-sampled coefficients moving far from `--coefficient-init`.
- `--cql-coef`: adds a conservative Q penalty when actor-sampled actions have higher Q than replay actions.

The log now prints `q`, `tq` (target Q), `vloss` (critic loss including CQL), `cql` (conservative penalty), `bell` (Bellman loss), `ploss` (actor loss), and `au` (actor updates in that episode). These same values are saved in `results.json` as `q_mean`, `target_q_mean`, `value_loss`, `cql_loss`, `bellman_loss`, `policy_loss`, and `actor_updates`.

Recommended debugging run for 8-dataset layer-wise SAC:

```bash
nohup python3 -m rl_methods.rl_mmcr_SAC_Actor_Critic.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --source-baseline-json results/source_baselines_8datasets_test.json \
  --output-dir rl_mmcr_SAC_Actor_Critic_runs/layer8_fullstate_conservative_seed2029 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --episodes 300 \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --reward-eval-interval 13 \
  --batch-size 128 \
  --random-steps 1000 \
  --updates-per-step 1 \
  --actor-update-delay 4 \
  --freeze-actor-during-random-steps \
  --action-anchor-coef 0.05 \
  --cql-coef 0.1 \
  --alpha 0.005 \
  --lr 1e-4 \
  --critic-lr 1e-4 \
  --activation-reward-coef 0.0 \
  --log-every 5 \
  --seed 2029 \
  --gpu 0 \
  --amp \
  > sac_layer8_fullstate_conservative_seed2029.log 2>&1 &
```

## Activation Reward

Add dense layer-wise activation guidance with a small coefficient:

```bash
--activation-reward-coef 0.01
```

This compares each merged layer activation with the corresponding source-model
activation using cosine similarity on the first reward batch for each dataset.
It adds extra forward passes, so expect slower training.
