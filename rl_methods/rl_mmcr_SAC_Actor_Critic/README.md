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

## Activation Reward

Add dense layer-wise activation guidance with a small coefficient:

```bash
--activation-reward-coef 0.01
```

This compares each merged layer activation with the corresponding source-model
activation using cosine similarity on the first reward batch for each dataset.
It adds extra forward passes, so expect slower training.
