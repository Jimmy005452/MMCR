# RL MMCR SAC Actor-Critic

This package implements a continuous value-based actor-critic baseline using SAC.
It reuses the maintained PPO-GAE environment and model-merging code, but replaces
PPO with off-policy SAC over continuous coefficient vectors.

The actor is a Dirichlet policy, so every action is directly a valid coefficient
vector on the simplex:

```text
coefficients >= 0
sum(coefficients) = 1
```

There is no binary gate in this method.

## Layer-wise SAC

```bash
python -m rl_methods.rl_mmcr_SAC_Actor_Critic.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
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
