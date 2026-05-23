# RL Model Merging Methods

RL-based model merging packages. Run commands from the repository root.

## PPO-GAE Actor-Critic RL-MMCR

```bash
python -m rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn_gtsrb_eurosat_dtd \
  --episodes 500 \
  --rollouts-per-update 4 \
  --ppo-epochs 4 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 1 \
  --coefficient-mode positive \
  --coefficient-init 1.0 \
  --episode-reward-only \
  --log-every 10 \
  --gpu 0 \
  --amp
```

The maintained implementation lives in `rl_methods/rl_mmcr_PPO_GAE_Actor-Critic/`.
