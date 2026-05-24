# RL MMCR GRPO RLOO

This package implements the recommended GRPO-style baseline for the current
model-merging setup: **global coefficients + group-relative advantages**.

It does not use a value network. For each iteration it samples a group of global
coefficient vectors, evaluates each one, computes a group-relative advantage,
and updates a positive softplus-normal coefficient policy with PPO-style clipping.

Each coefficient is constrained to be non-negative. The coefficient sum is not
constrained to 1.

Supported advantage modes:

- `rloo`: leave-one-out reward baseline, normalized by group std
- `zscore`: reward z-score within the group
- `rank`: rank-based advantage for noisy rewards

Notes:

- GRPO currently supports global coefficients only.
- The default `--coefficient-mode positive` leaves coefficients unnormalized during merging.
- `--episode-reward-only` uses only the terminal objective as reward.
- `--export-policy best` exports the best sampled coefficient vector; use
  `--export-policy final` when you want to evaluate the deterministic learned
  policy itself.

## Run

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_mmcr_GRPO_RLOO_runs/global_rloo_seed2026 \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rloo \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --log-every 10 \
  --seed 2026 \
  --gpu 0 \
  --amp
```

For noisier rewards, try rank advantages:

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_mmcr_GRPO_RLOO_runs/global_rank_seed2027 \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rank \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --log-every 10 \
  --seed 2027 \
  --gpu 1 \
  --amp
```

Outputs are saved under `--output-dir`:

- `encoder.pt`
- `results.json`
- `training_curves.png`
- `reward_curves.png`
