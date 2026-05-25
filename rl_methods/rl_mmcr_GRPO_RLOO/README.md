# RL MMCR GRPO RLOO

This package implements the recommended GRPO-style baseline for the current
model-merging setup: **global or layer-wise coefficients + group-relative advantages**.

It does not use a value network. For global runs, each iteration samples a group of coefficient vectors. For layer-wise runs, each iteration samples a group of complete coefficient trajectories, evaluates each trajectory, computes a group-relative advantage, and updates a positive softplus-normal coefficient policy with PPO-style clipping.

Each coefficient is constrained to be non-negative. The coefficient sum is not
constrained to 1.

Supported advantage modes:

- `rloo`: leave-one-out reward baseline, normalized by group std
- `zscore`: reward z-score within the group
- `rank`: rank-based advantage for noisy rewards

Notes:

- `--merge-granularity global` uses one coefficient vector shared by all layers.
- `--merge-granularity layer` uses trajectory-level RLOO: each group sample is a complete layer-wise coefficient trajectory, and every layer action in that trajectory shares the same group-relative advantage.
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

## Layer-wise Trajectory GRPO

This samples `group-size` full layer-wise trajectories per iteration. A trajectory contains one coefficient vector per merge layer. The terminal trajectory reward/objective is used to compute RLOO advantages across the group, then the same trajectory advantage is applied to every layer action from that trajectory.

```bash
nohup python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --source-baseline-json results/source_baselines_8datasets_test.json \
  --output-dir rl_mmcr_GRPO_RLOO_runs/layer8_positive_rloo_seed2030 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rloo \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --reward-eval-interval 13 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --log-every 5 \
  --seed 2030 \
  --gpu 0 \
  --amp \
  > grpo_layer8_positive_rloo_seed2030.log 2>&1 &
```

Layer-wise GRPO is slower than global GRPO because each candidate contains one action per merge layer. Increase `--reward-eval-interval` to reduce dense reward evaluations.

Outputs are saved under `--output-dir`:

- `encoder.pt`
- `results.json`
- `training_curves.png`
- `reward_curves.png`
