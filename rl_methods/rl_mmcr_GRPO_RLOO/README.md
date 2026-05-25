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


## Speed Options

Two safe speed options are available for long layer-wise runs:

- `--cache-task-vectors-device`: keeps stacked task vectors on the GPU and uses vectorized per-layer merging. This avoids repeated CPU-to-GPU transfers during `_apply_layer_coefficients`. It should not change the reward definition, but it uses more VRAM.
- `--batched-reward-eval`: concatenates reward images from multiple datasets and runs fewer encoder forwards per reward step, then splits features for each dataset head. This keeps the same coefficients and reward formula, with only small possible floating-point/AMP differences.
- `--batched-reward-max-samples`: caps the concatenated image count per encoder forward. The default `128` is safer for ViT-L on 24GB GPUs; use `0` to concatenate all datasets if memory allows.

Example:

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  ... \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128
```

## Data-free Reward Modes

GRPO can keep the original labeled accuracy-retention reward or switch to unlabeled proxy rewards:

- `--reward-mode accuracy_retention`: uses labeled reward batches and optimizes retention against source-model accuracies. This is the original supervised reward.
- `--reward-mode entropy`: uses real images but ignores labels. The per-dataset score is negative merged-model entropy, so higher is better.
- `--reward-mode synthetic_entropy`: uses synthetic inputs and ignores teacher logits except for loading the generated files. The score is negative merged-model entropy.
- `--reward-mode synthetic_proxy`: uses synthetic inputs from `synthesis_data/generated/<dataset>/inputs.pt` plus source-model `teacher_logits.pt`. The proxy score is `-kl_weight * KL + agreement_weight * agreement - entropy_weight * entropy`.

For `entropy` and `synthetic_proxy`, the existing `retention` fields in logs/results are compatibility aliases for the proxy objective; inspect `reward_stats.reward_mode` and `reward_stats.mean_proxy_score` for the real meaning.

### Layer-wise Entropy GRPO

This is unlabeled but still uses real dataset images. It is closest to AdaMerging-style entropy minimization.

```bash
nohup python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_mmcr_GRPO_RLOO_runs/layer8_fullcoef_entropy_seed2034 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --reward-mode entropy \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rank \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 4 \
  --reward-eval-interval 13 \
  --lr 2e-4 \
  --entropy-coef 0.005 \
  --target-kl 0.02 \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --log-every 5 \
  --seed 2034 \
  --gpu 0 \
  --amp \
  > grpo_layer8_fullcoef_entropy_seed2034.log 2>&1 &
```

### Layer-wise Synthetic Entropy GRPO

This uses only synthetic inputs and minimizes merged-model entropy. It is useful as a pure data-free entropy baseline, but it is riskier than `synthetic_proxy` because confidence on synthetic inputs does not guarantee real accuracy.

```bash
nohup python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --synthesis-root synthesis_data/generated \
  --output-dir rl_mmcr_GRPO_RLOO_runs/layer8_fullcoef_synthetic_entropy_seed2035 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --reward-mode synthetic_entropy \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rank \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --reward-eval-interval 13 \
  --lr 2e-4 \
  --entropy-coef 0.005 \
  --target-kl 0.02 \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --log-every 5 \
  --seed 2035 \
  --gpu 1 \
  --amp \
  > grpo_layer8_fullcoef_synthetic_entropy_seed2035.log 2>&1 &
```

### Layer-wise Synthetic Proxy GRPO

This is the data-free path. It uses generated synthetic inputs and teacher logits saved by `synthesis_data/synthesize_inputs.py`; no real images or labels are needed during reward calculation.

```bash
nohup python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --synthesis-root synthesis_data/generated \
  --output-dir rl_mmcr_GRPO_RLOO_runs/layer8_fullcoef_synthetic_proxy_seed2035 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --reward-mode synthetic_proxy \
  --kl-weight 1.0 \
  --agreement-weight 0.5 \
  --entropy-weight 0.1 \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --advantage-mode rank \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --reward-eval-interval 13 \
  --lr 2e-4 \
  --entropy-coef 0.005 \
  --target-kl 0.02 \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --log-every 5 \
  --seed 2035 \
  --gpu 1 \
  --amp \
  > grpo_layer8_fullcoef_synthetic_proxy_seed2035.log 2>&1 &
```

Outputs are saved under `--output-dir`:

- `encoder.pt`
- `results.json`
- `training_curves.png`
- `reward_curves.png`
