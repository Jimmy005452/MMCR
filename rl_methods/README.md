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
  --coefficient-init 0.3 \
  --episode-reward-only \
  --log-every 10 \
  --gpu 0 \
  --amp
```

The maintained implementation lives in `rl_methods/rl_mmcr_PPO_GAE_Actor-Critic/`.
Its CLI now uses `--coefficient-mode positive`, so coefficients are non-negative and are not normalized to sum to 1.


## SAC Actor-Critic RL-MMCR

Continuous value-based actor-critic baseline. It uses a positive softplus-normal
actor, so each task coefficient is non-negative and the per-layer coefficient
sum is not constrained to 1.

Layer-wise run:

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

Global coefficient sanity check:

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

The SAC implementation lives in `rl_methods/rl_mmcr_SAC_Actor_Critic/`.


## GRPO/RLOO Global Coefficients

GRPO-style global coefficient baseline without a value network. Each iteration
samples a group of global coefficient vectors, evaluates them, computes
relative advantages, and updates a positive softplus-normal policy. Coefficients
are constrained to be non-negative and are not normalized to sum to 1.

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

The GRPO/RLOO implementation lives in `rl_methods/rl_mmcr_GRPO_RLOO/`.

## State Modes

All RL methods support `--state-mode`:

- `minimal`: current compact state. It contains layer progress, current-layer task-vector geometry, and the mean coefficients used so far. For 8 datasets this is 25 dimensions.
- `full_coefficients`: adds the full layer-by-task coefficient table and a filled-layer mask. For 8 datasets with 27 merge layers this is 260 dimensions. This is mainly intended for SAC/PPO critics, because the value/Q network can see the partial merged-model configuration instead of only local layer information.

Start SAC/PPO debugging with:

```bash
--state-mode full_coefficients
```

GRPO has no value network, so `minimal` is usually enough unless you want a controlled comparison.

## Source Baseline Cache

Use this once to evaluate each source model on the full test split and save the
accuracies used as retention denominators during RL training:

```bash
python -m rl_methods.source_baselines \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --num-workers 4 \
  --gpu 0 \
  --amp \
  --output-json results/source_baselines_8datasets_test.json
```

Then pass the cache to any RL method:

```bash
--source-baseline-json results/source_baselines_8datasets_test.json
```

The merged model is still evaluated on the configured reward batches during
training; this cache only replaces the source-model baseline denominator.

## Activation Similarity Reward

Layer-wise RL methods can add a dense activation reward with:

```bash
--activation-reward-coef 0.01
```

When enabled, the environment precomputes source-model activations for each
layer using the first reward batch of each task. During each layer step it
compares the current merged model activation with the corresponding source
model activation by cosine similarity and adds the averaged score to reward:

```text
activation_reward_l = activation_reward_coef * mean_i cosine(A_merged,l(x_i), A_source_i,l(x_i))
```

This is only active for layer-wise runs. Use a small coefficient first because
it is added at every layer step and requires extra forward passes.

