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


## SAC Actor-Critic RL-MMCR

Continuous value-based actor-critic baseline. It uses a Dirichlet actor, so the
action is directly a valid coefficient vector without binary gates.

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
relative advantages, and updates a Dirichlet policy.

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
