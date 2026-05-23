# RL MMCR PPO GAE Actor-Critic

This package contains the maintained PPO-GAE Actor-Critic implementation for
layer-wise RL-based model merging. It keeps the same high-level method as the
old `rl_mmcr` package, but splits the code by responsibility:

- `cli.py`: command-line arguments and validation
- `merge.py`: layered task-vector loading and state merging
- `env.py`: layer-wise RL environment and retention reward
- `policy.py`: hybrid actor-critic policy
- `ppo.py`: rollout collection, GAE, PPO update, deterministic evaluation
- `train.py`: experiment orchestration and result export

It intentionally uses the shared `mmcr` package and does not import from
`toy_mmcr` or `vit_rl_merge`.

## Run

```powershell
python -m rl_mmcr_PPO_GAE_Actor-Critic.train --datasets mnist svhn --checkpoint-root checkpoints --data-root data --output-dir rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn --episodes 100 --rollouts-per-update 4 --ppo-epochs 4 --reward-batch-size 4 --reward-batches-per-dataset 1 --log-every 10 --gpu 0 --amp
```

For a quick smoke run:

```powershell
python -m rl_mmcr_PPO_GAE_Actor-Critic.train --datasets mnist svhn --checkpoint-root checkpoints --data-root data --output-dir rl_mmcr_PPO_GAE_Actor-Critic_runs/debug --episodes 20 --rollouts-per-update 2 --ppo-epochs 2 --reward-batch-size 2 --reward-batches-per-dataset 1 --log-every 5 --gpu 0 --amp --skip-final-eval
```

Outputs are saved under `--output-dir`:

- `encoder.pt`
- `results.json`
- `training_curves.png`
- `reward_curves.png`
