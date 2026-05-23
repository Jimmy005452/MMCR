# ViT RL Tensor Merge

This package keeps the RL structure explicit while reusing the existing `mmcr`
data, model, head, and checkpoint utilities.

## Concepts

- `Env.reset()` starts from the zeroshot state and uniform source coefficients.
- `Env.step(action)` applies one tensor's source coefficients and advances to
  the next tensor.
- Intermediate rewards are zero.
- The terminal reward is capability retention:
  `merged_accuracy / source_accuracy`, with worst-task and imbalance terms.

## Example

Task-level decisions, one coefficient vector for the whole model:

```bash
python -m vit_rl_merge.run --decision-level task --coefficient-mode positive --coefficient-init 1.0 --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn --data-root data --episodes 50 --batches-per-dataset 2 --batch-size 64 --lr 1e-3 --gpu 0 --amp --export-policy best --output checkpoint_another/converted/vit_rl_merge/task_mnist_svhn_encoder.pt --overwrite
```

Group-level decisions, one coefficient vector per tensor type:

```bash
python -m vit_rl_merge.run --decision-level group --coefficient-mode positive --coefficient-init 1.0 --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn --data-root data --episodes 50 --batches-per-dataset 2 --batch-size 64 --lr 1e-3 --gpu 0 --amp --export-policy best --output checkpoint_another/converted/vit_rl_merge/group_mnist_svhn_encoder.pt --overwrite
```

Layer-level decisions, one coefficient vector per ViT block:

```bash
python -m vit_rl_merge.run --decision-level layer --coefficient-mode positive --coefficient-init 1.0 --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn --data-root data --episodes 50 --batches-per-dataset 2 --batch-size 64 --lr 1e-3 --gpu 0 --amp --export-policy best --output checkpoint_another/converted/vit_rl_merge/layer_mnist_svhn_encoder.pt --overwrite
```

Tensor-level decisions, one coefficient vector per state_dict tensor:

```bash
python -m vit_rl_merge.run --decision-level tensor --coefficient-mode sigmoid --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn --data-root data --episodes 50 --batches-per-dataset 2 --batch-size 64 --lr 1e-3 --gpu 0 --amp --export-policy best --output checkpoint_another/converted/vit_rl_merge/tensor_mnist_svhn_encoder.pt --overwrite
```

Coefficient modes:

- `softmax`: coefficients are positive and sum to 1.
- `sigmoid`: each coefficient is independently constrained to `[0, 1]`.
- `positive`: each coefficient is positive and may exceed 1. This is the recommended default for TA-style task vectors.
- `unconstrained`: coefficients may be any real value.

The default `--scale` is `1.0`, so the RL coefficients directly decide how much
of each task vector to add. Passing another scale is only for ablations.

Reward modes:

- `balanced`: the original objective, combining mean retention, worst retention, and imbalance penalty.
- `worst`: uses only the lowest retention, useful when one dataset collapses.
- `mean_worst`: averages mean retention and worst retention.
- `harmonic`: harmonic mean of retentions, which strongly penalizes one bad dataset.

When the reward changes too little for policy-gradient updates, use
`--reward-scale 10` to amplify the learning signal without changing which
solution is considered better.

Objectives:

- `--objective supervised`: uses label-based accuracy retention reward.
- `--objective entropy`: label-free objective, rewards lower prediction entropy
  on unlabeled images. This is closer to AdaMerging's unsupervised signal.

The reward is terminal-only, so credit assignment is noisy. Use
`--rollouts-per-update 4` or `8` to sample several complete episodes before one
policy update. Advantages are normalized across those rollouts by default, so
the policy learns from which sampled merge was better within the same update.

To inspect one episode's decisions:

```bash
python -m vit_rl_merge.run --decision-level layer --coefficient-mode positive --coefficient-init 1.0 --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn --data-root data --episodes 1 --batches-per-dataset 1 --batch-size 32 --gpu 0 --amp --debug-decisions --output checkpoint_another/converted/vit_rl_merge/debug_encoder.pt --overwrite
```

By default, the selected `best` or `final` decisions are printed after training.
Use `--debug-decisions --debug-all-episodes` only when you want every sampled
decision from every episode. The full best/final coefficient tables are saved in
the `.lambdas.json` file.

Evaluate:

```bash
python eval_main.py --encoder checkpoint_another/converted/vit_rl_merge/mnist_svhn_encoder.pt --datasets mnist svhn --checkpoint-root checkpoint_another/converted --data-root data --batch-size 64 --gpu 0 --amp --results-json results/vit_rl_merge/mnist_svhn.json --results-txt results/vit_rl_merge/mnist_svhn.txt
```
