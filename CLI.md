## Generate Source Heads by Fine-tuning

Fine-tune one dataset and generate its `head.pt`:

```bash
python -m source_models.train_vit_l14 \
  --dataset mnist \
  --data-root data \
  --output-dir checkpoints \
  --freeze-encoder \
  --encoder-checkpoint checkpoints/mnist/encoder.pt \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-5 \
  --gpu 0 \
  --amp
```

Generate heads for all supported datasets:

```bash
for dataset in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
  python -m source_models.train_vit_l14 \
    --dataset "$dataset" \
    --data-root data \
    --output-dir checkpoints \
    --freeze-encoder \
    --encoder-checkpoint checkpoints/$dataset/encoder.pt \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-5 \
    --gpu 0 \
    --amp
done
```

`source_models.train_vit_l14` trains the dataset-specific classifier and saves the best checkpoint under `checkpoints/<dataset>/`. When validation accuracy improves, it writes:

```text
checkpoints/<dataset>/head.pt
checkpoints/<dataset>/metadata.json
checkpoints/<dataset>/metrics.json
```

## Model Merging Baselines

All baseline commands produce a merged encoder checkpoint. Evaluate the output with `eval.py` after each merge.

### Task Arithmetic

```bash
python -m model_merging.task_arithmetic \
  --checkpoint-root checkpoints \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --scale 0.3 \
  --output checkpoints/task_arithmetic/encoder_8datasets_scale0.3.pt

python eval.py \
  --encoder checkpoints/task_arithmetic/encoder_8datasets_scale0.3.pt \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_task_arithmetic_8datasets_scale0.3.json \
  --results-txt results/eval_task_arithmetic_8datasets_scale0.3.txt
```

### TIES

```bash
python -m model_merging.ties_merge \
  --checkpoint-root checkpoints \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --top-k 20 \
  --scale 1.0 \
  --merge-func dis-mean \
  --output checkpoints/ties/encoder_8datasets_k20_scale1.0.pt \
  --overwrite

python eval.py \
  --encoder checkpoints/ties/encoder_8datasets_k20_scale1.0.pt \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_ties_8datasets_k20_scale1.0.json \
  --results-txt results/eval_ties_8datasets_k20_scale1.0.txt
```

### DARE

```bash
python -m model_merging.dare_merge \
  --merge-method ta \
  --checkpoint-root checkpoints \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --drop-rate 0.9 \
  --scale 0.3 \
  --seed 2033 \
  --output checkpoints/dare_ties/encoder_8datasets_drop0.9_k20_scale0.3.pt \
  --overwrite

python eval.py \
  --encoder checkpoints/dare_ties/encoder_8datasets_drop0.9_k20_scale0.3.pt \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_dare_ties_8datasets_drop0.9_k20_scale0.3.json \
  --results-txt results/eval_dare_ties_8datasets_drop0.9_k20_scale0.3.txt
```

### NAN

```bash
python -m model_merging.nan_merge \
  --merge-method ta \
  --checkpoint-root checkpoints \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --scale 0.3 \
  --output checkpoints/nan_ties/encoder_8datasets_k20_scale0.3.pt \
  --overwrite

python eval.py \
  --encoder checkpoints/nan_ties/encoder_8datasets_k20_scale0.3.pt \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_nan_ties_8datasets_k20_scale0.3.json \
  --results-txt results/eval_nan_ties_8datasets_k20_scale0.3.txt
```

### AdaMerging (Tensor-wise)

```bash
python -m model_merging.adamerging_modes \
  --lambda-mode tensor \
  --checkpoint-root checkpoints \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --data-root data \
  --top-k 20 \
  --prior 0.3 \
  --epochs 500 \
  --batches-per-dataset 2 \
  --batch-size 32 \
  --lr 1e-3 \
  --gpu 0 \
  --amp \
  --no-download \
  --output checkpoints/adamerging_modes/encoder_8datasets_tensor.pt \
  --overwrite

python eval.py \
  --encoder checkpoints/adamerging_modes/encoder_8datasets_tensor.pt \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_adamerging_8datasets.json \
  --results-txt results/eval_adamerging_8datasets.txt
```


## RL-based Model Merging

### Base command

The following command is the base command. All settings described below are modified from it.

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --reward-mode entropy \
  --reward-split val \
  --reward-sampling-mode stratified_pool \
  --reward-pool-size 1024 \
  --selection-reward-pool-position 0 \
  --selection-candidates 1 \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --coefficient-init 0.3 \
  --task-vector-mode ties \
  --top-k-percent 20 \
  --policy-hidden-dim 128 \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --lr 3e-4 \
  --clip-eps 0.2 \
  --entropy-coef 0.005 \
  --target-kl 0.03 \
  --advantage-mode rloo \
  --terminal-bonus 1.0 \
  --reward-scale 1.0 \
  --step-reward-coef 0.25 \
  --score-imbalance-coef 0.5 \
  --reward-eval-interval 13 \
  --log-every 20 \
  --gpu 0 \
  --num-workers 4 \
  --seed 2033 \
  --amp \
  --no-download \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --trajectory-devices 0 1 \
  --output-dir rl_mmcr_GRPO_RLOO_runs/layer8_best_interval13_seed2033
```

> Please remember to change `--output-dir` for each run to avoid overwriting previous results.

### GRPO on Different Numbers of Datasets

Modify the `--datasets` argument to specify the target dataset combination:

```bash
# GRPO on 2 Datasets
--datasets eurosat svhn

# GRPO on 3 Datasets
--datasets eurosat svhn dtd
```

### GRPO with TIES Task Vectors

This is the default and recommended mode:

```bash
--task-vector-mode ties --top-k-percent 20
```

### GRPO with Raw Task Vectors

To disable TIES task-vector preprocessing and use raw fine-tuned minus zeroshot deltas:

```bash
# add or replace this flag in any base command
--task-vector-mode raw
```

### GRPO Reward Interval Ablation

Use the same base command and only change these flags:

```bash
# reward every 3 layer steps
--reward-eval-interval 3

# reward every 6 layer steps, current recommended 8-dataset setting
--reward-eval-interval 6

# reward every 13 layer steps
--reward-eval-interval 13

# terminal reward only
--reward-eval-interval 999 --episode-reward-only
```

### GSPO-like

GSPO-like updates use a trajectory-level policy ratio instead of a per-step ratio:

```bash
# add to the base command
--policy-loss-mode trajectory
```

### Dr.GRPO-like

Dr.GRPO-like removes group reward standard-deviation normalization:

```bash
# add to the base command
--advantage-mode rloo_no_std
```

### DAPO-lite

DAPO-lite enables asymmetric clipping and optional dynamic group resampling:

```bash
# add to the base command
--clip-eps-low 0.2 \
--clip-eps-high 0.28 \
--dynamic-sampling-min-std 0.01 \
--dynamic-sampling-max-resamples 2
```

### PPO

PPO uses the same shared environment, positive coefficient action parameterization, and reward modes as GRPO, but trains with a value network and GAE.

```bash
python -m rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --reward-mode entropy \
  --reward-split val \
  --reward-sampling-mode stratified_pool \
  --reward-pool-size 1024 \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --coefficient-init 0.3 \
  --task-vector-mode ties \
  --top-k-percent 20 \
  --policy-hidden-dim 128 \
  --episodes 300 \
  --rollouts-per-update 4 \
  --ppo-epochs 4 \
  --lr 3e-4 \
  --clip-eps 0.2 \
  --entropy-coef 0.005 \
  --target-kl 0.03 \
  --terminal-bonus 1.0 \
  --reward-scale 1.0 \
  --step-reward-coef 0.25 \
  --score-imbalance-coef 0.5 \
  --reward-eval-interval 13 \
  --log-every 20 \
  --gpu 0 \
  --num-workers 4 \
  --seed 2033 \
  --amp \
  --no-download \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --output-dir rl_mmcr_PPO_GAE_Actor-Critic_runs/layer8_entropy_seed2033
```

## Policy Transfer

### Train GRPO Policy

Example: train on `eurosat svhn`.

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.train \
  --datasets eurosat svhn \
  --checkpoint-root checkpoints \
  --data-root data \
  --reward-mode entropy \
  --reward-sampling-mode stratified_pool \
  --reward-pool-size 1024 \
  --selection-reward-pool-position 0 \
  --selection-candidates 1 \
  --reward-batch-size 64 \
  --reward-batches-per-dataset 1 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --coefficient-init 0.3 \
  --task-vector-mode ties \
  --top-k-percent 20 \
  --iterations 300 \
  --group-size 8 \
  --grpo-epochs 4 \
  --lr 3e-4 \
  --clip-eps 0.2 \
  --entropy-coef 0.005 \
  --target-kl 0.03 \
  --advantage-mode rloo \
  --reward-eval-interval 13 \
  --gpu 0 \
  --seed 2033 \
  --amp \
  --no-download \
  --cache-task-vectors-device \
  --batched-reward-eval \
  --batched-reward-max-samples 128 \
  --trajectory-devices 0 \
  --output-dir rl_mmcr_GRPO_RLOO_runs/pair_eurosat_svhn_transfer_source_seed2033
```

### Apply Policy to New Dataset Combination

Example: apply the 2-task policy to `resisc45 mnist` and evaluate immediately.

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.eval_policy_transfer \
  --policy-checkpoint rl_mmcr_GRPO_RLOO_runs/pair_eurosat_svhn_transfer_source_seed2033/best_policy.pt \
  --target-datasets resisc45 mnist \
  --checkpoint-root checkpoints \
  --data-root data \
  --gpu 0 \
  --amp \
  --no-download \
  --eval-batch-size 64 \
  --output-json results/transfer_pair_eurosat_svhn_to_resisc45_mnist_best.json \
  --output-txt results/transfer_pair_eurosat_svhn_to_resisc45_mnist_best.txt
```

### Evaluate Transferred Merged Model

`eval_policy_transfer.py` evaluates the transferred merged encoder in memory and writes the JSON/TXT outputs above. It does not need to save a merged encoder checkpoint.

If you want to evaluate coefficients stored in a GRPO `results.json` without writing an encoder checkpoint:

```bash
python -m rl_methods.rl_mmcr_GRPO_RLOO.eval_final_policy \
  --results-json rl_mmcr_GRPO_RLOO_runs/layer8_best_interval13_seed2033/results.json \
  --policy best_sample \
  --checkpoint-root checkpoints \
  --data-root data \
  --gpu 0 \
  --amp \
  --no-download \
  --output-json results/eval_grpo_best_policy.json \
  --output-txt results/eval_grpo_best_policy.txt
```

## Evaluate Any Merged Encoder

If the encoder is stored inside a run directory that contains `results.json`, `eval.py` can infer the dataset list automatically.

```bash
python eval.py \
  --encoder rl_mmcr_GRPO_RLOO_runs/layer8_best_interval13_seed2033/encoder.pt \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_layer8_best_interval13_seed2033.json \
  --results-txt results/eval_layer8_best_interval13_seed2033.txt
```

To compare a merged encoder against cached source baselines:

```bash
python eval.py \
  --encoder rl_mmcr_GRPO_RLOO_runs/layer8_best_interval13_seed2033/encoder.pt \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --single-model-results-json results/source_baselines_8datasets_test.json \
  --comparison-json results/compare_layer8_best_interval13_seed2033.json \
  --comparison-txt results/compare_layer8_best_interval13_seed2033.txt
```
