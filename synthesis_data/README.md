# Synthetic Data Tools

Generate model-ready synthetic inputs from the already trained source models.
This is the first step toward a data-free merge reward: source models create
proxy inputs, then later the merged model can be scored by matching each source
model's cached teacher logits on those inputs.

## Generate all 8 synthetic datasets

Run from the repository root:

```bash
python -m synthesis_data.synthesize_inputs \
  --checkpoint-root checkpoints \
  --output-dir synthesis_data/generated \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --samples-per-dataset 64 \
  --batch-size 16 \
  --steps 150 \
  --gpu 0 \
  --amp \
  --overwrite
```

Use fewer samples or steps for a quick smoke test:

```bash
python -m synthesis_data.synthesize_inputs \
  --checkpoint-root checkpoints \
  --output-dir synthesis_data/generated_test \
  --datasets mnist svhn \
  --samples-per-dataset 8 \
  --batch-size 4 \
  --steps 20 \
  --gpu 0 \
  --amp \
  --overwrite
```

## Generate class-balanced inputs

`--samples-per-class` overrides `--samples-per-dataset`. This is useful when
you want every class to appear at least once.

```bash
python -m synthesis_data.synthesize_inputs \
  --checkpoint-root checkpoints \
  --output-dir synthesis_data/generated_balanced \
  --datasets mnist svhn gtsrb eurosat dtd \
  --samples-per-class 1 \
  --batch-size 16 \
  --steps 150 \
  --gpu 0 \
  --amp \
  --overwrite
```

For large-class datasets such as `sun397` and `stanford_cars`, start with
`--samples-per-dataset` first because full class-balanced synthesis is more
expensive.

## Outputs

Each dataset is saved under `synthesis_data/generated/<dataset>/`:

```text
inputs.pt
pixels.pt
teacher_logits.pt
pseudo_labels.pt
accepted_mask.pt
metadata.json
```

- `inputs.pt`: normalized tensors ready to feed into the ViT encoder.
- `pixels.pt`: the same images in `[0, 1]` pixel space for inspection.
- `teacher_logits.pt`: cached source-model logits for future KL rewards.
- `pseudo_labels.pt`: target classes used during input synthesis.
- `accepted_mask.pt`: samples passing the confidence/entropy thresholds.
- `metadata.json`: synthesis settings and summary statistics.

The current generator optimizes image tensors directly while keeping the source
encoder and head frozen. The loss combines pseudo-label cross entropy, entropy
minimization, augmentation consistency, total variation, and L2 regularization.


## Validate synthetic data usefulness

The main question is whether synthetic proxy scores rank merged models similarly
to real validation/test accuracy. First prepare a few candidate encoders and run
`eval.py` for each one, then compare those real results with synthetic proxy
metrics.

Example with three candidate encoders:

```bash
python -m synthesis_data.validate_proxy \
  --synthesis-root synthesis_data/generated \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --candidate task_arithmetic=checkpoints/task_arithmetic/encoder_mnist_svhn_gtsrb_eurosat_dtd_scale0.3.pt \
  --candidate ties=checkpoints/ties/encoder_mnist_svhn_gtsrb_eurosat_dtd_k20_scale0.3.pt \
  --candidate rl_mmcr=rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn_gtsrb_eurosat_dtd/encoder.pt \
  --real-result task_arithmetic=results/real_acc_task_arithmetic_scale0.3.json \
  --real-result ties=results/real_acc_ties_k20_scale0.3.json \
  --real-result rl_mmcr=results/real_acc_rl_mmcr.json \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --output-json results/synthetic_proxy_validation.json \
  --output-txt results/synthetic_proxy_validation.txt
```


Example with five merged-model candidates:

```bash
python -m synthesis_data.validate_proxy \
  --synthesis-root synthesis_data/generated \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --candidate task_arithmetic_0.1=checkpoints/task_arithmetic/encoder_mnist_svhn_gtsrb_eurosat_dtd_scale0.1.pt \
  --candidate task_arithmetic_0.3=checkpoints/task_arithmetic/encoder_mnist_svhn_gtsrb_eurosat_dtd_scale0.3.pt \
  --candidate task_arithmetic_0.5=checkpoints/task_arithmetic/encoder_mnist_svhn_gtsrb_eurosat_dtd_scale0.5.pt \
  --candidate ties_0.3=checkpoints/ties/encoder_mnist_svhn_gtsrb_eurosat_dtd_k20_scale0.3.pt \
  --candidate dare_ties_0.3=checkpoints/dare_ties/encoder_mnist_svhn_gtsrb_eurosat_dtd_drop0.9_k20_scale0.3.pt \
  --real-result task_arithmetic_0.1=results/real_acc_task_arithmetic_scale0.1.json \
  --real-result task_arithmetic_0.3=results/real_acc_task_arithmetic_scale0.3.json \
  --real-result task_arithmetic_0.5=results/real_acc_task_arithmetic_scale0.5.json \
  --real-result ties_0.3=results/real_acc_ties_scale0.3.json \
  --real-result dare_ties_0.3=results/real_acc_dare_ties_scale0.3.json \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --output-json results/synthetic_proxy_validation_5models.json \
  --output-txt results/synthetic_proxy_validation_5models.txt
```

If you only want proxy scores and do not have real accuracy JSON yet, omit all
`--real-result` arguments:

```bash
python -m synthesis_data.validate_proxy \
  --synthesis-root synthesis_data/generated \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --candidate rl_mmcr=rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn_gtsrb_eurosat_dtd/encoder.pt \
  --batch-size 64 \
  --gpu 0 \
  --amp
```

The validator reports:

- `KL`: lower means the merged model is closer to the source teacher logits.
- `agreement`: higher means source and merged top-1 predictions match more often.
- `merged_entropy`: lower means the merged model is more confident.
- `proxy_score`: `-KL + 0.5 * agreement - 0.1 * merged_entropy` by default.
- `Pearson` / `Spearman`: optional correlations between proxy score and real accuracy.

A useful synthetic dataset should give higher proxy scores to models that also
have higher real accuracy. If Spearman correlation is near zero or negative, do
not use that synthetic proxy as an RL reward yet.

## Useful filters

The command always saves all samples, but `accepted_mask.pt` records which ones
pass these thresholds:

```bash
--min-confidence 0.8 --max-entropy 0.5
```

For data-free reward experiments, start by checking whether low KL on these
synthetic inputs correlates with real validation accuracy before replacing the
current RL reward.
