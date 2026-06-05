# MMCR: Sequential Layer-wise Model Merging via Reinforcement Learning

MMCR learns sequential layer-wise coefficients for task-vector based Vision Transformer model merging. The project supports standard model-merging baselines and reinforcement-learning-based merging with GRPO/RLOO, PPO-GAE, and GRPO-style variants such as GSPO-like, Dr.GRPO-like, and DAPO-lite.

The default setting uses TIES-selected task vectors as the merge basis, then learns non-negative layer-wise coefficients with an entropy-based reward. Pretrained source encoders can be converted from Hugging Face checkpoints, and dataset-specific heads can either be downloaded or generated with the included source-model fine-tuning utility.

## Overview

The workflow is:

1. Prepare datasets: download the five torchvision-compatible datasets automatically, then download the prepared SUN397, Stanford Cars, and RESISC45 mirror.
2. Prepare checkpoints: create the zeroshot encoder, convert pretrained source encoders, and download or generate dataset heads.
3. Evaluate the source models.
4. Run model-merging baselines: Task Arithmetic, TIES, DARE, NAN, and AdaMerging.
5. Run RL-based merging: GRPO/RLOO, PPO-GAE, GSPO-like, Dr.GRPO-like, and DAPO-lite.
6. Evaluate any merged encoder with `eval.py`.
7. Optionally transfer a trained GRPO policy to another dataset combination with the same number of tasks.

## Environment Setup

- Python 3.11+
- GPU recommended (CUDA supported)

This project was developed on two RTX3090.

For running this project on your local machine, follow the steps below.

### Step 0: Clone the Repository

```bash
git clone https://github.com/Jimmy005452/MMCR.git
cd MMCR
```
### Step1: Using Conda (Recommended do)

```bash
# Create a conda environment
conda create -n mmcr python=3.11 -y

# Activate environment
conda activate mmcr
```
### Step2: Install PyTorch

Please install PyTorch based on your system configuration:

1. Visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Select your OS, package (pip), and CUDA version
3. Run the generated command

Example (CUDA 12.6):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
### Step3: Install Other Dependencies
```bash
pip install -r requirements.txt
```

## Repository Layout

```text
MMCR/
├─ README.md
├─ CLI.md                         # runnable command examples for experiments
├─ requirements.txt
├─ eval.py                         # evaluate any merged encoder
├─ mmcr/
│  ├─ data.py                      # dataset loaders and transforms
│  ├─ models.py                    # CLIP/timm model builders
│  ├─ checkpoints.py               # checkpoint loading and saving helpers
│  ├─ task_vectors.py              # task-vector construction utilities
│  └─ evaluation.py                # shared evaluation utilities
├─ model_merging/
│  ├─ task_arithmetic.py           # Task Arithmetic baseline
│  ├─ ties_merge.py                # streaming TIES baseline
│  ├─ dare_merge.py                # DARE baseline
│  ├─ nan_merge.py                 # NAN baseline
│  └─ adamerging_modes.py          # AdaMerging task/tensor modes
├─ rl_methods/
│  ├─ source_baselines.py          # evaluate individual source models
│  ├─ rl_mmcr_GRPO_RLOO/           # GRPO/RLOO and GRPO-style variants
│  └─ rl_mmcr_PPO_GAE_Actor-Critic/ # PPO-GAE actor-critic baseline
├─ analysis_tools/
│  └─ task-vector analysis tools
├─ hf_tools/
│  └─ convert_hf_clip_to_timm.py   # convert HF CLIP checkpoints to encoder.pt
└─ source_models/
   ├─ save_zeroshot.py             # save the shared zeroshot encoder
   └─ train_vit_l14.py             # train/download-compatible dataset heads
```

## Supported Datasets

Use these dataset keys in all commands:

| Key | Dataset | Classes | Notes |
| --- | --- | ---: | --- |
| `sun397` | SUN397 | 397 | Expected under `data/sun397/train` and `data/sun397/test`. |
| `stanford_cars` | Stanford Cars | 196 | Alias is normalized internally to `cars`. |
| `resisc45` | NWPU-RESISC45 | 45 | Uses split files under `data/resisc45/`. |
| `eurosat` | EuroSAT | 10 | Torchvision-compatible. |
| `svhn` | SVHN | 10 | Torchvision-compatible. |
| `gtsrb` | GTSRB | 43 | Torchvision-compatible. |
| `mnist` | MNIST | 10 | Converted to RGB by the loader. |
| `dtd` | DTD | 47 | Torchvision-compatible. |

For convenience, the eight-dataset order used in the experiments is:

```bash
DATASETS_8="sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd"
```

## Data Preparation

This repository does not store raw datasets. Five datasets are torchvision-compatible and can be downloaded automatically by the loaders. Three datasets need the prepared directory layout used by this project.

### Download Torchvision Datasets

The following datasets can be downloaded automatically when you run training or evaluation without `--no-download`:

```text
eurosat
svhn
gtsrb
mnist
dtd
```

Download the five torchvision-compatible datasets before checkpoint-dependent commands:

```bash
python - <<'PY'
from pathlib import Path
from mmcr.data import build_transforms, make_dataset

transform = build_transforms(train=False)
for dataset in ["eurosat", "svhn", "gtsrb", "mnist", "dtd"]:
    for train in [True, False]:
        split = "train" if train else "test"
        print(f"Downloading {dataset} {split}...")
        make_dataset(dataset, Path("data"), train=train, transform=transform, download=True)
PY
```

After the data is present locally, pass `--no-download` to avoid repeated download checks.

### Download Prepared Datasets from Hugging Face

SUN397, Stanford Cars, and RESISC45 require the prepared layout below. Download those three datasets into `data/` from the project mirror:

```bash
export HF_XET_HIGH_PERFORMANCE=1
hf download jimmy0214/RL-final-MMCR-dataset \
  --repo-type dataset \
  --local-dir data \
  --max-workers 16
```

### Expected Data Layout

```text
data/
├─ sun397/
│  ├─ train/
│  │  └─ <class_name>/*.jpg
│  └─ test/
│     └─ <class_name>/*.jpg
├─ stanford_cars/
│  ├─ devkit/
│  │  ├─ cars_train_annos.mat
│  │  ├─ cars_test_annos_withlabels.mat
│  │  └─ cars_meta.mat
│  ├─ cars_train/*.jpg
│  └─ cars_test/*.jpg
├─ resisc45/
│  ├─ resisc45-train.txt
│  ├─ resisc45-val.txt
│  ├─ resisc45-test.txt
│  └─ NWPU-RESISC45/
│     └─ <class_name>/*.jpg
├─ eurosat/
├─ svhn/
├─ gtsrb/
├─ mnist/
└─ dtd/
```

When the data mirror is already present, pass `--no-download` to avoid torchvision downloads.

## Checkpoint Preparation

### Create Zeroshot Encoder

Create the base CLIP ViT-L/14 encoder used as the zeroshot initialization:

```bash
python -m source_models.save_zeroshot --output checkpoints/zeroshot.pt --overwrite
```

### Download and Convert Pretrained Source Encoders

The eight pretrained Hugging Face CLIP source encoders can be downloaded and converted to this repository's timm `encoder.pt` format with:

```bash
python -m hf_tools.convert_hf_clip_to_timm \
  --download-root checkpoints/hf_raw \
  --output-root checkpoints \
  --overwrite
```

By default this converts the following repositories:

```text
tanganke/clip-vit-large-patch14_sun397
tanganke/clip-vit-large-patch14_stanford-cars
tanganke/clip-vit-large-patch14_resisc45
tanganke/clip-vit-large-patch14_eurosat
tanganke/clip-vit-large-patch14_svhn
tanganke/clip-vit-large-patch14_gtsrb
tanganke/clip-vit-large-patch14_mnist
tanganke/clip-vit-large-patch14_dtd
```

### Download Source Heads

We recommend directly using the heads we have trained. Please download the provided `head.pt` files from Hugging Face:

```bash
hf download jimmy0214/RL-final-MMCR-heads \
  --repo-type model \
  --local-dir checkpoints
```

We also provide the code for training the heads. For more details, please refer to the **Generate Source Heads by Fine-tuning** section in `CLI.md`.

### Expected Checkpoint Layout

The checkpoint directory should follow a folder-based structure. Each dataset-specific checkpoint must be placed under its corresponding dataset folder. Every dataset folder should contain two files: `encoder.pt` and `head.pt`.

The `encoder.pt` file stores the task-specific encoder weights, while `head.pt` stores the corresponding classification head. The shared zero-shot checkpoint should be placed directly under the root `checkpoints/` directory as `zeroshot.pt`.

```text
checkpoints/
├─ zeroshot.pt
├─ sun397/
│  ├─ encoder.pt
│  └─ head.pt
├─ stanford_cars/
├─ resisc45/
├─ eurosat/
├─ svhn/
├─ gtsrb/
├─ mnist/
└─ dtd/
```

All other dataset folders follow the same structure as `sun397/`.


## Evaluate Source Models

Evaluate each source model on its own test split and cache the source baseline accuracies:

```bash
python -m rl_methods.source_baselines \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --output-json results/source_baselines_8datasets_test.json
```

## Model Merging Baselines

For the commands used to run model merging baselines, please refer to the **Model Merging Baselines** section in `CLI.md`.

We provide instructions for running the following five baseline methods:

* Task Arithmetic
* TIES
* DARE
* NAN
* AdaMerging (Tensor-wise)

## RL-based Model Merging (GRPO on 8 Datasets)

Below are our recommended GRPO settings:

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

> **Note: GPU Configuration for GRPO Training**
>
> `--gpu` specifies the main GPU used for policy training, while `--trajectory-devices` specifies the GPU devices used for trajectory generation.
>
> For single-GPU execution, use the same device for both arguments:
>
> ```bash
> --gpu 0 --trajectory-devices 0
> ```
>
> For two-GPU trajectory generation, keep `--gpu` as the main training device and list both devices in `--trajectory-devices`:
>
> ```bash
> --gpu 0 --trajectory-devices 0 1
> ```

GRPO writes the following artifacts to `--output-dir`:

```text
encoder.pt          # exported merged encoder, controlled by --export-policy
best_encoder.pt     # merged encoder from the best selected sampled policy
final_encoder.pt    # merged encoder from the final deterministic policy
best_policy.pt      # policy checkpoint from the iteration that produced the best selected sample
final_policy.pt     # final policy checkpoint
results.json        # config, coefficients, per-iteration history, final eval, and artifact paths
training_curves.png # score/loss/KL/entropy curves
reward_curves.png   # per-episode reward curve
```

For experiments with other settings, please refer to the **RL-based Model Merging** section in `CLI.md`. It includes commands for the following settings:

* GRPO on Different Numbers of Datasets
* GRPO with Raw Task Vectors (TIES is used by default)
* GRPO Reward Interval Ablation
* GSPO-like
* Dr.GRPO-like
* DAPO-lite
* PPO

## Policy Transfer

Policy transfer applies a saved GRPO policy network to a new dataset combination with the same number of tasks. For example, a policy trained on a 2-task merge can be transferred to another 2-task merge, while a policy trained on a 3-task merge can be transferred to another 3-task merge.

The full workflow includes three steps:

1. Train a GRPO policy on a source dataset combination.
2. Apply the saved policy checkpoint to a new target dataset combination.
3. Evaluate the transferred merged model.

For detailed commands and examples, please refer to the **Policy Transfer** section in `CLI.md`.

## Evaluate Any Merged Encoder

Use `eval.py` for any encoder checkpoint, including outputs from model merging baselines, GRPO/PPO-based methods, or downloaded merged encoders.

```bash
python eval.py \
  --encoder <path/to/encoder.pt> \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --no-download \
  --results-json results/eval_custom_encoder.json \
  --results-txt results/eval_custom_encoder.txt
```

For additional evaluation options, including automatic dataset inference and comparison against cached source baselines, please refer to the **Evaluate Any Merged Encoder** section in `CLI.md`.
