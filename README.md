# MMCR Source Model Training Framework

This framework fine-tunes one `ViT-L/16` source model per dataset. The resulting
checkpoints are intended to be merged later by baseline methods and MMCR.

## Datasets

The proposal uses 8 vision datasets:

| Dataset key | Dataset |
| --- | --- |
| `sun397` | SUN397 |
| `stanford_cars` | Stanford Cars |
| `resisc45` | RESISC45 |
| `eurosat` | EuroSAT |
| `svhn` | SVHN |
| `gtsrb` | GTSRB |
| `mnist` | MNIST |
| `dtd` | DTD |

`eurosat`, `svhn`, `gtsrb`, `mnist`, and `dtd` are loaded with torchvision.
Only `sun397`, `stanford_cars`, and `resisc45` use the local AdaMerging-style layout:

```text
data/
  sun397/
    train/class_name/image.jpg
    test/class_name/image.jpg
  stanford_cars/
    devkit/
    cars_train/
    cars_test/
  resisc45/
    resisc45-train.txt
    resisc45-test.txt
    NWPU-RESISC45/
      class_name/image.jpg
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Project Layout

Top-level executable code is kept minimal: `eval.py` is the evaluation entry point.
Other runnable tools are grouped by purpose:

- `source_models/`: train source encoders and save zeroshot weights.
- `synthesis_data/`: generate synthetic inputs and teacher logits for data-free experiments.
- `model_merging/`: Task Arithmetic, TIES, DARE, NAN, and AdaMerging scripts.
- `rl_methods/`: RL-based model merging packages.
- `analysis_tools/`: task-vector inspection utilities.
- `hf_tools/`: Hugging Face checkpoint comparison and conversion tools.
- `mmcr/`: shared helper library used by the scripts above.

Each tool folder has its own `README.md` with commands.

## Train One Dataset

```powershell
python -m source_models.train_vit_l16 --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 16
```

Common options:

```powershell
# Use GPU 2
python -m source_models.train_vit_l16 --dataset mnist --gpu 2 --amp

# Disable the learning-rate scheduler
python -m source_models.train_vit_l16 --dataset mnist --scheduler none

# Also save the pretrained encoder as checkpoints/zeroshot.pt
python -m source_models.train_vit_l16 --dataset mnist --save-zeroshot

# Print all options with defaults
python -m source_models.train_vit_l16 -h
```

## Train All 8 Source Models

Run this from the repository root:

```bash
for dataset in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
  python -m source_models.train_vit_l16 \
    --dataset "$dataset" \
    --data-root data \
    --output-dir checkpoints \
    --epochs 10 \
    --batch-size 64 \
    --gpu 0 \
    --amp
done
```

Save the zeroshot encoder once if it is not already available:

```bash
python -m source_models.save_zeroshot --output checkpoints/zeroshot.pt
```

The checkpoints will be saved under:

```text
checkpoints/
  zeroshot.pt
  mnist/
    encoder.pt
    head.pt
    metadata.json
    metrics.json
  svhn/
    encoder.pt
    head.pt
    metadata.json
    metrics.json
  ...
```

The weight files are intentionally split:

- `zeroshot.pt` stores the original pretrained encoder before fine-tuning.
- `encoder.pt` stores the fine-tuned image encoder for one dataset.
- `head.pt` stores that dataset's classification head.
- `metadata.json` and `metrics.json` store run information and training history.

## Reuse Modules in Merging or Evaluation

Data loading, model loading, and evaluation are separated into importable modules.
Future merging scripts should reuse these instead of rewriting dataset/model code.

### Load a Dataset

```python
from mmcr.data import build_loader, build_loaders

train_loader, val_loader, num_classes = build_loaders(
    dataset_key="mnist",
    data_root="data",
    batch_size=16,
)

eval_loader, num_classes = build_loader(
    dataset_key="mnist",
    data_root="data",
    split="val",
    batch_size=16,
)
```

### Build or Load a Model

```python
import torch
from mmcr.checkpoints import load_classification_head, load_image_encoder
from mmcr.models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(num_classes=10).to(device)
model.image_encoder = load_image_encoder(
    "checkpoints/mnist/encoder.pt",
    fallback_encoder=model.image_encoder,
).to(device)
model.classification_head = load_classification_head(
    "checkpoints/mnist/head.pt",
    fallback_head=model.classification_head,
).to(device)
```

The model is an `ImageClassifier`:

```text
ImageClassifier(
  image_encoder=ViT-L/16,
  classification_head=Linear(...)
)
```

For model merging, you will usually merge `model.image_encoder` across source
models and keep one classification head per dataset, because the 8 datasets have
different numbers of classes.

### Evaluate a Model

```python
import torch.nn as nn
from mmcr.checkpoints import load_classification_head, load_image_encoder
from mmcr.data import build_loader
from mmcr.engine import evaluate
from mmcr.models import build_model

device = "cuda"
model = build_model(num_classes=10).to(device)
model.image_encoder = load_image_encoder(
    "checkpoints/mnist/encoder.pt",
    fallback_encoder=model.image_encoder,
).to(device)
model.classification_head = load_classification_head(
    "checkpoints/mnist/head.pt",
    fallback_head=model.classification_head,
).to(device)
loader, _ = build_loader("mnist", "data", split="val", batch_size=16)
loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device, amp=True)
```

### Evaluate an Encoder

```powershell
# Single dataset: encoder + that dataset's head
python eval.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp

# Same encoder, different dataset heads
python eval.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets mnist svhn gtsrb --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

Cache single-source baselines and compare a merged encoder against them.
When `--datasets` is omitted, `eval.py` automatically reads datasets from
`results.json` next to the merged `encoder.pt`:

```powershell
python eval.py \
  --encoder rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn_gtsrb_eurosat_dtd/encoder.pt \
  --checkpoint-root checkpoints \
  --data-root data \
  --batch-size 64 \
  --gpu 0 \
  --amp \
  --single-model-results-json results/single_model_baselines_mnist_svhn_gtsrb_eurosat_dtd.json \
  --results-json results/merged_mnist_svhn_gtsrb_eurosat_dtd.json \
  --comparison-json results/merged_vs_single_mnist_svhn_gtsrb_eurosat_dtd.json \
  --comparison-txt results/merged_vs_single_mnist_svhn_gtsrb_eurosat_dtd.txt
```

If the run metadata lives somewhere else, pass it explicitly:

```powershell
python eval.py \
  --encoder outputs/encoder.pt \
  --run-results-json rl_mmcr_PPO_GAE_Actor-Critic_runs/mnist_svhn_gtsrb_eurosat_dtd/results.json \
  --checkpoint-root checkpoints \
  --data-root data \
  --single-model-results-json results/single_model_baselines_mnist_svhn_gtsrb_eurosat_dtd.json
```

`--single-model-results-json` stores each dataset source encoder accuracy.
If the file already exists, cached datasets are reused and only missing datasets
are evaluated. Add `--refresh-single-model-results` to recompute the cache. The
comparison table reports single-source accuracy, merged accuracy, per-dataset
difference, and average accuracy difference. You can still pass `--datasets`
manually to override automatic detection.


## Notes

`ViT-L/16` is large. If the GPU memory is not enough, reduce `--batch-size` to 4
or 8, enable `--amp`, and increase `--grad-accum-steps`.
