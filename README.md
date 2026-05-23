# MMCR Source Model Training Framework

This framework fine-tunes one `ViT-L/16` source model per dataset. The resulting
checkpoints are intended to be merged later by baseline methods and MMCR.

## Datasets

The proposal uses 8 vision datasets:

| Dataset key | Dataset |
| --- | --- |
| `sun397` | SUN397 |
| `stanford_cars` | Stanford Cars |
| `cars` | Stanford Cars alias |
| `resisc45` | RESISC45 |
| `eurosat` | EuroSAT |
| `svhn` | SVHN |
| `gtsrb` | GTSRB |
| `mnist` | MNIST |
| `dtd` | DTD |

`eurosat`, `svhn`, `gtsrb`, `mnist`, and `dtd` are loaded with torchvision.
Only `sun397`, `cars`, and `resisc45` use the local AdaMerging-style layout:

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

## Train One Dataset

```powershell
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 16
```

Common options:

```powershell
# Use GPU 2
python train_vit_l16.py --dataset mnist --gpu 2 --amp

# Disable the learning-rate scheduler
python train_vit_l16.py --dataset mnist --scheduler none

# Also save the pretrained encoder as checkpoints/zeroshot.pt
python train_vit_l16.py --dataset mnist --save-zeroshot

# Print all options with defaults
python train_vit_l16.py -h
```

## Train All 8 Source Models

```powershell
.\scripts\train_all_vit_l16.ps1
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
python eval_main.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp

# Same encoder, different dataset heads
python eval_main.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets mnist svhn gtsrb --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

Suggested future scripts:

```text
merge_baselines.py   -> import mmcr.checkpoints, mmcr.models, and mmcr.data
evaluate_model.py    -> import mmcr.checkpoints and mmcr.engine.evaluate
mmcr_toy.py          -> import model/data/checkpoint helpers and implement RL search only
```

## Notes

`ViT-L/16` is large. If the GPU memory is not enough, reduce `--batch-size` to 4
or 8, enable `--amp`, and increase `--grad-accum-steps`.
