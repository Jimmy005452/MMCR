# MMCR Source Model Training Framework

This framework fine-tunes one `ViT-L/16` source model per dataset. The resulting
checkpoints are intended to be merged later by baseline methods and MMCR.

## Datasets

The proposal uses 8 vision datasets:

- SUN397
- Stanford Cars
- RESISC45
- EuroSAT
- SVHN
- GTSRB
- MNIST
- DTD

`RESISC45` is not available as `torchvision.datasets.RESISC45` in this
environment. This project uses TorchGeo to download/extract it when available,
then reads the extracted files as an ImageFolder dataset.

Automatic download:

```powershell
python -m pip install torchgeo
python train_vit_l16.py --dataset resisc45 --data-root data --output-dir checkpoints --epochs 10 --batch-size 16 --amp
```

Expected extracted format:

```text
data/
  resisc45/
    NWPU-RESISC45/
      airplane/
        image_001.jpg
        ...
      airport/
        image_001.jpg
        ...
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train One Dataset

```powershell
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 16
```

## Train All 8 Source Models

```powershell
.\scripts\train_all_vit_l16.ps1
```

The checkpoints will be saved under:

```text
checkpoints/
  mnist/best.pt
  svhn/best.pt
  ...
```

Each checkpoint stores:

- model weights
- dataset name
- number of classes
- best validation accuracy
- training arguments

## Reuse Modules in Merging or Evaluation

Data loading, model loading, and evaluation are separated into importable modules.
Future merging scripts should reuse these instead of rewriting dataset/model code.

### Load a Dataset

```python
from mmcr.data import build_eval_loader, build_loaders

train_loader, val_loader, num_classes = build_loaders(
    dataset_key="mnist",
    data_root="data",
    batch_size=16,
)

eval_loader, num_classes = build_eval_loader(
    dataset_key="mnist",
    data_root="data",
    batch_size=16,
)
```

### Build or Load a Model

```python
import torch
from mmcr.models import build_model, load_model_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(num_classes=10).to(device)
model, ckpt = load_model_from_checkpoint("checkpoints/mnist/best.pt", device=device)
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
from mmcr.data import build_eval_loader
from mmcr.engine import evaluate
from mmcr.models import load_model_from_checkpoint

device = "cuda"
model, ckpt = load_model_from_checkpoint("checkpoints/mnist/best.pt", device=device)
loader, _ = build_eval_loader("mnist", "data", batch_size=16)
loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device, amp=True)
```

Suggested future scripts:

```text
merge_baselines.py   -> import mmcr.models and mmcr.data
evaluate_model.py    -> import mmcr.engine.evaluate
mmcr_toy.py          -> import model/data helpers and implement RL search only
```

## Notes

`ViT-L/16` is large. If the GPU memory is not enough, reduce `--batch-size` to 4
or 8, enable `--amp`, and increase `--grad-accum-steps`.
