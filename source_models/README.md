# Source Model Tools

Scripts for creating the per-dataset source checkpoints used by model merging.

## Fine-tune one source model

```bash
python -m source_models.train_vit_l16 --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp
```

## Save the zeroshot encoder

```bash
python -m source_models.save_zeroshot --output checkpoints/zeroshot.pt
```

Outputs are expected under `checkpoints/<dataset>/encoder.pt`, `checkpoints/<dataset>/head.pt`, and `checkpoints/zeroshot.pt`.

## Fine-tune all source models

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
