# Model Merging Methods

Executable scripts for baseline and learned model merging methods. Run commands from the repository root.

## Task Arithmetic

```bash
python -m model_merging.task_arithmetic --checkpoint-root checkpoints --datasets mnist svhn gtsrb --scale 0.3 --output checkpoints/task_arithmetic/encoder_scale_0.3.pt
```

## TIES

```bash
python -m model_merging.ties_merge --checkpoint-root checkpoints --datasets mnist svhn gtsrb --top-k 20 --scale 0.3 --merge-func dis-sum --output checkpoints/ties/encoder_k20_scale_0.3.pt --overwrite
```

## DARE

```bash
python -m model_merging.dare_merge --merge-method ties --checkpoint-root checkpoints --datasets mnist svhn gtsrb --drop-rate 0.9 --top-k 20 --merge-func dis-sum --scale 0.3 --output checkpoints/dare_ties/encoder_drop0.9_k20_scale0.3.pt --overwrite
```

## NAN

```bash
python -m model_merging.nan_merge --merge-method ties --checkpoint-root checkpoints --datasets mnist svhn gtsrb --top-k 20 --merge-func dis-sum --scale 0.3 --output checkpoints/nan_ties/encoder_k20_scale0.3.pt --overwrite
```

## AdaMerging

```bash
python -m model_merging.adamerging --checkpoint-root checkpoints --datasets mnist svhn gtsrb --data-root data --top-k 20 --epochs 5 --batches-per-dataset 1 --batch-size 8 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging/test_encoder.pt --overwrite
```

## AdaMerging modes

```bash
python -m model_merging.adamerging_modes --lambda-mode tensor --checkpoint-root checkpoints --datasets mnist svhn gtsrb --data-root data --top-k 20 --epochs 5 --batches-per-dataset 1 --batch-size 4 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging_modes/tensor_test_encoder.pt --overwrite
```
