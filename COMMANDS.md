# MMCR 指令庫

這份檔案放常用指令，之後新增 TIES、MMCR、其他 evaluation 指令也都補在這裡。

## 環境

啟動環境：

```bash
conda activate mmcr
```

確認 PyTorch 可以用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 訓練

訓練 MNIST：

```bash
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp
```

訓練 Cars：

```bash
python train_vit_l16.py --dataset cars --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp
```

訓練 SUN397：

```bash
python train_vit_l16.py --dataset sun397 --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp
```

訓練 RESISC45：

```bash
python train_vit_l16.py --dataset resisc45 --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp
```

訓練 8 個資料集時可用的 dataset 名稱：

```text
sun397 cars resisc45 eurosat svhn gtsrb mnist dtd
```

關掉 scheduler：

```bash
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --epochs 10 --batch-size 64 --gpu 0 --amp --scheduler none
```

載入舊 encoder/head 繼續訓練：

```bash
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --encoder-checkpoint checkpoints/mnist/encoder.pt --head-checkpoint checkpoints/mnist/head.pt --epochs 10 --batch-size 64 --gpu 0 --amp
```

用 resume-dir 載入舊 encoder/head 繼續訓練：

```bash
python train_vit_l16.py --dataset mnist --data-root data --output-dir checkpoints --resume-dir checkpoints/mnist --epochs 10 --batch-size 64 --gpu 0 --amp
```

固定 converted encoder，只重新訓練 SUN397 head，並把 head 暫時存回 `checkpoint_another/converted/sun397/head.pt`：

```bash
python train_vit_l16.py --dataset sun397 --data-root data --output-dir checkpoint_another/converted --encoder-checkpoint checkpoint_another/converted/sun397/encoder.pt --freeze-encoder --epochs 10 --batch-size 64 --lr 1e-3 --weight-decay 0.0 --scheduler none --gpu 0 --amp
```

固定 converted encoder，只重新訓練 Cars head：

```bash
python train_vit_l16.py --dataset cars --data-root data --output-dir checkpoint_another/converted --encoder-checkpoint checkpoint_another/converted/cars/encoder.pt --freeze-encoder --epochs 10 --batch-size 64 --lr 1e-3 --weight-decay 0.0 --scheduler none --gpu 0 --amp
```

固定 converted encoder，只重新訓練 RESISC45 head：

```bash
python train_vit_l16.py --dataset resisc45 --data-root data --output-dir checkpoint_another/converted --encoder-checkpoint checkpoint_another/converted/resisc45/encoder.pt --freeze-encoder --epochs 10 --batch-size 64 --lr 1e-3 --weight-decay 0.0 --scheduler none --gpu 0 --amp
```

存 pretrained zero-shot encoder：

```bash
python save_zeroshot.py --output checkpoints/zeroshot.pt
```

重新產生 zeroshot.pt：

```bash
rm -f checkpoints/zeroshot.pt
python save_zeroshot.py --output checkpoints/zeroshot.pt
```

## Task Arithmetic

合併 3 個任務：

```bash
python task_arithmetic.py --checkpoint-root checkpoints --datasets mnist svhn gtsrb --scale 0.3 --output checkpoints/task_arithmetic/encoder_scale_0.3.pt
```

合併 8 個任務：

```bash
python task_arithmetic.py --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --scale 0.3 --output checkpoints/task_arithmetic/encoder_scale_0.3.pt
```

指定自己的 zeroshot encoder：

```bash
python task_arithmetic.py --checkpoint-root checkpoints --zeroshot checkpoints/zeroshot.pt --datasets mnist svhn gtsrb --scale 0.3 --output checkpoints/task_arithmetic/encoder_scale_0.3.pt
```

## TIES

合併 3 個任務：

```bash
python ties_merge.py --checkpoint-root checkpoints --datasets mnist svhn gtsrb --top-k 20 --scale 0.3 --merge-func dis-sum --output checkpoints/ties/encoder_k20_scale_0.3.pt
```

合併 8 個任務：

```bash
python ties_merge.py --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --top-k 20 --scale 0.3 --merge-func dis-sum --output checkpoints/ties/encoder_k20_scale_0.3.pt
```

如果輸出檔已經存在，要覆蓋：

```bash
python ties_merge.py --checkpoint-root checkpoints --datasets mnist svhn gtsrb --top-k 20 --scale 0.3 --merge-func dis-sum --output checkpoints/ties/encoder_k20_scale_0.3.pt --overwrite
```

評估 TIES merged encoder：

```bash
python eval_main.py --encoder checkpoints/ties/encoder_k20_scale_0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

## Hugging Face 權重比對

下載 tanganke 的 SUN397 CLIP ViT-L/14 權重到 `checkpoint_another/`，並跟目前 timm default model 比對 key：

```bash
python compare_hf_timm_checkpoint.py --repo-id tanganke/clip-vit-large-patch14_sun397 --download-root checkpoint_another --report checkpoint_another/tanganke_sun397_compare_report.json
```

如果要指定 timm arch：

```bash
python compare_hf_timm_checkpoint.py --repo-id tanganke/clip-vit-large-patch14_sun397 --download-root checkpoint_another --arch vit_large_patch14_clip_224.openai --report checkpoint_another/tanganke_sun397_compare_report.json
```

## Hugging Face CLIP 轉 timm encoder

轉換 tanganke 的 8 個 CLIP ViT-L/14 fine-tuned encoder：

```bash
python convert_hf_clip_to_timm.py --download-root checkpoint_another --output-root checkpoint_another/converted --arch vit_large_patch14_clip_224.openai
```

轉換後會輸出：

```text
checkpoint_another/converted/
  sun397/encoder.pt
  cars/encoder.pt
  resisc45/encoder.pt
  eurosat/encoder.pt
  svhn/encoder.pt
  gtsrb/encoder.pt
  mnist/encoder.pt
  dtd/encoder.pt
```

只轉換其中幾個：

```bash
python convert_hf_clip_to_timm.py --repos tanganke/clip-vit-large-patch14_sun397 tanganke/clip-vit-large-patch14_mnist --download-root checkpoint_another --output-root checkpoint_another/converted --arch vit_large_patch14_clip_224.openai
```

如果輸出檔已存在，要覆蓋：

```bash
python convert_hf_clip_to_timm.py --download-root checkpoint_another --output-root checkpoint_another/converted --arch vit_large_patch14_clip_224.openai --overwrite
```

## 評估

評估單一資料集：

```bash
python eval_main.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

評估 Task Arithmetic merged encoder，並針對多個資料集換各自 head：

```bash
python eval_main.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

只評估幾個資料集：

```bash
python eval_main.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets mnist svhn gtsrb --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

指定單一 head：

```bash
python eval_main.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --head checkpoints/mnist/head.pt --data-root data --batch-size 64 --gpu 0 --amp
```

不用 GPU：

```bash
python eval_main.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu -1
```

不讓 torchvision 自動下載資料：

```bash
python eval_main.py --encoder checkpoints/mnist/encoder.pt --datasets mnist --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --no-download
```

存評估結果成 JSON：

```bash
python eval_main.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets mnist svhn gtsrb --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/task_arithmetic_scale_0.3.json
```

同時存 JSON 和 txt：

```bash
python eval_main.py --encoder checkpoints/task_arithmetic/encoder_scale_0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/task_arithmetic_scale_0.3.json --results-txt results/task_arithmetic_scale_0.3.txt
```

## 資料夾格式

一般 checkpoint：

```text
checkpoints/
  zeroshot.pt
  mnist/
    encoder.pt
    head.pt
  svhn/
    encoder.pt
    head.pt
```

Cars 本地資料：

```text
data/
  stanford_cars/
    devkit/
      cars_train_annos.mat
      cars_test_annos_withlabels.mat
      cars_meta.mat
    cars_train/
      *.jpg
    cars_test/
      *.jpg
```

SUN397 本地資料：

```text
data/
  sun397/
    train/class_name/image.jpg
    test/class_name/image.jpg
```

RESISC45 本地資料：

```text
data/
  resisc45/
    resisc45-train.txt
    resisc45-test.txt
    NWPU-RESISC45/
      class_name/image.jpg
```

## Task Vector 相似度

比較一般 `checkpoints/` 裡面的 task vector cosine similarity，並輸出 CSV、JSON、heatmap：

```bash
python task_vector_similarity.py --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --output-dir results/task_vector_similarity
```

比較 converted encoder 的 task vector similarity：

```bash
python task_vector_similarity.py --checkpoint-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --output-dir results/converted_task_vector_similarity
```

只輸出數字，不畫圖：

```bash
python task_vector_similarity.py --checkpoint-root checkpoints --datasets mnist svhn gtsrb --output-dir results/task_vector_similarity_small --no-plot
```

## AdaMerging

用 8 個資料集做 task-wise AdaMerging，輸出 merged encoder：

```bash
python adamerging.py --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --data-root data --top-k 20 --prior 0.3 --epochs 500 --batches-per-dataset 1 --batch-size 8 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging/encoder_k20.pt
```

用 converted encoder/head 做 AdaMerging：

```bash
python adamerging.py --checkpoint-root checkpoint_another/converted --head-root checkpoint_another/converted --zeroshot checkpoints/zeroshot.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --data-root data --top-k 20 --prior 0.3 --epochs 500 --batches-per-dataset 1 --batch-size 8 --lr 1e-3 --gpu 0 --amp --output checkpoint_another/converted/adamerging/encoder_k20.pt
```

快速試跑 5 epochs：

```bash
python adamerging.py --checkpoint-root checkpoints --datasets mnist svhn gtsrb --data-root data --top-k 20 --epochs 5 --batches-per-dataset 1 --batch-size 8 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging/test_encoder.pt --overwrite
```

如果 AdaMerging 還是 OOM，用更保守的設定：

```bash
python adamerging.py --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --data-root data --top-k 20 --prior 0.3 --epochs 500 --batches-per-dataset 1 --batch-size 4 --num-workers 0 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging/encoder_k20.pt --overwrite
```

評估 AdaMerging merged encoder：

```bash
python eval_main.py --encoder checkpoints/adamerging/encoder_k20.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp
```

評估 AdaMerging 並存結果：

```bash
python eval_main.py --encoder checkpoints/adamerging/encoder_k20.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/adamerging_k20.json --results-txt results/adamerging_k20.txt
```

## AdaMerging Modes

新版 AdaMerging，保留舊版 `adamerging.py`，這支可以選 lambda mode。

Task-wise mode：

```bash
python adamerging_modes.py --lambda-mode task --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --data-root data --top-k 20 --prior 0.3 --epochs 500 --batches-per-dataset 1 --batch-size 8 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging_modes/task_encoder_k20.pt
```

Tensor-wise mode，也就是比較接近 reference 的 layer-wise 寫法：

```bash
python adamerging_modes.py --lambda-mode tensor --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --data-root data --top-k 20 --prior 0.3 --epochs 500 --batches-per-dataset 1 --batch-size 4 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging_modes/tensor_encoder_k20.pt
```

Tensor-wise mode 快速試跑：

```bash
python adamerging_modes.py --lambda-mode tensor --checkpoint-root checkpoints --datasets mnist svhn gtsrb --data-root data --top-k 20 --prior 0.3 --epochs 5 --batches-per-dataset 1 --batch-size 4 --lr 1e-3 --gpu 0 --amp --output checkpoints/adamerging_modes/tensor_test_encoder.pt --overwrite
```

評估 tensor-wise AdaMerging：

```bash
python eval_main.py --encoder checkpoints/adamerging_modes/tensor_encoder_k20.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/adamerging_tensor_k20.json --results-txt results/adamerging_tensor_k20.txt
```

## DARE

DARE + Task Arithmetic：

```bash
python dare_merge.py --merge-method ta --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --drop-rate 0.9 --scale 0.3 --seed 42 --output checkpoints/dare_ta/encoder_drop0.9_scale0.3.pt
```

DARE + TIES：

```bash
python dare_merge.py --merge-method ties --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --drop-rate 0.9 --top-k 20 --merge-func dis-sum --scale 0.3 --seed 42 --output checkpoints/dare_ties/encoder_drop0.9_k20_scale0.3.pt
```

評估 DARE + Task Arithmetic：

```bash
python eval_main.py --encoder checkpoints/dare_ta/encoder_drop0.9_scale0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/dare_ta_drop0.9_scale0.3.json --results-txt results/dare_ta_drop0.9_scale0.3.txt
```

評估 DARE + TIES：

```bash
python eval_main.py --encoder checkpoints/dare_ties/encoder_drop0.9_k20_scale0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/dare_ties_drop0.9_k20_scale0.3.json --results-txt results/dare_ties_drop0.9_k20_scale0.3.txt
```

## NAN

NAN + Task Arithmetic:
```bash
python nan_merge.py --merge-method ta --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --scale 0.3 --output checkpoints/nan_ta/encoder_scale0.3.pt
```

NAN + TIES:
```bash
python nan_merge.py --merge-method ties --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --top-k 20 --merge-func dis-sum --scale 0.3 --output checkpoints/nan_ties/encoder_k20_scale0.3.pt
```

Use task-vector norms instead of fine-tuned model norms:
```bash
python nan_merge.py --merge-method ta --checkpoint-root checkpoints --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --norm-target task-vector --scale 0.3 --output checkpoints/nan_ta/encoder_tasknorm_scale0.3.pt
```

Evaluate NAN + TIES:
```bash
python eval_main.py --encoder checkpoints/nan_ties/encoder_k20_scale0.3.pt --datasets sun397 cars resisc45 eurosat svhn gtsrb mnist dtd --checkpoint-root checkpoints --data-root data --batch-size 64 --gpu 0 --amp --results-json results/nan_ties_k20_scale0.3.json --results-txt results/nan_ties_k20_scale0.3.txt
```
