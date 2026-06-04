# RL-MMCR SAC via Stable-Baselines3

這個方法使用 Stable-Baselines3 的 `SAC`，並沿用既有 MMCR environment builder。因此資料載入、reward、state、merge granularity、coefficient mode、final export/eval 都與原本 SAC 方法使用同一套環境程式碼。

執行入口：

```bash
python -m rl_method_sb3.rl_mmcr_SAC_Actor_Critic.train ...
```

## 相同的部分

- `RLMMCREnv` 來源相同，透過既有 PPO-GAE package 的 `build_environment()` 建立。
- Reward 計算相同：accuracy retention、step reward、terminal bonus、activation reward 等都由同一個 environment 處理。
- State mode 相同：`minimal`、`full_coefficients`。
- Action 維度相同：每一步輸出一個 task coefficient vector，`selected` 固定為全 1。
- Coefficient mode 相同：`--coefficient-mode positive`，merge 時不做 sum-to-one normalization。

## SB3 版 CLI

這個子方法只保留 SB3 SAC 真的會使用的訓練參數，避免舊手寫 SAC 的參數造成誤導：

- `--buffer-size` 對應 SB3 `buffer_size`
- `--learning-starts` 對應 SB3 `learning_starts`
- `--train-freq` 對應 SB3 `train_freq`
- `--gradient-steps` 對應 SB3 `gradient_steps`
- `--ent-coef` 對應 SB3 `ent_coef`，可用 float、`auto`、`auto_<initial_value>`
- `--target-entropy` 對應 SB3 `target_entropy`，可用 float 或 `auto`
- `--log-std-init` 對應 SB3 policy `log_std_init`
- `--action-max` 控制 Gym Box action upper bound，預設是 10.0

不再提供 `--critic-lr`、`--alpha-lr`、`--actor-update-delay`、`--freeze-actor-during-random-steps`、`--action-anchor-coef`、`--cql-coef`、`--log-std-min/max` 等 hand-written SAC 專用參數。

## 與手寫 SAC 的主要差異

- SB3 SAC 使用 bounded tanh-squashed Gaussian policy，action space 是 `Box(0, action_max)`；手寫版使用 positive softplus-normal actor。
- SB3 SAC 使用單一 `--lr` 作為 policy、critic、entropy coefficient optimizer 的 learning rate。
- SB3 warmup 在 `--learning-starts` 前不做 gradient update，並從 action space 隨機抽樣。
- `q_mean`、`target_q_mean`、`log_prob_mean` 這些手寫 SAC 才有的 detailed stats 無法從 SB3 logger 直接取得，會以 0.0 寫入以維持結果 schema。

## Layer-wise SAC

```bash
python -m rl_method_sb3.rl_mmcr_SAC_Actor_Critic.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_method_sb3_runs/rl_mmcr_SAC_Actor_Critic/layer_interval4_seed2026 \
  --merge-granularity layer \
  --episodes 300 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --reward-eval-interval 4 \
  --batch-size 128 \
  --buffer-size 50000 \
  --learning-starts 200 \
  --train-freq 1 \
  --gradient-steps 1 \
  --lr 3e-4 \
  --ent-coef 0.02 \
  --log-every 20 \
  --seed 2026 \
  --gpu 0 \
  --amp
```
