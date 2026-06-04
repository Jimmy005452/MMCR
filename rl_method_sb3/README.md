# RL-MMCR Stable-Baselines3

這個資料夾提供 Stable-Baselines3 版本的 RL-MMCR 訓練入口。它沿用既有的 MMCR environment、reward、merge/export 程式碼，但把原本手寫的 GRPO/RLOO policy update 改成套件呼叫。

Stable-Baselines3 沒有內建 GRPO，所以預設用 SB3 `PPO` 作為最接近的 policy-gradient 替代；也可用 `--algo sac` 改跑 SB3 `SAC`。

## 安裝

```bash
pip install stable-baselines3
```

或使用 repo 的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 範例：沿用 GRPO 風格參數

```bash
python -m rl_method_sb3.train \
  --datasets mnist svhn gtsrb eurosat dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --output-dir rl_method_sb3_runs/global8_ppo_seed2026 \
  --merge-granularity global \
  --iterations 300 \
  --group-size 8 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --log-every 10 \
  --seed 2026 \
  --gpu 0 \
  --amp
```

若沒有指定 `--episodes`，程式會用 `--iterations * --group-size` 當作總 episode 數，讓評估樣本數接近原本 GRPO 每 iteration 採樣一個 group 的設定。舊 GRPO 的 `--advantage-mode`、`--target-kl`、`--min-concentration`、`--log-std-min/max` 也保留為相容參數，其中 `--target-kl` 會傳給 SB3 PPO，其餘由 SB3 內部策略取代。

## 範例：直接指定 SB3 episode 數

```bash
python -m rl_method_sb3.train \
  --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd \
  --checkpoint-root checkpoints \
  --data-root data \
  --source-baseline-json results/source_baselines_8datasets_test.json \
  --output-dir rl_method_sb3_runs/layer8_ppo_seed2036 \
  --merge-granularity layer \
  --state-mode full_coefficients \
  --episodes 500 \
  --n-steps 216 \
  --batch-size 64 \
  --reward-batch-size 32 \
  --reward-batches-per-dataset 4 \
  --reward-eval-interval 4 \
  --action-max 2.0 \
  --seed 2036 \
  --gpu 0 \
  --amp
```

## 輸出

每次訓練會輸出：

- `results.json`
- `encoder.pt`
- `sb3_model.zip`
- `training_curves.png`
- `reward_curves.png`
