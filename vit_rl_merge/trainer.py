import torch
import torch.nn.functional as F
from tqdm import tqdm

from vit_rl_merge.policy import TensorCoefficientPolicy
from vit_rl_merge.reward import format_float_dict, format_percent_dict


def rollout(env, policy, deterministic: bool = False, debug_decisions: bool = False):
    state = env.reset()
    log_probs = []
    entropies = []
    values = []
    infos = []
    done = False

    while not done:
        state = state.to(env.device)
        if deterministic:
            action, value = policy.deterministic_action(state)
            log_prob = torch.tensor(0.0, device=env.device)
            entropy = torch.tensor(0.0, device=env.device)
        else:
            action, log_prob, entropy, value = policy.sample_action(state)
        next_state, reward, done, info = env.step(action)
        if debug_decisions:
            print(
                f"  step={len(infos):03d} group={info['group']} "
                f"num_tensors={info['num_tensors']} coeffs={env.format_coefficients(action.detach().cpu())}"
            )
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        infos.append(info)
        state = next_state

    return {
        "reward": reward,
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "values": torch.stack(values),
        "infos": infos,
        "state_dict": env.export_state(),
        "coefficients": env.coefficient_matrix(),
    }


def train_actor_critic(
    env,
    episodes: int,
    rollouts_per_update: int = 1,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    hidden_dim: int = 128,
    init_log_std: float = -0.5,
    coefficient_mode: str = "sigmoid",
    coefficient_init: float = 1.0,
    debug_decisions: bool = False,
    debug_first_episode_only: bool = True,
    normalize_advantages: bool = True,
):
    policy = TensorCoefficientPolicy(
        env.state_dim,
        env.num_sources,
        hidden_dim=hidden_dim,
        init_log_std=init_log_std,
        coefficient_mode=coefficient_mode,
        coefficient_init=coefficient_init,
    ).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    best = {"episode": None, "reward": float("-inf"), "state_dict": None, "coefficients": None, "info": None}

    rollouts_per_update = max(1, rollouts_per_update)
    sampled_episodes = 0
    update_index = 0
    progress = tqdm(total=episodes, desc="ViT RL merge")
    while sampled_episodes < episodes:
        update_index += 1
        batch = []
        for rollout_index in range(rollouts_per_update):
            if sampled_episodes >= episodes:
                break
            episode = sampled_episodes + 1
            should_debug = debug_decisions and (not debug_first_episode_only or episode == 1)
            if should_debug:
                print(f"[episode {episode}] sampled decisions")
            trajectory = rollout(env, policy, deterministic=False, debug_decisions=should_debug)
            reward = trajectory["reward"]
            terminal_info = trajectory["infos"][-1]

            if reward > best["reward"]:
                best = {
                    "episode": episode,
                    "reward": reward,
                    "state_dict": trajectory["state_dict"],
                    "coefficients": trajectory["coefficients"],
                    "info": terminal_info,
                }

            returns = torch.full_like(trajectory["values"], float(reward))
            advantages = returns - trajectory["values"].detach()
            batch.append(
                {
                    "episode": episode,
                    "rollout_index": rollout_index,
                    "trajectory": trajectory,
                    "returns": returns,
                    "advantages": advantages,
                    "terminal_info": terminal_info,
                    "reward": reward,
                }
            )
            sampled_episodes += 1

        if normalize_advantages and batch:
            flat_advantages = torch.cat([item["advantages"].reshape(-1) for item in batch])
            std = flat_advantages.std(unbiased=False)
            if std.item() > 1e-6:
                mean = flat_advantages.mean()
                for item in batch:
                    item["advantages"] = (item["advantages"] - mean) / std

        policy_loss = torch.stack(
            [
                -(item["trajectory"]["log_probs"] * item["advantages"]).mean()
                for item in batch
            ]
        ).mean()
        value_loss = torch.stack(
            [
                F.mse_loss(item["trajectory"]["values"], item["returns"])
                for item in batch
            ]
        ).mean()
        entropy = torch.stack([item["trajectory"]["entropies"].mean() for item in batch]).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for item in batch:
            terminal_info = item["terminal_info"]
            row = {
                "episode": item["episode"],
                "update": update_index,
                "rollout_index": item["rollout_index"],
                "reward": item["reward"],
                "accuracies": terminal_info["accuracies"],
                "retentions": terminal_info["retentions"],
                "entropies": terminal_info["entropies"],
                "reward_stats": terminal_info["reward_stats"],
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }
            history.append(row)
            message = f"episode={item['episode']} update={update_index} reward={item['reward']:.4f}"
            if terminal_info["accuracies"]:
                message += (
                    f" accuracies={format_percent_dict(terminal_info['accuracies'])}"
                    f" retentions={format_percent_dict(terminal_info['retentions'])}"
                )
            if terminal_info.get("entropies"):
                message += f" entropies={format_float_dict(terminal_info['entropies'])}"
            tqdm.write(message)
        progress.update(len(batch))
        progress.set_postfix(reward=f"{batch[-1]['reward']:.4f}", best=f"{best['reward']:.4f}")
    progress.close()

    final = rollout(env, policy, deterministic=True, debug_decisions=False)
    return {
        "policy": policy,
        "history": history,
        "best": best,
        "final": {
            "reward": final["reward"],
            "state_dict": final["state_dict"],
            "coefficients": final["coefficients"],
            "info": final["infos"][-1],
        },
    }
