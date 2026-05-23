from pathlib import Path

import torch

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from mmcr.adamerging import build_adamerging_loaders
from mmcr.checkpoints import load_head
from mmcr.models import build_image_encoder

from vit_rl_merge.checkpoint import build_decision_groups, build_group_features, load_merge_inputs, merge_tensor
from vit_rl_merge.reward import capability_retention_reward, entropy_reward, softmax_entropy


class ViTTensorMergeEnv:
    """Tensor-wise model merging as an RL environment.

    Each step chooses coefficients for one state_dict tensor. Intermediate
    rewards are zero; the terminal reward measures capability retention.
    """

    def __init__(
        self,
        arch: str,
        datasets: list[str],
        head_paths: list[Path | str],
        zeroshot_path: Path | str,
        encoder_paths: list[Path | str],
        data_root: Path | str,
        device,
        scale: float = 1.0,
        decision_level: str = "tensor",
        batches_per_dataset: int = 1,
        batch_size: int = 8,
        num_workers: int = 4,
        amp: bool = False,
        download: bool = True,
        objective: str = "supervised",
        worst_weight: float = 0.5,
        std_weight: float = 0.25,
        reward_mode: str = "balanced",
        reward_scale: float = 1.0,
    ):
        if objective not in {"supervised", "entropy"}:
            raise ValueError("objective must be one of: supervised, entropy")
        self.arch = arch
        self.datasets = datasets
        self.device = device
        self.scale = scale
        self.decision_level = decision_level
        self.amp = amp
        self.objective = objective
        self.worst_weight = worst_weight
        self.std_weight = std_weight
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale

        self.zeroshot_state, self.finetuned_states, self.task_vectors, self.keys = load_merge_inputs(
            zeroshot_path=zeroshot_path,
            encoder_paths=encoder_paths,
        )
        self.decision_groups = build_decision_groups(self.keys, decision_level=decision_level)
        self.group_features = build_group_features(self.task_vectors, self.decision_groups)
        self.num_decisions = len(self.decision_groups)
        self.num_sources = len(self.task_vectors)

        self.encoder = build_image_encoder(arch=arch, pretrained=False).to(device)
        self.heads = {dataset: load_head(path, device=device) for dataset, path in zip(datasets, head_paths)}
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad_(False)

        loaders = build_adamerging_loaders(
            datasets=datasets,
            data_root=data_root,
            arch=arch,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
        )
        self.probe_batches = self._collect_probe_batches(loaders, batches_per_dataset)
        self.source_accuracies = self._compute_source_accuracies() if self.objective == "supervised" else {}
        self.state_dim = self.group_features.shape[1] + 2 * self.num_sources
        self.reset()

    def reset(self):
        self.decision_index = 0
        self.previous_coefficients = torch.full((self.num_sources,), 1.0 / self.num_sources)
        self.running_coefficients = torch.zeros(self.num_sources)
        self.actions = []
        self.merged_state = {key: value.detach().cpu().clone() for key, value in self.zeroshot_state.items()}
        return self._state()

    def step(self, action):
        if self.decision_index >= self.num_decisions:
            raise RuntimeError("Episode is already done. Call reset().")

        coefficients = action.detach().cpu().float()
        group = self.decision_groups[self.decision_index]
        for key in group["keys"]:
            self.merged_state[key] = merge_tensor(
                base_tensor=self.zeroshot_state[key],
                task_vectors=self.task_vectors,
                key=key,
                coefficients=coefficients,
                scale=self.scale,
            )

        self.actions.append(coefficients)
        self.running_coefficients += coefficients
        self.previous_coefficients = coefficients
        self.decision_index += 1

        done = self.decision_index == self.num_decisions
        if done:
            evaluation = self.evaluate_current_state()
            if self.objective == "supervised":
                accuracies = evaluation["accuracies"]
                reward, retentions, reward_stats = capability_retention_reward(
                    accuracies,
                    self.source_accuracies,
                    worst_weight=self.worst_weight,
                    std_weight=self.std_weight,
                    reward_mode=self.reward_mode,
                    reward_scale=self.reward_scale,
                )
            else:
                accuracies = {}
                retentions = {}
                reward, reward_stats = entropy_reward(evaluation["entropies"], reward_scale=self.reward_scale)
            info = {
                "group": group["name"],
                "num_tensors": len(group["keys"]),
                "accuracies": accuracies,
                "retentions": retentions,
                "entropies": evaluation["entropies"],
                "reward_stats": reward_stats,
                "coefficients": coefficients.tolist(),
            }
            next_state = None
        else:
            reward = 0.0
            info = {"group": group["name"], "num_tensors": len(group["keys"]), "coefficients": coefficients.tolist()}
            next_state = self._state()
        return next_state, reward, done, info

    def export_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.merged_state.items()}

    def coefficient_matrix(self):
        if not self.actions:
            return torch.empty(0, self.num_sources)
        return torch.stack(self.actions)

    def format_coefficients(self, coefficients):
        return "{" + ", ".join(
            f"{dataset}: {coefficient:.4f}"
            for dataset, coefficient in zip(self.datasets, coefficients.tolist())
        ) + "}"

    @torch.inference_mode()
    def evaluate_weight_average_baseline(self):
        state = {key: value.detach().cpu().clone() for key, value in self.zeroshot_state.items()}
        for key in self.keys:
            averaged = torch.stack(
                [finetuned_state[key].detach().cpu().float() for finetuned_state in self.finetuned_states],
                dim=0,
            ).mean(dim=0)
            state[key] = averaged.to(dtype=self.zeroshot_state[key].dtype)
        state = {key: value.detach().to(self.device) for key, value in state.items()}
        evaluation = self._evaluate_state(state, self.datasets)
        if self.objective == "supervised":
            accuracies = evaluation["accuracies"]
            reward, retentions, reward_stats = capability_retention_reward(
                accuracies,
                self.source_accuracies,
                worst_weight=self.worst_weight,
                std_weight=self.std_weight,
                reward_mode=self.reward_mode,
                reward_scale=self.reward_scale,
            )
        else:
            accuracies = {}
            retentions = {}
            reward, reward_stats = entropy_reward(evaluation["entropies"], reward_scale=self.reward_scale)
        return {
            "reward": reward,
            "accuracies": accuracies,
            "retentions": retentions,
            "entropies": evaluation["entropies"],
            "reward_stats": reward_stats,
        }

    def _state(self):
        if self.decision_index >= self.num_decisions:
            raise RuntimeError("No state after terminal step.")
        if self.decision_index == 0:
            running_mean = self.previous_coefficients
        else:
            running_mean = self.running_coefficients / self.decision_index
        return torch.cat(
            [
                self.group_features[self.decision_index],
                self.previous_coefficients,
                running_mean,
            ],
            dim=0,
        )

    def _collect_probe_batches(self, loaders, batches_per_dataset):
        probe_batches = {}
        for dataset, loader in loaders.items():
            batches = []
            for batch_idx, (images, targets) in enumerate(loader):
                if batch_idx >= batches_per_dataset:
                    break
                batches.append((images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)))
            if not batches:
                raise ValueError(f"No probe batches collected for {dataset}.")
            probe_batches[dataset] = batches
        return probe_batches

    @torch.inference_mode()
    def evaluate_current_state(self):
        state = {key: value.detach().to(self.device) for key, value in self.merged_state.items()}
        return self._evaluate_state(state, self.datasets)

    @torch.inference_mode()
    def _evaluate_state(self, state, datasets):
        self.encoder.eval()
        for head in self.heads.values():
            head.eval()

        accuracies = {}
        entropies = {}
        for dataset in datasets:
            correct = 0
            total = 0
            entropy_sum = 0.0
            for images, targets in self.probe_batches[dataset]:
                with torch.autocast(device_type=self.device.type, enabled=self.amp and self.device.type == "cuda"):
                    features = functional_call(self.encoder, state, (images,))
                    logits = self.heads[dataset](features)
                    batch_entropy = softmax_entropy(logits)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                entropy_sum += batch_entropy.sum().item()
            accuracies[dataset] = correct / max(total, 1)
            entropies[dataset] = entropy_sum / max(total, 1)
        return {"accuracies": accuracies, "entropies": entropies}

    def _compute_source_accuracies(self):
        source_accuracies = {}
        for dataset, source_state in zip(self.datasets, self.finetuned_states):
            state = {key: value.detach().to(self.device) for key, value in source_state.items()}
            evaluation = self._evaluate_state(state, [dataset])
            source_accuracies[dataset] = evaluation["accuracies"][dataset]
        return source_accuracies
