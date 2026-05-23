from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from mmcr.adamerging import build_adamerging_loaders, load_ties_task_vectors, softmax_entropy
from mmcr.checkpoints import load_head
from mmcr.models import build_image_encoder


class AdaMergingModesModel(nn.Module):
    """AdaMerging with task-wise or tensor-wise lambdas.

    tensor mode matches the reference "layer-wise" implementation more closely:
    each floating state_dict tensor gets its own lambda per task.
    """

    def __init__(
        self,
        encoder,
        heads: dict[str, nn.Module],
        pretrained_state: dict[str, torch.Tensor],
        task_vectors: list[dict[str, torch.Tensor]],
        task_names: list[str],
        device,
        lambda_mode: str = "task",
        prior: float = 0.3,
    ):
        super().__init__()
        if lambda_mode not in {"task", "tensor"}:
            raise ValueError("lambda_mode must be one of: task, tensor")

        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.pretrained_state = pretrained_state
        self.task_vectors = task_vectors
        self.task_names = task_names
        self.device = device
        self.lambda_mode = lambda_mode
        self.lambda_keys = list(task_vectors[0].keys())
        self.key_to_lambda_index = {key: idx for idx, key in enumerate(self.lambda_keys)}

        if lambda_mode == "task":
            shape = (len(task_vectors),)
        else:
            shape = (len(task_vectors), len(self.lambda_keys))
        self.lambdas_raw = nn.Parameter(torch.full(shape, prior))

        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad_(False)

    def lambdas(self):
        return torch.clamp(self.lambdas_raw, min=0.0, max=1.0)

    def _lambda_for(self, lambdas, task_idx: int, key: str):
        if self.lambda_mode == "task":
            return lambdas[task_idx]
        return lambdas[task_idx, self.key_to_lambda_index[key]]

    def build_merged_state(self):
        lambdas = self.lambdas()
        merged_state = {}

        for key, base_value in self.pretrained_state.items():
            value = base_value.detach().to(self.device)
            if key in self.key_to_lambda_index:
                delta = None
                for task_idx, task_vector in enumerate(self.task_vectors):
                    weighted_delta = self._lambda_for(lambdas, task_idx, key) * task_vector[key].detach().to(self.device)
                    delta = weighted_delta if delta is None else delta + weighted_delta
                value = value + delta
            merged_state[key] = value

        return merged_state

    @torch.no_grad()
    def export_merged_state(self):
        lambdas = self.lambdas().detach().cpu()
        merged_state = {}

        for key, base_value in self.pretrained_state.items():
            value = base_value.detach().cpu().clone()
            if key in self.key_to_lambda_index:
                delta = None
                for task_idx, task_vector in enumerate(self.task_vectors):
                    weighted_delta = self._lambda_for(lambdas, task_idx, key) * task_vector[key].detach().cpu()
                    delta = weighted_delta if delta is None else delta + weighted_delta
                value = value + delta.to(dtype=value.dtype)
            merged_state[key] = value

        return merged_state

    @torch.no_grad()
    def export_lambdas(self):
        lambdas = self.lambdas().detach().cpu()
        if self.lambda_mode == "task":
            return dict(zip(self.task_names, lambdas.tolist()))

        exported = {}
        for task_idx, task_name in enumerate(self.task_names):
            exported[task_name] = dict(zip(self.lambda_keys, lambdas[task_idx].tolist()))
        return exported

    @torch.no_grad()
    def lambda_summary(self):
        lambdas = self.lambdas().detach().cpu()
        summary = {
            "mean": lambdas.mean().item(),
            "min": lambdas.min().item(),
            "max": lambdas.max().item(),
        }
        if self.lambda_mode == "task":
            summary["by_task"] = dict(zip(self.task_names, lambdas.tolist()))
        return summary

    def forward(self, images, dataset: str, merged_state=None):
        if merged_state is None:
            merged_state = self.build_merged_state()
        features = functional_call(self.encoder, merged_state, (images,))
        return self.heads[dataset](features)


def build_adamerging_modes_model(
    arch: str,
    datasets: list[str],
    head_paths: list[Path | str],
    zeroshot_path: Path | str,
    encoder_paths: list[Path | str],
    device,
    lambda_mode: str = "task",
    top_k_percent: float = 20,
    prior: float = 0.3,
):
    pretrained_state, task_vectors, _ = load_ties_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=encoder_paths,
        top_k_percent=top_k_percent,
    )
    encoder = build_image_encoder(arch=arch, pretrained=False).to(device)
    heads = {dataset: load_head(head_path, device=device) for dataset, head_path in zip(datasets, head_paths)}
    return AdaMergingModesModel(
        encoder=encoder,
        heads=heads,
        pretrained_state=pretrained_state,
        task_vectors=task_vectors,
        task_names=datasets,
        device=device,
        lambda_mode=lambda_mode,
        prior=prior,
    ).to(device)


def train_adamerging_modes(
    model: AdaMergingModesModel,
    loaders,
    epochs: int,
    batches_per_dataset: int,
    lr: float,
    amp: bool = False,
):
    optimizer = torch.optim.Adam([model.lambdas_raw], lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    history = []
    normalizer = max(1, len(loaders) * batches_per_dataset)

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"AdaMerging-{model.lambda_mode}")
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen_batches = 0
        batch_bar = tqdm(total=len(loaders) * batches_per_dataset, desc=f"epoch {epoch}", leave=False)

        for dataset, loader in loaders.items():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= batches_per_dataset:
                    break

                images = images.to(model.device, non_blocking=True)
                merged_state = model.build_merged_state()
                with torch.autocast(device_type=model.device.type, enabled=amp and model.device.type == "cuda"):
                    logits = model(images, dataset, merged_state=merged_state)
                    loss = softmax_entropy(logits).mean() / normalizer

                loss.backward()
                total_loss += loss.item() * normalizer
                seen_batches += 1
                batch_bar.update(1)
                batch_bar.set_postfix(dataset=dataset, loss=f"{total_loss / max(seen_batches, 1):.4f}")
                del merged_state, logits, loss, images

        batch_bar.close()
        optimizer.step()

        summary = model.lambda_summary()
        row = {
            "epoch": epoch,
            "entropy_loss": total_loss / max(seen_batches, 1),
            "lambda_summary": summary,
        }
        history.append(row)
        epoch_bar.set_postfix(entropy=f"{row['entropy_loss']:.4f}", lambda_mean=f"{summary['mean']:.4f}")
        tqdm.write(
            f"epoch={epoch} entropy={row['entropy_loss']:.4f} "
            f"lambda_mean={summary['mean']:.4f} "
            f"lambda_min={summary['min']:.4f} "
            f"lambda_max={summary['max']:.4f}"
        )

    return history


__all__ = [
    "AdaMergingModesModel",
    "build_adamerging_loaders",
    "build_adamerging_modes_model",
    "train_adamerging_modes",
]
