from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from mmcr.checkpoints import load_head
from mmcr.data import build_loader
from mmcr.models import build_image_encoder, build_model_transforms
from mmcr.task_vectors import load_state_dict
from mmcr.ties import get_merge_keys, state_dict_to_vector, ties_select_task_vectors


def softmax_entropy(logits: torch.Tensor):
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def vector_to_delta_state(vector: torch.Tensor, reference_state: dict[str, torch.Tensor], keys: list[str]):
    delta_state = {}
    offset = 0
    for key in keys:
        reference_value = reference_state[key]
        numel = reference_value.numel()
        delta_state[key] = vector[offset : offset + numel].reshape_as(reference_value).to(dtype=reference_value.dtype)
        offset += numel

    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} values, but only consumed {offset}.")
    return delta_state


def load_ties_task_vectors(
    zeroshot_path: Path | str,
    finetuned_paths: list[Path | str],
    top_k_percent: float = 20,
    map_location="cpu",
):
    pretrained_state = load_state_dict(zeroshot_path, map_location=map_location)
    finetuned_states = [load_state_dict(path, map_location=map_location) for path in finetuned_paths]
    keys = get_merge_keys(pretrained_state, finetuned_states)

    print(f"Flattening {len(finetuned_states)} checkpoints over {len(keys)} floating-point tensors.")
    flat_pretrained = state_dict_to_vector(pretrained_state, keys)
    flat_finetuned = torch.vstack([state_dict_to_vector(state, keys) for state in finetuned_states])
    task_vectors = flat_finetuned - flat_pretrained.unsqueeze(0)
    selected_vectors = ties_select_task_vectors(task_vectors, top_k_percent=top_k_percent)

    delta_states = [vector_to_delta_state(vector, pretrained_state, keys) for vector in selected_vectors]
    return pretrained_state, delta_states, keys


class AdaMergingModel(nn.Module):
    def __init__(
        self,
        encoder,
        heads: dict[str, nn.Module],
        pretrained_state: dict[str, torch.Tensor],
        task_vectors: list[dict[str, torch.Tensor]],
        task_names: list[str],
        device,
        prior: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.pretrained_state = pretrained_state
        self.task_vectors = task_vectors
        self.task_names = task_names
        self.device = device
        self.lambdas_raw = nn.Parameter(torch.full((len(task_vectors),), prior))

        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for head in self.heads.values():
            for param in head.parameters():
                param.requires_grad_(False)

    def lambdas(self):
        return torch.clamp(self.lambdas_raw, min=0.0, max=1.0)

    def build_merged_state(self):
        lambdas = self.lambdas()
        merged_state = {}
        for key, base_value in self.pretrained_state.items():
            value = base_value.detach().to(self.device)
            if key in self.task_vectors[0]:
                delta = 0
                for task_idx, task_vector in enumerate(self.task_vectors):
                    delta = delta + lambdas[task_idx] * task_vector[key].detach().to(self.device)
                value = value + delta
            merged_state[key] = value
        return merged_state

    @torch.no_grad()
    def export_merged_state(self):
        lambdas = self.lambdas().detach().cpu()
        merged_state = {}
        for key, base_value in self.pretrained_state.items():
            value = base_value.detach().cpu().clone()
            if key in self.task_vectors[0]:
                delta = 0
                for task_idx, task_vector in enumerate(self.task_vectors):
                    delta = delta + lambdas[task_idx] * task_vector[key].detach().cpu()
                value = value + delta.to(dtype=value.dtype)
            merged_state[key] = value
        return merged_state

    def forward(self, images, dataset: str, merged_state=None):
        if merged_state is None:
            merged_state = self.build_merged_state()
        features = functional_call(self.encoder, merged_state, (images,))
        return self.heads[dataset](features)


def build_adamerging_model(
    arch: str,
    datasets: list[str],
    head_paths: list[Path | str],
    zeroshot_path: Path | str,
    encoder_paths: list[Path | str],
    device,
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
    return AdaMergingModel(
        encoder=encoder,
        heads=heads,
        pretrained_state=pretrained_state,
        task_vectors=task_vectors,
        task_names=datasets,
        device=device,
        prior=prior,
    ).to(device)


def build_adamerging_loaders(
    datasets: list[str],
    data_root: Path | str,
    arch: str,
    batch_size: int,
    num_workers: int,
    download: bool = True,
):
    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)
    loaders = {}
    for dataset in datasets:
        loader, _ = build_loader(
            dataset,
            data_root,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            train_transform=eval_transform,
            eval_transform=eval_transform,
            shuffle=True,
        )
        loaders[dataset] = loader
    return loaders


def train_adamerging(
    model: AdaMergingModel,
    loaders,
    epochs: int,
    batches_per_dataset: int,
    lr: float,
    amp: bool = False,
):
    optimizer = torch.optim.Adam([model.lambdas_raw], lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    history = []
    normalizer = max(1, len(loaders) * batches_per_dataset)

    epoch_bar = tqdm(range(1, epochs + 1), desc="AdaMerging")
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen_batches = 0
        batch_bar = tqdm(
            total=len(loaders) * batches_per_dataset,
            desc=f"epoch {epoch}",
            leave=False,
        )

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

        row = {
            "epoch": epoch,
            "entropy_loss": total_loss / max(seen_batches, 1),
            "lambdas": dict(zip(model.task_names, model.lambdas().detach().cpu().tolist())),
        }
        history.append(row)
        epoch_bar.set_postfix(entropy=f"{row['entropy_loss']:.4f}")
        tqdm.write(f"epoch={epoch} entropy={row['entropy_loss']:.4f} lambdas={row['lambdas']}")

    return history
