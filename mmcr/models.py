from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config


DEFAULT_ARCH = "vit_large_patch16_224"


# ------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------


class ImageClassifier(nn.Module):
    """把 image encoder 和 classification head 接在一起。"""

    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

        if self.image_encoder is not None:
            if hasattr(self.image_encoder, "train_preprocess"):
                self.train_preprocess = self.image_encoder.train_preprocess
                self.val_preprocess = self.image_encoder.val_preprocess
            elif hasattr(self.image_encoder, "model") and hasattr(self.image_encoder.model, "train_preprocess"):
                self.train_preprocess = self.image_encoder.model.train_preprocess
                self.val_preprocess = self.image_encoder.model.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        return self.classification_head(features)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        torch.save(self, filename)

    @classmethod
    def load(cls, filename, map_location="cpu"):
        print(f"Loading image classifier from {filename}")
        return torch_load(filename, map_location=map_location, weights_only=False)


# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------


def build_image_encoder(arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """建立 timm image encoder，num_classes=0 代表只輸出 feature。"""
    return timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=0,
    )


def build_classification_head(in_features: int, num_classes: int):
    """建立 dataset-specific linear head。"""
    return nn.Linear(in_features, num_classes)


def build_model(num_classes: int, arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """建立完整 classifier：encoder + head。"""
    image_encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    classification_head = build_classification_head(image_encoder.num_features, num_classes)
    return ImageClassifier(image_encoder, classification_head)


def build_model_transforms(arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """用 timm 的設定建立 train/eval preprocessing。"""
    image_encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    data_config = resolve_model_data_config(image_encoder)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    return train_transform, eval_transform, data_config


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------


def torch_load(path: Path | str, map_location="cpu", weights_only=True):
    """相容不同 PyTorch 版本的 torch.load 包裝。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: Path | str, map_location="cpu"):
    """讀完整 checkpoint dict。"""
    return torch_load(path, map_location=map_location, weights_only=True)


def load_image_encoder_state(path: Path | str, map_location="cpu"):
    """
    從不同格式的 checkpoint 中抽出 encoder state_dict。

    支援：
    - 直接存 encoder state_dict
    - checkpoint["image_encoder"]
    - checkpoint["model"] 裡 image_encoder.* 的部分
    """
    obj = torch_load(path, map_location=map_location, weights_only=True)

    if isinstance(obj, dict) and "image_encoder" in obj:
        return obj["image_encoder"]
    if isinstance(obj, dict) and "model" in obj:
        model_state = obj["model"]
        return {
            key.removeprefix("image_encoder."): value
            for key, value in model_state.items()
            if key.startswith("image_encoder.")
        }
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported encoder checkpoint type: {type(obj)}")


def load_classification_head(path: Path | str, in_features: int, num_classes: int, map_location="cpu"):
    """
    從 checkpoint 載入 classification head。

    這裡會重新建立符合目前 dataset 類別數的 head，再載入權重。
    """
    obj = torch_load(path, map_location=map_location, weights_only=True)
    head = build_classification_head(in_features=in_features, num_classes=num_classes)

    if isinstance(obj, dict) and "classification_head" in obj:
        head.load_state_dict(obj["classification_head"])
    elif isinstance(obj, dict) and "model" in obj:
        model_state = obj["model"]
        head_state = {
            key.removeprefix("classification_head."): value
            for key, value in model_state.items()
            if key.startswith("classification_head.")
        }
        head.load_state_dict(head_state)
    elif isinstance(obj, dict):
        head.load_state_dict(obj)
    else:
        raise TypeError(f"Unsupported head checkpoint type: {type(obj)}")

    return head


def load_image_encoder(path: Path | str, fallback_encoder=None, map_location="cpu"):
    """把 encoder state_dict 載入到已建立好的 encoder module。"""
    if fallback_encoder is None:
        raise ValueError("fallback_encoder is required when loading an encoder state_dict.")

    encoder_state = load_image_encoder_state(path, map_location=map_location)
    fallback_encoder.load_state_dict(encoder_state)
    return fallback_encoder


def load_model_from_checkpoint(path: Path | str, device=None, pretrained: bool = False):
    """讀舊式完整 checkpoint，重建 ImageClassifier。"""
    ckpt = load_checkpoint(path, map_location="cpu")
    arch = ckpt.get("arch", DEFAULT_ARCH)
    num_classes = ckpt["num_classes"]
    model = build_model(num_classes=num_classes, arch=arch, pretrained=pretrained)
    model.load_state_dict(ckpt["model"])
    if device is not None:
        model = model.to(device)
    return model, ckpt


# ------------------------------------------------------------
# Base encoder states for merging
# ------------------------------------------------------------


def clone_state_to_cpu(state):
    """把 state_dict clone 到 CPU，避免後續 merge 不小心吃 GPU 記憶體。"""
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def build_base_encoder_state(
    arch: str = DEFAULT_ARCH,
    pretrained: bool = True,
    base_encoder_path: Path | str | None = None,
    map_location="cpu",
):
    """
    給 TA/TIES/NAN 用：只需要 base encoder 的 state_dict。

    如果有 base_encoder_path，就讀該 checkpoint；否則建立 timm pretrained encoder。
    """
    if base_encoder_path is not None:
        return clone_state_to_cpu(load_image_encoder_state(base_encoder_path, map_location=map_location))

    encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    return clone_state_to_cpu(encoder.state_dict())


def build_base_encoder_and_state(
    arch: str = DEFAULT_ARCH,
    pretrained: bool = True,
    base_encoder_path: Path | str | None = None,
    device=None,
):
    """
    給 AdaMerging 用：同時需要 encoder module 和 CPU base state。
    """
    encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    if base_encoder_path is not None:
        base_state = load_image_encoder_state(base_encoder_path, map_location="cpu")
        encoder.load_state_dict(base_state)

    base_state = clone_state_to_cpu(encoder.state_dict())
    if device is not None:
        encoder = encoder.to(device)
    return encoder, base_state


# ------------------------------------------------------------
# Checkpoint saving
# ------------------------------------------------------------


def save_checkpoint(path: Path, model, args, num_classes: int, best_acc: float, epoch: int):
    """存完整 model checkpoint，目前主要保留給舊流程相容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "image_classifier",
            "model": model.state_dict(),
            "image_encoder": model.image_encoder.state_dict(),
            "classification_head": model.classification_head.state_dict(),
            "dataset": args.dataset,
            "arch": getattr(args, "arch", DEFAULT_ARCH),
            "num_classes": num_classes,
            "best_acc": best_acc,
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def save_encoder(path: Path, model):
    """只存 encoder state_dict，merge 方法主要使用這個格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.image_encoder.state_dict(), path)


def save_head(path: Path, model):
    """只存 classification head state_dict。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.classification_head.state_dict(), path)
