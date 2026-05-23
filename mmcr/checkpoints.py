from pathlib import Path

import torch

from mmcr.models import build_classification_head, build_image_encoder


ENCODER_FILE = "encoder.pt"
HEAD_FILE = "head.pt"


def save_encoder(path: Path, model):
    """Save only model.image_encoder as a state_dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.image_encoder.state_dict(), path)


def save_head(path: Path, model):
    """Save only model.classification_head as a state_dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.classification_head.state_dict(), path)


def load_image_encoder(path: Path | str, fallback_encoder, map_location="cpu"):
    """Load an encoder.pt state_dict into an existing encoder module."""
    state = torch.load(path, map_location=map_location)
    fallback_encoder.load_state_dict(state)
    return fallback_encoder


def load_classification_head(path: Path | str, fallback_head, map_location="cpu"):
    """Load a head.pt state_dict into an existing classification head."""
    state = torch.load(path, map_location=map_location)
    fallback_head.load_state_dict(state)
    return fallback_head


def load_encoder(path: Path | str, arch: str, device, map_location="cpu"):
    encoder = build_image_encoder(arch=arch, pretrained=False)
    encoder = load_image_encoder(path, fallback_encoder=encoder, map_location=map_location)
    return encoder.to(device)


def load_head(path: Path | str, device, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    num_classes, in_features = state["weight"].shape
    head = build_classification_head(in_features, num_classes)
    head.load_state_dict(state)
    return head.to(device)
