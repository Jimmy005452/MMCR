import timm
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config

DEFAULT_ARCH = "vit_large_patch14_clip_224.openai"


class ImageClassifier(nn.Module):
    """Image encoder plus a dataset-specific classification head."""

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


def build_image_encoder(arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """Build a timm image encoder that returns features instead of logits."""
    return timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=0,
    )


def build_classification_head(in_features: int, num_classes: int):
    """Build a linear head for one dataset."""
    return nn.Linear(in_features, num_classes)


def build_model(num_classes: int, arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """Build the full classifier used during source-model fine-tuning."""
    image_encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    classification_head = build_classification_head(image_encoder.num_features, num_classes)
    return ImageClassifier(image_encoder, classification_head)


def build_model_transforms(arch: str = DEFAULT_ARCH, pretrained: bool = True):
    """Build train/eval preprocessing from timm's model data config."""
    image_encoder = build_image_encoder(arch=arch, pretrained=pretrained)
    data_config = resolve_model_data_config(image_encoder)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    return train_transform, eval_transform, data_config
