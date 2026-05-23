from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mmcr.dataset_sources import RESISC45Split, StanfordCars, make_imagefolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = {
    "sun397": 397,
    "cars": 196,
    "resisc45": 45,
    "eurosat": 10,
    "svhn": 10,
    "gtsrb": 43,
    "mnist": 10,
    "dtd": 47,
}

ALIASES = {
    "stanford_cars": "cars",
}


class RGBConvert:
    def __call__(self, image):
        return image.convert("RGB")


def normalize_dataset_key(dataset_key: str) -> str:
    key = dataset_key.lower()
    return ALIASES.get(key, key)


def get_num_classes(dataset_key: str) -> int:
    key = normalize_dataset_key(dataset_key)
    if key not in NUM_CLASSES:
        valid = ", ".join(sorted([*NUM_CLASSES, *ALIASES]))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid choices: {valid}")
    return NUM_CLASSES[key]


def build_transforms(train: bool):
    """Fallback transforms used only when the caller does not pass model transforms."""
    steps = [RGBConvert(), transforms.Resize((224, 224))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def with_rgb(transform):
    return transforms.Compose([RGBConvert(), transform])


def get_transforms(train_transform=None, eval_transform=None):
    """Use caller-provided model transforms, or fallback transforms if none are given."""
    train_transform = build_transforms(train=True) if train_transform is None else with_rgb(train_transform)
    eval_transform = build_transforms(train=False) if eval_transform is None else with_rgb(eval_transform)
    return train_transform, eval_transform


def make_dataset(dataset_key: str, data_root: Path, train: bool, transform, download: bool):
    key = normalize_dataset_key(dataset_key)
    split_name = "train" if train else "test"

    if key == "sun397":
        return make_imagefolder(data_root / "sun397" / split_name, transform)

    if key == "cars":
        return StanfordCars(str(data_root), split=split_name, transform=transform, download=False)

    if key == "resisc45":
        return RESISC45Split(str(data_root), split=split_name, transform=transform)

    if key == "eurosat":
        return datasets.EuroSAT(root=str(data_root / "eurosat"), transform=transform, download=download)

    if key == "svhn":
        return datasets.SVHN(root=str(data_root / "svhn"), split=split_name, transform=transform, download=download)

    if key == "gtsrb":
        return datasets.GTSRB(root=str(data_root / "gtsrb"), split=split_name, transform=transform, download=download)

    if key == "mnist":
        return datasets.MNIST(root=str(data_root / "mnist"), train=train, transform=transform, download=download)

    if key == "dtd":
        return datasets.DTD(root=str(data_root / "dtd"), split=split_name, transform=transform, download=download)

    valid = ", ".join(sorted([*NUM_CLASSES, *ALIASES]))
    raise ValueError(f"Unknown dataset '{dataset_key}'. Valid choices: {valid}")


def build_loader(
    dataset_key: str,
    data_root: Path | str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    download: bool = True,
    train_transform=None,
    eval_transform=None,
    shuffle: bool | None = None,
    shuffle_generator=None,
):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    is_train = split == "train"
    train_transform, eval_transform = get_transforms(train_transform, eval_transform)
    transform = train_transform if is_train else eval_transform
    dataset = make_dataset(dataset_key, Path(data_root), is_train, transform, download)

    if shuffle is None:
        shuffle = is_train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=shuffle_generator if shuffle else None,
    )
    return loader, get_num_classes(dataset_key)


def build_loaders(
    dataset_key: str,
    data_root: Path | str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
    train_transform=None,
    eval_transform=None,
    shuffle_generator=None,
):
    del seed
    train_loader, num_classes = build_loader(
        dataset_key,
        data_root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        train_transform=train_transform,
        eval_transform=eval_transform,
        shuffle_generator=shuffle_generator,
    )
    val_loader, _ = build_loader(
        dataset_key,
        data_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    return train_loader, val_loader, num_classes
