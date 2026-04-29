from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASET_CONFIG = Path("configs/datasets.yaml")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESISC45_DIRECTORY = "NWPU-RESISC45"


class RGBConvert:
    def __call__(self, image):
        return image.convert("RGB")


def load_dataset_config(path: Path = DATASET_CONFIG) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_num_classes(dataset_key: str) -> int:
    configs = load_dataset_config()
    if dataset_key not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid choices: {valid}")
    return configs[dataset_key]["num_classes"]


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                RGBConvert(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            RGBConvert(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def ensure_rgb_transform(transform):
    return transforms.Compose([RGBConvert(), transform])


def make_torchvision_dataset(name: str, root: Path, train: bool, transform, download: bool = True):
    if name == "SUN397":
        return datasets.SUN397(root=str(root), transform=transform, download=download)
    if name == "StanfordCars":
        split = "train" if train else "test"
        return datasets.StanfordCars(root=str(root), split=split, transform=transform, download=download)
    if name == "RESISC45":
        dataset_root = prepare_resisc45(root, download=download)
        if not dataset_root.exists():
            raise FileNotFoundError(
                "RESISC45 is not provided by torchvision in this environment. "
                f"Please prepare it as an ImageFolder dataset under: {dataset_root}\n"
                "Expected format: data/resisc45/NWPU-RESISC45/class_name/image.jpg"
            )
        return datasets.ImageFolder(root=str(dataset_root), transform=transform)
    if name == "EuroSAT":
        return datasets.EuroSAT(root=str(root), transform=transform, download=download)
    if name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(root=str(root), split=split, transform=transform, download=download)
    if name == "GTSRB":
        split = "train" if train else "test"
        return datasets.GTSRB(root=str(root), split=split, transform=transform, download=download)
    if name == "MNIST":
        return datasets.MNIST(root=str(root), train=train, transform=transform, download=download)
    if name == "DTD":
        split = "train" if train else "test"
        return datasets.DTD(root=str(root), split=split, transform=transform, download=download)
    raise ValueError(f"Unsupported dataset: {name}")


def has_local_dataset_files(root: Path) -> bool:
    return root.exists() and any(root.iterdir())


def make_dataset_prefer_local(name: str, root: Path, train: bool, transform, download: bool = True):
    if has_local_dataset_files(root):
        try:
            return make_torchvision_dataset(name, root, train, transform, download=False)
        except (FileNotFoundError, RuntimeError) as exc:
            if not download:
                raise
            print(
                f"Found local files under {root}, but {name} could not be loaded from them. "
                "Falling back to download."
            )

    return make_torchvision_dataset(name, root, train, transform, download=download)


def prepare_resisc45(root: Path, download: bool = True) -> Path:
    """Prepare RESISC45 via TorchGeo when available, then read it as ImageFolder."""
    dataset_root = root / RESISC45_DIRECTORY
    if dataset_root.exists():
        return dataset_root

    if not download:
        return dataset_root

    try:
        from torchgeo.datasets import RESISC45
    except ImportError as exc:
        raise ImportError(
            "RESISC45 auto-download requires optional TorchGeo, which may need "
            "native GIS dependencies such as PROJ/pyproj.\n"
            "Recommended install:\n"
            "conda install -c conda-forge torchgeo\n"
            "Alternative: manually extract RESISC45 to "
            f"{dataset_root} and rerun training."
        ) from exc

    # Instantiating the dataset triggers TorchGeo's download/extract routine.
    RESISC45(root=str(root), split="train", download=True)
    return dataset_root


def split_random_indices(dataset_size: int, val_fraction: float, seed: int):
    val_size = int(dataset_size * val_fraction)
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[:train_size], indices[train_size:]


def build_datasets(
    dataset_key: str,
    data_root: Path | str,
    seed: int = 42,
    train_transform=None,
    eval_transform=None,
    download: bool = True,
):
    configs = load_dataset_config()
    if dataset_key not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid choices: {valid}")

    cfg = configs[dataset_key]
    tv_name = cfg["torchvision_name"]
    dataset_root = Path(data_root) / dataset_key
    train_transform = build_transforms(train=True) if train_transform is None else ensure_rgb_transform(train_transform)
    eval_transform = build_transforms(train=False) if eval_transform is None else ensure_rgb_transform(eval_transform)

    if cfg["split_mode"] == "train_test":
        train_set = make_dataset_prefer_local(tv_name, dataset_root, True, train_transform, download)
        val_set = make_dataset_prefer_local(tv_name, dataset_root, False, eval_transform, download)
    elif cfg["split_mode"] == "random":
        train_full_set = make_dataset_prefer_local(tv_name, dataset_root, True, train_transform, download)
        eval_full_set = make_dataset_prefer_local(tv_name, dataset_root, True, eval_transform, download)
        train_indices, val_indices = split_random_indices(
            len(train_full_set),
            cfg.get("val_fraction", 0.1),
            seed,
        )
        train_set = Subset(train_full_set, train_indices)
        val_set = Subset(eval_full_set, val_indices)
    else:
        raise ValueError(f"Unsupported split mode: {cfg['split_mode']}")

    return train_set, val_set, cfg["num_classes"]


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
    train_set, val_set, num_classes = build_datasets(
        dataset_key,
        data_root,
        seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        download=download,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes


def build_eval_loader(
    dataset_key: str,
    data_root: Path | str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
    eval_transform=None,
):
    _, val_set, num_classes = build_datasets(
        dataset_key,
        data_root,
        seed,
        eval_transform=eval_transform,
        download=download,
    )
    loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, num_classes
