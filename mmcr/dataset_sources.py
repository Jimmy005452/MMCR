import os
import pathlib
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError as exc:
            raise RuntimeError("Stanford Cars requires scipy. Install it with: pip install scipy") from exc

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = devkit / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Stanford Cars not found. Please prepare the AdaMerging layout under "
                f"{self._base_folder}:\n"
                "  devkit/cars_train_annos.mat\n"
                "  devkit/cars_test_annos_withlabels.mat\n"
                "  devkit/cars_meta.mat\n"
                "  cars_train/*.jpg\n"
                "  cars_test/*.jpg"
            )

        self._samples = [
            (str(self._images_base_path / annotation["fname"]), annotation["class"] - 1)
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]
        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_path, target = self._samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def _check_exists(self) -> bool:
        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )


class RESISC45Split(datasets.ImageFolder):
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None) -> None:
        split_file = os.path.join(root, "resisc45", f"resisc45-{split}.txt")
        image_root = os.path.join(root, "resisc45", "NWPU-RESISC45")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Missing RESISC45 split file: {split_file}. "
                "Expected AdaMerging layout with resisc45-train.txt and resisc45-test.txt."
            )
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Missing RESISC45 image folder: {image_root}")

        valid_names = set()
        with open(split_file, "r", encoding="utf-8") as f:
            valid_names = {line.strip() for line in f if line.strip()}

        def is_valid_file(path: str) -> bool:
            return os.path.basename(path) in valid_names

        super().__init__(
            root=image_root,
            transform=transform,
            loader=default_loader,
            is_valid_file=is_valid_file,
        )


def make_imagefolder(root: pathlib.Path, transform):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing ImageFolder directory: {root}")
    return datasets.ImageFolder(str(root), transform=transform)
