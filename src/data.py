import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageDataset:
    """Image paths paired with integer IDs in deterministic filename order."""

    paths: tuple[Path, ...]
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def image_id(path: Path) -> int:
    """Parse IDs from both `00012.jpg` and `bbdd_00012.jpg`."""
    return int(path.stem.rsplit("_", 1)[-1])


def load_image_dataset(directory: str | Path) -> ImageDataset:
    directory = Path(directory)
    # Sorting is essential: submission row i must correspond to query image i.
    paths = tuple(sorted(directory.glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No JPG images found in {directory}")
    ids = np.asarray([image_id(path) for path in paths], dtype=int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate image IDs found in {directory}")
    return ImageDataset(paths=paths, ids=ids)


def load_ground_truth(path: str | Path) -> list[list[int]]:
    with Path(path).open("rb") as stream:
        values = pickle.load(stream)
    return [[int(image_id) for image_id in relevant] for relevant in values]


def read_bgr(path: str | Path) -> np.ndarray:
    """Read an image in OpenCV's native BGR channel order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def read_images(dataset: ImageDataset) -> list[np.ndarray]:
    """Read every image in a dataset without changing its deterministic order."""
    return [read_bgr(path) for path in dataset.paths]


def load_binary_masks(image_paths: tuple[Path, ...]) -> list[np.ndarray]:
    """Load same-stem PNG masks, returning ``True`` for foreground pixels."""
    masks: list[np.ndarray] = []
    for image_path in image_paths:
        mask_path = image_path.with_suffix(".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        image = read_bgr(image_path)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask/image size mismatch for {image_path.name}: "
                f"{mask.shape} vs {image.shape[:2]}"
            )
        masks.append(mask > 127)
    return masks
