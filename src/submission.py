"""Serialization and validation for C1 ranking and mask submissions."""

import pickle
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .data import ImageDataset


def save_rankings(
    rankings: Sequence[Sequence[int]],
    output_path: str | Path,
    expected_queries: int,
    k: int = 10,
) -> list[list[int]]:
    """Write rankings as a portable Python list-of-lists pickle."""
    clean = [[int(image_id) for image_id in row] for row in rankings]
    if len(clean) != expected_queries:
        raise ValueError(f"Expected {expected_queries} query rows, got {len(clean)}")
    if any(len(row) != k for row in clean):
        raise ValueError(f"Every query must contain exactly {k} results")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(clean, stream)
    return clean


def save_submission_masks(
    query_paths: Sequence[Path],
    masks: Sequence[np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    """Write same-stem, 0/255 PNG masks for an ordered query collection."""
    if len(query_paths) != len(masks):
        raise ValueError("The number of query images and masks must match")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for query_path, mask in zip(query_paths, masks):
        output_path = directory / f"{query_path.stem}.png"
        binary = np.asarray(mask, dtype=np.uint8) * 255
        if not cv2.imwrite(str(output_path), binary):
            raise OSError(f"Could not write {output_path}")
        written.append(output_path)
    return written


def validate_rankings(
    result_path: str | Path,
    query_dataset: ImageDataset,
    database_ids: Sequence[int],
    k: int = 10,
) -> list[list[int]]:
    """Reload and validate a ranking pickle against the submission contract."""
    path = Path(result_path)
    with path.open("rb") as stream:
        rankings = pickle.load(stream)

    valid_database_ids = {int(image_id) for image_id in database_ids}
    if type(rankings) is not list or len(rankings) != len(query_dataset):
        raise ValueError(f"{path}: expected {len(query_dataset)} query rows")
    for row_number, row in enumerate(rankings):
        if type(row) is not list or len(row) != k:
            raise ValueError(f"{path}: row {row_number} does not have {k} results")
        if not all(type(image_id) is int for image_id in row):
            raise TypeError(f"{path}: row {row_number} contains a non-integer ID")
        if len(set(row)) != k:
            raise ValueError(f"{path}: row {row_number} contains duplicate IDs")
        unknown = set(row) - valid_database_ids
        if unknown:
            raise ValueError(f"{path}: unknown database IDs {sorted(unknown)}")
    return rankings


def validate_submission_masks(
    query_dataset: ImageDataset,
    mask_dir: str | Path,
) -> int:
    """Validate mask filenames, dimensions, readability, and binary values."""
    directory = Path(mask_dir)
    expected_names = {f"{path.stem}.png" for path in query_dataset.paths}
    actual_names = {path.name for path in directory.glob("*.png")}
    if actual_names != expected_names:
        raise ValueError(
            f"Mask filenames differ: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    for query_path in query_dataset.paths:
        image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        mask_path = directory / f"{query_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not reopen {query_path} or {mask_path}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{mask_path}: shape {mask.shape} != {image.shape[:2]}")
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise ValueError(f"{mask_path}: mask is not binary; values={sorted(values)}")
    return len(actual_names)
