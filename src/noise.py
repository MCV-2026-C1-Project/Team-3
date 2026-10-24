""" Noise analysis and filtering used by Week 3 """

import numpy as np
import cv2
from typing import Callable
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

from .data import ImageDataset
from .color_histogram_retrieval import read_rgb


NOISE_FILTERS: dict[str, Callable] = {
    "No filter": lambda img: img,
    "Median Filter kernel 3": lambda img: cv2.medianBlur(img, 3),
    "Median Filter kernel 5": lambda img: cv2.medianBlur(img, 5),
    "Gaussian Filter σ=1": lambda img: cv2.GaussianBlur(img, (0, 0), 1),
    "Mean Filter kernel 3": lambda img: cv2.blur(img, (3, 3)),
    "Bilateral Filter": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
    "NLM h=5": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21),
    "NLM h=10": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
}


def noise_score(noisy_path, clean_path, sigma=3):
    diff = read_rgb(noisy_path).astype(np.float32) - read_rgb(clean_path).astype(np.float32)
    return (diff - cv2.GaussianBlur(diff, (0, 0), sigma)).std()  # high-frequency part only


def subset_dataset(ds: ImageDataset, mask) -> ImageDataset:
    idx = np.flatnonzero(mask)
    return ImageDataset(tuple(ds.paths[i] for i in idx), ds.ids[idx])


def split_noisy_images(noisy_dataset: ImageDataset, clean_dataset: ImageDataset, sigma=3):
    """Wrapper: Otsu split (on log10 of the noise score) into noisy / non-noisy images.

    Returns noisy_subset, clean_subset, scores, threshold.
    """
    clean_by_id = dict(zip(clean_dataset.ids.tolist(), clean_dataset.paths))
    scores = np.array([
        noise_score(path, clean_by_id[int(i)], sigma)
        for path, i in zip(noisy_dataset.paths, noisy_dataset.ids)
    ])
    threshold = 10 ** threshold_otsu(np.log10(scores + 1e-6))
    noisy_mask = scores > threshold
    return (
        subset_dataset(noisy_dataset, noisy_mask),
        subset_dataset(noisy_dataset, ~noisy_mask),
        scores,
        float(threshold),
    )


def evaluate_filters(
    noisy_dataset: ImageDataset,
    clean_dataset: ImageDataset,
    filter_functions: dict[str, Callable],
) -> dict[str, dict[str, np.ndarray | float]]:
    """Compute per-image MSE and SSIM for every filter."""
    clean_by_id = dict(zip(clean_dataset.ids.tolist(), clean_dataset.paths))
    missing_ids = sorted(set(noisy_dataset.ids.tolist()) - set(clean_by_id))
    if missing_ids:
        raise ValueError(f"No clean counterpart for image IDs: {missing_ids}")

    metrics = {}
    for filter_name, filter_function in filter_functions.items():
        mse_values = []
        ssim_values = []
        for noisy_path, image_id_value in zip(noisy_dataset.paths, noisy_dataset.ids):
            noisy = read_rgb(noisy_path)
            clean = read_rgb(clean_by_id[int(image_id_value)])
            if noisy.shape != clean.shape:
                raise ValueError(
                    f"Image shape mismatch for ID {image_id_value}: "
                    f"{noisy.shape} vs {clean.shape}"
                )

            filtered = filter_function(noisy)
            mse_values.append(np.mean((filtered.astype(np.float32) - clean) ** 2))
            ssim_values.append(
                structural_similarity(clean, filtered, channel_axis=2, data_range=255)
            )

        metrics[filter_name] = {
            "mse": np.asarray(mse_values),
            "ssim": np.asarray(ssim_values),
        }
    return metrics


def summarize_filter_metrics(filter_metrics) -> list[dict]:
    """Wrapper: one row (mean/std of MSE and SSIM) per filter."""
    return [
        {
            "Filter": name,
            "MSE mean": float(values["mse"].mean()),
            "MSE std": float(values["mse"].std()),
            "SSIM mean": float(values["ssim"].mean()),
            "SSIM std": float(values["ssim"].std()),
        }
        for name, values in filter_metrics.items()
    ]
