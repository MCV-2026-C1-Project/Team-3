"""Colour-only background removal used by Week 2 Tasks 3-6."""

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def border_strips(height: int, width: int, fraction: float) -> np.ndarray:
    """Return a mask covering an outer strip on all four image borders."""
    band_width = max(3, int(round(fraction * min(height, width))))
    border = np.zeros((height, width), dtype=bool)
    border[:band_width] = True
    border[-band_width:] = True
    border[:, :band_width] = True
    border[:, -band_width:] = True
    return border


def background_distance(
    lab: np.ndarray,
    border: np.ndarray,
    k: int = 3,
    min_share: float = 0.25,
    w_lab: tuple[float, float, float] = (0.5, 1.0, 1.0),
    sigma_floor: tuple[float, float, float] = (2.5, 1.5, 1.5),
) -> np.ndarray:
    """Fit border-colour clusters and return the nearest standardized distance."""
    weights = np.asarray(w_lab, dtype=np.float32)
    border_pixels = lab[border]
    cluster_count = min(k, len(border_pixels))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(
        (border_pixels * weights).astype(np.float32),
        cluster_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    shares = np.bincount(labels, minlength=cluster_count) / len(border_pixels)
    kept = [
        cluster
        for cluster in range(cluster_count)
        if shares[cluster] >= min_share or cluster == int(shares.argmax())
    ]

    flat = lab.reshape(-1, 3)
    distance = np.full(flat.shape[0], np.inf, dtype=np.float32)
    for cluster in kept:
        points = border_pixels[labels == cluster]
        mean = points.mean(axis=0)
        scale = np.maximum(points.std(axis=0), sigma_floor)
        cluster_distance = np.sqrt((((flat - mean) / scale * weights) ** 2).sum(axis=1))
        distance = np.minimum(distance, cluster_distance)
    return distance.reshape(lab.shape[:2])


def adaptive_threshold(
    distance: np.ndarray,
    threshold_range: tuple[float, float] = (2.0, 6.0),
) -> float:
    """Choose a background-distance threshold using clipped log-space Otsu."""
    maximum = max(float(distance.max()), 1.0)
    scaled = np.clip(np.log1p(distance) / np.log1p(maximum) * 255, 0, 255).astype(np.uint8)
    threshold_u8, _ = cv2.threshold(
        scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = np.expm1(threshold_u8 / 255 * np.log1p(maximum))
    return float(np.clip(threshold, *threshold_range))


def foreground_mask(
    image: np.ndarray,
    work_size: int = 400,
    border_frac: float = 0.04,
    k: int = 3,
    min_share: float = 0.25,
    w_lab: tuple[float, float, float] = (0.5, 1.0, 1.0),
    sigma_floor: tuple[float, float, float] = (2.5, 1.5, 1.5),
    tau_range: tuple[float, float] = (2.0, 6.0),
    morph_frac: float = 0.012,
    min_area: float = 0.05,
    keep: str = "all",
    rectangular: bool = False,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Estimate a painting mask from border colour without an object detector."""
    original_height, original_width = image.shape[:2]
    scale = min(1.0, work_size / max(original_height, original_width))
    small = (
        cv2.resize(
            image,
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else image
    )
    small = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]

    border = border_strips(height, width, border_frac)
    distance = background_distance(lab, border, k, min_share, w_lab, sigma_floor)
    threshold = adaptive_threshold(distance, tau_range)

    background = (distance < threshold).astype(np.uint8)
    _, components = cv2.connectedComponents(background, connectivity=4)
    touching = np.unique(components[border & (background > 0)])
    connected_background = np.isin(components, touching[touching > 0])
    foreground = (~connected_background).astype(np.uint8)

    kernel_size = max(3, int(round(morph_frac * min(height, width))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = binary_fill_holes(foreground).astype(np.uint8)

    _, components, statistics, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    areas = statistics[1:, cv2.CC_STAT_AREA]
    kept_labels = [
        index + 1
        for index, area in enumerate(areas)
        if area >= min_area * height * width
    ]
    if keep == "largest" and len(areas):
        kept_labels = [1 + int(areas.argmax())]
    output = np.isin(components, kept_labels)

    if rectangular and output.any():
        rectangle = np.zeros_like(output)
        for label in kept_labels:
            y, x = np.where(components == label)
            rectangle[y.min() : y.max() + 1, x.min() : x.max() + 1] = True
        output = rectangle

    foreground_fraction = output.mean()
    if foreground_fraction < 0.05 or foreground_fraction > 0.98:
        output = np.ones((height, width), dtype=bool)

    mask = cv2.resize(
        output.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    ) > 0.5
    if return_debug:
        return mask, {"distance": distance, "threshold": threshold, "border": border}
    return mask


def predict_foreground_masks(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Apply the default colour-only segmenter to an ordered image collection."""
    return [foreground_mask(image) for image in images]
