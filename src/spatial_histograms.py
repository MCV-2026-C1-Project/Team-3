"""Spatial Lab histogram descriptors shared by Week 2 tasks."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


_HIST_TYPES = ("1d", "2d", "3d")


@dataclass(frozen=True)
class SpatialHistogramConfig:
    """Configuration for a concatenated block or pyramid Lab descriptor."""

    levels: tuple[int, ...] = (5,)
    bins: int = 48
    sigma: float = 0.5
    channel_weights: tuple[float, float, float] = (0.125, 0.5, 0.375)
    hist_type: str = "1d"
    level_weights: tuple[float, ...] | str | None = None

    def __post_init__(self) -> None:
        if not self.levels or any(level <= 0 for level in self.levels):
            raise ValueError("levels must contain positive grid sizes")
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if self.sigma < 0:
            raise ValueError("sigma cannot be negative")
        if len(self.channel_weights) != 3 or any(weight < 0 for weight in self.channel_weights):
            raise ValueError("channel_weights must contain three non-negative values")
        if not np.isclose(sum(self.channel_weights), 1.0):
            raise ValueError("channel_weights must sum to one")
        if self.hist_type not in _HIST_TYPES:
            raise ValueError(f"hist_type must be one of {_HIST_TYPES}")
        if isinstance(self.level_weights, str):
            if self.level_weights != "pyramid":
                raise ValueError("level_weights must be None, 'pyramid' or a sequence")
        elif self.level_weights is not None and (
            len(self.level_weights) != len(self.levels)
            or any(weight < 0 for weight in self.level_weights)
            or sum(self.level_weights) <= 0
        ):
            raise ValueError("level_weights needs one non-negative value per level")


WEEK2_BLOCK_CONFIG = SpatialHistogramConfig()


def _grid_edges(length: int, level: int) -> np.ndarray:
    return np.linspace(0, length, level + 1, dtype=int)


def _resolve_level_weights(level_weights, n_levels: int) -> np.ndarray:
    """Return pyramid level weights that sum to one."""
    if level_weights is None:
        w = np.ones(n_levels, dtype=np.float64)  # equal weights
    elif isinstance(level_weights, str):
        if level_weights != "pyramid":
            raise ValueError("level_weights must be None, 'pyramid' or a sequence")
        # Lazebnik et al.: coarsest level 1/2^L, level i (i>=1) 1/2^(L-i+1)
        L = n_levels - 1
        w = np.array([1 / 2**L] + [1 / 2 ** (L - i + 1) for i in range(1, n_levels)])
    else:
        w = np.asarray(level_weights, dtype=np.float64)
        if w.shape != (n_levels,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("level_weights needs one non-negative value per level")
    return w / w.sum()


def simple_segment_and_histogram(
    image: np.ndarray,
    level: int | Sequence[int],
    bins: int = 48,
    plot: bool = False,
    normalize: bool = True,
    weights: Sequence[float] | None = None,
    hist_type: str = "1d",
    level_weights: Sequence[float] | str | None = None,
) -> np.ndarray:
    """Split a BGR image into grid blocks and concatenate per-block Lab histograms.

    Parameters
    ----------
    level : int or sequence of int
        Grid size(s). An int gives a single block representation; a sequence
        (e.g. ``(1, 2, 4)``) gives a spatial pyramid whose per-level vectors are
        concatenated, coarsest to finest in the order given.
    hist_type : {"1d", "2d", "3d"}
        "1d": three independent L, a, b histograms (``3 * bins`` per block).
        "2d": joint a-b histogram (``bins**2`` per block).
        "3d": joint L-a-b histogram (``bins**3`` per block). Use small ``bins``
        (8-16) for 2d/3d, since 48 bins in 3D is 110,592 values per block.
    weights : per-channel Lab weights, only valid for ``hist_type="1d"``.
    level_weights : None (equal), "pyramid" (Lazebnik weights, finer levels
        weigh more) or one non-negative value per level. Normalised to sum to one.

    Each level is laid out in row-major block order, so the output length is
    ``sum(level_i**2 * features_per_block)``. A single level with the defaults
    reproduces the original output.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if hist_type not in _HIST_TYPES:
        raise ValueError(f"hist_type must be one of {_HIST_TYPES}")
    levels = (int(level),) if np.isscalar(level) else tuple(int(l) for l in level)
    if not levels or any(l <= 0 for l in levels):
        raise ValueError("level must contain positive grid sizes")
    if weights is not None and hist_type != "1d":
        raise ValueError("per-channel weights are only defined for hist_type='1d'")
    if plot and hist_type != "1d":
        raise ValueError("plot is only supported for hist_type='1d'")

    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lw = _resolve_level_weights(level_weights, len(levels))
    channel_weights = None
    if weights is not None:
        channel_weights = np.asarray(weights, dtype=np.float32)
        if channel_weights.shape != (3,):
            raise ValueError("weights must contain one value for each Lab channel")

    block_shape = {"1d": (3, bins), "2d": (bins, bins), "3d": (bins, bins, bins)}[hist_type]
    parts: list[np.ndarray] = []

    for lvl, level_weight in zip(levels, lw):
        y_edges = _grid_edges(height, lvl)
        x_edges = _grid_edges(width, lvl)
        histograms = np.empty((lvl * lvl, *block_shape), dtype=np.float32)

        for row in range(lvl):
            for column in range(lvl):
                index = row * lvl + column
                block = np.ascontiguousarray(
                    lab[
                        y_edges[row] : y_edges[row + 1],
                        x_edges[column] : x_edges[column + 1],
                    ]
                )
                if hist_type == "1d":
                    block_hist = np.stack(
                        [
                            cv2.calcHist([block], [c], None, [bins], [0, 256]).ravel()
                            for c in range(3)
                        ]
                    )
                elif hist_type == "2d":
                    block_hist = cv2.calcHist(
                        [block], [1, 2], None, [bins, bins], [0, 256, 0, 256]
                    )
                else:
                    block_hist = cv2.calcHist(
                        [block], [0, 1, 2], None, [bins] * 3, [0, 256] * 3
                    )
                if normalize:
                    if hist_type == "1d":  # each channel sums to one, as before
                        block_hist = block_hist / np.maximum(
                            block_hist.sum(axis=1, keepdims=True), 1e-12
                        )
                    else:  # joint histogram sums to one
                        block_hist = block_hist / max(float(block_hist.sum()), 1e-12)
                histograms[index] = block_hist

        if channel_weights is not None:
            histograms *= channel_weights[None, :, None]

        if plot:
            _plot_block_histograms(image, histograms, y_edges, x_edges, normalize)
        parts.append(histograms.ravel() * np.float32(level_weight))

    return np.concatenate(parts)


def _plot_block_histograms(
    image: np.ndarray,
    histograms: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    normalize: bool,
) -> None:
    """Render the Task 1 grid and its per-block histograms."""
    import matplotlib.pyplot as plt

    level = len(y_edges) - 1
    bins = histograms.shape[2]
    figure = plt.figure(figsize=(4 * (level + 1), 2.8 * level), layout="constrained")
    outer = figure.add_gridspec(1, 2, width_ratios=[1, level])
    image_axis = figure.add_subplot(outer[0])
    image_axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for x in x_edges[1:-1]:
        image_axis.axvline(x - 0.5, color="red", linewidth=1.5)
    for y in y_edges[1:-1]:
        image_axis.axhline(y - 0.5, color="red", linewidth=1.5)
    for row in range(level):
        for column in range(level):
            center_x = (x_edges[column] + x_edges[column + 1]) / 2
            center_y = (y_edges[row] + y_edges[row + 1]) / 2
            image_axis.text(
                center_x,
                center_y,
                f"({row},{column})",
                color="white",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1.5},
            )
    image_axis.set_title(f"Segmentation ({level}x{level} blocks)")
    image_axis.axis("off")

    inner = outer[1].subgridspec(level, level)
    channel_names = ("L", "a", "b")
    channel_colors = ("black", "green", "blue")
    for row in range(level):
        for column in range(level):
            index = row * level + column
            axis = figure.add_subplot(inner[row, column])
            for channel, (name, color) in enumerate(zip(channel_names, channel_colors)):
                axis.plot(
                    np.arange(channel * bins, (channel + 1) * bins),
                    histograms[index, channel],
                    color=color,
                    linewidth=1.3,
                    label=name,
                )
            axis.axvline(bins - 0.5, color="gray", linestyle="--", linewidth=0.8)
            axis.axvline(2 * bins - 0.5, color="gray", linestyle="--", linewidth=0.8)
            axis.set_title(f"Block ({row}, {column})", fontsize=10)
            axis.set_xlim(0, 3 * bins - 1)
            axis.set_xticks([bins / 2 - 0.5, 1.5 * bins - 0.5, 2.5 * bins - 0.5])
            axis.set_xticklabels(channel_names)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Frequency" if normalize else "Pixel count")
            if row == 0 and column == level - 1:
                axis.legend(fontsize=8, loc="upper right")
    plt.show()


def descriptor(
    image: np.ndarray,
    levels: tuple[int, ...] = (1,),
    bins: int = 48,
    sigma: float = 0.5,
    weights: tuple[float, float, float] = (0.125, 0.5, 0.375),
    hist_type: str = "1d",
    level_weights: Sequence[float] | str | None = None,
) -> np.ndarray:
    """Compute a complete-image block or pyramid descriptor."""
    if hist_type != "1d":
        features = simple_segment_and_histogram(
            image,
            levels,
            bins=bins,
            hist_type=hist_type,
            level_weights=level_weights,
        )
        return features / max(float(features.sum()), 1e-12)

    parts: list[np.ndarray] = []
    resolved_level_weights = _resolve_level_weights(level_weights, len(levels))
    for level, level_weight in zip(levels, resolved_level_weights):
        histograms = simple_segment_and_histogram(image, level, bins=bins).reshape(
            level * level, 3, bins
        )
        if sigma > 0:
            histograms = gaussian_filter1d(histograms, sigma, axis=2, mode="nearest")
            histograms /= np.maximum(histograms.sum(axis=2, keepdims=True), 1e-12)
        histograms *= np.asarray(weights, dtype=np.float32)[None, :, None]
        parts.append(histograms.ravel() * np.float32(level_weight))
    return np.concatenate(parts)


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image and foreground mask to the smallest foreground box."""
    if image.shape[:2] != mask.shape:
        raise ValueError("image and mask dimensions must match")
    y, x = np.where(mask)
    if len(y) == 0:
        raise ValueError("cannot crop to an empty mask")
    y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def descriptor_masked(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    levels: tuple[int, ...] = (5,),
    bins: int = 48,
    sigma: float = 0.5,
    weights: tuple[float, float, float] = (0.125, 0.5, 0.375),
    crop: bool = True,
    hist_type: str = "1d",
    level_weights: Sequence[float] | str | None = None,
) -> np.ndarray:
    """Compute the block Lab descriptor using only foreground pixels.

    Each block is weighted by its foreground-pixel count. Cropping aligns the
    spatial grid with the predicted painting rather than the surrounding wall.
    """
    if mask is None or not np.asarray(mask).any():
        mask = np.ones(image.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if crop:
        image, mask = crop_to_mask(image, mask)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    height, width = mask_u8.shape
    if hist_type not in _HIST_TYPES:
        raise ValueError(f"hist_type must be one of {_HIST_TYPES}")
    if hist_type != "1d" and weights != (0.125, 0.5, 0.375):
        raise ValueError("per-channel weights are only defined for hist_type='1d'")
    resolved_level_weights = _resolve_level_weights(level_weights, len(levels))
    channel_weights = np.asarray(weights, dtype=np.float32)
    parts: list[np.ndarray] = []

    for level, level_weight in zip(levels, resolved_level_weights):
        y_edges = _grid_edges(height, level)
        x_edges = _grid_edges(width, level)
        block_shape = {
            "1d": (3, bins),
            "2d": (bins, bins),
            "3d": (bins, bins, bins),
        }[hist_type]
        histograms = np.zeros((level * level, *block_shape), dtype=np.float32)
        foreground_area = np.zeros(level * level, dtype=np.float32)

        for row in range(level):
            for column in range(level):
                index = row * level + column
                ys = slice(y_edges[row], y_edges[row + 1])
                xs = slice(x_edges[column], x_edges[column + 1])
                block_mask = np.ascontiguousarray(mask_u8[ys, xs])
                foreground_area[index] = block_mask.sum()
                if foreground_area[index] == 0:
                    continue
                block = np.ascontiguousarray(lab[ys, xs])
                if hist_type == "1d":
                    for channel in range(3):
                        histogram = cv2.calcHist(
                            [block], [channel], block_mask, [bins], [0, 256]
                        ).ravel()
                        histograms[index, channel] = histogram / max(
                            float(histogram.sum()), 1e-12
                        )
                elif hist_type == "2d":
                    histogram = cv2.calcHist(
                        [block],
                        [1, 2],
                        block_mask,
                        [bins, bins],
                        [0, 256, 0, 256],
                    )
                    histograms[index] = histogram / max(float(histogram.sum()), 1e-12)
                else:
                    histogram = cv2.calcHist(
                        [block],
                        [0, 1, 2],
                        block_mask,
                        [bins] * 3,
                        [0, 256] * 3,
                    )
                    histograms[index] = histogram / max(float(histogram.sum()), 1e-12)

        if sigma > 0 and hist_type == "1d":
            histograms = gaussian_filter1d(histograms, sigma, axis=2, mode="nearest")
            histograms /= np.maximum(histograms.sum(axis=2, keepdims=True), 1e-12)
        histograms *= (foreground_area / max(float(foreground_area.sum()), 1e-12))[
            (...,) + (None,) * len(block_shape)
        ]
        if hist_type == "1d":
            histograms *= channel_weights[None, :, None]
        parts.append(histograms.ravel() * np.float32(level_weight))
    return np.concatenate(parts)


def describe_images(
    images: Sequence[np.ndarray],
    config: SpatialHistogramConfig = WEEK2_BLOCK_CONFIG,
    masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Describe an ordered image collection, optionally with foreground masks."""
    if masks is not None and len(images) != len(masks):
        raise ValueError("images and masks must contain the same number of elements")
    if masks is None:
        return np.asarray(
            [
                descriptor(
                    image,
                    levels=config.levels,
                    bins=config.bins,
                    sigma=config.sigma,
                    weights=config.channel_weights,
                    hist_type=config.hist_type,
                    level_weights=config.level_weights,
                )
                for image in images
            ]
        )
    return np.asarray(
        [
            descriptor_masked(
                image,
                mask,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
                hist_type=config.hist_type,
                level_weights=config.level_weights,
            )
            for image, mask in zip(images, masks)
        ]
    )
