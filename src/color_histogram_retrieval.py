from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import ImageDataset
from .distances import pairwise_distances
from .evaluation import evaluate_rankings


@dataclass(frozen=True)
class HistogramConfig:
    """Configuration for one global, concatenated 1D color-histogram method."""

    name: str
    color_space: str
    bins: int
    sigma: float
    metric: str
    channel_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.bins <= 0:
            raise ValueError("bins must be positive")
        if self.sigma < 0:
            raise ValueError("sigma cannot be negative")
        if any(weight < 0 for weight in self.channel_weights):
            raise ValueError("channel weights cannot be negative")
        if not abs(sum(self.channel_weights) - 1.0) < 1e-9:
            raise ValueError("channel weights must sum to 1")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "color_space": self.color_space,
            "bins": self.bins,
            "sigma": self.sigma,
            "metric": self.metric,
            "channel_weights": list(self.channel_weights),
        }


@dataclass(frozen=True)
class ColorSpace:
    conversion: int | None
    channel_names: tuple[str, ...]
    value_counts: tuple[int, ...]
    circular_channels: frozenset[int] = frozenset()


COLOR_SPACES = {
    "gray": ColorSpace(cv2.COLOR_RGB2GRAY, ("gray",), (256,)),
    "rgb": ColorSpace(None, ("R", "G", "B"), (256, 256, 256)),
    "hsv": ColorSpace(cv2.COLOR_RGB2HSV, ("H", "S", "V"), (180, 256, 256), frozenset({0})),
    "lab": ColorSpace(cv2.COLOR_RGB2LAB, ("L", "a", "b"), (256, 256, 256)),
    "ycrcb": ColorSpace(cv2.COLOR_RGB2YCrCb, ("Y", "Cr", "Cb"), (256, 256, 256)),
}


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rebin_histograms(histograms: np.ndarray, bins: int) -> np.ndarray:
    """Merge intensity values into equal-width bins without modifying the input."""
    n_values = histograms.shape[1]
    bins = min(bins, n_values)
    destination = np.arange(n_values) * bins // n_values
    return np.stack([
        np.bincount(destination, weights=row, minlength=bins)
        for row in histograms
    ])


def smooth_histograms(histograms: np.ndarray, sigma: float, circular: bool) -> np.ndarray:
    """
    Reduce sensitivity to small colour and quantization shifts.

    Similar pixel values can fall into neighbouring bins even when the underlying
    colours are nearly identical. Gaussian smoothing shares a small amount of mass
    between those bins. Hue uses circular padding because its endpoints represent
    neighbouring colours.
    """
    if sigma == 0:
        return histograms
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    mode = "wrap" if circular else "reflect"
    padded = np.pad(histograms, ((0, 0), (radius, radius)), mode=mode)
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded])


class ColorHistogramRetrieval:
    """Global 1D color-histogram retrieval with a fit/retrieve interface."""

    def __init__(self, config: HistogramConfig):
        self.config = config
        self.database: ImageDataset | None = None
        self.database_features: np.ndarray | None = None

    def _histogram_bank(self, paths: tuple[Path, ...]) -> tuple[np.ndarray, ...]:
        specification = COLOR_SPACES[self.config.color_space]
        channel_rows: list[list[np.ndarray]] = [[] for _ in specification.channel_names]

        for path in paths:
            # Keep full-resolution counts here so bin choices can be applied later.
            rgb = read_rgb(path)
            converted = rgb if specification.conversion is None else cv2.cvtColor(rgb, specification.conversion)
            if converted.ndim == 2:
                converted = converted[..., None]
            for channel, n_values in enumerate(specification.value_counts):
                counts = np.bincount(converted[..., channel].ravel(), minlength=n_values)[:n_values]
                channel_rows[channel].append(counts.astype(np.float64))

        return tuple(np.stack(rows) for rows in channel_rows)

    def _describe(self, paths: tuple[Path, ...]) -> np.ndarray:
        specification = COLOR_SPACES[self.config.color_space]
        bank = self._histogram_bank(paths)
        if len(bank) != len(self.config.channel_weights):
            raise ValueError(
                f"{self.config.color_space} requires {len(specification.channel_names)} channel weights"
            )

        parts = []
        for channel, (histograms, weight) in enumerate(zip(bank, self.config.channel_weights)):
            part = rebin_histograms(histograms, self.config.bins)
            # Make nearby colour values compare more smoothly without altering the image.
            part = smooth_histograms(
                part,
                self.config.sigma,
                channel in specification.circular_channels,
            )
            # Normalize before weighting so image resolution cannot dominate distance.
            part /= part.sum(axis=1, keepdims=True)
            parts.append(weight * part)
        # Weighted parts still form one global concatenated 1D descriptor.
        return np.concatenate(parts, axis=1)

    def fit(self, database: ImageDataset) -> "ColorHistogramRetrieval":
        self.database = database
        self.database_features = self._describe(database.paths)
        return self

    def retrieve(
        self,
        queries: ImageDataset,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.database is None or self.database_features is None:
            raise RuntimeError("Call fit(database) before retrieve(queries)")
        query_features = self._describe(queries.paths)
        distances = pairwise_distances(query_features, self.database_features, self.config.metric)
        # Every metric exposed here is a distance, so smaller values rank first.
        order = np.argsort(distances, axis=1, kind="stable")
        if top_k is not None:
            order = order[:, :top_k]
        return self.database.ids[order], np.take_along_axis(distances, order, axis=1)

    def evaluate(
        self,
        queries: ImageDataset,
        ground_truth: list[list[int]],
    ) -> tuple[dict[str, float], np.ndarray]:
        rankings, _ = self.retrieve(queries)
        return evaluate_rankings(rankings, ground_truth), rankings
