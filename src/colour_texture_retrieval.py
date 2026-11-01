"""Reusable colour-texture retrieval pipeline selected for Week 3.

The final Task 4 method combines the best QSD2-W3 texture descriptor with a
global Lab descriptor. Each distance vector is normalised independently for
every query painting before fusion so the descriptor families contribute on a
comparable scale.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .spatial_histograms import SpatialHistogramConfig, descriptor as lab_descriptor
from .texture_histograms import (
    WEEK3_FINAL_TEXTURE_CONFIG,
    TextureDescriptorConfig,
    compute_distances,
    describe_images,
)


@dataclass(frozen=True)
class ColourTextureFusionConfig:
    """Configuration for the selected Week 3 Task 4 retrieval method."""

    name: str = "60% LBP radius 2 + 40% global Lab"
    texture: TextureDescriptorConfig = WEEK3_FINAL_TEXTURE_CONFIG
    colour: SpatialHistogramConfig = field(
        default_factory=lambda: SpatialHistogramConfig(
            levels=(1,),
            bins=48,
            sigma=0.5,
            channel_weights=(0.125, 0.5, 0.375),
        )
    )
    colour_distance: str = "hellinger"
    texture_weight: float = 0.60
    colour_weight: float = 0.40

    def __post_init__(self) -> None:
        if self.texture_weight < 0 or self.colour_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if not np.isclose(self.texture_weight + self.colour_weight, 1.0):
            raise ValueError("fusion weights must sum to one")


WEEK3_FINAL_RETRIEVAL = ColourTextureFusionConfig()


def describe_colour_images(
    images: Sequence[np.ndarray],
    config: SpatialHistogramConfig = WEEK3_FINAL_RETRIEVAL.colour,
) -> np.ndarray:
    """Compute global Lab descriptors for an ordered BGR image collection."""
    return np.asarray(
        [
            lab_descriptor(
                image,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
                hist_type=config.hist_type,
                level_weights=config.level_weights,
            )
            for image in images
        ],
        dtype=np.float32,
    )


def minmax_rows(values: np.ndarray) -> np.ndarray:
    """Min-max normalise every query's distances over the database."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("distance matrix must be two-dimensional")
    minima = values.min(axis=1, keepdims=True)
    ranges = values.max(axis=1, keepdims=True) - minima
    return np.divide(
        values - minima,
        ranges,
        out=np.zeros_like(values),
        where=ranges > 1e-12,
    )


def fuse_distances(
    texture_distances: np.ndarray,
    colour_distances: np.ndarray,
    config: ColourTextureFusionConfig = WEEK3_FINAL_RETRIEVAL,
) -> np.ndarray:
    """Normalise and fuse aligned texture and colour distance matrices."""
    if texture_distances.shape != colour_distances.shape:
        raise ValueError("texture and colour distance matrices must have the same shape")
    return (
        config.texture_weight * minmax_rows(texture_distances)
        + config.colour_weight * minmax_rows(colour_distances)
    )


def final_retrieval_distances(
    query_images: Sequence[np.ndarray],
    database_texture_features: np.ndarray,
    database_colour_features: np.ndarray,
    config: ColourTextureFusionConfig = WEEK3_FINAL_RETRIEVAL,
) -> np.ndarray:
    """Return the selected fused query-by-database distance matrix."""
    query_texture = describe_images(query_images, config.texture)
    query_colour = describe_colour_images(query_images, config.colour)
    texture_distances = compute_distances(
        query_texture, database_texture_features, config.texture.distance
    )
    colour_distances = compute_distances(
        query_colour, database_colour_features, config.colour_distance
    )
    return fuse_distances(texture_distances, colour_distances, config)
