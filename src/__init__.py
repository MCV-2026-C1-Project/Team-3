"""Shared data, descriptor, retrieval, evaluation, and submission utilities."""

from .color_histogram_retrieval import ColorHistogramRetrieval, HistogramConfig
from .data import (
    ImageDataset,
    load_binary_masks,
    load_ground_truth,
    load_image_dataset,
    read_bgr,
    read_images,
)
from .evaluation import evaluate_masks, evaluate_rankings, map_at_k
from .retrieval import hellinger
from .segmentation import foreground_mask, predict_foreground_masks
from .spatial_histograms import (
    SpatialHistogramConfig,
    descriptor,
    descriptor_masked,
    simple_segment_and_histogram,
)
from .submission import (
    save_rankings,
    save_submission_masks,
    validate_rankings,
    validate_submission_masks,
)

__all__ = [
    "ColorHistogramRetrieval",
    "HistogramConfig",
    "ImageDataset",
    "SpatialHistogramConfig",
    "descriptor",
    "descriptor_masked",
    "evaluate_masks",
    "evaluate_rankings",
    "foreground_mask",
    "hellinger",
    "load_binary_masks",
    "load_ground_truth",
    "load_image_dataset",
    "map_at_k",
    "predict_foreground_masks",
    "read_bgr",
    "read_images",
    "retrieve_from_features",
    "save_rankings",
    "save_submission_masks",
    "simple_segment_and_histogram",
    "validate_rankings",
    "validate_submission_masks",
]
