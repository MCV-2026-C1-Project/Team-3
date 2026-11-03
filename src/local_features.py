"""Local-feature extraction and geometric retrieval for Week 4 experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .painting_detection import find_paintings_refined
from .retrieval import order_box_points


@dataclass(frozen=True)
class LocalFeatureRetrievalConfig:
    """Frozen parameters for the selected local-feature retrieval pipeline."""

    name: str = "rectified SIFT + Lowe ratio + RANSAC + unknown rejection"
    thinness_threshold: float = 0.30
    frame_min_area: float = 0.08
    rectify: bool = True
    nfeatures: int = 3000
    ratio: float = 0.75
    ransac_threshold: float = 4.0
    shortlist: int = 30
    topk: int = 10
    unknown_inlier_threshold: float = 4.5

    def __post_init__(self) -> None:
        if not 0 < self.thinness_threshold <= 1:
            raise ValueError("thinness_threshold must be in (0, 1]")
        if not 0 < self.frame_min_area <= 1:
            raise ValueError("frame_min_area must be in (0, 1]")
        if self.nfeatures <= 0 or self.shortlist <= 0 or self.topk <= 0:
            raise ValueError("feature, shortlist and ranking sizes must be positive")
        if not 0 < self.ratio < 1:
            raise ValueError("ratio must be in (0, 1)")
        if self.ransac_threshold <= 0 or self.unknown_inlier_threshold < 0:
            raise ValueError("RANSAC and rejection thresholds must be non-negative")


WEEK4_FINAL_RETRIEVAL = LocalFeatureRetrievalConfig()


def rectify_painting(image: np.ndarray, polygon: np.ndarray | None) -> np.ndarray:
    """Warp a detected rotated rectangle into an upright painting crop."""
    if polygon is None:
        return image.copy()
    source = order_box_points(polygon)
    top_left, top_right, bottom_right, bottom_left = source
    width = max(
        int(round(np.linalg.norm(top_right - top_left))),
        int(round(np.linalg.norm(bottom_right - bottom_left))),
        2,
    )
    height = max(
        int(round(np.linalg.norm(bottom_left - top_left))),
        int(round(np.linalg.norm(bottom_right - top_right))),
        2,
    )
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def prepare_query_parts(
    images: Iterable[np.ndarray],
    config: LocalFeatureRetrievalConfig = WEEK4_FINAL_RETRIEVAL,
) -> tuple[list[list[dict[str, object]]], list[np.ndarray], list[dict[str, object]]]:
    """Detect, order and optionally rectify every painting in every query."""
    prepared_queries: list[list[dict[str, object]]] = []
    masks: list[np.ndarray] = []
    debug_rows: list[dict[str, object]] = []
    for image in images:
        parts, debug = find_paintings_refined(
            image,
            thinness_threshold=config.thinness_threshold,
            frame_min_area=config.frame_min_area,
        )
        prepared: list[dict[str, object]] = []
        for part in parts:
            if config.rectify and part["polygon"] is not None:
                crop = rectify_painting(image, part["polygon"])
                mask = None
            else:
                crop = part["crop"]
                mask = part["mask"]
            prepared.append({**part, "crop": crop, "mask": mask})
        prepared_queries.append(prepared)
        masks.append(np.asarray(debug["combined"], dtype=bool))
        debug_rows.append(debug)
    return prepared_queries, masks, debug_rows


def sift_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    nfeatures: int = 3000,
) -> dict[str, np.ndarray | None]:
    """Extract SIFT point coordinates and descriptors from one image."""
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    points = np.asarray(
        [keypoint.pt for keypoint in keypoints],
        dtype=np.float32,
    ).reshape(-1, 2)
    return {"points": points, "descriptors": descriptors}


def describe_parts(
    parts_per_query: Iterable[list[dict[str, object]]],
    nfeatures: int = 3000,
) -> list[list[dict[str, np.ndarray | None]]]:
    """Extract SIFT features for each detected painting, preserving query order."""
    return [
        [
            sift_features(part["crop"], part["mask"], nfeatures=nfeatures)
            for part in parts
        ]
        for parts in parts_per_query
    ]


def describe_database(
    images: Iterable[np.ndarray],
    identifiers: Iterable[int],
    nfeatures: int = 3000,
) -> dict[int, dict[str, np.ndarray | None]]:
    """Extract whole-image SIFT features for the reference database."""
    return {
        int(identifier): sift_features(image, nfeatures=nfeatures)
        for identifier, image in zip(identifiers, images)
    }


def ratio_matches(
    query_descriptors: np.ndarray | None,
    reference_descriptors: np.ndarray | None,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return query/reference indices accepted by Lowe's ratio test."""
    if (
        query_descriptors is None
        or reference_descriptors is None
        or len(query_descriptors) < 2
        or len(reference_descriptors) < 2
    ):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = [
        pair
        for pair in matcher.knnMatch(
            query_descriptors,
            reference_descriptors,
            k=2,
        )
        if len(pair) == 2
    ]
    accepted = [
        pair[0]
        for pair in pairs
        if pair[0].distance < ratio * pair[1].distance
    ]
    return (
        np.asarray([match.queryIdx for match in accepted], dtype=int),
        np.asarray([match.trainIdx for match in accepted], dtype=int),
    )


def plausible_homography(
    homography: np.ndarray | None,
    max_anisotropy: float = 4.0,
) -> bool:
    """Reject mirrored, collapsed, or extremely stretched transformations."""
    if homography is None or abs(homography[2, 2]) < 1e-9:
        return False
    linear = (homography / homography[2, 2])[:2, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return bool(
        np.linalg.det(linear) > 0
        and singular_values[1] > 1e-3
        and singular_values[0] / singular_values[1] < max_anisotropy
    )


def ransac_inliers(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
    threshold: float = 4.0,
) -> int:
    """Count geometrically consistent tentative matches."""
    if len(query_indices) < 4:
        return 0
    homography, inlier_mask = cv2.findHomography(
        query_points[query_indices],
        reference_points[reference_indices],
        cv2.RANSAC,
        threshold,
    )
    if inlier_mask is None or not plausible_homography(homography):
        return 0
    return int(inlier_mask.sum())


def rank_references(
    query: dict[str, np.ndarray | None],
    database: dict[int, dict[str, np.ndarray | None]],
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    shortlist: int = 30,
    topk: int = 10,
) -> dict[str, object]:
    """Rank references by RANSAC inliers, using ratio matches for shortlisting."""
    matches = {
        identifier: ratio_matches(
            query["descriptors"],
            reference["descriptors"],
            ratio=ratio,
        )
        for identifier, reference in database.items()
    }
    candidates = sorted(
        matches,
        key=lambda identifier: len(matches[identifier][0]),
        reverse=True,
    )[:shortlist]
    inliers = {
        identifier: ransac_inliers(
            query["points"],
            database[identifier]["points"],
            *matches[identifier],
            threshold=ransac_threshold,
        )
        for identifier in candidates
    }
    ranked = sorted(
        database,
        key=lambda identifier: (
            inliers.get(identifier, -1),
            len(matches[identifier][0]),
        ),
        reverse=True,
    )[:topk]
    scores = sorted(inliers.values(), reverse=True)
    return {
        "ranked": [int(identifier) for identifier in ranked],
        "best_inliers": int(scores[0]) if scores else 0,
        "second_inliers": int(scores[1]) if len(scores) > 1 else 0,
    }


def retrieve_parts(
    query_features: list[list[dict[str, np.ndarray | None]]],
    database_features: dict[int, dict[str, np.ndarray | None]],
    **ranking_parameters,
) -> list[list[dict[str, object]]]:
    """Retrieve one ranking per detected painting, preserving query structure."""
    return [
        [
            rank_references(part, database_features, **ranking_parameters)
            for part in query
        ]
        for query in query_features
    ]


def retrieve_query_images(
    images: Iterable[np.ndarray],
    database_features: dict[int, dict[str, np.ndarray | None]],
    config: LocalFeatureRetrievalConfig = WEEK4_FINAL_RETRIEVAL,
) -> tuple[
    list[list[list[int]]],
    list[list[dict[str, object]]],
    list[list[dict[str, object]]],
    list[np.ndarray],
]:
    """Run the frozen Week 4 query pipeline and apply unknown rejection.

    Returns submission rankings, raw retrieval records, prepared painting
    parts, and full-query foreground masks in dataset order.
    """
    parts, masks, _ = prepare_query_parts(images, config)
    query_features = describe_parts(parts, nfeatures=config.nfeatures)
    results = retrieve_parts(
        query_features,
        database_features,
        ratio=config.ratio,
        ransac_threshold=config.ransac_threshold,
        shortlist=config.shortlist,
        topk=config.topk,
    )
    rankings = [
        [
            [-1]
            if int(result["best_inliers"]) < config.unknown_inlier_threshold
            else [int(identifier) for identifier in result["ranked"]]
            for result in query
        ]
        for query in results
    ]
    return rankings, results, parts, masks
