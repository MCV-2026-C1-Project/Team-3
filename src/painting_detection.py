"""Painting-region detection shared by the Week 4 retrieval experiments."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .segmentation import foreground_mask


def edge_foreground(
    image: np.ndarray,
    work_size: int = 400,
    close_frac: float = 0.025,
) -> np.ndarray:
    """Return a filled foreground proposal derived from image edges."""
    height, width = image.shape[:2]
    scale = min(1.0, work_size / max(height, width))
    small = (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else image
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    edges = cv2.Canny(
        gray,
        int(max(0, 0.66 * median)),
        int(min(255, 1.33 * median)),
    )

    kernel_size = max(3, int(round(close_frac * min(gray.shape))) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    filled = binary_fill_holes(closed > 0).astype(np.uint8)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    return (
        cv2.resize(
            opened.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        > 0.5
    )


def _boxes_near(
    box_a: np.ndarray,
    box_b: np.ndarray,
    gap: float,
) -> bool:
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    return not (
        xa + wa + gap < xb
        or xb + wb + gap < xa
        or ya + ha + gap < yb
        or yb + hb + gap < ya
    )


def _polygon_aspect_ratio(polygon: np.ndarray) -> float:
    """Return short-side / long-side for a four-point rotated rectangle."""
    edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    return float(edges.min() / max(edges.max(), 1e-12))


def _part_from_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Build a detector part and its full-image mask from a polygon."""
    height, width = image.shape[:2]
    filled = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(filled, np.round(polygon).astype(np.int32), 1)
    rows, columns = np.nonzero(filled)
    x0, x1 = int(columns.min()), int(columns.max() + 1)
    y0, y1 = int(rows.min()), int(rows.max() + 1)
    return (
        {
            "bbox": (x0, y0, x1, y1),
            "polygon": polygon,
            "crop": image[y0:y1, x0:x1],
            "mask": filled[y0:y1, x0:x1] * 255,
        },
        filled.astype(bool),
    )


def _reading_order(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    """Use left-to-right order, or top-to-bottom for strongly overlapping columns."""
    if len(parts) < 2:
        return parts

    geometry = []
    for part in parts:
        x0, y0, x1, y1 = part["bbox"]
        geometry.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0))

    horizontal_separation = abs(geometry[0][0] - geometry[1][0])
    if horizontal_separation < 0.5 * min(geometry[0][2], geometry[1][2]):
        return [
            part
            for _, part in sorted(
                zip(geometry, parts),
                key=lambda item: item[0][1],
            )
        ]
    return [
        part
        for _, part in sorted(
            zip(geometry, parts),
            key=lambda item: item[0][0],
        )
    ]


def _frame_candidates(
    image: np.ndarray,
    work_size: int = 800,
) -> list[dict[str, object]]:
    """Return non-duplicate rectangular frame candidates from image edges."""
    height, width = image.shape[:2]
    scale = min(1.0, work_size / max(height, width))
    small = (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else image
    )
    small_height, small_width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[dict[str, object]] = []
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rectangle
        rectangle_area = rect_width * rect_height
        if rectangle_area <= 0:
            continue

        area_fraction = rectangle_area / (small_height * small_width)
        short_side, long_side = sorted((rect_width, rect_height))
        aspect_ratio = short_side / max(long_side, 1e-12)
        rectangularity = cv2.contourArea(contour) / rectangle_area
        if not (
            0.025 <= area_fraction <= 0.85
            and aspect_ratio >= 0.25
            and rectangularity >= 0.45
        ):
            continue

        polygon = cv2.boxPoints(rectangle) / scale
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
        candidates.append(
            {
                "polygon": polygon,
                "mask": mask.astype(bool),
                "area_fraction": float(area_fraction),
            }
        )

    candidates.sort(
        key=lambda candidate: float(candidate["area_fraction"]),
        reverse=True,
    )
    kept: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_mask = candidate["mask"]
        duplicate = False
        for other in kept:
            other_mask = other["mask"]
            intersection = int(np.count_nonzero(candidate_mask & other_mask))
            union = int(np.count_nonzero(candidate_mask | other_mask))
            smaller = min(
                int(np.count_nonzero(candidate_mask)),
                int(np.count_nonzero(other_mask)),
            )
            if (
                intersection / max(union, 1) > 0.55
                or intersection / max(smaller, 1) > 0.82
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def find_paintings(
    image: np.ndarray,
    max_paintings: int = 2,
    component_min_area: float = 0.005,
    painting_min_area: float = 0.02,
    merge_gap_fraction: float = 0.04,
    use_edges: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Detect up to two painting regions and order them from left to right.

    Each returned part contains its bounding box, rotated polygon, crop, and
    crop-local mask. The debug dictionary exposes the proposal masks and whether
    the whole-image fallback was required.
    """
    height, width = image.shape[:2]

    colour = foreground_mask(
        image,
        min_area=component_min_area,
        keep="all",
    )
    colour_used = bool(colour.mean() <= 0.98)
    if not colour_used:
        colour = np.zeros_like(colour)

    edge = edge_foreground(image) if use_edges else np.zeros_like(colour)
    edge_used = bool(use_edges and edge.mean() <= 0.90)
    if not edge_used:
        edge = np.zeros_like(colour)

    proposal = colour | edge
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        proposal.astype(np.uint8),
        connectivity=8,
    )
    component_ids = [
        component_id
        for component_id in range(1, count)
        if statistics[component_id, cv2.CC_STAT_AREA]
        >= component_min_area * height * width
    ]

    parent = {component_id: component_id for component_id in component_ids}

    def root(component_id: int) -> int:
        while parent[component_id] != component_id:
            parent[component_id] = parent[parent[component_id]]
            component_id = parent[component_id]
        return component_id

    gap = merge_gap_fraction * min(height, width)
    for position, component_a in enumerate(component_ids):
        for component_b in component_ids[position + 1 :]:
            if _boxes_near(
                statistics[component_a, :4],
                statistics[component_b, :4],
                gap,
            ):
                parent[root(component_b)] = root(component_a)

    groups: dict[int, list[int]] = {}
    for component_id in component_ids:
        groups.setdefault(root(component_id), []).append(component_id)

    regions: list[tuple[int, np.ndarray]] = []
    for members in groups.values():
        rows, columns = np.nonzero(np.isin(labels, members))
        if len(rows) < painting_min_area * height * width:
            continue
        points = np.column_stack((columns, rows)).astype(np.float32)
        polygon = cv2.boxPoints(cv2.minAreaRect(points))
        regions.append((len(rows), polygon))

    regions = sorted(regions, key=lambda region: region[0], reverse=True)[
        :max_paintings
    ]
    regions.sort(key=lambda region: float(region[1][:, 0].mean()))

    parts: list[dict[str, object]] = []
    combined = np.zeros((height, width), dtype=bool)
    for _, polygon in regions:
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(filled, np.round(polygon).astype(np.int32), 1)
        rows, columns = np.nonzero(filled)
        if len(columns) == 0:
            continue
        x0, x1 = int(columns.min()), int(columns.max() + 1)
        y0, y1 = int(rows.min()), int(rows.max() + 1)
        combined |= filled.astype(bool)
        parts.append(
            {
                "bbox": (x0, y0, x1, y1),
                "polygon": polygon,
                "crop": image[y0:y1, x0:x1],
                "mask": filled[y0:y1, x0:x1] * 255,
            }
        )

    fallback = not parts
    if fallback:
        parts = [
            {
                "bbox": (0, 0, width, height),
                "polygon": None,
                "crop": image,
                "mask": None,
            }
        ]
        combined[:] = True

    debug: dict[str, object] = {
        "colour": colour,
        "edge": edge,
        "combined": combined,
        "colour_used": colour_used,
        "edge_used": edge_used,
        "fallback": fallback,
    }
    return parts, debug


def find_paintings_refined(
    image: np.ndarray,
    thinness_threshold: float = 0.30,
    frame_min_area: float = 0.08,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Refine the baseline detector for thin false regions and stacked frames.

    Thin-region rejection is applied only when the baseline proposes more than
    one painting. A single proposal is split only when edge contours produce
    exactly two distinct, sufficiently large rectangular frame candidates.
    """
    baseline_parts, baseline_debug = find_paintings(image)
    parts = list(baseline_parts)

    rejected = 0
    if len(parts) > 1:
        retained = [
            part
            for part in parts
            if part["polygon"] is None
            or _polygon_aspect_ratio(part["polygon"]) >= thinness_threshold
        ]
        if retained:
            rejected = len(parts) - len(retained)
            parts = retained

    split_used = False
    candidates = []
    if len(parts) == 1:
        candidates = [
            candidate
            for candidate in _frame_candidates(image)
            if float(candidate["area_fraction"]) >= frame_min_area
        ]
        if len(candidates) == 2:
            parts = [
                _part_from_polygon(image, candidate["polygon"])[0]
                for candidate in candidates
            ]
            split_used = True

    parts = _reading_order(parts)
    combined = np.zeros(image.shape[:2], dtype=bool)
    for part in parts:
        polygon = part["polygon"]
        if polygon is None:
            combined[:] = True
        else:
            filled = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(
                filled,
                np.round(polygon).astype(np.int32),
                1,
            )
            combined |= filled.astype(bool)

    debug = {
        **baseline_debug,
        "combined": combined,
        "thin_regions_rejected": rejected,
        "frame_candidate_count": len(candidates),
        "split_used": split_used,
    }
    return parts, debug
