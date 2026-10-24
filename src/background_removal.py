"""Colour-only background removal used by Week 2 Tasks 3-6."""

import cv2
import numpy as np

from .segmentation import foreground_mask

def _boxes_are_near(box_a, box_b, gap):
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    return not (
        xa + wa + gap < xb
        or xb + wb + gap < xa
        or ya + ha + gap < yb
        or yb + hb + gap < ya
    )


def detect_painting_mask(
    image,
    max_paintings=2,
    component_min_area=0.005,
    painting_min_area=0.02,
    merge_gap_fraction=0.04,
):
    """Predict one combined mask and up to two left-to-right regions."""
    raw_mask = foreground_mask(image, min_area=component_min_area, keep="all")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask.astype(np.uint8), connectivity=8
    )

    image_area = raw_mask.size
    component_ids = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= component_min_area * image_area
    ]

    parent = {label: label for label in component_ids}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(label_a, label_b):
        root_a, root_b = find(label_a), find(label_b)
        if root_a != root_b:
            parent[root_b] = root_a

    merge_gap = merge_gap_fraction * min(raw_mask.shape)
    for position, label_a in enumerate(component_ids):
        box_a = tuple(int(value) for value in stats[label_a, :4])
        for label_b in component_ids[position + 1:]:
            box_b = tuple(int(value) for value in stats[label_b, :4])
            if _boxes_are_near(box_a, box_b, merge_gap):
                union(label_a, label_b)

    groups = {}
    for label in component_ids:
        groups.setdefault(find(label), []).append(label)

    regions = []
    for group_labels in groups.values():
        component_mask = np.isin(labels, group_labels)
        component_area = int(component_mask.sum())
        if component_area < painting_min_area * image_area:
            continue

        rows, columns = np.where(component_mask)
        points = np.column_stack((columns, rows)).astype(np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(points))
        regions.append({"area": component_area, "box": box})

    regions = sorted(regions, key=lambda region: region["area"], reverse=True)[:max_paintings]
    regions.sort(key=lambda region: float(region["box"][:, 0].mean()))

    combined_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    for region in regions:
        region_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
        polygon = np.round(region["box"]).astype(np.int32)
        cv2.fillConvexPoly(region_mask, polygon, 1)
        region["mask"] = region_mask.astype(bool)
        combined_mask |= region_mask

    return combined_mask.astype(bool), regions