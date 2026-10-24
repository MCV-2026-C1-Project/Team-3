"""Feature-matrix retrieval helpers shared by Week 2 experiments."""
""" Multi painting retrieval done in week 3"""

import numpy as np
import cv2

from .evaluation import map_at_k

TEXTURE_SIZE = 256
K = 10


def hellinger(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Return Hellinger-proportional distances from one query to each database row."""
    query = np.asarray(query, dtype=np.float64)
    database = np.asarray(database, dtype=np.float64)
    return np.sqrt(((np.sqrt(query) - np.sqrt(database)) ** 2).sum(axis=1))


def order_box_points(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(coordinate_sum)]         # top-left
    ordered[1] = points[np.argmin(coordinate_difference)]  # top-right
    ordered[2] = points[np.argmax(coordinate_sum)]         # bottom-right
    ordered[3] = points[np.argmax(coordinate_difference)]  # bottom-left
    return ordered


def extract_painting(image, region):
    source = order_box_points(region["box"])
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

    foreground = image.copy()
    foreground[~region["mask"]] = 255
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        foreground,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )



def flatten_ground_truth(query_ground_truth):
    """One list per painting (one database ID each), in left-to-right order."""
    return [[int(identifier)] for query in query_ground_truth for identifier in query]


def rankings_to_queries(flat_rankings, paintings_per_query):
    """Regroup a flat per-painting list into query -> painting -> ranking."""
    nested = []
    position = 0
    for count in paintings_per_query:
        nested.append(flat_rankings[position:position + count])
        position += count
    assert position == len(flat_rankings)
    return nested


def rankings_from_distances(distance_matrix, db_ids, paintings_per_query, k=K):
    """query -> painting -> top-k database identifiers, from a (paintings x database) distance matrix."""
    flat_rankings = [
        [int(db_ids[index]) for index in np.argsort(row, kind="stable")[:k]]
        for row in distance_matrix
    ]
    return rankings_to_queries(flat_rankings, paintings_per_query)


def correspondence_metrics(results, query_ground_truth):
    rankings = [ranking for query_result in results for ranking in query_result]
    painting_ground_truth = flatten_ground_truth(query_ground_truth)
    complete_query_accuracy = np.mean([
        all(ranking[0] == identifier for ranking, identifier in zip(query_result, matches))
        for query_result, matches in zip(results, query_ground_truth)
    ])
    return {
        "paintings": len(rankings),
        "mAP@1": map_at_k(rankings, painting_ground_truth, 1),
        "mAP@5": map_at_k(rankings, painting_ground_truth, 5),
        "mAP@10": map_at_k(rankings, painting_ground_truth, 10),
        "complete-query top-1 accuracy": float(complete_query_accuracy),
    }


def ranking_key(scores):
    return (scores["mAP@1"], scores["mAP@5"], scores["mAP@10"])
