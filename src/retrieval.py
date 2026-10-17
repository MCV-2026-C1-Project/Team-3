"""Feature-matrix retrieval helpers shared by Week 2 experiments."""

import numpy as np


def hellinger(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Return Hellinger-proportional distances from one query to each database row."""
    query = np.asarray(query, dtype=np.float64)
    database = np.asarray(database, dtype=np.float64)
    return np.sqrt(((np.sqrt(query) - np.sqrt(database)) ** 2).sum(axis=1))


def retrieve_from_features(
    query_features: np.ndarray,
    database_features: np.ndarray,
    database_ids: np.ndarray,
    k: int = 10,
) -> list[list[int]]:
    """Rank database IDs for every query feature using Hellinger distance."""
    if k <= 0 or k > len(database_ids):
        raise ValueError(f"k must be between 1 and {len(database_ids)}")
    if query_features.ndim != 2 or database_features.ndim != 2:
        raise ValueError("query_features and database_features must be matrices")
    if query_features.shape[1] != database_features.shape[1]:
        raise ValueError("query and database descriptor dimensions differ")
    if len(database_features) != len(database_ids):
        raise ValueError("database features and IDs must have equal length")

    results: list[list[int]] = []
    for query in query_features:
        order = np.argsort(hellinger(query, database_features), kind="stable")[:k]
        results.append([int(database_ids[index]) for index in order])
    return results
