import numpy as np


def pairwise_distances(query: np.ndarray, database: np.ndarray, metric: str) -> np.ndarray:
    """Return a query-by-database distance matrix; lower is always better."""
    q = query[:, None, :].astype(np.float64)
    x = database[None, :, :].astype(np.float64)

    if metric == "euclidean":
        return np.sqrt(((q - x) ** 2).sum(axis=2))
    if metric == "l1":
        return np.abs(q - x).sum(axis=2)
    if metric == "chi2":
        numerator = (q - x) ** 2
        denominator = q + x
        return 0.5 * np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        ).sum(axis=2)
    if metric == "intersection":
        # With L1-normalized histograms this has the same ranking as L1.
        return 1.0 - np.minimum(q, x).sum(axis=2)
    if metric == "hellinger":
        # Bhattacharyya coefficient converted to a distance in [0, 1].
        coefficient = np.sqrt(q * x).sum(axis=2)
        return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))
    if metric == "jensen_shannon":
        middle = 0.5 * (q + x)
        q_ratio = np.divide(q, middle, out=np.ones_like(q + x), where=q > 0)
        x_ratio = np.divide(x, middle, out=np.ones_like(q + x), where=x > 0)
        divergence = 0.5 * (
            np.where(q > 0, q * np.log(q_ratio), 0).sum(axis=2)
            + np.where(x > 0, x * np.log(x_ratio), 0).sum(axis=2)
        )
        return np.sqrt(np.maximum(divergence, 0.0))
    raise ValueError(f"Unknown metric: {metric}")
