import numpy as np


def average_precision_at_k(relevant: list[int], ranked: np.ndarray, k: int) -> float:
    relevant_ids = set(relevant)
    hits = 0
    score = 0.0
    for rank, image_id in enumerate(ranked[:k], start=1):
        if int(image_id) in relevant_ids:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_ids), k) if relevant_ids else 0.0


def evaluate_rankings(rankings: np.ndarray, ground_truth: list[list[int]]) -> dict[str, float]:
    if len(rankings) != len(ground_truth):
        raise ValueError("Rankings and ground truth must contain the same number of queries")
    # QSD1 has one relevant painting per query, so mAP@1 is also top-1 accuracy.
    ap1 = [average_precision_at_k(relevant, ranked, 1) for relevant, ranked in zip(ground_truth, rankings)]
    ap5 = [average_precision_at_k(relevant, ranked, 5) for relevant, ranked in zip(ground_truth, rankings)]
    top5 = [bool(set(map(int, ranked[:5])) & set(relevant)) for relevant, ranked in zip(ground_truth, rankings)]
    return {
        "mAP@1": float(np.mean(ap1)),
        "mAP@5": float(np.mean(ap5)),
        "top5_hit_rate": float(np.mean(top5)),
    }


def map_at_k(results: list[list[int]], ground_truth: list[list[int]], k: int) -> float:
    """Compute mean AP@k for list-based rankings used by the Week 2 notebook."""
    if len(results) != len(ground_truth):
        raise ValueError("results and ground_truth must contain the same number of queries")
    scores = [
        average_precision_at_k(relevant, np.asarray(ranked), k)
        for ranked, relevant in zip(results, ground_truth)
    ]
    return float(np.mean(scores))


def rank_configs(df, group_cols):
    """Order: mAP@1, mAP@5, and smallest descriptor."""
    df = df.copy()
    return df.sort_values(["mAP@1", "mAP@5", "dim"],
                          ascending=[False, False, True]).reset_index(drop=True)


def mask_counts(annotation: np.ndarray, output: np.ndarray) -> dict[str, int]:
    """Return TP, FP, FN, and TN with foreground as the positive class."""
    annotation = np.asarray(annotation, dtype=bool)
    output = np.asarray(output, dtype=bool)
    if annotation.shape != output.shape:
        raise ValueError(f"Mask shapes differ: {annotation.shape} and {output.shape}")
    return {
        "tp": int(np.count_nonzero(annotation & output)),
        "fp": int(np.count_nonzero(~annotation & output)),
        "fn": int(np.count_nonzero(annotation & ~output)),
        "tn": int(np.count_nonzero(~annotation & ~output)),
    }


def scores_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Convert foreground confusion counts into precision, recall, and F1."""
    true_positive = counts["tp"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_masks(
    annotations: list[np.ndarray],
    outputs: list[np.ndarray],
) -> tuple[dict[str, dict[str, float] | dict[str, int]], list[dict[str, int | float]]]:
    """Evaluate binary masks using macro and micro foreground metrics."""
    if len(annotations) != len(outputs):
        raise ValueError("The number of annotations and outputs must match")

    per_image: list[dict[str, int | float]] = []
    totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    for query_index, (annotation, output) in enumerate(zip(annotations, outputs)):
        counts = mask_counts(annotation, output)
        scores = scores_from_counts(counts)
        per_image.append({"query": query_index, **counts, **scores})
        for name in totals:
            totals[name] += counts[name]

    macro = {
        name: float(np.mean([row[name] for row in per_image]))
        for name in ("precision", "recall", "f1")
    }
    return {
        "macro": macro,
        "micro": scores_from_counts(totals),
        "totals": totals,
    }, per_image



def evaluate_painting_masks(references, predictions, regions_per_query, correspondences, query_ids):
    """Per-image and macro / micro mask scores plus painting-count accuracy.

    The expected painting count comes from the correspondences and is used
    only for evaluation, never by the detector.
    Returns (summary, per_image_rows).
    """
    rows = []
    totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    for query_id, reference, prediction, regions, matches in zip(
        query_ids, references, predictions, regions_per_query, correspondences
    ):
        counts = mask_counts(reference, prediction)
        scores = scores_from_counts(counts)
        for name in totals:
            totals[name] += counts[name]
        rows.append({
            "query": int(query_id),
            "expected paintings": len(matches),
            "detected paintings": len(regions),
            "precision": scores["precision"],
            "recall": scores["recall"],
            "F1": scores["f1"],
        })

    micro = scores_from_counts(totals)
    summary = {
        "macro": {
            "precision": float(np.mean([r["precision"] for r in rows])),
            "recall": float(np.mean([r["recall"] for r in rows])),
            "F1": float(np.mean([r["F1"] for r in rows])),
            "painting_count_accuracy": float(np.mean([
                r["expected paintings"] == r["detected paintings"] for r in rows
            ])),
        },
        "micro": {
            "precision": micro["precision"],
            "recall": micro["recall"],
            "F1": micro["f1"],
        },
    }
    return summary, rows