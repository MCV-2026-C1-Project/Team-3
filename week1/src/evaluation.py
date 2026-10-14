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
