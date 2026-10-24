"""Evaluate Week 3: noise filtering (Task 1), texture descriptors on QSD1-W3 (Task 2),
painting detection / background removal on QSD2-W3 (Task 3) and
multi-painting retrieval on QSD2-W3 (Task 4).

Every texture method is a preset of `src.texture_histograms.TEXTURE_PRESETS`.
Database descriptors are computed once per preset and reused by Tasks 2 and 4.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.background_removal import detect_painting_mask
from src.data import load_binary_masks, load_ground_truth, load_image_dataset, read_images
from src.evaluation import evaluate_painting_masks, map_at_k
from src.noise import NOISE_FILTERS, evaluate_filters, split_noisy_images, summarize_filter_metrics
from src.retrieval import (
    correspondence_metrics,
    extract_painting,
    ranking_key,
    rankings_from_distances,
)
from src.texture_histograms import (
    WEEK3_FINAL_CONFIG,
    TEXTURE_PRESETS,
    WEEK3_LBP_R2_G12_L1,
    WEEK3_LBP_R2_G16_HELLINGER,
    compute_distances,
    describe_images,
)


# Task 2 uses the presets tuned on QSD1-W3; the radius-2 LBP variants only appear in Task 4.
TASK4_ONLY_CONFIGS = (WEEK3_LBP_R2_G12_L1, WEEK3_LBP_R2_G16_HELLINGER)
TASK2_PRESETS = {
    name: config for name, config in TEXTURE_PRESETS.items() if config not in TASK4_ONLY_CONFIGS
}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Week 3 Tasks 1, 2, 3 and 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qsd1-dir", type=Path, default=Path("data/qsd1_w3"))
    parser.add_argument(
        "--non-augmented-dir",
        type=Path,
        default=None,
        help="Clean versions of QSD1-W3 (default: <qsd1-dir>/non_augmented)",
    )
    parser.add_argument("--qsd2-dir", type=Path, default=Path("data/qsd2_w3"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week3/evaluation.json"),
        help="JSON summary written after evaluation",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip Task 1 (the filter comparison is slow because of non-local means)",
    )
    return parser


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.round(3).to_string())


# ----------------------------------------------------------------------
# Task 1 - noise filtering against the non-augmented dataset
# ----------------------------------------------------------------------
def run_task1(qsd1, non_augmented_dir: Path) -> dict:
    non_augmented = load_image_dataset(non_augmented_dir)
    _, _, scores, threshold = split_noisy_images(qsd1, non_augmented)
    noisy = scores > threshold  # same criterion as split_noisy_images

    # Filters are evaluated once on every image (NLM is slow); the noisy-only
    # table is just a per-image selection of those results.
    metrics_all = evaluate_filters(qsd1, non_augmented, NOISE_FILTERS)
    metrics_noisy = {
        name: {key: values[noisy] for key, values in metrics.items()}
        for name, metrics in metrics_all.items()
    }
    rows_all = summarize_filter_metrics(metrics_all)
    rows_noisy = summarize_filter_metrics(metrics_noisy)

    n_noisy = int(noisy.sum())
    print(
        f"Task 1: {n_noisy} noisy / {len(noisy) - n_noisy} without noise "
        f"(threshold = {threshold:.2g})"
    )
    print_table("Filter metrics (all QSD1-W3)", pd.DataFrame(rows_all).set_index("Filter"))
    print_table("Filter metrics (noisy images only)", pd.DataFrame(rows_noisy).set_index("Filter"))
    return {
        "noisy_images": n_noisy,
        "clean_images": len(noisy) - n_noisy,
        "threshold": threshold,
        "filters_all": rows_all,
        "filters_noisy_only": rows_noisy,
    }


# ----------------------------------------------------------------------
# Task 2 - texture descriptors on QSD1-W3 (median 3x3 + resize 256, both sides)
# ----------------------------------------------------------------------
def run_task2(db_features: dict, qsd1_imgs, qsd1_ground_truth, db_ids) -> list[dict]:
    rows = []
    for name, config in TASK2_PRESETS.items():
        query_features = describe_images(qsd1_imgs, config)
        distances = compute_distances(query_features, db_features[name], config.distance)
        # One painting per query: take the single ranking of each query.
        rankings = [
            ranking[0]
            for ranking in rankings_from_distances(distances, db_ids, [1] * len(distances))
        ]
        rows.append({
            "descriptor": name,
            "values per image": db_features[name].shape[1],
            "mAP@1": map_at_k(rankings, qsd1_ground_truth, 1),
            "mAP@5": map_at_k(rankings, qsd1_ground_truth, 5),
        })
    print_table("Task 2: texture descriptors on QSD1-W3", pd.DataFrame(rows).set_index("descriptor"))
    return rows


# ----------------------------------------------------------------------
# Task 3 - painting detection and background removal on QSD2-W3
# ----------------------------------------------------------------------
def run_task3(qsd2, qsd2_imgs, qsd2_ground_truth, reference_masks):
    """Return (summary, regions per query); the regions are reused by Task 4."""
    detections = [detect_painting_mask(image) for image in qsd2_imgs]
    predicted_masks = [mask for mask, _ in detections]
    all_regions = [regions for _, regions in detections]

    mask_summary, per_image = evaluate_painting_masks(
        reference_masks, predicted_masks, all_regions, qsd2_ground_truth, qsd2.ids
    )
    macro, micro = mask_summary["macro"], mask_summary["micro"]
    print(
        "\nTask 3 macro: precision %.3f  recall %.3f  F1 %.3f  painting-count accuracy %.3f"
        % (macro["precision"], macro["recall"], macro["F1"], macro["painting_count_accuracy"])
    )
    print(
        "Task 3 micro: precision %.3f  recall %.3f  F1 %.3f"
        % (micro["precision"], micro["recall"], micro["F1"])
    )

    count_errors = [
        row for row in per_image if row["expected paintings"] != row["detected paintings"]
    ]
    if count_errors:
        print("Queries with an incorrect painting count:")
        for row in count_errors:
            print(
                "  query %05d: expected %d, detected %d"
                % (row["query"], row["expected paintings"], row["detected paintings"])
            )
    else:
        print("The predicted painting count is correct for every development query.")
    return {**mask_summary, "count_errors": count_errors}, all_regions


# ----------------------------------------------------------------------
# Task 4 - multi-painting retrieval on QSD2-W3
# ----------------------------------------------------------------------
def run_task4(qsd2_imgs, all_regions, qsd2_ground_truth, db_features: dict, db_ids) -> dict:
    query_paintings = [
        [extract_painting(image, region) for region in regions]
        for image, regions in zip(qsd2_imgs, all_regions)
    ]
    paintings_per_query = [len(paintings) for paintings in query_paintings]
    flat_paintings = [painting for paintings in query_paintings for painting in paintings]
    print(f"\nTask 4: {len(flat_paintings)} paintings in {len(qsd2_imgs)} queries")

    # (paintings x database) distance matrix per preset.
    distance_matrices = {
        name: compute_distances(describe_images(flat_paintings, config), db_features[name], config.distance)
        for name, config in TEXTURE_PRESETS.items()
    }

    method_scores = {
        name: correspondence_metrics(
            rankings_from_distances(matrix, db_ids, paintings_per_query), qsd2_ground_truth
        )
        for name, matrix in distance_matrices.items()
    }
    comparison = (
        pd.DataFrame([{"method": name, **scores} for name, scores in method_scores.items()])
        .sort_values(["mAP@1", "mAP@5", "mAP@10"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    print_table(
        "Task 4: QSD2-W3 retrieval (one query per detected painting)",
        comparison.set_index("method"),
    )

    best_key = max(method_scores, key=lambda name: ranking_key(method_scores[name]))
    print(f"Best method by (mAP@1, mAP@5, mAP@10): {best_key}")
    # Preset name of the final configuration (JSON-serialisable, unlike the config object).
    final_name = next(name for name, config in TEXTURE_PRESETS.items() if config == WEEK3_FINAL_CONFIG)
    print(f"Selected method (frozen for submission): {final_name}")
    return {"selected_method": final_name, "methods": method_scores}


def main() -> int:
    args = build_parser().parse_args()
    qsd1_dir = project_path(args.qsd1_dir)
    qsd2_dir = project_path(args.qsd2_dir)
    non_augmented_dir = (
        project_path(args.non_augmented_dir)
        if args.non_augmented_dir is not None
        else qsd1_dir / "non_augmented"
    )

    database = load_image_dataset(project_path(args.database_dir))
    qsd1 = load_image_dataset(qsd1_dir)
    qsd1_ground_truth = load_ground_truth(qsd1_dir / "gt_corresps.pkl")
    qsd2 = load_image_dataset(qsd2_dir)
    qsd2_ground_truth = load_ground_truth(qsd2_dir / "gt_corresps.pkl")
    reference_masks = load_binary_masks(qsd2.paths)

    db_imgs = read_images(database)
    qsd1_imgs = read_images(qsd1)
    qsd2_imgs = read_images(qsd2)
    assert len(qsd2_imgs) == len(qsd2_ground_truth) == len(reference_masks)

    # Database descriptors: computed once per preset, shared by Tasks 2 and 4.
    db_features = {name: describe_images(db_imgs, config) for name, config in TEXTURE_PRESETS.items()}

    summary: dict = {}
    if not args.skip_filters:
        summary["task1_noise"] = run_task1(qsd1, non_augmented_dir)
    summary["task2_qsd1_texture"] = run_task2(db_features, qsd1_imgs, qsd1_ground_truth, database.ids)
    summary["task3_masks"], all_regions = run_task3(qsd2, qsd2_imgs, qsd2_ground_truth, reference_masks)
    summary["task4_qsd2_retrieval"] = run_task4(
        qsd2_imgs, all_regions, qsd2_ground_truth, db_features, database.ids
    )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())