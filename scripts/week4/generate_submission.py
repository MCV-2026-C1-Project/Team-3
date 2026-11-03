"""Generate and validate the Team 3 Week 4 QST1 submission.

Submission format: query -> detected painting -> top-10 BBDD identifiers.
An unknown painting is represented by ``[-1]``. Only the selected method1 is
written, as required by the Week 4 task specification.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_ground_truth, load_image_dataset, read_images
from src.evaluation import map_at_k
from src.local_features import (
    WEEK4_FINAL_RETRIEVAL,
    describe_database,
    retrieve_query_images,
)
from src.submission import check_result, save_pkl


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the validated QST1-W4 method1 submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qsd1-dir", type=Path, default=Path("data/qsd1_w4"))
    parser.add_argument(
        "--qst1-dir",
        type=Path,
        default=Path("data/qst_w4/qst1_w4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submissions/Team3/week4"),
    )
    parser.add_argument("--method-name", default="method1")
    parser.add_argument(
        "--skip-dev-check",
        action="store_true",
        help="Skip the QSD1-W4 sanity check before generating the blind result.",
    )
    return parser


def aligned_lists(rankings, ground_truth):
    """Flatten query/painting predictions and score missed detections as empty."""
    flat_rankings, targets = [], []
    for predicted, expected in zip(rankings, ground_truth):
        for position, identifier in enumerate(expected):
            flat_rankings.append(
                predicted[position] if position < len(predicted) else []
            )
            targets.append([int(identifier)])
    return flat_rankings, targets


def main() -> int:
    args = build_parser().parse_args()
    config = WEEK4_FINAL_RETRIEVAL
    database = load_image_dataset(project_path(args.database_dir))
    qst1 = load_image_dataset(project_path(args.qst1_dir))
    database_images = read_images(database)
    database_features = describe_database(
        database_images,
        database.ids,
        nfeatures=config.nfeatures,
    )

    print(f"Method: {config.name}")
    print(
        "Parameters: SIFT %d, ratio %.2f, RANSAC %.1f px, shortlist %d, "
        "unknown if inliers < %.1f"
        % (
            config.nfeatures,
            config.ratio,
            config.ransac_threshold,
            config.shortlist,
            config.unknown_inlier_threshold,
        )
    )

    if not args.skip_dev_check:
        qsd1_dir = project_path(args.qsd1_dir)
        development = load_image_dataset(qsd1_dir)
        development_images = read_images(development)
        development_ground_truth = load_ground_truth(
            qsd1_dir / "gt_corresps.pkl"
        )
        development_result, _, _, _ = retrieve_query_images(
            development_images,
            database_features,
            config,
        )
        dev_rankings, dev_targets = aligned_lists(
            development_result, development_ground_truth
        )
        print(
            "QSD1-W4 dev: mAP@1 %.3f  mAP@5 %.3f  mAP@10 %.3f"
            % tuple(
                map_at_k(dev_rankings, dev_targets, k)
                for k in (1, 5, 10)
            )
        )

    qst1_images = read_images(qst1)
    qst1_result, _, qst1_parts, _ = retrieve_query_images(
        qst1_images,
        database_features,
        config,
    )
    check_result(
        qst1_result,
        len(qst1),
        database.ids,
        k=config.topk,
        allow_unknown=True,
    )

    method_dir = (
        project_path(args.output_dir) / "QST1" / args.method_name
    )
    save_pkl(qst1_result, method_dir)
    with (method_dir / "result.pkl").open("rb") as stream:
        written = pickle.load(stream)
    check_result(
        written,
        len(qst1),
        database.ids,
        k=config.topk,
        allow_unknown=True,
    )

    painting_count = sum(map(len, qst1_parts))
    unknown_count = sum(
        ranking == [-1]
        for query in qst1_result
        for ranking in query
    )
    two_painting_queries = sum(len(query) == 2 for query in qst1_result)
    print("VALIDATION PASSED")
    print(
        f"QST1-W4: {len(qst1_result)} queries, {painting_count} paintings, "
        f"{two_painting_queries} queries with two paintings"
    )
    print(f"Unknown decisions: {unknown_count}")
    print(f"Saved {method_dir / 'result.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
