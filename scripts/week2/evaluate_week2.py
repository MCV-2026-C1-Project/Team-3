"""Evaluate the selected Week 2 descriptor, masks, and QSD2 retrieval."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    descriptor,
    descriptor_masked,
    evaluate_masks,
    load_binary_masks,
    load_ground_truth,
    load_image_dataset,
    map_at_k,
    predict_foreground_masks,
    read_images,
    retrieve_from_features,
)
from src.methods import WEEK2_BLOCK_5X5


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def describe_complete(images: list[np.ndarray], config) -> np.ndarray:
    return np.asarray(
        [
            descriptor(
                image,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
                hist_type=config.hist_type,
                level_weights=config.level_weights,
            )
            for image in images
        ]
    )


def describe_masked(
    images: list[np.ndarray], masks: list[np.ndarray] | None, config
) -> np.ndarray:
    if masks is None:
        masks = [None] * len(images)
    return np.asarray(
        [
            descriptor_masked(
                image,
                mask,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
                hist_type=config.hist_type,
                level_weights=config.level_weights,
            )
            for image, mask in zip(images, masks)
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Week 2 Tasks 2, 4, and 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qsd1-dir", type=Path, default=Path("data/qsd1_w1"))
    parser.add_argument("--qsd2-dir", type=Path, default=Path("data/qsd2_w1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week2/evaluation.json"),
        help="JSON summary written after evaluation",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="One or more spatial grid sizes, from coarsest to finest",
    )
    parser.add_argument(
        "--hist-type",
        choices=("1d", "2d", "3d"),
        default=None,
        help="Histogram representation used for retrieval descriptors",
    )
    level_weights = parser.add_mutually_exclusive_group()
    level_weights.add_argument(
        "--pyramid-weights",
        action="store_true",
        help="Use Lazebnik spatial-pyramid level weights",
    )
    level_weights.add_argument(
        "--level-weights",
        type=float,
        nargs="+",
        default=None,
        help="Explicit non-negative weight for each selected level",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    database = load_image_dataset(project_path(args.database_dir))
    qsd1 = load_image_dataset(project_path(args.qsd1_dir))
    qsd2 = load_image_dataset(project_path(args.qsd2_dir))
    database_images = read_images(database)
    qsd1_images = read_images(qsd1)
    qsd2_images = read_images(qsd2)
    config = replace(
        WEEK2_BLOCK_5X5,
        levels=tuple(args.levels) if args.levels is not None else WEEK2_BLOCK_5X5.levels,
        hist_type=args.hist_type or WEEK2_BLOCK_5X5.hist_type,
        level_weights=(
            "pyramid"
            if args.pyramid_weights
            else tuple(args.level_weights)
            if args.level_weights is not None
            else WEEK2_BLOCK_5X5.level_weights
        ),
    )

    qsd1_ground_truth = load_ground_truth(project_path(args.qsd1_dir) / "gt_corresps.pkl")
    complete_database = describe_complete(database_images, config)
    qsd1_features = describe_complete(qsd1_images, config)
    qsd1_rankings = retrieve_from_features(
        qsd1_features, complete_database, database.ids, k=10
    )
    task2 = {
        "mAP@1": map_at_k(qsd1_rankings, qsd1_ground_truth, 1),
        "mAP@5": map_at_k(qsd1_rankings, qsd1_ground_truth, 5),
    }
    print("Task 2: mAP@1 %.3f  mAP@5 %.3f" % (task2["mAP@1"], task2["mAP@5"]))

    predicted_masks = predict_foreground_masks(qsd2_images)
    reference_masks = load_binary_masks(qsd2.paths)
    mask_summary, _ = evaluate_masks(reference_masks, predicted_masks)
    macro = mask_summary["macro"]
    micro = mask_summary["micro"]
    print(
        "Task 4 macro: precision %.3f  recall %.3f  F1 %.3f"
        % (macro["precision"], macro["recall"], macro["f1"])
    )
    print(
        "Task 4 micro: precision %.3f  recall %.3f  F1 %.3f"
        % (micro["precision"], micro["recall"], micro["f1"])
    )

    qsd2_ground_truth = load_ground_truth(project_path(args.qsd2_dir) / "gt_corresps.pkl")
    masked_database = describe_masked(database_images, masks=None, config=config)
    qsd2_features = describe_masked(qsd2_images, predicted_masks, config=config)
    qsd2_rankings = retrieve_from_features(
        qsd2_features, masked_database, database.ids, k=10
    )
    task5 = {
        "mAP@1": map_at_k(qsd2_rankings, qsd2_ground_truth, 1),
        "mAP@5": map_at_k(qsd2_rankings, qsd2_ground_truth, 5),
    }
    print("Task 5: mAP@1 %.3f  mAP@5 %.3f" % (task5["mAP@1"], task5["mAP@5"]))

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "method": {
                    "levels": list(config.levels),
                    "bins": config.bins,
                    "sigma": config.sigma,
                    "channel_weights": list(config.channel_weights),
                    "hist_type": config.hist_type,
                    "level_weights": config.level_weights,
                },
                "task2_qsd1_retrieval": task2,
                "task4_mask_macro": macro,
                "task4_mask_micro": micro,
                "task5_qsd2_retrieval": task5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
