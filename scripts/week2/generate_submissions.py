"""Generate and validate Team 3 Week 2 blind-test submissions."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    descriptor,
    descriptor_masked,
    load_image_dataset,
    predict_foreground_masks,
    read_images,
    retrieve_from_features,
    save_rankings,
    save_submission_masks,
    validate_rankings,
    validate_submission_masks,
)
from src.methods import WEEK2_BLOCK_5X5


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validated QST1-W2 and QST2-W2 submission files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qst1-dir", type=Path, default=Path("data/qst_w2/qst1_w2"))
    parser.add_argument("--qst2-dir", type=Path, default=Path("data/qst_w2/qst2_w2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submissions/Team3/week2"),
    )
    parser.add_argument("-k", "--top-k", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    database = load_image_dataset(project_path(args.database_dir))
    qst1 = load_image_dataset(project_path(args.qst1_dir))
    qst2 = load_image_dataset(project_path(args.qst2_dir))
    database_images = read_images(database)
    qst1_images = read_images(qst1)
    qst2_images = read_images(qst2)
    config = WEEK2_BLOCK_5X5

    complete_database = np.asarray(
        [
            descriptor(
                image,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
            )
            for image in database_images
        ]
    )
    qst1_features = np.asarray(
        [
            descriptor(
                image,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
            )
            for image in qst1_images
        ]
    )
    qst1_results = retrieve_from_features(
        qst1_features, complete_database, database.ids, k=args.top_k
    )

    qst2_masks = predict_foreground_masks(qst2_images)
    masked_database = np.asarray(
        [
            descriptor_masked(
                image,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
            )
            for image in database_images
        ]
    )
    qst2_features = np.asarray(
        [
            descriptor_masked(
                image,
                mask,
                levels=config.levels,
                bins=config.bins,
                sigma=config.sigma,
                weights=config.channel_weights,
            )
            for image, mask in zip(qst2_images, qst2_masks)
        ]
    )
    qst2_results = retrieve_from_features(
        qst2_features, masked_database, database.ids, k=args.top_k
    )

    output_root = project_path(args.output_dir)
    qst1_method = output_root / "QST1" / "method1"
    qst2_method = output_root / "QST2" / "method1"
    save_rankings(qst1_results, qst1_method / "result.pkl", len(qst1), args.top_k)
    save_rankings(qst2_results, qst2_method / "result.pkl", len(qst2), args.top_k)
    save_submission_masks(qst2.paths, qst2_masks, qst2_method)

    validate_rankings(qst1_method / "result.pkl", qst1, database.ids, args.top_k)
    validate_rankings(qst2_method / "result.pkl", qst2, database.ids, args.top_k)
    mask_count = validate_submission_masks(qst2, qst2_method)
    print(f"QST1: {len(qst1_results)} rows x {args.top_k} results")
    print(f"QST2: {len(qst2_results)} rows x {args.top_k} results")
    print(f"QST2: {mask_count} validated binary masks")
    print(f"Saved submission under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
