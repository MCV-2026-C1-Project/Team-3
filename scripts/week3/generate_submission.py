"""Generate and validate Team 3 Week 3 blind-test submissions (QST1-W3 and QST2-W3).

Single method for both test sets (Task 4 conclusion): LBP, 8 neighbours, radius 2,
16x16 blocks, Hellinger distance, on median-filtered 256x256 grayscale images
(`WEEK3_FINAL_CONFIG` in `src.texture_histograms`).
Result format: query -> painting -> top-K database IDs.
"""

import argparse
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.background_removal import detect_painting_mask
from src.data import load_ground_truth, load_image_dataset, read_images
from src.evaluation import map_at_k
from src.retrieval import extract_painting, rankings_from_distances
from src.submission import (
    check_result,
    save_pkl,
    save_submission_masks,
    validate_submission_masks,
)
from src.texture_histograms import (
    WEEK3_FINAL_CONFIG,
    WEEK3_FINAL_CONFIG,
    compute_distances,
    describe_images,
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validated QST1-W3 and QST2-W3 submission files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qsd1-dir", type=Path, default=Path("data/qsd1_w3"))
    parser.add_argument("--qst1-dir", type=Path, default=Path("data/qst1_w3"))
    parser.add_argument("--qst2-dir", type=Path, default=Path("data/qst2_w3"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submissions/week3"),
    )
    parser.add_argument("--method-name", default="method1")
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument(
        "--skip-dev-check",
        action="store_true",
        help="Skip the sanity check on QSD1-W3 (needs its ground truth)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    k = args.top_k
    config = WEEK3_FINAL_CONFIG

    database = load_image_dataset(project_path(args.database_dir))
    qst1 = load_image_dataset(project_path(args.qst1_dir))
    qst2 = load_image_dataset(project_path(args.qst2_dir))
    qst1_images = read_images(qst1)
    qst2_images = read_images(qst2)

    print("Method:", WEEK3_FINAL_CONFIG)
    db_features = describe_images(read_images(database), config)

    def retrieve(paintings, paintings_per_query):
        """query -> painting -> top-k database IDs for a flat list of paintings."""
        distances = compute_distances(describe_images(paintings, config), db_features, config.distance)
        return rankings_from_distances(distances, database.ids, paintings_per_query, k)

    # Sanity check on the development set (has ground truth).
    if not args.skip_dev_check:
        qsd1_dir = project_path(args.qsd1_dir)
        qsd1_images = read_images(load_image_dataset(qsd1_dir))
        qsd1_ground_truth = load_ground_truth(qsd1_dir / "gt_corresps.pkl")
        dev = [query[0] for query in retrieve(qsd1_images, [1] * len(qsd1_images))]
        print(
            "QSD1-W3 dev: mAP@1 %.3f  mAP@5 %.3f"
            % (map_at_k(dev, qsd1_ground_truth, 1), map_at_k(dev, qsd1_ground_truth, 5))
        )

    # QST1: one painting per image, so every query has a single inner list.
    qst1_result = retrieve(qst1_images, [1] * len(qst1_images))
    check_result(qst1_result, len(qst1_images), database.ids, k)

    # QST2: detect up to two paintings (ordered left to right), retrieve each one.
    qst2_masks, qst2_paintings, paintings_per_query = [], [], []
    for image in qst2_images:
        mask, regions = detect_painting_mask(image)
        paintings = [extract_painting(image, region) for region in regions]
        qst2_masks.append(mask)
        qst2_paintings.extend(paintings)
        paintings_per_query.append(len(paintings))
    qst2_result = retrieve(qst2_paintings, paintings_per_query)
    check_result(qst2_result, len(qst2_images), database.ids, k)

    output_root = project_path(args.output_dir)
    qst1_method = output_root / "QST1" / args.method_name
    qst2_method = output_root / "QST2" / args.method_name
    save_pkl(qst1_result, qst1_method)
    save_pkl(qst2_result, qst2_method)
    save_submission_masks(qst2.paths, qst2_masks, qst2_method)

    # Re-open what was written and validate it.
    for method_dir, query_dataset in ((qst1_method, qst1), (qst2_method, qst2)):
        with (method_dir / "result.pkl").open("rb") as stream:
            check_result(pickle.load(stream), len(query_dataset), database.ids, k)
    mask_count = validate_submission_masks(qst2, qst2_method)

    n_paintings = len(qst2_paintings)
    two_painting = sum(count == 2 for count in paintings_per_query)
    print("VALIDATION PASSED")
    print(f"QST1: {len(qst1_result)} queries, 1 painting each, {k} results")
    print(f"QST2: {len(qst2_result)} queries, {n_paintings} paintings ({two_painting} queries with 2)")
    print(f"QST2: {mask_count} validated binary masks")
    print(f"Saved submission under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())