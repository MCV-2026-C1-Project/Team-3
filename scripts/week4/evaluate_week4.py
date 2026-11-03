"""Evaluate the frozen Week 4 retrieval pipeline on QSD1-W4.

The report covers painting detection, known-painting retrieval, unknown
rejection, and the required comparison with the selected Week 3 system.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.background_removal import detect_painting_mask
from src.colour_texture_retrieval import (
    WEEK3_FINAL_RETRIEVAL,
    describe_colour_images,
    final_retrieval_distances,
)
from src.data import (
    load_binary_masks,
    load_ground_truth,
    load_image_dataset,
    read_images,
)
from src.evaluation import evaluate_painting_masks, map_at_k
from src.local_features import (
    WEEK4_FINAL_RETRIEVAL,
    describe_database,
    retrieve_query_images,
)
from src.retrieval import extract_painting, rankings_from_distances
from src.texture_histograms import describe_images


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected Week 4 system on QSD1-W4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database-dir", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--qsd1-dir", type=Path, default=Path("data/qsd1_w4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week4/evaluation.json"),
    )
    return parser


def aligned_painting_lists(nested_rankings, ground_truth, known_only=False):
    """Flatten query rankings while scoring missed detections as empty rows."""
    rankings, targets = [], []
    for predicted, expected in zip(nested_rankings, ground_truth):
        for painting_position, identifier in enumerate(expected):
            identifier = int(identifier)
            if known_only and identifier == -1:
                continue
            ranking = (
                predicted[painting_position]
                if painting_position < len(predicted)
                else []
            )
            rankings.append([int(value) for value in ranking])
            targets.append([identifier])
    return rankings, targets


def evaluate_rankings(label, nested_rankings, ground_truth) -> dict[str, object]:
    all_rankings, all_targets = aligned_painting_lists(
        nested_rankings, ground_truth
    )
    known_rankings, known_targets = aligned_painting_lists(
        nested_rankings, ground_truth, known_only=True
    )
    row: dict[str, object] = {
        "system": label,
        "detected_paintings": sum(map(len, nested_rankings)),
        "correct_query_counts": sum(
            len(predicted) == len(expected)
            for predicted, expected in zip(nested_rankings, ground_truth)
        ),
    }
    row.update({
        f"known_mAP@{k}": map_at_k(known_rankings, known_targets, k)
        for k in (1, 5, 10)
    })
    row.update({
        f"all_mAP@{k}": map_at_k(all_rankings, all_targets, k)
        for k in (1, 5, 10)
    })
    return row


def week3_rankings(
    query_crops,
    paintings_per_query,
    database,
    database_texture,
    database_colour,
):
    config = WEEK3_FINAL_RETRIEVAL
    distances = final_retrieval_distances(
        query_crops,
        database_texture,
        database_colour,
        config,
    )
    return rankings_from_distances(
        distances,
        database.ids,
        paintings_per_query,
        k=WEEK4_FINAL_RETRIEVAL.topk,
    )


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    args = build_parser().parse_args()
    qsd1_dir = project_path(args.qsd1_dir)
    database = load_image_dataset(project_path(args.database_dir))
    queries = load_image_dataset(qsd1_dir)
    ground_truth = load_ground_truth(qsd1_dir / "gt_corresps.pkl")
    reference_masks = load_binary_masks(queries.paths)
    database_images = read_images(database)
    query_images = read_images(queries)

    config = WEEK4_FINAL_RETRIEVAL
    database_features = describe_database(
        database_images,
        database.ids,
        nfeatures=config.nfeatures,
    )
    week3_database_texture = describe_images(
        database_images, WEEK3_FINAL_RETRIEVAL.texture
    )
    week3_database_colour = describe_colour_images(
        database_images, WEEK3_FINAL_RETRIEVAL.colour
    )
    week4_final, week4_results, week4_parts, predicted_masks = (
        retrieve_query_images(query_images, database_features, config)
    )
    week4_no_rejection = [
        [[int(identifier) for identifier in result["ranked"]] for result in query]
        for query in week4_results
    ]

    mask_summary, mask_rows = evaluate_painting_masks(
        reference_masks,
        predicted_masks,
        week4_parts,
        ground_truth,
        queries.ids,
    )
    mask_summary["detected_paintings"] = sum(map(len, week4_parts))
    mask_summary["ground_truth_paintings"] = sum(map(len, ground_truth))
    print("\nPainting detection")
    print(pd.DataFrame(mask_summary).round(3).to_string())

    # Week 3 complete pipeline: its original detector, rectifier and descriptors.
    week3_crops = []
    week3_counts = []
    for image in query_images:
        _, regions = detect_painting_mask(image)
        crops = [extract_painting(image, region) for region in regions]
        week3_crops.extend(crops)
        week3_counts.append(len(crops))
    week3_as_is = week3_rankings(
        week3_crops,
        week3_counts,
        database,
        week3_database_texture,
        week3_database_colour,
    )

    # Controlled comparison: Week 3 receives the exact Week 4 rectified crops.
    shared_crops = [
        part["crop"] for query_parts in week4_parts for part in query_parts
    ]
    shared_counts = [len(query_parts) for query_parts in week4_parts]
    week3_shared = week3_rankings(
        shared_crops,
        shared_counts,
        database,
        week3_database_texture,
        week3_database_colour,
    )

    systems = {
        "Week 3 full pipeline": week3_as_is,
        "Week 3 retrieval on Week 4 crops": week3_shared,
        "Week 4 retrieval without rejection": week4_no_rejection,
        "Week 4 final with unknown rejection": week4_final,
    }
    comparison = pd.DataFrame([
        evaluate_rankings(name, rankings, ground_truth)
        for name, rankings in systems.items()
    ]).set_index("system")
    print("\nQSD1-W4 comparison")
    print(comparison.round(3).to_string())

    summary = {
        "week4_config": asdict(config),
        "week3_config": asdict(WEEK3_FINAL_RETRIEVAL),
        "painting_detection": {
            **mask_summary,
            "per_query": mask_rows,
        },
        "retrieval_comparison": comparison.reset_index().to_dict("records"),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
