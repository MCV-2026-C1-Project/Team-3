import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ColorHistogramRetrieval, HistogramConfig, load_ground_truth, load_image_dataset
from src.methods import AUDIT_METHODS


def project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the project root."""
    return path if path.is_absolute() else ROOT / path


def evaluate_methods(
    database_dir: Path,
    query_dir: Path,
    ground_truth_path: Path,
    configs: Sequence[HistogramConfig] = AUDIT_METHODS,
) -> pd.DataFrame:
    """Evaluate several histogram configurations on one development set."""
    database = load_image_dataset(database_dir)
    queries = load_image_dataset(query_dir)
    ground_truth = load_ground_truth(ground_truth_path)

    rows = []
    for config in configs:
        retriever = ColorHistogramRetrieval(config).fit(database)
        metrics, _ = retriever.evaluate(queries, ground_truth)
        rows.append({**config.as_dict(), **metrics})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for QSD1 evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate all named Week 1 color-histogram methods on QSD1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/BBDD"),
        help="database image directory",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=Path("data/qsd1_w1"),
        help="development-query image directory",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/qsd1_w1/gt_corresps.pkl"),
        help="development ground-truth pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/week1/qsd1_results.csv"),
        help="output CSV file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run evaluation from command-line arguments."""
    args = build_parser().parse_args(argv)
    output = project_path(args.output)
    results = evaluate_methods(
        database_dir=project_path(args.database_dir),
        query_dir=project_path(args.query_dir),
        ground_truth_path=project_path(args.ground_truth),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results[["name", "mAP@1", "mAP@5", "top5_hit_rate"]].to_string(index=False))
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
