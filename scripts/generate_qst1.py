import argparse
import json
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ColorHistogramRetrieval, HistogramConfig, load_image_dataset
from src.methods import SUBMISSION_METHODS


def project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the project root."""
    return path if path.is_absolute() else ROOT / path


def validate_result(
    result: list[list[int]],
    n_queries: int,
    database_ids: set[int],
    top_k: int,
) -> None:
    """Check that rankings follow the required pickle submission format."""
    if type(result) is not list or len(result) != n_queries:
        raise RuntimeError(f"Result must be a list with one entry per query ({n_queries})")
    for query_index, row in enumerate(result):
        if type(row) is not list or len(row) != top_k:
            raise RuntimeError(f"Query {query_index}: expected a list of {top_k} IDs, got {row!r}")
        if any(type(image_id) is not int for image_id in row):
            raise RuntimeError(f"Query {query_index}: IDs must be Python integers")
        if len(set(row)) != top_k:
            raise RuntimeError(f"Query {query_index}: duplicated IDs in {row}")
        if not set(row) <= database_ids:
            raise RuntimeError(f"Query {query_index}: IDs not present in the database: {row}")


def generate_submissions(
    database_dir: Path,
    query_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    top_k: int = 10,
    methods: Mapping[str, HistogramConfig] = SUBMISSION_METHODS,
) -> dict[str, object]:
    """Create, reload, and validate one result pickle per submission method."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    database = load_image_dataset(database_dir)
    queries = load_image_dataset(query_dir)
    if top_k > len(database):
        raise ValueError(f"top_k={top_k} exceeds the database size ({len(database)})")

    database_ids = {int(image_id) for image_id in database.ids}
    manifest_methods: dict[str, object] = {}
    manifest: dict[str, object] = {
        "k": top_k,
        "query_ids": queries.ids.tolist(),
        "methods": manifest_methods,
    }

    for folder_name, config in methods.items():
        retriever = ColorHistogramRetrieval(config).fit(database)
        rankings, _ = retriever.retrieve(queries, top_k=top_k)
        # Store plain Python integers because the evaluator expects a portable pickle.
        result = [[int(image_id) for image_id in row] for row in rankings]
        validate_result(result, len(queries), database_ids, top_k)

        output = output_dir / folder_name / "result.pkl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            pickle.dump(result, stream)
        with output.open("rb") as stream:
            loaded = pickle.load(stream)
        if loaded != result:
            raise RuntimeError(f"Could not verify {output}")
        validate_result(loaded, len(queries), database_ids, top_k)

        manifest_methods[folder_name] = config.as_dict()
        print(f"{folder_name}: {config.name} -> {output}")
        print(" first query:", result[0])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(queries)} queries (ordered by filename); manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for QST1 prediction generation."""
    parser = argparse.ArgumentParser(
        description="Generate the two validated Week 1 QST1 result.pkl files.",
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
        default=Path("data/qst1_w1"),
        help="blind-test query image directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/week1/QST1"),
        help="output directory for method folders",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/week1/qst1_manifest.json"),
        help="output JSON manifest",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="number of retrieved database IDs per query",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate blind-test files from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    generate_submissions(
        database_dir=project_path(args.database_dir),
        query_dir=project_path(args.query_dir),
        output_dir=project_path(args.output_dir),
        manifest_path=project_path(args.manifest),
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
