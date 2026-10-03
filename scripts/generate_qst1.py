import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ColorHistogramRetrieval, load_image_dataset
from src.methods import SUBMISSION_METHODS

K = 10

def validate_result(result, n_queries: int, database_ids: set[int]) -> None:
    """Check that `result` follows the required submission format."""
    if type(result) is not list or len(result) != n_queries:
        raise RuntimeError(f"Result must be a list with one entry per query ({n_queries})")
    for q, row in enumerate(result):
        if type(row) is not list or len(row) != K:
            raise RuntimeError(f"Query {q}: expected a list of {K} IDs, got {row!r}")
        if any(type(i) is not int for i in row):
            raise RuntimeError(f"Query {q}: IDs must be Python integers")
        if len(set(row)) != K:
            raise RuntimeError(f"Query {q}: duplicated IDs in {row}")
        if not set(row) <= database_ids:
            raise RuntimeError(f"Query {q}: IDs not present in the database: {row}")


database = load_image_dataset(ROOT / "data/BBDD")
queries = load_image_dataset(ROOT / "data/qst1_w1")
database_ids = {int(i) for i in database.ids}
submission_root = ROOT / "outputs/week1/QST1"
# Persist the query order alongside the blind predictions for reproducibility.
manifest = {"k": K, "query_ids": queries.ids.tolist(), "methods": {}}

for method_name, config in SUBMISSION_METHODS.items():
    retriever = ColorHistogramRetrieval(config).fit(database)
    rankings, _ = retriever.retrieve(queries, top_k=K)
    # Plain Python list of lists of ints (not numpy types), best match first.
    result = [[int(image_id) for image_id in row] for row in rankings]
    validate_result(result, len(queries), database_ids)

    output = submission_root / method_name / "result.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(result, stream)
    with output.open("rb") as stream:
        loaded = pickle.load(stream)
    if loaded != result:
        raise RuntimeError(f"Could not verify {output}")
    validate_result(loaded, len(queries), database_ids)

    manifest["methods"][method_name] = config.as_dict()
    print(f"{method_name}: {config.name} -> {output.relative_to(ROOT)}")
    print(" first query:", result[0])

# Kept outside the delivery tree so only result.pkl files are uploaded.
manifest_path = ROOT / "outputs/week1/qst1_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Saved {len(queries)} queries (ordered by numeric query ID); manifest: {manifest_path.relative_to(ROOT)}")
