import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ColorHistogramRetrieval, load_image_dataset
from src.methods import SUBMISSION_METHODS


database = load_image_dataset(ROOT / "data/BBDD")
queries = load_image_dataset(ROOT / "data/qst1_w1")
submission_root = ROOT / "submissions/Team3/week1/QST1"
# Persist the query order alongside the blind predictions for reproducibility.
manifest = {"query_ids": queries.ids.tolist(), "methods": {}}

for method_name, config in SUBMISSION_METHODS.items():
    retriever = ColorHistogramRetrieval(config).fit(database)
    rankings, _ = retriever.retrieve(queries, top_k=10)
    result = [[int(image_id) for image_id in row] for row in rankings]
    if len(result) != len(queries) or any(len(row) != 10 for row in result):
        raise RuntimeError("Submission must contain ten database IDs per query")

    output = submission_root / method_name / "result.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(result, stream)
    with output.open("rb") as stream:
        if pickle.load(stream) != result:
            raise RuntimeError(f"Could not verify {output}")

    manifest["methods"][method_name] = config.as_dict()
    print(f"{method_name}: {config.name} -> {output.relative_to(ROOT)}")
    print(" first query:", result[0])

submission_root.mkdir(parents=True, exist_ok=True)
(submission_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Saved manifest for {len(queries)} queries in sorted query-ID order")
