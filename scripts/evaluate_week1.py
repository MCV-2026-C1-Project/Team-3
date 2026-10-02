from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ColorHistogramRetrieval, load_ground_truth, load_image_dataset
from src.methods import AUDIT_METHODS


database = load_image_dataset(ROOT / "data/BBDD")
queries = load_image_dataset(ROOT / "data/qsd1_w1")
ground_truth = load_ground_truth(ROOT / "data/qsd1_w1/gt_corresps.pkl")

rows = []
for config in AUDIT_METHODS:
    retriever = ColorHistogramRetrieval(config).fit(database)
    metrics, _ = retriever.evaluate(queries, ground_truth)
    rows.append({**config.as_dict(), **metrics})

results = pd.DataFrame(rows)
output = ROOT / "outputs/week1/qsd1_results.csv"
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)
print(results[["name", "mAP@1", "mAP@5", "top5_hit_rate"]].to_string(index=False))
print(f"\nSaved {output.relative_to(ROOT)}")
