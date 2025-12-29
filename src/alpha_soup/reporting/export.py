"""Export utilities for run outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_soup.search.robustness import best_candidates


def export_best(results: pd.DataFrame, out_path: Path, top_k: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = best_candidates(results, top_k)
    out_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
