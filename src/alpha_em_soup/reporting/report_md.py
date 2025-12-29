"""Markdown reporting for EM soup runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_report(input_dir: str | Path, output_path: str | Path) -> Path:
    input_dir = Path(input_dir)
    summary_path = input_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    best = summary.sort_values("score").head(5)
    lines = ["# EM Soup Report", "", "## Top candidates", ""]
    for _, row in best.iterrows():
        lines.append(
            f"- Config {row['config_id']}: Q={row['q_ratio']:.3f}, g21={row['g21_eff']:.4f}, "
            f"g23={row['g23_eff']:.4f}, score={row['score']:.3f}"
        )
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))
    return output_path
