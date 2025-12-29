"""Report generation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_soup.search.robustness import summarize_configs, stability_filter
from alpha_soup.constants import ALPHA_TARGET


def write_report(results: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_configs(results)
    best = summary.sort_values("loss_mean").head(5)
    stable = stability_filter(results, tolerance=0.2)

    lines = ["# Alpha Soup Report", "", "## Overview", ""]
    lines.append(f"Target alpha: {ALPHA_TARGET:.12f}")
    lines.append(f"Total runs: {len(results)}")
    lines.append("")
    lines.append("## Best candidates (by mean loss)")
    lines.append("")
    lines.append(best.to_markdown(index=False))
    lines.append("")
    lines.append("## Filter outcomes")
    lines.append("")
    lines.append(f"Configs passing stability filter: {len(stable)} / {summary.shape[0]}")
    lines.append(
        "Configs failing stability filter: {0} / {1}".format(
            summary.shape[0] - len(stable), summary.shape[0]
        )
    )
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "Candidates are filtered by mean loss. Robustness should be assessed with repeated seeds and neighborhood perturbations."
    )
    out_path.write_text("\n".join(lines))
