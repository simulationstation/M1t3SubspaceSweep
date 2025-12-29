# Alpha Soup

This repository simulates a degree-stratified computing "soup" (M3/M2/M1 analogs) and searches for regimes where an emergent ratio of internal scales matches the fine-structure constant target (α ≈ 1/137). The implementation uses an XY-like phase model, degree-stratified couplings, emergent M2 detection via coherence/closure, and a search pipeline with robustness checks.

## Quick start

```bash
poetry install
poetry run alpha-soup demo --out outputs/demo
```

## CLI commands

```bash
alpha-soup demo --out outputs/demo
alpha-soup sweep --config configs/sweep_small.json --out outputs/sweep_small
alpha-soup refine --in outputs/sweep_small --out outputs/refined
alpha-soup report --in outputs/refined --out outputs/refined/report.md
```

## Outputs

Each run writes to `outputs/run_<timestamp>/` (or the specified output path) including:

- `configs_used.json` and `results.csv`
- `best_candidates.json`
- `figures/` containing `soup_best.png`, `alpha_scatter.png`, `robustness.png`
- `report.md`

## Tests

```bash
poetry run pytest
```
