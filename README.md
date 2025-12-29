# Alpha Soup Wave (v2)

This repository simulates an overlapping substrate "soup" with:

- **M3**: degree-3 backbone network.
- **M2**: emergent coherent/loop-closed subset.
- **M1**: degree-1 strings (1D waveguides) attached to anchors.

Coupling is mediated by impedance mismatch and resonance. The pipeline searches for emergent ratios of internal scales that approximate the fine-structure constant (without hardcoding it).

## Quick start

```bash
poetry install
poetry run alpha-wave demo --out outputs/demo
```

## CLI

```bash
alpha-wave demo --out outputs/demo
alpha-wave sweep --out outputs/sweep_small
alpha-wave refine --in outputs/sweep_small --out outputs/refined
alpha-wave report --in outputs/refined
```

## Outputs

Each run writes to the output directory (or `outputs/run_<timestamp>/`) containing:

- `results.csv`
- `best_candidates.json`
- `report.md`
- `figures/` with `soup_best.png`, `alpha_hist.png`, `robustness.png`

## Tests

```bash
poetry run pytest
```
