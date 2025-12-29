# Alpha Soup Wave (v3)

This repository simulates an overlapping substrate "soup" and **searches** for emergent ratios that can land near the fine-structure constant. It does **not** derive alpha from first principles or hardcode 1/137 anywhere.

## Model overview

- **M3**: degree-3 backbone network (XY phases on a random regular graph).
- **M2**: emergent coherent/loop-closed subset detected from M3.
- **M1**: string-like waveguides attached to anchors (length/tension/density/damping/boundary conditions).

### Mechanisms (v3)

Each mechanism is a *physically meaningful small-parameter pathway* producing a bounded ratio in (0,1):

1. **Exponential attenuation**: transmission amplitude decays as `exp(-L/ℓ_d)`.
2. **Spectral mismatch**: overlap of M2 spectral density with string-mode Lorentzians.
3. **Selection-rule second order**: direct M2↔M3 coupling is orthogonal; effective coupling appears via string modes.
4. **Duty-cycle gating**: rare coherence and resonance events multiply into a small amplitude proxy.

Robustness filters reject fine-tuned needle hits by checking local plateaus and variance explained by any single parameter.

## Quick start

```bash
poetry install
poetry run alpha-soup-wave demo --out outputs/demo
```

## CLI

```bash
alpha-soup-wave demo --out outputs/demo
alpha-soup-wave sweep --config configs/sweep_small.json --out outputs/sweep_small
alpha-soup-wave refine --in outputs/sweep_small --out outputs/refined
alpha-soup-wave report --in outputs/refined
```

## Outputs

Each run writes to the output directory (or `outputs/run_<timestamp>/`) containing:

- `results.csv` (per-config + per-seed metrics)
- `summary.csv` (aggregated metrics + robustness flags)
- `best_candidates.json`
- `report.md`
- `figures/` with `soup_best.png`, `alpha_hist_*.png`, `seed_stability.png`, `sensitivity.png`

## Tests

```bash
poetry run pytest
```
