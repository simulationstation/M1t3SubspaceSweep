# Alpha EM Soup (v4)

This repository simulates an overlapping substrate "soup" where coupling emerges from EM-like wave geometry (phase accumulation, interference, resonance, attenuation). It **never** hardcodes 1/137 into the dynamics—`ALPHA_TARGET` is only used for reporting comparison.

## Model overview

- **M3**: loop-rich degree-3 substrate graph.
- **M2**: emergent or constructed coherent interface (graph-wave medium).
- **M1**: 1D wave paths (paths + loops) anchored on M2 nodes.

### Coupling definitions (v4)

At frequency ω, we solve a damped Helmholtz response on the coupled block system and compute layer-dissipated power. Effective coupling ratios use RMS across the band:

```
P_layer(ω) = ω^2 Σ_i D_i |ψ_i|^2

g21(ω) = sqrt(P_M1 / P_total)
g23(ω) = sqrt(P_M3 / P_total)
Q = g23_eff / g21_eff^2
```

The search enforces **Q ≈ 1** structurally before comparing `g21_eff` to `ALPHA_TARGET`.

## Quick start

```bash
poetry install
poetry run alpha-em demo --out outputs/demo
```

## CLI

```bash
alpha-em demo --out outputs/demo
alpha-em sweep --config configs/sweep_small.json --out outputs/sweep_small
alpha-em refine --in outputs/sweep_small --out outputs/refined
alpha-em report --in outputs/refined
```

## Outputs

Each run writes:

- `results.csv` (per-seed metrics)
- `summary.csv` (aggregated metrics)
- `report.md`
- `figures/` (`soup_best.png`, distributions, stability plots)

## Tests

```bash
poetry run pytest
```
