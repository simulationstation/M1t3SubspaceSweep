"""Gate computations for coherence and resonance duty cycle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    pA: float
    pB: float
    pAB: float


def gate_probabilities(m2_coherence: float, resonance_fraction: float, coherence_min: float,
                       resonance_min: float) -> GateResult:
    pA = float(min(1.0, max(0.0, m2_coherence / max(coherence_min, 1e-6))))
    pB = float(min(1.0, max(0.0, resonance_fraction / max(resonance_min, 1e-6))))
    pAB = float(min(1.0, pA * pB))
    return GateResult(pA=pA, pB=pB, pAB=pAB)
