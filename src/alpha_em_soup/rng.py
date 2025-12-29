"""Deterministic RNG helpers."""

from __future__ import annotations

import numpy as np


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))
