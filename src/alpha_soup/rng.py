"""Random helpers."""
from __future__ import annotations

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = 0
    return np.random.default_rng(seed)
