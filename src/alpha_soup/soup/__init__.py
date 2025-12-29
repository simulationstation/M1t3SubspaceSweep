"""Soup simulation components."""
from .graph_gen import generate_degree_graph, apply_decay
from .dynamics import initialize_state, evolve, SoupState
from .emergent_m2 import detect_m2
from .observables import compute_observables

__all__ = [
    "generate_degree_graph",
    "apply_decay",
    "initialize_state",
    "evolve",
    "SoupState",
    "detect_m2",
    "compute_observables",
]
