from alpha_soup.config import DegreeProfile
from alpha_soup.soup.graph_gen import generate_degree_graph


def test_graph_generation_counts():
    profile = DegreeProfile(frac_deg3=0.6, frac_deg2=0.25, frac_deg1=0.15)
    bundle = generate_degree_graph(100, profile, seed=10)
    counts = {1: 0, 2: 0, 3: 0}
    for deg in bundle.degree_labels.values():
        counts[deg] += 1
    assert sum(counts.values()) == 100
    assert counts[3] >= counts[1]
