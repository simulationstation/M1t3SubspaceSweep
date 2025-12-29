from alpha_em_soup.physics.effective_coupling import compute_q_ratio


def test_q_ratio_synthetic():
    g21 = 0.2
    g23 = g21**2
    assert compute_q_ratio(g21, g23) == 1.0
