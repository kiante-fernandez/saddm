"""Properties the Numba reference must hold before it can serve as an oracle."""

import numpy as np
import pytest

from reference import DDMModel, ddm_pdf_core, integrate

A, Z, V, TER = 1.2, 0.5, 0.3, 0.3


@pytest.fixture
def model():
    return DDMModel(n_points=15)


def test_no_variability_matches_core(model):
    got = integrate(0.5, A, Z, V, TER, 0.15, 0.0, 0.0, 0.0, model.nodes, model.weights)
    assert abs(got - ddm_pdf_core(0.5, A, Z, V, TER, 0.15)) < 1e-10


def test_tiny_widths_match_no_variability(model):
    """Widths just above the eps=1e-6 activation threshold change nothing."""
    base = model.pdf(0.5, A, Z, V, TER, sv=0.15)
    tiny = model.pdf(0.5, A, Z, V, TER, sv=0.15, sz=2e-6, st=2e-6, sa=2e-6)
    assert abs(tiny - base) / base < 1e-2


def test_each_width_changes_the_density(model):
    base = model.pdf(0.5, A, Z, V, TER)
    ps = [model.pdf(0.5, A, Z, V, TER, **w) for w in (dict(sv=0.15), dict(sz=0.1),
                                                     dict(st=0.05), dict(sa=0.2))]
    assert all(abs(p - base) > 1e-6 for p in ps) and len(set(ps)) == 4


def test_both_boundaries_integrate_to_one():
    dt = 0.001
    rts = np.arange(TER + dt, 5.0, dt)
    lower = sum(ddm_pdf_core(rt, A, Z, V, TER, 0.0) for rt in rts) * dt
    upper = sum(ddm_pdf_core(rt, A, 1 - Z, -V, TER, 0.0) for rt in rts) * dt
    assert 0 < lower < 1 and abs(lower + upper - 1.0) < 0.05


def test_zero_before_ter(model):
    assert ddm_pdf_core(TER, A, Z, V, TER, 0.0) == 0.0
    assert model.pdf(TER - 0.05, A, Z, V, TER) == model.min_p


def test_choice_coding(model):
    """choice=1 with (z, v) must equal choice=0 with (1-z, -v), and positive drift
    favours the upper response."""
    z, v = 0.3, 0.4
    up = model.log_likelihood([A, z, v, TER, 0, 0, 0, 0], [(0.5, 1)])
    assert abs(up - model.log_likelihood([A, 1 - z, -v, TER, 0, 0, 0, 0], [(0.5, 0)])) < 1e-10
    p = [A, Z, V, TER, 0, 0, 0, 0]
    assert model.log_likelihood(p, [(0.5, 1)]) > model.log_likelihood(p, [(0.5, 0)])


def test_invalid_params_give_neg_inf(model):
    data = [(0.5, 0), (0.6, 1)]
    for i in (0, 1, 3, 4, 5, 6, 7):
        p = [A, Z, V, TER, 0, 0, 0, 0]
        p[i] = -1.0
        assert model.log_likelihood(p, data) == -np.inf
    assert model.log_likelihood([A, 1.5, V, TER, 0, 0, 0, 0], data) == -np.inf


def test_slow_task_ter(model):
    """ter above 1 s is accepted whenever every RT exceeds it."""
    data = [(1.6, 0), (1.8, 1), (2.1, 0)]
    assert np.isfinite(model.log_likelihood([A, Z, V, 1.1, 0, 0, 0, 0], data))
    assert model.log_likelihood([A, Z, V, 2.5, 0, 0, 0, 0], data) == 3 * np.log(model.min_p)
