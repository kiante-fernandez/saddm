import pytest
import numpy as np
from math import sqrt, pi, exp
from saddm.integrator import DDMIntegrator
from saddm.core import ddm_pdf_core

# Parameters in s=1 scale:
#   a ~ 0.65-2.4, v ~ 0.1-0.5, ter ~ 0.1-0.6, sv ~ 0.1-0.3

@pytest.fixture
def integrator():
    return DDMIntegrator(n_points=15)

def _quad_args(integrator):
    """Helper to build the full quadrature args tuple."""
    return (
        integrator.sz_nodes, integrator.sz_weights,
        integrator.st_nodes, integrator.st_weights,
        integrator.sa_nodes, integrator.sa_weights,
    )

def test_integrator_initialization(integrator):
    """Test proper initialization of quadrature nodes and weights."""
    assert len(integrator.sz_nodes) == integrator.n_points
    assert len(integrator.sz_weights) == integrator.n_points
    assert len(integrator.st_nodes) == integrator.n_points
    assert len(integrator.st_weights) == integrator.n_points
    assert len(integrator.sa_nodes) == integrator.n_points
    assert len(integrator.sa_weights) == integrator.n_points

def test_integrator_no_variability(integrator):
    """Test that integration matches core PDF when sz, st, sa are zero."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    integrated = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    direct = ddm_pdf_core(rt, a, z, v, ter, sv)

    assert abs(integrated - direct) < 1e-10, "Should exactly match core PDF when no variability"

def test_integrator_sz_effects(integrator):
    """Test effects of starting point variability."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    base = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    with_sz = integrator.integrate(rt, a, z, v, ter, sv, 0.1, 0.0, 0.0, *args)

    assert with_sz > 0 and np.isfinite(with_sz)
    assert abs(with_sz - base) > 1e-6

def test_integrator_st_effects(integrator):
    """Test effects of non-decision time variability."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    base = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    with_st = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.1, 0.0, *args)

    assert with_st > 0 and np.isfinite(with_st)
    assert abs(with_st - base) > 1e-6

def test_integrator_sa_effects(integrator):
    """Test effects of boundary separation variability."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    base = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    with_sa = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.2, *args)

    assert with_sa > 0 and np.isfinite(with_sa)
    assert abs(with_sa - base) > 1e-6

def test_integrator_combined_variability(integrator):
    """Test combined effects of sz, st, and sa."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    base = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    sz_only = integrator.integrate(rt, a, z, v, ter, sv, 0.1, 0.0, 0.0, *args)
    st_only = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.1, 0.0, *args)
    sa_only = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.2, *args)
    all_var = integrator.integrate(rt, a, z, v, ter, sv, 0.1, 0.1, 0.2, *args)

    assert len(set([base, sz_only, st_only, sa_only, all_var])) == 5
    assert all(p > 0 and np.isfinite(p) for p in [base, sz_only, st_only, sa_only, all_var])

def test_integrator_extreme_cases(integrator):
    """Test that very small variability matches no-variability case."""
    rt, a, z, v, ter, sv = 0.5, 1.2, 0.5, 0.3, 0.3, 0.15
    args = _quad_args(integrator)

    no_var = integrator.integrate(rt, a, z, v, ter, sv, 0.0, 0.0, 0.0, *args)
    tiny_var = integrator.integrate(rt, a, z, v, ter, sv, 1e-10, 1e-10, 1e-10, *args)

    assert abs(tiny_var - no_var) / max(abs(no_var), 1e-10) < 1e-2

def test_integrator_boundary_times(integrator):
    """Test integration near temporal boundaries."""
    a, z, v, ter, sv = 1.2, 0.5, 0.3, 0.3, 0.15
    sz, st = 0.1, 0.05
    args = _quad_args(integrator)

    near_ter = integrator.integrate(ter + 0.02, a, z, v, ter, sv, sz, st, 0.0, *args)
    assert near_ter > 0 and np.isfinite(near_ter)

    long_rt = integrator.integrate(2.0, a, z, v, ter, sv, sz, st, 0.0, *args)
    assert long_rt > 0 and np.isfinite(long_rt)
