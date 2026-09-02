import pytest
import numpy as np
from saddm.model import DDMModel

# Parameters in s=1 scale:
#   a ~ 0.65-2.4, v ~ 0.1-0.5, ter ~ 0.1-0.6, sv ~ 0.1-0.3

@pytest.fixture
def model():
    return DDMModel(n_points=15)

def test_model_initialization():
    """Test model initialization with different quadrature points."""
    model_default = DDMModel()
    model_custom = DDMModel(n_points=20)

    assert model_default.integrator.n_points == 15
    assert model_custom.integrator.n_points == 20

def test_model_pdf_basic(model):
    """Test basic PDF calculations with s=1 params."""
    rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

    p = model.pdf(rt, a, z, v, ter)
    assert p > model.min_p
    assert np.isfinite(p)

    p_neg = model.pdf(rt, a, z, -v, ter)
    assert p_neg > model.min_p
    assert p_neg != p

def test_model_pdf_sv(model):
    """Test PDF with drift variability."""
    rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

    p_base = model.pdf(rt, a, z, v, ter)
    p_sv = model.pdf(rt, a, z, v, ter, sv=0.15)

    assert abs(p_sv - p_base) > 1e-6
    assert p_sv > model.min_p

def test_model_pdf_sz_st(model):
    """Test PDF with starting point and non-decision time variability."""
    rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

    p_base = model.pdf(rt, a, z, v, ter)
    p_sz = model.pdf(rt, a, z, v, ter, sz=0.1)
    p_st = model.pdf(rt, a, z, v, ter, st=0.05)
    p_both = model.pdf(rt, a, z, v, ter, sz=0.1, st=0.05)

    assert len(set([p_base, p_sz, p_st, p_both])) == 4
    assert all(p > model.min_p for p in [p_base, p_sz, p_st, p_both])

def test_model_pdf_sa(model):
    """Test PDF with boundary separation variability."""
    rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

    p_base = model.pdf(rt, a, z, v, ter)
    p_sa = model.pdf(rt, a, z, v, ter, sa=0.2)

    assert p_sa > model.min_p
    assert abs(p_sa - p_base) > 1e-6

def test_model_parameter_boundaries(model):
    """Test parameter boundary handling."""
    rt = 0.5

    # Below min a -> should return min_p
    p1 = model.pdf(rt, 0.005, 0.5, 0.3, 0.3)
    assert p1 == model.min_p

    # Negative ter -> should return min_p
    p2 = model.pdf(rt, 1.2, 0.5, 0.3, -0.1)
    assert p2 == model.min_p

    # z out of range -> should return min_p
    p3 = model.pdf(rt, 1.2, 1.5, 0.3, 0.3)
    assert p3 == model.min_p

def test_model_log_likelihood(model):
    """Test log likelihood calculations."""
    data = [
        (0.5, 0),
        (0.6, 1),
        (0.45, 0),
    ]

    # params: [a, z, v, ter, sv, sz, st, sa]
    params = [1.2, 0.5, 0.3, 0.3, 0.15, 0.0, 0.0, 0.0]

    ll = model.log_likelihood(params, data)
    assert np.isfinite(ll)

    params_neg = params.copy()
    params_neg[2] = -0.3
    ll_neg = model.log_likelihood(params_neg, data)
    assert ll_neg != ll

def test_model_log_likelihood_with_variability(model):
    """Test log likelihood with all variability parameters."""
    data = [
        (0.5, 0),
        (0.6, 1),
        (0.45, 0),
    ]

    params = [1.2, 0.5, 0.3, 0.3, 0.15, 0.1, 0.05, 0.2]

    ll = model.log_likelihood(params, data)
    assert np.isfinite(ll)

def test_model_response_types(model):
    """Test model handling of different response types."""
    data = [(0.5, 0), (0.5, 1)]
    params = [1.2, 0.5, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]

    ll = model.log_likelihood(params, data)
    assert np.isfinite(ll)

def test_model_extreme_rts(model):
    """Test handling of extreme response times."""
    a, z, v, ter = 1.2, 0.5, 0.3, 0.3

    p_near = model.pdf(0.35, a, z, v, ter)
    p_long = model.pdf(5.0, a, z, v, ter)

    assert p_near > model.min_p
    assert p_long >= model.min_p
    assert p_near > p_long

def test_model_full_variability(model):
    """Test model with all variability parameters active."""
    rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

    p = model.pdf(rt, a, z, v, ter, sv=0.15, sz=0.1, st=0.05, sa=0.2)
    assert p > model.min_p
    assert np.isfinite(p)

def test_model_invalid_params(model):
    """Test model behavior with invalid parameters."""
    data = [(0.5, 0), (0.6, 1)]

    invalid_params_list = [
        [-1.0, 0.5, 0.3, 0.3, 0, 0, 0, 0],   # Negative boundary
        [1.2, -0.5, 0.3, 0.3, 0, 0, 0, 0],    # Negative starting point
        [1.2, 0.5, 0.3, -0.1, 0, 0, 0, 0],    # Negative non-decision time
        [1.2, 0.5, 0.3, 0.3, -1, 0, 0, 0],    # Negative sv
        [1.2, 0.5, 0.3, 0.3, 0, -1, 0, 0],    # Negative sz
        [1.2, 0.5, 0.3, 0.3, 0, 0, -1, 0],    # Negative st
        [1.2, 0.5, 0.3, 0.3, 0, 0, 0, -1],    # Negative sa
    ]

    for params in invalid_params_list:
        ll = model.log_likelihood(params, data)
        assert ll == -np.inf or np.isfinite(ll)
