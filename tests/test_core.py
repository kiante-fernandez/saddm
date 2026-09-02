import pytest
import numpy as np
from math import sqrt, pi, exp
from saddm.core import ddm_pdf_core, ftt_01w

# Parameters in s=1 scale:
#   a ~ 0.65-2.4, v ~ 0.1-0.5, ter ~ 0.1-0.6, sv ~ 0.1-0.3

def test_ftt_01w_basic():
    """Test the Navarro-Fuss first-passage time density."""
    p = ftt_01w(0.5, 0.5, 1e-4)
    assert p > 0 and np.isfinite(p)

    p_lo = ftt_01w(0.5, 0.2, 1e-4)
    p_hi = ftt_01w(0.5, 0.8, 1e-4)
    assert p_lo > 0 and p_hi > 0
    assert p_lo != p_hi

def test_ddm_pdf_core_basic():
    """Test basic PDF calculation without variability (s=1 params)."""
    rt = 0.5
    a = 1.2
    z = 0.5
    v = 0.3
    ter = 0.3

    p = ddm_pdf_core(rt, a, z, v, ter, 0.0)
    assert p > 0 and np.isfinite(p)

    p_neg = ddm_pdf_core(rt, a, z, -v, ter, 0.0)
    assert p_neg > 0 and np.isfinite(p_neg)
    assert p != p_neg

def test_ddm_pdf_core_sv():
    """Test PDF calculation with drift variability."""
    rt = 0.5
    a = 1.2
    z = 0.5
    v = 0.3
    ter = 0.3
    sv = 0.15

    p_sv = ddm_pdf_core(rt, a, z, v, ter, sv)
    p_no_sv = ddm_pdf_core(rt, a, z, v, ter, 0.0)

    assert p_sv > 0 and np.isfinite(p_sv)
    assert abs(p_sv - p_no_sv) > 1e-6

def test_ddm_pdf_core_edge_cases():
    """Test edge cases in PDF calculation."""
    assert ddm_pdf_core(0.3, 1.2, 0.5, 0.3, 0.35, 0.0) == 0.0
    assert ddm_pdf_core(0.0, 1.2, 0.5, 0.3, 0.1, 0.0) == 0.0
    assert ddm_pdf_core(0.5, -1.0, 0.5, 0.3, 0.1, 0.0) == 0.0

    p = ddm_pdf_core(0.5, 1.2, 0.5, 1e-10, 0.3, 0.0)
    assert p > 0 and np.isfinite(p)

    p_sv = ddm_pdf_core(0.5, 1.2, 0.5, 1e-10, 0.3, 0.15)
    assert p_sv > 0 and np.isfinite(p_sv)

def test_ddm_pdf_core_scaling():
    """Test that changing boundary changes the PDF."""
    rt = 0.5
    z = 0.5
    v = 0.3
    ter = 0.3

    p_small_a = ddm_pdf_core(rt, 0.8, z, v, ter, 0.0)
    p_large_a = ddm_pdf_core(rt, 2.0, z, v, ter, 0.0)
    assert p_small_a > 0 and p_large_a > 0
    assert abs(p_small_a - p_large_a) > 1e-6

def test_ddm_pdf_core_starting_point():
    """Test starting point effects on PDF."""
    rt = 0.5
    a = 1.2
    v = 0.3
    ter = 0.3

    lower = ddm_pdf_core(rt, a, 0.2, v, ter, 0.0)
    center = ddm_pdf_core(rt, a, 0.5, v, ter, 0.0)
    upper = ddm_pdf_core(rt, a, 0.8, v, ter, 0.0)

    assert all(p > 0 for p in [lower, center, upper])
    assert len(set([lower, center, upper])) == 3

def test_ddm_pdf_core_positive_density_range():
    """Test that PDF is positive over a range of reasonable RTs."""
    a = 1.2
    z = 0.5
    v = 0.3
    ter = 0.3

    for rt in [0.35, 0.4, 0.5, 0.6, 0.8, 1.0]:
        p = ddm_pdf_core(rt, a, z, v, ter, 0.0)
        assert p > 0 and np.isfinite(p), f"PDF should be positive at rt={rt}"
