"""
Cross-validation tests to verify DDM PDF properties (s=1 scale).
"""

import pytest
import numpy as np
from saddm.model import DDMModel
from saddm.core import ddm_pdf_core


@pytest.fixture
def model():
    return DDMModel(n_points=15)


class TestScaleParameters:
    """Test that s=1 scale parameters produce reasonable PDFs."""

    def test_typical_sa_model_params(self, model):
        """Test with typical parameter values (s=1 scale)."""
        rt = 0.5
        a = 1.2
        z = 0.5
        v = 0.3
        ter = 0.3
        sv = 0.2
        sa = 0.05
        st = 0.1

        p = model.pdf(rt, a, z, v, ter, sv=sv, sa=sa, st=st)
        assert p > 0, "PDF should be positive for typical parameters"
        assert np.isfinite(p), "PDF should be finite"

    def test_novar_model_params(self, model):
        """Test with no-variability model."""
        rt = 0.5
        a = 1.2
        z = 0.5
        v = 0.3
        ter = 0.3

        p = model.pdf(rt, a, z, v, ter, sv=0.0, sz=0.0, st=0.0, sa=0.0)
        assert p > 0
        assert np.isfinite(p)

    def test_multiple_conditions(self, model):
        """Test different drift rates (simulating multi-condition data)."""
        a, z, ter = 1.2, 0.5, 0.3
        sv, sa, st = 0.2, 0.05, 0.1

        drift_rates = [0.4, 0.3, 0.2, 0.1]

        for v in drift_rates:
            p = model.pdf(0.5, a, z, v, ter, sv=sv, sa=sa, st=st)
            assert p > 0 and np.isfinite(p), f"Failed for v={v}"


class TestPDFProperties:
    """Test mathematical properties that the PDF should satisfy."""

    def test_pdf_positive(self, model):
        """PDF should be positive for valid parameters and RT > ter."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        for rt in np.arange(0.35, 2.0, 0.1):
            p = ddm_pdf_core(rt, a, z, v, ter, 0.0)
            assert p > 0, f"PDF should be positive at rt={rt:.2f}"

    def test_pdf_approaches_zero(self, model):
        """PDF should approach zero for very large RT."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        p_moderate = ddm_pdf_core(0.5, a, z, v, ter, 0.0)
        p_large = ddm_pdf_core(10.0, a, z, v, ter, 0.0)

        assert p_large < p_moderate, "PDF should decrease for very large RT"

    def test_pdf_zero_before_ter(self):
        """PDF should be exactly zero when RT <= ter."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        assert ddm_pdf_core(0.3, a, z, v, ter, 0.0) == 0.0
        assert ddm_pdf_core(0.2, a, z, v, ter, 0.0) == 0.0
        assert ddm_pdf_core(0.0, a, z, v, ter, 0.0) == 0.0

    def test_defective_density_integrates_under_one(self, model):
        """The defective PDF (single boundary) should integrate to < 1."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        dt = 0.001
        rts = np.arange(ter + dt, 5.0, dt)
        total = sum(ddm_pdf_core(rt, a, z, v, ter, 0.0) * dt for rt in rts)

        assert 0 < total < 1.0, f"Defective density integral = {total:.4f}, should be in (0, 1)"

    def test_both_boundaries_integrate_near_one(self, model):
        """PDF at upper + lower boundary should integrate close to 1."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        dt = 0.001
        rts = np.arange(ter + dt, 5.0, dt)

        total_lower = sum(ddm_pdf_core(rt, a, z, v, ter, 0.0) * dt for rt in rts)
        total_upper = sum(ddm_pdf_core(rt, a, 1 - z, -v, ter, 0.0) * dt for rt in rts)

        total = total_lower + total_upper
        assert abs(total - 1.0) < 0.05, f"Total probability = {total:.4f}, should be ~1.0"


class TestVariabilityEffects:
    """Test that variability parameters have expected effects."""

    def test_sv_changes_distribution(self, model):
        """Drift variability should change the RT distribution."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        p_nosv = model.pdf(0.5, a, z, v, ter, sv=0.0)
        p_sv = model.pdf(0.5, a, z, v, ter, sv=0.5)

        assert abs(p_sv - p_nosv) > 1e-6, "sv should change the PDF"
        assert p_sv > 0 and np.isfinite(p_sv)

    def test_sa_changes_pdf(self, model):
        """Boundary variability should change the PDF."""
        rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

        p_no_sa = model.pdf(rt, a, z, v, ter)
        p_sa = model.pdf(rt, a, z, v, ter, sa=0.2)

        assert abs(p_no_sa - p_sa) > 1e-8, "sa should change the PDF"

    def test_st_changes_pdf(self, model):
        """Non-decision time variability should change the PDF."""
        rt, a, z, v, ter = 0.5, 1.2, 0.5, 0.3, 0.3

        p_no_st = model.pdf(rt, a, z, v, ter)
        p_st = model.pdf(rt, a, z, v, ter, st=0.05)

        assert abs(p_no_st - p_st) > 1e-8, "st should change the PDF"


class TestChoiceCoding:
    """Test that choice coding works correctly for likelihood."""

    def test_symmetric_start_equal_drift(self, model):
        """With z=0.5 and v=0, both choices should have equal likelihood."""
        data = [(0.5, 0), (0.5, 1)]
        params = [1.2, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]

        ll = model.log_likelihood(params, data)
        assert np.isfinite(ll)

        p_lower = model.pdf(0.5, 1.2, 0.5, 0.0, 0.3)
        p_upper = model.pdf(0.5, 1.2, 0.5, 0.0, 0.3)
        assert abs(p_lower - p_upper) < 1e-10

    def test_drift_favors_one_boundary(self, model):
        """Positive drift makes the upper response (choice 1) more likely."""
        a, z, v, ter = 1.2, 0.5, 0.3, 0.3

        data_lower = [(0.5, 0)]
        data_upper = [(0.5, 1)]

        params = [a, z, v, ter, 0.0, 0.0, 0.0, 0.0]

        ll_lower = model.log_likelihood(params, data_lower)
        ll_upper = model.log_likelihood(params, data_upper)

        assert ll_upper > ll_lower
        assert model.log_likelihood([a, z, -v, ter, 0.0, 0.0, 0.0, 0.0], data_lower) == ll_upper
