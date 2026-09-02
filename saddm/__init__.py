"""
saddm - Drift Diffusion Model with across-trial variability in boundary separation.

Implements the DDM with variability parameters (sv, sz, st, sa).
Diffusion coefficient s=1.

Parameters:
    a:  boundary separation (~0.65-2.4)
    z:  relative starting point (0 to 1)
    v:  drift rate (~0.1-0.5)
    ter: non-decision time (seconds)
    sv: Gaussian drift rate variability
    sz: uniform starting point variability
    st: uniform non-decision time variability
    sa: uniform boundary separation variability
"""

__version__ = "0.1.0"

from .core import ddm_pdf_core, ftt_01w
from .integrator import DDMIntegrator
from .model import DDMModel
from .ddmsa import (
    DDMSA,
    ddmsa_logp,
    ddmsa_potential,
    make_ddmsa_model,
    sample_ddmsa,
    sample_ddmsa_exact,
    simulate_ddmsa,
)

__all__ = [
    "ddm_pdf_core",
    "ftt_01w",
    "DDMIntegrator",
    "DDMModel",
    "ddmsa_logp",
    "ddmsa_potential",
    "DDMSA",
    "make_ddmsa_model",
    "sample_ddmsa",
    "sample_ddmsa_exact",
    "simulate_ddmsa",
]
