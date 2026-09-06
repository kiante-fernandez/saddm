"""
saddm - Drift Diffusion Model with across-trial variability in boundary separation.

Parameters (s = 1): a boundary separation, z relative start point in (0, 1),
v drift rate, t non-decision time (seconds); sv Gaussian drift variability;
sa, st, sz uniform full-width variability of boundary, non-decision time, and
start point.
"""

__version__ = "0.2.0"

from .ddmsa import *  # noqa: F401,F403
from .ddmsa import __all__  # noqa: F401
