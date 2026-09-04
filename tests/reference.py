"""Numba reference for the DDM-SA density, ported from fortran/fit_sa_simplex.f90
(FC/GQ/FFC/COR lineage). Shares no code with saddm.ddmsa; test_ddmsa.py holds
the PyTensor likelihood to it."""

from math import ceil, exp, floor, log, pi, sin, sqrt

import numpy as np
from numba import float64, njit
from scipy.special import roots_legendre


@njit(float64(float64, float64, float64), cache=True)
def ftt_01w(tt, w, err):
    """f(t | 0, 1, w) following Navarro & Fuss (2009)."""
    if pi * tt * err < 1:
        kl = sqrt(-2 * log(pi * tt * err) / (pi * pi * tt))
        kl = max(kl, 1. / (pi * sqrt(tt)))
    else:
        kl = 1. / (pi * sqrt(tt))

    if 2 * sqrt(2 * pi * tt) * err < 1:
        ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * pi * tt) * err))
        ks = max(ks, sqrt(tt) + 1)
    else:
        ks = 2

    p = 0.0
    if ks < kl:
        K = int(ceil(ks))
        lower = -int(floor((K - 1) / 2))
        upper = int(ceil((K - 1) / 2))
        for k in range(lower, upper + 1):
            p += (w + 2 * k) * exp(-(pow(w + 2 * k, 2)) / 2 / tt)
        p /= sqrt(2 * pi * pow(tt, 3))
    else:
        K = int(ceil(kl))
        for k in range(1, K + 1):
            p += k * exp(-(k * k) * (pi * pi) * tt / 2) * sin(k * pi * w)
        p *= pi

    return p


@njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
def ddm_pdf_core(rt, a, z, v, ter, sv):
    """Lower-boundary density at s = 1 with Gaussian drift variability."""
    if rt <= 0 or rt <= ter or a <= 0:
        return 0.0

    tt = (rt - ter) / (a * a)
    p = ftt_01w(tt, z, 1e-4)

    if sv <= 1e-10:
        return p * exp(-v * a * z - (v * v * (rt - ter)) / 2.) / (a * a)

    sv_squared_rt = sv * sv * (rt - ter)
    if sv_squared_rt > 100:
        return 0.0
    exp_term = ((a * z * sv) ** 2 - 2 * a * v * z - (v * v) * (rt - ter)) / (2 * (sv_squared_rt + 1))
    if exp_term < -500 or exp_term > 500:
        return 0.0
    return p * exp(exp_term) / sqrt(sv_squared_rt + 1) / (a * a)


@njit
def integrate(rt, a, z, v, ter, sv, sz, st, sa, nodes, weights):
    """One triple loop over the sz/st/sa grids. An inactive axis collapses to a
    single node with weight 1 and its validity guard is skipped."""
    eps = 1e-6

    if (sz > 0 and sz > 2 * min(z, 1 - z)) or \
       (st > 0 and st > 2 * ter) or \
       (sa > 0 and sa > 2 * a):
        return 0.0
    if rt <= 0 or rt <= ter:
        return 0.0

    sz_active = sz >= eps
    st_active = st >= eps
    sa_active = sa >= eps
    if not sz_active and not st_active and not sa_active:
        return ddm_pdf_core(rt, a, z, v, ter, sv)

    if sz_active:
        z_grid = z + nodes * (sz / 2)
        z_wts = weights * 0.5
    else:
        z_grid = np.array([z], dtype=np.float64)
        z_wts = np.array([1.0], dtype=np.float64)

    if st_active:
        ter_grid = ter + nodes * (st / 2)
        ter_wts = weights * 0.5
    else:
        ter_grid = np.array([ter], dtype=np.float64)
        ter_wts = np.array([1.0], dtype=np.float64)

    if sa_active:
        a_grid = a + nodes * (sa / 2)
        a_wts = weights * 0.5
    else:
        a_grid = np.array([a], dtype=np.float64)
        a_wts = np.array([1.0], dtype=np.float64)

    total = 0.0
    for i in range(len(z_grid)):
        z_val = z_grid[i]
        if sz_active and (z_val < 0 or z_val > 1):
            continue
        for j in range(len(ter_grid)):
            ter_val = ter_grid[j]
            if st_active and (ter_val < 0 or ter_val >= rt):
                continue
            for k in range(len(a_grid)):
                a_val = a_grid[k]
                if sa_active and a_val <= 0.01:
                    continue
                total += z_wts[i] * ter_wts[j] * a_wts[k] * \
                    ddm_pdf_core(rt, a_val, z_val, v, ter_val, sv)

    return total


class DDMModel:
    min_p = 1e-10

    def __init__(self, n_points=15):
        self.n_points = n_points
        self.nodes, self.weights = roots_legendre(n_points)

    @staticmethod
    def valid(a, z, v, ter, sv, sz, st, sa):
        return bool(np.all(np.isfinite([a, z, v, ter, sv, sz, st, sa]))
                    and a > 0.01 and 0 <= z <= 1 and ter >= 0
                    and min(sv, sz, st, sa) >= 0
                    and sa <= 2 * a and st <= 2 * ter and sz <= 2 * min(z, 1 - z))

    def pdf(self, rt, a, z, v, ter, sv=0.0, sz=0.0, st=0.0, sa=0.0, validate=True):
        if validate and not self.valid(a, z, v, ter, sv, sz, st, sa):
            return self.min_p
        if rt <= ter:
            return self.min_p
        sv, sz, st, sa = (0.0 if w < 1e-6 else w for w in (sv, sz, st, sa))
        if sz == st == sa == 0.0:
            p = ddm_pdf_core(rt, a, z, v, ter, sv)
        else:
            p = integrate(rt, a, z, v, ter, sv, sz, st, sa, self.nodes, self.weights)
        return max(p, self.min_p)

    def log_likelihood(self, params, data):
        a, z, v, ter, sv, sz, st, sa = params
        if not self.valid(a, z, v, ter, sv, sz, st, sa):
            return -np.inf
        total = 0.0
        for rt, choice in np.asarray(data, dtype=float):
            flip = choice > 0
            total += log(self.pdf(abs(rt), a, 1 - z if flip else z, -v if flip else v,
                                  ter, sv, sz, st, sa))
        return total
