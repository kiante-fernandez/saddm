"""
Fully differentiable DDM-SA likelihood in pure PyTensor.

Implements the Navarro-Fuss (2009) first-passage time density and
DDM with across-trial variability (sv, sa, st) using PyTensor ops,
enabling automatic differentiation for NUTS sampling in PyMC.

Parameters use diffusion coefficient s=1:
    a:   boundary separation (~0.65-2.4)
    z:   relative starting point (0 to 1)
    v:   drift rate (~0.1-0.5)
    ter: non-decision time (seconds)
    sv:  Gaussian drift rate variability
    sa:  uniform boundary separation variability (half-range)
    st:  uniform non-decision time variability (half-range)
"""

import numpy as np
import pytensor.tensor as pt
from scipy.special import roots_legendre

# Series truncation: fixed K for static computation graph
K_LARGE = 30  # terms for large-t (sinusoidal) series
K_SMALL = 15  # half-range for small-t (Gaussian) series

# Threshold for switching between series (~1/(2*pi) ≈ 0.159)
TT_SWITCH = 0.159

# Pre-compute Gauss-Legendre quadrature nodes and weights
_GL_NODES_7, _GL_WEIGHTS_7 = roots_legendre(7)
GL_NODES = pt.as_tensor_variable(np.float64(_GL_NODES_7))
GL_WEIGHTS = pt.as_tensor_variable(np.float64(_GL_WEIGHTS_7))

PI = np.float64(np.pi)
LOG_MIN = np.float64(-1000.0)


def ftt_01w_pytensor(tt, w):
    """Navarro-Fuss f(t|0,1,w) in pure PyTensor.

    Computes both small-t and large-t series with fixed K terms,
    selects via pt.switch. Fully differentiable.

    Args:
        tt: normalized time (rt - ter) / a^2, any shape
        w:  relative starting point, broadcastable with tt

    Returns:
        density values, same shape as tt
    """
    # Large-t series: pi * sum_{k=1}^{K} k * exp(-k^2 * pi^2 * t / 2) * sin(k * pi * w)
    k_large = pt.arange(1, K_LARGE + 1, dtype='float64')  # (K,)
    # Broadcast: tt[..., None] against k_large
    k_sq_pi2_t = k_large ** 2 * (PI ** 2) * tt[..., None] / 2.0
    large_terms = k_large * pt.exp(-k_sq_pi2_t) * pt.sin(k_large * PI * w[..., None])
    large_sum = PI * pt.sum(large_terms, axis=-1)

    # Small-t series: 1/sqrt(2*pi*t^3) * sum_k (w+2k) * exp(-(w+2k)^2/(2t))
    k_small = pt.arange(-K_SMALL, K_SMALL + 1, dtype='float64')  # (2K+1,)
    w_plus_2k = w[..., None] + 2.0 * k_small
    small_terms = w_plus_2k * pt.exp(-w_plus_2k ** 2 / (2.0 * tt[..., None]))
    small_sum = pt.sum(small_terms, axis=-1) / pt.sqrt(2.0 * PI * tt ** 3)

    # Select based on normalized time
    result = pt.switch(pt.gt(tt, TT_SWITCH), large_sum, small_sum)

    return pt.maximum(result, 1e-30)


def ddm_logp_pytensor(rt, choice, a, z, v, ter, sv, sa, st, sz=None):
    """Fully differentiable DDM-SA log-likelihood.

    Integrates analytically over drift variability (sv) and numerically
    (Gauss-Legendre) over uniform boundary (sa), non-decision (st), and
    optionally starting-point (sz) variability.

    Args:
        rt:     (N,) response times in seconds
        choice: (N,) binary choices (0=lower, 1=upper boundary)
        a, z, v, ter, sv, sa, st: scalar PyTensor variables (s=1 scale)
        sz:     optional scalar starting-point variability (uniform full-width,
                in relative-z units). When None the sz axis collapses (Qz=1) and
                the result is numerically identical to the sv/sa/st model.

    Returns:
        (N,) per-trial log-likelihoods
    """
    # Flip drift and starting point for upper boundary responses
    eff_v = pt.switch(pt.gt(choice, 0.5), -v, v)      # (N,)
    eff_z = pt.switch(pt.gt(choice, 0.5), 1.0 - z, z)  # (N,)

    # Gauss-Legendre quadrature grids for sa and st (axes 1, 2)
    sa_grid = a + GL_NODES * (sa / 2.0)    # (Qa,)
    sa_wts = GL_WEIGHTS * 0.5
    st_grid = ter + GL_NODES * (st / 2.0)  # (Qst,)
    st_wts = GL_WEIGHTS * 0.5
    sa_grid = pt.maximum(sa_grid, 0.01)
    st_grid = pt.maximum(st_grid, 0.001)

    # Starting-point axis (axis 3). z varies per-trial because eff_z depends on
    # choice, so the grid is (N, Qz). When sz is None it degenerates to (N, 1).
    if sz is None:
        sz_nodes = pt.as_tensor_variable(np.array([0.0]))
        sz_wts = pt.as_tensor_variable(np.array([1.0]))
        sz_half = 0.0
    else:
        sz_nodes = GL_NODES
        sz_wts = GL_WEIGHTS * 0.5
        sz_half = sz / 2.0
    z_grid = eff_z[:, None] + sz_nodes[None, :] * sz_half   # (N, Qz)
    z_grid = pt.clip(z_grid, 1e-4, 1.0 - 1e-4)

    # Broadcast to (N, Qa, Qst, Qz)
    rt_4 = rt[:, None, None, None]
    v_4 = eff_v[:, None, None, None]
    a_4 = sa_grid[None, :, None, None]     # (1, Qa, 1, 1)
    ter_4 = st_grid[None, None, :, None]   # (1, 1, Qst, 1)
    z_4 = z_grid[:, None, None, :]         # (N, 1, 1, Qz)

    # Decision time and normalized time
    dt = rt_4 - ter_4                # (N, 1, Qst, 1)
    dt_safe = pt.maximum(dt, 1e-10)
    tt = dt_safe / (a_4 ** 2)        # (N, Qa, Qst, 1)
    valid = dt > 1e-10

    # f(tt | 0, 1, w) via Navarro-Fuss -> (N, Qa, Qst, Qz)
    ftt = ftt_01w_pytensor(tt, z_4)

    # Analytical sv integration (Gaussian mixture); reduces to sv=0 smoothly
    sv_sq_dt = sv ** 2 * dt_safe
    denom = sv_sq_dt + 1.0
    exp_numerator = ((a_4 * z_4 * sv) ** 2
                     - 2.0 * a_4 * v_4 * z_4
                     - v_4 ** 2 * dt_safe)
    log_sv_factor = exp_numerator / (2.0 * denom) - 0.5 * pt.log(denom)

    log_pdf = pt.log(ftt) + log_sv_factor - 2.0 * pt.log(a_4)
    log_pdf = pt.switch(valid, log_pdf, LOG_MIN)

    # Combined quadrature weights (Qa, Qst, Qz) and logsumexp over all axes
    w_q = sa_wts[:, None, None] * st_wts[None, :, None] * sz_wts[None, None, :]
    log_w = pt.log(pt.maximum(w_q, 1e-30))
    log_integrand = log_pdf + log_w[None, :, :, :]   # (N, Qa, Qst, Qz)

    n_trials = rt.shape[0]
    log_integrand_flat = log_integrand.reshape((n_trials, -1))
    logp = pt.logsumexp(log_integrand_flat, axis=-1)  # (N,)

    return logp


def ddm_logp_contam_pytensor(rt, choice, a, z, v, ter, sv, sa, st,
                             p_outlier, w_outlier=0.1, sz=None):
    """DDM log-likelihood with a lapse / p_outlier contamination mixture.

    Follows the HDDM convention: a fraction p_outlier of responses are outliers
    drawn from a flat (uniform) density w_outlier (default 0.1 ~ 0.5/5s over both
    choices), the rest from the DDM:

        density = (1 - p_outlier) * f_DDM(rt, choice | theta) + p_outlier * w_outlier

    Fully differentiable; reduces to the plain DDM at p_outlier = 0.
    """
    base = ddm_logp_pytensor(rt, choice, a, z, v, ter, sv, sa, st, sz=sz)  # (N,) log-density
    dens = (1.0 - p_outlier) * pt.exp(base) + p_outlier * w_outlier
    return pt.log(pt.maximum(dens, 1e-300))


def ddm_logp_asym_pytensor(rt, choice, a, z, v, ter, sv, sa_u, sa_l, st):
    """Differentiable log-likelihood with ASYMMETRIC boundary variability.

    The two boundary-to-start distances vary independently:
        B_u = a*(1-z) + U(-sa_u/2, sa_u/2)   (start -> upper boundary)
        B_l = a*z     + U(-sa_l/2, sa_l/2)   (start -> lower boundary)
    Each grid point defines a plain DDM with separation a' = B_u + B_l and
    relative start z' = B_l / a'. Integrates over (B_u, B_l, st); sv analytic.

    sa_u == sa_l reduces to independent (correlation-0) boundary noise, which is
    NOT the same as the symmetric-sa model (that is the correlation-+1 case).
    """
    eff_v = pt.switch(pt.gt(choice, 0.5), -v, v)   # (N,)

    Bu0 = a * (1.0 - z)
    Bl0 = a * z
    Bu_grid = pt.maximum(Bu0 + GL_NODES * (sa_u / 2.0), 0.01)   # (Qu,)
    Bl_grid = pt.maximum(Bl0 + GL_NODES * (sa_l / 2.0), 0.01)   # (Ql,)
    Bu_wts = GL_WEIGHTS * 0.5
    Bl_wts = GL_WEIGHTS * 0.5
    st_grid = pt.maximum(ter + GL_NODES * (st / 2.0), 0.001)    # (Qst,)
    st_wts = GL_WEIGHTS * 0.5

    # axes: (N, Qu, Ql, Qst)
    Bu_4 = Bu_grid[None, :, None, None]
    Bl_4 = Bl_grid[None, None, :, None]
    a_4 = Bu_4 + Bl_4                       # (1, Qu, Ql, 1)
    ch_4 = choice[:, None, None, None]
    v_4 = eff_v[:, None, None, None]
    # relative start in the response's frame: lower resp uses Bl/a', upper Bu/a'
    z_4 = pt.switch(pt.gt(ch_4, 0.5), Bu_4 / a_4, Bl_4 / a_4)   # (N, Qu, Ql, 1)
    rt_4 = rt[:, None, None, None]
    ter_4 = st_grid[None, None, None, :]

    dt = rt_4 - ter_4
    dt_safe = pt.maximum(dt, 1e-10)
    tt = dt_safe / (a_4 ** 2)
    valid = dt > 1e-10

    ftt = ftt_01w_pytensor(tt, z_4)         # (N, Qu, Ql, Qst)

    sv_sq_dt = sv ** 2 * dt_safe
    denom = sv_sq_dt + 1.0
    exp_numerator = ((a_4 * z_4 * sv) ** 2
                     - 2.0 * a_4 * v_4 * z_4
                     - v_4 ** 2 * dt_safe)
    log_sv_factor = exp_numerator / (2.0 * denom) - 0.5 * pt.log(denom)

    log_pdf = pt.log(ftt) + log_sv_factor - 2.0 * pt.log(a_4)
    log_pdf = pt.switch(valid, log_pdf, LOG_MIN)

    w_q = Bu_wts[:, None, None] * Bl_wts[None, :, None] * st_wts[None, None, :]
    log_w = pt.log(pt.maximum(w_q, 1e-30))
    log_integrand = log_pdf + log_w[None, :, :, :]

    n_trials = rt.shape[0]
    logp = pt.logsumexp(log_integrand.reshape((n_trials, -1)), axis=-1)
    return logp
