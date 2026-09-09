"""
DDM-SA: drift diffusion model with across-trial variability in boundary separation.

Canonical fully differentiable implementation. Built from PyTensor ops only, so the
log-likelihood has exact analytic gradients, compiles to the C, Numba and JAX
backends unchanged, and supports NUTS via PyMC, numpyro, nutpie or blackjax.

Per trial, with diffusion coefficient s = 1:

    a_i ~ Uniform(a - sa/2, a + sa/2)     boundary separation
    t_i ~ Uniform(t - st/2, t + st/2)     non-decision time
    z_i ~ Uniform(z - sz/2, z + sz/2)     relative start point
    v_i ~ Normal(v, sv)                   drift rate

sa, st and sz are full widths, matching simulate_ddmsa. The drift integral is
analytic (Ratcliff's Gaussian-mixture form); the uniform integrals use
Gauss-Legendre quadrature, with the t panel truncated at rt. At the default
7 nodes the per-trial error is below 1e-2 nats at the published ITC operating
points (tests/test_ddmsa.py::test_quad_default); n_quad=15 is below 1e-3 and
n_quad=31 below 1e-6.

Ratcliff's s = 0.1 convention converts to this module by multiplying a, v, sv
and sa by 10; t, st and relative z are unchanged.

    from saddm import DDMSA

    with pm.Model():                         # priors are yours; data: (N, 2) [rt, response]
        a, z, v, t = (pm.HalfNormal("a", 3), pm.Beta("z", 3, 3), pm.Normal("v", 0, 2),
                      pm.Uniform("t", 0, data[:, 0].min()))
        DDMSA("y", a, z, v, t, sv=pm.HalfNormal("sv", 1.5), sa=pm.HalfNormal("sa", 1),
              st=pm.HalfNormal("st", 0.5), observed=data)
        idata = pm.sample(nuts_sampler="numpyro")
"""

from __future__ import annotations

import functools

import numpy as np
import pytensor.tensor as pt

__all__ = [
    "ddmsa_logp",
    "hssm_loglik",
    "HSSM_PARAMS",
    "DDMSA",
    "sample_ddmsa_exact",
    "simulate_ddmsa",
]

K_LARGE = 30
K_SMALL = 15
TT_SWITCH = 0.159
N_QUAD = 7

_PI = np.float64(np.pi)
_LOG_TINY = np.float64(-1e3)
_FTT_FLOOR = np.float64(1e-30)


def wfpt_01w(tt, w):
    """Navarro & Fuss (2009) density f(t | 0, 1, w) for the unit-scale Wiener process.

    Both series are evaluated at fixed term counts and selected with pt.switch, which
    keeps the graph shape static so it compiles to JAX.

    Args:
        tt: normalized decision time (rt - t) / a**2, any shape.
        w:  relative start point in (0, 1), broadcastable with tt.

    Returns:
        Density with the broadcast shape of tt and w, floored at 1e-30.
    """
    k_l = pt.arange(1, K_LARGE + 1, dtype="float64")
    large = _PI * pt.sum(
        k_l * pt.exp(-(k_l ** 2) * (_PI ** 2) * tt[..., None] / 2.0)
        * pt.sin(k_l * _PI * w[..., None]),
        axis=-1,
    )

    k_s = pt.arange(-K_SMALL, K_SMALL + 1, dtype="float64")
    wk = w[..., None] + 2.0 * k_s
    small = pt.sum(wk * pt.exp(-(wk ** 2) / (2.0 * tt[..., None])), axis=-1) / pt.sqrt(
        2.0 * _PI * tt ** 3
    )

    return pt.maximum(pt.switch(pt.gt(tt, TT_SWITCH), large, small), _FTT_FLOOR)


def _is_static_zero(x) -> bool:
    """True when x is known to be exactly 0 while the graph is being built.

    pm.CustomDist hands constant parameters to logp as TensorConstants, so those
    count too; otherwise every zero-width axis would still cost n_quad nodes.
    """
    if isinstance(x, pt.TensorConstant):
        x = x.data
    if isinstance(x, (int, float, np.number, np.ndarray)):
        x = np.asarray(x)
        return x.size > 0 and bool(np.all(x == 0.0))
    return False


def _uniform_axis(center, width, n_quad):
    """Gauss-Legendre grid for Uniform(center - width/2, center + width/2).

    Returns (grid, log_weights) with grid of shape (N, Q) and weights that already
    absorb the 1/width density, so they sum to 1. A statically-zero width collapses
    the axis to one node.
    """
    if _is_static_zero(width):
        return center[:, None], np.zeros(1)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = pt.as_tensor_variable(np.asarray(nodes, dtype="float64"))
    log_w = np.log(np.asarray(weights, dtype="float64") * 0.5)
    grid = center[:, None] + nodes[None, :] * (width[:, None] / 2.0)
    return grid, log_w


def ddmsa_logp(rt, response, a, z, v, t,
               sv=0.0, sa=0.0, st=0.0, sz=0.0, n_quad=N_QUAD):
    """Per-trial log-likelihood of the DDM-SA. Pure PyTensor, fully differentiable.

    Every parameter may be a scalar or an (N,) vector, so the same function serves
    single-condition fits, per-trial regressions and hierarchical models.

    Args:
        rt:        (N,) response times in seconds, always positive.
        response:  (N,) responses; > 0.5 is the upper boundary, so both 0/1 and
            -1/1 coding work.
        a, z, v, t: boundary separation, relative start point in (0, 1), drift rate,
            non-decision time.
        sv: SD of the Gaussian across-trial drift distribution.
        sa, st, sz: full widths of the uniform across-trial distributions of boundary
            separation, non-decision time and relative start point.
        n_quad: Gauss-Legendre nodes per active variability dimension.

    Returns:
        (N,) tensor of log-densities. -1e3 where rt is below every possible
        non-decision time; -inf where a width is negative or exceeds its support
        (sa <= 2a, st <= 2t, sz <= 2 min(z, 1 - z)), so samplers reject rather
        than plateau there. The quadrature grids are clipped at a >= 1e-3 and
        1e-4 <= z <= 1 - 1e-4; below those the density is constant in a or z.

    Raises:
        ValueError: if a concrete rt contains non-finite or non-positive values.
    """
    if not isinstance(rt, pt.Variable) or isinstance(rt, pt.TensorConstant):
        r = np.asarray(getattr(rt, "data", rt), dtype="float64")
        bad = np.flatnonzero(~(np.isfinite(r) & (r > 0.0)))
        if bad.size:
            raise ValueError(f"{bad.size} rt values are not finite and positive, "
                             f"first at indices {bad[:5].tolist()}")
    rt = pt.as_tensor_variable(rt).astype("float64")
    response = pt.as_tensor_variable(response).astype("float64")
    ones = pt.ones_like(rt)

    def vec(x):
        return x if _is_static_zero(x) else pt.as_tensor_variable(x).astype("float64") * ones

    a_v, z_v, v_v, t_v, sv_v = (pt.as_tensor_variable(x).astype("float64") * ones
                                for x in (a, z, v, t, sv))
    sa_w, st_w, sz_w = vec(sa), vec(st), vec(sz)

    upper = pt.gt(response, 0.5)
    v_eff = pt.switch(upper, -v_v, v_v)
    z_eff = pt.switch(upper, 1.0 - z_v, z_v)

    a_grid, log_wa = _uniform_axis(a_v, sa_w, n_quad)
    z_grid, log_wz = _uniform_axis(z_eff, sz_w, n_quad)

    if _is_static_zero(st_w):
        t_grid, log_wt = _uniform_axis(t_v, st_w, n_quad)
    else:
        # The t integrand vanishes for t_i >= rt, and a Gauss-Legendre panel that
        # straddles that step loses its convergence (errors of several nats on
        # fast trials at 7 nodes). Integrate only the fraction of the panel below
        # rt and rescale the weights by it. frac == 0 (rt below the whole panel)
        # keeps the full grid, where every node is then invalid.
        t_lo = t_v - st_w / 2.0
        frac = pt.clip((rt - t_lo) / pt.maximum(st_w, 1e-12), 0.0, 1.0)
        frac = pt.switch(pt.gt(frac, 0.0), frac, 1.0)
        t_grid, log_wt = _uniform_axis(t_lo + frac * st_w / 2.0, frac * st_w, n_quad)
        log_wt = log_wt[None, :] + pt.log(frac)[:, None]

    a_grid = pt.maximum(a_grid, 1e-3)
    z_grid = pt.clip(z_grid, 1e-4, 1.0 - 1e-4)

    a4 = a_grid[:, :, None, None]
    t4 = t_grid[:, None, :, None]
    z4 = z_grid[:, None, None, :]
    rt4 = rt[:, None, None, None]
    v4 = v_eff[:, None, None, None]
    sv4 = sv_v[:, None, None, None]

    dt = rt4 - t4
    valid = pt.gt(dt, 1e-10)
    dt_s = pt.switch(valid, dt, 1.0)

    log_f = pt.log(wfpt_01w(dt_s / a4 ** 2, z4))

    denom = sv4 ** 2 * dt_s + 1.0
    log_sv = (((a4 * z4 * sv4) ** 2 - 2.0 * a4 * v4 * z4 - v4 ** 2 * dt_s)
              / (2.0 * denom) - 0.5 * pt.log(denom))

    log_pdf = pt.switch(valid, log_f + log_sv - 2.0 * pt.log(a4), _LOG_TINY)

    log_w = (pt.as_tensor_variable(log_wa)[None, :, None, None]
             + pt.atleast_2d(pt.as_tensor_variable(log_wt))[:, None, :, None]
             + pt.as_tensor_variable(log_wz)[None, None, None, :])

    lw = (log_pdf + log_w).reshape((rt.shape[0], -1))
    m = pt.max(lw, axis=-1)  # explicit shift: pt.logsumexp underflows to -inf unrewritten
    result = m + pt.log(pt.sum(pt.exp(lw - m[:, None]), axis=-1))
    # Widths are non-negative by definition; the grid is symmetric under a sign
    # flip, so without the lower bound a negative width returns the density at |w|.
    ok = pt.and_(pt.ge(sv_v, 0.0), pt.and_(pt.ge(sa_w, 0.0), pt.le(sa_w, 2.0 * a_v)))
    ok = pt.and_(ok, pt.and_(pt.ge(sz_w, 0.0), pt.le(sz_w, 2.0 * pt.minimum(z_v, 1.0 - z_v))))
    ok = pt.and_(ok, pt.and_(pt.ge(st_w, 0.0), pt.le(st_w, 2.0 * t_v)))
    return pt.switch(ok, result, -np.inf)


# Pass a copy (list(HSSM_PARAMS)) to hssm.HSSM: it appends "p_outlier" to the
# list it is given in place, which breaks the next model built in the process.
HSSM_PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]


def hssm_loglik(data, v, a, z, t, sv, sa, st, n_quad=N_QUAD):
    """ddmsa_logp as an HSSM loglik_kind="analytical" likelihood.

    Pass with model_config["list_params"] = HSSM_PARAMS. a and the widths are in
    saddm's full units; t is the lower edge of the non-decision distribution, so
    the actual t0 is t + st/2. Fix a width to 0.0 in hssm.HSSM for a plain DDM.
    functools.partial(hssm_loglik, n_quad=15) raises the node count.
    """
    data = pt.reshape(data, (-1, 2))
    return ddmsa_logp(pt.abs(data[:, 0]), data[:, 1], a=a, z=z, v=v,
                      t=t + st / 2.0, sv=sv, sa=sa, st=st, n_quad=n_quad)


def simulate_ddmsa(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0,
                   n_trials=500, dt=1e-4, max_time=10.0, seed=None):
    """Vectorized Euler-Maruyama simulator for the DDM-SA at s = 1.

    Widths sa, st and sz are full widths, matching ddmsa_logp. seed is anything
    np.random.default_rng accepts, including a Generator.

    Returns:
        (M, 2) array of [rt, response] with timed-out trials dropped.
    """
    rng = np.random.default_rng(seed)
    n = int(n_trials)

    v_i = rng.normal(v, sv, n) if sv > 0 else np.full(n, float(v))
    a_i = a + rng.uniform(-sa / 2, sa / 2, n) if sa > 0 else np.full(n, float(a))
    a_i = np.maximum(a_i, 1e-3)
    t_i = t + rng.uniform(-st / 2, st / 2, n) if st > 0 else np.full(n, float(t))
    t_i = np.maximum(t_i, 0.0)
    z_i = z + rng.uniform(-sz / 2, sz / 2, n) if sz > 0 else np.full(n, float(z))
    z_i = np.clip(z_i, 1e-4, 1 - 1e-4)

    x = a_i * z_i
    live = np.ones(n, dtype=bool)
    decision_time = np.zeros(n)
    response = np.full(n, -1)
    sqrt_dt = np.sqrt(dt)

    for _ in range(int(max_time / dt)):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        x[idx] += v_i[idx] * dt + sqrt_dt * rng.standard_normal(idx.size)
        decision_time[idx] += dt
        hit_up = idx[x[idx] >= a_i[idx]]
        hit_lo = idx[x[idx] <= 0.0]
        response[hit_up] = 1
        response[hit_lo] = 0
        live[hit_up] = False
        live[hit_lo] = False

    ok = response >= 0
    return np.column_stack([t_i[ok] + decision_time[ok], response[ok].astype(float)])


@functools.cache
def _icdf_density_fn():
    """Compile a scalar-parameter density for the exact sampler, once."""
    import pytensor

    rt, ch = pt.dvector("rt"), pt.dvector("ch")
    ps = [pt.dscalar(n) for n in ["a", "z", "v", "t", "sv", "sa", "st", "sz"]]
    return pytensor.function([rt, ch] + ps, pt.exp(ddmsa_logp(rt, ch, *ps)))


def sample_ddmsa_exact(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0, n_trials=500,
                       seed=None, n_grid=8000, max_dt=30.0):
    """Draw exact samples by inverting the analytic CDF. Scalar parameters only.

    Preferred over simulate_ddmsa for parameter recovery. Euler-Maruyama overshoots
    the boundary by O(sqrt(dt)), which inflates a and sv enough to masquerade as a
    recovery failure; at dt=1e-4 the mean RT is biased by roughly +0.4%. This
    sampler draws from the same density the likelihood evaluates, so any residual
    recovery error is a property of the model rather than of the simulator.

    Returns:
        (n_trials, 2) array of [rt, response].
    """
    rng = np.random.default_rng(seed)
    f = _icdf_density_fn()
    a, z, v, t, sv, sa, st, sz = (float(x) for x in (a, z, v, t, sv, sa, st, sz))

    lo = max(t - st / 2.0, 0.0)
    grid = lo + np.geomspace(1e-5, max_dt, n_grid)

    dens = np.stack([f(grid, np.full(n_grid, float(c)), a, z, v, t, sv, sa, st, sz)
                     for c in (0.0, 1.0)])
    cum = np.concatenate(
        [np.zeros((2, 1)), np.cumsum((dens[:, 1:] + dens[:, :-1]) / 2.0 * np.diff(grid),
                                     axis=1)], axis=1)

    mass = cum[:, -1]
    total = mass.sum()
    if not 0.99 < total < 1.01:
        raise ValueError(f"density integrates to {total:.4f}; widen max_dt or n_grid")

    p_upper = mass[1] / total
    resp = (rng.random(n_trials) < p_upper).astype(float)
    u = rng.random(n_trials)
    rt = np.empty(n_trials)
    for c in (0, 1):
        idx = np.flatnonzero(resp == c)
        if idx.size:
            rt[idx] = np.interp(u[idx] * mass[c], cum[c], grid)

    return np.column_stack([rt, resp])


def DDMSA(name, a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0, n_quad=N_QUAD,
          observed=None, **kwargs):
    """DDM-SA as a pm.CustomDist over an (N, 2) matrix of [rt, response].

    Records per-trial log-likelihoods, so az.loo and az.compare work. No random
    method: draw data with sample_ddmsa_exact.

    pm.CustomDist hands logp fresh symbolic inputs, so a width that is a constant
    0 would still be integrated over n_quad nodes. Only the non-zero parameters
    become CustomDist inputs; the zeros are baked into the graph.
    """
    import pymc as pm

    given = dict(a=a, z=z, v=v, t=t, sv=sv, sa=sa, st=st, sz=sz)
    free = [k for k, x in given.items() if not _is_static_zero(x)]

    def params(args):
        return {**given, **dict(zip(free, args))}

    def logp(value, *args):
        return ddmsa_logp(value[:, 0], value[:, 1], n_quad=n_quad, **params(args))

    return pm.CustomDist(
        name, *[given[k] for k in free],
        logp=logp,
        signature=",".join("()" for _ in free) + "->(2)",
        observed=observed, **kwargs,
    )
