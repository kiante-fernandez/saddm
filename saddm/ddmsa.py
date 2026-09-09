"""
DDM-SA: drift diffusion model with across-trial variability in boundary separation.

PyTensor likelihood with analytic gradients; compiles to the C, Numba and JAX
backends. Per trial, with diffusion coefficient s = 1:

    a_i ~ Uniform(a - sa/2, a + sa/2)     boundary separation
    t_i ~ Uniform(t - st/2, t + st/2)     non-decision time
    z_i ~ Uniform(z - sz/2, z + sz/2)     relative start point
    v_i ~ Normal(v, sv)                   drift rate

sa, st and sz are full widths. The drift integral is analytic; the uniform
integrals are Gauss-Legendre with the t panel truncated at rt (per-trial error
< 1e-2 nats at 7 nodes, < 1e-3 at 15, < 1e-6 at 31). Ratcliff's s = 0.1 units
convert by multiplying a, v, sv and sa by 10.

    from saddm import DDMSA

    with pm.Model():                         # data: (N, 2) array of [rt, response]
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
    """Navarro & Fuss (2009) f(tt | 0, 1, w), tt = (rt - t) / a**2, floored at 1e-30.

    Both series run at fixed term counts (static graph shape for JAX).
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
    """True when x is a concrete 0 (Python number, array or TensorConstant)."""
    if isinstance(x, pt.TensorConstant):
        x = x.data
    if isinstance(x, (int, float, np.number, np.ndarray)):
        x = np.asarray(x)
        return x.size > 0 and bool(np.all(x == 0.0))
    return False


def _uniform_axis(center, width, n_quad):
    """(N, Q) Gauss-Legendre grid and log-weights (summing to 1) for
    Uniform(center - width/2, center + width/2); one node if width is a static 0."""
    if _is_static_zero(width):
        return center[:, None], np.zeros(1)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = pt.as_tensor_variable(np.asarray(nodes, dtype="float64"))
    log_w = np.log(np.asarray(weights, dtype="float64") * 0.5)
    grid = center[:, None] + nodes[None, :] * (width[:, None] / 2.0)
    return grid, log_w


def ddmsa_logp(rt, response, a, z, v, t,
               sv=0.0, sa=0.0, st=0.0, sz=0.0, n_quad=N_QUAD):
    """Per-trial DDM-SA log-likelihood. Parameters may be scalars or (N,) vectors.

    Args:
        rt: (N,) response times in seconds.
        response: (N,); > 0.5 is the upper boundary (0/1 or -1/1 coding).
        a, z, v, t: boundary separation, start point in (0, 1), drift, non-decision time.
        sv: SD of the across-trial drift distribution.
        sa, st, sz: full widths of the uniform across-trial distributions.
        n_quad: Gauss-Legendre nodes per active variability dimension.

    Returns:
        (N,) log-densities. -1e3 where rt precedes every possible non-decision
        time; -inf where a width is negative or exceeds its support (sa <= 2a,
        st <= 2t, sz <= 2 min(z, 1 - z)). Grids are clipped at a >= 1e-3 and
        1e-4 <= z <= 1 - 1e-4.

    Raises:
        ValueError: concrete rt with non-finite or non-positive entries.
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
        # Integrate only the part of the t panel below rt (the integrand is 0
        # above it); frac == 0 keeps the full grid, where every node is invalid.
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
    m = pt.max(lw, axis=-1)  # pt.logsumexp underflows to -inf without the shift
    result = m + pt.log(pt.sum(pt.exp(lw - m[:, None]), axis=-1))
    # a negative width would silently return the density at |width|
    ok = pt.and_(pt.ge(sv_v, 0.0), pt.and_(pt.ge(sa_w, 0.0), pt.le(sa_w, 2.0 * a_v)))
    ok = pt.and_(ok, pt.and_(pt.ge(sz_w, 0.0), pt.le(sz_w, 2.0 * pt.minimum(z_v, 1.0 - z_v))))
    ok = pt.and_(ok, pt.and_(pt.ge(st_w, 0.0), pt.le(st_w, 2.0 * t_v)))
    return pt.switch(ok, result, -np.inf)


# Pass hssm.HSSM a copy, list(HSSM_PARAMS): it appends "p_outlier" in place.
HSSM_PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]


def hssm_loglik(data, v, a, z, t, sv, sa, st, n_quad=N_QUAD):
    """ddmsa_logp as an HSSM loglik_kind="analytical" likelihood.

    Full units; t is the lower edge of the non-decision distribution (t0 = t + st/2).
    Fix a width to 0.0 for a plain DDM; functools.partial sets n_quad.
    """
    data = pt.reshape(data, (-1, 2))
    return ddmsa_logp(pt.abs(data[:, 0]), data[:, 1], a=a, z=z, v=v,
                      t=t + st / 2.0, sv=sv, sa=sa, st=st, n_quad=n_quad)


def simulate_ddmsa(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0,
                   n_trials=500, dt=1e-4, max_time=10.0, seed=None):
    """Euler-Maruyama simulator at s = 1. Returns (M, 2) [rt, response], timeouts dropped."""
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
    """Compiled scalar-parameter density, cached."""
    import pytensor

    rt, ch = pt.dvector("rt"), pt.dvector("ch")
    ps = [pt.dscalar(n) for n in ["a", "z", "v", "t", "sv", "sa", "st", "sz"]]
    return pytensor.function([rt, ch] + ps, pt.exp(ddmsa_logp(rt, ch, *ps)))


def sample_ddmsa_exact(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0, n_trials=500,
                       seed=None, n_grid=8000, max_dt=30.0):
    """Exact samples by inverse CDF of the analytic density. Scalar parameters.

    Use this for recovery studies: Euler-Maruyama overshoots the boundary and
    biases a and sv. Returns (n_trials, 2) [rt, response].
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

    Per-trial log-likelihoods for az.loo; no random method. Static-zero widths
    are baked into the graph rather than passed as inputs, so they cost no nodes.
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
