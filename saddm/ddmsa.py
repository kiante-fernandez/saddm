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
Gauss-Legendre quadrature, accurate to ~1e-6 at the default 7 nodes.

There is no lapse (contaminant) mixture here: HSSM adds its own p_outlier
mixture on top of any analytical likelihood, and in plain PyMC a lapse model is
one pm.logaddexp away from ddmsa_logp.

Ratcliff's s = 0.1 convention (used by fortran/fit_sa_simplex.f90) converts to
this module by multiplying a, v, sv (eta) and sa by 10 and leaving t, st and
relative z alone: Fortran a=0.11, v=0.01..0.32, eta=0.23, sa=0.10 becomes
a=1.1, v=0.1..3.2, sv=2.3, sa=1.0.

    from saddm.ddmsa import make_ddmsa_model, sample_ddmsa

    model = make_ddmsa_model(data)          # data: (N, 2) array of [rt, response]
    idata = sample_ddmsa(model, backend="numpyro")
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from scipy.special import roots_legendre

__all__ = [
    "ddmsa_logp",
    "ddmsa_potential",
    "DDMSA",
    "make_ddmsa_model",
    "sample_ddmsa",
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

    nodes, weights = roots_legendre(n_quad)
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
        (N,) tensor of log-densities; -1e3 where rt is below every possible
        non-decision time or a width exceeds its support (sa <= 2a,
        st <= 2t, sz <= 2 min(z, 1 - z)).
    """
    rt = pt.as_tensor_variable(rt).astype("float64")
    response = pt.as_tensor_variable(response).astype("float64")
    ones = pt.ones_like(rt)

    a_v = pt.as_tensor_variable(a).astype("float64") * ones
    z_v = pt.as_tensor_variable(z).astype("float64") * ones
    v_v = pt.as_tensor_variable(v).astype("float64") * ones
    t_v = pt.as_tensor_variable(t).astype("float64") * ones
    sv_v = pt.as_tensor_variable(sv).astype("float64") * ones

    sa_w = sa if _is_static_zero(sa) else pt.as_tensor_variable(sa).astype("float64") * ones
    st_w = st if _is_static_zero(st) else pt.as_tensor_variable(st).astype("float64") * ones
    sz_w = sz if _is_static_zero(sz) else pt.as_tensor_variable(sz).astype("float64") * ones

    upper = pt.gt(response, 0.5)
    v_eff = pt.switch(upper, -v_v, v_v)
    z_eff = pt.switch(upper, 1.0 - z_v, z_v)

    a_grid, log_wa = _uniform_axis(a_v, sa_w, n_quad)
    t_grid, log_wt = _uniform_axis(t_v, st_w, n_quad)
    z_grid, log_wz = _uniform_axis(z_eff, sz_w, n_quad)

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
             + pt.as_tensor_variable(log_wt)[None, None, :, None]
             + pt.as_tensor_variable(log_wz)[None, None, None, :])

    result = pt.logsumexp((log_pdf + log_w).reshape((rt.shape[0], -1)), axis=-1)
    sa_valid = pt.le(sa_w, 2.0 * a_v)
    sz_valid = pt.le(sz_w, 2.0 * pt.minimum(z_v, 1.0 - z_v))
    st_valid = pt.le(st_w, 2.0 * t_v)
    return pt.switch(pt.and_(pt.and_(sa_valid, sz_valid), st_valid), result, _LOG_TINY)


def ddmsa_potential(data, **kwargs):
    """Total log-likelihood for pm.Potential.

    Args:
        data: (N, 2) tensor or array with columns [rt, response].
        **kwargs: passed to ddmsa_logp.

    Returns:
        Scalar tensor.
    """
    data = pt.as_tensor_variable(data)
    return pt.sum(ddmsa_logp(data[:, 0], data[:, 1], **kwargs))


def simulate_ddmsa(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0,
                   n_trials=500, dt=1e-4, max_time=10.0, rng=None, seed=None):
    """Vectorized Euler-Maruyama simulator for the DDM-SA at s = 1.

    Widths sa, st and sz are full widths, matching ddmsa_logp.

    Returns:
        (M, 2) array of [rt, response] with timed-out trials dropped.
    """
    if rng is None:
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


_ICDF_FN = None


def _icdf_density_fn():
    """Compile and cache a scalar-parameter density used by the exact sampler."""
    global _ICDF_FN
    if _ICDF_FN is None:
        import pytensor

        rt = pt.dvector("rt")
        ch = pt.dvector("ch")
        names = ["a", "z", "v", "t", "sv", "sa", "st", "sz"]
        ps = [pt.dscalar(n) for n in names]
        _ICDF_FN = pytensor.function(
            [rt, ch] + ps, pt.exp(ddmsa_logp(rt, ch, *ps)))
    return _ICDF_FN


def sample_ddmsa_exact(a, z, v, t, sv=0.0, sa=0.0, st=0.0, sz=0.0, n_trials=500,
                       rng=None, seed=None, n_grid=8000, max_dt=30.0):
    """Draw exact samples by inverting the analytic CDF. Scalar parameters only.

    Preferred over simulate_ddmsa for parameter recovery. Euler-Maruyama overshoots
    the boundary by O(sqrt(dt)), which inflates a and sv enough to masquerade as a
    recovery failure; at dt=1e-4 the mean RT is biased by roughly +0.4%. This
    sampler draws from the same density the likelihood evaluates, so any residual
    recovery error is a property of the model rather than of the simulator.

    Returns:
        (n_trials, 2) array of [rt, response].
    """
    if rng is None:
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

    Preferred over a bare pm.Potential because it records per-trial log-likelihoods,
    so az.loo and az.compare work, and it supports posterior predictive sampling
    (through sample_ddmsa_exact, so scalar parameters only).

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

    def scalar(x):
        x = np.unique(np.asarray(x, dtype="float64"))
        if x.size != 1:
            raise ValueError("DDMSA.random needs scalar parameters")
        return float(x[0])

    def random(*args, rng=None, size=None):
        n = 1 if size is None else int(np.prod(size))
        out = sample_ddmsa_exact(n_trials=n, rng=rng,
                                 **{k: scalar(x) for k, x in params(args).items()})
        return out if size is None else out.reshape(tuple(size) + (2,))

    return pm.CustomDist(
        name, *[given[k] for k in free],
        logp=logp, random=random,
        signature=",".join("()" for _ in free) + "->(2)",
        observed=observed, **kwargs,
    )


def make_ddmsa_model(data, sz=False, n_quad=N_QUAD, constrained=True,
                     use_potential=False):
    """Build a single-condition PyMC model for the DDM-SA.

    Args:
        data: (N, 2) array with columns [rt in seconds, response 0/1].
        sz: include across-trial start-point variability.
        n_quad: Gauss-Legendre nodes per variability dimension.
        constrained: sample sa as a fraction of a, so it stays inside its
            support by construction. Set False to sample the width directly.
        use_potential: attach the likelihood with pm.Potential instead of the
            CustomDist; cheaper, but gives up per-trial log-likelihoods.

    Non-decision time is parameterized by the lower edge of its uniform
    distribution, t_edge = t - st/2, bounded above by the fastest observed RT.
    Bounding t itself by min(RT) would be wrong: with st > 0 the earliest possible
    response is at t - st/2, so the true t routinely exceeds min(RT) and a prior on
    t capped at min(RT) can exclude it outright. That mis-specification inflates a
    and sv and drives sa toward zero.

    Returns:
        pm.Model with named variables a, z, v, t, sv, sa, st and optionally sz.
    """
    import pymc as pm

    data = np.asarray(data, dtype="float64")
    min_rt = float(data[:, 0].min())

    with pm.Model() as model:
        a = pm.Uniform("a", lower=0.3, upper=5.0)
        z = pm.Beta("z", alpha=3.0, beta=3.0)
        v = pm.Normal("v", mu=0.0, sigma=2.0)
        sv = pm.HalfNormal("sv", sigma=1.5)

        if constrained:
            sa_frac = pm.Beta("sa_frac", alpha=1.5, beta=3.0)
            sa = pm.Deterministic("sa", sa_frac * a)
        else:
            sa = pm.HalfNormal("sa", sigma=1.0)

        st = pm.HalfNormal("st", sigma=0.15)
        t_edge = pm.Uniform("t_edge", lower=0.0, upper=min_rt)
        t = pm.Deterministic("t", t_edge + st / 2.0)

        if sz:
            sz_frac = pm.Beta("sz_frac", alpha=1.5, beta=3.0)
            sz_val = pm.Deterministic("sz", sz_frac * 2.0 * pt.minimum(z, 1.0 - z))
        else:
            sz_val = 0.0

        kw = dict(sv=sv, sa=sa, st=st, sz=sz_val, n_quad=n_quad)
        if use_potential:
            pm.Potential("ddmsa", ddmsa_potential(data, a=a, z=z, v=v, t=t, **kw))
        else:
            DDMSA("ddmsa", a, z, v, t, observed=data, **kw)

    return model


def sample_ddmsa(model, backend="numpyro", draws=1000, tune=1000, chains=4,
                 target_accept=0.9, random_seed=None, **kwargs):
    """Sample a DDM-SA model with gradient-based NUTS.

    backend="numpyro" compiles the log-likelihood to JAX and is several times faster
    per gradient evaluation than the default C backend on this model.

    Args:
        model: model from make_ddmsa_model.
        backend: "numpyro", "nutpie", "blackjax" or "pymc".
        draws, tune, chains, target_accept, random_seed: passed to pm.sample.

    Returns:
        arviz.InferenceData.
    """
    import pymc as pm

    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            nuts_sampler=backend, random_seed=random_seed, **kwargs,
        )
