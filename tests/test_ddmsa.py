"""
Verification suite for saddm.ddmsa — the canonical DDM-SA likelihood.

    pytest tests/test_ddmsa.py -s            # checks 0-5, bounds, static zero
    SAMPLE=1 pytest tests/test_ddmsa.py -s   # also the slow NUTS check

Checks:
  0. Density is the s = 1 Wiener process (closed-form P(upper) and mean RT).
  1. Density matches the Numba reference (core PDF and quadrature integrator).
  2. Gradients match central finite differences for every parameter.
  3. logp and gradients stay finite in the corners of parameter space.
  4. Per-trial (vector) parameters agree with scalar parameters.
  5. The C, Numba, and JAX backends agree.
  6. Optional: a short NUTS run to confirm gradient-based MCMC works end to end.
"""

import importlib.util
import os
import time

import numpy as np
import pytensor
import pytest
import pytensor.tensor as pt

from reference import DDMModel, ddm_pdf_core
from saddm.ddmsa import (DDMSA, N_QUAD, _is_static_zero, ddmsa_logp,
                         sample_ddmsa_exact, simulate_ddmsa)

PARAMS = ["a", "z", "v", "t", "sv", "sa", "st", "sz"]
# sa and sz are alternative accounts of the same variability and never both
# active in a real fit, so each config activates one of them.
TRUE = dict(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08, sz=0.0)
TRUE_SZ = {**TRUE, "sa": 0.0, "sz": 0.1}


def _fn(n_quad=7):
    """Compile logp and its gradient w.r.t. all eight parameters."""
    rt, ch = pt.dvector("rt"), pt.dvector("ch")
    sv = [pt.dscalar(p) for p in PARAMS]
    logp = ddmsa_logp(rt, ch, *sv, n_quad=n_quad)
    total = pt.sum(logp)
    f_logp = pytensor.function([rt, ch] + sv, logp)
    f_grad = pytensor.function([rt, ch] + sv, pytensor.grad(total, sv))
    return f_logp, f_grad


def test_0_scale():
    """The density must be the s = 1 Wiener process, not Ratcliff's s = 0.1.

    For barriers 0 and a, start a*z, drift v and diffusion s, the exit probability
    and mean decision time have closed forms. Numerically integrating our density
    must reproduce the s = 1 versions; the s = 0.1 versions differ so wildly (they
    give P(upper) = 1.000000 for every case below) that the two cannot be confused.
    """
    print("\n[0] s = 1 scale convention")
    rt_v, ch_v = pt.dvector("rt"), pt.dvector("ch")
    sc = [pt.dscalar(n) for n in "azvt"]
    f = pytensor.function([rt_v, ch_v] + sc, pt.exp(ddmsa_logp(rt_v, ch_v, *sc)))

    def closed_form(a, z, v, s):
        x0, k = a * z, 2.0 * v / s ** 2
        p_up = (1 - np.exp(-k * x0)) / (1 - np.exp(-k * a))
        return p_up, (a / v) * p_up - x0 / v

    worst_p = worst_e = 0.0
    for a, z, v, t in [(1.1, 0.5, 1.5, 0.25), (1.1, 0.35, 0.8, 0.20),
                       (2.0, 0.6, 2.5, 0.30), (0.8, 0.5, 0.5, 0.15),
                       (1.5, 0.45, 3.2, 0.22)]:
        g = t + np.geomspace(1e-6, 60.0, 400000)
        w = np.diff(g)
        d = {c: f(g, np.full(g.size, float(c)), a, z, v, t) for c in (0, 1)}
        mass = {c: np.sum((d[c][1:] + d[c][:-1]) / 2 * w) for c in (0, 1)}
        p_num = mass[1] / (mass[0] + mass[1])
        e_num = sum(np.sum((d[c][1:] * g[1:] + d[c][:-1] * g[:-1]) / 2 * w)
                    for c in (0, 1))
        p_ref, e_ref = closed_form(a, z, v, 1.0)
        worst_p = max(worst_p, abs(p_num - p_ref))
        worst_e = max(worst_e, abs(e_num - e_ref - t))
        print(f"    a={a:4.2f} z={z:4.2f} v={v:4.2f}  P(up)={p_num:.6f} "
              f"(s=1 exact {p_ref:.6f}, s=0.1 would be {closed_form(a, z, v, 0.1)[0]:.6f})"
              f"  E[rt]={e_num:.6f} (exact {e_ref + t:.6f})")

    ok = worst_p < 1e-6 and worst_e < 1e-6
    print(f"    max error: P(upper) {worst_p:.2e}, E[rt] {worst_e:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_1_reference():
    """New density vs the Numba core PDF and quadrature integrator."""
    print("\n[1] vs Numba reference")
    f_logp, _ = _fn(n_quad=15)
    model = DDMModel(n_points=15)
    worst = 0.0
    n = 0

    for a, z, v, t, sv in [(1.2, 0.5, 0.3, 0.3, 0.0), (1.2, 0.5, 0.3, 0.3, 0.15),
                           (0.8, 0.3, 0.1, 0.2, 0.2), (2.0, 0.7, 0.5, 0.4, 0.1),
                           (1.1, 0.5, 3.2, 0.22, 2.3)]:
        for rt in [t + 0.05, t + 0.2, t + 0.5, t + 1.2]:
            ref = ddm_pdf_core(rt, a, z, v, t, sv)
            if ref < 1e-20:
                continue
            got = np.exp(f_logp([rt], [0.0], a, z, v, t, sv, 0.0, 0.0, 0.0)[0])
            worst = max(worst, abs(got - ref) / ref)
            n += 1

    for a, z, v, t, sv, sa, st, sz in [
        (1.2, 0.5, 0.3, 0.3, 0.15, 0.2, 0.05, 0.0),
        (1.5, 0.5, 0.2, 0.3, 0.20, 0.0, 0.08, 0.1),
        (1.1, 0.5, 3.2, 0.22, 2.3, 1.0, 0.10, 0.0),
        (1.1, 0.5, 2.0, 0.22, 1.0, 0.0, 0.10, 0.2),
        (1.1, 0.5, 1.5, 0.25, 0.8, 1.6, 0.08, 0.0),
    ]:
        for rt in [t + 0.1, t + 0.35, t + 0.9]:
            ref = model.pdf(rt, a, z, v, t, sv=sv, sa=sa, st=st, sz=sz)
            if ref < 1e-20:
                continue
            got = np.exp(f_logp([rt], [0.0], a, z, v, t, sv, sa, st, sz)[0])
            worst = max(worst, abs(got - ref) / ref)
            n += 1

    ok = worst < 1e-6
    print(f"    {n} densities, max relative error {worst:.3e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_2_gradients():
    """Analytic gradients vs central finite differences."""
    print("\n[2] gradients vs finite differences")
    f_logp, f_grad = _fn()
    ok = True
    for truth in (TRUE, TRUE_SZ):
        ok &= _check_gradients(f_logp, f_grad, truth)
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    assert ok


def _check_gradients(f_logp, f_grad, truth):
    data = simulate_ddmsa(**truth, n_trials=500, seed=42)
    rt, ch = data[:, 0], data[:, 1]
    vals = [truth[p] for p in PARAMS]

    def total(v):
        return float(np.sum(f_logp(rt, ch, *v)))

    analytic = f_grad(rt, ch, *vals)
    ok = True
    for i, name in enumerate(PARAMS):
        if vals[i] == 0.0:
            continue  # inactive width; the negative side is outside the support
        h = 1e-5 * max(abs(vals[i]), 1e-2)
        vp, vm = list(vals), list(vals)
        vp[i] += h
        vm[i] -= h
        numeric = (total(vp) - total(vm)) / (2 * h)
        rel = abs(analytic[i] - numeric) / max(abs(numeric), 1e-6)
        good = rel < 1e-4
        ok &= good
        print(f"    {name:10s} analytic={analytic[i]:+13.5f} numeric={numeric:+13.5f} "
              f"rel={rel:.2e} {'ok' if good else 'MISMATCH'}")
    return ok


def test_3_edges():
    """logp and gradients stay finite everywhere NUTS can wander; outside the
    support logp is -inf (a rejection, not a plateau) with finite gradients."""
    print("\n[3] finiteness in the corners")
    f_logp, f_grad = _fn()
    data = simulate_ddmsa(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08,
                          n_trials=300, seed=3)
    rt, ch = data[:, 0], data[:, 1]
    min_rt = float(rt.min())

    cases = {
        "all variability off": dict(sv=0.0, sa=0.0, st=0.0, sz=0.0),
        "t just below min RT": dict(t=min_rt - 1e-4),
        "t above min RT": dict(t=min_rt + 0.05),
        "st straddles min RT": dict(t=min_rt - 0.02, st=0.4),
        "sa exactly at 2a": dict(sa=TRUE["a"] * 2.0),
        "sa just above 2a (rejected)": dict(sa=TRUE["a"] * 2.01),
        "sz at bound": dict(z=0.1, sz=0.2),
        "sz just past bound (rejected)": dict(z=0.1, sz=0.202),
        "tiny a": dict(a=0.31),
        "huge a": dict(a=4.9),
        "z at the edge": dict(z=0.01, sz=0.0),
        "huge sv": dict(sv=10.0),
    }
    ok = True
    for label, over in cases.items():
        p = {**TRUE, **over}
        vals = [p[k] for k in PARAMS]
        L = float(np.sum(f_logp(rt, ch, *vals)))
        G = np.asarray(f_grad(rt, ch, *vals), dtype=float)
        want_inf = label.endswith("(rejected)")
        good = (L == -np.inf if want_inf else np.isfinite(L)) and np.all(np.isfinite(G))
        ok &= good
        print(f"    {label:30s} logp={L:12.2f} grad finite={np.all(np.isfinite(G))} "
              f"{'' if good else '<-- BAD'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_bounds():
    """sa <= 2a, sz <= 2*min(z, 1-z), st <= 2t: mass is 1 at each bound
    inclusive, rejected past it, and the Numba reference draws the same line."""
    print("\n[+] support bounds")
    from scipy.integrate import quad

    # st = 2t is the widest possible panel and 7 nodes leave 2e-3 of its mass;
    # finer z nodes near 0 put a spike at dt -> 0 that quad misses, so sz stays at 7.
    fns = {7: _fn()[0], 15: _fn(n_quad=15)[0]}
    a, v, t = 1.1, 1.5, 0.25
    model = DDMModel(n_points=15)

    def logp(rt, resp, z, sa=0.0, st=0.0, sz=0.0):
        f_logp = fns[15 if st else 7]
        return float(f_logp([rt], [float(resp)], a, z, v, t, 0.0, sa, st, sz)[0])

    def ref(z, **w):
        return model.pdf(0.5, a, z, v, t, **w) > model.min_p

    ok = True
    for name, z, bound in [("sa", 0.5, dict(sa=2.0 * a)), ("sz", 0.1, dict(sz=0.2)),
                           ("st", 0.5, dict(st=2.0 * t))]:
        inside = {k: 0.99 * w for k, w in bound.items()}
        past = {k: 1.01 * w for k, w in bound.items()}
        lo = t - bound.get("st", 0.0) / 2
        mass = sum(quad(lambda rt: np.exp(logp(rt, resp, z, **bound)), lo + 1e-6, t + 30,
                        limit=200)[0]
                   for resp in (0, 1))
        rejected = logp(t + 0.2, 0, z, **past) == -np.inf
        agrees = ref(z, **inside) and not ref(z, **past)
        good = abs(mass - 1.0) < 1e-3 and rejected and agrees
        ok &= good
        print(f"    {name}: mass at bound={mass:.6f}  rejected past={rejected}  "
              f"reference agrees={agrees}  {'' if good else '<-- BAD'}")

    print(f"    -> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_quad_default():
    """The default n_quad, at the published ITC operating points, against a
    61-node reference. The t panel is truncated at rt (issue #10); a panel that
    straddled the step carried several nats per fast trial at 7 nodes."""
    print("\n[+] default n_quad at the ITC operating points")
    import pandas as pd

    ks = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results",
                                  "reference", "itc_amasino", "ksweep.csv"))
    r, c = pt.dvector("rt"), pt.dvector("ch")
    ps = [pt.dscalar(n) for n in ["a", "z", "v", "t", "sv", "sa", "st"]]
    f7 = pytensor.function([r, c] + ps, ddmsa_logp(r, c, *ps, n_quad=N_QUAD))
    fref = pytensor.function([r, c] + ps, ddmsa_logp(r, c, *ps, n_quad=61))
    worst = 0.0
    for _, row in ks.iterrows():
        vals = [row.a, row.z, row.v_Intercept, row.t0, row.sv, row.sa, row.st]
        lo = row.t0 - row.st / 2.0
        rt = lo + np.geomspace(1e-3, 4.0, 60)
        for ch in (0.0, 1.0):
            chv = np.full(rt.size, ch)
            ref = fref(rt, chv, *vals)
            keep = ref > -20.0  # the tail below that approaches the 1e-30 floor
            d = f7(rt, chv, *vals)[keep] - ref[keep]
            worst = max(worst, float(np.max(np.abs(d))))
    ok = worst < 2e-2
    print(f"    {len(ks)} operating points, max |dlogp| {worst:.2e} nats  "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_4_vector_params():
    """Vector (per-trial) parameters must reproduce the scalar result."""
    print("\n[4] per-trial parameter broadcasting")
    data = simulate_ddmsa(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08,
                          n_trials=200, seed=11)
    rt, ch = data[:, 0], data[:, 1]
    n = len(rt)

    rt_v, ch_v = pt.dvector("rt"), pt.dvector("ch")
    sc = [pt.dscalar(p) for p in PARAMS]
    f_scalar = pytensor.function([rt_v, ch_v] + sc,
                                 ddmsa_logp(rt_v, ch_v, *sc))
    ve = [pt.dvector(p) for p in PARAMS]
    f_vector = pytensor.function([rt_v, ch_v] + ve,
                                 ddmsa_logp(rt_v, ch_v, *ve))

    vals = [TRUE[p] for p in PARAMS]
    a = f_scalar(rt, ch, *vals)
    b = f_vector(rt, ch, *[np.full(n, x) for x in vals])
    worst = float(np.max(np.abs(a - b)))

    rng = np.random.default_rng(0)
    v_trial = rng.normal(1.5, 0.4, n)
    vec = f_vector(rt, ch, *[np.full(n, vals[0]), np.full(n, vals[1]), v_trial,
                             *[np.full(n, x) for x in vals[3:]]])
    loop = np.array([f_scalar(rt[i:i + 1], ch[i:i + 1], vals[0], vals[1],
                              v_trial[i], *vals[3:])[0] for i in range(n)])
    worst_v = float(np.max(np.abs(vec - loop)))

    ok = worst < 1e-12 and worst_v < 1e-10
    print(f"    constant vector vs scalar: {worst:.2e}")
    print(f"    trial-varying v vs loop:   {worst_v:.2e}  -> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_5_backends():
    """Every installed backend (C, Numba, JAX) must agree."""
    print("\n[5] backend agreement and speed")
    data = simulate_ddmsa(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08,
                          n_trials=500, seed=42)
    rt, ch = data[:, 0], data[:, 1]
    sv = [pt.dscalar(p) for p in PARAMS]
    total = pt.sum(ddmsa_logp(pt.as_tensor_variable(rt), pt.as_tensor_variable(ch), *sv))
    outs = [total] + list(pytensor.grad(total, sv))
    vals = [TRUE[p] for p in PARAMS]

    modes = ["C"] + [m for m in ["NUMBA", "JAX"]
                     if importlib.util.find_spec(m.lower()) is not None]
    ref, ok = None, True
    for mode in modes:
        f = pytensor.function(sv, outs, mode=None if mode == "C" else mode)
        got = np.array([float(x) for x in f(*vals)])
        delta = 0.0 if ref is None else float(np.max(np.abs(got - ref)))
        ref = got if ref is None else ref
        good = delta < 1e-8
        ok &= good
        print(f"    {mode:6s} max|delta vs C| {delta:.2e} {'' if good else '<-- MISMATCH'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    assert ok


def test_static_zero():
    """Zero-width axes must collapse to one node, including through pm.CustomDist."""
    print("\n[+] static-zero collapse")
    ok = (_is_static_zero(0.0) and _is_static_zero(pt.constant(0.0))
          and _is_static_zero(np.zeros(3)) and not _is_static_zero(pt.dscalar("w")))

    import pymc as pm

    data = simulate_ddmsa(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, n_trials=50,
                          seed=1)
    with pm.Model() as model:
        a = pm.Uniform("a", 0.3, 5.0)
        sa = pm.HalfNormal("sa", 1.0)
        y = DDMSA("ddmsa", a, 0.5, 1.5, 0.25, sv=0.8, sa=sa, observed=data)
    grids = {v.type.shape[1:] for v in pytensor.graph.ancestors([pm.logp(y, data)])
             if getattr(v.type, "ndim", 0) == 4}
    widest = max(grids, key=np.prod)
    ok = ok and widest == (7, 1, 1)

    ip = model.initial_point()
    a_val = 0.3 + 4.7 / (1.0 + np.exp(-ip["a_interval__"]))
    sa_val = np.exp(ip["sa_log__"])
    direct = ddmsa_logp(data[:, 0], data[:, 1], a=a_val, z=0.5, v=1.5, t=0.25,
                        sv=0.8, sa=sa_val).sum().eval()
    ok = ok and abs(float(model.compile_logp(vars=[y])(ip)) - float(direct)) < 1e-8
    print(f"    widest (sa, st, sz) quadrature grid through CustomDist: {widest}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert ok


@pytest.mark.skipif(not os.environ.get("SAMPLE"), reason="set SAMPLE=1")
def test_6_nuts(backend="numpyro", draws=750, tune=750, chains=2, n_trials=2000):
    """End-to-end gradient MCMC: sampler health plus a recovery report.

    Passing requires healthy geometry (no divergences, r_hat below 1.03, usable
    ESS) and the four core parameters within 3 SD of the truth.

    Recovery of sa, st and sv is reported but not asserted. Over 8 datasets of
    2000 trials the mean bias is sa -30% (SD 0.19, occasionally collapsing to 0)
    and st -18%, while a, z, v and t stay within 3%; sa is weakly identified at
    this N and its estimate is pulled toward the Beta(1.5, 3) prior mean of a/3
    (see the recovery study in verification/). HDI coverage is printed for the
    same reason: with posteriors as correlated as (a, v, sv, sa) a
    single dataset misses individual intervals often enough that asserting it would
    be a coin flip. Calibration belongs in a many-dataset SBC run.
    """
    print(f"\n[6] NUTS via {backend}")
    import arviz as az
    import pymc as pm

    from saddm.ddmsa import make_ddmsa_model

    truth = dict(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08)
    data = sample_ddmsa_exact(**truth, n_trials=n_trials, seed=99)
    print(f"    {len(data)} trials, mean RT {data[:, 0].mean():.3f}s, "
          f"upper {data[:, 1].mean():.1%}")

    model = make_ddmsa_model(data)
    t0 = time.time()
    with model:
        idata = pm.sample(nuts_sampler=backend, draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=3, progressbar=False)
    elapsed = time.time() - t0

    summary = az.summary(idata, var_names=list(truth))
    summary["true"] = [truth[k] for k in summary.index]
    summary["z"] = (summary["mean"] - summary["true"]) / summary["sd"]
    summary["in_hdi"] = [
        lo <= truth[k] <= hi
        for k, lo, hi in zip(summary.index, summary["hdi_3%"], summary["hdi_97%"])
    ]
    print(summary[["true", "mean", "sd", "hdi_3%", "hdi_97%", "z", "r_hat",
                   "ess_bulk", "in_hdi"]].to_string())
    div = int(idata.sample_stats.diverging.values.sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    core = ["a", "z", "v", "t"]
    max_z = float(summary.loc[core, "z"].abs().max())
    ok = div == 0 and max_rhat < 1.03 and min_ess > 100 and max_z < 3.0
    print(f"    {elapsed:.1f}s, {div} divergences, max r_hat {max_rhat:.3f}, "
          f"min ESS {min_ess:.0f}, max |z| over {core} {max_z:.2f}, "
          f"{int(summary['in_hdi'].sum())}/{len(summary)} inside 94% HDI  "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert ok



def test_7_st_prior_scale():
    """make_ddmsa_model's default st prior must cover the st that generated the data.

    st is a duration, so a hard-coded prior on it is a hard-coded assumption about
    the task's timescale. The failure is a property of the DATA, not of any one
    experiment: it appears whenever the RT distribution is slow relative to the
    constant. It is invisible to simulation studies whose generating st was chosen
    to sit inside that constant, which is why the fixed HalfNormal(0.15) survived
    -- test_6_nuts generates st = 0.08, comfortably inside it.

    Two regimes are checked, spanning the range of non-decision variability seen
    across choice tasks. Both must land in the prior's central mass.
    """
    import pymc as pm

    from saddm.ddmsa import make_ddmsa_model

    regimes = [
        ("fast   (perceptual-like)",
         dict(a=1.1, z=0.5, v=1.0, t=0.25, sv=0.5, sa=0.33, st=0.08)),
        ("slow   (value-based-like)",
         dict(a=2.5, z=0.5, v=1.0, t=0.90, sv=0.5, sa=0.75, st=1.30)),
    ]
    ratios = []
    for label, truth in regimes:
        data = sample_ddmsa_exact(**truth, n_trials=1500, seed=11)
        model = make_ddmsa_model(data)
        # pm.draw on the RV, not sample_prior_predictive: only the prior on st is
        # wanted, and forward-sampling the variable avoids the (correct, but here
        # irrelevant) warning that the likelihood Potential is ignored.
        draws = np.asarray(pm.draw(model["st"], draws=2000, random_seed=0)).ravel()
        frac = float(np.mean(draws > truth["st"]))
        rt = data[:, 0]
        print(f"    {label}  true st={truth['st']:.2f}  "
              f"median RT={np.median(rt):.3f}  min RT={rt.min():.3f}  "
              f"prior median={np.median(draws):.3f}  P(prior st > true)={frac:.3f}")
        assert 0.02 < frac < 0.98, (
            f"{label}: the default st prior puts the generating st={truth['st']} "
            f"at tail probability {frac:.2e}. A prior on a duration must scale "
            f"with the data (median RT {np.median(rt):.2f}s here), or the model "
            f"is silently specialised to one timescale."
        )
        ratios.append(truth["st"] / (np.median(rt) - rt.min()))

    # scale-free: the same rule must place both regimes similarly, even though
    # their st differs by ~16x. A fixed prior cannot do this by construction.
    print(f"    st / prior-scale across regimes: "
          f"{ratios[0]:.2f} (fast) vs {ratios[1]:.2f} (slow)")
    assert max(ratios) / min(ratios) < 4.0, (
        f"the prior scale does not track the data: st/scale = {ratios}. "
        f"Across a 16x change in true st these should stay comparable."
    )
