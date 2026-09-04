"""
Verification suite for saddm.ddmsa — the canonical DDM-SA likelihood.

Runs under pytest (checks 0-6; the slow NUTS check is CLI-only) or directly:
    pytest tests/test_ddmsa.py
    python tests/test_ddmsa.py [--sample]

Checks, in order:
  0. Density is the s = 1 Wiener process (closed-form P(upper) and mean RT).
  1. Density matches the Numba reference (core PDF and quadrature integrator).
  2. Gradients match central finite differences for every parameter.
  3. logp and gradients stay finite in the corners of parameter space.
  4. Per-trial (vector) parameters agree with scalar parameters.
  5. The C, Numba, and JAX backends agree, with timings.
  6. Optional: a short NUTS run to confirm gradient-based MCMC works end to end.
"""

import argparse
import importlib.util
import sys
import time

import numpy as np
import pytensor
import pytensor.tensor as pt

from saddm.ddmsa import (DDMSA, _LOG_TINY, _is_static_zero, ddmsa_logp,
                         sample_ddmsa_exact, simulate_ddmsa)

PARAMS = ["a", "z", "v", "t", "sv", "sa", "st", "sz"]
TRUE = dict(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08, sz=0.1)


def _fn(n_quad=7):
    """Compile logp and its gradient w.r.t. all eight parameters."""
    rt, ch = pt.dvector("rt"), pt.dvector("ch")
    sv = [pt.dscalar(p) for p in PARAMS]
    logp = ddmsa_logp(rt, ch, *sv, n_quad=n_quad)
    total = pt.sum(logp)
    f_logp = pytensor.function([rt, ch] + sv, logp)
    f_grad = pytensor.function([rt, ch] + sv, pytensor.grad(total, sv))
    return f_logp, f_grad


def check_0_scale():
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
    return ok


def check_1_reference():
    """New density vs the Numba core PDF and quadrature integrator."""
    print("\n[1] vs Numba reference")
    from saddm.core import ddm_pdf_core
    from saddm.model import DDMModel

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
        (1.5, 0.5, 0.2, 0.3, 0.20, 0.15, 0.08, 0.1),
        (1.1, 0.5, 3.2, 0.22, 2.3, 1.0, 0.10, 0.0),
        (1.1, 0.5, 2.0, 0.22, 1.0, 0.5, 0.10, 0.2),
        # a < sa < 2a: legal (a_i = a +- sa/2 stays positive) but used to be
        # rejected outright by the Numba reference's old sa < a bound.
        (1.1, 0.5, 1.5, 0.25, 0.8, 1.6, 0.08, 0.0),
    ]:
        for rt in [t + 0.1, t + 0.35, t + 0.9]:
            ref = model.pdf(rt, a, z, v, t, sv=sv, sa=sa, st=st, sz=sz,
                            validate=False)
            if ref < 1e-20:
                continue
            got = np.exp(f_logp([rt], [0.0], a, z, v, t, sv, sa, st, sz)[0])
            worst = max(worst, abs(got - ref) / ref)
            n += 1

    ok = worst < 1e-6
    print(f"    {n} densities, max relative error {worst:.3e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_2_gradients():
    """Analytic gradients vs central finite differences."""
    print("\n[2] gradients vs finite differences")
    f_logp, f_grad = _fn()
    data = simulate_ddmsa(**{k: TRUE[k] for k in ["a", "z", "v", "t", "sv", "sa", "st", "sz"]},
                          n_trials=500, seed=42)
    rt, ch = data[:, 0], data[:, 1]
    vals = [TRUE[p] for p in PARAMS]

    def total(v):
        return float(np.sum(f_logp(rt, ch, *v)))

    analytic = f_grad(rt, ch, *vals)
    ok = True
    for i, name in enumerate(PARAMS):
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
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_3_edges():
    """logp and gradients must stay finite everywhere NUTS can wander."""
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
        good = np.isfinite(L) and np.all(np.isfinite(G))
        ok &= good
        print(f"    {label:22s} logp={L:12.2f} grad finite={np.all(np.isfinite(G))} "
              f"{'' if good else '<-- BAD'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_sa_boundary():
    """sa <= 2a: mass is 1 at the bound inclusive, rejected past it, and the
    Numba reference enforces the same bound."""
    print("\n[+] sa boundary (a_i = a +/- sa/2 stays positive)")
    from scipy.integrate import quad

    from saddm.model import DDMModel

    f_logp, _ = _fn()
    a, z, v, t = 1.1, 0.5, 1.5, 0.25

    def logp(rt, resp, sa):
        return float(f_logp([rt], [float(resp)], a, z, v, t, 0.0, sa, 0.0, 0.0)[0])

    # Mass is exactly 1 at sa = 2a; past it the a_grid floor used to eat it.
    mass = sum(quad(lambda rt: np.exp(logp(rt, resp, 2.0 * a)), t + 1e-6, t + 30,
                    limit=200)[0]
               for resp in (0, 1))
    rejected = np.isclose(logp(t + 0.2, 0, 2.01 * a), float(_LOG_TINY))

    # The njit reference must draw the line in the same place (validate=False
    # exercises the kernel's own check, not DDMModel's validator).
    model = DDMModel(n_points=15)
    ref = (model.pdf(0.5, a, z, v, t, sa=1.99 * a, validate=False) > model.min_p
           and model.pdf(0.5, a, z, v, t, sa=2.01 * a, validate=False) == model.min_p)

    ok = abs(mass - 1.0) < 1e-3 and rejected and ref
    print(f"    mass at sa=2a={mass:.6f}  rejected past 2a={rejected}  "
          f"reference bound matches={ref}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_4_vector_params():
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
    return ok


def check_5_backends():
    """Every installed backend (C, Numba, JAX) must agree; report timings."""
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
        reps = 20
        t0 = time.time()
        for _ in range(reps):
            float(f(*vals)[0])
        ms = (time.time() - t0) / reps * 1000
        good = delta < 1e-8
        ok &= good
        print(f"    {mode:6s} logp+grad {ms:8.2f} ms   max|delta vs C| {delta:.2e} "
              f"{'' if good else '<-- MISMATCH'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_static_zero():
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
    ok = ok and pm.draw(y, random_seed=0).shape == data.shape
    print(f"    widest (sa, st, sz) quadrature grid through CustomDist: {widest}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_6_nuts(backend="numpyro", draws=750, tune=750, chains=2, n_trials=2000):
    """End-to-end gradient MCMC: sampler health plus a recovery report.

    Passing requires healthy geometry (no divergences, r_hat below 1.03, usable
    ESS) and the four core parameters within 3 SD of the truth.

    Recovery of sa, st and sv is reported but not asserted. Their MLE is genuinely
    biased at this sample size: over 8 datasets of 2000 trials the mean bias is
    sa -30% (SD 0.19, occasionally collapsing to 0) and st -18%, while a, z, v and
    t stay within 3%. The bias shrinks with N (sa -5% at 20k and at 200k), so it is
    a small-sample property of the estimator rather than a defect. HDI coverage is
    printed for the same reason: with posteriors as correlated as (a, v, sv, sa) a
    single dataset misses individual intervals often enough that asserting it would
    be a coin flip. Calibration belongs in a many-dataset SBC run.
    """
    print(f"\n[6] NUTS via {backend}")
    import arviz as az

    from saddm.ddmsa import make_ddmsa_model, sample_ddmsa

    truth = dict(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08)
    data = sample_ddmsa_exact(**truth, n_trials=n_trials, seed=99)
    print(f"    {len(data)} trials, mean RT {data[:, 0].mean():.3f}s, "
          f"upper {data[:, 1].mean():.1%}")

    model = make_ddmsa_model(data, use_potential=True)
    t0 = time.time()
    idata = sample_ddmsa(model, backend=backend, draws=draws, tune=tune,
                         chains=chains, random_seed=3, progressbar=False)
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
    return ok


def test_scale_convention():
    assert check_0_scale()


def test_reference_agreement():
    assert check_1_reference()


def test_gradients():
    assert check_2_gradients()


def test_edge_finiteness():
    assert check_3_edges()


def test_vector_params():
    assert check_4_vector_params()


def test_backends():
    assert check_5_backends()


def test_static_zero():
    assert check_static_zero()


def test_sa_boundary():
    assert check_sa_boundary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="also run the NUTS check")
    ap.add_argument("--backend", default="numpyro")
    args = ap.parse_args()

    results = {
        "0 s=1 scale": check_0_scale(),
        "1 reference": check_1_reference(),
        "2 gradients": check_2_gradients(),
        "3 edges": check_3_edges(),
        "4 broadcasting": check_4_vector_params(),
        "5 backends": check_5_backends(),
        "static zero": check_static_zero(),
        "sa boundary": check_sa_boundary(),
    }
    if args.sample:
        results["6 NUTS"] = check_6_nuts(backend=args.backend)

    print("\n" + "=" * 52)
    for k, v in results.items():
        print(f"  {k:16s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
