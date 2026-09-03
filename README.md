# saddm

Diffusion decision model with across-trial variability in boundary separation
(`sa`), drift (`sv`), and non-decision time (`st`) — the DDM-SA — as a fully
differentiable PyTensor likelihood for gradient-based Bayesian estimation with
[PyMC](https://www.pymc.io) and [HSSM](https://github.com/lnccbrown/HSSM).

The likelihood is analytic (Navarro–Fuss density; drift variability integrated
in closed form, uniform variability by Gauss–Legendre quadrature), so NUTS gets
exact gradients. It is validated against the Fortran implementation the model
was originally developed in; the Fortran programs, the data, and reference
results ship in this repository so every validation is reproducible from a
clone.

## Install

```bash
pip install -e .                # core: numpy, scipy, pytensor
pip install -e ".[sampling]"    # + pymc, arviz
pip install -e ".[hssm]"        # + hssm, with pinned jax/numpyro
pip install -e ".[test]"        # + pytest, numba (reference backend)
```

## Quickstart

```python
from saddm.ddmsa import make_ddmsa_model, sample_ddmsa, sample_ddmsa_exact

data  = sample_ddmsa_exact(a=1.1, z=0.5, v=1.5, t=0.25,
                           sv=0.8, sa=0.5, st=0.08, n_trials=2000)
idata = sample_ddmsa(make_ddmsa_model(data), backend="numpyro")
```

With HSSM, register `saddm.ddmsa.ddmsa_logp` as a `loglik_kind="analytical"`
likelihood; `examples/estimate_HSSM_saddm.py` is the minimal adapter.

## Conventions

- Diffusion coefficient **s = 1**; `a` is the full boundary separation;
  `sa`/`st`/`sz` are full widths of uniform across-trial distributions.
  (Ratcliff s = 0.1 values convert by ×10 on a, v, sv, sa.)
- **Non-decision time is sampled by its lower edge**: with `st > 0` the
  earliest response is at `t − st/2`, so bounding `t` by min(RT) excludes the
  truth (and spuriously collapses `sa`). Report `t0 = t_edge + st/2`.
- HSSM adapters apply the same rule inside the likelihood (`t + st/2`), and
  HSSM's built-in DDM defines `a` as the half separation (pass `2*a`).
- Hierarchical models in HSSM need bounded links
  (`link_settings="log_logit"`) on `a`/`t` random effects.

## Layout

| path | contents |
|---|---|
| `saddm/` | `ddmsa.py`: the likelihood and PyMC glue. `core.py`/`integrator.py`/`model.py`: the Numba reference implementation. |
| `tests/` | `test_ddmsa.py`: verification suite — s = 1 closed forms, agreement with the Numba/Fortran reference, finite-difference gradients, corner finiteness, per-trial broadcasting, C/Numba/JAX backend agreement (run directly with `--sample` for an end-to-end NUTS check). Remaining `test_*.py` cover the Numba reference, including `test_fortran_match.py`. |
| `verification/` | `parameter_recovery.py`: 100-config NUTS recovery study. `recovery_figure.py`, `compare_to_fortran.py`: analysis and figures (read `results/reference/` by default; set `RESULTS` for a fresh run). |
| `examples/` | HSSM applications: flat fit on cavanagh_theta, the per-subject + k-sweep replication of the Fortran intertemporal-choice analysis, and hierarchical variants. |
| `fortran/` | The Fortran programs that produced the benchmarks, with build notes. |
| `data/itc_amasino/` | Amasino et al. (2019) trials, the Fortran benchmarks, and the exact k-sweep permutation files. |
| `results/reference/` | Reference outputs: recovery, ITC, and hierarchical results with figures. Fresh runs write to gitignored `results/` subdirectories. |

## Validation summary

- Density matches the s = 1 closed-form choice probability and mean decision
  time to 1e-16, and the Fortran-derived Numba reference to 1.6e-8; gradients
  match finite differences to ~1e-9; C, Numba, and JAX backends agree to 2e-13.
- Parameter recovery (100 simulated datasets, 500 trials): a/z/v/t at
  r = 0.93–0.98 with near-nominal coverage; sv/sa/st at r = 0.65–0.71.
- Empirical: reproduces the Fortran per-subject benchmark (r = 0.95 on the
  value coefficient, 212 subjects) and the population k-sweep on identical
  trial samples; hierarchical DDM-SA converges on the full dataset.

## Citation

```bibtex
@unpublished{shevlin2026little,
  author = {Shevlin, Blair R. K. and Fernandez, Kiant{\'e} and Ratcliff, Roger and Krajbich, Ian},
  title  = {A little goes a long way: Fitting one-shot decisions with cognitive models},
  note   = {Manuscript in preparation},
  year   = {2026},
}
```

Blair R. K. Shevlin\* and Kianté Fernandez\* contributed equally.

:Author: Kianté Fernandez, Blair R K Shevlin, Roger Ratcliff, Ian Krajbich

:Contact: kiante@ucla.edu, blair.shevlin@mssm.edu, ratcliff.22@osu.edu, krajbich@ucla.edu
