# Implementation of Diffusion decision model with across-trial variability in boundary separation

[![PyPI](https://img.shields.io/pypi/v/saddm.svg)](https://pypi.org/project/saddm/)
[![Python](https://img.shields.io/pypi/pyversions/saddm.svg)](https://pypi.org/project/saddm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/kiante-fernandez/saddm/actions/workflows/tests.yml/badge.svg)](https://github.com/kiante-fernandez/saddm/actions/workflows/tests.yml)

Diffusion decision model with across-trial variability in boundary separation
(`sa`), drift (`sv`), and non-decision time (`st`). This codebase presents the 
DDM-SA as a fully differentiable PyTensor likelihood for gradient-based Bayesian estimation.

The likelihood is analytic (Navarro–Fuss density; drift variability integrated
in closed form, uniform variability by Gauss–Legendre quadrature), so NUTS gets
exact gradients. It is validated against the Fortran implementation the model
was originally developed in; the Fortran programs, the data, and reference
results ship in this repository so every validation is reproducible from a
clone.

## Install

From PyPI:

```bash
pip install saddm               # core: numpy, scipy, pytensor
pip install "saddm[sampling]"   # + pymc, arviz, numpyro (pinned jax)
pip install "saddm[hssm]"       # + hssm
```

From a clone, for development or to run the verification and examples:

```bash
pip install -e ".[test]"        # + pytest, numba (reference backend)
pip install -e ".[hssm]"        # everything the examples need
```

The `sampling` and `hssm` extras pin `jax==0.5.3` / `numpyro==0.19.0`
deliberately: newer jax silently freezes numpyro's NUTS at its initial point.

## Quickstart

```python
from saddm import make_ddmsa_model, sample_ddmsa, sample_ddmsa_exact

data  = sample_ddmsa_exact(a=1.1, z=0.5, v=1.5, t=0.25,
                           sv=0.8, sa=0.5, st=0.08, n_trials=2000)
idata = sample_ddmsa(make_ddmsa_model(data), backend="numpyro")
```

`saddm.ddmsa_logp(rt, response, a, z, v, t, sv, sa, st, sz)` is the per-trial
log-likelihood; every parameter may be a scalar or a per-trial vector.

With HSSM, register `saddm.ddmsa_logp` as a `loglik_kind="analytical"`
likelihood; `examples/estimate_HSSM_saddm.py` is the minimal adapter.

## Layout

| path | contents |
|---|---|
| `saddm/` | `ddmsa.py`: the likelihood and PyMC glue. `core.py`/`integrator.py`/`model.py`: the Numba reference implementation (`reference` extra). |
| `tests/` | `test_ddmsa.py`: verification suite — s = 1 closed forms, agreement with the Numba/Fortran reference, finite-difference gradients, corner finiteness, per-trial broadcasting, backend agreement, static-zero collapse (run directly with `--sample` for an end-to-end NUTS check). Remaining `test_*.py` cover the Numba reference. |
| `verification/` | `parameter_recovery.py`: 100-config NUTS recovery study. `recovery_figure.py`, `compare_to_fortran.py`, `likelihood_figure.py`: analysis and figures (read `results/reference/` by default; set `RESULTS` for a fresh run). |
| `examples/` | HSSM applications: flat fit on cavanagh_theta, the per-subject + k-sweep replication of the Fortran intertemporal-choice analysis, hierarchical variants, and the random-effects figure. |
| `fortran/` | The Fortran programs that produced the benchmarks, with build notes. |
| `data/itc_amasino/` | Amasino et al. (2019) trials, the Fortran benchmarks, and the exact k-sweep permutation files. |
| `results/reference/` | Reference outputs: recovery, ITC, hierarchical, and cavanagh results with figures. Everything else under `results/` is gitignored, and every script writes there by default. |

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
