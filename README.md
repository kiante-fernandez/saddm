# Implementation of Diffusion decision model with across-trial variability in boundary separation

[![PyPI](https://img.shields.io/pypi/v/saddm.svg)](https://pypi.org/project/saddm/)
[![Python](https://img.shields.io/pypi/pyversions/saddm.svg)](https://pypi.org/project/saddm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/kiante-fernandez/saddm/actions/workflows/tests.yml/badge.svg)](https://github.com/kiante-fernandez/saddm/actions/workflows/tests.yml)

Diffusion decision model with across-trial variability in boundary separation
(`sa`), drift (`sv`), and non-decision time (`st`). This codebase presents the 
DDM-SA as a fully differentiable PyTensor likelihood for gradient-based Bayesian estimation.

## Install

From PyPI:

```bash
pip install saddm               # core: numpy, scipy, pytensor
pip install "saddm[sampling]"   # + pymc, arviz, numpyro (pinned jax)
pip install "saddm[hssm]"       # + hssm
pip install -e ".[test]"        # + pytest, numba (from a clone, for development or to run the verification and example)
```

## Quickstart

```python
import pymc as pm
from saddm import DDMSA, sample_ddmsa_exact

data = sample_ddmsa_exact(a=1.1, z=0.5, v=1.5, t=0.25,
                          sv=0.8, sa=0.5, st=0.08, n_trials=2000)

with pm.Model():
    a, z, v = pm.HalfNormal("a", 3.0), pm.Beta("z", 3.0, 3.0), pm.Normal("v", 0.0, 2.0)
    t = pm.Uniform("t", 0.0, data[:, 0].min())
    DDMSA("y", a, z, v, t, sv=pm.HalfNormal("sv", 1.5), sa=pm.HalfNormal("sa", 1.0),
          st=pm.HalfNormal("st", 0.5), observed=data)
    idata = pm.sample(nuts_sampler="numpyro")
```

`saddm.ddmsa_logp(rt, response, a, z, v, t, sv, sa, st, sz)` is the per-trial
log-likelihood; every parameter may be a scalar or a per-trial vector.

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
