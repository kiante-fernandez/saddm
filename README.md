# Implementation of Diffusion decision model with across-trial variability in boundary separation

Diffusion decision model with across-trial variability in boundary separation
(`sa`), drift (`sv`), and non-decision time (`st`). This codebase presents the 
DDM-SA as a fully differentiable PyTensor likelihood for gradient-based Bayesian estimation.

## Install

```bash
pip install -e .                # core: numpy, scipy, pytensor
pip install -e ".[sampling]"    # + pymc, arviz
pip install -e ".[hssm]"        # + hssm, with pinned jax/numpyro
pip install -e ".[test]"        # + pytest, numba (reference backend)
```

<<<<<<< Updated upstream
## Quickstart
=======
## Usage
>>>>>>> Stashed changes

```python
from saddm.ddmsa import make_ddmsa_model, sample_ddmsa, sample_ddmsa_exact

data  = sample_ddmsa_exact(a=1.1, z=0.5, v=1.5, t=0.25,
                           sv=0.8, sa=0.5, st=0.08, n_trials=2000)
idata = sample_ddmsa(make_ddmsa_model(data), backend="numpyro")
```

To use the model inside HSSM (custom analytical likelihood, regressions,
hierarchical designs), see [`examples/`](examples/).

## Layout

| path | contents |
|---|---|
| `saddm/` | The package: `ddmsa.py` (likelihood + PyMC glue) and the Numba reference implementation. |
| `tests/` | Verification suite and unit tests (`pytest tests`). |
| `verification/` | Parameter-recovery study and Fortran-comparison analyses. |
| `examples/` | HSSM applications, from a flat fit to hierarchical models. |
| `fortran/` | The Fortran reference programs behind the benchmarks. |
| `data/`, `results/reference/` | The data and reference outputs that make every validation reproducible from a clone. |

<<<<<<< Updated upstream
=======
The likelihood is validated against closed forms, the Fortran implementation,
and simulation-based parameter recovery; see [`verification/`](verification/)
and [`tests/`](tests/).

>>>>>>> Stashed changes
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
