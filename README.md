# saddm

Diffusion decision model with across-trial variability in boundary separation
(`sa`), drift (`sv`), and non-decision time (`st`) — the DDM-SA — as a fully
differentiable PyTensor likelihood.

## Install

```bash
pip install -e .                # core: numpy, scipy, pytensor
pip install -e ".[sampling]"    # + pymc, arviz
pip install -e ".[hssm]"        # + hssm and PINNED jax/numpyro (see below)
pip install -e ".[test]"        # + pytest, numba (the reference backend)
```

## Quickstart

```python
from saddm.ddmsa import make_ddmsa_model, sample_ddmsa, sample_ddmsa_exact

data  = sample_ddmsa_exact(a=1.1, z=0.5, v=1.5, t=0.25,
                           sv=0.8, sa=0.5, st=0.08, n_trials=2000)
idata = sample_ddmsa(make_ddmsa_model(data), backend="numpyro")
```

For HSSM, register `saddm.ddmsa.ddmsa_logp` as a `loglik_kind="analytical"`
likelihood — see `examples/estimate_HSSM_saddm.py` for the minimal adapter
(mind HSSM's half-boundary `a` convention and the `t + st/2` lower-edge rule).

## Layout

| path | contents |
|---|---|
| `saddm/` | The package. `ddmsa.py` is the canonical likelihood + PyMC glue; `core.py`/`integrator.py`/`model.py` are the Numba reference (Fortran-validated); `likelihood_pytensor.py` is an older PyTensor path kept as a second cross-check; `simulate.py` is the Euler simulator (biased by boundary overshoot — prefer `sample_ddmsa_exact` for recovery work). |
| `tests/` | `pytest` suite. `test_ddmsa.py` is the verification suite for the canonical likelihood: s = 1 closed forms, agreement with the Numba/Fortran reference and the legacy path, finite-difference gradients, corner finiteness, per-trial broadcasting, C/Numba/JAX backend agreement. Run directly with `--sample` for an end-to-end NUTS recovery check. The remaining `test_*.py` cover the Numba reference, including `test_fortran_match.py`. |
| `verification/` | The heavy validation studies. `parameter_recovery.py`: 100-config NUTS recovery (LHS grid at s = 1, exact sampler; `OUT_DIR`/`SHARD`/`N_SHARDS` env vars for cluster sharding). Reference results: `results/reference/recovery/`. |
| `data/itc_amasino/` | Amasino_2019 intertemporal-choice trials, the Fortran per-subject benchmark values, and the exact k-sweep permutation files the Fortran campaign used. |
| `results/reference/` | Reference outputs of the validation campaign: recovery study CSVs + figure, ITC per-subject/k-sweep results + figures, hierarchical fit summaries + trace plots. Fresh runs write to `results/{recovery,itc_amasino,hier}/` (gitignored) so reproduction never collides with the reference copies. |
| `examples/` | HSSM applications (data under `data/itc_amasino/`; cavanagh_theta ships with HSSM): `estimate_HSSM_saddm.py` (flat fit, cavanagh_theta), `HSSM_estimate_ITC_Amasino.py` (per-subject + k-sweep replication of the Fortran ITC analysis), `HSSM_hierarchical.py` (hierarchical variants; bounded links required on `a`/`t` random effects). |


:Author: Kianté Fernandez, Blair R K Shevlin, Roger Ratcliff, Ian Krajbich
:Contact: kiante@ucla.edu, blair.shevlin@mssm.edu, ratcliff.22@osu.edu, krajbich@ucla.edu

