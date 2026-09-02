# Fortran reference implementation

The Fortran programs the Python package is validated against. Each file below
is the source of a specific artifact shipped in this repo.

| program | produces | shipped artifact |
|---|---|---|
| `fit_ddm_itc.f90` | per-subject plain-DDM ITC fits (drift regression, ML via simplex) | `data/itc_amasino/benchmark_targets_amasino.csv` |
| `fit_ddm_itc_sa.f90` | population DDM-SA fits on k trials per subject (the k-sweep) | `data/itc_amasino/fortran_ksweep_summary.csv` |
| `fit_sa_simplex.f90` | the DDM-SA density routines (FC/GQ/FFC/COR lineage) | the reference values hardcoded in `tests/test_fortran_match.py`, via the Numba port in `saddm/core.py` / `saddm/integrator.py` |

Build (gfortran, OpenMP):

```bash
gfortran -O2 -fopenmp -o fit_ddm_itc     fit_ddm_itc.f90
gfortran -O2 -fopenmp -o fit_ddm_itc_sa  fit_ddm_itc_sa.f90
```

The ITC fitters read whitespace-separated trial files with columns
`choice rt val_diff time_diff` — the format of
`data/itc_amasino/perms/k<k>/perm_NNN.csv`. Environment variables:
`DDM_NSZ` (quadrature points, 15 in the campaign), `DDM_NSTARTS` (simplex
restarts, 5), `OMP_NUM_THREADS`.

Scale note: these programs work at s = 1 for the ITC models; `fit_sa_simplex.f90`
follows Ratcliff's s = 0.1 convention (multiply a, v, sv, sa by 10 to compare
with the Python package). In `fit_ddm_itc.f90` output, the starting-point
convention is mirrored relative to the package (z_fortran = 1 - z).
