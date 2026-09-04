# Fortran reference implementation

The Fortran programs the Python package is validated against. Each file below
is the source of a specific artifact in this repo.

| program | produces | artifact |
|---|---|---|
| `fit_ddm_itc.f90` | per-subject plain-DDM ITC fits (drift regression, ML via simplex) | `data/itc_amasino/benchmark_targets_amasino.csv` |
| `fit_ddm_itc_sa.f90` | population DDM-SA fits on k trials per subject (the k-sweep) | `data/itc_amasino/fortran_ksweep_summary.csv` |

Build (gfortran, OpenMP):

```bash
gfortran -O2 -fopenmp -o fit_ddm_itc     fit_ddm_itc.f90
gfortran -O2 -fopenmp -o fit_ddm_itc_sa  fit_ddm_itc_sa.f90
```

Both fitters read whitespace-separated trial files with columns
`choice rt val_diff time_diff`. The k-sweep inputs are
`data/itc_amasino/perms/k<k>/perm_NNN.csv`; per-subject inputs are one such
file per subject, extracted from `data/itc_amasino/itc_amasino.csv` (columns
`choice`, `rt`, `val_diff_usd`, `time_diff_days`). Running `fit_ddm_itc` on
subject `003_0001` reproduces its benchmark row to all printed digits.

`fit_ddm_itc_sa` reads `DDM_NSZ` (quadrature points, 15 in the campaign) and
`DDM_NSTARTS` (simplex restarts, 5) from the environment and honours
`OMP_NUM_THREADS`; `fit_ddm_itc` reads no environment variables and sets
four OpenMP threads itself.

Both programs work at s = 1 and report the starting point mirrored relative
to the package (z_fortran = 1 - z). The DDM-SA density routines (FC/GQ/FFC/COR)
in `fit_ddm_itc_sa.f90` are the ones ported to `tests/reference.py`.
