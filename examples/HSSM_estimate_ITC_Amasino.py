"""HSSM replication of the Fortran ITC DDM-SA analysis (Amasino_2019).

Two stages, matching the fit_ddm_itc_sa.f90 / ksweep_v2 design:
  1. Per-subject fits: each subject's full trials, 9-parameter model
     (a, t, z, sv, sa, st, and drift delta_i = v0 + v_val*val + v_time*tim).
  2. Population k-sweep: all subjects pooled, k trials sampled per subject,
     fitted on the SAME perm files the Fortran campaign used
     (data/itc_amasino/perms/k<k>/perm_NNN.csv).

Units are the Fortran ones throughout: a is the FULL boundary separation and
sa/st are FULL widths of the uniform across-trial/participant distributions,
so estimates compare to data/itc_amasino/benchmark_targets_amasino.csv
directly. t is sampled as the lower edge of the non-decision distribution
(actual t0 = t + st/2).

Environment: STAGE (all | persubject | ksweep), PLAIN=1 for the plain DDM,
SHARD / N_SHARDS to split the per-subject stage across workers, OUT_DIR.
Both stages checkpoint to CSV and skip completed rows on restart.
"""

import os
import time

import numpyro

numpyro.set_host_device_count(2)

import arviz as az
import hssm
import pandas as pd

from _hssm import ddm, ddmsa

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_CSV = os.path.join(ROOT, "data/itc_amasino/itc_amasino.csv")
PERM_DIR = os.path.join(ROOT, "data/itc_amasino/perms")
OUT_DIR = os.environ.get("OUT_DIR",
                         os.path.join(ROOT, "results/itc_amasino"))
os.makedirs(OUT_DIR, exist_ok=True)

# Optional sharding for parallel per-subject workers: SHARD in [0, N_SHARDS).
# Each shard fits subjects where index % N_SHARDS == SHARD and writes its own
# checkpoint CSV, so concurrent workers never touch the same file.
SHARD = int(os.environ.get("SHARD", "-1"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
STAGE = os.environ.get("STAGE", "all")  # all | persubject | ksweep
# PLAIN=1 drops sv/sa/st (plain DDM + drift regression). This is the model the
# Fortran per-subject benchmark actually fitted (benchmark_targets.csv has no
# variability columns), so it is the apples-to-apples per-subject comparison.
PLAIN = os.environ.get("PLAIN", "0") == "1"

KS = [1, 2, 4, 10, 50]
N_PERMS = 5
DRAWS, TUNE, CHAINS = 1000, 1000, 2
SUFFIX = "_plain" if PLAIN else ""
if PLAIN:
    PARAMS = ["v", "a", "z", "t"]
    REPORT = ["v_Intercept", "v_val", "v_tim", "a", "z", "t"]
else:
    PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]
    REPORT = ["v_Intercept", "v_val", "v_tim", "a", "z", "t", "sv", "sa", "st"]


def fit_one(df, tag):
    """df columns: rt, response (+-1), val, tim. Returns one summary row."""
    min_rt = float(df.rt.min())
    bounds = {"v": (-10.0, 10.0), "a": (0.3, 6.0), "z": (0.05, 0.95),
              "t": (0.0, min_rt), "sv": (0.0, 2.0), "sa": (0.0, 3.0),
              "st": (0.0, 2.0)}
    model = hssm.HSSM(
        data=df,
        model="ddm_itc" if PLAIN else "ddmsa_itc",
        loglik=ddm if PLAIN else ddmsa,
        loglik_kind="analytical",
        model_config={
            "response": ["rt", "response"],
            "list_params": list(PARAMS),
            "choices": (-1, 1),
            "bounds": {k: bounds[k] for k in PARAMS},
        },
        p_outlier=0.05,
        include=[{
            "name": "v",
            "formula": "v ~ 1 + val + tim",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 2.0},
                "val": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                "tim": {"name": "Normal", "mu": 0.0, "sigma": 0.05},
            },
        }],
    )
    t0 = time.time()
    idata = model.sample(sampler="numpyro", draws=DRAWS, tune=TUNE,
                         chains=CHAINS, progressbar=False, random_seed=20240101)
    s = az.summary(idata, var_names=list(REPORT))
    row = {"tag": tag, "n_trials": len(df), "sec": round(time.time() - t0),
           "div": int(idata.sample_stats.diverging.values.sum()),
           "max_rhat": round(float(s.r_hat.max()), 3),
           "min_ess": int(s.ess_bulk.min())}
    for p in REPORT:
        row[p] = float(s.loc[p, "mean"])
        row[p + "_sd"] = float(s.loc[p, "sd"])
    row["t0"] = row["t"] if PLAIN else row["t"] + row["st"] / 2.0
    return row


def append_row(row, path):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not os.path.exists(path))


def done_tags(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path).tag)


def report(tag, row):
    print(f"{tag} n={row['n_trials']} {row['sec']}s div={row['div']} "
          f"rhat={row['max_rhat']} | "
          + " ".join(f"{p}={row[p]:.4f}" for p in REPORT + ["t0"]), flush=True)


def run_per_subject():
    shard = f"_shard{SHARD}" if SHARD >= 0 else ""
    path = os.path.join(OUT_DIR, f"persubject{SUFFIX}{shard}.csv")
    done = done_tags(path)
    d = pd.read_csv(SRC_CSV)
    d = d[d.paper == "Amasino_2019"]
    for i, (subj, g) in enumerate(d.groupby("subj_ident")):
        if SHARD >= 0 and i % N_SHARDS != SHARD:
            continue
        tag = f"subj_{subj}"
        if tag in done:
            continue
        df = pd.DataFrame({
            "rt": g.rt.values,
            "response": 2.0 * g.choice.values - 1.0,
            "val": g.val_diff_usd.values.astype(float),
            "tim": g.time_diff_days.values.astype(float),
        })
        try:
            row = fit_one(df, tag)
            append_row(row, path)
            report(tag, row)
        except Exception as e:
            print(f"{tag} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


def run_ksweep():
    path = os.path.join(OUT_DIR, f"ksweep{SUFFIX}.csv")
    done = done_tags(path)
    for k in KS:
        for perm in range(1, N_PERMS + 1):
            tag = f"k{k}_perm{perm:03d}"
            if tag in done:
                continue
            fp = os.path.join(PERM_DIR, f"k{k}", f"perm_{perm:03d}.csv")
            raw = pd.read_csv(fp, sep=" ", header=None,
                              names=["choice", "rt", "val", "tim"])
            df = pd.DataFrame({
                "rt": raw.rt.values,
                "response": 2.0 * raw.choice.values - 1.0,
                "val": raw.val.values.astype(float),
                "tim": raw.tim.values.astype(float),
            })
            try:
                row = fit_one(df, tag)
                row["k"], row["perm"] = k, perm
                append_row(row, path)
                report(tag, row)
            except Exception as e:
                print(f"{tag} FAILED {type(e).__name__}: {str(e)[:150]}",
                      flush=True)


def summarize():
    import glob
    parts = sorted(glob.glob(os.path.join(OUT_DIR, f"persubject{SUFFIX}.csv"))
                   + glob.glob(os.path.join(OUT_DIR, f"persubject{SUFFIX}_shard*.csv")))
    if parts:
        ps = pd.concat([pd.read_csv(f) for f in parts])
        print(f"\n=== per-subject (n={len(ps)} subjects) ===")
        print(ps[REPORT + ["t0"]].mean().round(4).to_string())
    ks_path = os.path.join(OUT_DIR, f"ksweep{SUFFIX}.csv")
    if os.path.exists(ks_path):
        ks = pd.read_csv(ks_path)
        print("\n=== k-sweep (mean over perms) ===")
        print(ks.groupby("k")[REPORT + ["t0", "div"]].mean().round(4).to_string())


if __name__ == "__main__":
    if STAGE in ("all", "persubject"):
        run_per_subject()
    if STAGE in ("all", "ksweep"):
        run_ksweep()
    summarize()
