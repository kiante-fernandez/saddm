"""Hierarchical (participant random-effect) DDM-SA fits under HSSM.

VARIANT selects the model:
  cav_loglogit  cavanagh_theta, DDM-SA, random intercepts on v, a, t;
                link_settings="log_logit" keeps a > 0 and t inside its bounds.
  cav_va        cavanagh_theta, DDM-SA, random intercepts on v and a only
                (identity links); t, sv, sa, st shared.
  itc_hier      Amasino_2019 ITC, plain DDM with drift regression
                v ~ 1 + val + tim + (1|subj), random intercepts on a and t,
                link_settings="log_logit".
  itc_hier_sa   As itc_hier but the full DDM-SA: sv, sa, st estimated (shared
                across subjects, across-trial within subject).

DRAWS / TUNE / TARGET_ACCEPT environment variables override the defaults.

The flat cavanagh_theta fit with identity-link random effects on v, a, t gave
2000/2000 divergences: unbounded participant offsets push a and t into the
invalid region, where the likelihood is flat. The variants above are the two
ways around that.
"""

import os
import time

import numpyro

numpyro.set_host_device_count(2)

import arviz as az
import hssm
import pandas as pd
import pytensor.tensor as pt

from saddm import ddmsa_logp

VARIANT = os.environ.get("VARIANT", "cav_loglogit")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hier"))
os.makedirs(OUT_DIR, exist_ok=True)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DRAWS = int(os.environ.get("DRAWS", "1000"))
TUNE = int(os.environ.get("TUNE", "1000"))
TARGET_ACCEPT = float(os.environ.get("TARGET_ACCEPT", "0.8"))
CHAINS = 2


def ddmsa(data, v, a, z, t, sv, sa, st):
    """ddmsa_logp as an HSSM loglik. a and the widths are in saddm's full units;
    t is the lower edge of the non-decision distribution, so t0 = t + st/2."""
    data = pt.reshape(data, (-1, 2))
    return ddmsa_logp(pt.abs(data[:, 0]), data[:, 1], a=a, z=z, v=v,
                      t=t + st / 2.0, sv=sv, sa=sa, st=st)


def build():
    if VARIANT.startswith("cav"):
        cav = hssm.load_data("cavanagh_theta")
        params = ["v", "a", "z", "t", "sv", "sa", "st"]
        min_rt = float(cav.rt.min())
        bounds = {"v": (-10.0, 10.0), "a": (0.6, 6.0), "z": (0.05, 0.95),
                  "sv": (0.0, 3.0), "sa": (0.0, 2.0), "st": (0.0, 2.0)}
        if VARIANT == "cav_loglogit":
            bounds["t"] = (0.0, 1.0)
            include = [
                {"name": "v", "formula": "v ~ 1 + (1|participant_id)", "link": "identity"},
                {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
                {"name": "t", "formula": "t ~ 1 + (1|participant_id)"},
            ]
            link = "log_logit"
        else:
            bounds["t"] = (0.0, min_rt)
            include = [
                {"name": "v", "formula": "v ~ 1 + (1|participant_id)"},
                {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
            ]
            link = None
        return hssm.HSSM(
            data=cav, model="ddmsa", loglik=ddmsa, loglik_kind="analytical",
            model_config={"response": ["rt", "response"], "list_params": params,
                          "choices": (-1, 1), "bounds": bounds},
            p_outlier=0.05, include=include, link_settings=link,
        )

    src = os.path.join(ROOT, "data/itc_amasino/itc_amasino.csv")
    d = pd.read_csv(src)
    d = d[d.paper == "Amasino_2019"]
    df = pd.DataFrame({
        "rt": d.rt.values,
        "response": 2.0 * d.choice.values - 1.0,
        "val": d.val_diff_usd.values.astype(float),
        "tim": d.time_diff_days.values.astype(float),
        "subj": d.subj_ident.values,
    })
    params = ["v", "a", "z", "t", "sv", "sa", "st"]
    bounds = {"v": (-10.0, 10.0), "a": (0.3, 6.0), "z": (0.05, 0.95),
              "t": (0.0, 1.6), "sv": (0.0, 2.0), "sa": (0.0, 3.0), "st": (0.0, 2.0)}
    fixed = {} if VARIANT == "itc_hier_sa" else dict(sv=0.0, sa=0.0, st=0.0)
    include = [
        {"name": "v", "formula": "v ~ 1 + val + tim + (1|subj)", "link": "identity",
         "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 2.0},
                   "val": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                   "tim": {"name": "Normal", "mu": 0.0, "sigma": 0.05}}},
        {"name": "a", "formula": "a ~ 1 + (1|subj)"},
        {"name": "t", "formula": "t ~ 1 + (1|subj)"},
    ]
    return hssm.HSSM(
        data=df, model="ddm_itc",
        loglik=ddmsa, loglik_kind="analytical",
        model_config={"response": ["rt", "response"], "list_params": params,
                      "choices": (-1, 1), "bounds": bounds},
        p_outlier=0.05, include=include, link_settings="log_logit", **fixed,
    )


model = build()
print(model, flush=True)
ip = model.pymc_model.initial_point()
print("logp at init:", float(model.pymc_model.compile_logp()(ip)), flush=True)

t0 = time.time()
idata = model.sample(sampler="numpyro", draws=DRAWS, tune=TUNE, chains=CHAINS,
                     target_accept=TARGET_ACCEPT, progressbar=False, random_seed=20240101)
elapsed = time.time() - t0
idata.to_netcdf(os.path.join(OUT_DIR, f"{VARIANT}_idata.nc"))

s = az.summary(idata)
s.to_csv(os.path.join(OUT_DIR, f"{VARIANT}_summary.csv"))
div = int(idata.sample_stats.diverging.values.sum())
print(f"\n### {VARIANT}: {elapsed:.0f}s div={div} max_rhat={float(s.r_hat.max()):.3f} "
      f"min_ess={float(s.ess_bulk.min()):.0f} "
      f"mean_treedepth={float(idata.sample_stats.tree_depth.values.mean()):.1f}", flush=True)
keep = [i for i in s.index if "offset" not in i and "|" not in i or "sigma" in i]
print(s.loc[keep, ["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].to_string())
