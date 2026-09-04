"""Fit the DDM-SA (across-trial variability in a, v, t) to cavanagh_theta via HSSM."""

import numpyro

numpyro.set_host_device_count(2)

import arviz as az
import hssm
import matplotlib.pyplot as plt

import os

from _hssm import ddmsa_half_a

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "cavanagh"))
PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]


cav = hssm.load_data("cavanagh_theta")

model = hssm.HSSM(
    data=cav,
    model="ddmsa",
    loglik=ddmsa_half_a,
    loglik_kind="analytical",
    model_config={
        "response": ["rt", "response"],
        "list_params": list(PARAMS),
        "choices": (-1, 1),
        "bounds": {
            "v": (-10.0, 10.0),
            "a": (0.3, 3.0),
            "z": (0.05, 0.95),
            "t": (0.0, float(cav.rt.min())),
            "sv": (0.0, 3.0),
            "sa": (0.0, 1.0),
            "st": (0.0, 2.0),
        },
    },
    p_outlier=0.05,
)

idata = model.sample(sampler="numpyro", draws=1000, tune=1000, chains=2, random_seed=20240101)

print(az.summary(idata, var_names=list(PARAMS)))
print("divergences:", int(idata.sample_stats.diverging.values.sum()))

az.plot_trace(idata, var_names=list(PARAMS))
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "figure_cavanagh_flat_traces.png"), dpi=150)
print("saved", os.path.join(OUT_DIR, "figure_cavanagh_flat_traces.png"))
