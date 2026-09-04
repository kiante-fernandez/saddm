"""Fit the DDM-SA (across-trial variability in a, v, t) to cavanagh_theta via HSSM."""

import numpyro

numpyro.set_host_device_count(2)

import arviz as az
import hssm
import matplotlib.pyplot as plt
import pytensor.tensor as pt

import os

from saddm import ddmsa_logp

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "cavanagh"))
PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]


def ddmsa(data, v, a, z, t, sv, sa, st):
    """ddmsa_logp as an HSSM loglik. a and the widths are in saddm's full units;
    t is the lower edge of the non-decision distribution, so t0 = t + st/2."""
    data = pt.reshape(data, (-1, 2))
    return ddmsa_logp(pt.abs(data[:, 0]), data[:, 1], a=a, z=z, v=v,
                      t=t + st / 2.0, sv=sv, sa=sa, st=st)


cav = hssm.load_data("cavanagh_theta")

model = hssm.HSSM(
    data=cav,
    model="ddmsa",
    loglik=ddmsa,
    loglik_kind="analytical",
    model_config={
        "response": ["rt", "response"],
        "list_params": list(PARAMS),
        "choices": (-1, 1),
        "bounds": {
            "v": (-10.0, 10.0),
            "a": (0.6, 6.0),
            "z": (0.05, 0.95),
            "t": (0.0, float(cav.rt.min())),
            "sv": (0.0, 3.0),
            "sa": (0.0, 2.0),
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
