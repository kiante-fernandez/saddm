"""Fit the DDM-SA (across-trial variability in a, v, t) to cavanagh_theta via HSSM."""

import numpyro

numpyro.set_host_device_count(2)

import arviz as az
import hssm
import matplotlib.pyplot as plt
import pytensor.tensor as pt

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saddm.ddmsa import ddmsa_logp

PARAMS = ["v", "a", "z", "t", "sv", "sa", "st"]


def logp_ddmsa(data, v, a, z, t, sv, sa, st):
    # HSSM conventions -> saddm conventions: HSSM's a is the half boundary
    # separation (saddm uses the full one), and t here is the LOWER EDGE of the
    # uniform non-decision distribution (actual non-decision time = t + st/2).
    # Sampling the edge decorrelates t from st; sampling them independently makes
    # the posterior a ridge that NUTS cannot traverse.
    data = pt.reshape(data, (-1, 2))
    return ddmsa_logp(
        pt.abs(data[:, 0]),
        data[:, 1],
        a=2.0 * a,
        z=z,
        v=v,
        t=t + st / 2.0,
        sv=sv,
        sa=2.0 * sa,
        st=st,
    )


cav = hssm.load_data("cavanagh_theta")

model = hssm.HSSM(
    data=cav,
    model="ddmsa",
    loglik=logp_ddmsa,
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
            "st": (0.0, 0.6),
        },
    },
    p_outlier=0.05,
)

idata = model.sample(sampler="numpyro", draws=1000, tune=1000, chains=2)

print(az.summary(idata, var_names=list(PARAMS)))
print("divergences:", int(idata.sample_stats.diverging.values.sum()))

az.plot_trace(idata, var_names=list(PARAMS))
plt.tight_layout()
plt.savefig("ddmsa_cavanagh_traces.png", dpi=150)
print("saved ddmsa_cavanagh_traces.png")
