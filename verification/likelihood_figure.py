"""Analytic DDM-SA density against an Euler-Maruyama simulation that shares no
code with the likelihood. Writes figure_likelihood_validation.png to OUT
(default results/figures).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e8e7e2", "axes.axisbelow": True})
import numpy as np
import pytensor
import pytensor.tensor as pt

from saddm.ddmsa import ddmsa_logp, simulate_ddmsa

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.environ.get("OUT", os.path.join(ROOT, "results/figures"))
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1a1a19"
P = dict(a=1.1, z=0.5, v=1.5, t=0.25, sv=0.8, sa=0.5, st=0.08)
N_TRIALS, DT = 200_000, 1e-5

sim = simulate_ddmsa(**P, n_trials=N_TRIALS, dt=DT, seed=7)
grid = np.linspace(0.15, 2.5, 800)
rt, ch = pt.dvector("rt"), pt.dvector("ch")
dens = pytensor.function([rt, ch], pt.exp(ddmsa_logp(rt, ch, **P)))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, (c, name) in zip(axes, [(1, "upper"), (0, "lower")]):
    rts = sim[sim[:, 1] == c, 0]
    frac_sim = rts.size / sim.shape[0]
    f = dens(grid, np.full(grid.size, float(c)))
    frac_an = np.sum((f[1:] + f[:-1]) / 2 * np.diff(grid))
    counts, edges = np.histogram(rts, bins=120, range=(grid[0], grid[-1]))
    ax.stairs(counts / (sim.shape[0] * np.diff(edges)), edges, fill=True, color=BLUE,
              alpha=0.55,
              label=f"Euler–Maruyama simulation\n({N_TRIALS // 1000}k trials, dt={DT:g})")
    ax.plot(grid, f, color=ORANGE, lw=2.2,
            label="analytic DDM-SA density\n(saddm.ddmsa_logp)")
    ax.set_title(f"{name} boundary   sim {frac_sim:.3f} vs analytic {frac_an:.3f}",
                 fontsize=11, color=INK)
    ax.set_xlabel("response time (s)")
    print(f"{name}: P(sim) = {frac_sim:.4f}, P(analytic) = {frac_an:.4f}")
axes[0].set_ylabel("density")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("DDM-SA likelihood validation: analytic density vs simulation  ("
             + ", ".join(f"{k}={v}" for k, v in P.items()) + ")", fontsize=12, color=INK)
plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "figure_likelihood_validation.png"), dpi=160,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "figure_likelihood_validation.png"))
