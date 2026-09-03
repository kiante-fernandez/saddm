"""Seven-panel posterior-vs-truth figure for the parameter recovery study.

Reads results/reference/recovery by default; point RESULTS at a fresh
parameter_recovery.py output directory (sharded CSVs are concatenated) to plot
a reproduction. Writes to OUT (default results/figures, gitignored).
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results/reference/recovery"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "results/figures"))
BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#8a8983", "#1a1a19"
PARAMS = ["a", "z", "v", "t", "sv", "sa", "st"]
LABELS = dict(a="a (boundary)", z="z (start point)", v="v (drift)",
              t="t (non-decision, s)", sv="sv (drift variability)",
              sa="sa (boundary variability)", st="st (non-decision var.)")

files = sorted(glob.glob(os.path.join(RESULTS, "ddm_sa_recovery_nuts*.csv")))
d = pd.concat([pd.read_csv(f) for f in files]).sort_values("config_id")
if not d.config_id.is_unique:
    raise ValueError(f"duplicate config_id across {files}")
d["max_rhat"] = d[[f"rhat_{p}" for p in PARAMS]].max(axis=1)
ok = d.max_rhat <= 1.05
print(f"{len(d)} configs, {int((~ok).sum())} non-converged, "
      f"{int(d.n_divergences.sum())} total divergences")

fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
for ax, p in zip(axes.flat, PARAMS):
    t_, e_ = d[f"true_{p}"], d[f"post_median_{p}"]
    lo, hi = min(t_.min(), e_.min()), max(t_.max(), e_.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(t_[ok], e_[ok], s=18, color=BLUE, alpha=0.75,
               edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(t_[~ok], e_[~ok], s=22, facecolor="none", edgecolor=ORANGE, lw=1.2, zorder=4)
    r = np.corrcoef(t_[ok], e_[ok])[0, 1]
    cov = d.loc[ok, f"coverage_{p}"].mean()
    print(f"  {p:3s} r={r:.3f} coverage={cov:.2f}")
    ax.text(0.04, 0.95, f"r = {r:.2f}\ncover = {cov:.2f}", transform=ax.transAxes,
            fontsize=9, color=INK, va="top")
    ax.set_title(LABELS[p], fontsize=10)
    ax.set_xlabel("true"); ax.set_ylabel("posterior median")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
axes.flat[7].axis("off")
axes.flat[7].legend(handles=[
    plt.Line2D([], [], marker="o", ls="", color=BLUE,
               label=f"converged (r̂ ≤ 1.05), n={int(ok.sum())}"),
    plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=ORANGE, label=f"non-converged, n={int((~ok).sum())}")],
    loc="center", fontsize=10, frameon=False)
fig.suptitle("DDM-SA parameter recovery: NUTS posteriors vs truth "
             f"({len(d)} simulated datasets, 500 trials each, exact sampler, s = 1)",
             fontsize=12, color=INK)
plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "figure_recovery.png"), dpi=160, bbox_inches="tight")
print("saved", os.path.join(OUT, "figure_recovery.png"))
