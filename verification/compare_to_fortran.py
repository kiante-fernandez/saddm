"""Compare HSSM ITC results (from examples/HSSM_estimate_ITC_Amasino.py) to the
Fortran benchmark on every shared parameter: per-subject scatter and k-sweep
curves.

Reads results/reference/itc_amasino by default; point RESULTS at a fresh run
to compare a reproduction. Writes to OUT (default results/figures, gitignored).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results/reference/itc_amasino"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "results/figures"))
BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#8a8983", "#1a1a19"

# Fortran column -> HSSM column. Fortran reports the start point mirrored
# (z_fortran = 1 - z) and the k-sweep reports sa and st as standard deviations
# of the uniform (sd_a, sd_t0); fortran_frame undoes both.
COLS = {"v0": "v_Intercept", "v_val": "v_val", "v_time": "v_tim",
        "a": "a", "t0": "t0", "z": "z", "sv": "sv", "sa": "sa", "st": "st"}
LABELS = {"v0": "v0 (drift intercept)", "v_val": "v_val (USD$^{-1}$)",
          "v_time": "v_time (day$^{-1}$)", "a": "a (boundary)", "t0": "t0 (s)",
          "z": "z (start point)", "sv": "sv (drift variability)",
          "sa": "sa (boundary variability)", "st": "st (non-decision var.)"}


def fortran_frame(df):
    df = df.copy()
    df["z"] = 1.0 - df["z"]
    for sd, width in (("sd_a", "sa"), ("sd_t0", "st")):
        if sd in df:
            df[width] = df.pop(sd) * np.sqrt(12.0)
    return df


def hssm_frame(df):
    return df.rename(columns={h: f for f, h in COLS.items()})


BENCH = fortran_frame(pd.read_csv(os.path.join(ROOT, "data/itc_amasino/benchmark_targets_amasino.csv")))
FORTRAN_KSWEEP = fortran_frame(pd.read_csv(os.path.join(ROOT, "data/itc_amasino/fortran_ksweep_summary.csv")))


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#e8e7e2", lw=0.8)
    ax.set_axisbelow(True)


def per_subject():
    path = os.path.join(RESULTS, "persubject_plain.csv")
    if not os.path.exists(path):
        print("no persubject_plain.csv; skipping per-subject comparison")
        return
    ps = hssm_frame(pd.read_csv(path))
    ps["subj_ident"] = ps.tag.str.replace("subj_", "")
    m = ps.merge(BENCH, on="subj_ident", suffixes=("_h", "_f"))
    conv = m[m.max_rhat <= 1.05]
    params = [p for p in COLS if p in BENCH.columns]
    print(f"per-subject: {len(conv)}/{len(m)} converged")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    for ax, p in zip(axes.flat, params):
        f, h = conv[p + "_f"], conv[p + "_h"]
        lo, hi = min(f.min(), h.min()), max(f.max(), h.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(f, h, s=18, color=BLUE, alpha=0.7, edgecolor="white", lw=0.5, zorder=3)
        r = np.corrcoef(f, h)[0, 1]
        print(f"  {p:6s} fortran={f.mean():8.4f} hssm={h.mean():8.4f} r={r:.3f}")
        ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_xlabel(f"{LABELS[p]}  Fortran (ML)"); ax.set_ylabel("HSSM (NUTS)")
        style(ax)
    fig.suptitle(f"Per-subject plain DDM + drift regression: HSSM vs Fortran "
                 f"(Amasino 2019, n = {len(conv)} converged)", fontsize=11, color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_persubject_hssm_vs_fortran.png"),
                dpi=160, bbox_inches="tight")
    plt.close()


def ksweep():
    path = os.path.join(RESULTS, "ksweep.csv")
    if not os.path.exists(path):
        print("no ksweep.csv; skipping k-sweep comparison")
        return
    ks = hssm_frame(pd.read_csv(path))
    params = [p for p in COLS if p in FORTRAN_KSWEEP.columns]
    g = ks.groupby("k")[params].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.6))
    for ax, p in zip(axes.flat, params):
        if p in BENCH.columns:
            ax.axhline(BENCH[p].mean(), color=MUTED, lw=1, ls="--",
                       label="per-subject benchmark")
        ax.errorbar(g.index, g[(p, "mean")], yerr=g[(p, "std")], color=BLUE, lw=2,
                    marker="o", ms=6, capsize=3, label="HSSM (NUTS), mean ± SD over perms")
        ax.plot(FORTRAN_KSWEEP.k, FORTRAN_KSWEEP[p], color=ORANGE, lw=2, marker="s",
                ms=6, label="Fortran (ML), mean over perms")
        ax.set_xscale("log"); ax.set_xticks(FORTRAN_KSWEEP.k)
        ax.set_xticklabels(FORTRAN_KSWEEP.k)
        ax.set_xlabel("k trials per subject"); ax.set_ylabel(LABELS[p])
        style(ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    fig.suptitle("Population DDM-SA on k trials per subject: HSSM vs Fortran "
                 "(identical permutation files)", fontsize=11, color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_ksweep_hssm_vs_fortran.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("k-sweep, HSSM mean vs Fortran by k:")
    table = g.xs("mean", axis=1, level=1).join(FORTRAN_KSWEEP.set_index("k")[params],
                                               lsuffix="_hssm", rsuffix="_fortran")
    print(table.round(4).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    per_subject()
    ksweep()
