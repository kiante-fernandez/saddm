"""Compare HSSM ITC results (from examples/HSSM_estimate_ITC_Amasino.py) to the
Fortran benchmark: per-subject scatter and k-sweep recovery curves.

Reads results from results/reference/itc_amasino by default (the shipped
reference outputs); point RESULTS at a fresh run to compare a reproduction.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.environ.get("RESULTS", os.path.join(ROOT, "results/reference/itc_amasino"))
OUT = os.environ.get("OUT", RESULTS)
BENCH = pd.read_csv(os.path.join(ROOT, "data/itc_amasino/benchmark_targets_amasino.csv"))
FORTRAN_KSWEEP = pd.read_csv(os.path.join(ROOT, "data/itc_amasino/fortran_ksweep_summary.csv"))
BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#8a8983", "#1a1a19"


def per_subject():
    path = os.path.join(RESULTS, "persubject_plain.csv")
    if not os.path.exists(path):
        print("no persubject_plain.csv; skipping per-subject comparison")
        return
    ps = pd.read_csv(path)
    ps["subj_ident"] = ps.tag.str.replace("subj_", "")
    m = ps.merge(BENCH, on="subj_ident", suffixes=("_h", "_f"))
    conv = m[m.max_rhat <= 1.05]
    print(f"per-subject: {len(conv)}/{len(m)} converged")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (f, h, lab) in zip(axes, [("v_val_f", "v_val_h", "v_val"),
                                      ("a_f", "a_h", "a"), ("t0_f", "t0_h", "t0")]):
        lo = min(conv[f].min(), conv[h].min()); hi = max(conv[f].max(), conv[h].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(conv[f], conv[h], s=18, color=BLUE, alpha=0.7,
                   edgecolor="white", lw=0.5, zorder=3)
        r = np.corrcoef(conv[f], conv[h])[0, 1]
        print(f"  {lab:6s} fortran={conv[f].mean():8.4f} hssm={conv[h].mean():8.4f} r={r:.3f}")
        ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_xlabel(f"{lab}  Fortran (ML)"); ax.set_ylabel(f"{lab}  HSSM (NUTS)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
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
    ks = pd.read_csv(path)
    g = ks.groupby("k").agg(**{p: (p, "mean") for p in ["v_val", "a", "t0"]},
                            v_val_sd=("v_val", "std"), a_sd=("a", "std"), t0_sd=("t0", "std"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (p, ylabel) in zip(axes, [("v_val", "v_val (USD$^{-1}$)"),
                                      ("a", "a (boundary)"), ("t0", "t0 (s)")]):
        bm = BENCH[p if p != "t0" else "t0"].mean()
        ax.axhline(bm, color=MUTED, lw=1, ls="--")
        ax.errorbar(g.index, g[p], yerr=g[p + "_sd"], color=BLUE, lw=2, marker="o",
                    ms=6, capsize=3, label="HSSM (NUTS)")
        ax.plot(FORTRAN_KSWEEP.k, FORTRAN_KSWEEP[p], color=ORANGE, lw=2, marker="s",
                ms=6, label="Fortran (ML)")
        ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 10, 50])
        ax.set_xticklabels([1, 2, 4, 10, 50])
        ax.set_xlabel("k trials per subject"); ax.set_ylabel(ylabel)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Population DDM-SA on k trials per subject: HSSM vs Fortran "
                 "(identical permutation files)", fontsize=11, color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_ksweep_hssm_vs_fortran.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("k-sweep comparison (HSSM mean vs Fortran, by k):")
    for k, row in g.iterrows():
        f = FORTRAN_KSWEEP[FORTRAN_KSWEEP.k == k].iloc[0]
        print(f"  k={k:3d}  v_val {row.v_val:.4f} vs {f.v_val:.4f}   "
              f"a {row.a:.3f} vs {f.a:.3f}   t0 {row.t0:.3f} vs {f.t0:.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    per_subject()
    ksweep()
