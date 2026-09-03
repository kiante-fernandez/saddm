"""Participant random effects from a HSSM_hierarchical.py summary CSV.

Reads results/reference/hierarchical/cav_loglogit_summary.csv by default;
point SUMMARY at a fresh run. Writes figure_hier_participant_effects.png to
OUT (default results/figures).
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUMMARY = os.environ.get("SUMMARY", os.path.join(
    ROOT, "results/reference/hierarchical/cav_loglogit_summary.csv"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "results/figures"))
BLUE, MUTED, INK = "#2a78d6", "#8a8983", "#1a1a19"
LABELS = dict(v="drift v", a="boundary a (link scale)", t="non-decision t (link scale)")

s = pd.read_csv(SUMMARY, index_col=0)
s = s[~s.index.str.contains("_offset[", regex=False)]
effects = {p: s[s.index.str.match(rf"^{p}_1\|.*\[\d+\]$")] for p in LABELS}
effects = {p: e for p, e in effects.items() if len(e)}
n = len(next(iter(effects.values())))

fig, axes = plt.subplots(1, len(effects), figsize=(5 * len(effects), 4.8), sharey=True)
for ax, (p, e) in zip(axes, effects.items()):
    e = e.sort_values("mean", ascending=False)
    y = range(len(e))
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.errorbar(e["mean"], y, xerr=[e["mean"] - e["hdi_3%"], e["hdi_97%"] - e["mean"]],
                fmt="o", color=BLUE, ms=5, lw=1.5)
    ax.set_title(LABELS[p], fontsize=11, color=INK)
    ax.set_xlabel("participant offset (94% HDI)")
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#e8e7e2", lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("participant (sorted per panel)")
variant = re.sub(r"_summary\.csv$", "", os.path.basename(SUMMARY))
fig.suptitle(f"Hierarchical DDM-SA ({variant}): participant random effects, "
             f"{n} participants", fontsize=12, color=INK)
plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "figure_hier_participant_effects.png"), dpi=160,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "figure_hier_participant_effects.png"))
